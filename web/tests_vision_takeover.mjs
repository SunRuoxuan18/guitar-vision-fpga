import assert from "node:assert/strict";
import { createRequire } from "node:module";

const { chromium } = createRequire(import.meta.url)(process.argv[2]);
const browser = await chromium.launch({ headless: true, executablePath: process.argv[3] });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on("pageerror", error => errors.push(error.message));

let sourceSequence = 0;
let processCalls = 0;
let busyInjected = false;
let busyIndex = -1;
const processSequences = [];
const processOwners = [];

try {
  const image = Buffer.from(await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const context = canvas.getContext("2d");
    context.fillStyle = "#6b2637";
    context.fillRect(0, 0, 640, 480);
    context.fillStyle = "#d8c8aa";
    context.fillRect(120, 60, 320, 360);
    return canvas.toDataURL("image/jpeg").split(",")[1];
  }), "base64");

  await page.route("**/vision/stream.mjpeg", route =>
    route.fulfill({ status: 503, body: "takeover test uses polling" }));
  await page.route("**/vision/frame.jpg*", async route => {
    sourceSequence += 1;
    await new Promise(resolve => setTimeout(resolve, 8));
    await route.fulfill({
      status: 200,
      contentType: "image/jpeg",
      body: image,
      headers: {
        "X-Vision-Sequence": String(sourceSequence),
        "X-Vision-Captured-Ms": String(performance.now()),
        "X-Vision-Source": "camera",
        "X-Vision-Stream": "takeover-stream"
      }
    });
  });
  await page.route("**/api/vision/pc/status", route =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ link_speed_mbps: 100 })
    }));
  await page.route("**/api/vision/pc/process", async route => {
    const request = route.request();
    const options = JSON.parse(request.headers()["x-vision-options"]);
    processCalls += 1;
    processSequences.push(options.sequence);
    processOwners.push(options.owner);
    if (!busyInjected && processCalls >= 1) {
      busyInjected = true;
      busyIndex = processSequences.length - 1;
      await route.fulfill({ status: 409, body: "another stream owns PL" });
      return;
    }

    const tracking = busyInjected && processSequences.length > busyIndex + 1 &&
      options.sequence === 1;
    const corners = [
      { x: 0.16, y: 0.13 },
      { x: 0.82, y: 0.18 },
      { x: 0.84, y: 0.86 },
      { x: 0.14, y: 0.82 }
    ];
    const analysis = {
      status: tracking ? "tracking" : "searching",
      reason: tracking ? null : "no_color_neck_candidate",
      roi_status: tracking ? "tracking" : "searching",
      roi_corners: tracking ? corners : [],
      strings: [],
      frets: [],
      stream_id: options.stream,
      sequence: options.sequence,
      generation: options.generation,
      processing_location: "ps_linux"
    };
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "{}",
      headers: {
        "X-Vision-Backend": "pl",
        "X-Vision-Source": "pc_camera",
        "X-Vision-Token": request.headers()["x-vision-token"],
        "X-Vision-Processing-Ms": "12",
        "X-Vision-Fretboard": JSON.stringify(analysis)
      }
    });
  });

  await page.goto("http://127.0.0.1:8001/?camera=board&transport=jpeg&v=takeover-test");
  await page.waitForFunction(() => window.FretVision?.active, null, { timeout: 8000 });

  const deadline = Date.now() + 8000;
  while (Date.now() < deadline &&
      !(busyInjected && processSequences.slice(busyIndex + 1).includes(1))) {
    await page.waitForTimeout(50);
  }

  assert.equal(busyInjected, true, "The fixture must exercise one 409 response");
  assert.ok(processSequences.slice(busyIndex + 1).includes(1),
    `Recognition must restart at sequence 1 after 409; saw ${processSequences.join(",")}`);
  assert.ok(processOwners.every(owner => typeof owner === "string" && owner.length >= 8),
    "Every recognition request must carry an owner");
  await page.waitForFunction(() =>
    window.FretVision.trackingSnapshot().automatic?.roi_status === "tracking",
  null, { timeout: 5000 });
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    takeover: "pass",
    process_sequences: processSequences,
    owner_takeover: "pass",
    automatic_status: await page.evaluate(() =>
      window.FretVision.trackingSnapshot().automatic?.roi_status)
  }, null, 2));
} finally {
  await page.close();
  await browser.close();
}
