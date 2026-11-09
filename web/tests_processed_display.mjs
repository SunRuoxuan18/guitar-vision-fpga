import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require(join(process.env.USERPROFILE,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
const browser = await chromium.launch({ headless: true, channel: "msedge" });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.route("**/api/**", route => route.fulfill({ status: 503, body: "offline" }));
  let count = 0, release, hold = true, badIdentity = false;
  const pending = [];
  await page.route("**/api/vision/pc/status", route => route.fulfill({
    contentType: "application/json", body: '{"link_speed_mbps":1000}' }));
  await page.route("**/api/vision/pc/process", async route => {
    count++;
    const request = route.request(), options = JSON.parse(request.headers()["x-vision-options"]);
    const data = request.postDataBuffer();
    pending.push({ options, rgb: [...data.subarray(0, 3)] });
    if (hold) await new Promise(resolve => { release = resolve; });
    else await new Promise(resolve => setTimeout(resolve, 200));
    const analysis = { sequence: options.sequence, stream_id: options.stream,
      generation: options.generation, processing_location: "ps_linux", status: "searching",
      roi_status: "tracking", roi_corners: [{x:.1,y:.2},{x:.9,y:.2},{x:.9,y:.8},{x:.1,y:.8}],
      hand_region: { mode: "neck_local_lightweight", status: "tracking",
        sequence: options.sequence, stream_id: options.stream, generation: options.generation,
        regions: [{ corners: [{x:.5,y:.3},{x:.8,y:.3},{x:.8,y:.7},{x:.5,y:.7}] }] } };
    await route.fulfill({ contentType: "application/x-fretlens-planes",
      body: Buffer.alloc(320 * 240 * 2, 80), headers: {
        "X-Vision-Backend": "pl", "X-Vision-Source": "pc_camera",
        "X-Vision-Token": badIdentity ? "wrong-token" : request.headers()["x-vision-token"],
        "X-Vision-Processing-Ms": "180", "X-Vision-Fretboard": JSON.stringify(analysis) } });
  });
  await page.goto("http://127.0.0.1:8001/");
  await page.evaluate(async () => {
    window.FretStopCamera?.(); window.FretVision.stop();
    document.querySelector("#pcPlEnabled").checked = true;
    document.querySelector("#autoFretboard").checked = true;
    document.querySelector("#visionView").value = "color";
    const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 480;
    const context = canvas.getContext("2d");
    window.paintCapture = color => { context.fillStyle = color; context.fillRect(0, 0, 640, 480); };
    window.paintCapture("rgb(220,40,30)");
    window.fixtureStream = canvas.captureStream(30);
    const video = document.querySelector("#cameraVideo"); video.srcObject = window.fixtureStream;
    await video.play(); window.FretVision.attachCamera(window.fixtureStream);
  });
  await page.waitForFunction(() => window.FretVision.metrics().pl_status !== "not-used");
  while (!release) await new Promise(resolve => setTimeout(resolve, 20));
  const pixel = () => page.evaluate(() => {
    const c = document.querySelector("#fretOverlay");
    return [...c.getContext("2d").getImageData(c.width / 2, c.height / 2, 1, 1).data];
  });
  assert.deepEqual(await pixel(), [17, 21, 22, 255], "No raw camera image before a board reply");
  await page.evaluate(() => window.paintCapture("rgb(30,50,220)"));
  hold = false; release();
  await page.waitForFunction(() => window.FretVision.metrics().frames >= 1);
  const first = await pixel();
  assert.ok(first[0] > 180 && first[2] < 70, `Accepted frame must still be red: ${first}`);
  assert.deepEqual(pending[0].rgb, [220, 40, 30]);
  await page.waitForTimeout(1400);
  const metrics = await page.evaluate(() => window.FretVision.metrics());
  assert.ok(metrics.display_fps > 1 && metrics.display_fps < 8, JSON.stringify(metrics));
  assert.match(await page.locator("#videoRate").textContent(), /板卡处理画面/);
  assert.ok(metrics.frames <= count, "Only accepted board frames count toward FPS");
  await page.locator("#autoFretboard").evaluate(node => {
    node.checked = false; node.dispatchEvent(new Event("change"));
  });
  await page.waitForTimeout(350);
  const withoutDetection = await pixel();
  assert.ok(withoutDetection[2] > 180, "PL-only display also uses its submitted camera frame");
  badIdentity = true;
  await page.waitForTimeout(1500);
  assert.deepEqual(await pixel(), [17, 21, 22, 255], "Invalid/stale replies must not expose live input");
  assert.deepEqual(errors, []);
  await page.evaluate(() => { window.FretVision.stop(); window.fixtureStream.getTracks().forEach(t => t.stop()); });
  console.log(JSON.stringify({ matching_capture: "pass", startup_cover: "pass", honest_fps: metrics.display_fps,
    detection_disabled: "pass", invalid_reply_cover: "pass", errors }, null, 2));
} finally { await browser.close(); }
