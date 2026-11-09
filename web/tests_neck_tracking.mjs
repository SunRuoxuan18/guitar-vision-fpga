import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createRequire } from "node:module";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || "playwright");
const output = await mkdtemp(join(tmpdir(), "neck-tracking-"));
const fixture = spawn("vision/.venv/Scripts/python.exe", ["-B", "diagnostics/tracking_fixture.py", "18082"], { stdio: ["ignore", "pipe", "pipe"] });
const proxy = spawn(process.execPath, ["server.mjs"], { cwd: "web", env: { ...process.env,
  PORT: "18000", FPGA_VISION_URL: "http://127.0.0.1:18082",
  FPGA_API_URL: "http://127.0.0.1:18082", FPGA_PL_VISION_URL: "http://127.0.0.1:18082"
}, stdio: ["ignore", "pipe", "pipe"] });
const children = [fixture, proxy], report = [];
let browser;
const logs = [];
for (const child of children) child.stderr.on("data", data => logs.push(String(data)));
async function pose(step, blank = false) {
  const response = await fetch("http://127.0.0.1:18082/fixture/pose", { method: "POST",
    body: JSON.stringify({ step, blank }) });
  assert.equal(response.status, 200, "Fixture must publish the requested pose");
  return response.json();
}
async function waitForPose(page, expected) {
  await page.waitForFunction(({ sequence, stream_id }) => {
    const tracking = window.FretVision.trackingSnapshot().tracking;
    return tracking?.status === "tracking" && tracking.stream_id === stream_id && tracking.sequence >= sequence;
  }, expected);
}
async function checkPose(page, expected, label) {
  await waitForPose(page, expected);
  const check = await page.evaluate(({ corners, width, height }) => {
    const { calibration, tracking } = window.FretVision.trackingSnapshot();
    if (!calibration) return { tracking };
    const target = Fretboard.createCalibration(corners);
    const tip = Fretboard.transform(target.toImage,
      { x: (target.boundaries[2].x + target.boundaries[3].x) / 2, y: .6 });
    const location = Fretboard.locateTip(calibration, tip);
    return { tracking, location, error: Math.max(...corners.map((point, index) =>
      Math.hypot((point.x - calibration.corners[index].x) * (width - 1),
        (point.y - calibration.corners[index].y) * (height - 1)))) };
  }, expected);
  assert.equal(check.tracking.status, "tracking", JSON.stringify(check));
  assert.ok(check.error < 4, `${label}, quad error ${check.error}px, ${JSON.stringify(check)}`);
  assert.equal(check.location[0].fret, 3);
  assert.equal(check.location[0].string, 3);
  return check.error;
}
async function calibrate(page, { corners, width: frameWidth, height: frameHeight }) {
  await page.locator("#autoFretboard").uncheck();
  await page.locator("#calibrationSettings").evaluate(element => { element.open = true; });
  await page.locator("#calibrateButton").click();
  await page.locator("#fretOverlay").scrollIntoViewIfNeeded();
  const box = await page.locator("#fretOverlay").evaluate((canvas, { frameWidth, frameHeight }) => {
    const rect = canvas.getBoundingClientRect();
    const width = Math.min(canvas.width, canvas.height * frameWidth / frameHeight), height = width * frameHeight / frameWidth;
    return { x: rect.x, y: rect.y, scaleX: rect.width / canvas.width, scaleY: rect.height / canvas.height,
      left: (canvas.width - width) / 2, top: (canvas.height - height) / 2, width, height };
  }, { frameWidth, frameHeight });
  for (const point of corners) await page.mouse.click(box.x + (box.left + point.x * box.width) * box.scaleX,
    box.y + (box.top + point.y * box.height) * box.scaleY);
}
try {
  await Promise.all(children.map(child => once(child.stdout, "data")));
  browser = await chromium.launch({ channel: "msedge", headless: true });
  for (const width of [1440, 360]) {
    const initial = await pose(0);
    const page = await browser.newPage({ viewport: { width, height: 1000 } }), errors = [];
    let automaticRequests = 0, trackingRequests = 0;
    page.on("request", request => {
      const url = new URL(request.url());
      if (url.pathname !== "/vision/frame.jpg") return;
      if (url.searchParams.has("auto")) automaticRequests += 1;
      if (url.searchParams.has("tracking")) trackingRequests += 1;
    });
    // The camera-origin fixture exercises manual tracking without a PC/PL service.
    await page.route("**/api/vision/pc/**", route => route.fulfill({ status: 503,
      contentType: "application/json", body: '{"error":"PC/PL is outside the local tracking fixture"}' }));
    page.on("pageerror", error => errors.push(error.message));
    await page.goto("http://127.0.0.1:18000/");
    await page.locator("#debugDetails").evaluate(element => { element.open = true; });
    await page.locator("#boardVisionButton").click();
    await page.waitForFunction(() => window.FretVision.active);
    await calibrate(page, initial);
    const initialError = await checkPose(page, initial, `width=${width}, initial calibration`);
    let maximumError = 0;
    for (let step = 1; step <= 12; step += 1) {
      const expected = await pose(step);
      const error = await checkPose(page, expected, `width=${width}, step=${step}`);
      maximumError = Math.max(maximumError, error);
    }
    await page.screenshot({ path: join(output, `neck_tracking_${width}.png`), fullPage: true });
    const overlayPixels = await page.locator("#fretOverlay").evaluate(canvas => {
      const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let grid = 0;
      for (let index = 0; index < pixels.length; index += 4)
        if (pixels[index] > 175 && pixels[index + 1] > 200 && pixels[index + 2] < 150) grid += 1;
      return grid;
    });
    assert.ok(overlayPixels > 50, "Moving grid must be visible");
    const overlaps = await page.locator(".vision-controls").evaluate(element => {
      const controls = [...element.children].map(node => node.getBoundingClientRect());
      return controls.some((a, index) => controls.slice(index + 1).some(b =>
        a.left < b.right - 1 && a.right > b.left + 1 && a.top < b.bottom - 1 && a.bottom > b.top + 1));
    });
    assert.equal(overlaps, false);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await pose(12, true);
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().tracking?.status === "lost");
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().calibration), null);
    // Tracking state changes in fetch completion; the label follows on the next render.
    await page.waitForFunction(() => document.getElementById("visualEvidence").textContent.includes("跟踪丢失"));
    assert.match(await page.locator("#visualEvidence").textContent(), /跟踪丢失/);
    await pose(0);
    await page.waitForTimeout(450);
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().tracking.status), "lost");
    await calibrate(page, initial);
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().tracking?.status === "tracking");
    const freeze = await fetch("http://127.0.0.1:18000/vision/frame.jpg?tracking=" +
      await page.evaluate(() => window.FretVision.trackingSnapshot().tracking.tracker_id));
    const frozenBody = Buffer.from(await freeze.arrayBuffer());
    await page.route("**/vision/frame.jpg?tracking=*", route => route.fulfill({
      status: 200, headers: Object.fromEntries(freeze.headers), body: frozenBody }));
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().tracking?.status === "lost");
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().calibration), null);
    await page.unroute("**/vision/frame.jpg?tracking=*");
    await page.waitForTimeout(200);
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().tracking.status), "lost");
    await page.locator("#followNeck").uncheck();
    await calibrate(page, initial);
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().calibration !== null);
    const fixed = await page.evaluate(() => window.FretVision.trackingSnapshot().calibration.corners);
    await pose(10); await page.waitForTimeout(350);
    assert.deepEqual(await page.evaluate(() => window.FretVision.trackingSnapshot().calibration.corners), fixed);
    await page.locator("#visionStopButton").click();
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().tracking), null);
    assert.deepEqual(errors, []);
    assert.equal(automaticRequests, 0, "Camera-origin frames must not request the legacy automatic detector");
    assert.ok(trackingRequests > 0, "Manual tracking must still request tracker metadata");
    report.push({ width, source: "rendered known poses through real OpenCV, HTTP proxy and browser",
      frame_resolution: [initial.width, initial.height], initial_corner_error_px: initialError,
      maximum_corner_error_px: maximumError, moving_fret_string_mapping: "pass",
      loss_clears_geometry: "pass", stale_clears_geometry: "pass", explicit_recalibration: "pass",
      fixed_option: "pass", stop: "pass", legacy_auto_requests: automaticRequests,
      manual_tracking_requests: trackingRequests });
    await page.close();
  }
  await writeFile(join(output, "neck_tracking_results.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  console.log(`Evidence: ${output}`);
} finally {
  await browser?.close();
  for (const child of children) child.kill();
  await Promise.all(children.map(child => child.exitCode === null && child.signalCode === null ? once(child, "exit") : null));
  if (logs.length) console.error(logs.join(""));
}
