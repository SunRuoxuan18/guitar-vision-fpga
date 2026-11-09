import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createRequire } from "node:module";
import { writeFile } from "node:fs/promises";

const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || "playwright");
const fixture = spawn("vision/.venv/Scripts/python.exe", ["-B", "diagnostics/auto_fretboard_fixture.py", "18182"],
  { env: { ...process.env, OPENBLAS_NUM_THREADS: "1", OMP_NUM_THREADS: "1" }, stdio: ["ignore", "pipe", "pipe"] });
const proxy = spawn(process.execPath, ["server.mjs"], { cwd: "web", env: { ...process.env,
  PORT: "18100", FPGA_VISION_URL: "http://127.0.0.1:18182" }, stdio: ["ignore", "pipe", "pipe"] });
const children = [fixture, proxy], report = [], logs = [];
for (const child of children) child.stderr.on("data", data => logs.push(String(data)));
let browser;
async function pose(step, blank = false) {
  const response = await fetch("http://127.0.0.1:18182/fixture/pose", { method: "POST",
    body: JSON.stringify({ step, blank }) });
  assert.equal(response.ok, true);
}
async function acquired(page) {
  await page.waitForFunction(() => {
    const a = window.FretVision.trackingSnapshot().automatic;
    return a?.status === "tracking" && a.strings.length === 6 && a.frets.length === 22;
  });
}
try {
  await Promise.all(children.map(child => once(child.stdout, "data")));
  browser = await chromium.launch({ channel: "msedge", headless: true });
  for (const width of [1440, 360]) {
    await pose(0); await new Promise(resolve => setTimeout(resolve, 300));
    const page = await browser.newPage({ viewport: { width, height: 1000 } }), errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto("http://127.0.0.1:18100/");
    assert.equal(await page.locator("#autoFretboard").isChecked(), true);
    await page.locator("#debugDetails > summary").click();
    await page.locator("#boardVisionButton").click();
    await acquired(page);
    const first = await page.evaluate(() => window.FretVision.trackingSnapshot().automatic.corners);
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().calibration), null);
    let previous = first;
    for (let step = 1; step <= 4; step += 1) {
      await pose(step);
      await page.waitForFunction(corners => {
        const a = window.FretVision.trackingSnapshot().automatic;
        return a?.status === "tracking" && Math.hypot(a.corners[0].x - corners[0].x,
          a.corners[0].y - corners[0].y) > .01;
      }, previous);
      await acquired(page);
      previous = await page.evaluate(() => window.FretVision.trackingSnapshot().automatic.corners);
    }
    const pixels = await page.locator("#fretOverlay").evaluate(canvas => {
      const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let cyan = 0, yellow = 0;
      for (let p = 0; p < data.length; p += 4) {
        if (data[p + 1] > 150 && data[p + 2] > 150 && data[p] < 160) cyan += 1;
        if (data[p] > 180 && data[p + 1] > 210 && data[p + 2] < 150) yellow += 1;
      }
      return { cyan, yellow };
    });
    assert.ok(pixels.cyan > 100, JSON.stringify(pixels));
    assert.match(await page.locator("#frameStatusText").textContent(), /SYNTHETIC/);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({ path: `diagnostics/vision_browser/auto_fretboard_${width}.png`, fullPage: true });
    await pose(1, true);
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().automatic?.status === "searching");
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().calibration), null);
    await pose(0); await acquired(page);
    const frozen = await fetch("http://127.0.0.1:18100/vision/frame.jpg?auto=1");
    const body = Buffer.from(await frozen.arrayBuffer());
    await page.route("**/vision/frame.jpg*", route => route.fulfill({
      status: 200, headers: Object.fromEntries(frozen.headers), body }));
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().automatic === null);
    assert.equal(await page.evaluate(() => window.FretVision.snapshot()), null);
    await page.unroute("**/vision/frame.jpg*");
    await acquired(page);
    let automaticRequests = 0, previewRequests = 0;
    await page.route("**/vision/frame.jpg*", async route => {
      const automatic = new URL(route.request().url()).searchParams.get("auto") === "1";
      if (!automatic) { previewRequests += 1; await route.continue(); return; }
      automaticRequests += 1;
      const response = await route.fetch();
      const headers = response.headers();
      const geometry = JSON.stringify({ mode: "auto", status: "searching", reason: "detector_too_slow",
        sequence: Number(headers["x-vision-sequence"]), stream_id: headers["x-vision-stream"],
        corners: null, strings: [], frets: [] });
      await route.fulfill({ response, headers: { ...headers, "x-vision-fretboard": geometry,
        "x-vision-tracking": geometry } });
    });
    await page.waitForFunction(() => document.querySelector("#visionStatus").textContent.includes("检测较慢"));
    const requestsAtBackoff = automaticRequests;
    await page.waitForTimeout(800);
    assert.ok(previewRequests > 0, "Slow detection must fall back to ordinary preview");
    assert.equal(automaticRequests, requestsAtBackoff, "Slow analysis must back off after the fresh slow result");
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().automatic?.corners ?? null), null);
    await page.locator("#visionStopButton").click();
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().automatic), null);
    assert.deepEqual(errors, []);
    report.push({ width, source: "independent synthetic guitar through actual OpenCV, HTTP proxy and browser",
      strings: 6, frets: 22, no_manual_clicks: true, moving_pose: "pass", visible_overlay: pixels,
      disappearance: "pass", reacquire: "pass", duplicate_expiry: "pass", slow_detector_preview: "pass",
      stop: "pass", errors });
    await page.close();
  }
  await writeFile("diagnostics/vision_browser/auto_fretboard_results.json", JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser?.close();
  for (const child of children) child.kill();
  if (logs.length) console.error(logs.join(""));
}
