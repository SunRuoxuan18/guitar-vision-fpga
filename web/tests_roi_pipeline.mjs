import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createRequire } from "node:module";
import { writeFile } from "node:fs/promises";
const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || "playwright");
const fixture = spawn("vision/.venv/Scripts/python.exe", ["-B", "diagnostics/auto_fretboard_fixture.py", "18192"],
  { env: { ...process.env, OPENBLAS_NUM_THREADS: "1", OMP_NUM_THREADS: "1" }, stdio: ["ignore", "pipe", "pipe"] });
const proxy = spawn(process.execPath, ["server.mjs"], { cwd: "web", env: { ...process.env,
  PORT: "18110", FPGA_VISION_URL: "http://127.0.0.1:18192" }, stdio: ["ignore", "pipe", "pipe"] });
const children = [fixture, proxy], logs = [], results = [];
for (const child of children) child.stderr.on("data", data => logs.push(String(data)));
let browser;
async function pose(step, blank = false) {
  const response = await fetch("http://127.0.0.1:18192/fixture/pose", {
    method: "POST", body: JSON.stringify({ step, blank }) });
  assert.equal(response.ok, true);
}
try {
  await Promise.all(children.map(child => once(child.stdout, "data")));
  browser = await chromium.launch({ channel: "msedge", headless: true });
  for (const width of [1440, 360]) {
    await pose(0);
    const page = await browser.newPage({ viewport: { width, height: 1000 } }), errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto("http://127.0.0.1:18110/?camera=board");
    await page.waitForFunction(() => window.FretVision.active);
    await page.locator("#debugDetails").evaluate(element => { element.open = true; });
    await page.locator("#focusFretboard").uncheck();
    await page.locator("#selectNeckButton").click();
    await page.locator("#fretOverlay").scrollIntoViewIfNeeded();
    const box = await page.locator("#fretOverlay").evaluate(canvas => {
      const rect = canvas.getBoundingClientRect();
      const width = Math.min(canvas.width, canvas.height * 640 / 400), height = width * 400 / 640;
      return { x: rect.x, y: rect.y, sx: rect.width / canvas.width, sy: rect.height / canvas.height,
        left: (canvas.width - width) / 2, top: (canvas.height - height) / 2, width, height };
    });
    for (const [x, y] of [[65, 130], [585, 105], [570, 267], [55, 245]]) {
      await page.mouse.click(box.x + (box.left + x / 639 * box.width) * box.sx,
        box.y + (box.top + y / 399 * box.height) * box.sy);
    }
    await page.waitForFunction(() => {
      const a = window.FretVision.trackingSnapshot().automatic;
      return a?.backend === "roi_lk_sobel_projection" && a.status === "tracking";
    });
    assert.equal(await page.locator("#autoFretboard").isChecked(), true);
    const initial = await page.evaluate(() => window.FretVision.trackingSnapshot().automatic);
    assert.equal(initial.strings.length, 6);
    assert.equal(initial.frets.length, 22);
    for (let step = 1; step <= 3; step++) {
      const previous = await page.evaluate(() => window.FretVision.trackingSnapshot().automatic.corners[0]);
      await pose(step);
      await page.waitForFunction(previous => {
        const a = window.FretVision.trackingSnapshot().automatic;
        return a?.status === "tracking" && Math.hypot(a.corners[0].x - previous.x, a.corners[0].y - previous.y) > .005;
      }, previous);
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({ path: `diagnostics/vision_browser/roi_pipeline_${width}.png`, fullPage: true });
    await pose(3, true);
    await page.waitForFunction(() => {
      const a = window.FretVision.trackingSnapshot().automatic;
      return a?.roi_status === "lost" && !a.corners && !a.roi_corners;
    });
    await page.locator("#visionStopButton").click();
    assert.deepEqual(errors, []);
    results.push({ width, source: "synthetic fixture, real ROI API and LK pipeline",
      manual_roi: "pass", six_strings_22_frets: "pass", moving_roi: "pass", loss_clears: "pass", errors });
    await page.close();
  }
  await writeFile("diagnostics/vision_browser/roi_pipeline_results.json", JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
} finally {
  await browser?.close();
  for (const child of children) child.kill();
  if (logs.length) console.error(logs.join(""));
}
