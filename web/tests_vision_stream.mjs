import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const { chromium } = await import(pathToFileURL(process.argv[2]).href);
const browser = await chromium.launch({ headless: true, executablePath: process.argv[3] });
const results = [];
try {
  for (const width of [1440, 360]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(process.argv[4] || "http://127.0.0.1:8000/");
    await page.locator("#debugDetails").evaluate(node => { node.open = true; });
    // This regression measures plain preview, independently of detector throughput.
    await page.locator("#autoFretboard").uncheck();
    const image = Buffer.from(await page.evaluate(() => {
      const canvas = document.createElement("canvas");
      canvas.width = 640; canvas.height = 480;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#752840"; ctx.fillRect(0, 0, 640, 480);
      ctx.fillStyle = "#27d3a0"; ctx.fillRect(100, 80, 300, 280);
      return canvas.toDataURL("image/jpeg").split(",")[1];
    }), "base64");
    let requests = 0, sequence = 0, active = 0, maxActive = 0, frozen = false, failed = false;
    let stream = "fixture-before-restart";
    // This fixture exercises the JPEG polling fallback; the dedicated board
    // rate test covers the persistent multipart stream path.
    await page.route("**/vision/stream.mjpeg", route =>
      route.fulfill({ status: 503, body: "fixture uses polling fallback" }));
    await page.route("**/vision/frame.jpg*", async (route) => {
      active += 1; maxActive = Math.max(maxActive, active);
      try {
        await new Promise((resolve) => setTimeout(resolve, 8));
        requests += 1;
        if (!frozen && !failed) sequence += 1;
        await route.fulfill(failed ? { status: 503, body: "offline" } : {
          status: 200, contentType: "image/jpeg", body: image,
          headers: { "X-Vision-Sequence": String(sequence),
            "X-Vision-Captured-Ms": String(performance.now()), "X-Vision-Source": "synthetic",
            "X-Vision-Stream": stream }
        });
      } finally { active -= 1; }
    });
    await page.locator("#boardVisionButton").click();
    await page.waitForFunction(() => window.FretVision.active);
    const started = performance.now(), initial = requests;
    await page.waitForTimeout(3000);
    const fps = (requests - initial) / ((performance.now() - started) / 1000);
    assert.ok(fps > 18, `Expected useful preview throughput, got ${fps.toFixed(1)} fps`);
    assert.equal(maxActive, 1, "Requests must not build up in parallel");
    assert.ok(await page.evaluate(() => window.FretVision.metrics().display_fps) > 18);
    for (const view of ["gray", "edge", "color"]) {
      await page.locator("#visionView").selectOption(view);
      await page.waitForFunction((expectedView) => {
        const canvas = document.querySelector("#fretOverlay");
        const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
        let bright = 0, colored = 0;
        for (let p = 0; p < data.length; p += 16) {
          if (data[p] > 20) bright += 1;
          if (Math.abs(data[p] - data[p + 1]) > 20) colored += 1;
        }
        return bright > 250 && (colored > 250) === (expectedView === "color");
      }, view);
      const pixels = await page.locator("#fretOverlay").evaluate((canvas) => {
        const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
        let bright = 0, colored = 0;
        for (let p = 0; p < data.length; p += 4) {
          if (data[p] > 20) bright += 1;
          if (Math.abs(data[p] - data[p + 1]) > 20) colored += 1;
        }
        return { bright, colored };
      });
      assert.ok(pixels.bright > 1000, `${view} must render actual pixels`);
      assert.equal(pixels.colored > 1000, view === "color");
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    frozen = true;
    await page.waitForTimeout(2200);
    assert.equal(await page.evaluate(() => window.FretVision.metrics().display_fps), 0,
      "Repeated sequence numbers must not count as fresh frames");
    const stalePixels = await page.locator("#fretOverlay").evaluate((canvas) => {
      const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      return data.some((value) => value !== 0);
    });
    assert.equal(stalePixels, false, "Stale source images must be cleared");
    failed = true;
    await page.waitForTimeout(150);
    const failedStart = requests;
    await page.waitForTimeout(1200);
    assert.ok(requests - failedStart <= 2, "Offline retries should back off");
    assert.match(await page.locator("#visionStatus").textContent(), /正在自动重连/);
    stream = "fixture-after-restart";
    failed = false;
    await page.waitForFunction(() => {
      const canvas = document.querySelector("#fretOverlay");
      const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      return document.querySelector("#visionStatus").textContent.includes("已恢复") &&
        data.some((value) => value !== 0);
    }, null, { timeout: 5000 });
    frozen = false;
    await page.waitForTimeout(1500);
    assert.ok(await page.evaluate(() => window.FretVision.metrics().display_fps) > 0,
      "A restarted service must resume even when its first sequence repeats");
    await page.locator("#visionStopButton").click();
    await page.waitForTimeout(50);
    const stopped = requests;
    await page.waitForTimeout(1100);
    assert.equal(requests, stopped);
    assert.deepEqual(errors, []);
    results.push({ width, fixture: "synthetic 640x480 JPEG, 8ms response; no hand inference",
      preview_fps: fps, max_parallel_requests: maxActive, views: "pass",
      stale_duplicate_frames: "pass", error_backoff: "pass", service_restart_recovery: "pass", stop: "pass" });
    await page.close();
  }
  await writeFile(new URL("../diagnostics/vision_browser/stream_performance_results.json", import.meta.url),
    JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
} finally { await browser.close(); }
