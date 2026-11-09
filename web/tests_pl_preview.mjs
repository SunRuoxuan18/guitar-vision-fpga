import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
let playwright;
try { playwright = require(process.argv[2] || "playwright"); }
catch (error) {
  if (process.argv[2] || !process.env.USERPROFILE) throw error;
  playwright = require(join(process.env.USERPROFILE,
    ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
}
const root = fileURLToPath(new URL(".", import.meta.url));
const evidence = await mkdtemp(join(tmpdir(), "fretlens-pl-"));
const report = [], servers = [], children = [];
let browser;
async function listen(server, port = 0, host = "127.0.0.1") {
  servers.push(server);
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(port, host, resolve); });
  return `http://${host}:${server.address().port}`;
}
async function startProxy(cameraUrl, plUrl) {
  const env = { ...process.env, PORT: "0", FPGA_API_URL: cameraUrl, FPGA_VISION_URL: cameraUrl };
  delete env.FPGA_BIND_ADDRESS; delete env.FPGA_PL_VISION_URL;
  if (plUrl) env.FPGA_PL_VISION_URL = plUrl;
  const child = spawn(process.execPath, ["server.mjs"], { cwd: root, env, windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"] });
  children.push(child);
  return new Promise((resolve, reject) => {
    let output = "", errors = "";
    const timer = setTimeout(() => reject(new Error(`Proxy startup timeout: ${errors}`)), 5000);
    child.stderr.on("data", chunk => { errors += chunk; });
    child.stdout.on("data", chunk => {
      output += chunk;
      const match = output.match(/http:\/\/127\.0\.0\.1:\d+/);
      if (match) { clearTimeout(timer); resolve(match[0]); }
    });
    child.once("error", error => { clearTimeout(timer); reject(error); });
    child.once("exit", code => { clearTimeout(timer); reject(new Error(`Proxy exit ${code}: ${errors}`)); });
  });
}
const json = (response, payload, status = 200) => {
  response.writeHead(status, { "Content-Type": "application/json" }); response.end(JSON.stringify(payload));
};
async function pixels(page) {
  return page.evaluate(() => Object.fromEntries(["grayPreview", "edgePreview", "fretOverlay"].map(id => {
    const c = document.getElementById(id), ctx = c.getContext("2d");
    return [id, { center: [...ctx.getImageData(c.width / 2, c.height / 2, 1, 1).data],
      nonzero: ctx.getImageData(0, 0, c.width, c.height).data.some(value => value !== 0) }];
  })));
}
async function waitPl(page) {
  await page.waitForFunction(() => document.getElementById("grayPreviewBackend").textContent === "PL");
}
async function waitBrowser(page) {
  await page.waitForFunction(() => document.getElementById("grayPreviewBackend").textContent === "网页预览");
}

try {
  browser = await playwright.chromium.launch({ headless: true,
    ...(process.argv[3] ? { executablePath: process.argv[3] } : process.platform === "win32" ? { channel: "msedge" } : {}) });
  const generator = await browser.newPage();
  const generated = await generator.evaluate(() => {
    const canvas = document.createElement("canvas"), ctx = canvas.getContext("2d");
    canvas.width = 640; canvas.height = 480;
    ctx.fillStyle = "#752840"; ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = "#27d3a0"; ctx.fillRect(100, 80, 300, 280);
    const camera = canvas.toDataURL("image/jpeg", .95).split(",")[1];
    canvas.width = 1280;
    ctx.fillStyle = "rgb(180,180,180)"; ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = "rgb(45,45,45)"; ctx.fillRect(640, 0, 640, 480);
    return { camera, pl: canvas.toDataURL("image/jpeg", .95).split(",")[1] };
  });
  await generator.close();
  const cameraJpeg = Buffer.from(generated.camera, "base64"), plJpeg = Buffer.from(generated.pl, "base64");
  let sequence = 0, captured = 0, stream = "camera-first", source = "camera", cameraMode = "ok", mode = "ok";
  let cameraRequests = 0, cameraActive = 0, cameraMax = 0, plActive = 0, plMax = 0, delay = 0, snapshot = null;
  const pollTimes = [];
  const cameraUrl = await listen(createServer(async (request, response) => {
    if (request.url.startsWith("/vision/frame.jpg")) {
      cameraRequests++; cameraActive++; cameraMax = Math.max(cameraMax, cameraActive);
      try {
        if (cameraMode === "error") { json(response, { error: "camera offline" }, 503); return; }
        if (cameraMode !== "frozen") { sequence++; captured = performance.now(); }
        await new Promise(resolve => setTimeout(resolve, 8));
        response.writeHead(200, { "Content-Type": "image/jpeg", "X-Vision-Sequence": String(sequence),
          "X-Vision-Captured-Ms": String(captured), "X-Vision-Stream": stream, "X-Vision-Source": source });
        response.end(cameraJpeg);
      } finally { cameraActive--; }
    } else if (request.url === "/api/audio") json(response, { status: "hardware_unavailable", stale: true, notes: [] });
    else json(response, { error: "Unexpected camera route" }, 404);
  }));
  const plUrl = await listen(createServer(async (request, response) => {
    plActive++; plMax = Math.max(plMax, plActive);
    const requestMode = mode;
    try {
      if (delay) await new Promise(resolve => setTimeout(resolve, delay));
      if (request.url === "/api/vision/pl/status") {
        pollTimes.push(performance.now());
        if (mode !== "duplicate") snapshot = { sequence, captured_ms: captured, stream_id: stream, source };
        if (mode === "http_error") { json(response, { error: "hardware unavailable" }, 503); return; }
        if (mode === "invalid_json") { response.end("not JSON"); return; }
        json(response, { backend: mode === "status_cpu" ? "cpu" : "pl",
          status: mode === "error" ? "error" : mode === "processing" ? "processing" : "ready",
          stale: mode === "stale", age_ms: mode === "stale" ? 1500 : mode === "future" ? 0 : 15,
          error: mode === "error" ? "DMA failed" : null, ...snapshot,
          ...(mode === "wrong_stream" ? { stream_id: "obsolete" } : {}),
          ...(mode === "wrong_source" ? { source: "synthetic" } : {}),
          ...(mode === "future" ? { captured_ms: captured + 10000 } : {}),
          hardware: { dma: "fixture" }, timings: { dma_ms: 3, packing_ms: 2 }, processing_ms: 5, processing_fps: 5 });
      } else if (request.url === "/vision/pl.jpg") {
        const headers = { "Content-Type": "image/jpeg", "X-Vision-Sequence": String(snapshot.sequence),
          "X-Vision-Captured-Ms": String(snapshot.captured_ms), "X-Vision-Stream": snapshot.stream_id,
          "X-Vision-Source": snapshot.source, "X-Vision-Backend": requestMode === "cpu" ? "cpu" : "pl",
          "X-Vision-Processing-Ms": "5.25" };
        if (requestMode === "missing") delete headers["X-Vision-Captured-Ms"];
        if (requestMode === "missing_backend") delete headers["X-Vision-Backend"];
        if (requestMode === "bad_timing") headers["X-Vision-Processing-Ms"] = "NaN";
        if (requestMode === "image_stream") headers["X-Vision-Stream"] = "obsolete";
        if (requestMode === "image_sequence") headers["X-Vision-Sequence"] = String(snapshot.sequence - 1);
        if (requestMode === "image_capture") headers["X-Vision-Captured-Ms"] = String(snapshot.captured_ms + 1);
        response.writeHead(200, headers); response.end(requestMode === "bad_size" ? cameraJpeg : plJpeg);
      } else if (request.url === "/api/vision/pc/process") {
        for await (const _chunk of request) { /* Drain the fixture camera frame. */ }
        const options = JSON.parse(request.headers["x-vision-options"] || "{}");
        const planes = Buffer.alloc(320 * 240 * 2);
        planes.fill(180, 0, 320 * 240);
        planes.fill(45, 320 * 240);
        response.writeHead(200, {
          "Content-Type": options.coordinates_only ? "application/json" : "application/x-fretlens-planes",
          "X-Vision-Backend": "pl",
          "X-Vision-Source": "pc_camera",
          "X-Vision-Token": request.headers["x-vision-token"] || "",
          "X-Vision-Processing-Ms": "5.25"
        });
        response.end(options.coordinates_only ? "{}" : planes);
      } else json(response, { error: "Unexpected PL route" }, 404);
    } finally { plActive--; }
  }));
  const base = await startProxy(cameraUrl, plUrl);
  const statusResponse = await fetch(base + "/api/vision/pl/status");
  assert.equal((await statusResponse.json()).backend, "pl");
  assert.equal(statusResponse.headers.get("Cache-Control"), "no-store");
  const imageResponse = await fetch(base + "/vision/pl.jpg");
  assert.deepEqual(Buffer.from(await imageResponse.arrayBuffer()), plJpeg);
  for (const name of ["Sequence", "Captured-Ms", "Stream", "Source", "Backend", "Processing-Ms"])
    assert.ok(imageResponse.headers.has("X-Vision-" + name), name);
  const cameraResponse = await fetch(base + "/vision/frame.jpg");
  assert.equal(cameraResponse.headers.get("X-Vision-Backend"), null);
  assert.equal(cameraResponse.headers.get("X-Vision-Source"), "camera");
  for (const path of ["/api/vision/pl/status", "/vision/pl.jpg"])
    assert.equal((await fetch(base + path, { method: "POST" })).status, 405);

  for (const width of [1440, 360]) {
    mode = "ok"; cameraMode = "ok"; delay = 0;
    const page = await browser.newPage({ viewport: { width, height: 960 }, serviceWorkers: "block" });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(() => {
      const original = window.fetch.bind(window);
      window.plRequests = { active: 0, max: 0, starts: [] };
      window.fetch = async (...args) => {
        const pl = args[0] === "/api/vision/pl/status" || args[0] === "/vision/pl.jpg";
        if (pl) {
          if (args[0] === "/api/vision/pl/status") window.plRequests.starts.push(performance.now());
          window.plRequests.active++;
          window.plRequests.max = Math.max(window.plRequests.max, window.plRequests.active);
        }
        try { return await original(...args); }
        finally { if (pl) window.plRequests.active--; }
      };
    });
    await page.route("**/*", route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
    await page.goto(base);
    await page.locator("#debugDetails").evaluate(node => { node.open = true; });
    await page.locator("#autoFretboard").uncheck();
    await page.locator("#boardVisionButton").click();
    await waitPl(page);
    let values = await pixels(page);
    assert.ok(Math.abs(values.grayPreview.center[0] - 180) < 3, JSON.stringify(values));
    assert.ok(Math.abs(values.edgePreview.center[0] - 45) < 3, JSON.stringify(values));
    assert.ok(Math.abs(values.fretOverlay.center[0] - values.fretOverlay.center[1]) > 30,
      "Original view must retain the camera color pixels");
    await page.locator("#debugDetails").evaluate(node => { node.open = true; });
    for (const [view, expected] of [["gray", 154], ["edge", 0]]) {
      await page.locator("#visionView").selectOption(view);
      await page.waitForFunction(value => {
        const c = document.getElementById("fretOverlay");
        return Math.abs(c.getContext("2d").getImageData(c.width / 2, c.height / 2, 1, 1).data[0] - value) < 3;
      }, expected);
      assert.match(await page.locator(`#visionView option[value="${view}"]`).textContent(), /网页预览/);
    }
    // The main view keeps browser pixels and camera geometry even while PL tiles are valid.
    await page.locator("#followNeck").uncheck();
    await page.locator("#calibrationSettings").evaluate(node => { node.open = true; });
    await page.locator("#calibrateButton").click();
    await waitPl(page);
    await page.locator("#fretOverlay").scrollIntoViewIfNeeded();
    const r = await page.locator("#fretOverlay").boundingBox();
    for (const [x, y] of [[.3, .3], [.7, .3], [.7, .7], [.3, .7]])
      await page.mouse.click(r.x + r.width * x, r.y + r.height * y);
    assert.ok(await page.evaluate(() => window.FretVision.trackingSnapshot().calibration));
    await page.waitForFunction(() => document.getElementById("cameraStage").dataset.processingBackend === "browser");
    const annotationPixels = await page.locator("#fretOverlay").evaluate(c => {
      const data = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      let colored = 0;
      for (let i = 0; i < data.length; i += 4)
        if (Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]) > 20) colored++;
      return { colored, center: [...c.getContext("2d").getImageData(c.width / 2, c.height / 2, 1, 1).data],
        view: document.getElementById("visionView").value, state: document.getElementById("cameraStage").dataset.processingBackend,
        status: document.getElementById("visionStatus").textContent };
    });
    assert.ok(annotationPixels.colored > 20, JSON.stringify(annotationPixels));
    await page.screenshot({ path: join(evidence, `pl-${width}.png`), fullPage: true });
    const countBefore = cameraRequests, throughputStart = performance.now();
    const browserThroughputStart = await page.evaluate(() => performance.now());
    await page.waitForTimeout(1200);
    const cameraFps = (cameraRequests - countBefore) * 1000 / (performance.now() - throughputStart);
    assert.ok(cameraFps > 18, `Camera ${cameraFps.toFixed(1)} fps`);
    const recent = await page.evaluate(since => window.plRequests.starts.filter(time => time > since), browserThroughputStart);
    const measuredWindow = await page.evaluate(since => performance.now()-since, browserThroughputStart);
    assert.ok(recent.length <= Math.ceil(measuredWindow/33)+2);
    for (let i = 1; i < recent.length; i++) assert.ok(recent[i] - recent[i - 1] >= 25);

    const invalidCases = ["cpu", "status_cpu", "stale", "error", "processing", "wrong_stream", "image_stream",
      "future", "missing", "bad_timing", "bad_size", "http_error", "invalid_json", "missing_backend",
      "wrong_source", "image_sequence", "image_capture"];
    for (const bad of invalidCases) {
      mode = bad;
      await waitBrowser(page);
      assert.equal(await page.locator("#cameraStage").getAttribute("data-processing-backend"), "browser", bad);
      assert.ok((await pixels(page)).grayPreview.nonzero, `${bad}: camera fallback must remain visible`);
      mode = "ok"; await waitPl(page);
    }
    mode = "duplicate";
    await waitBrowser(page);
    assert.ok((await pixels(page)).fretOverlay.nonzero, "Expired PL duplicates must fall back to the live camera");
    mode = "ok"; await waitPl(page);
    cameraMode = "frozen";
    await page.waitForFunction(() => document.getElementById("grayPreviewBackend").textContent === "不可用");
    values = await pixels(page);
    assert.ok(Object.values(values).every(value => !value.nonzero), "Stale camera must clear all outputs");
    cameraMode = "ok"; await waitPl(page);
    mode = "duplicate"; stream = "camera-restarted-" + width; sequence = 0;
    await waitBrowser(page);
    assert.ok((await pixels(page)).fretOverlay.nonzero, "Camera restart must continue while PL catches up");
    mode = "ok"; await waitPl(page);
    cameraMode = "error";
    await page.waitForFunction(() => document.getElementById("grayPreviewBackend").textContent === "不可用");
    assert.ok(Object.values(await pixels(page)).every(value => !value.nonzero));
    cameraMode = "ok"; await waitPl(page);
    // Crossing ten minutes on the browser clock must not mix it with board capture timestamps.
    await page.evaluate(() => {
      const original = performance.now.bind(performance);
      Object.defineProperty(performance, "now", { configurable: true, value: () => original() + 600000 });
    });
    await waitBrowser(page); await waitPl(page);
    delay = 1500;
    await waitBrowser(page);
    assert.ok((await pixels(page)).fretOverlay.nonzero, "PL deadline must leave the current camera available");
    // Let the fixture's aborted upstream finish before measuring single-request behavior again.
    await page.waitForTimeout(1600);
    delay = 0; await waitPl(page);
    delay = 300;
    const slowStart = cameraRequests;
    await page.waitForTimeout(1200);
    assert.ok(cameraRequests - slowStart > 20, "Slow PL requests must not stall original frames");
    await page.locator("#visionStopButton").click();
    await page.waitForTimeout(400);
    const stoppedAt = pollTimes.length;
    await page.waitForTimeout(800);
    assert.equal(pollTimes.length, stoppedAt, "Stop must cancel PL polling");
    assert.equal(await page.locator("#grayPreviewBackend").textContent(), "不可用");
    assert.ok(Object.values(await pixels(page)).every(value => !value.nonzero));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.equal(await page.evaluate(() => window.plRequests.max), 1, "Browser PL requests must remain serial even after timeouts");
    assert.deepEqual(errors, []);
    report.push({ width, camera_fps: cameraFps, composite_pixels: "180 gray / 45 edge", invalid_metadata_cases: invalidCases.length,
      stale_duplicates: "pass", camera_stale_error_restart: "pass", main_view: "browser pixels and camera geometry; PL tiles only",
      slow_pl_camera_independence: "pass", request_timeout_fallback: "pass", ten_minute_browser_clock: "pass", stop: "pass" });
    await page.close();
  }
  assert.equal(cameraMax, 1);
  // Verify the default follows FPGA_VISION_URL's host while selecting port 8083.
  const defaultUrl = await listen(createServer((request, response) => json(response, { backend: "pl", default_port: true })),
    8083, "127.0.0.2");
  const defaultProxy = await startProxy("http://127.0.0.2:18082");
  assert.equal((await (await fetch(defaultProxy + "/api/vision/pl/status")).json()).default_port, true, defaultUrl);
  console.log(JSON.stringify({ proxy_override_and_default_8083: "pass", max_camera_requests: cameraMax,
    max_browser_pl_requests: 1, max_fixture_pl_requests_including_aborted: plMax, results: report, screenshots: evidence }, null, 2));
} finally {
  await browser?.close();
  for (const child of children) if (child.exitCode === null && child.signalCode === null) {
    const exited = once(child, "exit"); child.kill(); await exited;
  }
  for (const server of servers) {
    server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
  }
}
