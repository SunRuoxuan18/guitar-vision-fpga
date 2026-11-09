import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdtemp } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Self-contained HTTP/JPEG/model fixtures: this suite never proxies to hardware.
const require = createRequire(import.meta.url);
let playwright;
try { playwright = require(process.argv[2] || "playwright"); }
catch (error) {
  if (process.argv[2] || !process.env.USERPROFILE) throw error;
  playwright = require(join(process.env.USERPROFILE,
    ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
}
const root = dirname(fileURLToPath(import.meta.url));
const server = createServer(async (request, response) => {
  const pathname = new URL(request.url, "http://localhost").pathname;
  const target = resolve(root, "." + (pathname === "/" ? "/index.html" : pathname));
  if (!target.startsWith(root + "/") && !target.startsWith(root + "\\")) {
    response.writeHead(403); response.end(); return;
  }
  try {
    const data = await readFile(target);
    response.writeHead(200, { "Content-Type": ({ ".html": "text/html", ".js": "text/javascript",
      ".mjs": "text/javascript", ".css": "text/css" })[extname(target)] || "application/octet-stream" });
    response.end(data);
  } catch { response.writeHead(404); response.end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const evidence = await mkdtemp(join(tmpdir(), "fretlens-roi-"));
const corners = [{ x: .23, y: .38 }, { x: .76, y: .32 }, { x: .8, y: .56 }, { x: .26, y: .62 }];
const report = [];
let browser;

async function drawing(page) {
  return page.evaluate(() => window.roiFixture.draw);
}
function isFull(draw) {
  return Math.abs(draw.region.x) < .001 && Math.abs(draw.region.y) < .001 &&
    Math.abs(draw.region.width - 1) < .001 && Math.abs(draw.region.height - 1) < .001;
}
function checkCrop(draw, points) {
  const r = draw.region;
  assert.ok(r.width < .8 && r.height < .6, JSON.stringify(draw));
  assert.ok(r.x >= 0 && r.y >= 0 && r.x + r.width <= 1.00001 && r.y + r.height <= 1.00001);
  for (const p of points) assert.ok(p.x > r.x && p.x < r.x + r.width && p.y > r.y && p.y < r.y + r.height,
    "All four corners need visible padding");
}
async function waitCrop(page) {
  await page.waitForFunction(() => window.roiFixture.draw?.region.width < .8);
  return drawing(page);
}
async function waitFull(page) {
  await page.waitForFunction(() => window.roiFixture.draw?.region.width === 1);
  assert.equal(isFull(await drawing(page)), true);
}
async function clickPoints(page, points) {
  await page.locator("#fretOverlay").scrollIntoViewIfNeeded();
  const draw = await drawing(page);
  const rect = await page.locator("#fretOverlay").evaluate(canvas => {
    const r = canvas.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width / canvas.width, height: r.height / canvas.height };
  });
  for (const p of points) {
    const x = rect.x + (draw.x + (p.x - draw.region.x) / draw.region.width * draw.width) * rect.width;
    const y = rect.y + (draw.y + (p.y - draw.region.y) / draw.region.height * draw.height) * rect.height;
    await page.mouse.click(x, y);
  }
}
async function beginCalibration(page) {
  await page.locator("#calibrationSettings").evaluate(node => { node.open = true; });
  await page.locator("#calibrateButton").click();
  await page.waitForFunction(() => document.querySelector("#cameraStage").dataset.calibrating === "true");
}
async function outlinePixels(page, points, expected = true) {
  const counts = await page.evaluate(points => {
    const c = document.querySelector("#fretOverlay"), ctx = c.getContext("2d");
    const draw = window.roiFixture.draw, r = draw.region;
    return points.map((p, i) => {
      const q = points[(i + 1) % points.length];
      const x = Math.round(draw.x + ((p.x + q.x) / 2 - r.x) / r.width * draw.width);
      const y = Math.round(draw.y + ((p.y + q.y) / 2 - r.y) / r.height * draw.height);
      const data = ctx.getImageData(Math.max(0, x - 4), Math.max(0, y - 4), 9, 9).data;
      let red = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i] > 200 && data[i + 1] < 90 && data[i + 2] < 90) red++;
      return red;
    });
  }, points);
  assert.ok(expected ? counts.every(count => count > 3) : counts.every(count => count === 0), JSON.stringify(counts));
}
async function checkPreviews(page) {
  await page.waitForFunction(() => document.querySelector("#edgePreview").getContext("2d")
    .getImageData(0, 0, document.querySelector("#edgePreview").width, document.querySelector("#edgePreview").height)
    .data.some((value, i) => i % 4 !== 3 && value > 200));
  const values = await page.evaluate(() => ["grayPreview", "edgePreview"].map(id => {
    const canvas = document.getElementById(id), data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let bright = 0, black = 0, nonGray = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 20) bright++;
      if (data[i] === 0) black++;
      if (data[i] !== data[i + 1] || data[i] !== data[i + 2]) nonGray++;
    }
    return { id, width: canvas.width, height: canvas.height, bright, black, nonGray };
  }));
  for (const value of values) {
    assert.ok(value.width <= 320 && value.height <= 180);
    assert.ok(value.bright > 30, JSON.stringify(value));
    assert.equal(value.nonGray, 0);
  }
  assert.ok(values[1].black > values[1].bright, "Sobel should contain thin edges with a dark background");
  return values;
}
async function cleared(page) {
  await page.waitForFunction(() => ["grayPreview", "edgePreview", "fretOverlay"].every(id => {
    const c = document.getElementById(id);
    return c.getContext("2d").getImageData(0, 0, c.width, c.height).data.every(value => value === 0);
  }));
}

try {
  browser = await playwright.chromium.launch({ headless: true,
    ...(process.argv[3] ? { executablePath: process.argv[3] } : process.platform === "win32" ? { channel: "msedge" } : {}) });
  for (const width of [1440, 360]) {
    const page = await browser.newPage({ viewport: { width, height: 960 }, isMobile: width === 360,
      hasTouch: width === 360, serviceWorkers: "block" });
    const errors = [], external = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(() => {
      window.roiFixture = { draw: null, reads: [], inferences: [], tip: { x: .5, y: .46 } };
      const originalDraw = CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage = function (frame, ...args) {
        if (this.canvas.id === "fretOverlay" && args.length === 8) {
          const fw = frame.videoWidth || frame.width, fh = frame.videoHeight || frame.height;
          const [sx, sy, sw, sh, x, y, width, height] = args;
          window.roiFixture.draw = { x, y, width, height,
            region: { x: sx / fw, y: sy / fh, width: sw / fw, height: sh / fh } };
        }
        return originalDraw.call(this, frame, ...args);
      };
      const originalRead = CanvasRenderingContext2D.prototype.getImageData;
      CanvasRenderingContext2D.prototype.getImageData = function (...args) {
        if (!this.canvas.id && this.canvas.width <= 320 && this.canvas.height <= 180)
          window.roiFixture.reads.push({ time: performance.now(), width: this.canvas.width, height: this.canvas.height });
        return originalRead.apply(this, args);
      };
    });
    await page.route("**/*", async route => {
      if (new URL(route.request().url()).origin !== base) { external.push(route.request().url()); await route.abort(); return; }
      const pathname = new URL(route.request().url()).pathname;
      if (pathname.startsWith("/api/")) { await route.fulfill({ status: 503, body: "Offline fixture" }); return; }
      await route.continue();
    });
    await page.route("**/vendor/mediapipe/vision_bundle.mjs", route => route.fulfill({
      contentType: "text/javascript", body: `
        export const FilesetResolver = { forVisionTasks: async () => ({}) };
        export const HandLandmarker = { createFromOptions: async () => ({
          detectForVideo(frame) {
            const canvas = document.createElement("canvas");
            canvas.width = 640; canvas.height = 400;
            const ctx = canvas.getContext("2d"); ctx.drawImage(frame, 0, 0, 640, 400);
            window.roiFixture.inferences.push({ width: frame.videoWidth || frame.width, height: frame.videoHeight || frame.height,
              originalCornerPixel: [...ctx.getImageData(10, 10, 1, 1).data] });
            const landmarks = Array.from({length: 21}, () => ({x: -1, y: -1, z: 0}));
            landmarks[8] = {...window.roiFixture.tip, z: 0};
            return {landmarks: [landmarks], handedness: [[{score: .99, categoryName: "Fixture"}]]};
          }, close() {}
        }) };
      ` }));
    await page.goto(base);
    const fixture = await page.evaluate(points => {
      const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 400;
      const ctx = canvas.getContext("2d"); ctx.fillStyle = "#314a7a"; ctx.fillRect(0, 0, 640, 400);
      ctx.fillStyle = "#ff00ff"; ctx.fillRect(0, 0, 40, 40);
      ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p.x * 640, p.y * 400) : ctx.moveTo(p.x * 640, p.y * 400));
      ctx.closePath(); ctx.fillStyle = "#8c9a3f"; ctx.fill();
      const geometry = Fretboard.createCalibration(points);
      const line = (a, b) => {
        ctx.beginPath(); ctx.moveTo(a.x * 640, a.y * 400); ctx.lineTo(b.x * 640, b.y * 400);
        ctx.strokeStyle = "white"; ctx.lineWidth = 2; ctx.stroke();
        return { observed: true, start: a, end: b };
      };
      const point = (x, y) => Fretboard.transform(geometry.toImage, { x, y });
      const strings = Array.from({ length: 6 }, (_, i) => line(point(0, i / 5), point(1, i / 5)));
      const frets = [0, .2, .4, .6, .8, 1].map(x => line(point(x, 0), point(x, 1)));
      const tip = point(.5, .6);
      window.roiFixture.tip = tip;
      return { image: canvas.toDataURL("image/jpeg").split(",")[1], strings, frets, tip };
    }, corners);
    const image = Buffer.from(fixture.image, "base64");
    let sequence = 0, stream = "offline-roi", mode = "valid", trackingCorners = corners;
    let tracker = null, active = 0, maxActive = 0;
    await page.route("**/api/vision/tracking", async route => {
      const body = route.request().postDataJSON();
      assert.ok(body.sequence >= 0 && body.stream_id === stream);
      trackingCorners = body.corners; tracker = "offline-tracker";
      await new Promise(resolve => setTimeout(resolve, 150));
      await route.fulfill({ json: { tracker_id: tracker, status: "tracking" } });
    });
    await page.route("**/api/vision/tracking/stop", route => route.fulfill({ json: { status: "stopped" } }));
    await page.route("**/vision/frame.jpg*", async route => {
      active++; maxActive = Math.max(maxActive, active);
      try {
        if (mode === "error") { await route.fulfill({ status: 503, body: "offline" }); return; }
        if (mode !== "frozen") sequence++;
        const auto = { status: mode === "searching" ? "searching" : "tracking", sequence,
          stream_id: stream, corners, strings: fixture.strings, frets: fixture.frets };
        if (mode === "wrong_sequence") auto.sequence--;
        if (mode === "wrong_stream") auto.stream_id = "obsolete";
        if (mode === "invalid") auto.corners = [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }];
        const tracking = { tracker_id: tracker, sequence, stream_id: stream,
          status: mode === "lost" ? "lost" : "tracking", corners: trackingCorners };
        const url = new URL(route.request().url());
        await route.fulfill({ contentType: "image/jpeg", body: image, headers: {
          "X-Vision-Sequence": String(sequence), "X-Vision-Source": "camera", "X-Vision-Stream": stream,
          "X-Vision-Captured-Ms": String(performance.now()),
          ...(url.searchParams.has("auto") && mode !== "missing" ? { "X-Vision-Fretboard": JSON.stringify(auto) } : {}),
          ...(url.searchParams.has("tracking") ? { "X-Vision-Tracking": JSON.stringify(tracking) } : {})
        } });
      } finally { active--; }
    });
    assert.equal(await page.locator("#focusFretboard").isChecked(), true);
    await page.locator("#boardVisionButton").click();
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().automatic?.geometry);
    checkCrop(await waitCrop(page), corners);
    await outlinePixels(page, corners);
    const previews = await checkPreviews(page);
    await page.locator("#handModelButton").click();
    await page.waitForFunction(() => window.FretVision.snapshot()?.source === "board_camera");
    const tip = await page.evaluate(() => window.FretVision.snapshot().hands[0].landmarks[8]);
    assert.deepEqual(tip, { ...fixture.tip, z: 0 });
    await page.waitForFunction(() => (window.FretVision.snapshot()?.positions.length || 0) > 0);
    await page.waitForTimeout(1100);
    const reads = await page.evaluate(() => window.roiFixture.reads);
    assert.ok(reads.length >= 5, "Live previews must continue updating");
    for (let i = 1; i < reads.length; i++) assert.ok(reads[i].time - reads[i - 1].time >= 99,
      `Processing exceeded 10 Hz: ${reads[i].time - reads[i - 1].time}ms`);
    assert.ok(reads.every(read => read.width <= 320 && read.height <= 180));
    assert.ok((await page.evaluate(() => window.roiFixture.inferences)).every(size => size.width === 640 && size.height === 400));
    assert.ok((await page.evaluate(() => window.roiFixture.inferences)).every(({ originalCornerPixel: p }) =>
      p[0] > 220 && p[1] < 40 && p[2] > 220), "Inference must retain the uncropped image pixels");
    await page.screenshot({ path: join(evidence, `roi-${width}.png`), fullPage: true });

    await page.locator("#focusFretboard").uncheck();
    await waitFull(page); await outlinePixels(page, corners);
    await page.locator("#focusFretboard").check();
    checkCrop(await waitCrop(page), corners);
    await page.waitForFunction(() => window.FretVision.snapshot()?.positions.length > 0);
    const marker = await page.evaluate(tip => {
      window.FretVision.render();
      const c = document.querySelector("#fretOverlay"), d = window.roiFixture.draw, r = d.region;
      const x = d.x + (tip.x - r.x) / r.width * d.width, y = d.y + (tip.y - r.y) / r.height * d.height;
      const pixels = c.getContext("2d").getImageData(Math.round(x - 9), Math.round(y - 9), 19, 19).data;
      let yellow = 0;
      for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > 180 && pixels[i + 1] > 200 && pixels[i + 2] < 140) yellow++;
      return yellow;
    }, fixture.tip);
    assert.ok(marker > 10, "Normalized fingertip must project into the cropped view");
    for (const invalid of ["searching", "missing", "invalid", "wrong_sequence", "wrong_stream"]) {
      mode = invalid;
      await waitFull(page); await outlinePixels(page, corners, false);
      mode = "valid";
      await waitCrop(page);
    }
    mode = "frozen";
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().automatic === null);
    await cleared(page);
    assert.equal(await page.evaluate(() => window.FretVision.snapshot()), null);
    mode = "valid"; await waitCrop(page);

    await page.locator("#followNeck").uncheck();
    await waitCrop(page);
    const previous = (await drawing(page)).region;
    await beginCalibration(page);
    assert.deepEqual((await drawing(page)).region, previous, "Calibration must hold the crop used for clicking");
    await page.locator("#focusFretboard").uncheck(); await waitFull(page);
    await page.locator("#focusFretboard").check(); await waitCrop(page);
    // A letterbox click cannot become one of the four normalized corners.
    await page.locator("#fretOverlay").dispatchEvent("pointerdown", { clientX: 0, clientY: 0 });
    await clickPoints(page, corners);
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().calibration !== null);
    const manual = await page.evaluate(() => window.FretVision.trackingSnapshot().calibration.corners);
    manual.forEach((p, i) => assert.ok(Math.hypot(p.x - corners[i].x, p.y - corners[i].y) < .003));
    checkCrop(await waitCrop(page), manual);
    await outlinePixels(page, manual);
    await page.locator("#clearCalibrationButton").click(); await waitFull(page);
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().calibration), null);

    await page.locator("#followNeck").check();
    await beginCalibration(page); await clickPoints(page, corners);
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().tracking?.status === "tracking");
    checkCrop(await waitCrop(page), corners);
    trackingCorners = corners.map(p => ({ x: p.x - .08, y: p.y + .05 }));
    await page.waitForFunction(x => Math.abs(window.FretVision.trackingSnapshot().calibration.corners[0].x - x) < .002, trackingCorners[0].x);
    await outlinePixels(page, trackingCorners);
    checkCrop(await drawing(page), trackingCorners);
    mode = "lost";
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().tracking?.status === "lost");
    await waitFull(page);
    assert.equal(await page.evaluate(() => window.FretVision.trackingSnapshot().calibration), null);
    mode = "valid"; await page.waitForTimeout(150); await waitFull(page);
    await beginCalibration(page); await clickPoints(page, corners);
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().tracking?.status === "tracking");
    mode = "frozen";
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().tracking?.status === "lost");
    await cleared(page);
    mode = "valid";
    await waitFull(page);
    await page.locator("#autoFretboard").check(); await waitCrop(page);
    stream = "restarted-offline-roi"; mode = "missing";
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().automatic === null);
    await waitFull(page);
    mode = "valid"; await waitCrop(page);
    mode = "error";
    await page.waitForFunction(() => document.querySelector("#visionStatus").textContent.includes("正在自动重连"));
    await cleared(page);
    mode = "valid"; await waitCrop(page);
    await page.locator("#visionStopButton").click(); await cleared(page);
    assert.equal(await page.evaluate(() => window.FretVision.active), false);
    assert.equal(maxActive, 1);

    // A local canvas stream exercises the video element path without a real camera.
    await page.evaluate(async image => {
      const img = new Image(); img.src = "data:image/jpeg;base64," + image; await img.decode();
      const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 400;
      canvas.getContext("2d").drawImage(img, 0, 0);
      const stream = canvas.captureStream(15), video = document.querySelector("#cameraVideo");
      video.srcObject = stream; await video.play();
      window.roiFixture.localStream = stream;
      window.FretVision.attachCamera(stream);
    }, fixture.image);
    await waitFull(page);
    await beginCalibration(page); await clickPoints(page, corners);
    await page.waitForFunction(() => window.FretVision.trackingSnapshot().calibration !== null);
    checkCrop(await waitCrop(page), corners);
    await outlinePixels(page, corners);
    await checkPreviews(page);
    await page.locator("#handModelButton").click();
    await page.waitForFunction(() => window.FretVision.snapshot()?.source === "camera");
    assert.deepEqual(await page.evaluate(() => window.FretVision.snapshot().hands[0].landmarks[8]), tip);
    await page.locator("#debugDetails").evaluate(node => { node.open = true; });
    for (const view of ["gray", "edge", "color"]) {
      await page.locator("#visionView").selectOption(view);
      await page.waitForTimeout(120);
      checkCrop(await drawing(page), corners);
      await outlinePixels(page, corners);
    }
    const layout = await page.evaluate(() => {
      const ids = ["cameraStage", "visionStatus", "grayPreview", "edgePreview"];
      const rects = Object.fromEntries(ids.map(id => {
        const r = document.getElementById(id).getBoundingClientRect();
        return [id, { left: r.left, right: r.right, top: r.top, bottom: r.bottom }];
      }));
      const controls = [...document.querySelector(".vision-controls").children].map(node => node.getBoundingClientRect());
      return { overflow: document.documentElement.scrollWidth > innerWidth,
        overlap: controls.some((a, i) => controls.slice(i + 1).some(b => a.left < b.right - 1 && a.right > b.left + 1 && a.top < b.bottom - 1 && a.bottom > b.top + 1)),
        previewsBelow: rects.grayPreview.top > rects.cameraStage.bottom && rects.edgePreview.top > rects.cameraStage.bottom };
    });
    assert.deepEqual(layout, { overflow: false, overlap: false, previewsBelow: true });
    await page.evaluate(() => window.roiFixture.localStream.getTracks().forEach(track => track.stop()));
    await page.locator("#visionStopButton").click(); await cleared(page);
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    report.push({ width, crop: "pass", red_perimeter: "pass", original_frame_inference: "pass",
      cropped_manual_clicks: "pass", invalid_geometry_fallback: "pass", tracking_loss_stale_restart: "pass",
      local_video: "pass", max_parallel_requests: maxActive, thumbnails: previews,
      minimum_processing_interval_ms: Math.min(...reads.slice(1).map((read, i) => read.time - reads[i].time)), layout });
    await page.close();
  }
  console.log(JSON.stringify({ report, screenshots: evidence }, null, 2));
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
