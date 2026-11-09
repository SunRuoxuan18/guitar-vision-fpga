import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { pathToFileURL, fileURLToPath } from "node:url";
const { chromium } = await import(pathToFileURL(process.argv[2]).href);
const evidence = new URL("../diagnostics/vision_browser/", import.meta.url);
const baseUrl = process.argv[4] || "http://127.0.0.1:8000/";
await mkdir(evidence, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: process.argv[3],
  args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
const errors = [], results = [];
try {
  for (const width of [1440, 1024, 360]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, permissions: ["camera"] });
    page.on("console", (message) => { if (message.type() === "error") console.log(`BROWSER: ${message.text()}`); });
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(baseUrl);
    assert.equal(await page.locator("#chordName").textContent(), "--");
    await page.locator("#visionTestButton").click();
    await page.waitForFunction(() => window.FretVision.snapshot()?.positions.length === 3);
    assert.match(await page.locator("#visualEvidence").textContent(), /6弦3品.*5弦2品.*1弦3品.*SYNTHETIC/);
    assert.equal(await page.locator("#chordName").textContent(), "--");
    const layout = await page.evaluate(() => {
      const canvas = document.querySelector("#fretOverlay");
      const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      let bright = 0;
      for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > 90 || pixels[i + 1] > 90) bright++;
      const controls = [...document.querySelectorAll(".vision-controls button,.vision-controls label")]
        .map((node) => { const r = node.getBoundingClientRect(); return { x: r.x, y: r.y, right: r.right, bottom: r.bottom }; });
      const overlaps = controls.some((r, i) => controls.slice(i + 1).some((b) =>
        r.x < b.right - 1 && r.right > b.x + 1 && r.y < b.bottom - 1 && r.bottom > b.y + 1));
      return { bright, overflow: document.documentElement.scrollWidth > innerWidth, overlaps };
    });
    assert.ok(layout.bright > 1500);
    assert.equal(layout.overflow, false);
    assert.equal(layout.overlaps, false);
    await page.screenshot({ path: fileURLToPath(new URL(`synthetic-${width}.png`, evidence)), fullPage: true });
    await page.locator("#clearCalibrationButton").click();
    await page.waitForFunction(() => window.FretVision.snapshot()?.status === "uncalibrated");
    assert.equal(await page.locator("#chordName").textContent(), "--");
    await page.locator("#visionStopButton").click();
    assert.equal(await page.evaluate(() => window.FretVision.active), false);
    assert.equal(await page.evaluate(() => window.FretVision.snapshot()), null);
    results.push({ width, synthetic: "pass", clear: "pass", stop: "pass", layout });
    console.log(`PASS ${width}px: synthetic geometry, empty result, stop, layout`);
    if (width === 1440) {
      // A generated recording validates decoding and geometry, without inventing hand detections.
      const recording = await page.evaluate(async () => {
        const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 480;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#181818"; ctx.fillRect(0, 0, 640, 480);
        ctx.fillStyle = "#bbbbbb"; ctx.fillRect(64, 96, 512, 288);
        ctx.fillStyle = "#ffffff"; ctx.fillRect(192, 96, 4, 288);
        const stream = canvas.captureStream(0), chunks = [];
        const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp8" });
        recorder.ondataavailable = (event) => chunks.push(event.data);
        const stopped = new Promise((resolve) => recorder.onstop = resolve);
        recorder.start();
        for (let i = 0; i < 20; i++) {
          ctx.fillStyle = i % 2 ? "#fefefe" : "#ffffff"; ctx.fillRect(192, 96, 4, 288);
          stream.getVideoTracks()[0].requestFrame();
          await new Promise((r) => setTimeout(r, 50));
        }
        recorder.stop(); await stopped; stream.getTracks().forEach((t) => t.stop());
        return [...new Uint8Array(await new Blob(chunks).arrayBuffer())];
      });
      console.log(`Generated recording: ${recording.length} bytes`);
      await page.locator("#visionFile").setInputFiles({ name: "synthetic-recording.webm", mimeType: "video/webm", buffer: Buffer.from(recording) });
      await page.waitForFunction(() => document.querySelector("#cameraStage").dataset.visionSource === "video_file");
      assert.equal(await page.locator("#cameraVideo").evaluate((node) => getComputedStyle(node).objectFit), "contain");
      assert.equal(await page.locator("#cameraVideo").evaluate((node) => getComputedStyle(node).transform), "none");
      await page.locator("#calibrationSettings").evaluate(element => { element.open = true; });
      await page.locator("#calibrateButton").click();
      await page.locator("#fretOverlay").scrollIntoViewIfNeeded();
      for (const point of [{ x: .1, y: .2 }, { x: .9, y: .2 }, { x: .9, y: .8 }, { x: .1, y: .8 }]) {
        const position = await page.evaluate((p) => {
          const canvas = document.querySelector("#fretOverlay"), r = canvas.getBoundingClientRect(), ratio = 4 / 3;
          const w = Math.min(canvas.width, canvas.height * ratio), h = w / ratio;
          return { x: r.x + ((canvas.width - w) / 2 + p.x * w) * r.width / canvas.width,
            y: r.y + ((canvas.height - h) / 2 + p.y * h) * r.height / canvas.height };
        }, point);
        await page.mouse.click(position.x, position.y);
      }
      assert.match(await page.locator("#visionStatus").textContent(), /标定完成/);
      console.log("PASS recorded video: import and four-point calibration");
      await page.locator("#visionView").selectOption("edge");
      await page.waitForTimeout(150);
      const edgeCount = await page.locator("#fretOverlay").evaluate((canvas) => {
        const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
        let whites = 0; for (let i = 0; i < data.length; i += 4) if (data[i] === 255 && data[i + 1] === 255) whites++;
        return whites;
      });
      assert.ok(edgeCount > 100, "recorded edges should render");
      await page.screenshot({ path: fileURLToPath(new URL("recorded-edges.png", evidence)), fullPage: true });
      await page.locator("#handModelButton").click();
      await page.waitForFunction(() => /检测已启动|手部检测已启动|指尖模型|模型加载失败|指尖检测失败/.test(document.querySelector("#visionStatus").textContent), null, { timeout: 60000 });
      assert.match(await page.locator("#visionStatus").textContent(), /检测已启动|手部检测已启动|指尖模型/);
      try { await page.waitForFunction(() => window.FretVision.snapshot()?.source === "video_file", null, { timeout: 10000 }); }
      catch (error) {
        console.log(await page.evaluate(() => ({ status: document.querySelector("#visionStatus").textContent,
          snapshot: window.FretVision.snapshot(), ready: document.querySelector("#cameraVideo").readyState,
          time: document.querySelector("#cameraVideo").currentTime })));
        console.log(errors); throw error;
      }
      assert.equal(await page.evaluate(() => window.FretVision.snapshot().positions.length), 0);
      assert.equal(await page.evaluate(() => window.FretVision.snapshot().hands.length), 0);
      await page.locator("#visionStopButton").click();
      await page.locator("#cameraButton").click();
      await page.waitForFunction(() => document.querySelector("#cameraStage").dataset.visionSource === "camera");
      await page.locator("#handModelButton").click();
      await page.waitForFunction(() => window.FretVision.snapshot()?.source === "camera");
      assert.equal(await page.evaluate(() => window.FretVision.snapshot().positions.length), 0);
      await page.locator("#visionStopButton").click();
      if (process.argv[5] === "--board-synthetic") {
        await page.locator("#boardVisionButton").click();
        await page.waitForFunction(() => document.querySelector("#cameraStage").dataset.visionSource === "board");
        assert.match(await page.locator("#frameStatusText").textContent(), /SYNTHETIC/);
        await page.locator("#handModelButton").click();
        await page.waitForFunction(() => window.FretVision.snapshot()?.source === "board_synthetic");
        assert.equal(await page.evaluate(() => window.FretVision.snapshot().positions.length), 0);
        assert.equal(await page.locator("#chordName").textContent(), "--");
        await page.screenshot({ path: fileURLToPath(new URL("board-interface.png", evidence)), fullPage: true });
        await page.locator("#visionStopButton").click();
        assert.equal(await page.evaluate(() => window.FretVision.snapshot()), null);
        results.push({ board_python_node_browser: "pass", synthetic_provenance: "pass" });
      }
      results.push({ recorded_video: "pass", four_points: "pass", edges: "pass", local_model_blank_frame: "pass", simulated_camera: "pass" });
    }
    await page.close();
  }
  assert.deepEqual(errors, []);
  await writeFile(new URL(process.argv[5] === "--board-synthetic" ? "board_results.json" : "results.json", evidence), JSON.stringify({ results, errors }, null, 2));
  console.log(JSON.stringify(results));
} finally { await browser.close(); }
