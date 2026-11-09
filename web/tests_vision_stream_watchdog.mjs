import assert from "node:assert/strict";
import { createRequire } from "node:module";

const { chromium } = createRequire(import.meta.url)(process.argv[2]);
const browser = await chromium.launch({ headless: true, executablePath: process.argv[3] });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on("pageerror", error => errors.push(error.message));

try {
  const jpegBase64 = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const context = canvas.getContext("2d");
    context.fillStyle = "#6b2637";
    context.fillRect(0, 0, 640, 480);
    context.fillStyle = "#d8c8aa";
    context.fillRect(120, 60, 320, 360);
    return canvas.toDataURL("image/jpeg").split(",")[1];
  });

  await page.addInitScript(({ jpegBase64 }) => {
    const jpeg = Uint8Array.from(atob(jpegBase64), character => character.charCodeAt(0));
    const encoder = new TextEncoder();
    const originalFetch = window.fetch.bind(window);
    window.__streamConnections = 0;
    window.fetch = async (input, init) => {
      const url = typeof input === "string" ? input : input.url;
      if (!url.includes("/vision/stream.mjpeg")) return originalFetch(input, init);
      const connection = ++window.__streamConnections;
      let sequence = connection === 1 ? 1 : 100;
      let timer = null;
      const stream = new ReadableStream({
        start(controller) {
          const enqueueFrame = () => {
            const headers = encoder.encode(
              "--frame\r\n" +
              "Content-Type: image/jpeg\r\n" +
              `Content-Length: ${jpeg.length}\r\n` +
              `X-Vision-Sequence: ${sequence++}\r\n` +
              `X-Vision-Captured-Ms: ${performance.now()}\r\n` +
              "X-Vision-Source: camera\r\n" +
              "X-Vision-Stream: watchdog-stream\r\n\r\n");
            const frame = new Uint8Array(headers.length + jpeg.length + 2);
            frame.set(headers, 0);
            frame.set(jpeg, headers.length);
            frame.set([13, 10], headers.length + jpeg.length);
            controller.enqueue(frame);
          };
          enqueueFrame();
          // The first connection stays open but stops producing frames.
          if (connection > 1) timer = setInterval(enqueueFrame, 100);
        },
        cancel() {
          clearInterval(timer);
        }
      });
      return new Response(stream, {
        status: 200,
        headers: { "Content-Type": "multipart/x-mixed-replace; boundary=frame" }
      });
    };
  }, { jpegBase64 });

  await page.route("**/vision/frame.jpg*", route => route.fulfill({
    status: 503,
    body: "watchdog test uses the multipart stream"
  }));
  await page.route("**/api/vision/pc/status", route => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ link_speed_mbps: 1000 })
  }));
  await page.route("**/api/vision/pc/process", route => route.fulfill({
    status: 503,
    body: "not needed for stream watchdog"
  }));

  await page.goto("http://127.0.0.1:8001/?camera=board&v=stream-watchdog-test");
  await page.waitForFunction(() => window.__streamConnections >= 1, null, { timeout: 5000 });
  await page.waitForFunction(() => window.__streamConnections >= 2, null, { timeout: 8000 });
  await page.waitForFunction(() => window.FretVision.metrics().display_fps > 0, null, { timeout: 5000 });
  assert.ok(await page.evaluate(() => window.__streamConnections) >= 2,
    "A silent multipart stream must reconnect after the stall timeout");
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    stream_watchdog: "pass",
    connections: await page.evaluate(() => window.__streamConnections),
    display_fps: await page.evaluate(() => window.FretVision.metrics().display_fps)
  }, null, 2));
} finally {
  await page.close();
  await browser.close();
}
