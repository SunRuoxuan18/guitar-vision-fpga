import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { chromium } = require(join(process.env.USERPROFILE,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
const baseUrl = process.argv[2] || "http://127.0.0.1:8001/?camera=board&v=20261029";

const browser = await chromium.launch({ channel: "msedge", headless: true });
const evidence = join(root, "../diagnostics/board_preview_rate");
await mkdir(evidence, { recursive: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window);
    window.__boardFrameResponses = [];
    window.fetch = async (...args) => {
      const response = await nativeFetch(...args);
      if (String(args[0]).startsWith("/vision/frame.jpg")) {
        window.__boardFrameResponses.push({
          at: performance.now(),
          sequence: Number(response.headers.get("X-Vision-Sequence"))
        });
      }
      return response;
    };
  });
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#cameraStage")?.dataset.visionSource === "board",
    null, { timeout: 15000 });
  await page.waitForFunction(() => window.FretVision?.metrics().frames > 0,
    null, { timeout: 10000 });
  await page.waitForTimeout(3000);
  const metrics = await page.evaluate(() => window.FretVision.metrics());
  const frameTransport = await page.evaluate(() => {
    const samples = window.__boardFrameResponses.slice(-160);
    const unique = [...new Set(samples.map(item => item.sequence).filter(Number.isSafeInteger))];
    const changes = unique.slice(1).map((sequence, index) => sequence - unique[index]).filter(delta => delta > 0);
    return {
      responses: samples.length,
      unique_sequences: unique.length,
      duplicate_responses: samples.length - unique.length,
      median_sequence_step: changes.sort((a, b) => a - b)[Math.floor(changes.length / 2)] ?? null
    };
  });
  const status = await page.evaluate(() => ({
    source: document.querySelector("#cameraStage").dataset.visionSource,
    videoRate: document.querySelector("#videoRate").textContent,
    frameStatus: document.querySelector("#frameStatusText").textContent
  }));
  assert.equal(status.source, "board");
  assert.ok(metrics.display_fps >= 20, JSON.stringify({ metrics, status, frameTransport }));
  assert.deepEqual(errors, []);
  await page.screenshot({ path: join(evidence, "live-c922.png") });
  const report = { checkedAt: new Date().toISOString(), metrics, frameTransport, status, errors };
  await writeFile(join(evidence, "live-c922.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
}
