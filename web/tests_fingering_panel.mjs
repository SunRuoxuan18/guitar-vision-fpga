import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { extname, resolve } from "node:path";

const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || "playwright");
const root = fileURLToPath(new URL(".", import.meta.url));
const evidence = fileURLToPath(new URL("../diagnostics/fingering_panel/", import.meta.url));
await mkdir(evidence, { recursive: true });
const server = createServer(async (req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  const target = resolve(root, path === "/" ? "index.html" : "." + path);
  if (!target.startsWith(root)) { res.writeHead(403).end(); return; }
  try {
    const body = await readFile(target);
    res.writeHead(200, { "Content-Type": ({ ".js": "text/javascript", ".html": "text/html", ".css": "text/css" })[extname(target)] || "application/octet-stream" });
    res.end(body);
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ channel: "msedge", headless: true });
const results = [];
try {
  for (const width of [1440, 360]) {
    const page = await browser.newPage({ viewport: { width, height: 1050 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    assert.equal(await page.locator("#observedFingering svg").count(), 0);
    await page.evaluate(() => {
      const fixture = document.createElement("p"); fixture.id = "fixtureLabel";
      fixture.textContent = "离线合成证据测试 · 非实琴拍摄";
      fixture.style.cssText = "color:#a53b23;font-weight:bold";
      document.querySelector("main").prepend(fixture);
      window.__fixtureAudio = { status: "recognized", testMode: 0, source: "hardware",
        chordLabel: "C", notes: [{ midi: 48 }, { midi: 52 }, { midi: 55 }],
        scaleCandidates: [{ tonic: "C", scale: "major" }, { tonic: "A", scale: "natural_minor" }] };
      window.__fixtureVision = { source: "board_camera", status: "candidates", positions: [
        { string: 5, fret: 3, midi_note: 48, finger: "ring", ambiguous: false },
        { string: 4, fret: 2, midi_note: 52, finger: "middle", ambiguous: false },
        { string: 2, fret: 1, midi_note: 60, finger: "index", ambiguous: false }] };
      window.FretAudioSnapshot = () => ({ ...window.__fixtureAudio, receivedAt: performance.now() });
      window.FretVision.snapshot = () => ({ ...window.__fixtureVision, receivedAt: performance.now() });
      window.FretFingering.update();
    });
    assert.equal(await page.locator('#observedFingering [data-evidence="audio_supported_projection"]').count(), 2);
    assert.equal(await page.locator('#observedFingering [data-evidence="fingertip_projection"]').count(), 1);
    assert.equal(await page.locator('#observedFingering [data-state="open"],#observedFingering [data-state="muted"]').count(), 0);
    assert.equal(await page.locator('#referenceFingering svg[data-kind="chord-diagram"]').count(), 1);
    assert.match(await page.locator("#referenceFingeringNote").textContent(),
      /3 个观察位置与此指型一致.*2 处同时获音频支持/);
    await page.evaluate(() => {
      window.__fixtureAudio.chordLabel = "C/E";
      window.__fixtureAudio.notes = [{ midi: 52 }, { midi: 55 }, { midi: 60 }];
      window.FretFingering.update();
    });
    assert.equal(await page.locator('#referenceFingering svg[data-chord="C/E"]').count(), 1);
    await page.evaluate(() => {
      window.__fixtureAudio.chordLabel = "C";
      window.__fixtureAudio.notes = [{ midi: 48 }, { midi: 52 }, { midi: 55 }];
      window.FretFingering.update();
    });
    assert.equal(await page.locator('#scaleFingering svg[data-kind="scale_reference"]').count(), 1);
    const layout = await page.evaluate(() => ({ overflow: document.documentElement.scrollWidth > innerWidth,
      diagramCount: document.querySelectorAll(".fingering-panel svg").length,
      boxes: [...document.querySelectorAll(".fingering-panel svg")].map(svg => {
        const r = svg.getBoundingClientRect(); return { x: r.x, right: r.right, width: r.width, height: r.height };
      }) }));
    assert.equal(layout.overflow, false);
    assert.ok(layout.boxes.slice(0, 2).every(r => r.x >= 0 && r.right <= width && r.width > 100 && r.height > 50));
    assert.equal(await page.locator("#scaleFingering").isVisible(), false);
    await page.locator("#debugDetails > summary").click();
    assert.equal(await page.locator("#scaleFingering").isVisible(), true);
    if (width < 600) assert.equal(await page.locator("#scaleFingering").evaluate(node =>
      getComputedStyle(node).overflowX === "auto" && node.scrollWidth > node.clientWidth), true);
    await page.locator("#debugDetails > summary").click();
    await page.screenshot({ path: `${evidence}/fusion-${width}.png`, fullPage: true });
    await page.evaluate(() => { window.__fixtureAudio.age_seconds = 2.5; window.FretFingering.update(); });
    assert.equal(await page.locator('#observedFingering [data-evidence="audio_supported_projection"]').count(), 0);
    assert.equal(await page.locator("#referenceFingering svg").count(), 0);
    assert.equal(await page.locator("#scaleFingering svg").count(), 0);
    await page.evaluate(() => { delete window.__fixtureAudio.age_seconds; window.FretFingering.update(); });
    await page.evaluate(() => { window.__fixtureAudio.notes = [{ midi: 49 }]; window.FretFingering.update(); });
    assert.equal(await page.locator('#observedFingering [data-evidence="audio_supported_projection"]').count(), 0);
    await page.evaluate(() => {
      window.__fixtureVision = { source: "board_camera", status: "relative_positions", positions: [
        { relative: true, string_line: 2, fret_interval: 4 }] };
      window.FretFingering.update();
    });
    assert.equal(await page.locator('#observedFingering [data-line="2"][data-interval="4"]').count(), 1);
    assert.equal(await page.locator('#observedFingering [data-fret]').count(), 0);
    await page.evaluate(() => { window.FretVision.snapshot = () => null; window.FretFingering.update(); });
    assert.equal(await page.locator("#observedFingering svg").count(), 0);
    assert.equal(await page.locator("#referenceFingering svg").count(), 1);
    await page.evaluate(() => { window.FretAudioSnapshot = () => null; window.FretFingering.update(); });
    assert.equal(await page.locator(".fingering-panel svg").count(), 0);
    assert.deepEqual(errors, []);
    results.push({ width, source: "offline fabricated inputs through real browser scripts",
      exactMidiSupport: true, relativeNumberingUnknown: true, staleCleared: true, layout, errors });
    await page.close();
  }
  await writeFile(`${evidence}/results.json`, JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
} finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
