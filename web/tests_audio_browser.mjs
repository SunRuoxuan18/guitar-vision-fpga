import assert from "node:assert/strict";
import { readFile, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

// Supply the existing Playwright module and browser paths; nothing is installed.
const { chromium } = await import(process.argv[2] ? pathToFileURL(process.argv[2]).href : "playwright");
const browser = await chromium.launch({ headless: true, ...(process.argv[3] ? { executablePath: process.argv[3] } : {}) });
const screenshots = await mkdtemp(join(tmpdir(), "fretlens-audio-"));
const files = new Map(await Promise.all(["index.html", "styles.css", "fretboard.js", "fingering.js", "chord_library.js", "app.js", "vision.js", "fingering_panel.js"].map(async (file) =>
  [file, await readFile(new URL(file, import.meta.url), "utf8")])));
const recorded = process.argv[4] ? JSON.parse(await readFile(process.argv[4], "utf8")) : null;
let sequence = 0;
let apiMode = "live";
const pageErrors = [];

try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 1024, height: 768 }, { width: 360, height: 800 }]) {
    const page = await browser.newPage({ viewport });
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await page.route("http://fretlens.test/**", async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/api/audio") {
        const body = { source: "fpga", status: apiMode === "offline" ? "error" : "recognized",
          error: apiMode === "offline" ? "Board unavailable" : null, sequence: ++sequence,
          stale: false, age_seconds: 0.1, config: { sample_rate: 48000, fft_size: 4096 },
          notes: [{ note_name: "C3", frequency_hz: 130.813, confidence: 0.92 },
            { note_name: "E3", frequency_hz: 164.814, confidence: 0.90 },
            { note_name: "G3", frequency_hz: 195.998, confidence: 0.88 }],
          chord: { status: "recognized", label: "C", confidence: 0.90, alternatives: [] },
          tracking: { chord_status: "confirmed" },
          stable_chord: { status: "recognized", label: "C", confidence: 0.90, alternatives: [] },
          scale_evidence: { window_seconds: 12, compatible_group_count: 1,
            candidates: [{ tonic: "C", scale: "major" }, { tonic: "A", scale: "natural_minor" },
              { tonic: "D", scale: "dorian" }, { tonic: "E", scale: "phrygian" },
              { tonic: "F", scale: "lydian" }, { tonic: "G", scale: "mixolydian" },
              { tonic: "B", scale: "locrian" }] },
          spectrum: { bins: Array.from({ length: 2048 }, (_, bin) => [11, 14, 17].includes(bin) ? 1200 : 0) } };
        if (recorded && apiMode === "live") Object.assign(body, recorded, { sequence, age_seconds: 0.1, stale: false });
        if (["holding", "switching"].includes(apiMode)) Object.assign(body, {
          status: "unknown", signal_active: true, notes: [],
          chord: { status: "unknown", label: null }, tracking: { chord_status: apiMode },
          stable_chord: { status: "recognized", label: "C", confidence: 0.9 }
        });
        await route.fulfill({ json: body });
        return;
      }
      const file = path === "/" ? "index.html" : path.slice(1);
      await route.fulfill({ status: files.has(file) ? 200 : 404, body: files.get(file) || "",
        contentType: file.endsWith(".css") ? "text/css" : file.endsWith(".js") ? "text/javascript" : "text/html" });
    });
    await page.goto("http://fretlens.test/");
    assert.equal(await page.locator("#debugDetails").getAttribute("open"), null);
    assert.equal(await page.locator("#cameraButton").isVisible(), true);
    assert.equal(await page.locator("button:visible").count(), 2);
    assert.equal(await page.locator("#cameraSelect").isVisible(), false);
    assert.equal(await page.locator("#boardVisionButton").isVisible(), false);
    assert.equal(await page.locator("#calibrateButton").isVisible(), false);
    assert.equal(await page.locator("#autoFretboard").isVisible(), false);
    assert.equal(await page.locator("#autoFretboard").isChecked(), true);
    assert.equal(await page.locator("#handModelButton").isVisible(), false);
    assert.equal(await page.locator("#clearCalibrationButton").isVisible(), false);
    assert.match(await page.locator("#provenanceBadge").textContent(), /^IDLE/);
    assert.equal(await page.locator("#chordName").textContent(), "--");
    assert.equal(await page.locator(".event:visible").count(), 0);
    apiMode = "live";
    await page.locator("#hardwareButton").click();
    await page.waitForFunction(() => document.querySelector("#chordName").textContent === "C");
    assert.match(await page.locator("#noteResult").textContent(), recorded ? /C4.*E4.*G4/ : /C3.*E3.*G3/);
    if (!recorded || recorded.scale_evidence?.candidates?.length) {
      assert.match(await page.locator("#scaleAlternative").textContent(), /近 12 秒.*主音未确定/);
    } else {
      assert.match(await page.locator("#scaleAlternative").textContent(), /等待更多不同音级/);
    }
    if (recorded) assert.match(await page.locator("#provenanceBadge").textContent(), /内部 C 和弦测试/);
    const layout = await page.evaluate(() => {
      const buttons = [...document.querySelectorAll(".input-controls button,.vision-controls > button")].map((button) => {
        const r = button.getBoundingClientRect();
        return { x: r.x, y: r.y, right: r.right, bottom: r.bottom, height: r.height };
      });
      const overlap = buttons.some((r, i) => buttons.slice(i + 1).some((other) =>
        r.x < other.right && r.right > other.x && r.y < other.bottom && r.bottom > other.y));
      const data = document.querySelector("#spectrumCanvas").getContext("2d").getImageData(0, 0, 900, 250).data;
      let colored = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i] > 90 || data[i + 1] > 90 || data[i + 2] > 90) colored += 1;
      return { overflow: document.documentElement.scrollWidth > innerWidth, buttons, overlap, colored };
    });
    assert.equal(layout.overflow, false, `page overflow at ${viewport.width}px`);
    assert.equal(layout.overlap, false, `button overlap at ${viewport.width}px`);
    assert.ok(layout.buttons.length >= 4);
    assert.ok(layout.buttons.every((button) => button.height >= 38 && button.x >= 0 && button.right <= viewport.width));
    assert.ok(layout.colored > 100, "hardware spectrum should contain rendered evidence");
    await page.screenshot({ path: join(screenshots, `hardware-${viewport.width}.png`), fullPage: true });
    apiMode = "holding";
    await page.waitForFunction(() => document.querySelector("#agreementText").textContent === "延音保持");
    assert.equal(await page.locator("#chordName").textContent(), "C");
    assert.equal(await page.locator("#chordCandidates").textContent(), "--");
    apiMode = "switching";
    await page.waitForFunction(() => document.querySelector("#agreementText").textContent === "换和弦确认中");
    assert.equal(await page.locator("#chordName").textContent(), "C");
    apiMode = "offline";
    await page.waitForFunction(() => document.querySelector("#provenanceBadge").textContent.includes("错误"));
    assert.equal(await page.locator("#chordName").textContent(), "--");
    assert.equal(await page.locator("#noteResult").textContent(), "--");
    await page.locator("#hardwareButton").click();
    assert.match(await page.locator("#provenanceBadge").textContent(), /^IDLE/);
    await page.locator("#debugDetails > summary").click();
    await page.locator("#demoButton").click();
    assert.match(await page.locator("#provenanceBadge").textContent(), /^DEMO/);
    assert.equal(await page.locator("#chordName").textContent(), "G Major");
    await page.close();
    console.log(`PASS ${viewport.width}px: idle, hardware, spectrum, sustain, switching, offline, stop, demo, layout`);
  }
  assert.deepEqual(pageErrors, []);
  console.log(`Screenshots: ${screenshots}`);
} finally {
  await browser.close();
}
