import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import test from "node:test";
import "./fretboard.js";
import GuitarFingering from "./fingering.js";

const F = globalThis.GuitarFingering;
const pcs = notes => [...new Set(notes.map(note => ((note % 12) + 12) % 12))].sort((a, b) => a - b);
const now = 1000;
const vision = (positions = [{ string: 6, fret: 3, midi_note: 43, ambiguous: false }]) =>
  ({ source: "camera", receivedAt: 950, status: "candidates", positions });
const audio = (notes = [{ midi: 43, ambiguous: false }]) =>
  ({ source: "fpga", receivedAt: 950, status: "recognized", testMode: 0, notes });

test("one dependency-free API works as a native browser script, Node import and CommonJS module", () => {
  assert.equal(F, GuitarFingering);
  const source = readFileSync(new URL("./fingering.js", import.meta.url), "utf8");
  const browser = vm.createContext({});
  vm.runInContext("globalThis.window = globalThis", browser);
  vm.runInContext(source, browser);
  assert.equal(typeof browser.window.GuitarFingering.fuseEvidence, "function");
  assert.equal(browser.window.GuitarFingering.chooseReference("C").frets.join(","), ",3,2,0,1,0");
  const commonjs = vm.createContext({ module: { exports: {} } });
  vm.runInContext(source, commonjs);
  assert.equal(commonjs.module.exports, commonjs.GuitarFingering);
});

test("all catalog voicings contain exactly their declared chord tones and omissions", () => {
  const expected = {
    C: [null, 3, 2, 0, 1, 0], D: [null, null, 0, 2, 3, 2], E: [0, 2, 2, 1, 0, 0],
    G: [3, 2, 0, 0, 0, 3], A: [null, 0, 2, 2, 2, 0], Am: [null, 0, 2, 2, 1, 0],
    Dm: [null, null, 0, 2, 3, 1], Em: [0, 2, 2, 0, 0, 0]
  };
  const catalog = F.getChordCatalog();
  assert.ok(catalog.length >= 23);
  for (const chord of catalog) {
    const included = chord.intervals.filter(interval => !chord.omittedIntervals.includes(interval));
    assert.deepEqual(pcs(chord.positions.map(position => position.midi_note)),
      pcs(included.map(interval => chord.rootPc + interval)), chord.label);
    assert.equal(chord.kind, "reference");
    assert.equal(chord.strings.length, 6);
    assert.equal(chord.source, "verified_open_catalog");
    for (const position of chord.strings) {
      assert.equal(position.pressed, null);
      assert.equal(position.sounding, null);
      assert.equal(position.fret, chord.frets[6 - position.string]);
      if (position.fret !== null) assert.equal(position.midi_note, F.TUNING[6 - position.string] + position.fret);
    }
    if (expected[chord.label]) assert.deepEqual(chord.frets, expected[chord.label]);
    assert.deepEqual(F.chooseReference(chord.label).frets, chord.frets);
  }
  assert.deepEqual(F.chooseReference("C7").omittedIntervals, [7]);
});

test("E/A barre transposition covers every chromatic root and five qualities with correct pitches and bass", () => {
  const roots = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
  const qualities = [["", [0, 4, 7]], ["m", [0, 3, 7]], ["7", [0, 4, 7, 10]],
    ["maj7", [0, 4, 7, 11]], ["m7", [0, 3, 7, 10]]];
  for (let root = 0; root < roots.length; root += 1) for (const [suffix, intervals] of qualities) {
    for (const shape of ["E", "A"]) {
      const voicings = F.getChordVoicings(roots[root] + suffix, { shape });
      assert.ok(voicings.length > 0, roots[root] + suffix + shape);
      for (const chord of voicings) {
        assert.equal(chord.shape, shape);
        assert.equal(chord.frets.includes(0), false);
        assert.ok(Math.max(...chord.frets) <= 24);
        assert.deepEqual(pcs(chord.positions.map(position => position.midi_note)), pcs(intervals.map(interval => root + interval)));
        assert.equal(Math.min(...chord.positions.map(position => position.midi_note)) % 12, root);
        assert.equal(chord.barres[0].fromString, shape === "E" ? 6 : 5);
        assert.equal(chord.barres[0].fret, chord.baseFret);
        assert.ok(chord.positions.every(position => position.fret >= chord.baseFret));
      }
    }
  }
  assert.deepEqual(F.chooseReference("F").frets, [1, 3, 3, 2, 1, 1]);
  assert.deepEqual(F.chooseReference("Bm").frets, [null, 2, 4, 4, 3, 2]);
  assert.equal(F.getChordVoicings("C", { maxFret: 2 }).length, 0);
});

test("chord labels preserve quality, enharmonic root and slash-bass meaning without guessing unsupported chords", () => {
  for (const label of ["C Major", "Cmaj", "CM"]) assert.equal(F.chooseReference(label).label, "C");
  assert.equal(F.chooseReference("C Minor Seventh").quality, "minor_seventh");
  assert.equal(F.chooseReference("CM7").quality, "major_seventh");
  assert.equal(F.chooseReference("Cm7").quality, "minor_seventh");
  assert.equal(F.chooseReference("B\u266d7").label, "Bb7");
  assert.equal(F.chooseReference("Cb").rootPc, 11);
  assert.equal(F.chooseReference("C/C").label, "C");
  for (const label of ["C/E", "D/F#", "C/X", "C/", "C/E/G", "Cdim", "C9", "C6", "Cconstructor", "C__proto__", "H", "", null, 42]) {
    assert.equal(F.chooseReference(label), null, String(label));
  }
  assert.throws(() => F.getChordVoicings("C", { maxFret: 25 }), RangeError);
  assert.throws(() => F.getChordVoicings("C", { shape: "D" }), RangeError);
  const c = F.chooseReference("C");
  c.frets[1] = 24;
  c.strings[0].state = "open";
  c.pitchClasses.push(1);
  assert.deepEqual(F.chooseReference("C").frets, [null, 3, 2, 0, 1, 0]);
  assert.equal(F.chooseReference("C").strings[0].state, "muted");
});

test("fusion uses exact MIDI rather than pitch classes and never proves pressure or sound", () => {
  const projection = vision().positions[0];
  const v = vision([{ ...projection, pressed: true, sounding: true }]);
  const a = audio();
  const before = JSON.stringify({ v, a });
  const result = F.fuseEvidence(v, a, now);
  assert.equal(result.status, "supported_positions");
  assert.equal(result.matches.length, 1);
  assert.equal(result.matches[0].evidence, "audio_supported_projection");
  assert.equal(result.matches[0].pressed, null);
  assert.equal(result.matches[0].sounding, null);
  assert.equal(result.synchronized, false);
  assert.equal(result.reception_gap_ms, 0);
  assert.match(result.limitation, /do not prove a pressed/);
  assert.equal(JSON.stringify({ v, a }), before);
  for (const notes of [[{ midi: 55 }], [{ midi: 43, ambiguous: true }],
    [{ midi_note: 43, octave_ambiguous: true }], [{ midi: 43.1 }],
    [{ midi: "43" }], [{ midi: 43, midi_note: 55 }], [null], []]) {
    assert.equal(F.fuseEvidence(vision(), audio(notes), now).matches.length, 0);
  }
  assert.equal(F.fuseEvidence(vision(), { ...audio([{ midi_note: 43 }]), testMode: undefined,
    fft: { analysis_test_mode: 0 } }, now).matches.length, 1);
  assert.equal(F.fuseEvidence(vision(), { ...audio(), fft: { analysis_test_mode: 2 } }, now).matches.length, 0);
});

test("no fingertips means unknown strings, never six inferred open or muted strings", () => {
  for (const v of [null, vision([]), { ...vision(), positions: null }]) {
    const result = F.fuseEvidence(v, audio(), now);
    assert.equal(result.positions.length, 0);
    assert.equal(result.matches.length, 0);
    assert.equal(result.strings.length, 6);
    assert.ok(result.strings.every(string => string.state === "unknown" && string.fret === null &&
      string.midi_note === null && string.pressed === null && string.sounding === null));
  }
  const result = F.fuseEvidence(vision(), audio(), now);
  assert.equal(result.strings[0].state, "projection");
  assert.ok(result.strings.slice(1).every(string => string.state === "unknown"));
  for (const position of [{ string: 6, fret: 0, midi_note: 40 }, { string: 0, fret: 3 },
    { string: 7, fret: 3 }, { string: 6, fret: 25 }, { string: 6, fret: "3" },
    { string: 6, fret: 3, midi_note: 55 }]) {
    const invalid = F.fuseEvidence(vision([position]), audio([{ midi: position.midi_note }]), now);
    assert.equal(invalid.matches.length, 0);
    assert.ok(invalid.strings.every(string => string.state === "unknown"));
  }
});

test("automatic relative geometry stays unnumbered, even when a chord reference or exact MIDI is available", () => {
  const relative = { string_line: 1, fret_interval: 3, relative: true, ambiguous: false,
    string: 6, fret: 3, midi_note: 43, pressed: true, sounding: true };
  const result = F.fuseEvidence(vision([relative]), audio(), now);
  assert.equal(result.status, "relative_positions");
  assert.equal(result.positions.length, 0);
  assert.equal(result.matches.length, 0);
  assert.equal(result.relativePositions.length, 1);
  assert.equal(result.relativePositions[0].string, undefined);
  assert.equal(result.relativePositions[0].fret, undefined);
  assert.equal(result.relativePositions[0].midi_note, undefined);
  assert.equal(result.relativePositions[0].pressed, null);
  assert.equal(result.absoluteFretsKnown, false);
  assert.ok(result.strings.every(string => string.state === "unknown"));
  F.chooseReference("G");
  assert.deepEqual(F.fuseEvidence(vision([relative]), audio(), now), result);
});

test("freshness gates reject stale, future, missing and nonfinite reception times and reception gaps", () => {
  for (const update of [{ receivedAt: 499 }, { receivedAt: now + 1 }, { receivedAt: undefined },
    { receivedAt: NaN }, { receivedAt: -1 }, { stale: true }, { age_seconds: .501 },
    { age_seconds: NaN }, { status: "lost" }, { status: "uncalibrated" }]) {
    const result = F.fuseEvidence({ ...vision(), ...update }, audio(), now);
    assert.equal(result.status, "vision_unavailable");
    assert.equal(result.matches.length, 0);
    assert.equal(result.positions.length, 0);
  }
  for (const update of [{ receivedAt: 249 }, { receivedAt: now + 1 }, { receivedAt: undefined },
    { receivedAt: Infinity }, { receivedAt: -1 }, { stale: true }, { age_seconds: .751 },
    { age_seconds: -1 }, { status: "unknown" }, { status: "silence" }, { testMode: undefined }]) {
    const result = F.fuseEvidence(vision(), { ...audio(), ...update }, now);
    assert.equal(result.status, "audio_unavailable");
    assert.equal(result.matches.length, 0);
    assert.equal(result.positions.length, 1, "fresh visual projections survive audio unavailability");
  }
  assert.equal(F.fuseEvidence({ ...vision(), receivedAt: 500 }, { ...audio(), receivedAt: 250 }, now).matches.length, 1);
  assert.equal(F.fuseEvidence(vision(), { ...audio(), age_seconds: .71 }, now).matches.length, 0,
    "server age plus elapsed reception time exceeds freshness");
  assert.equal(F.fuseEvidence(vision(), { ...audio(), age_seconds: .65 }, now).matches.length, 1);
  assert.equal(F.fuseEvidence(vision(), { ...audio(), age_seconds: .71, ageReportedAt: now }, now).matches.length, 1);
  assert.equal(F.fuseEvidence(vision(), { ...audio(), receivedAt: 700 }, now, { maximumGapMs: 100 }).matches.length, 0);
  for (const invalidNow of [undefined, NaN, Infinity, -1]) assert.equal(F.fuseEvidence(vision(), audio(), invalidNow).matches.length, 0);
  assert.throws(() => F.fuseEvidence(vision(), audio(), now, { audioMaxAgeMs: Infinity }), RangeError);
});

test("synthetic, recording and diagnostic inputs never become current guitar observations", () => {
  for (const source of ["synthetic", "board_synthetic", "demo", "board", undefined]) {
    const result = F.fuseEvidence({ ...vision(), source }, audio(), now);
    assert.equal(result.status, "demo_only");
    assert.equal(result.positions.length, 0);
    assert.equal(result.matches.length, 0);
  }
  for (const source of ["video_file", "board_video_file"]) {
    assert.equal(F.fuseEvidence({ ...vision(), source }, audio(), now).status, "recording_not_synchronized");
  }
  for (const update of [{ synthetic: true }, { demo: true }, { provenance: { synthetic: true } }, { recording: true }]) {
    assert.equal(F.fuseEvidence({ ...vision(), ...update }, audio(), now).matches.length, 0);
  }
  for (const update of [{ source: "synthetic" }, { source: "recording" }, { synthetic: true },
    { provenance: { synthetic: true } }, { demo: true }, { recording: true }, { recorded: true },
    { testMode: 1 }, { testMode: 2 }, { testMode: 3 }, { testMode: "0" }]) {
    assert.equal(F.fuseEvidence(vision(), { ...audio(), ...update }, now).matches.length, 0);
  }
  assert.equal(F.fuseEvidence({ ...vision(), source: "board_camera" }, audio(), now).matches.length, 1);
});

test("ambiguous fret/string projections and conflicting finger identities remain unsupported", () => {
  assert.equal(F.fuseEvidence(vision([{ string: 6, fret: 3, ambiguous: true }]), audio(), now).matches.length, 0);
  const positions = [{ string: 6, fret: 3, finger: "index", hand_index: 0 },
    { string: 5, fret: 3, finger: "index", hand_index: 0 }];
  const result = F.fuseEvidence(vision(positions), audio([{ midi: 43 }, { midi: 48 }]), now);
  assert.equal(result.matches.length, 0);
  assert.ok(result.positions.every(position => position.ambiguous));
  const multipleFrets = F.fuseEvidence(vision([{ string: 6, fret: 3 }, { string: 6, fret: 5 }]), audio(), now);
  assert.equal(multipleFrets.strings[0].state, "ambiguous_projection");
  assert.equal(multipleFrets.strings[0].fret, null);
});

test("real calibrated fretboard payloads fuse without a renderer or controller dependency", () => {
  const calibration = globalThis.Fretboard.createCalibration([
    { x: .1, y: .1 }, { x: .9, y: .1 }, { x: .9, y: .9 }, { x: .1, y: .9 }
  ]);
  const x = (calibration.boundaries[2].x + calibration.boundaries[3].x) / 2;
  const tip = globalThis.Fretboard.transform(calibration.toImage, { x, y: 0 });
  const positions = globalThis.Fretboard.locateTip(calibration, tip);
  assert.equal(F.fuseEvidence(vision(positions), audio(), now).matches.length, 1);
  const dropD = [38, 45, 50, 55, 59, 64];
  const alternate = F.fuseEvidence(vision([{ string: 6, fret: 3, midi_note: 41 }]), audio([{ midi: 41 }]), now, { tuning: dropD });
  assert.equal(alternate.matches[0].note_name, "F2");
});

test("seven natural modes and both pentatonics have their independent expected intervals and note spellings", () => {
  const expected = {
    ionian: [0, 2, 4, 5, 7, 9, 11], dorian: [0, 2, 3, 5, 7, 9, 10],
    phrygian: [0, 1, 3, 5, 7, 8, 10], lydian: [0, 2, 4, 6, 7, 9, 11],
    mixolydian: [0, 2, 4, 5, 7, 9, 10], aeolian: [0, 2, 3, 5, 7, 8, 10],
    locrian: [0, 1, 3, 5, 6, 8, 10], major_pentatonic: [0, 2, 4, 7, 9], minor_pentatonic: [0, 3, 5, 7, 10]
  };
  for (const [mode, intervals] of Object.entries(expected)) assert.deepEqual(F.getScale("C", mode).pitchClasses, intervals);
  assert.deepEqual(F.getScale("F", "major").notes.map(note => note.name), ["F", "G", "A", "Bb", "C", "D", "E"]);
  assert.deepEqual(F.getScale("C#", "ionian").notes.map(note => note.name), ["C#", "D#", "E#", "F#", "G#", "A#", "B#"]);
  assert.deepEqual(F.getScale("A", "minor pentatonic").notes.map(note => note.name), ["A", "C", "D", "E", "G"]);
  assert.deepEqual(F.getScale("A", "minor_pentatonic").notes.map(note => note.degree), ["1", "b3", "4", "5", "b7"]);
  assert.deepEqual(F.getScale("A", "natural_minor").pitchClasses, F.getScale("A", "aeolian").pitchClasses);
  assert.deepEqual(pcs(F.getScale("D", "dorian").pitchClasses), pcs(F.getScale("C", "ionian").pitchClasses));
  for (const [tonic, mode] of [["H", "ionian"], ["C", "harmonic_minor"], ["C", "__proto__"], [null, "ionian"]]) {
    assert.equal(F.getScale(tonic, mode), null);
  }
});

test("fret maps include every and only requested scale pitch across all strings, frets and tunings", () => {
  const roots = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
  for (const tonic of roots) for (const mode of Object.keys(F.MODES)) {
    const map = F.scaleFretmap(tonic, mode, { firstFret: 0, lastFret: 24 });
    const actual = new Map(map.positions.map(position => [`${position.string}:${position.fret}`, position]));
    assert.equal(actual.size, map.positions.length);
    for (let string = 1; string <= 6; string += 1) for (let fret = 0; fret <= 24; fret += 1) {
      const midi = F.TUNING[6 - string] + fret, present = map.pitchClasses.includes(midi % 12);
      assert.equal(actual.has(`${string}:${fret}`), present, `${tonic} ${mode}, string ${string} fret ${fret}`);
      if (present) {
        const position = actual.get(`${string}:${fret}`);
        assert.equal(position.midi_note, midi);
        assert.equal(position.root, midi % 12 === map.rootPc);
      }
    }
  }
  const partial = F.scaleFretmap("D", "minor_pentatonic", { firstFret: 5, lastFret: 9, tuning: [38, 45, 50, 55, 59, 64] });
  assert.ok(partial.positions.every(position => position.fret >= 5 && position.fret <= 9));
  assert.ok(partial.positions.some(position => position.string === 6 && position.fret === 5 && position.midi_note === 43));
  assert.throws(() => F.scaleFretmap("C", "major", { firstFret: 9, lastFret: 5 }), RangeError);
  assert.throws(() => F.scaleFretmap("C", "major", { tuning: [40] }), RangeError);
  assert.equal(F.scaleFretmap("C", "unknown"), null);
});

test("SVGs label reference and observed evidence separately, show barres and never draw observed open/mute markers", () => {
  const reference = F.renderChordChart(F.chooseReference("C"));
  assert.match(reference, /data-kind="reference"/);
  assert.match(reference, /Reference voicing/);
  assert.match(reference, /data-state="open"/);
  assert.match(reference, /data-state="muted"/);
  assert.match(F.renderChordChart(F.chooseReference("F")), /reference_barre/);
  const observed = F.renderChordChart(F.fuseEvidence(vision(), audio(), now));
  assert.match(observed, /data-kind="observed"/);
  assert.match(observed, /audio_supported_projection/);
  assert.match(observed, /pressure and sound unknown/);
  assert.doesNotMatch(observed, /data-state="(?:open|muted)"|reference_barre/);
  assert.equal((observed.match(/data-state="unknown"/g) || []).length, 6);
  const visualOnlyModel = F.fuseEvidence(vision(), null, now);
  const visualOnly = F.renderChordChart(visualOnlyModel);
  assert.equal(visualOnlyModel.positions.length, 1);
  assert.equal(visualOnlyModel.audioAvailable, false);
  assert.match(visualOnly, /data-evidence="fingertip_projection"[^>]+fill="none"/);
  assert.doesNotMatch(visualOnly, /data-evidence="audio_supported_projection"/);
  const relative = F.renderChordChart(F.fuseEvidence(vision([
    { relative: true, string_line: 2, fret_interval: 4 }
  ]), audio(), now));
  assert.match(relative, /1 relative projections; numbering unknown/);
  assert.match(relative, /data-line="2" data-interval="4"/);
  assert.match(relative, /String and fret numbering unknown/);
  assert.doesNotMatch(relative, /data-fret=/);
  const malicious = { ...F.chooseReference("C"), label: '<script onload="evil()">&' };
  assert.doesNotMatch(F.renderChordChart(malicious), /<script|aria-label="<|onload="evil/);
  assert.match(F.renderChordChart(malicious), /&lt;script/);
  assert.match(F.renderChordChart(null), /No reference voicing/);
  const map = F.scaleFretmap("A", "minor_pentatonic");
  const svg = F.renderScaleFretmap(map);
  assert.match(svg, /Scale reference/);
  assert.equal((svg.match(/data-midi=/g) || []).length, map.positions.length);
  assert.equal((svg.match(/data-root="true"/g) || []).length, map.positions.filter(position => position.root).length);
  assert.match(F.renderScaleFretmap(map, { labels: "degrees" }), />b3<\/text>/);
  assert.match(F.renderScaleFretmap(null), /No scale reference/);
});

// Optional: FINGERING_BROWSER_TESTS=1; use existing Playwright and browser installs only.
test("SVGs render in a real browser at mobile and desktop sizes without clipping or numbering relative observations", {
  skip: process.env.FINGERING_BROWSER_TESTS !== "1"
}, async () => {
  const require = createRequire(import.meta.url);
  let playwright;
  try { playwright = require(process.env.FINGERING_PLAYWRIGHT_PATH || "playwright"); }
  catch (error) {
    if (process.env.FINGERING_PLAYWRIGHT_PATH) throw error;
    playwright = require(join(process.env.USERPROFILE || process.env.HOME,
      ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
  }
  const browser = await playwright.chromium.launch({ headless: true,
    ...(process.env.FINGERING_BROWSER_PATH ? { executablePath: process.env.FINGERING_BROWSER_PATH } : {}) });
  const screenshotDirectory = await mkdtemp(join(tmpdir(), "guitar-fingering-"));
  const errors = [];
  try {
    for (const width of [360, 768, 1440]) {
      const page = await browser.newPage({ viewport: { width, height: 1100 } });
      page.on("pageerror", error => errors.push(error.message));
      await page.setContent('<!doctype html><html lang="en"><meta charset="utf-8"><style>body{margin:12px;background:#fff;color:#303b39}main{display:flex;flex-wrap:wrap;gap:12px}section{max-width:100%;min-width:0}aside{width:100%;max-width:640px}svg{display:block}</style><main></main></html>');
      await page.addScriptTag({ path: fileURLToPath(new URL("./fingering.js", import.meta.url)) });
      const report = await page.evaluate(async () => {
        const F = window.GuitarFingering;
        const v = { source: "camera", receivedAt: 100, positions: [{ string: 6, fret: 3, midi_note: 43 }] };
        const a = { status: "recognized", receivedAt: 100, testMode: 0, notes: [{ midi: 43 }] };
        const relative = { ...v, positions: [{ relative: true, string_line: 2, fret_interval: 3 },
          { relative: true, string_line: 4, fret_interval: 4, ambiguous: true }] };
        const panels = [F.renderChordChart(F.chooseReference("C")), F.renderChordChart(F.chooseReference("F")),
          F.renderChordChart(F.fuseEvidence(v, null, 120)), F.renderChordChart(F.fuseEvidence(v, a, 120)),
          F.renderChordChart(F.fuseEvidence(relative, a, 120))];
        document.querySelector("main").innerHTML = panels.map(svg => `<section>${svg}</section>`).join("") +
          `<aside>${F.renderScaleFretmap(F.scaleFretmap("A", "minor_pentatonic"))}</aside>`;
        await document.fonts.ready;
        const parser = new DOMParser();
        const xml = [...F.getChordCatalog(), ...F.getChordVoicings("Bbmaj7")].map(chord => F.renderChordChart(chord));
        xml.push(...Object.keys(F.MODES).map(mode => F.renderScaleFretmap(F.scaleFretmap("C#", mode))));
        const invalidXml = xml.filter(svg => parser.parseFromString(svg, "image/svg+xml").querySelector("parsererror")).length;
        const clips = [];
        for (const svg of document.querySelectorAll("svg")) {
          const view = svg.viewBox.baseVal;
          for (const element of svg.querySelectorAll("text,circle,line")) {
            const box = element.getBBox();
            if (box.x < 0 || box.y < 0 || box.x + box.width > view.width || box.y + box.height > view.height) {
              clips.push({ text: element.textContent, x: box.x, y: box.y, width: box.width, height: box.height });
            }
          }
        }
        return { invalidXml, clips, documentWidth: document.documentElement.scrollWidth,
          viewportWidth: innerWidth, chartCount: document.querySelectorAll("svg").length,
          visualOnlyCount: document.querySelectorAll("section:nth-child(3) circle[data-evidence=fingertip_projection]").length,
          supportedCount: document.querySelectorAll("section:nth-child(4) circle[data-evidence=audio_supported_projection]").length,
          relativeCount: document.querySelectorAll("section:nth-child(5) circle[data-line]").length,
          relativeInventedFrets: document.querySelectorAll("section:nth-child(5) [data-fret]").length };
      });
      assert.equal(report.invalidXml, 0);
      assert.deepEqual(report.clips, [], `clipped SVG elements at width ${width}`);
      assert.ok(report.documentWidth <= report.viewportWidth, `horizontal overflow at ${width}`);
      assert.equal(report.chartCount, 6);
      assert.equal(report.visualOnlyCount, 1);
      assert.equal(report.supportedCount, 1);
      assert.equal(report.relativeCount, 2);
      assert.equal(report.relativeInventedFrets, 0);
      await page.screenshot({ path: join(screenshotDirectory, `${width}.png`), fullPage: true });
      await page.evaluate(async source => {
        delete window.GuitarFingering;
        await import(`data:text/javascript;base64,${btoa(source)}`);
      }, readFileSync(new URL("./fingering.js", import.meta.url), "utf8"));
      assert.equal(await page.evaluate(() => window.GuitarFingering.chooseReference("C").label), "C");
      await page.close();
    }
    assert.deepEqual(errors, []);
    console.log(`Fingering browser screenshots: ${screenshotDirectory}`);
  } finally { await browser.close(); }
});
