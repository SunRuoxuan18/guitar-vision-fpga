import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("app.js", import.meta.url), "utf8");
const markup = await readFile(new URL("index.html", import.meta.url), "utf8");

function harness(fetch = async () => { throw new TypeError("Network offline"); }) {
  const context2d = new Proxy({
    createLinearGradient: () => ({ addColorStop() {} }),
    measureText: (text) => ({ width: text.length * 8 })
  }, { get: (target, key) => target[key] || (() => {}) });
  const elements = new Map();
  function element() {
    const children = new Map();
    return {
      textContent: "", innerHTML: "", dataset: {}, style: {}, hidden: false, width: 900, height: 250,
      clientWidth: 900, disabled: false, value: "", listeners: {}, attributes: {},
      classList: { add() {}, remove() {} }, getContext: () => context2d,
      querySelector(selector) { if (!children.has(selector)) children.set(selector, element()); return children.get(selector); },
      addEventListener(name, listener) { this.listeners[name] = listener; },
      setAttribute(name, value) { this.attributes[name] = value; }, appendChild() {}
    };
  }
  for (const [, id] of markup.matchAll(/\bid="([^"]+)"/g)) elements.set(`#${id}`, element());
  const timers = new Map();
  let timerId = 0;
  let now = 0;
  const setTimer = (callback, delay) => { timers.set(++timerId, { callback, delay }); return timerId; };
  const window = { setTimeout: setTimer, setInterval: setTimer, clearTimeout: (id) => timers.delete(id),
    clearInterval: (id) => timers.delete(id), addEventListener() {}, devicePixelRatio: 1 };
  const sandbox = vm.createContext({ document: { querySelector: (selector) => elements.get(selector), createElement: element },
    window, navigator: {}, performance: { now: () => now }, Date, AbortController, fetch,
    setInterval: setTimer, requestAnimationFrame: setTimer, cancelAnimationFrame: (id) => timers.delete(id) });
  vm.runInContext(source, sandbox);
  return { elements, timers, sandbox, run: (code) => vm.runInContext(code, sandbox),
    now: (value) => { now = value; }, text: (id) => elements.get(`#${id}`).textContent };
}

function frame(overrides = {}) {
  return {
    source: "fpga", status: "recognized", stale: false, age_seconds: 0.1, sequence: 1,
    config: { sample_rate: 48000, fft_size: 4096 },
    notes: [{ note_name: "C3", frequency_hz: 130.813, confidence: 0.92 },
      { note_name: "E3", frequency_hz: 164.814, confidence: 0.91 },
      { note_name: "G3", frequency_hz: 195.998, confidence: 0.90 }],
    chord: { status: "recognized", label: "C", confidence: 0.91, ambiguous: true,
      alternatives: [{ label: "Csus", confidence: 0.7 }] },
    spectrum: { bins: Array.from({ length: 2048 }, (_, bin) => bin === 11 ? 1200 : 0) },
    ...overrides
  };
}

function apply(h, payload, now = 0) {
  h.sandbox.payload = payload;
  h.run(`acceptHardwarePayload(payload, ${now})`);
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

test("camera waiting state describes board processing without claiming a chord", () => {
  const h = harness();
  h.elements.get("#pcPlEnabled").checked = true;
  h.run("state.cameraStream = {}; showLiveWaitingResult()");
  assert.match(h.text("provenanceBadge"), /板卡处理/);
  assert.match(h.elements.get("#truthNote").innerHTML, /等待板卡/);
  assert.doesNotMatch(h.elements.get("#truthNote").innerHTML, /模型仍待接入/);
  assert.equal(h.text("chordName"), "等待融合");
  h.run("state.hardware = true; showLiveWaitingResult()");
  assert.match(h.text("visualEvidence"), /板卡/);
  assert.doesNotMatch(h.text("visualEvidence"), /识别待接入/);
});

test("audio snapshot retains the board feedback for the video renderer", () => {
  const h = harness();
  h.run("state.hardware = true");
  apply(h, frame({ fusion: { processing_location: "ps_linux",
    position_reason: "hand_outside_measured_lattice" } }));
  const snapshot = h.run("window.FretAudioSnapshot()");
  assert.equal(snapshot.visualFeedback, h.text("visualEvidence"));
});

test("joint conflicts and time mismatches do not display visual agreement", () => {
  const h = harness();
  h.run("state.hardware = true");
  apply(h, frame({ fusion: { processing_location: "ps_linux", status: "audio_visual_supported",
    joint_decision: { status: "visual_audio_conflict" } } }));
  assert.match(h.text("visualEvidence"), /不一致/);
  apply(h, frame({ sequence: 2, fusion: { processing_location: "ps_linux",
    reason: "audio_visual_time_mismatch" } }));
  assert.match(h.text("visualEvidence"), /时刻不匹配/);
  apply(h, frame({ sequence: 3, fusion: { processing_location: "ps_linux",
    position_reason: "hand_outside_measured_lattice" } }));
  assert.match(h.text("visualEvidence"), /不在已测品丝范围/);
});

test("joint feedback includes high fret candidates and distinguishes awaiting audio", () => {
  const h = harness();
  h.run("state.hardware = true");
  apply(h, frame({ fusion: { processing_location: "ps_linux", hand_fret_span: [15, 17],
    joint_decision: { status: "audio_visual_consistent", supported_fingers: 3 } } }));
  assert.match(h.text("visualEvidence"), /15.*17/);
  assert.match(h.text("visualEvidence"), /3/);
  apply(h, frame({ sequence: 2, fusion: { processing_location: "ps_linux",
    joint_decision: { status: "awaiting_audio_transition" } } }));
  assert.match(h.text("visualEvidence"), /等待当前音高确认/);
});

test("server frame age survives normalization and prevents false fresh fusion", async () => {
  const h = harness();
  h.run("state.hardware = true");
  apply(h, frame({ age_seconds: 2.5, notes: [{ note_name: "C3", frequency_hz: 130.813, midi_note: 48 }] }), 100);
  h.now(100);
  const F = (await import("./fingering.js")).default;
  const audio = h.run("window.FretAudioSnapshot()");
  assert.equal(audio.age_seconds, 2.5);
  assert.equal(audio.ageReportedAt, 100);
  const vision = { source: "camera", receivedAt: 100, status: "candidates",
    positions: [{ string: 5, fret: 3, midi_note: 48 }] };
  assert.equal(F.fuseEvidence(vision, audio, 100).matches.length, 0);
  await import("./fretboard.js");
  assert.equal(globalThis.Fretboard.compareEvidence(vision, audio, 100).matches.length, 0);
  for (const age of [NaN, Infinity, -1]) assert.throws(() => apply(h, frame({ age_seconds: age })));
});

test("startup is idle with no fabricated notes or events", () => {
  const h = harness();
  assert.match(h.text("provenanceBadge"), /^IDLE/);
  for (const id of ["chordName", "audioEvidence", "chordCandidates", "noteResult"]) assert.equal(h.text(id), "--");
  for (const id of ["visualEvent", "audioEvent", "resultEvent"]) assert.equal(h.elements.get(`#${id}`).hidden, true);
  assert.equal(h.run("state.demo"), false);
  assert.equal(h.run("state.hardware"), false);
});

test("temporal confirmation separates raw candidate, confirmed chord and silence history", () => {
  const h = harness();
  const tracked = { tracking: { chord_status: "pending" }, stable_chord: { status: "unknown" },
    scale_evidence: { window_seconds: 12, compatible_group_count: 1,
      candidates: [{ tonic: "C", scale: "major" }, { tonic: "A", scale: "natural_minor" },
        { tonic: "D", scale: "dorian" }] } };
  apply(h, frame(tracked));
  assert.equal(h.text("chordName"), "--");
  assert.match(h.text("chordCandidates"), /C/);
  assert.match(h.text("agreementText"), /确认中/);
  assert.match(h.text("scaleResult"), /C 大调.*A 自然小调/);
  assert.match(h.text("scaleAlternative"), /近 12 秒.*主音未确定/);
  apply(h, frame({ ...tracked, sequence: 2, tracking: { chord_status: "confirmed" },
    stable_chord: { status: "recognized", label: "C", confidence: .91 } }));
  assert.equal(h.text("chordName"), "C");
  apply(h, frame({ ...tracked, sequence: 3, status: "silence" }));
  assert.equal(h.text("chordName"), "--");
  assert.equal(h.text("chordCandidates"), "--");
  assert.match(h.text("scaleResult"), /C 大调/);
});

test("active temporal holding and switching show only the stable chord", () => {
  const h = harness();
  const tracking = { chord_status: "holding" };
  apply(h, frame({ status: "unknown", signal_active: true, notes: [{ note_name: "D3", frequency_hz: 146.832 }],
    chord: { status: "recognized", label: "D", confidence: .8, alternatives: [] }, tracking,
    stable_chord: { status: "recognized", label: "C", confidence: .91 } }));
  assert.equal(h.text("chordName"), "C");
  assert.equal(h.text("chordNotes"), "未识别");
  assert.equal(h.text("chordCandidates"), "--");
  assert.equal(h.text("agreementText"), "延音保持");
  apply(h, frame({ status: "unknown", signal_active: true, sequence: 2, tracking: { chord_status: "switching" },
    stable_chord: { status: "recognized", label: "C", confidence: .91 } }));
  assert.equal(h.text("chordName"), "C");
  assert.equal(h.text("agreementText"), "换和弦确认中");
});

test("inactive, silent, and stale temporal frames cannot retain a chord", () => {
  const h = harness();
  const held = { signal_active: true, tracking: { chord_status: "holding" },
    stable_chord: { status: "recognized", label: "C", confidence: .91 } };
  apply(h, frame({ status: "unknown", ...held }));
  assert.equal(h.text("chordName"), "C");
  apply(h, frame({ status: "unknown", signal_active: false, sequence: 2, tracking: { chord_status: "holding" },
    stable_chord: held.stable_chord }));
  assert.equal(h.text("chordName"), "--");
  apply(h, frame({ signal_active: false, sequence: 3, tracking: { chord_status: "switching" },
    stable_chord: held.stable_chord }));
  assert.equal(h.text("chordName"), "--");
  apply(h, frame({ status: "silence", sequence: 4, ...held }));
  assert.equal(h.text("chordName"), "--");
  apply(h, frame({ sequence: 5, stale: true, ...held }));
  assert.equal(h.text("chordName"), "--");
});

test("chromatic history is distinguished from insufficient notes", () => {
  const h = harness();
  apply(h, frame({ scale_evidence: { status: "mixed_or_chromatic", candidates: [] } }));
  assert.match(h.text("scaleAlternative"), /混合或半音化/);
  assert.doesNotMatch(h.text("scaleAlternative"), /等待更多/);
});

test("PS polyphony, chord alternatives, confidence and sequence are shown without fusion claims", () => {
  const h = harness();
  apply(h, frame());
  assert.match(h.text("provenanceBadge"), /^FPGA \/ PS/);
  assert.equal(h.text("chordName"), "C");
  assert.equal(h.text("fusionConfidence"), "91%");
  assert.match(h.text("noteResult"), /C3.*130\.81.*E3.*164\.81.*G3.*196\.00/);
  assert.match(h.text("chordCandidates"), /C · 91% \/ Csus · 70%/);
  assert.match(h.text("processingSequence"), /#1/);
  assert.match(h.text("waveformStatus"), /未提供时域波形/);
  assert.match(h.text("spectrumRange"), /4 kHz/);
  assert.match(h.text("visualEvidence"), /尚无可靠视觉位置.*仅用音频/);
  assert.equal(h.elements.get("#visualEvent").hidden, true);
});

test("non-hardware sources and malformed payloads cannot assert hardware recognition", () => {
  const h = harness();
  for (const payload of [null, frame({ source: "demo" }), frame({ source: undefined }), frame({ sequence: null }),
    frame({ config: null }), frame({ notes: [{ note_name: "C3", frequency_hz: -1 }] }),
    frame({ spectrum: { bins: [1, -2] } }), frame({ status: "banana" })]) {
    h.sandbox.payload = payload;
    assert.throws(() => h.run("normalizeHardwareFrame(payload)"));
  }
});

test("internal PL diagnostic signals are labelled separately from audio input", () => {
  const h = harness();
  apply(h, frame({ source: "hardware", fft: { analysis_test_mode: 2 } }));
  assert.match(h.text("provenanceBadge"), /内部 C 和弦测试/);
  assert.match(h.text("controlMessage"), /内部 C 和弦测试/);
  apply(h, frame({ source: "hardware", sequence: 2, fft: { analysis_test_mode: 0 } }));
  assert.match(h.text("provenanceBadge"), /音频输入/);
  assert.doesNotMatch(h.text("provenanceBadge"), /内部/);
});

test("silence and unknown clear notes and chord even if payload contains old assertions", () => {
  const h = harness();
  for (const status of ["silence", "unknown"]) {
    apply(h, frame());
    apply(h, frame({ status, sequence: 2 }));
    assert.equal(h.text("chordName"), "--");
    assert.equal(h.text("noteResult"), "--");
    assert.equal(h.text("chordCandidates"), "--");
    assert.equal(h.text("fusionConfidence"), "--");
  }
});

test("unavailable and expired snapshots clear all recognition", () => {
  const h = harness();
  for (const fields of [{ status: "starting" }, { status: "idle" }, { status: "offline" },
    { status: "error", error: "FFT unavailable" }, { stale: true }, { age_seconds: 4 }]) {
    apply(h, frame());
    apply(h, frame(fields));
    for (const id of ["chordName", "noteResult", "chordCandidates", "fusionConfidence"]) assert.equal(h.text(id), "--");
    assert.equal(h.elements.get("#resultEvent").hidden, true);
  }
});

test("unchanged sequence expires; a fresh wrapped sequence recovers", () => {
  const h = harness();
  apply(h, frame({ sequence: 0xFFFFFFFF }), 0);
  apply(h, frame({ sequence: 0xFFFFFFFF }), 3100);
  assert.match(h.text("provenanceBadge"), /过期/);
  assert.equal(h.text("noteResult"), "--");
  apply(h, frame({ sequence: 0 }), 3200);
  assert.equal(h.text("chordName"), "C");
  assert.match(h.text("processingSequence"), /#0/);
});

test("backend text is never inserted as HTML", () => {
  const h = harness();
  const label = "<img src=x onerror=alert(1)>";
  apply(h, frame({ chord: { status: "recognized", label, confidence: 0.5, alternatives: [] } }));
  assert.equal(h.text("chordName"), label);
  assert.equal(h.elements.get("#chordName").innerHTML, "");
});

test("polling uses same origin and stop prevents late replies and new requests", async () => {
  let resolve;
  const requests = [];
  const h = harness((url, options) => { requests.push({ url, options }); return new Promise((done) => { resolve = done; }); });
  h.run("startHardware()");
  assert.equal(requests[0].url, "/api/audio");
  assert.equal(requests[0].options.cache, "no-store");
  h.run("stopHardware()");
  assert.equal(requests[0].options.signal.aborted, true);
  resolve({ ok: true, json: async () => frame() });
  await flush();
  assert.match(h.text("provenanceBadge"), /^IDLE/);
  assert.equal(h.text("noteResult"), "--");
  assert.equal(h.run("state.hardwareRequest"), null);
  assert.equal(requests.length, 1);
  assert.equal([...h.timers.values()].filter((timer) => timer.delay === 500).length, 0);
});

test("HTTP failures clear a previous result while polling can recover", async () => {
  let response = { ok: false, status: 503 };
  const h = harness(async () => response);
  h.run("startHardware()");
  apply(h, frame());
  await flush();
  assert.equal(h.text("chordName"), "--");
  assert.match(h.text("controlMessage"), /HTTP 503/);
  response = { ok: true, json: async () => frame({ sequence: 2 }) };
  await h.run("pollHardware(state.hardwareSession)");
  assert.equal(h.text("chordName"), "C");
  h.run("stopHardware()");
});

test("demo explicitly stops hardware and cannot be overwritten by a pending reply", async () => {
  let resolve;
  const h = harness(() => new Promise((done) => { resolve = done; }));
  h.run("startHardware()");
  h.elements.get("#demoButton").listeners.click();
  resolve({ ok: true, json: async () => frame() });
  await flush();
  assert.equal(h.run("state.hardware"), false);
  assert.match(h.text("provenanceBadge"), /^DEMO/);
  assert.equal(h.text("chordName"), "G Major");
});
