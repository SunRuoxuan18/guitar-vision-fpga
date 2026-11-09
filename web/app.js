const $ = (selector) => document.querySelector(selector);

const refs = {
  cameraButton: $("#cameraButton"), audioButton: $("#audioButton"), hardwareButton: $("#hardwareButton"), demoButton: $("#demoButton"),
  cameraSelect: $("#cameraSelect"), inputSelect: $("#inputSelect"), controlMessage: $("#controlMessage"), provenance: $("#provenanceBadge"),
  cameraChip: $("#cameraChip"), audioChip: $("#audioChip"), fpgaChip: $("#fpgaChip"),
  cameraStage: $("#cameraStage"), cameraVideo: $("#cameraVideo"), overlay: $("#fretOverlay"),
  overlayState: $("#overlayState"), videoRate: $("#videoRate"), frameStatusText: $("#frameStatusText"),
  cameraHint: $("#cameraHint"), frameClock: $("#frameClock"), chordName: $("#chordName"),
  chordNotes: $("#chordNotes"), fusionConfidence: $("#fusionConfidence"), agreementText: $("#agreementText"),
  agreementBar: $("#agreementBar"), visualEvidence: $("#visualEvidence"), audioEvidence: $("#audioEvidence"),
  scaleResult: $("#scaleResult"), scaleAlternative: $("#scaleAlternative"), noteResult: $("#noteResult"),
  truthNote: $("#truthNote"), sampleRate: $("#sampleRateText"), waveCanvas: $("#waveformCanvas"),
  spectrumCanvas: $("#spectrumCanvas"), toast: $("#toast"), clock: $("#footerClock"),
  cameraEmptyTitle: $("#cameraEmptyTitle"), candidates: $("#chordCandidates"), confidenceLabel: $("#confidenceLabel"),
  timelineStatus: $("#timelineStatus"), timelineTitle: $("#timelineTitle"), timelineAxis: $("#timelineAxis"),
  visualEvent: $("#visualEvent"), audioEvent: $("#audioEvent"), resultEvent: $("#resultEvent"),
  waveformStatus: $("#waveformStatus"), spectrumRange: $("#spectrumRange"), processingSequence: $("#processingSequence")
};

const state = {
  demo: false, cameraStream: null, audioStream: null, audioContext: null, analyser: null,
  audioSource: null, animationId: 0, demoPhase: 0, lastPitchAt: 0, audioStarting: false, audioSession: 0,
  hardware: false, hardwareTimer: 0, hardwareWatchdog: 0, hardwareRequest: null, hardwareSession: 0,
  hardwareSequence: null, hardwareUpdatedAt: 0, hardwareResult: null
};

const noteNames = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

function setChip(element, value) { element.dataset.state = value; }
function showToast(message) {
  refs.toast.textContent = message;
  refs.toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => refs.toast.classList.remove("is-visible"), 3200);
}
function setProvenance(kind, text) { refs.provenance.dataset.kind = kind; refs.provenance.textContent = text; }

function setEvent(element, title, detail) {
  element.hidden = !title;
  if (!title) return;
  element.querySelector("strong").textContent = title;
  element.querySelector("small").textContent = detail;
  element.title = `${title} · ${detail}`;
}

function resetTimeline(status) {
  refs.timelineStatus.textContent = status;
  refs.timelineTitle.textContent = "音频处理序列";
  refs.timelineAxis.textContent = "--";
  setEvent(refs.visualEvent, null);
  setEvent(refs.audioEvent, null);
  setEvent(refs.resultEvent, null);
}

function clearResult(message) {
  setFusionResult({
    chord: "--", notes: message, confidence: 0, agreement: "尚无结论",
    visual: state.cameraStream ? "画面已连接 · 视觉识别待接入" : "视觉识别待接入",
    audio: "--", scale: "--", alternative: "等待连续音符", note: "--", truth: "尚无识别结果。"
  });
  refs.fusionConfidence.textContent = "--";
  refs.candidates.textContent = "--";
  refs.confidenceLabel.textContent = "视听一致性";
}

function showIdleResult() {
  state.demo = false;
  cancelAnimationFrame(state.animationId);
  setProvenance("idle", "IDLE · 无音频输入");
  clearResult("等待音频输入");
  refs.controlMessage.textContent = "等待音频输入。";
  refs.processingSequence.textContent = "IDLE · 无处理帧";
  refs.sampleRate.textContent = "--";
  refs.waveformStatus.textContent = "无输入";
  refs.spectrumRange.textContent = "--";
  refs.overlayState.textContent = "待标定";
  refs.cameraEmptyTitle.textContent = "摄像头未连接";
  if (!state.cameraStream) {
    refs.cameraStage.dataset.mode = "idle";
    refs.frameStatusText.textContent = "CAMERA OFFLINE";
  }
  setChip(refs.audioChip, "idle");
  setChip(refs.fpgaChip, "idle");
  resetTimeline("IDLE");
  drawFretOverlay(false);
  drawQuietSignals();
}

function updateClocks() {
  const now = new Date();
  refs.clock.textContent = now.toLocaleTimeString("zh-CN", { hour12: false });
  refs.frameClock.textContent = `${now.toLocaleTimeString("zh-CN", { hour12: false })}.${String(now.getMilliseconds()).padStart(3, "0")}`;
}
setInterval(updateClocks, 83);
updateClocks();

function setFusionResult({ chord, notes, confidence, agreement, visual, audio, scale, alternative, note, truth }) {
  refs.chordName.textContent = chord;
  refs.chordNotes.textContent = notes;
  refs.fusionConfidence.textContent = `${confidence}%`;
  refs.agreementText.textContent = agreement;
  refs.agreementBar.style.width = `${confidence}%`;
  refs.visualEvidence.textContent = visual;
  refs.audioEvidence.textContent = audio;
  refs.scaleResult.textContent = scale;
  refs.scaleAlternative.textContent = alternative;
  refs.noteResult.textContent = note;
  refs.truthNote.innerHTML = truth;
}

function showDemoResult() {
  state.demo = true;
  setProvenance("demo", "DEMO · 模拟识别结果");
  setFusionResult({
    chord: "G Major", notes: "构成音 G · B · D", confidence: 94, agreement: "高度一致",
    visual: "6弦3品 · 5弦2品 · 1弦3品", audio: "G3 · B3 · D4", scale: "G Ionian",
    alternative: "次选：E Aeolian", note: "G3 · 196.00 Hz",
    truth: "<strong>当前为演示：</strong>指法、和弦和音阶均为模拟数据，用来展示最终系统应该怎样工作。"
  });
  refs.overlayState.textContent = "GUIDE + DEMO";
  refs.cameraEmptyTitle.textContent = "DEMO FRETBOARD";
  refs.candidates.textContent = "G Major · 94% · DEMO";
  refs.confidenceLabel.textContent = "视听一致性";
  refs.sampleRate.textContent = "48 kHz · DEMO";
  refs.waveformStatus.textContent = "DEMO · 模拟波形";
  refs.spectrumRange.textContent = "0 — 6 kHz · DEMO";
  refs.processingSequence.textContent = "DEMO · 模拟处理帧";
  refs.timelineStatus.textContent = "DEMO · 最近 4 秒";
  refs.timelineTitle.textContent = "最近识别时间线";
  refs.timelineAxis.textContent = "0 ms · DEMO";
  setEvent(refs.visualEvent, "视觉落指 · DEMO", "6弦3品");
  setEvent(refs.audioEvent, "音频起音 · DEMO", "G3");
  setEvent(refs.resultEvent, "融合稳定 · DEMO", "G Major · 94%");
  setChip(refs.fpgaChip, "idle");
  if (!state.cameraStream) {
    refs.cameraStage.dataset.mode = "demo";
    refs.frameStatusText.textContent = "SIMULATED FRAME";
  }
  refs.controlMessage.textContent = "完整演示：视觉落指 + G/B/D 音频峰值 → G Major，全部带 DEMO 标记。";
  drawFretOverlay(true);
}

function showLiveWaitingResult() {
  const boardCamera = Boolean(state.cameraStream && $("#pcPlEnabled").checked);
  if (state.hardware) {
    refs.visualEvidence.textContent = boardCamera ? "画面已连接 · 等待板卡视觉结果" :
      state.cameraStream ? "画面已连接 · 板卡处理未启用" : "摄像头未连接";
    return;
  }
  if (!state.analyser && !state.cameraStream && !state.audioStarting) {
    showIdleResult();
    return;
  }
  state.demo = false;
  if (!state.analyser) cancelAnimationFrame(state.animationId);
  setProvenance("live", state.analyser ? "BROWSER MIC · 单音检测" :
    boardCamera ? "LIVE CAMERA · 等待板卡处理" : "LIVE CAMERA · 仅预览");
  setFusionResult({
    chord: "等待融合", notes: "需要视觉关键点与多个音频峰值", confidence: 0, agreement: "尚无结论",
    visual: boardCamera ? "画面已连接 · 等待板卡视觉结果" :
      state.cameraStream ? "画面已连接 · 板卡处理未启用" : "摄像头未连接",
    audio: state.analyser ? "音频已连接 · 正在检测单音" : "等待板卡音频连接",
    scale: "等待连续音符", alternative: "积累音级后给出调式候选", note: "--",
    truth: boardCamera ? "<strong>真实输入状态：</strong>板卡 C922 采集并传图，等待板卡 PS/PL 返回视觉结果；连接板卡音频后才能进行音视频联合判断。" :
      "<strong>真实输入状态：</strong>当前仅预览摄像头或检测浏览器单音；请连接板卡处理链路后查看联合判断。"
  });
  refs.candidates.textContent = "--";
  refs.confidenceLabel.textContent = "视听一致性";
  refs.processingSequence.textContent = boardCamera ? "PC → PS / PL · 等待板卡结果" : "BROWSER · 非 FPGA 输出";
  refs.waveformStatus.textContent = state.analyser ? "BROWSER MIC · 真实输入" : "无音频输入";
  refs.spectrumRange.textContent = state.analyser ? `0 — ${(state.audioContext.sampleRate / 8 / 1000).toFixed(1)} kHz` : "--";
  refs.cameraEmptyTitle.textContent = "CAMERA OFFLINE";
  refs.overlayState.textContent = boardCamera ? "等待板卡视觉结果" : "仅预览";
  resetTimeline(boardCamera ? "等待板卡音频" : "BROWSER · 单音 / 预览");
  drawFretOverlay(false);
  if (!state.analyser) drawQuietSignals();
  refs.controlMessage.textContent = boardCamera ? "画面已连接；等待板卡处理结果与音频连接。" :
    "真实输入已连接；等待板卡处理链路。";
}

const HARDWARE_POLL_MS = 100;
const HARDWARE_STALE_MS = 3000;

function confidencePercent(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
    ? Math.round(value * 100) : null;
}

function normalizeHardwareFrame(payload) {
  if (!payload || typeof payload !== "object" || !["fpga", "hardware"].includes(payload.source)) {
    throw new Error("音频来源未经确认：需要 FPGA / PS 数据");
  }
  const statuses = ["recognized", "silence", "unknown", "starting", "idle", "offline", "stale", "error", "hardware_unavailable"];
  if (!statuses.includes(payload.status)) throw new Error("音频状态格式无效");
  if (["starting", "idle", "offline", "stale", "error", "hardware_unavailable"].includes(payload.status)) {
    return { status: payload.status === "hardware_unavailable" ? "offline" : payload.status, error: typeof payload.error === "string" ? payload.error : null };
  }
  if (payload.age_seconds !== undefined && (!Number.isFinite(payload.age_seconds) || payload.age_seconds < 0)) {
    throw new Error("板卡处理帧时间无效");
  }
  if (payload.stale === true || (typeof payload.age_seconds === "number" && payload.age_seconds > HARDWARE_STALE_MS / 1000)) {
    return { status: "stale", error: "板卡音频暂无新数据，请检查音频服务；此状态与摄像头无关" };
  }
  const sequence = payload.sequence;
  if (!Number.isInteger(sequence) || sequence < 0 || sequence > 0xFFFFFFFF) throw new Error("缺少有效处理帧序号");
  const sampleRate = payload.config?.sample_rate;
  const fftSize = payload.config?.fft_size;
  if (!(Number.isFinite(sampleRate) && sampleRate > 0 && Number.isInteger(fftSize) && fftSize >= 64 && fftSize % 2 === 0)) {
    throw new Error("缺少采样率或 FFT 长度");
  }
  if (!Array.isArray(payload.notes)) throw new Error("音符结果格式无效");
  const notes = payload.status === "recognized" ? payload.notes.map((note) => {
    if (!note || typeof note.note_name !== "string" || !note.note_name.trim() ||
        !(Number.isFinite(note.frequency_hz) && note.frequency_hz > 0)) {
      throw new Error("音符名称或频率格式无效");
    }
    return { name: note.note_name, frequency: note.frequency_hz, midi: note.midi_note,
      confidence: confidencePercent(note.confidence), ambiguous: note.octave_ambiguous === true };
  }) : [];
  const rawChord = payload.status === "recognized" && payload.chord?.status === "recognized" ? payload.chord : null;
  const tracked = payload.tracking && typeof payload.tracking === "object";
  const chordStatus = tracked ? payload.tracking.chord_status : null;
  const temporalHold = ["holding", "switching"].includes(chordStatus);
  const held = tracked && payload.signal_active === true && payload.status !== "silence" &&
    temporalHold && payload.stable_chord?.status === "recognized";
  const chord = tracked ? (payload.stable_chord?.status === "recognized" &&
    (held || (payload.status === "recognized" && !temporalHold)) ? payload.stable_chord : null) : rawChord;
  for (const item of [rawChord, chord]) {
    if (item && (typeof item.label !== "string" || !item.label.trim())) throw new Error("和弦名称格式无效");
  }
  const candidates = rawChord ? [rawChord, ...(Array.isArray(rawChord.alternatives) ? rawChord.alternatives : [])]
    .filter((candidate) => typeof candidate.label === "string" && candidate.label.trim())
    .map((candidate) => ({ label: candidate.label, confidence: confidencePercent(candidate.confidence) })) : [];
  const power = payload.spectrum?.bins;
  if (power !== undefined && (!Array.isArray(power) || power.length !== fftSize / 2 ||
      power.some((value) => !Number.isFinite(value) || value < 0))) {
    throw new Error("频谱必须为 FFT 正频率线性功率数据");
  }
  const scaleCandidates = payload.scale_evidence?.candidates;
  const modeNames = { major: "大调", natural_minor: "自然小调", dorian: "多利亚", phrygian: "弗里几亚",
    lydian: "利底亚", mixolydian: "混合利底亚", locrian: "洛克里亚" };
  const scales = Array.isArray(scaleCandidates) ? scaleCandidates
    .filter((candidate) => typeof candidate.tonic === "string" && typeof candidate.scale === "string")
    .map((candidate) => `${candidate.tonic} ${modeNames[candidate.scale] || candidate.scale}`) : [];
  const testMode = payload.fft?.analysis_test_mode ?? 0;
  if (!Number.isInteger(testMode) || testMode < 0 || testMode > 3) throw new Error("分析输入模式无效");
  return { status: payload.status, sequence, sampleRate, fftSize, notes, candidates, scales, testMode,
    boardFusion: payload.fusion?.processing_location === "ps_linux" ? payload.fusion : null,
    age_seconds: payload.age_seconds ?? 0,
    droppedFrames: Number.isInteger(payload.fft?.drops) && payload.fft.drops >= 0 ? payload.fft.drops : null,
    scaleCandidates: Array.isArray(scaleCandidates) ? scaleCandidates.map(candidate => ({
      tonic: candidate.tonic, scale: candidate.scale
    })) : [],
    chordLabel: chord?.label || null, tracked, pending: chordStatus === "pending", heldStatus: held ? chordStatus : null,
    rollingScales: tracked && Number.isFinite(payload.scale_evidence?.window_seconds),
    scaleStatus: payload.scale_evidence?.status,
    scaleWindow: payload.scale_evidence?.window_seconds, scaleGroups: payload.scale_evidence?.compatible_group_count,
    confidence: chord ? confidencePercent(chord.confidence) : null, ambiguous: chord?.ambiguous === true,
    power: power || null };
}

function showHardwareUnavailable(status, message) {
  state.hardwareResult = null;
  $("#protocolStatus").textContent = "等待有效板卡数据";
  const labels = { starting: "连接中", idle: "等待帧", offline: "离线", stale: "数据过期", error: "错误" };
  const label = labels[status] || "错误";
  setProvenance("hardware", `FPGA / PS · ${label}`);
  clearResult(`板卡音频 · ${label}`);
  refs.confidenceLabel.textContent = "音频和弦置信度";
  refs.truthNote.textContent = "板卡音频暂不可用；视觉状态见视频";
  refs.controlMessage.textContent = message || `FPGA / PS · ${label}`;
  refs.sampleRate.textContent = "--";
  refs.waveformStatus.textContent = "无有效硬件波形";
  refs.spectrumRange.textContent = "--";
  refs.processingSequence.textContent = `PL → PS → WEB · ${label}`;
  setChip(refs.fpgaChip, ["starting", "idle"].includes(status) ? "pending" : "error");
  setChip(refs.audioChip, "idle");
  resetTimeline(`FPGA · ${label}`);
  drawQuietSignals();
}

function showHardwareFrame(frame) {
  state.hardwareResult = frame;
  $("#protocolStatus").textContent = `I2S → PCM → FFT ${frame.fftSize} 点 · ${frame.sampleRate / 1000} kHz · AXI 快照 #${frame.sequence}${frame.droppedFrames === null ? "" : ` · 丢帧 ${frame.droppedFrames}`}`;
  const noteNamesText = frame.notes.map((note) => `${note.name}${note.ambiguous ? "?" : ""}`).join(" · ");
  const candidateText = frame.candidates.map((candidate) => `${candidate.label}${candidate.confidence === null ? "" : ` · ${candidate.confidence}%`}`).join(" / ");
  const statusText = { recognized: "多音推断", silence: "静音", unknown: "未识别" }[frame.status];
  const sourceText = frame.testMode ? ["", "内部 A4 测试", "内部 C 和弦测试", "内部 Am 和弦测试"][frame.testMode] : "音频输入";
  const fusion = frame.boardFusion;
  const span = fusion?.hand_fret_span;
  const fretText = Array.isArray(span) && span.length === 2 && span.every(Number.isInteger)
    ? ` · ${span[0]}–${span[1]} 品候选` : "";
  const supportedFingers = Number.isInteger(fusion?.joint_decision?.supported_fingers)
    ? fusion.joint_decision.supported_fingers : "?";
  const visualText = fusion?.joint_decision?.status === "awaiting_audio_transition" ? "手部正在换位，等待当前音高确认" :
      fusion?.joint_decision?.status === "visual_audio_conflict" ? "手指投影与当前音高不一致，联合判断待确认" :
      fusion?.joint_decision?.status === "audio_visual_consistent" ? `音视频支持 ${supportedFingers} 根手指位置${fretText} · 按压状态待确认` :
      fusion?.status === "audio_visual_aligned" ? `手指位置与音高已联合对齐${fretText} · 按压状态待确认` :
      fusion?.status === "audio_visual_supported" ? "手指位置与当前音高吻合 · 按压状态待确认" :
    ["mapping_candidate", "mapping_ambiguous"].includes(fusion?.status) ?
      `音视频推算 ${Number.isInteger(fusion.mapping_alternatives) ? fusion.mapping_alternatives : "?"} 种弦品映射候选${fretText}，尚未确认绝对品号` :
    fusion?.reason === "vision_stale" ? "视觉结果过期，当前仅用音频" :
    fusion?.reason === "audio_visual_time_mismatch" ? "音视频采样时刻不匹配，当前仅用音频" :
    fusion?.position_reason === "hand_outside_measured_lattice" ? "手已检测到，但不在已测品丝范围内" :
    "尚无可靠视觉位置，当前仅用音频";
  frame.visualFeedback = visualText;
  setProvenance("hardware", `FPGA / PS · ${sourceText} · ${statusText}`);
  setFusionResult({
    chord: frame.chordLabel || "--", notes: noteNamesText || statusText,
    confidence: frame.confidence ?? 0, agreement: frame.heldStatus === "holding" ? "延音保持" : frame.heldStatus === "switching" ? "换和弦确认中" : frame.pending ? "候选确认中" : frame.ambiguous ? "存在多个解释" : frame.confidence === null ? "尚无和弦结论" : `${frame.confidence}%`,
    visual: visualText,
    audio: frame.notes.map((note) => `${note.name}${note.confidence === null ? "" : ` ${note.confidence}%`}${note.ambiguous ? "（八度不确定）" : ""}`).join(" · ") || statusText,
    scale: frame.scales.slice(0, 2).join(" / ") || "--",
    alternative: frame.scales.length ? (frame.rollingScales
      ? `近 ${frame.scaleWindow} 秒 · ${frame.scaleGroups || 1} 组候选 · 同组调式：${frame.scales.slice(0, 7).join(" / ")} · 主音未确定`
      : "单帧音级相容性 · 调性未确定") : frame.scaleStatus === "mixed_or_chromatic"
        ? "混合或半音化音级，暂无相容调式" : "等待更多不同音级",
    note: frame.notes.map((note) => `${note.name} · ${note.frequency.toFixed(2)} Hz`).join(" / ") || "--",
    truth: frame.testMode ? "板卡内部测试信号，非吉他输入。" : visualText
  });
  refs.fusionConfidence.textContent = frame.confidence === null ? "--" : `${frame.confidence}%`;
  refs.confidenceLabel.textContent = "音频和弦置信度";
  refs.candidates.textContent = candidateText || "--";
  refs.candidates.title = candidateText;
  refs.scaleResult.title = frame.scales.join(" / ");
  refs.sampleRate.textContent = `${frame.sampleRate / 1000} kHz · PL`;
  refs.waveformStatus.textContent = "FPGA API 未提供时域波形";
  refs.spectrumRange.textContent = frame.power ? `0 — ${Math.min(4000, frame.sampleRate / 2) / 1000} kHz · 相对功率` : "频谱未提供";
  refs.controlMessage.textContent = `${sourceText} · ${statusText}`;
  refs.processingSequence.textContent = `PL → PS → WEB · #${frame.sequence}`;
  refs.timelineStatus.textContent = `FPGA · #${frame.sequence}`;
  refs.timelineTitle.textContent = "音频处理序列";
  refs.timelineAxis.textContent = `#${frame.sequence} · PL → PS`;
  setEvent(refs.visualEvent, null);
  setEvent(refs.audioEvent, "PL 频谱帧", `#${frame.sequence}`);
  setEvent(refs.resultEvent, "PS 音频推断", candidateText || noteNamesText || statusText);
  setChip(refs.fpgaChip, "live");
  setChip(refs.audioChip, "live");
  drawWaveform(new Float32Array(256));
  if (frame.power) drawHardwareSpectrum(frame); else drawSpectrum(new Uint8Array(256));
}

function acceptHardwarePayload(payload, now = performance.now()) {
  const frame = normalizeHardwareFrame(payload);
  frame.ageReportedAt = now;
  if (frame.sequence === undefined) {
    showHardwareUnavailable(frame.status, frame.error);
    return;
  }
  if (frame.sequence !== state.hardwareSequence) {
    state.hardwareSequence = frame.sequence;
    state.hardwareUpdatedAt = now;
  } else if (now - state.hardwareUpdatedAt > HARDWARE_STALE_MS) {
    showHardwareUnavailable("stale", "FPGA 处理帧已停止更新");
    return;
  }
  showHardwareFrame(frame);
}

async function pollHardware(session) {
  if (!state.hardware || session !== state.hardwareSession) return;
  const started = performance.now();
  const controller = new AbortController();
  state.hardwareRequest = controller;
  const timeout = window.setTimeout(() => controller.abort(), HARDWARE_STALE_MS);
  try {
    const response = await fetch("/api/audio", { cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (!state.hardware || session !== state.hardwareSession) return;
    acceptHardwarePayload(payload);
  } catch (error) {
    if (state.hardware && session === state.hardwareSession) {
      showHardwareUnavailable(error.name === "AbortError" || error instanceof TypeError ? "offline" : "error",
        `FPGA 音频不可用：${error.name === "AbortError" ? "请求超时" : error.message}`);
    }
  } finally {
    window.clearTimeout(timeout);
    if (state.hardwareRequest === controller) state.hardwareRequest = null;
    if (state.hardware && session === state.hardwareSession) {
      state.hardwareTimer = window.setTimeout(() => pollHardware(session),
        Math.max(0, HARDWARE_POLL_MS - (performance.now() - started)));
    }
  }
}

function stopHardware() {
  state.hardware = false;
  state.hardwareSession += 1;
  window.clearTimeout(state.hardwareTimer);
  window.clearInterval(state.hardwareWatchdog);
  state.hardwareRequest?.abort();
  state.hardwareRequest = null;
  state.hardwareResult = null;
  state.hardwareSequence = null;
  refs.hardwareButton.textContent = "连接板卡音频";
  refs.hardwareButton.setAttribute("aria-pressed", "false");
  refs.hardwareButton.title = "连接板卡音频";
  showIdleResult();
}

function startHardware() {
  if (state.hardware) { stopHardware(); return; }
  state.demo = false;
  stopAudio();
  cancelAnimationFrame(state.animationId);
  state.hardware = true;
  state.hardwareSession += 1;
  state.hardwareSequence = null;
  state.hardwareUpdatedAt = 0;
  refs.hardwareButton.textContent = "断开音频";
  refs.hardwareButton.setAttribute("aria-pressed", "true");
  refs.hardwareButton.title = "停止板卡音频显示";
  showHardwareUnavailable("starting");
  state.hardwareWatchdog = window.setInterval(() => {
    if (state.hardwareResult && performance.now() - state.hardwareUpdatedAt > HARDWARE_STALE_MS) {
      showHardwareUnavailable("stale", "FPGA 处理帧已停止更新");
    }
  }, HARDWARE_POLL_MS);
  pollHardware(state.hardwareSession);
}

function drawFretOverlay(showFingers = state.demo) {
  if (window.FretVision?.active) { window.FretVision.render(); return; }
  const canvas = refs.overlay;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!state.demo) return;

  if (state.demo && !state.cameraStream) {
    const bg = ctx.createLinearGradient(0, 0, w, h);
    bg.addColorStop(0, "#151b19");
    bg.addColorStop(1, "#080b0a");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
  }

  const left = w * .08;
  const right = w * .94;
  const top = h * .22;
  const bottom = h * .79;
  ctx.lineCap = "round";
  for (let string = 0; string < 6; string += 1) {
    const y = top + (bottom - top) * string / 5;
    ctx.strokeStyle = string === 0 || string === 5 ? "rgba(102,214,209,.78)" : "rgba(102,214,209,.52)";
    ctx.lineWidth = 1 + string * .22;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(237,241,233,.7)";
    ctx.font = "18px Consolas";
    ctx.fillText(String(6 - string), left - 30, y + 6);
  }
  for (let fret = 0; fret <= 9; fret += 1) {
    const ratio = fret / 9;
    const x = left + (right - left) * (1 - Math.pow(1 - ratio, 1.14));
    ctx.strokeStyle = fret === 0 ? "rgba(220,255,88,.9)" : "rgba(237,241,233,.32)";
    ctx.lineWidth = fret === 0 ? 5 : 2;
    ctx.beginPath();
    ctx.moveTo(x, top - 24);
    ctx.lineTo(x, bottom + 24);
    ctx.stroke();
    if (fret > 0) {
      ctx.fillStyle = "rgba(237,241,233,.55)";
      ctx.font = "16px Consolas";
      ctx.fillText(String(fret), x - 4, bottom + 54);
    }
  }

  if (showFingers) {
    const points = [
      { string: 0, fret: 3, label: "G" },
      { string: 1, fret: 2, label: "B" },
      { string: 5, fret: 3, label: "G" }
    ];
    points.forEach((point, index) => {
      const startRatio = (point.fret - 1) / 9;
      const endRatio = point.fret / 9;
      const x1 = left + (right - left) * (1 - Math.pow(1 - startRatio, 1.14));
      const x2 = left + (right - left) * (1 - Math.pow(1 - endRatio, 1.14));
      const x = (x1 + x2) / 2;
      const y = top + (bottom - top) * point.string / 5;
      ctx.fillStyle = index === 1 ? "#dcff58" : "#b7a5ff";
      ctx.shadowColor = ctx.fillStyle;
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.arc(x, y, 17, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#0b0e0e";
      ctx.font = "700 15px system-ui";
      ctx.textAlign = "center";
      ctx.fillText(point.label, x, y + 5);
      ctx.textAlign = "start";
    });
  }
}

function keepOverlaySized() {
  const width = Math.max(640, Math.round(refs.cameraStage.clientWidth * (window.devicePixelRatio || 1)));
  const height = Math.round(width * refs.cameraStage.clientHeight / Math.max(1, refs.cameraStage.clientWidth));
  if (refs.overlay.width !== width || refs.overlay.height !== height) {
    refs.overlay.width = width;
    refs.overlay.height = height;
  }
  drawFretOverlay(state.demo);
}
window.addEventListener("resize", keepOverlaySized);

function populateCameraDevices(devices, selectedId) {
  const cameras = devices.filter((device) => device.kind === "videoinput");
  refs.cameraSelect.innerHTML = "";
  cameras.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `摄像头 ${index + 1}`;
    option.selected = device.deviceId === selectedId;
    refs.cameraSelect.appendChild(option);
  });
  refs.cameraSelect.disabled = cameras.length === 0;
}

async function startCamera(deviceId) {
  if (!navigator.mediaDevices?.getUserMedia) {
    setChip(refs.cameraChip, "error");
    refs.controlMessage.textContent = "当前浏览器不支持摄像头访问，请使用 localhost 下的现代浏览器。";
    return;
  }
  if (state.cameraStream) {
    stopCamera();
    return;
  }
  refs.cameraButton.textContent = "正在连接…";
  try {
    let stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 }
      },
      audio: false
    });
    if (!deviceId) {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const external = devices.find(device => device.kind === "videoinput" && /c922/i.test(device.label));
      if (external && stream.getVideoTracks()[0].getSettings().deviceId !== external.deviceId) {
        stream.getTracks().forEach(track => track.stop());
        stream = await navigator.mediaDevices.getUserMedia({ video: {
          deviceId: { exact: external.deviceId }, width: { ideal: 640 }, height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 }
        }, audio: false });
      }
    }
    state.cameraStream = stream;
    window.FretVision?.stop();
    refs.cameraVideo.removeAttribute("src");
    refs.cameraVideo.srcObject = stream;
    await refs.cameraVideo.play();
    window.FretVision?.attachCamera(stream);
    refs.cameraStage.dataset.mode = "live";
    refs.cameraButton.textContent = "关闭摄像头";
    const settings = stream.getVideoTracks()[0].getSettings();
    populateCameraDevices(await navigator.mediaDevices.enumerateDevices(), settings.deviceId);
    refs.videoRate.textContent = `${settings.width || 1280}×${settings.height || 720} · ${Math.round(settings.frameRate || 0) || "?"}fps`;
    setChip(refs.cameraChip, "live");
    showLiveWaitingResult();
    keepOverlaySized();
    showToast("摄像头已连接，正在自动定位琴颈并检测手部。");
  } catch (error) {
    refs.cameraButton.textContent = "打开摄像头";
    setChip(refs.cameraChip, "error");
    refs.controlMessage.textContent = `摄像头启动失败：${error.name || "权限或设备不可用"}`;
    showToast("无法访问摄像头，请检查浏览器权限和设备占用情况。");
  }
}

function stopCamera() {
  window.FretVision?.stop();
  state.cameraStream?.getTracks().forEach((track) => track.stop());
  state.cameraStream = null;
  refs.cameraVideo.srcObject = null;
  refs.cameraStage.dataset.mode = state.demo ? "demo" : "idle";
  refs.cameraButton.textContent = "打开摄像头";
  refs.videoRate.textContent = "--";
  refs.frameStatusText.textContent = state.demo ? "SIMULATED FRAME" : "CAMERA OFFLINE";
  refs.cameraHint.textContent = "蓝线表示琴弦，竖线表示品格，圆点表示演示中的指尖落点。";
  setChip(refs.cameraChip, "idle");
  if (!state.demo) showLiveWaitingResult();
  drawFretOverlay(state.demo);
}

function populateAudioDevices(devices) {
  const inputs = devices.filter((device) => device.kind === "audioinput");
  refs.inputSelect.innerHTML = "";
  inputs.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `音频输入 ${index + 1}`;
    refs.inputSelect.appendChild(option);
  });
  refs.inputSelect.disabled = inputs.length === 0;
}

function stopAudio() {
  state.audioSession += 1;
  state.audioStarting = false;
  state.audioStream?.getTracks().forEach((track) => track.stop());
  state.audioContext?.close();
  state.audioStream = null;
  state.audioContext = null;
  state.analyser = null;
  state.audioSource = null;
  cancelAnimationFrame(state.animationId);
  refs.audioButton.innerHTML = '<span aria-hidden="true">◉</span> 浏览器音频';
  refs.sampleRate.textContent = "48 kHz target";
  setChip(refs.audioChip, "idle");
  if (!state.demo) showLiveWaitingResult();
  if (state.demo) renderDemoAudio(); else drawQuietSignals();
}

async function startAudio(deviceId) {
  if (state.audioStarting) { stopAudio(); return; }
  if (state.audioStream) {
    stopAudio();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    setChip(refs.audioChip, "error");
    return;
  }
  if (state.hardware) stopHardware();
  state.demo = false;
  cancelAnimationFrame(state.animationId);
  state.audioStarting = true;
  const session = ++state.audioSession;
  showLiveWaitingResult();
  refs.audioButton.textContent = "正在启动音频…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { deviceId: deviceId ? { exact: deviceId } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false
    });
    if (session !== state.audioSession) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 4096;
    analyser.smoothingTimeConstant = .72;
    const source = context.createMediaStreamSource(stream);
    source.connect(analyser);
    state.audioStream = stream;
    state.audioContext = context;
    state.analyser = analyser;
    state.audioSource = source;
    state.audioStarting = false;
    state.demo = false;
    refs.audioButton.innerHTML = '<span aria-hidden="true">■</span> 停止音频';
    refs.sampleRate.textContent = `${Math.round(context.sampleRate / 1000)} kHz actual`;
    setChip(refs.audioChip, "live");
    populateAudioDevices(await navigator.mediaDevices.enumerateDevices());
    if (session !== state.audioSession) return;
    showLiveWaitingResult();
    renderLiveAudio();
    showToast("浏览器音频已连接 · 单音检测");
  } catch (error) {
    if (session !== state.audioSession) return;
    stopAudio();
    setChip(refs.audioChip, "error");
    refs.controlMessage.textContent = `音频启动失败：${error.name || "权限或设备不可用"}`;
    showToast("无法访问音频输入，请检查浏览器权限和 Scarlett 是否被占用。");
  }
}

function frequencyToPitch(frequency) {
  if (!Number.isFinite(frequency) || frequency <= 0) return null;
  const midi = Math.round(69 + 12 * Math.log2(frequency / 440));
  const target = 440 * 2 ** ((midi - 69) / 12);
  const cents = 1200 * Math.log2(frequency / target);
  const pitchClass = ((midi % 12) + 12) % 12;
  return { name: noteNames[pitchClass], octave: Math.floor(midi / 12) - 1, target, cents };
}

function autoCorrelate(buffer, sampleRate) {
  const size = Math.min(buffer.length, 2048);
  let rms = 0;
  for (let i = 0; i < size; i += 1) rms += buffer[i] ** 2;
  rms = Math.sqrt(rms / size);
  if (rms < .008) return -1;
  const minOffset = Math.max(2, Math.floor(sampleRate / 1400));
  const maxOffset = Math.min(Math.floor(size / 2), Math.floor(sampleRate / 60));
  let bestOffset = -1;
  let bestCorrelation = 0;
  for (let offset = minOffset; offset <= maxOffset; offset += 1) {
    let correlation = 0;
    for (let i = 0; i < size - offset; i += 1) correlation += buffer[i] * buffer[i + offset];
    correlation /= size - offset;
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestOffset = offset;
    }
  }
  return bestOffset > 0 && bestCorrelation > .01 ? sampleRate / bestOffset : -1;
}

function drawGrid(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0d1211";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(146,157,150,.12)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= width; x += width / 8) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += height / 4) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawWaveform(data) {
  const ctx = refs.waveCanvas.getContext("2d");
  const w = refs.waveCanvas.width;
  const h = refs.waveCanvas.height;
  drawGrid(ctx, w, h);
  ctx.strokeStyle = "#dcff58";
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((value, index) => {
    const x = index / (data.length - 1) * w;
    const y = (0.5 + value * .41) * h;
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawSpectrum(data, peaks = [], maxBins = 512) {
  const ctx = refs.spectrumCanvas.getContext("2d");
  const w = refs.spectrumCanvas.width;
  const h = refs.spectrumCanvas.height;
  drawGrid(ctx, w, h);
  const gradient = ctx.createLinearGradient(0, h, 0, 0);
  gradient.addColorStop(0, "#66d6d1");
  gradient.addColorStop(.72, "#dcff58");
  gradient.addColorStop(1, "#ff7152");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.moveTo(0, h);
  const count = Math.min(data.length, maxBins);
  for (let i = 0; i < count; i += 1) {
    const x = i / (count - 1) * w;
    const y = h - (data[i] / 255) ** 1.6 * h * .9;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fill();
  peaks.forEach((peak) => {
    if (peak.bin < 0 || peak.bin >= count) return;
    const x = peak.bin / (count - 1) * w;
    ctx.strokeStyle = "#b7a5ff";
    ctx.beginPath();
    ctx.moveTo(x, 12);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillStyle = "#edf1e9";
    ctx.font = "13px Consolas";
    const labelWidth = ctx.measureText(peak.label).width;
    ctx.fillText(peak.label, Math.max(0, Math.min(x + 5, w - labelWidth)), 24);
  });
}

function drawHardwareSpectrum(frame) {
  const binHz = frame.sampleRate / frame.fftSize;
  const count = Math.min(frame.power.length, Math.floor(4000 / binHz) + 1);
  const power = frame.power.slice(0, count);
  const maximum = Math.max(...power);
  const display = power.map((value) => maximum > 0 ? Math.log1p(99 * value / maximum) / Math.log(100) * 255 : 0);
  const peaks = frame.notes.map((note) => ({ bin: Math.round(note.frequency / binHz), label: note.name }));
  drawSpectrum(display, peaks, count);
}

function demoBuffers() {
  const wave = new Float32Array(2048);
  const spectrum = new Uint8Array(1024);
  const frequencies = [196, 246.94, 293.66];
  for (let i = 0; i < wave.length; i += 1) {
    const t = i / 48000;
    wave[i] = frequencies.reduce((sum, frequency, index) => sum + (.38 - index * .05) * Math.sin(2 * Math.PI * frequency * t + state.demoPhase), 0) / 1.35;
  }
  state.demoPhase += .025;
  spectrum.fill(7);
  const peaks = frequencies.map((frequency, index) => {
    const bin = Math.round(frequency / 48000 * 4096);
    spectrum[bin] = 235 - index * 24;
    if (bin * 2 < spectrum.length) spectrum[bin * 2] = 110 - index * 12;
    return { bin, label: ["G3", "B3", "D4"][index] };
  });
  return { wave, spectrum, peaks };
}

function renderDemoAudio() {
  cancelAnimationFrame(state.animationId);
  if (!state.demo || state.analyser) return;
  const buffers = demoBuffers();
  drawWaveform(buffers.wave);
  drawSpectrum(buffers.spectrum, buffers.peaks);
  state.animationId = requestAnimationFrame(renderDemoAudio);
}

function renderLiveAudio() {
  if (!state.analyser) return;
  const timeData = new Float32Array(state.analyser.fftSize);
  const freqData = new Uint8Array(state.analyser.frequencyBinCount);
  state.analyser.getFloatTimeDomainData(timeData);
  state.analyser.getByteFrequencyData(freqData);
  drawWaveform(timeData);
  drawSpectrum(freqData);
  if (performance.now() - state.lastPitchAt > 120) {
    const frequency = autoCorrelate(timeData, state.audioContext.sampleRate);
    const pitch = frequencyToPitch(frequency);
    if (pitch) {
      refs.noteResult.textContent = `${pitch.name}${pitch.octave} · ${frequency.toFixed(2)} Hz`;
      refs.audioEvidence.textContent = `${pitch.name}${pitch.octave}（浏览器单音检测）`;
    } else {
      refs.noteResult.textContent = "--";
      refs.audioEvidence.textContent = "尚无稳定单音";
    }
    state.lastPitchAt = performance.now();
  }
  state.animationId = requestAnimationFrame(renderLiveAudio);
}

function drawQuietSignals() {
  drawWaveform(new Float32Array(256));
  drawSpectrum(new Uint8Array(256));
}

refs.cameraButton.addEventListener("click", () => startCamera(refs.cameraSelect.disabled ? undefined : refs.cameraSelect.value));
refs.cameraSelect.addEventListener("change", () => {
  if (state.cameraStream) {
    stopCamera();
    startCamera(refs.cameraSelect.value);
  }
});
refs.audioButton.addEventListener("click", () => startAudio(refs.inputSelect.disabled ? undefined : refs.inputSelect.value));
refs.hardwareButton.addEventListener("click", startHardware);
refs.inputSelect.addEventListener("change", () => {
  if (state.audioStream) {
    stopAudio();
    startAudio(refs.inputSelect.value);
  }
});
refs.demoButton.addEventListener("click", () => {
  if (state.cameraStream) stopCamera();
  window.FretVision?.stop();
  if (state.hardware) stopHardware();
  state.demo = false;
  stopAudio();
  showDemoResult();
  renderDemoAudio();
  showToast("已运行完整视听演示：模拟 G Major 指法、G/B/D 音频峰值和融合结论。");
});

window.addEventListener("beforeunload", () => {
  state.hardware = false;
  state.hardwareSession += 1;
  state.audioSession += 1;
  window.clearTimeout(state.hardwareTimer);
  window.clearInterval(state.hardwareWatchdog);
  state.hardwareRequest?.abort();
  state.cameraStream?.getTracks().forEach((track) => track.stop());
  state.audioStream?.getTracks().forEach((track) => track.stop());
});

window.FretAudioSnapshot = () => state.hardwareResult ? {
  ...state.hardwareResult, receivedAt: state.hardwareUpdatedAt, stale: !state.hardware
} : null;
window.FretStopCamera = () => { if (state.cameraStream) stopCamera(); };
window.FretPrepareVision = () => {
  if (state.demo) {
    state.demo = false;
    cancelAnimationFrame(state.animationId);
    showIdleResult();
    drawQuietSignals();
  }
};

showIdleResult();
keepOverlaySized();
