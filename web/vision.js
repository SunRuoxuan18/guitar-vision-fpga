/* Local vision prototype: explicit sources, calibrated geometry, no pressure claims. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const stage = $("cameraStage"), video = $("cameraVideo"), overlay = $("fretOverlay");
  const status = $("visionStatus"), modelButton = $("handModelButton");
  const F = globalThis.Fretboard;
  const scratch = document.createElement("canvas");
  const scratchCtx = scratch.getContext("2d", { willReadFrequently: true });
  const grayFrame = document.createElement("canvas"), edgeFrame = document.createElement("canvas");
  const grayPreview = $("grayPreview"), edgePreview = $("edgePreview");
  const syntheticFrame = document.createElement("canvas");
  const fullRegion = { x: 0, y: 0, width: 1, height: 1 };
  let processedAt = -Infinity, processedRegion = null;
  const boardFrame = document.createElement("canvas");
  // Poll slightly faster than the 30 fps source so capture/transport jitter
  // cannot leave every other slot waiting on a duplicate sequence.
  const boardFramePeriod = 1000 / 40;
  const boardRecognitionPeriod = 1000 / 6;
  const boardRecognitionSlowPeriod = 1000 / 3;
  const plGrayFrame = document.createElement("canvas"), plEdgeFrame = document.createElement("canvas");
  const pcMatched = document.createElement("canvas"), pcSeed = document.createElement("canvas");
  const generationSeed = new Uint32Array(1);
  crypto.getRandomValues(generationSeed);
  const PC = { generation: generationSeed[0] & 0x7ffffffe, corners: null, seedPending: false,
    matchedAt: 0, sequence: -1 };
  let noteCalibration = null;
  function currentNoteCalibration(analysis = V.automatic) {
    if (noteCalibration && (!analysis || analysis.status !== "tracking" ||
        ["stream_id", "generation", "lattice_epoch"].some(key => analysis[key] !== noteCalibration[key]))) {
      noteCalibration = null;
      $("noteCalibrationStatus").textContent = "指板结构或视频已改变，请重新确认音符标定";
    }
    return noteCalibration;
  }
  function acceptFingertipNotes(analysis) {
    currentNoteCalibration(analysis);
    if (analysis?.status === "tracking" && !analysis.geometry) {
      try { analysis.geometry = F.createObservedGeometry(analysis); }
      catch (_) { /* Unsupported measured lines must not become a grid. */ }
    }
    const notes = analysis?.fingertip_notes;
    if (V.result?.fourTipProjection) V.result = null;
    if (notes?.status === "current" && notes.source_sequence === analysis.sequence &&
        notes.stream_id === analysis.stream_id && notes.generation === analysis.generation &&
        Array.isArray(notes.projection?.positions)) {
      V.result = { ...notes.projection, hands: [], source: "camera", receivedAt: performance.now(),
        frame_sequence: analysis.sequence, inference_location: "ps_linux", synchronized: false,
        fourTipProjection: true };
    }
  }
  const plFramePeriod = 33, plMaxAge = 1000, latticeMaxAgeMs = 1500;
  const PL = { timer: null, request: null, generation: 0, loopKey: null,
    frame: null, last: null, status: "未连接",
    planePixels: [null, null] };
  const littleEndian = new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 4;
  const tips = [8, 12, 16, 20];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
  const V = { source: null, session: 0, request: 0, calibration: null, clicks: null,
    result: null, handObservation: null, model: null, modelLoading: false, modelError: null, detect: false, lastTime: -1,
    lastInference: 0, lastFrame: 0, frameCount: 0, fpsStart: 0, url: null,
    stream: null, shape: null, modelGeneration: 0, boardTimer: null, boardRequest: null,
    boardSequence: -1, boardReceivedAt: 0, boardOrigin: null, identity: null, boardCapturedMs: null,
    fpsFrames: 0, displayFps: 0, inferenceCount: 0, fpsInferences: 0, inferenceFps: 0,
    processedToken: null, clickRegion: null, boardStream: null, boardError: null,
    boardStreamActive: false,
    tracking: null, trackingConfig: null, trackingGeneration: 0, automatic: null, autoRetryAt: 0,
    requestMs: 0, decodeMs: 0, selectingRoi: false, roiError: null, searchFeedback: null,
    lastPublish: -Infinity, videoLayout: null, lastOverlayAt: -Infinity, lastOverlayKey: null };
  const cornerLabels = ["6弦起点边界", "6弦终点边界", "1弦终点边界", "1弦起点边界"];
  const fingertipNames = ["index", "middle", "ring", "little"];
  const fingertipPreview = { stream: null, generation: null, state: "waiting", result: null,
    key: null, receivedAt: 0, age: null, ageIsSource: false, ageAt: 0, image: null,
    imageKey: null, imageStatus: null, revision: 0, maxAgeMs: 250, completions: [], lastResultSequence: -1 };

  function clearFingertipPreview(reason = "等待板卡推理", state = "waiting") {
    fingertipPreview.revision += 1;
    Object.assign(fingertipPreview, { stream: null, generation: null, state, result: null,
      key: null, receivedAt: 0, age: null, ageIsSource: false, ageAt: 0, image: null,
      imageKey: null, imageStatus: null, maxAgeMs: 250, completions: [], lastResultSequence: -1 });
    const panel = $("fingertipPreview"), canvas = $("fingertipPreviewCanvas");
    if (!panel || !canvas) return;
    delete canvas.dataset.resultKey;
    panel.dataset.state = state;
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    $("fingertipPreviewFigure").hidden = true;
    $("fingertipPreviewStatus").textContent = reason;
    for (const id of ["Age", "Graph", "Inference", "Rate", "Tracking"]) $("fingertipPreview" + id).textContent = "--";
    $("fingertipPreviewPoints").textContent = "1 食指 · 2 中指 · 3 无名指 · 4 小指";
    $("fingertipPreviewNote").textContent = "等待同一视频会话的 PL 结果；原始峰值不是正确率。";
  }

  function updateFingertipPreview(snapshot, stream, generation, sequence, requestStarted) {
    if (!$("fingertipPreview")) return;
    const statuses = { idle: "waiting", pending: snapshot?.running ? "running" : "waiting", ok: "ready",
      waiting: "waiting", running: "running", ready: "ready", stale: "stale", sequence_mismatch: "stale",
      source_mismatch: "waiting", stopping: "stopping", stopped: "stopped", fault: "fault" };
    if (!snapshot || !Object.hasOwn(statuses, snapshot.status)) {
      clearFingertipPreview("四指尖 PL 推理未启用"); return;
    }
    if (fingertipPreview.stream !== stream || fingertipPreview.generation !== generation) {
      clearFingertipPreview();
      fingertipPreview.stream = stream; fingertipPreview.generation = generation;
    }
    const source = snapshot.current_source;
    if (!source || source.stream_id !== stream || source.generation !== generation || source.sequence !== sequence) {
      clearFingertipPreview("结果会话不匹配，已清空", "stale"); return;
    }
    const now = performance.now(), result = snapshot.last_result;
    fingertipPreview.state = statuses[snapshot.status];
    fingertipPreview.receivedAt = now;
    fingertipPreview.maxAgeMs = Number.isFinite(snapshot.max_age_ms) && snapshot.max_age_ms > 0 ? snapshot.max_age_ms : 250;
    if (snapshot.status === "fault") { clearFingertipPreview("PL 推理故障，历史结果已清空", "fault"); return; }
    if (!result) {
      fingertipPreview.result = null; fingertipPreview.image = null;
      fingertipPreview.imageKey = null; fingertipPreview.imageStatus = null;
      fingertipPreview.key = null; fingertipPreview.revision += 1;
      renderFingertipPreview(now); return;
    }
    const roi = result.roi, sourceSequence = result.sequence;
    const valid = result.stream_id === stream && result.generation === generation &&
      Number.isSafeInteger(sourceSequence) && sourceSequence >= 0 && sourceSequence <= sequence &&
      sourceSequence >= fingertipPreview.lastResultSequence && Array.isArray(roi) && roi.length === 4 &&
      roi.every(Number.isFinite) && roi[0] >= 0 && roi[1] >= 0 && roi[2] > 0 && roi[3] > 0 &&
      fingertipNames.every(name => {
        const point = result.points?.[name];
        return point && [point.x, point.y, point.raw_score].every(Number.isFinite) &&
          point.x >= roi[0] && point.x < roi[0] + roi[2] && point.y >= roi[1] && point.y < roi[1] + roi[3];
      });
    if (!valid) { clearFingertipPreview("无效或跨会话结果，已清空", "stale"); return; }
    const key = JSON.stringify([stream, generation, sourceSequence]);
    const newResult = fingertipPreview.key !== key;
    if (newResult) {
      fingertipPreview.completions.push(now);
      fingertipPreview.completions = fingertipPreview.completions.slice(-8);
      fingertipPreview.lastResultSequence = sourceSequence;
      fingertipPreview.image = null; fingertipPreview.imageKey = null; fingertipPreview.imageStatus = "missing";
    }
    const sourceAge = result.source_age_ms ?? snapshot.source_age_ms;
    const age = sourceAge ?? result.age_ms ?? snapshot.age_ms;
    const previousAge = !newResult && fingertipPreview.age != null ?
      fingertipPreview.age + now - fingertipPreview.ageAt : 0;
    fingertipPreview.age = Number.isFinite(age) && age >= 0 ?
      Math.max(previousAge, age + Math.max(0, now - requestStarted)) : null;
    fingertipPreview.ageIsSource = sourceAge != null;
    fingertipPreview.ageAt = now;
    fingertipPreview.result = result; fingertipPreview.key = key;
    if (result.stale || sourceSequence < sequence) fingertipPreview.state = "stale";
    if (newResult) {
      const revision = ++fingertipPreview.revision;
      const preview = result.preview;
      const encoded = preview?.jpeg_base64;
      if (preview?.mime_type === "image/jpeg" && preview.width === 208 && preview.height === 208 &&
          typeof encoded === "string" && encoded.length <= 200000 && /^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) {
        const image = new Image();
        fingertipPreview.imageStatus = "decoding";
        image.onload = () => {
          if (revision !== fingertipPreview.revision || key !== fingertipPreview.key) return;
          if (image.naturalWidth !== 208 || image.naturalHeight !== 208) {
            fingertipPreview.imageStatus = "fault"; renderFingertipPreview(performance.now()); return;
          }
          fingertipPreview.image = image; fingertipPreview.imageKey = key; fingertipPreview.imageStatus = "ready";
          renderFingertipPreview(performance.now());
        };
        image.onerror = () => {
          if (revision !== fingertipPreview.revision || key !== fingertipPreview.key) return;
          fingertipPreview.imageStatus = "fault"; renderFingertipPreview(performance.now());
        };
        image.src = "data:image/jpeg;base64," + encoded;
      }
    }
    renderFingertipPreview(now);
  }

  function renderFingertipPreview(now) {
    const panel = $("fingertipPreview"), state = fingertipPreview;
    if (!panel || state.stream == null) return;
    const activeVisionSource = V.source === "camera" && $("pcPlEnabled").checked ||
      V.source === "board" && PL.frame?.source === "board_camera" && PL.frame.session === V.session;
    if (!activeVisionSource || state.generation !== PC.generation ||
        now - state.receivedAt > plMaxAge) {
      clearFingertipPreview("视频中断或会话已变，历史结果已清空", "stale"); return;
    }
    const age = state.age == null ? null : state.age + now - state.ageAt;
    const displayState = state.result && age != null && age > state.maxAgeMs ? "stale" : state.state;
    panel.dataset.state = displayState;
    $("fingertipPreviewStatus").textContent = ({ waiting: "等待同会话手部 ROI", running: "PL 推理中",
      ready: "历史推理结果", stale: "已过期 · 仅历史参考", stopping: "PL 推理正在停止", stopped: "PL 推理已停止" })[displayState] +
      (displayState === "running" && state.result ? " · 保留历史结果" : "");
    const result = state.result;
    $("fingertipPreviewAge").textContent = result && age != null ?
      `${state.ageIsSource ? "源帧" : "板端 ROI"} ≤${Math.ceil(age)} ms` : "未提供";
    const graphMs = result?.timings_ms?.graph, inferenceMs = result?.inference_ms ?? result?.timings_ms?.total;
    $("fingertipPreviewGraph").textContent = Number.isFinite(graphMs) && graphMs >= 0 ? `${graphMs.toFixed(1)} ms` : "未提供";
    $("fingertipPreviewInference").textContent = Number.isFinite(inferenceMs) && inferenceMs >= 0 ? `${inferenceMs.toFixed(1)} ms` : "未提供";
    const times = state.completions, elapsed = times.length > 1 ? (now - times[0]) / 1000 : 0;
    $("fingertipPreviewRate").textContent = elapsed > 0 ? `${((times.length - 1) / elapsed).toFixed(2)} Hz · 实收` : "待第 2 次结果";
    $("fingertipPreviewTracking").textContent = Number.isFinite(PL.frame?.fps) ? `${PL.frame.fps.toFixed(1)} Hz` : "--";
    const imageNote = ({ missing: " · 源图未提供", decoding: " · 源图解码中", fault: " · 源图不可用" })[state.imageStatus] || "";
    $("fingertipPreviewNote").textContent = result ?
      `源帧 #${result.sequence} · ${state.ageIsSource ? "源帧计时" : "板端提交起算，非完整采集延迟"} + 本次往返上界 · 原始峰值非置信度${imageNote}` :
      "等待同一视频会话的 PL 结果；原始峰值不是正确率。";
    $("fingertipPreviewPoints").textContent = result ? fingertipNames.map((name, i) =>
      `${["食", "中", "无名", "小"][i]} ${result.points[name].raw_score.toFixed(3)}`).join(" · ") :
      "1 食指 · 2 中指 · 3 无名指 · 4 小指";
    const canvas = $("fingertipPreviewCanvas"), context = canvas.getContext("2d");
    const available = result && state.image && state.imageKey === state.key;
    $("fingertipPreviewFigure").hidden = !available;
    if (!available) { context.clearRect(0, 0, canvas.width, canvas.height); delete canvas.dataset.resultKey; return; }
    if (canvas.dataset.resultKey === state.key) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(state.image, 0, 0, 208, 208);
    for (const [i, name] of fingertipNames.entries()) {
      const point = result.points[name], [x, y, width, height] = result.roi;
      const px = (point.x - x) / width * 208, py = (point.y - y) / height * 208;
      context.beginPath(); context.arc(px, py, 4, 0, Math.PI * 2);
      context.fillStyle = ["#ff4b4b", "#ffdb58", "#51e5ba", "#7abfff"][i]; context.fill();
      context.strokeStyle = "#111516"; context.lineWidth = 1.5; context.stroke();
      context.font = "bold 12px sans-serif"; context.lineWidth = 3;
      const labelX = Math.max(2, Math.min(194, px + 6)), labelY = Math.max(13, Math.min(203, py - 5));
      context.strokeText(String(i + 1), labelX, labelY); context.fillStyle = "#fff";
      context.fillText(String(i + 1), labelX, labelY);
    }
    canvas.dataset.resultKey = state.key;
  }

  function message(text) { status.textContent = text; }
  function autoEnabled() { return (V.source === "board" || V.source === "camera" && $("pcPlEnabled").checked) && $("autoFretboard").checked; }
  function pcTracking() { return V.source === "camera" && autoEnabled(); }
  function pcProcessedDisplay() { return V.source === "camera" && $("pcPlEnabled").checked; }
  function colorLiveView() { return !V.clicks && $("visionView").value === "color"; }
  function boardProcessedDisplay() { return V.source === "board" && V.boardOrigin === "camera" &&
    PC.sequence >= 0 && pcMatched.width === 640 && PL.frame?.source === "board_camera" && validPl(); }
  function pcFresh() { return PC.sequence >= 0 && performance.now() - PC.matchedAt <= plMaxAge && pcMatched.width === 640; }
  let displayTrack = null;
  function clearAuto() { V.automatic = null; if (autoEnabled()) V.calibration = null; }
  function attachAutomaticGeometry(result) {
    if (!result || result.status !== "tracking") return result;
    try { return { ...result, geometry: F.createObservedGeometry(result) }; }
    catch (_) { return { ...result, status: "searching", reason: "invalid_geometry", geometry: null }; }
  }
  function applyAuto(result, sequence, stream) {
    clearAuto(); clearEvidence(); V.lastTime = -1;
    if (!result || result.sequence !== sequence || result.stream_id !== stream) return;
    const projectedAge = result.lattice_age_ms;
    const staleLattice = result.status === "tracking" && Number.isFinite(projectedAge) &&
      projectedAge > Math.min(latticeMaxAgeMs,
        Number.isFinite(result.lattice_ttl_ms) ? result.lattice_ttl_ms : latticeMaxAgeMs);
    const accepted = staleLattice ? { ...result, status: "searching",
      reason: "stale_lattice_geometry", strings: [], frets: [], geometry: null,
      string_count: 0, fret_count: 0, estimated_strings: null } : result;
    if (staleLattice) V.searchFeedback = {
      reason: "stale_lattice_geometry", elapsedMs: projectedAge, receivedAt: performance.now()
    };
    const handDetection = accepted.hand_detection;
    if (V.source === "camera" || V.source === "board") {
      V.handObservation = null;
      if (["ps_linux", "ps_pl"].includes(handDetection?.inference_location) && handDetection.stream_id === stream &&
          handDetection.sequence === sequence && handDetection.generation === PC.generation && Array.isArray(handDetection.hands)) {
        const hands = handDetection.hands.filter(h => h.tracked_sequence === sequence &&
          h.landmarks?.length === 21 && h.landmarks.every(p => Number.isFinite(p.x) && Number.isFinite(p.y)));
        V.handObservation = { hands, source: "ps_linux", receivedAt: performance.now(),
          token: sequence, frame: pcMatched, detection: handDetection };
        const projected = accepted.visual_positions;
        if (projected?.processing_location === "ps_linux" && Array.isArray(projected.positions)) {
          V.result = { ...projected, hands, source: "camera", receivedAt: PC.matchedAt,
            frame_sequence: sequence, inference_location: "ps_linux", synchronized: false };
        }
      }
    }
    if (accepted.roi_status === "tracking" && accepted.roi_corners?.length === 4) {
      displayTrack = { corners: accepted.roi_corners, session: V.session, at: performance.now() };
    } else if (accepted.roi_status === "lost") displayTrack = null;
    if (accepted.reason === "detector_too_slow") V.autoRetryAt = performance.now() + 30000;
    V.automatic = attachAutomaticGeometry(accepted);
  }
  function followsNeck() { return V.source === "board" && V.boardOrigin === "camera" && $("followNeck").checked; }
  function resetTracking() {
    V.trackingGeneration += 1;
    const identity = V.tracking?.tracker_id;
    V.tracking = null; V.trackingConfig = null;
    if (identity) fetch("/api/vision/tracking/stop", { method: "POST", keepalive: true,
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tracker_id: identity }) }).catch(() => {});
  }
  function loseTracking(reason) {
    if (!V.tracking) return;
    V.tracking = { ...V.tracking, status: "lost", corners: null, reason };
    V.calibration = null; clearEvidence();
    message("琴颈跟踪丢失，请保持指板可见并重新标定");
  }
  async function startTracking(calibration) {
    resetTracking();
    const generation = V.trackingGeneration, session = V.session;
    V.trackingConfig = calibration;
    V.tracking = { status: "pending" };
    V.calibration = null; clearEvidence();
    message("正在锁定琴颈");
    try {
      const response = await fetch("/api/vision/tracking", { method: "POST",
        headers: { "Content-Type": "application/json" }, signal: AbortSignal.timeout(5000),
        body: JSON.stringify({ corners: calibration.corners, sequence: V.boardSequence, stream_id: V.boardStream }) });
      const result = await response.json();
      if (generation !== V.trackingGeneration || session !== V.session) {
        if (result.tracker_id) fetch("/api/vision/tracking/stop", { method: "POST",
          headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tracker_id: result.tracker_id }) }).catch(() => {});
        return;
      }
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      if (!result.tracker_id || result.status !== "tracking") throw new Error("指板纹理不足或发生遮挡");
      // Geometry is applied only when it arrives with the corresponding JPEG.
      V.tracking = { ...result, status: "pending", corners: null };
    } catch (error) {
      if (generation !== V.trackingGeneration || session !== V.session) return;
      loseTracking(error.message);
      message(`无法锁定琴颈，请重新标定：${error.message}`);
    }
  }
  function applyTracking(tracking, sequence, stream) {
    if (!V.tracking?.tracker_id || V.tracking.status === "lost") return;
    if (!tracking || tracking.tracker_id !== V.tracking.tracker_id ||
      tracking.sequence !== sequence || tracking.stream_id !== stream || tracking.status !== "tracking") {
      loseTracking(tracking?.reason || "frame_mismatch"); return;
    }
    try {
      const config = V.trackingConfig;
      const wasPending = V.tracking.status === "pending";
      V.calibration = F.createCalibration(tracking.corners, config.firstFret, config.lastFret, config.tuning);
      V.tracking = tracking;
      if (wasPending) message("琴颈跟随中");
      // Do not keep fingertip positions projected through a previous frame's geometry.
      clearEvidence(); V.lastTime = -1;
    } catch (_) { loseTracking("invalid_geometry"); }
  }
  function inputSize() {
    // The transport uses a fixed 4:3 frame. Freshness gates evidence, not layout.
    if (V.source === "camera" && (V.selectingRoi && V.clicks || pcProcessedDisplay())) return { width: 640, height: 480 };
    return V.source === "board" ? { width: 640, height: 480 } : V.source === "synthetic" ? { width: 960, height: 540 } :
    V.source === "board" ? V.shape : { width: video.videoWidth, height: video.videoHeight }; }
  function inputFrame() { return V.source === "synthetic" ? syntheticFrame : V.source === "board" ? boardFrame :
    V.source === "camera" && V.selectingRoi && V.clicks ? pcSeed :
    pcProcessedDisplay() && PC.sequence >= 0 ? pcMatched : video; }
  function displayFrame() {
    // C922 color preview is the live source frame. Recognition runs
    // asynchronously and must not lower the displayed camera cadence.
    if (V.source === "board" && !V.clicks && $("visionView").value === "color")
      return boardFrame;
    if (V.source === "camera" && colorLiveView()) return video;
    // Present only the camera capture whose board response has been accepted.
    if ((pcProcessedDisplay() || boardProcessedDisplay()) && !V.clicks) return pcMatched;
    return V.source === "camera" && !V.clicks ? video : inputFrame();
  }
  function placeLiveVideo(box, enabled) {
    const layout = enabled ? [box.x / overlay.width * 100, box.y / overlay.height * 100,
      box.width / overlay.width * 100, box.height / overlay.height * 100].map(n => `${n}%`) : null;
    const key = layout?.join(",") || "default";
    if (V.videoLayout === key) return;
    V.videoLayout = key;
    if (layout) {
      [video.style.left, video.style.top, video.style.width, video.style.height] = layout;
      video.style.objectFit = "fill";
    } else {
      for (const name of ["left", "top", "width", "height", "object-fit"]) video.style.removeProperty(name);
    }
  }
  function inputToken() {
    if (V.source === "board") return V.boardSequence;
    const quality = video.getVideoPlaybackQuality?.();
    return quality && quality.totalVideoFrames > 0 ? quality.totalVideoFrames : video.currentTime;
  }
  function resultSource() { return V.source === "board" ? `board_${V.boardOrigin}` : V.source; }
  function displayCorners() {
    if (V.clicks || (V.source === "board" && (V.boardError || performance.now() - V.boardReceivedAt > 500))) return null;
    if (V.source === "board" && V.boardOrigin === "camera" && !boardProcessedDisplay()) return null;
    if (pcTracking() && !pcFresh()) return displayTrack?.session === V.session &&
      performance.now()-displayTrack.at <= 1500 ? displayTrack.corners : null;
    return autoEnabled() ? V.automatic?.geometry?.corners ||
      (V.automatic?.roi_status === "tracking" ? V.automatic.roi_corners : null) ||
      (displayTrack?.session === V.session && performance.now()-displayTrack.at <= 1500 ? displayTrack.corners : null) : V.calibration?.corners || null;
  }
  function handRegionCorners() {
    if (V.source === "board") {
      const region = V.automatic?.hand_region;
      if (!boardProcessedDisplay() || performance.now() - (V.boardProcessedAt || 0) > 1800 ||
          region?.status !== "tracking" || region.stream_id !== PC.stream ||
          region.generation !== PC.generation || region.sequence !== PC.sequence) return null;
      const corners = region.regions?.[0]?.corners;
      return corners?.length === 4 && corners.every(p => Number.isFinite(p.x) && Number.isFinite(p.y) &&
        p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1) ? corners : null;
    }
    if (V.source === "camera" && pcTracking()) {
      const region = V.automatic?.hand_region;
      if (region?.mode === "neck_local_lightweight" || region?.regions) {
        if (!pcFresh() || region.sequence !== PC.sequence || region.generation !== PC.generation ||
            region.stream_id !== V.automatic.stream_id || region.status !== "tracking") return null;
        const corners = region.regions?.[0]?.corners;
        return corners?.length === 4 && corners.every(p => Number.isFinite(p.x) && Number.isFinite(p.y) &&
          p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1) ? corners : null;
      }
    }
    const neck = displayCorners();
    const hands = visibleHands();
    if (!hands.length) return null;
    const candidates = [];
    for (const hand of hands) {
      const points = [5, 8, 9, 12, 13, 16, 17, 20].map(i => hand.landmarks[i])
        .filter(p => p && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
      if (points.length < 4) continue;
      const center = { x: points.reduce((s,p)=>s+p.x,0)/points.length,
        y: points.reduce((s,p)=>s+p.y,0)/points.length };
      // The delayed neck estimate selects the nearer hand only. It never
      // constrains the current hand box to the old neck position.
      const distance = neck ? Math.min(...neck.map(p=>Math.hypot(p.x-center.x,p.y-center.y))) : 0;
      candidates.push({ points, distance });
    }
    if (!candidates.length) return null;
    const { points } = candidates.sort((a,b)=>a.distance-b.distance)[0];
    const left = Math.max(0,Math.min(...points.map(p=>p.x))-.025);
    const right = Math.min(1,Math.max(...points.map(p=>p.x))+.025);
    const top = Math.max(0,Math.min(...points.map(p=>p.y))-.025);
    const bottom = Math.min(1,Math.max(...points.map(p=>p.y))+.025);
    return [{x:left,y:top},{x:right,y:top},{x:right,y:bottom},{x:left,y:bottom}];
  }
  function displayRegion() {
    if (V.source === "camera" && !V.clicks) return fullRegion;
    // Detection geometry must not make the board preview zoom in and out.
    if (V.source === "board") return fullRegion;
    if (!$("focusFretboard").checked) return fullRegion;
    if (V.source === "board" && (V.boardError || performance.now() - V.boardReceivedAt > 500)) return fullRegion;
    if (V.clicks) return V.clickRegion || fullRegion;
    const neck = displayCorners(), hand = handRegionCorners();
    const corners = neck && hand ? [...neck, ...hand] : hand || neck;
    if (!corners) return fullRegion;
    const xs = corners.map(p => p.x), ys = corners.map(p => p.y);
    const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
    const padX = Math.max(.015, (right - left) * .08), padY = Math.max(.015, (bottom - top) * .15);
    const x = Math.max(0, left - padX), y = Math.max(0, top - padY);
    return { x, y, width: Math.min(1, right + padX) - x, height: Math.min(1, bottom + padY) - y };
  }
  function projectionPoint(p, box) {
    const region = box.region || fullRegion;
    return { x: box.x + (p.x - region.x) / region.width * box.width,
      y: box.y + (p.y - region.y) / region.height * box.height };
  }
  function frameBox() {
    const region = displayRegion(), size = inputSize();
    const ratio = ((size.width || 16) * region.width) / ((size.height || 9) * region.height);
    const width = Math.min(overlay.width, overlay.height * ratio), height = width / ratio;
    return { x: (overlay.width - width) / 2, y: (overlay.height - height) / 2, width, height, region };
  }
  function clearPreviews() {
    V.processedToken = null; processedRegion = null;
    for (const canvas of [grayPreview, edgePreview]) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }
  function clearPl(reason) {
    if (V.source !== "board") clearFingertipPreview(reason === "未连接" ? "等待 PL 视频处理" : `PL 视频处理：${reason}`,
      reason === "未连接" ? "waiting" : "stale");
    if (PL.frame) for (const canvas of [plGrayFrame, plEdgeFrame])
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    PL.frame = null; PL.status = reason;
    if (V.handObservation?.source === "ps_linux") V.handObservation = null;
    if (pcTracking() || V.source === "board" && V.boardOrigin === "camera") {
      V.automatic = null; displayTrack = null; clearEvidence();
    }
  }
  function stopPl() {
    PL.generation += 1;
    clearTimeout(PL.timer); PL.timer = null;
    PL.request?.abort(); PL.request = null;
    PL.loopKey = null;
    PL.last = null; clearPl("未连接");
    labelPreviews(false, false);
  }
  function cameraFresh(now = performance.now()) {
    return V.source === "board" && !V.boardError && V.boardSequence >= 0 &&
      now - V.boardReceivedAt <= 500 && V.boardStream && Number.isFinite(V.boardCapturedMs);
  }
  function plAge(frame, now = performance.now()) {
    if (frame.source === "pc_camera" || frame.source === "board_camera") return now - frame.started;
    // Both capture timestamps use the board's monotonic clock; the PC clock is unrelated.
    const cameraAge = now - V.boardReceivedAt + V.requestMs + V.decodeMs;
    const relativeAge = V.boardCapturedMs - frame.capturedMs + cameraAge;
    if (relativeAge < -50) return Infinity;
    return Math.max(0, relativeAge, now - frame.started + frame.ageMs);
  }
  function validPl() {
    const frame = PL.frame;
    if (!frame) return false;
    if (frame.source === "board_camera") return V.source === "board" && frame.session === V.session &&
      frame.sequence === PC.sequence && performance.now() < frame.expiresAt &&
      performance.now() - (V.boardProcessedAt || 0) <= 1800;
    if (frame.source === "pc_camera") {
      if (V.source !== "camera" || frame.session !== V.session || video.readyState < 2 ||
          performance.now() >= frame.expiresAt || plAge(frame) > plMaxAge) {
        clearPl("PL 暂无新结果 · 等待板卡处理帧"); return false;
      }
      return true;
    }
    if (!cameraFresh() || frame.stream !== V.boardStream || frame.source !== V.boardOrigin ||
        performance.now() >= frame.expiresAt || plAge(frame) > plMaxAge) {
      clearPl("板卡摄像头暂无有效新图像；电脑 USB 摄像头请使用打开摄像头"); return false;
    }
    return true;
  }
  function labelPreviews(hardware, available = true) {
    if (V.source === "camera" && !hardware) available = false;
    $("grayPreviewBackend").textContent = $("edgePreviewBackend").textContent =
      available ? hardware ? "PL" : "网页预览" : "不可用";
    for (const [value, name] of [["gray", "灰度"], ["edge", "边缘"]]) {
      const option = $("visionView").querySelector(`option[value="${value}"]`);
      const text = `${name} · ${available ? hardware && V.source === "camera" ? "PL" : "网页预览" : "不可用"}`;
      if (option.textContent !== text) option.textContent = text;
    }
    stage.dataset.processingBackend = available ? V.source === "camera" && hardware ? "pl" : $("visionView").value === "color" ? V.source : "browser" : "unavailable";
    $("plVisionStatus").textContent = hardware ?
      `PL · 板端 ${PL.frame.processingMs.toFixed(1)} ms` +
      (Number.isFinite(PL.frame.roundtripMs) ? ` · 往返 ${PL.frame.roundtripMs.toFixed(1)} ms` : "") +
      ` · 处理 ${PL.frame.fps.toFixed(1)} 次/秒 · ${Math.round(plAge(PL.frame))} ms 帧龄` :
      `PL ${PL.status}${available ? " · 网页预览" : ""}`;
  }
  function updatePlaneCanvas(canvas, plane, cacheIndex) {
    if (canvas.width !== 320 || canvas.height !== 240) {
      canvas.width = 320; canvas.height = 240;
      PL.planePixels[cacheIndex] = null;
    }
    const context = canvas.getContext("2d");
    let image = PL.planePixels[cacheIndex];
    if (!image) image = PL.planePixels[cacheIndex] = context.createImageData(320, 240);
    if (littleEndian) {
      const rgba = new Uint32Array(image.data.buffer);
      for (let p = 0; p < plane.length; p++) {
        const value = plane[p];
        rgba[p] = 0xff000000 | value * 0x00010101;
      }
    } else {
      for (let p = 0, i = 0; p < plane.length; p++, i += 4) {
        const value = plane[p];
        image.data[i] = image.data[i + 1] = image.data[i + 2] = value;
        image.data[i + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }
  function startPcPl(session) {
    const generation = PL.generation;
    const boardSource = V.source === "board";
    const boardCameraStream = boardSource ? V.boardStream : null;
    // A copied tab can inherit the same random generation. Give every
    // recognition loop its own owner so the board can reject the stale loop.
    const owner = crypto.randomUUID();
    const loopKey = boardSource
      ? `board:${boardCameraStream}:${PC.generation}:${generation}:${owner}`
      : `camera:${PC.generation}:${generation}:${owner}`;
    if (PL.loopKey === loopKey) return;
    PL.loopKey = loopKey;
    const current = () => generation === PL.generation && session === V.session &&
      (boardSource ? V.source === "board" && V.boardStream === boardCameraStream :
        V.source === "camera" && $("pcPlEnabled").checked);
    const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 480;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const stream = boardSource ? `board-${boardCameraStream}` : crypto.randomUUID();
    let sequence = 0, lastToken = null, lastDone = null, takeoverAttempted = false;
    let consecutiveRequestFailures = 0;
    const transportMode = new URLSearchParams(location.search).get("transport") || "auto";
    let fastLink = false;
    fetch("/api/vision/pc/status").then(r => r.ok ? r.json() : null)
      .then(status => { fastLink = status?.link_speed_mbps >= 1000; }).catch(() => {});
    const poll = async () => {
      if (!current()) return;
      const started = performance.now(), controller = new AbortController();
      PL.request = controller;
      // Initial whole-frame acquisition may be slower than steady LK updates.
      // Old responses cannot supply geometry for recognition.
      // Acquisition is asynchronous on the board. A slow/missing neck is not
      // a camera failure, so give a board request time to return its searching
      // state instead of repeatedly aborting work that is still running there.
      const requestLimit = boardSource ? 30000 :
        autoEnabled() && V.automatic?.roi_status !== "tracking" ? 7000 : 2500;
      const timeout = setTimeout(() => controller.abort(), requestLimit);
      let bitmap;
      try {
        if (document.hidden) return;
        if (V.clicks) return;
        const videoToken = boardSource ? V.boardSequence :
          video.getVideoPlaybackQuality?.().totalVideoFrames || video.currentTime;
        if (boardSource ? V.boardError || V.boardSequence < 0 || videoToken === lastToken :
          video.readyState < 2 || !PC.seedPending && videoToken === lastToken) return;
        lastToken = videoToken;
        const trackingGeneration = PC.generation, seed = PC.seedPending;
        context.drawImage(boardSource ? boardFrame : seed ? pcSeed : video, 0, 0, 640, 480);
        const rawTransport = transportMode === "raw" || transportMode === "auto" && fastLink;
        const jpegTransport = typeof canvas.toBlob === "function" &&
          transportMode !== "lossless" && !rawTransport;
        sequence += 1;
        const token = `${stream}-${sequence}`;
        const options = { stream, sequence, generation: trackingGeneration, owner,
          enabled: autoEnabled(), corners: PC.corners, preview_scale: 2, preview_encoding: "planes",
          fast_tracking: true,
          coordinates_only: autoEnabled() && !$("preprocessDetails").open && $("visionView").value === "color",
          ...(boardSource ? { board_camera: true } : {}) };
        const musicalCalibration = currentNoteCalibration();
        if (musicalCalibration?.stream_id === stream && musicalCalibration.generation === trackingGeneration)
          options.note_calibration = musicalCalibration;
        let payload = jpegTransport ? null : context.getImageData(0, 0, 640, 480).data;
        const quad = V.automatic?.status === "tracking" && V.automatic.roi_corners;
        // Experimental: reconstruction currently costs more than the bytes saved.
        if (new URLSearchParams(location.search).get("roiTransport") === "1" &&
            options.coordinates_only && !seed && Array.isArray(quad) && quad.length === 4) {
          const x = Math.max(0,Math.floor(Math.min(...quad.map(p=>p.x))*639)-64);
          const y = Math.max(0,Math.floor(Math.min(...quad.map(p=>p.y))*479)-48);
          const right = Math.min(640,Math.ceil(Math.max(...quad.map(p=>p.x))*639)+65);
          const bottom = Math.min(480,Math.ceil(Math.max(...quad.map(p=>p.y))*479)+49);
          options.transport_roi = {x,y,width:right-x,height:bottom-y};
          payload = context.getImageData(x,y,right-x,bottom-y).data;
        }
        // Encoding transports the camera frame; all recognition remains on PS/PL.
        if (jpegTransport) {
          payload = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", .92));
          if (!payload) throw new Error("相机帧编码失败");
        }
        if (!current() || controller.signal.aborted) return;
        const response = await fetch("/api/vision/pc/process", { method: "POST",
          headers: { "Content-Type": jpegTransport ? "image/jpeg" : "application/octet-stream", "X-Vision-Token": token,
            ...(rawTransport ? {"X-Vision-Transport":"identity"} : {}),
            "X-Vision-Options": JSON.stringify(options) },
          body: payload, signal: controller.signal });
        if (response.status === 429) {
          if (boardSource) PL.retryDelay = 120;
          return;
        }
        if (response.status === 409) {
          // Let a page that never acquired the board retry once with the
          // sequence-1 takeover handshake. A page that has already produced
          // results is the displaced owner and must yield to the new page.
          if (lastDone !== null || takeoverAttempted) {
            PL.generation += 1;
            clearPl("另一个摄像头页面已接管琴颈识别；本页已暂停传图");
            message("另一个摄像头页面已接管琴颈识别；本页已暂停传图");
            return;
          }
          takeoverAttempted = true;
          sequence = 0;
          lastToken = null;
          consecutiveRequestFailures = 0;
          if (boardSource) PL.retryDelay = 120;
          clearPl("检测到旧页面占用琴颈识别；本页正在自动接管");
          message("检测到旧页面占用琴颈识别；本页正在自动接管");
          return;
        }
        if (!response.ok) throw new Error(`板卡接口 ${response.status}`);
        consecutiveRequestFailures = 0;
        const processingMs = Number(response.headers.get("X-Vision-Processing-Ms"));
        const planeResponse = /^application\/x-fretlens-planes(?:;|$)/i.test(response.headers.get("Content-Type") || "");
        if (response.headers.get("X-Vision-Backend") !== "pl" ||
            response.headers.get("X-Vision-Source") !== "pc_camera" ||
            response.headers.get("X-Vision-Token") !== token ||
            !response.headers.get("X-Vision-Processing-Ms") || !Number.isFinite(processingMs) || processingMs < 0 ||
            !(options.coordinates_only ? /^application\/json(?:;|$)/i :
              options.preview_encoding === "planes" ? /^(?:application\/x-fretlens-planes|image\/jpeg)(?:;|$)/i : /^image\/jpeg(?:;|$)/i)
              .test(response.headers.get("Content-Type") || "")) throw new Error("PL 返回来源无效");
        let planes;
        if (options.coordinates_only) await response.arrayBuffer();
        else if (planeResponse) {
          planes = new Uint8Array(await response.arrayBuffer());
          if (planes.length !== 320 * 240 * 2) throw new Error("PL 返回图像尺寸无效");
        }
        else bitmap = await createImageBitmap(await response.blob());
        const now = performance.now();
        if (!current() || trackingGeneration !== PC.generation || V.clicks) return;
        const resultMaxAge = boardSource ? 1800 : plMaxAge;
        if (now-started > resultMaxAge && options.enabled) {
          const analysis = JSON.parse(response.headers.get("X-Vision-Fretboard") || "null");
          if (analysis?.stream_id === stream && analysis.sequence === sequence &&
              analysis.generation === trackingGeneration && analysis.processing_location === "ps_linux") {
            V.searchFeedback = { reason: analysis.reason, elapsedMs: now-started, receivedAt: now };
          }
          // A delayed search result is useful diagnostics, never current geometry.
        } else V.searchFeedback = null;
        const previewScale = bitmap?.width === 640 && bitmap?.height === 240 ? 2 : 1;
        if (controller.signal.aborted) return;
        if (now-started > resultMaxAge) {
          clearPl(`PL 结果延迟 ${Math.round(now-started)} ms · 等待板卡新结果`);
          return;
        }
        if (bitmap && (bitmap.width !== 1280 / previewScale || bitmap.height !== 480 / previewScale))
          throw new Error("PL 返回图像尺寸无效");
        const analysis = options.enabled ? JSON.parse(response.headers.get("X-Vision-Fretboard") || "null") : null;
        if (options.enabled && (!analysis || analysis.stream_id !== stream || analysis.sequence !== sequence ||
            analysis.generation !== trackingGeneration || analysis.processing_location !== "ps_linux"))
          throw new Error("跟踪结果与图像不匹配");
        if (analysis) V.searchFeedback = analysis.status === "tracking" ? null :
          { reason: analysis.reason, elapsedMs: now-started, receivedAt: now };
        if (planes) {
          // Reuse ImageData and write each grayscale pixel in one packed store.
          updatePlaneCanvas(plGrayFrame, planes.subarray(0, 320 * 240), 0);
          updatePlaneCanvas(plEdgeFrame, planes.subarray(320 * 240), 1);
        }
        for (const [target, x] of bitmap ? [[plGrayFrame, 0], [plEdgeFrame, 640]] : []) {
          target.width = 640; target.height = 480;
          target.getContext("2d").drawImage(bitmap, x / previewScale, 0, 640 / previewScale, 480 / previewScale, 0, 0, 640, 480);
        }
        if (bitmap || planes) PL.previewAt = now;
        PL.frame = { source: boardSource ? "board_camera" : "pc_camera", session,
          sequence: boardSource ? sequence : null, started,
          expiresAt: started + (boardSource ? 1800 : plMaxAge),
          processingMs, roundtripMs: now - started, fps: lastDone ? 1000 / (now - lastDone) : 0 };
        if (pcMatched.width !== 640 || pcMatched.height !== 480) { pcMatched.width = 640; pcMatched.height = 480; }
        pcMatched.getContext("2d").drawImage(canvas, 0, 0);
        if (boardSource) {
          PC.stream = stream; PC.sequence = sequence; PC.generation = trackingGeneration;
          PC.matchedAt = now; V.boardProcessedAt = now;
          if (analysis?.stream_id === stream && analysis.sequence === sequence && analysis.generation === trackingGeneration) {
            if (options.enabled) applyAuto(analysis, sequence, stream);
            if (analysis.hand_region?.stream_id === stream && analysis.hand_region.generation === trackingGeneration &&
                analysis.hand_region.sequence === sequence) {
              const liveSnapshot = analysis.fingertip_neural;
              updateFingertipPreview(liveSnapshot, stream, trackingGeneration, sequence, started);
            }
          }
        } else {
          PC.stream = stream; PC.matchedAt = started; PC.sequence = sequence; PC.seedPending = false;
          if (options.enabled) applyAuto(analysis, sequence, stream);
        }
        if (!boardSource) updateFingertipPreview(analysis?.fingertip_neural, stream, trackingGeneration, sequence, started);
        if (analysis) acceptFingertipNotes(analysis);
        takeoverAttempted = false;
        lastDone = now; PL.status = boardSource ? "C922 → PS/PL" : "电脑采集 → PL";
      } catch (error) {
        if (current()) {
          if (boardSource) {
            consecutiveRequestFailures = Math.min(consecutiveRequestFailures + 1, 4);
            const retryMs = Math.min(4000, 250 * (2 ** (consecutiveRequestFailures - 1)));
            PL.retryDelay = retryMs;
            clearPl(controller.signal.aborted ?
              `琴颈分析请求超时；摄像头视频继续，${(retryMs / 1000).toFixed(2)} 秒后重试` :
              `琴颈分析暂不可用；摄像头视频继续，稍后重试：${error.message}`);
          } else clearPl(controller.signal.aborted ? "电脑传图超时" : error.message);
        }
      } finally {
        bitmap?.close(); clearTimeout(timeout);
        if (current()) {
          PL.request = null;
          const boardPeriod = boardSource && PL.frame?.roundtripMs > 120 ?
            boardRecognitionSlowPeriod : boardRecognitionPeriod;
          const boardInterval = PL.retryDelay ||
            Math.max(0, (boardSource ? boardPeriod : plFramePeriod) -
              (performance.now() - started));
          PL.retryDelay = 0;
          PL.timer = setTimeout(poll, boardInterval);
        }
      }
    };
    poll();
  }
  function startPlPoll(session) {
    const generation = PL.generation;
    const loopKey = `board:${V.boardStream}:${PC.generation}:${generation}`;
    if (PL.loopKey === loopKey) return;
    PL.loopKey = loopKey;
    const current = () => generation === PL.generation && session === V.session && V.source === "board";
    const poll = async () => {
      if (!current()) return;
      const started = performance.now(), controller = new AbortController();
      PL.request = controller;
      const timeout = setTimeout(() => controller.abort(), plMaxAge);
      let bitmap;
      try {
        if (!cameraFresh()) throw new Error("等待新相机帧");
        const response = await fetch("/api/vision/pl/status", { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error("不可用");
        const info = await response.json();
        if (!current()) return;
        if (info.backend !== "pl" || info.status !== "ready" || info.stale === true || info.fresh === false || info.error ||
            !(info.stale === false || info.fresh === true)) throw new Error("不可用");
        if (typeof info.age_ms !== "number" || !Number.isFinite(info.age_ms) || info.age_ms < 0 || info.age_ms > plMaxAge ||
            !Number.isSafeInteger(info.sequence) || info.sequence < 0 || info.stream_id !== V.boardStream ||
            info.source !== V.boardOrigin || typeof info.captured_ms !== "number" ||
            !Number.isFinite(info.captured_ms) || info.captured_ms < 0) throw new Error("板卡摄像头图像延迟或来源不匹配；此入口不读取电脑摄像头");
        const image = await fetch("/vision/pl.jpg", { cache: "no-store", signal: controller.signal });
        if (!image.ok || !/^image\/jpeg(?:;|$)/i.test(image.headers.get("Content-Type") || "")) throw new Error("图像不可用");
        const sequenceHeader = image.headers.get("X-Vision-Sequence"), capturedHeader = image.headers.get("X-Vision-Captured-Ms");
        const processingHeader = image.headers.get("X-Vision-Processing-Ms");
        const frame = { sequence: Number(sequenceHeader), capturedMs: Number(capturedHeader),
          processingMs: Number(processingHeader), stream: image.headers.get("X-Vision-Stream"),
          source: image.headers.get("X-Vision-Source"), ageMs: info.age_ms, started,
          fps: Number.isFinite(info.processing_fps) && info.processing_fps >= 0 ? info.processing_fps : 0 };
        if (!sequenceHeader?.trim() || !capturedHeader?.trim() || !processingHeader?.trim() ||
            image.headers.get("X-Vision-Backend") !== "pl" || frame.stream !== info.stream_id || frame.source !== info.source ||
            !Number.isSafeInteger(frame.sequence) || frame.sequence < info.sequence ||
            !Number.isFinite(frame.capturedMs) || frame.capturedMs < info.captured_ms ||
            (frame.sequence === info.sequence && frame.capturedMs !== info.captured_ms) ||
            !Number.isFinite(frame.processingMs) || frame.processingMs < 0) throw new Error("图像元数据无效");
        bitmap = await createImageBitmap(await image.blob());
        if (!current()) return;
        if (controller.signal.aborted || bitmap.width !== 1280 || bitmap.height !== 480 ||
            !cameraFresh() || frame.stream !== V.boardStream || frame.source !== V.boardOrigin || plAge(frame) > plMaxAge)
          throw new Error("板卡摄像头图像延迟或来源不匹配；此入口不读取电脑摄像头");
        const previous = PL.last?.stream === frame.stream ? PL.last : null;
        if (previous && (frame.sequence < previous.sequence || frame.capturedMs < previous.capturedMs ||
            (frame.sequence === previous.sequence && frame.capturedMs !== previous.capturedMs))) throw new Error("帧序号无效");
        frame.expiresAt = performance.now() + plMaxAge - plAge(frame);
        if (previous?.sequence === frame.sequence) frame.expiresAt = Math.min(frame.expiresAt, previous.expiresAt);
        if (frame.expiresAt <= performance.now()) throw new Error("板卡摄像头暂无有效新图像");
        for (const [canvas, x] of [[plGrayFrame, 0], [plEdgeFrame, 640]]) {
          if (canvas.width !== 640 || canvas.height !== 480) { canvas.width = 640; canvas.height = 480; }
          canvas.getContext("2d").drawImage(bitmap, x, 0, 640, 480, 0, 0, 640, 480);
        }
        PL.frame = frame; PL.last = frame; PL.status = "已连接";
      } catch (error) {
        if (current()) clearPl(controller.signal.aborted ? "请求超时" : error.message);
      } finally {
        bitmap?.close(); clearTimeout(timeout);
        if (current()) {
          PL.request = null;
          PL.timer = setTimeout(poll, Math.max(0, plFramePeriod - (performance.now() - started)));
        }
      }
    };
    poll();
  }
  function drawFrame(ctx, frame, box) {
    const region = box.region || fullRegion;
    const size = frame === video ? { width: video.videoWidth, height: video.videoHeight } : inputSize();
    ctx.drawImage(frame, region.x * size.width, region.y * size.height,
      region.width * size.width, region.height * size.height, box.x, box.y, box.width, box.height);
  }
  function saveCalibration() {
    try {
      localStorage.setItem("fretlens-calibration-v1", JSON.stringify({ corners: V.calibration.corners,
        firstFret: V.calibration.firstFret, lastFret: V.calibration.lastFret,
        tuning: V.calibration.tuning, shape: V.shape, identity: V.identity, source: V.source,
        dynamic: followsNeck() }));
    } catch (_) { message("已标定；当前浏览器无法保存本地标定"); }
  }
  function loadCalibration(size) {
    V.calibration = null;
    try {
      const saved = JSON.parse(localStorage.getItem("fretlens-calibration-v1"));
      if (saved && !saved.dynamic && !followsNeck() && saved.source === V.source && saved.identity === V.identity &&
        saved.shape?.width === size.width && saved.shape?.height === size.height) {
        V.calibration = F.createCalibration(saved.corners, saved.firstFret, saved.lastFret, saved.tuning);
        $("firstFret").value = saved.firstFret;
        $("lastFret").value = saved.lastFret;
      }
    } catch (_) { /* Invalid saved geometry never becomes recognition evidence. */ }
  }
  function clearEvidence() {
    V.result = null;
    $("visualEvidence").textContent = V.source ? "尚无指尖位置候选" : "视觉未连接";
    $("visualEvent").hidden = true;
  }
  function stop() {
    displayTrack = null;
    resetTracking();
    stopPl();
    PC.generation += 1; PC.corners = null; PC.seedPending = false; PC.sequence = -1; PC.matchedAt = 0;
    V.session += 1;
    cancelAnimationFrame(V.request);
    clearTimeout(V.boardTimer);
    V.boardRequest?.abort(); V.boardRequest = null;
    V.source = null;
    placeLiveVideo(null, false);
    V.searchFeedback = null;
    V.handObservation = null;
    V.result = null;
    V.clicks = null;
    V.stream = null;
    V.detect = false;
    V.boardOrigin = null; V.boardSequence = -1;
    V.boardStream = null; V.boardError = null; V.boardProcessedAt = 0;
    V.boardStreamActive = false;
    V.automatic = null; V.autoRetryAt = 0;
    clearPreviews();
    if (V.url) { video.pause(); video.removeAttribute("src"); video.load(); URL.revokeObjectURL(V.url); V.url = null; }
    delete stage.dataset.visionSource;
    stage.dataset.mode = "idle";
    stage.dataset.calibrating = "false";
    modelButton.textContent = "启动指尖检测";
    clearEvidence();
    message("视觉已停止");
    $("overlayState").textContent = "待标定";
    $("frameStatusText").textContent = "CAMERA OFFLINE";
    $("videoRate").textContent = "视觉已停止";
    overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  }
  function begin(source, size, calibration = null, identity = null, preserveSession = false) {
    clearFingertipPreview();
    window.FretPrepareVision?.();
    if (!preserveSession) V.session += 1;
    cancelAnimationFrame(V.request);
    V.source = source;
    V.shape = size;
    V.identity = identity;
    V.lastTime = -1;
    V.lastInference = 0;
    V.lastFrame = -1;
    V.frameCount = 0;
    V.fpsFrames = 0; V.displayFps = 0;
    V.inferenceCount = 0; V.fpsInferences = 0; V.inferenceFps = 0;
    V.processedToken = null;
    clearPreviews();
    V.fpsStart = performance.now();
    V.lastPublish = -Infinity;
    stage.dataset.visionSource = source;
    stage.dataset.mode = source === "synthetic" ? "demo" : "live";
    if (source !== "camera") $("videoRate").textContent =
      `${size.width}×${size.height} · ${source === "board" ? "PS 图像接口" : source === "synthetic" ? "合成图" : "录像"}`;
    if (calibration) V.calibration = calibration;
    else if (autoEnabled()) V.calibration = null;
    else loadCalibration(size);
    $("followNeck").disabled = !(source === "board" && V.boardOrigin === "camera");
    clearEvidence();
    message(autoEnabled() ? "正在寻找六根琴弦和可见品丝" :
      V.calibration ? "已载入标定；请确认相机与琴颈位置未改变" : "先标定指板，再启动指尖检测");
    tick(V.session);
  }
  function attachCamera(stream) {
    if (V.url) { URL.revokeObjectURL(V.url); V.url = null; }
    V.stream = stream;
    begin("camera", { width: video.videoWidth, height: video.videoHeight }, null,
      stream.getVideoTracks()[0].getSettings().deviceId || "local_camera");
    if ($("pcPlEnabled").checked) startPcPl(V.session);
    message("电脑仅采集传图；等待板卡 PS 手部检测与 PL 跟踪结果");
  }
  function syntheticHands(now) {
    if (!V.calibration) return [];
    const landmarks = Array.from({ length: 21 }, () => ({ x: -.5, y: -.5, z: 0 }));
    const target = [{ fret: 3, string: 6 }, { fret: 2, string: 5 }, { fret: 3, string: 1 }];
    target.forEach((p, i) => {
      const a = V.calibration.boundaries[p.fret - 1], b = V.calibration.boundaries[p.fret];
      landmarks[tips[i]] = { ...F.transform(V.calibration.toImage,
        { x: (a.x + b.x) / 2 + .002 * Math.sin(now / 450), y: (6 - p.string) / 5 }), z: 0 };
    });
    return [{ score: .99, label: "Synthetic", landmarks }];
  }
  function infer(now) {
    if (V.source === "camera") return;
    if (V.source === "synthetic") {
      V.result = { ...F.mapHands(V.calibration, syntheticHands(now)), source: "synthetic", receivedAt: now,
        frame_sequence: V.frameCount, inference_ms: 0 };
      return;
    }
    if (!V.detect || !V.model || (V.source !== "board" && video.readyState < 2) ||
      (V.source === "board" && now - V.boardReceivedAt > 500) || now - V.lastInference < 66 || inputToken() === V.lastTime) return;
    const started = performance.now();
    const detected = V.model.detectForVideo(inputFrame(), now);
    V.inferenceCount += 1;
    const hands = (detected.landmarks || []).map((landmarks, i) => ({ landmarks,
      score: 1, handedness_score: detected.handedness?.[i]?.[0]?.score ?? null,
      score_source: "accepted_by_model_thresholds_not_calibrated_probability",
      label: detected.handedness?.[i]?.[0]?.categoryName || "Hand" }));
    V.handObservation = { hands, receivedAt: performance.now(), frame: inputFrame(), token: inputToken() };
    // Live-camera landmarks and the board's older capture are not synchronized.
    // Keep display tracking independent; do not infer pressed strings across them.
    const geometry = pcTracking() ? null : V.automatic?.geometry;
    const mapped = autoEnabled() ? F.mapObservedHands(geometry, hands) : F.mapHands(V.calibration, hands);
    V.result = { ...mapped, source: resultSource(),
      receivedAt: V.source === "board" ? V.boardReceivedAt : pcTracking() && pcFresh() ? PC.matchedAt : now,
      frame_sequence: V.source === "board" ? V.boardSequence : pcTracking() ? PC.sequence : V.frameCount,
      media_time_seconds: V.source === "board" ? null : video.currentTime,
      captured_monotonic_ms: V.source === "board" ? V.boardCapturedMs : null,
      inference_location: "computer_browser", synchronized: false,
      inference_ms: performance.now() - started };
    V.lastInference = performance.now();
    V.lastTime = inputToken();
  }
  function tick(session) {
    if (session !== V.session || !V.source) return;
    const now = performance.now();
    if (now - V.fpsStart >= 1000) {
      const seconds = (now - V.fpsStart) / 1000;
      V.displayFps = (V.frameCount - V.fpsFrames) / seconds;
      V.inferenceFps = (V.inferenceCount - V.fpsInferences) / seconds;
      V.fpsFrames = V.frameCount; V.fpsInferences = V.inferenceCount; V.fpsStart = now;
      const size = inputSize();
      const synchronizedRate = colorLiveView() ?
        `实时视频 ${V.displayFps.toFixed(1)} fps` : pcProcessedDisplay() ?
        `网页/板卡同步 ${V.displayFps.toFixed(1)} fps` : `预览 ${V.displayFps.toFixed(1)} fps`;
      const cameraRate = V.source === "camera" ? video.srcObject?.getVideoTracks?.()[0]?.getSettings?.().frameRate : null;
      const inputRate = Number.isFinite(cameraRate) ? ` · 摄像头输入 ${Math.round(cameraRate)} fps` : "";
      const boardRate = (pcProcessedDisplay() || V.source === "board" && V.boardOrigin === "camera") && PL.frame ?
        ` · 识别 ${PL.frame.fps.toFixed(1)} 次/秒` : "";
      $("videoRate").textContent = `${size.width}×${size.height} · ${synchronizedRate}${inputRate}${boardRate}` +
        (V.detect ? ` · 检测 ${V.inferenceFps.toFixed(1)} fps` : "");
    }
    if (V.result && now - V.result.receivedAt > 500) clearEvidence();
    if (V.source === "board" && V.tracking?.status === "tracking" && now - V.boardReceivedAt > 500)
      loseTracking("stale_frame");
    if (V.source === "board" && autoEnabled() && now - V.boardReceivedAt > 500 && V.automatic) { clearAuto(); clearEvidence(); }
    try { infer(now); } catch (error) {
      V.detect = false;
      clearEvidence();
      modelButton.textContent = "启动指尖检测";
      message(`指尖检测失败：${error.message}`);
      V.modelError = `检测失败：${error.message}`;
    }
    render();
    if (now - V.lastPublish >= 100) { publish(now); V.lastPublish = now; }
    V.request = requestAnimationFrame(() => tick(session));
  }
  function drawSynthetic(ctx, box) {
    ctx.fillStyle = "#222827";
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.fillStyle = "#444b49";
    if (V.calibration) {
      const corners = V.calibration.corners.map((p) => projectionPoint(p, box));
      ctx.beginPath(); corners.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.closePath(); ctx.fill();
    }
    ctx.fillStyle = "#dcff58";
    ctx.font = "bold 18px sans-serif";
    ctx.fillText("SYNTHETIC / 几何测试帧", box.x + 16, box.y + 30);
  }
  function updatePreviews(box) {
    const now = performance.now(), token = V.source === "synthetic" ? V.frameCount : inputToken();
    const regionKey = JSON.stringify(box.region);
    if (now - processedAt < 100 || (V.processedToken === token && processedRegion === regionKey)) return;
    processedAt = now;
    const size = inputSize();
    const scale = Math.min(1, 320 / size.width, 180 / size.height);
    const width = Math.max(1, Math.round(size.width * scale)), height = Math.max(1, Math.round(size.height * scale));
    if (scratch.width !== width || scratch.height !== height) { scratch.width = width; scratch.height = height; }
    scratchCtx.drawImage(inputFrame(), 0, 0, scratch.width, scratch.height);
    const image = scratchCtx.getImageData(0, 0, scratch.width, scratch.height);
    const gray = new Uint8Array(scratch.width * scratch.height);
    for (let p = 0; p < gray.length; p++) gray[p] = (77 * image.data[p * 4] + 150 * image.data[p * 4 + 1] + 29 * image.data[p * 4 + 2] + 128) >> 8;
    for (let p = 0; p < gray.length; p++) image.data[p * 4] = image.data[p * 4 + 1] = image.data[p * 4 + 2] = gray[p];
    for (const canvas of [grayFrame, edgeFrame]) {
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    }
    grayFrame.getContext("2d").putImageData(image, 0, 0);
    for (let y = 0; y < scratch.height; y++) for (let x = 0; x < width; x++) {
      const p = y * width + x;
      let value = 0;
      if (x > 0 && x < width - 1 && y > 0 && y < scratch.height - 1) {
        const gx = -gray[p - width - 1] + gray[p - width + 1] - 2 * gray[p - 1] + 2 * gray[p + 1] - gray[p + width - 1] + gray[p + width + 1];
        const gy = -gray[p - width - 1] - 2 * gray[p - width] - gray[p - width + 1] + gray[p + width - 1] + 2 * gray[p + width] + gray[p + width + 1];
        value = Math.abs(gx) + Math.abs(gy) >= 100 ? 255 : 0;
      }
      image.data[p * 4] = image.data[p * 4 + 1] = image.data[p * 4 + 2] = value;
    }
    edgeFrame.getContext("2d").putImageData(image, 0, 0);
    V.processedToken = token; processedRegion = regionKey;
  }
  function drawPreviews(box, hardware) {
    if (hardware && V.source === "camera" && performance.now() - (PL.previewAt || 0) > plMaxAge) {
      for (const canvas of [grayPreview, edgePreview]) canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height);
      return;
    }
    const frames = hardware ? [plGrayFrame, plEdgeFrame] : [grayFrame, edgeFrame];
    const width = frames[0].width, height = frames[0].height;
    const region = box.region, cropWidth = region.width * width, cropHeight = region.height * height;
    const previewScale = Math.min(1, 320 / cropWidth, 180 / cropHeight);
    for (const [canvas, frame] of [[grayPreview, frames[0]], [edgePreview, frames[1]]]) {
      const w = Math.max(1, Math.round(cropWidth * previewScale)), h = Math.max(1, Math.round(cropHeight * previewScale));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      canvas.getContext("2d").drawImage(frame, region.x * width, region.y * height, cropWidth, cropHeight, 0, 0, w, h);
    }
  }
  function render() {
    if (!V.source) return;
    const now = performance.now();
    const synchronized = !colorLiveView() &&
      (V.source === "camera" && pcProcessedDisplay() && validPl() || boardProcessedDisplay());
    if (V.source === "camera" && !V.clicks && $("visionView").value === "color") {
      if (!synchronized) {
        const token = inputToken();
        if (video.readyState >= 2 && token !== V.lastFrame) { V.frameCount += 1; V.lastFrame = token; }
      }
      const key = [PC.generation, PC.sequence, PL.frame?.started, pcFresh(), validPl(),
        overlay.width, overlay.height, autoEnabled(), $("preprocessDetails").open].join(":");
      if (key === V.lastOverlayKey && now - V.lastOverlayAt < 100) return;
      V.lastOverlayKey = key; V.lastOverlayAt = now;
    } else V.lastOverlayKey = null;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (synchronized && (!pcFresh() || !validPl())) { clearPreviews(); labelPreviews(false, false); }
    const fresh = V.source === "synthetic" || (V.source === "board" ?
      !V.boardError && V.boardSequence >= 0 && performance.now() - V.boardReceivedAt <= 500 : video.readyState >= 2);
    const staleBoardFrame = V.source === "board" && V.boardOrigin === "camera" && boardFrame.width > 1 &&
      performance.now() - V.boardReceivedAt <= 5000;
    if (!fresh && !staleBoardFrame) {
      if (V.source === "board") {
        if (V.boardOrigin === "camera") {
          ctx.fillStyle = "#111516"; ctx.fillRect(0, 0, overlay.width, overlay.height);
        }
        clearPreviews(); labelPreviews(false, false);
      } else if (V.source === "camera") {
        placeLiveVideo(null, false);
        clearPreviews(); labelPreviews(false, false);
      }
      return;
    }
    const view = $("visionView").value, hardware = validPl();
    const box = frameBox();
    if (V.source === "synthetic") {
      const size = inputSize();
      if (syntheticFrame.width !== size.width || syntheticFrame.height !== size.height) {
        syntheticFrame.width = size.width; syntheticFrame.height = size.height;
      }
      drawSynthetic(syntheticFrame.getContext("2d"), { x: 0, y: 0, ...size });
    }
    const showIntermediate = !document.hidden && $("preprocessDetails").open;
    const previewFresh = hardware;
    if (V.source !== "camera" && ((!previewFresh && (showIntermediate || view !== "color")) || V.source === "board" && view !== "color")) updatePreviews(box);
    if (showIntermediate && (hardware || V.processedToken !== null))
      drawPreviews(hardware && V.source === "board" ? { region: fullRegion } : box, hardware);
    labelPreviews(hardware);
    const processed = (hardware || V.processedToken !== null) && view !== "color";
    const liveVideo = V.source === "camera" && !synchronized && !V.clicks && !processed;
    placeLiveVideo(box, liveVideo);
    if (!liveVideo && !(staleBoardFrame && !fresh)) {
      ctx.fillStyle = "#111516"; ctx.fillRect(0, 0, overlay.width, overlay.height);
    }
    ctx.save(); ctx.beginPath(); ctx.rect(box.x, box.y, box.width, box.height); ctx.clip();
    if (processed) {
      const frame = hardware && V.source === "camera" ? (view === "edge" ? plEdgeFrame : plGrayFrame) :
        (view === "edge" ? edgeFrame : grayFrame), region = box.region;
      ctx.drawImage(frame, region.x * frame.width, region.y * frame.height,
        region.width * frame.width, region.height * frame.height, box.x, box.y, box.width, box.height);
    } else if (!liveVideo) drawFrame(ctx, displayFrame(), box);
    const displayToken = V.clicks ? `selection:${PC.generation}:${V.clicks.length}` :
      V.source === "board" && view === "color" ? `camera:${V.boardSequence}` :
      synchronized ? `board:${PC.generation}:${PC.sequence}` :
      V.source === "camera" && processed ? `pl:${PL.frame?.started ?? PC.sequence}` : inputToken();
    if (V.source === "synthetic" || displayToken !== V.lastFrame) {
      V.frameCount += 1; V.lastFrame = displayToken;
    }
    if (view === "color" && showIntermediate) drawProcessingOverlay(ctx, hardware, box);
    const line = (a, b, color, thickness = 1) => {
      const p = projectionPoint(a, box), q = projectionPoint(b, box);
      ctx.strokeStyle = color; ctx.lineWidth = thickness; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
    };
    const neckOutline = displayCorners();
    if (neckOutline && autoEnabled()) {
      line(neckOutline[0], neckOutline[1], "#66d6d1", 4);
      line(neckOutline[3], neckOutline[2], "#66d6d1", 4);
      const estimated = V.automatic?.estimated_strings;
      const backendStrings = V.automatic?.strings?.length === 6;
      if (!backendStrings && estimated?.status === "estimated" && V.automatic.roi_status === "tracking" &&
          estimated.stream_id === V.automatic.stream_id &&
          estimated.generation === V.automatic.generation &&
          estimated.sequence === V.automatic.sequence &&
          estimated.strings?.length === 6 && estimated.boundaries?.length === 7) {
        ctx.save();
        ctx.setLineDash([7, 5]);
        for (const item of estimated.strings) line(item.start, item.end, "#ffcf55", 1.5);
        ctx.setLineDash([2, 6]); ctx.globalAlpha = .5;
        for (const item of estimated.boundaries.slice(1, -1)) line(item.start, item.end, "#66d6d1", 1);
        ctx.restore();
      }
    }
    if (autoEnabled() && V.automatic?.geometry && (V.source !== "board" || boardProcessedDisplay())) {
      for (const item of V.automatic.strings) {
        const estimated = item.estimated === true || item.observed === false &&
          item.source === "live_neck_width_six_equal_lanes";
        ctx.save();
        if (estimated) ctx.setLineDash([7, 5]);
        line(item.start, item.end, estimated ? "#ffcf55" : "#66d6d1", estimated ? 1.5 : 2);
        ctx.restore();
      }
      for (const item of V.automatic.frets) {
        const estimated = item.estimated === true;
        ctx.save();
        if (estimated) ctx.setLineDash([5, 4]);
        line(item.start, item.end, estimated ? "#d9e86a" : "#dcff58", estimated ? 1 : 1.5);
        ctx.restore();
      }
      const calibration = currentNoteCalibration();
      if (calibration) {
        const wires = [...V.automatic.frets].sort((a, b) =>
          F.transform(V.automatic.geometry.toBoard, a.start).x -
          F.transform(V.automatic.geometry.toBoard, b.start).x);
        wires.forEach((wire, i) => {
          const p = projectionPoint(wire.start, box);
          ctx.font = "12px sans-serif"; ctx.fillStyle = "#edf1e9";
          ctx.fillText(String(calibration.first_boundary_fret + (calibration.frets_reversed ? -i : i)),
            Math.max(2, Math.min(overlay.width - 25, p.x + 2)), Math.max(14, p.y - 5));
        });
      }
    }
    if (V.calibration) {
      for (let string = 0; string < 6; string++) line(F.transform(V.calibration.toImage, { x: 0, y: string / 5 }),
        F.transform(V.calibration.toImage, { x: 1, y: string / 5 }), "#66d6d1", 1.5);
      for (const boundary of V.calibration.boundaries) {
        const a = F.transform(V.calibration.toImage, { x: boundary.x, y: 0 });
        line(a, F.transform(V.calibration.toImage, { x: boundary.x, y: 1 }), "#dcff58");
        const p = projectionPoint(a, box); ctx.fillStyle = "#edf1e9"; ctx.font = "12px sans-serif";
        ctx.fillText(String(boundary.fret), p.x + 2, p.y - 5);
      }
    }
    for (const hand of visibleHands()) {
      for (const [a, b] of edges) {
        if (hand.joint_valid && (!hand.joint_valid[a] || !hand.joint_valid[b])) continue;
        if (hand.landmarks[a]?.x >= 0 && hand.landmarks[b]?.x >= 0) line(hand.landmarks[a], hand.landmarks[b], "#b7a5ff", 2);
      }
      for (const index of [0, 4, 8, 12, 16, 20]) {
        if (hand.joint_valid && !hand.joint_valid[index]) continue;
        const point = hand.landmarks[index];
        if (!point || point.x < 0 || point.y < 0) continue;
        const p = projectionPoint(point, box);
        ctx.fillStyle = index === 0 ? "#fff" : "#ffcf55";
        ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI*2); ctx.fill();
      }
    }
    const audioSnapshot = window.FretAudioSnapshot?.();
    const mapped = audioSnapshot?.boardFusion;
    const sameMappingFrame = audioSnapshot && !audioSnapshot.stale && audioSnapshot.testMode === 0 &&
      performance.now()-audioSnapshot.receivedAt <= 500 && performance.now() >= audioSnapshot.receivedAt &&
      mapped?.status === "audio_visual_aligned" && mapped.visual_sequence === V.automatic?.sequence &&
      mapped.visual_identity?.[0] === V.automatic?.stream_id &&
      mapped.visual_identity?.[1] === V.automatic?.generation &&
      mapped.visual_identity?.[2] === V.automatic?.lattice_epoch;
    for (const position of ((V.source !== "board" || boardProcessedDisplay()) ? V.result?.positions : []) || []) {
      const p = projectionPoint(position, box);
      ctx.strokeStyle = position.ambiguous ? "#ff7152" : "#dcff58"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#edf1e9"; ctx.font = "12px sans-serif";
      const candidate = sameMappingFrame && mapped.aligned_positions?.find(item =>
        item.finger === position.finger && item.hand_index === position.hand_index &&
        item.string_line === position.string_line && item.fret_interval === position.fret_interval &&
        Number.isInteger(item.fret) && Number.isInteger(item.string));
      ctx.fillText(candidate ? `${candidate.string}弦${candidate.fret}品?` : position.relative ? `弦线${position.string_line}·格${position.fret_interval}` :
        `${position.string}弦${position.fret}品 ${position.note_name || ""}${position.ambiguous ? "?" : ""}`, Math.min(p.x + 9, overlay.width - 115), Math.max(14, p.y - 8));
    }
    const tracked = currentFingertips();
    if (tracked) {
      for (const [i, name] of fingertipNames.entries()) {
        const point = tracked.points[name];
        if (!point) continue;
        const p = projectionPoint({x:point.x/640,y:point.y/480}, box);
        ctx.fillStyle = ["#ff4b4b", "#ffdb58", "#51e5ba", "#7abfff"][i];
        ctx.strokeStyle = "#111516"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.font = "bold 12px sans-serif"; ctx.fillStyle = "#fff";
        ctx.fillText(String(i + 1), p.x + 9, Math.max(14, p.y - 8));
      }
      const pointLabel = `${tracked.keypoint_source === "optical_flow" ? "PL 指尖跟踪" : "PL 神经网络"} · 当前帧 #${tracked.sequence} · 定位龄 ${Math.ceil(tracked.detector_age_ms)} ms`;
      ctx.font = "13px sans-serif"; ctx.textBaseline = "top";
      const labelWidth = ctx.measureText(pointLabel).width;
      ctx.fillStyle = "rgba(17,21,22,.82)";
      ctx.fillRect(box.x + 8, box.y + 8, labelWidth + 12, 24);
      ctx.fillStyle = "#edf1e9"; ctx.fillText(pointLabel, box.x + 14, box.y + 13);
    }
    if (!fresh && staleBoardFrame) {
      const age = Math.max(0, Math.round(performance.now() - V.boardReceivedAt));
      const text = `视频暂时中断 · 保留上一帧 ${age} ms`;
      ctx.font = "14px sans-serif"; ctx.textBaseline = "top";
      ctx.fillStyle = "rgba(17,21,22,.86)";
      ctx.fillRect(box.x + 8, box.y + 8, ctx.measureText(text).width + 16, 28);
      ctx.fillStyle = "#fff"; ctx.fillText(text, box.x + 16, box.y + 14);
    }
    for (const [i, point] of (V.clicks || []).entries()) {
      const p = projectionPoint(point, box); ctx.fillStyle = "#ff7152";
      ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.fill();
      ctx.fillText(String(i + 1), p.x + 8, p.y);
    }
    const corners = handRegionCorners();
    if (corners) {
      ctx.strokeStyle = "#ff3030"; ctx.lineWidth = 3; ctx.beginPath();
      corners.map(p => projectionPoint(p, box)).forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      ctx.closePath(); ctx.stroke();
      const p = projectionPoint(corners[0], box);
      ctx.font = "16px sans-serif"; ctx.fillStyle = "#ff6060";
      ctx.fillText(V.source === "camera" ?
        (V.automatic?.hand_region ? `手附近区域 · ${Math.round(performance.now()-PC.matchedAt)} ms 前` : V.handObservation?.detection?.inference_location === "ps_pl" ? "手部 · PS/PL" : "手部 · PS") : "按弦手附近",
        Math.max(4, Math.min(p.x, overlay.width-180)), Math.max(20,p.y-8));
    }
    ctx.restore();
  }
  function currentFingertips() {
    const analysis = V.automatic, tracked = analysis?.fingertip_tracking;
    if (!validPl() || !(pcProcessedDisplay() || boardProcessedDisplay()) ||
        !analysis || analysis.sequence !== PC.sequence || analysis.generation !== PC.generation ||
        analysis.stream_id !== PC.stream || tracked?.status !== "tracking" ||
        tracked.sequence !== PC.sequence || tracked.generation !== PC.generation ||
        tracked.stream_id !== PC.stream || !["optical_flow", "neural"].includes(tracked.keypoint_source) ||
        !Number.isFinite(tracked.detector_age_ms) || tracked.detector_age_ms < 0 ||
        tracked.detector_age_ms + performance.now() - PC.matchedAt > 4000 ||
        !Number.isSafeInteger(tracked.source_sequence) || tracked.source_sequence > PC.sequence ||
        tracked.source_sequence < 0 || !tracked.points || typeof tracked.points !== "object") return null;
    const points = Object.fromEntries(fingertipNames.filter(name => {
      const point = tracked.points[name];
      return point?.visible !== false && Number.isFinite(point?.x) && Number.isFinite(point?.y) &&
        point.x >= 0 && point.x < 640 && point.y >= 0 && point.y < 480;
    }).map(name => [name, tracked.points[name]]));
    return Object.keys(points).length ? { ...tracked, points } : null;
  }
  function drawProcessingOverlay(ctx, hardware, box) {
    const freshPreview = hardware && (V.source !== "camera" || pcFresh());
    const frames = freshPreview ? [plGrayFrame, plEdgeFrame] : V.source === "camera" ? [] : [grayFrame, edgeFrame];
    const corners = handRegionCorners();
    if (!corners || frames.length !== 2) return;
    ctx.save(); ctx.beginPath();
    corners.map(p => projectionPoint(p, box)).forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.closePath(); ctx.clip();
    const region = box.region;
    if (V.source === "camera") {
      // The PL patch and its box describe the same capture, never the newer
      // live background. Do not manufacture current-frame tracking on the PC.
      ctx.drawImage(pcMatched, region.x*pcMatched.width, region.y*pcMatched.height,
        region.width*pcMatched.width, region.height*pcMatched.height,
        box.x, box.y, box.width, box.height);
    }
    for (const [index, frame] of frames.entries()) {
      if (frame.width <= 1 || frame.height <= 1) continue;
      ctx.globalAlpha = index ? .35 : .65;
      ctx.globalCompositeOperation = index ? "screen" : "source-over";
      ctx.drawImage(frame, region.x*frame.width, region.y*frame.height,
        region.width*frame.width, region.height*frame.height,
        box.x, box.y, box.width, box.height);
    }
    ctx.restore();
  }
  function visibleHands() {
    const observation = V.handObservation;
    if (observation?.source === "ps_linux") return pcFresh() && observation.token === PC.sequence &&
      observation.frame === inputFrame() ? observation.hands : [];
    return V.detect && observation && performance.now()-observation.receivedAt <= 500 &&
      observation.frame === inputFrame() && (observation.token === inputToken() ||
      observation.frame === video && performance.now()-observation.receivedAt <= 250) ? observation.hands : [];
  }
  function publish(now) {
    renderFingertipPreview(now);
    const recentHands = V.detect && V.handObservation && now-V.handObservation.receivedAt <= 500 ?
      V.handObservation.hands.length : 0;
    const neckStatus = $("neckTrackingStatus");
    if (neckStatus) {
      const fresh = V.source === "camera" ? pcFresh() : V.source === "board" && cameraFresh(now);
      const automatic = fresh ? V.automatic : null;
      const following = automatic?.roi_status === "tracking";
      neckStatus.textContent = !V.source ? "琴颈跟踪：等待画面" :
        !autoEnabled() ? "琴颈跟踪：自动定位未启用" : following ?
        `${automatic.status === "tracking" ? "琴颈已锁定" : "琴颈候选跟踪"}：${automatic.flow_backend === "pl_pyramidal_lk_ps_ransac" ? "PL LK + PS RANSAC" : "区域跟随"}${Number.isFinite(automatic.flow?.inliers) ? ` · ${automatic.flow.inliers} 个内点` : ""}${automatic.reason === "stale_lattice_geometry" ? " · 旧弦品几何已清除" : automatic.estimated_strings?.status === "estimated" ? " · 六弦位置估计" : automatic.status !== "tracking" ? " · 六弦待确认" : ""}` :
        V.searchFeedback && now-V.searchFeedback.receivedAt < 6000 ?
        `琴颈搜索：${V.searchFeedback.reason === "no_supported_lattice" ? "找到琴颈候选，但未找到合格弦品结构" :
          V.searchFeedback.reason === "stale_lattice_geometry" ? "旧弦品几何已过期，正在重新测量" :
          V.searchFeedback.reason === "no_color_neck_candidate" ? "画面中没有完整琴颈" :
          "正在重新寻找琴颈"} · ${(V.searchFeedback.elapsedMs/1000).toFixed(1)} 秒` :
        `琴颈跟踪：${automatic?.reason === "roi_tracking_lost" ? "丢失，重新寻找" : "寻找琴颈"}${PL.frame ? ` · 板端 ${PL.frame.processingMs.toFixed(0)} ms` : ` · ${PL.status}`}`;
    }
    const handStatus = $("handDetectionStatus");
    if (handStatus) handStatus.textContent = V.source === "camera" || V.source === "board" ?
      V.automatic?.hand_region ? (handRegionCorners() ? "手部 ROI · PL 四指尖推理" :
        V.automatic.hand_region.status === "waiting_for_neck" ? "先锁定琴颈，再定位手附近区域" :
        "正在寻找手附近的指板区域") :
      visibleHands().length ? "已检测手部 · 跟踪中" :
      V.handObservation?.detection?.error ? `板卡手部检测错误：${V.handObservation.detection.error}` :
      V.handObservation?.detection?.detector_hand_count > 0 ? "已识别手部 · 正在跟踪到当前画面" :
      "正在寻找完整手部" : !V.source ? "手部检测：等待摄像头" :
      V.modelLoading ? "手部检测：正在加载模型" : V.modelError ? `手部检测：${V.modelError}` :
      !V.detect ? "手部检测：未启动" : recentHands ?
      `手部检测：已找到 ${recentHands} 只手${V.result?.positions?.length ? ` · ${V.result.positions.length} 个指尖位置候选` : " · 尚无可对应的弦品位置"}` :
      "手部检测：运行中，尚未检测到手部";
    if (!V.source) return;
    if (pcTracking() && !pcFresh()) { V.automatic = null; clearEvidence(); }
    const synthetic = V.source === "synthetic" || (V.source === "board" && V.boardOrigin === "synthetic");
    $("overlayState").textContent = synthetic ? "SYNTHETIC" : autoEnabled() ?
      V.automatic?.status === "tracking" ? `6 条弦线 · ${V.automatic.frets.length} 条品丝` : "寻找弦与品丝" : V.tracking ?
      ({ tracking: "跟随中", pending: "正在锁定", lost: "跟踪丢失" }[V.tracking.status]) :
      V.calibration ? "已标定" : "待标定";
    $("frameStatusText").textContent = synthetic ? "SYNTHETIC FRAME" : V.source === "board" ?
      performance.now() - V.boardReceivedAt > 500 ? "BOARD FRAME STALE" : `BOARD / ${V.boardOrigin?.toUpperCase()}` :
      V.source === "video_file" ? "RECORDED VIDEO" : "LIVE CAMERA";
    const result = V.result && now - V.result.receivedAt <= 500 ? V.result : null;
    const text = result?.positions.map((p) => p.relative ? `弦线${p.string_line}·相对格${p.fret_interval}` :
      `${p.string}弦${p.fret}品 ${p.note_name || ""}${p.ambiguous ? "?" : ""}`).join(" · ");
    const audio = window.FretAudioSnapshot?.();
    const feedbackAge = audio ? (audio.age_seconds ?? 0)*1000 + now-(audio.ageReportedAt ?? audio.receivedAt) : Infinity;
    const boardFeedback = !synthetic && audio && !audio.stale &&
      Number.isFinite(feedbackAge) && feedbackAge >= 0 && feedbackAge <= 750 &&
      Number.isFinite(audio.receivedAt) && now >= audio.receivedAt && now-audio.receivedAt <= 750 ?
      audio.visualFeedback : null;
    $("visualEvidence").textContent = synthetic ? `${text || "--"} · SYNTHETIC` :
      boardFeedback || (result?.reason === "hand_outside_measured_lattice" ? "手已检测到，但不在已测品丝范围内" :
      text ? `${text}（位置候选）` : V.tracking?.status === "lost" ? "琴颈跟踪丢失，请重新标定" :
      autoEnabled() ? V.automatic?.status === "tracking" ? "弦与品丝已定位；绝对品号未确定" : "未找到可靠的六弦与品丝结构" :
      V.calibration ? V.detect ? "未检测到指板上的指尖" : "指板已标定，检测未启动" : "尚未标定指板");
    const tipNotes = V.automatic?.fingertip_notes;
    const historical = tipNotes?.status === "historical" && tipNotes.stream_id === PC.stream &&
      tipNotes.generation === PC.generation && now-PC.matchedAt <= 1800;
    if (!text && historical) {
      const fingers = { index: "食指", middle: "中指", ring: "无名指", little: "小指" };
      const positions = tipNotes.projection?.positions || [];
      const description = positions.map(p => `${fingers[p.finger] || p.finger} ` + (p.relative ?
        `弦线${p.string_line}·格${p.fret_interval}` : `${p.string}弦${p.fret}品 ${p.note_name}`) +
        (p.ambiguous ? "?" : "")).join(" · ");
      $("visualEvidence").textContent = description ?
        `历史源帧 #${tipNotes.source_sequence} · ${description}（候选）` :
        "指尖源帧尚无可靠弦品结构，暂不能换算音符";
    }
    $("cameraHint").textContent = synthetic ? V.source === "board" ? "板卡接口合成图；尚无真实视觉验证" :
      "合成图与合成关键点；尚无真实视觉验证" : "位置投影不等于压弦；遮挡、横按与弯音仍需实琴验证";
    const comparison = F.compareEvidence(result, audio, now);
    if (!boardFeedback && comparison.matches.length) {
      $("visualEvidence").textContent += ` · 音频支持 ${comparison.matches.length} 个位置`;
    }
    if (autoEnabled() && !V.boardError && now < V.autoRetryAt) $("visionStatus").textContent =
      "板卡检测较慢，暂时仅预览；稍后自动重试";
    else if (autoEnabled() && !V.boardError && !V.clicks) $("visionStatus").textContent = V.automatic?.status === "tracking" ?
      `${V.automatic.backend === "roi_lk_sobel_projection" ? "区域跟踪" : "自动定位"} · 6 条弦线 · ${V.automatic.frets.length} 条可见品丝 · 绝对品号未确定` :
      V.automatic?.roi_status === "tracking" ? "琴颈跟随中；尚未找到可靠的六弦与品丝，请调整取景与光照" :
      V.automatic?.reason === "roi_tracking_lost" ? "琴颈跟踪已丢失，正在自动重新发现；可框选琴颈缩小搜索范围" :
      V.automatic?.reason === "confirming_lattice" ? "琴颈跟随中，正在确认弦与品丝" :
      V.automatic?.reason === "stale_lattice_geometry" ? "琴颈仍在跟随；旧弦品几何已过期并清除，正在重新测量" :
      V.automatic?.reason === "no_supported_lattice" ? "尚未找到符合六弦与品丝规律的结构" :
      V.source === "camera" && !PC.corners ? "正在自动发现琴颈" :
      "正在寻找六根琴弦和可见品丝；可框选琴颈缩小搜索范围";
    else if (result && !synthetic && !V.boardError && !V.clicks && !["lost", "pending"].includes(V.tracking?.status)) $("visionStatus").textContent =
      `${V.tracking?.status === "tracking" ? "琴颈跟随中" : V.calibration ? "已标定" : "未标定"} · 指尖检测中`;
    if (result?.positions.length && !synthetic) {
      $("visualEvent").hidden = false;
      $("visualEvent").querySelector("strong").textContent = "指尖位置候选";
      $("visualEvent").querySelector("small").textContent = text;
    } else $("visualEvent").hidden = true;
  }
  async function loadModel() {
    if (V.source === "camera") { message("手部模型由板卡运行，电脑只显示结果"); return; }
    if (!V.source || V.source === "synthetic") { message("先连接摄像头或导入视频"); return; }
    if (V.detect) {
      V.detect = false; clearEvidence(); modelButton.textContent = "启动指尖检测"; message("指尖检测已停止"); return;
    }
    if (V.modelLoading) return;
    const session = V.session;
    V.modelLoading = true;
    V.modelError = null;
    modelButton.disabled = true;
    message("正在加载本地手部模型");
    try {
      if (!V.model) {
        const module = await import("./vendor/mediapipe/vision_bundle.mjs");
        const files = await module.FilesetResolver.forVisionTasks("./vendor/mediapipe/wasm");
        V.model = await module.HandLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: "./models/hand_landmarker.task", delegate: "CPU" },
          runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: .65,
          minHandPresenceConfidence: .65, minTrackingConfidence: .65 });
      }
      if (session !== V.session || !V.source) return;
      V.detect = true;
      modelButton.textContent = "停止指尖检测";
      message(V.calibration ? "检测已启动；结果是指尖位置候选" : "手部检测已启动；请标定指板");
    } catch (error) {
      V.modelError = `模型加载失败：${error.message}`;
      V.detect = false; clearEvidence(); message(V.modelError);
    } finally { V.modelLoading = false; modelButton.disabled = false; }
  }
  function calibrate() {
    if (!V.source || V.source === "synthetic") { message("先连接真实摄像头或导入视频，再标定"); return; }
    if (V.source === "board" && (V.boardError || performance.now() - V.boardReceivedAt > 500)) {
      message("请等待板卡画面恢复后再标定"); return;
    }
    V.clickRegion = displayRegion();
    if (V.source === "camera") {
      PC.generation += 1; PC.corners = null; PC.seedPending = false; PC.sequence = -1;
    }
    V.selectingRoi = false; V.roiError = null;
    $("autoFretboard").checked = false; V.automatic = null;
    resetTracking();
    V.clicks = [];
    V.calibration = null;
    stage.dataset.calibrating = "true";
    clearEvidence();
    message(`标定时保持琴颈不动，点选 ${cornerLabels[0]}（起点 0 表示琴枕）`);
  }
  $("selectNeckButton").addEventListener("click", () => {
    if (V.source === "camera") {
      if (!$("pcPlEnabled").checked || video.readyState < 2) {
        message("先连接电脑摄像头并启用板卡 PL 处理"); return;
      }
      PC.generation += 1; PC.corners = null; PC.seedPending = false; PC.sequence = -1;
      pcSeed.width = 640; pcSeed.height = 480;
      pcSeed.getContext("2d").drawImage(video, 0, 0, 640, 480);
    } else if (V.source !== "board" || V.boardError || performance.now() - V.boardReceivedAt > 500) {
      message("先连接摄像头，再框选琴颈"); return;
    }
    resetTracking(); V.calibration = null; V.automatic = null; V.autoRetryAt = 0;
    $("autoFretboard").checked = false;
    V.selectingRoi = true; V.roiError = null; V.clickRegion = fullRegion; V.clicks = [];
    stage.dataset.calibrating = "true"; clearEvidence();
    message("保持琴颈不动，依次点选指板四角：6弦起点、6弦终点、1弦终点、1弦起点");
  });
  overlay.addEventListener("pointerdown", async (event) => {
    if (!V.clicks) return;
    const rect = overlay.getBoundingClientRect(), box = frameBox();
    const x = (event.clientX - rect.left) * overlay.width / rect.width;
    const y = (event.clientY - rect.top) * overlay.height / rect.height;
    if (x < box.x || x > box.x + box.width || y < box.y || y > box.y + box.height) { message("请点在视频画面内部"); return; }
    const point = { x: box.region.x + (x - box.x) / box.width * box.region.width,
      y: box.region.y + (y - box.y) / box.height * box.region.height };
    V.clicks.push(point);
    if (V.clicks.length < 4) { message(`点选 ${cornerLabels[V.clicks.length]}`); return; }
    if (V.selectingRoi) {
      const corners = V.clicks, session = V.session;
      V.clicks = null; V.selectingRoi = false; stage.dataset.calibrating = "false";
      message("正在锁定琴颈区域");
      if (V.source === "camera") {
        PC.generation += 1; PC.corners = corners; PC.seedPending = true; PC.sequence = -1;
        $("autoFretboard").checked = true;
        message("琴颈区域已提交，板卡正在建立跟踪");
        return;
      }
      try {
        const response = await fetch("/api/vision/roi", { method: "POST",
          headers: { "Content-Type": "application/json" }, signal: AbortSignal.timeout(5000),
          body: JSON.stringify({ corners, sequence: V.boardSequence, stream_id: V.boardStream }) });
        const result = await response.json();
        if (session !== V.session) return;
        if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
        $("autoFretboard").checked = true; V.autoRetryAt = 0; V.roiError = null;
        message("琴颈区域已提交，正在跟踪并检测弦与品丝");
      } catch (error) {
        if (session === V.session) { V.roiError = error.message; message(`框选失败：${error.message}`); }
      }
      return;
    }
    try {
      V.calibration = F.createCalibration(V.clicks, Number($("firstFret").value), Number($("lastFret").value));
      V.clicks = null; stage.dataset.calibrating = "false"; saveCalibration(); clearEvidence();
      if (followsNeck()) startTracking(V.calibration);
      else message("固定标定完成；移动相机或琴颈后需要重新标定");
    } catch (error) { V.clicks = []; message(`${error.message}；重新点选 ${cornerLabels[0]}`); }
  });
  $("calibrateButton").addEventListener("click", calibrate);
  $("focusFretboard").addEventListener("change", () => { clearPreviews(); render(); });
  $("preprocessDetails").addEventListener("toggle", () => { clearPreviews(); render(); });
  $("autoFretboard").addEventListener("change", () => {
    if (V.source === "camera") { PC.generation += 1; PC.corners = null; PC.sequence = -1; PC.seedPending = false; }
    V.selectingRoi = false; V.roiError = null;
    resetTracking(); V.automatic = null; V.autoRetryAt = 0; V.calibration = null; V.clicks = null;
    stage.dataset.calibrating = "false"; clearEvidence();
    message($("autoFretboard").checked ? "自动定位已启用" : "可在标定设置中手动标定");
  });
  $("visionStopButton").addEventListener("click", () => { window.FretStopCamera?.(); stop(); });
  $("clearCalibrationButton").addEventListener("click", () => {
    if (V.source === "camera") { PC.generation += 1; PC.corners = null; PC.sequence = -1; PC.seedPending = false; V.automatic = null; }
    resetTracking();
    V.calibration = null; V.clicks = null; stage.dataset.calibrating = "false"; clearEvidence();
    try { localStorage.removeItem("fretlens-calibration-v1"); } catch (_) { /* Local storage is optional. */ }
    message("标定已清除");
  });
  for (const id of ["firstFret", "lastFret"]) $(id).addEventListener("change", () => {
    resetTracking(); V.clicks = null; stage.dataset.calibrating = "false";
    V.calibration = null; clearEvidence(); message("品位范围已改变，请重新标定");
  });
  $("applyNoteCalibration").addEventListener("click", () => {
    const a = V.automatic, first = Number($("noteFirstBoundary").value);
    const reversed = $("noteFretDirection").value === "descending";
    if (!a || a.status !== "tracking" || a.strings?.length !== 6 || a.frets?.length < 5 ||
        !Number.isInteger(a.lattice_epoch) || !Number.isInteger(first) || first < 0 || first > 24 ||
        $("noteFirstBoundary").value.trim() === "" ||
        first + (reversed ? 1-a.frets.length : a.frets.length-1) < 0 ||
        first + (reversed ? 1-a.frets.length : a.frets.length-1) > 24) {
      $("noteCalibrationStatus").textContent = "请先定位六弦与连续品丝，并填写有效的首条品丝品号";
      return;
    }
    noteCalibration = { stream_id: a.stream_id, generation: a.generation, lattice_epoch: a.lattice_epoch,
      first_boundary_fret: first, strings_reversed: $("noteTopString").value === "1",
      frets_reversed: reversed, tuning: [40, 45, 50, 55, 59, 64] };
    $("noteCalibrationStatus").textContent = "音符标定已应用 · 标准调弦 E A D G B E";
  });
  const clearNotes = () => {
    noteCalibration = null;
    $("noteCalibrationStatus").textContent = "音符未标定 · 标准调弦 E A D G B E";
  };
  $("clearNoteCalibration").addEventListener("click", clearNotes);
  for (const id of ["noteFirstBoundary", "noteTopString", "noteFretDirection"])
    $(id).addEventListener("change", clearNotes);
  $("followNeck").addEventListener("change", () => {
    resetTracking(); V.calibration = null; V.clicks = null; stage.dataset.calibrating = "false";
    clearEvidence(); message("标定方式已改变，请重新标定");
  });
  modelButton.addEventListener("click", loadModel);
  $("visionTestButton").addEventListener("click", () => {
    window.FretStopCamera?.(); stop();
    const calibration = F.createCalibration([{ x: .08, y: .28 }, { x: .91, y: .2 }, { x: .92, y: .74 }, { x: .1, y: .8 }]);
    $("firstFret").value = 0; $("lastFret").value = 12;
    begin("synthetic", { width: 960, height: 540 }, calibration);
    message("SYNTHETIC：只验证指板坐标换算，不是真实手部识别");
  });
  function mjpegHeader(headers, name) {
    const match = headers.match(new RegExp(`^${name}:\\s*(.+)$`, "im"));
    return match?.[1]?.trim() || null;
  }
  function mjpegDelimiter(bytes) {
    for (let i = 0; i <= bytes.length - 4; i++) {
      if (bytes[i] === 13 && bytes[i + 1] === 10 && bytes[i + 2] === 13 && bytes[i + 3] === 10)
        return i;
    }
    return -1;
  }
  async function startBoardMjpeg(session) {
    if (session !== V.session) return;
    const controller = new AbortController();
    V.boardRequest?.abort();
    V.boardRequest = controller;
    const decoder = new TextDecoder("ascii");
    let receivedFrame = false;
    let lastFrameAt = performance.now();
    let stalled = false;
    let rejectStall;
    const stallFailure = new Promise((_, reject) => { rejectStall = reject; });
    // A dead UVC source can leave the HTTP response and its ReadableStream
    // open forever. Abort the stream when no complete JPEG arrives so the
    // normal reconnect path can restore the camera without a page reload.
    const stallWatchdog = setInterval(() => {
      if (performance.now() - lastFrameAt < 4000) return;
      stalled = true;
      V.boardStreamActive = false;
      rejectStall(new Error("板卡视频流超过 4 秒没有新帧"));
      controller.abort();
    }, 500);
    try {
      const response = await fetch("/vision/stream.mjpeg", {
        cache: "no-store", signal: controller.signal
      });
      if (!response.ok || !response.body) throw new Error(`板卡视频流 HTTP ${response.status}`);
      V.boardStreamActive = true;
      const reader = response.body.getReader();
      let pending = new Uint8Array(0);
      while (session === V.session && !controller.signal.aborted) {
        const { value, done } = await Promise.race([reader.read(), stallFailure]);
        if (done) throw new Error("板卡视频流已结束");
        if (!value?.length) continue;
        const joined = new Uint8Array(pending.length + value.length);
        joined.set(pending); joined.set(value, pending.length); pending = joined;
        while (session === V.session && !controller.signal.aborted) {
          const headerEnd = mjpegDelimiter(pending);
          if (headerEnd < 0) break;
          const headers = decoder.decode(pending.subarray(0, headerEnd));
          const length = Number(mjpegHeader(headers, "Content-Length"));
          if (!Number.isSafeInteger(length) || length < 4) {
            pending = pending.slice(headerEnd + 4);
            continue;
          }
          const frameStart = headerEnd + 4;
          if (pending.length < frameStart + length + 2) break;
          const jpeg = pending.slice(frameStart, frameStart + length);
          pending = pending.slice(frameStart + length + 2);
          const sequence = Number(mjpegHeader(headers, "X-Vision-Sequence"));
          const capturedMs = Number(mjpegHeader(headers, "X-Vision-Captured-Ms"));
          const origin = mjpegHeader(headers, "X-Vision-Source");
          const stream = mjpegHeader(headers, "X-Vision-Stream");
          if (!Number.isSafeInteger(sequence) || !stream || !["camera", "video_file", "synthetic"].includes(origin))
            continue;
          if (stream === V.boardStream && sequence <= V.boardSequence) continue;
          const bitmap = await createImageBitmap(new Blob([jpeg], { type: "image/jpeg" }));
          if (session !== V.session || controller.signal.aborted) { bitmap.close(); return; }
          if (boardFrame.width !== bitmap.width) boardFrame.width = bitmap.width;
          if (boardFrame.height !== bitmap.height) boardFrame.height = bitmap.height;
          boardFrame.getContext("2d").drawImage(bitmap, 0, 0); bitmap.close();
          receivedFrame = true;
          lastFrameAt = performance.now();
          const first = V.source !== "board";
          const restarted = !first && V.boardStream !== null && stream !== V.boardStream;
          if (restarted) {
            V.automatic = null;
            resetTracking(); V.calibration = null; V.clicks = null; stage.dataset.calibrating = "false";
            V.lastTime = -1; V.lastFrame = -1; V.processedToken = null; clearEvidence();
          }
          V.boardSequence = sequence; V.boardOrigin = origin;
          V.boardReceivedAt = performance.now(); V.boardStream = stream;
          V.boardCapturedMs = Number.isFinite(capturedMs) ? capturedMs : null;
          V.boardError = null;
          if (origin === "camera") PC.seedPending = true;
          if (first || restarted) {
            begin("board", { width: boardFrame.width, height: boardFrame.height }, null,
              `board_${origin}`, true);
            if (origin === "camera") startPcPl(session);
            else startPlPoll(session);
            message(origin === "synthetic" ? "板卡接口合成图；不是实际摄像头输入" :
              "C922 视频流已连接，正在自动定位");
          } else {
            V.shape = { width: boardFrame.width, height: boardFrame.height };
          }
        }
      }
    } catch (error) {
      V.boardStreamActive = false;
      if (session === V.session && (stalled || !controller.signal.aborted)) {
        const reason = stalled ? "板卡视频流超过 4 秒没有新帧" : error.message;
        V.boardError = reason;
        message(`板卡视频流中断，正在自动重连：${reason}`);
        // Before the first streamed frame, the initial JPEG request remains
        // the fallback and must be allowed to continue undisturbed.
        if (receivedFrame || stalled) {
          clearTimeout(V.boardTimer);
          V.boardTimer = setTimeout(() => startBoardMjpeg(session), 1000);
        }
      }
    } finally {
      clearInterval(stallWatchdog);
      if (V.boardRequest === controller) V.boardRequest = null;
    }
  }
  $("boardVisionButton").addEventListener("click", async () => {
    window.FretStopCamera?.(); stop();
    const session = V.session;
    const fetchFrame = async () => {
      if (session !== V.session) return;
      const started = performance.now();
      let retryDelay = 0;
      const controller = new AbortController(); V.boardRequest = controller;
      const timeout = setTimeout(() => controller.abort(), 3000);
      try {
        const trackingId = V.tracking?.tracker_id;
        // Live board frames are analysed by startPcPl. The camera server's
        // legacy whole-frame search competes with that tracker on the same PS.
        const bridgeCamera = V.boardOrigin === "camera";
        const automatic = V.boardOrigin !== null && !bridgeCamera && $("autoFretboard").checked && performance.now() >= V.autoRetryAt;
        const response = await fetch("/vision/frame.jpg" + (automatic ? "?auto=1" : trackingId ? `?tracking=${encodeURIComponent(trackingId)}` : ""),
          { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(`板卡视频 HTTP ${response.status}`);
        const sequence = Number(response.headers.get("X-Vision-Sequence"));
        const origin = response.headers.get("X-Vision-Source");
        const stream = response.headers.get("X-Vision-Stream");
        const capturedMs = Number(response.headers.get("X-Vision-Captured-Ms"));
        let tracking = null;
        try { tracking = JSON.parse(response.headers.get("X-Vision-Tracking")); } catch (_) { /* Invalid tracking fails closed. */ }
        let fretboard = null;
        try { fretboard = JSON.parse(response.headers.get("X-Vision-Fretboard")); } catch (_) { /* Unsupported frames have no geometry. */ }
        if (!fretboard && tracking?.mode === "auto") fretboard = tracking;
        if (!response.headers.has("X-Vision-Sequence") || !Number.isSafeInteger(sequence) || sequence < 0 ||
          !["camera", "video_file", "synthetic"].includes(origin)) throw new Error("板卡视频来源或序号无效");
        if (V.source === "board" && sequence <= V.boardSequence && stream === V.boardStream) {
          if (sequence === V.boardSequence && automatic && $("autoFretboard").checked && fretboard?.status === "tracking" &&
              performance.now() - V.boardReceivedAt <= 500) applyAuto(fretboard, sequence, stream);
          return;
        }
        const blob = await response.blob();
        V.requestMs = performance.now() - started;
        const decodeStarted = performance.now();
        const bitmap = await createImageBitmap(blob);
        V.decodeMs = performance.now() - decodeStarted;
        if (session !== V.session) { bitmap.close(); return; }
        if (boardFrame.width !== bitmap.width) boardFrame.width = bitmap.width;
        if (boardFrame.height !== bitmap.height) boardFrame.height = bitmap.height;
        boardFrame.getContext("2d").drawImage(bitmap, 0, 0); bitmap.close();
        const first = V.source !== "board";
        const restarted = !first && stream !== V.boardStream;
        if (restarted) {
          V.automatic = null;
          resetTracking(); V.calibration = null; V.clicks = null; stage.dataset.calibrating = "false";
          V.lastTime = -1; V.lastFrame = -1;
          V.processedToken = null; clearEvidence();
        }
        if (!first && (V.shape.width !== boardFrame.width || V.shape.height !== boardFrame.height || origin !== V.boardOrigin)) {
          resetTracking(); V.clicks = null; stage.dataset.calibrating = "false";
          $("followNeck").disabled = origin !== "camera";
          V.calibration = null; clearEvidence(); message("板卡取景或来源改变，请重新标定");
        }
        V.boardSequence = sequence; V.boardOrigin = origin; V.boardReceivedAt = performance.now();
        V.boardStream = stream;
        const recovered = V.boardError !== null;
        V.boardError = null;
        if (origin === "camera") PC.seedPending = true;
        V.boardCapturedMs = response.headers.has("X-Vision-Captured-Ms") && Number.isFinite(capturedMs) ? capturedMs : null;
        if (automatic && $("autoFretboard").checked && fretboard) applyAuto(fretboard, sequence, stream);
        if (trackingId && trackingId === V.tracking?.tracker_id) applyTracking(tracking, sequence, stream);
        if (first) {
          begin("board", { width: boardFrame.width, height: boardFrame.height }, null, `board_${origin}`, true);
          if (origin === "camera") startPcPl(session);
          else startPlPoll(session);
          message(origin === "synthetic" ? "板卡接口合成图；不是实际摄像头输入" :
            autoEnabled() ? "板卡图像已连接，正在自动定位" : "板卡图像已连接，请标定指板");
        } else {
          V.shape = { width: boardFrame.width, height: boardFrame.height };
          if (recovered || restarted) message("板卡视频已恢复" + (followsNeck() ? "，请重新标定琴颈" : ""));
        }
      } catch (error) {
        retryDelay = 1000;
        if (session === V.session) {
          V.boardError = error.message;
          clearAuto();
          V.clicks = null; stage.dataset.calibrating = "false";
          loseTracking("video_interrupted");
          clearEvidence(); message(`板卡视频中断，正在自动重连：${error.message}`);
        }
      } finally {
        clearTimeout(timeout);
        if (session === V.session && !V.boardStreamActive) V.boardTimer = setTimeout(fetchFrame,
          Math.max(retryDelay, boardFramePeriod - (performance.now() - started), 0));
      }
    };
    message("正在连接板卡视频");
    startBoardMjpeg(session);
    fetchFrame();
  });
  $("visionFile").addEventListener("change", async (event) => {
    const file = event.target.files?.[0]; if (!file) return;
    window.FretStopCamera?.(); stop();
    V.url = URL.createObjectURL(file);
    video.srcObject = null; video.src = V.url; video.loop = true;
    try {
      await video.play();
      begin("video_file", { width: video.videoWidth, height: video.videoHeight }, null, `${file.name}:${file.size}:${file.lastModified}`);
      message("录像已加载；先标定指板，再启动检测");
    } catch (error) { stop(); message(`视频打开失败：${error.message}`); }
    event.target.value = "";
  });
  window.addEventListener("beforeunload", () => { stop(); V.model?.close(); });
  window.FretVision = { get active() { return Boolean(V.source); }, render, attachCamera, stop,
    metrics() { return { frames: V.frameCount, display_fps: V.displayFps, inference_fps: V.inferenceFps,
      hand_inferences: V.inferenceCount, hand_detection_enabled: V.detect,
      model_loading: V.modelLoading, model_error: V.modelError, pl_status: PL.status,
      request_ms: V.requestMs, decode_ms: V.decodeMs,
      board_neural_inference_ms: V.handObservation?.detection?.last_inference_ms ?? null,
      board_neural_completed: V.handObservation?.detection?.detector_completed ?? null,
      pl_processing_ms: PL.frame?.processingMs ?? null,
      pl_roundtrip_ms: PL.frame?.roundtripMs ?? null,
      pl_updates_per_second: PL.frame?.fps ?? null,
      pl_result_age_ms: PL.frame ? plAge(PL.frame) : null }; },
    trackingSnapshot() { return { tracking: V.tracking, calibration: V.calibration, automatic: V.automatic,
      display_region: handRegionCorners(), display_corners: displayCorners() }; },
    snapshot() { return V.result; } };
  $("pcPlEnabled").addEventListener("change", () => {
    if (V.source !== "camera") return;
    displayTrack = null;
    PC.generation += 1; PC.corners = null; PC.seedPending = false; PC.sequence = -1; V.automatic = null;
    stopPl();
    if ($("pcPlEnabled").checked) startPcPl(V.session);
  });
  const cameraEntry = new URLSearchParams(window.location.search).get("camera");
  if (cameraEntry === "pc") {
    $("pcPlEnabled").checked = true;
    $("autoFretboard").checked = true;
    $("cameraButton").click();
  } else if (cameraEntry === "board") {
    $("autoFretboard").checked = true;
    $("boardVisionButton").click();
  }
})();
