/* Geometry and evidence rules shared by live vision and offline tests. */
(function (root) {
  "use strict";
  const TUNING = [40, 45, 50, 55, 59, 64]; // String 6 through string 1.
  const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const finitePoint = (p) => p && Number.isFinite(p.x) && Number.isFinite(p.y);
  const noteName = (midi) => `${NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;

  function solve(matrix, target) {
    const a = matrix.map((row, i) => [...row, target[i]]);
    for (let column = 0; column < target.length; column += 1) {
      let pivot = column;
      for (let row = column + 1; row < a.length; row += 1) {
        if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
      }
      if (Math.abs(a[pivot][column]) < 1e-10) throw new Error("指板四点退化，请重新标定");
      [a[column], a[pivot]] = [a[pivot], a[column]];
      const divisor = a[column][column];
      a[column] = a[column].map((value) => value / divisor);
      for (let row = 0; row < a.length; row += 1) {
        if (row === column) continue;
        const factor = a[row][column];
        a[row] = a[row].map((value, i) => value - factor * a[column][i]);
      }
    }
    return a.map((row) => row[row.length - 1]);
  }

  function homography(from, to) {
    const rows = [], target = [];
    from.forEach((p, i) => {
      const q = to[i];
      rows.push([p.x, p.y, 1, 0, 0, 0, -q.x * p.x, -q.x * p.y]);
      target.push(q.x);
      rows.push([0, 0, 0, p.x, p.y, 1, -q.y * p.x, -q.y * p.y]);
      target.push(q.y);
    });
    return [...solve(rows, target), 1];
  }

  function transform(h, p) {
    const denominator = h[6] * p.x + h[7] * p.y + h[8];
    if (Math.abs(denominator) < 1e-9) throw new Error("点位位于无效透视区域");
    return { x: (h[0] * p.x + h[1] * p.y + h[2]) / denominator,
      y: (h[3] * p.x + h[4] * p.y + h[5]) / denominator };
  }

  function createCalibration(corners, firstFret = 0, lastFret = 12, tuning = TUNING) {
    if (!Array.isArray(corners) || corners.length !== 4 || corners.some((p) =>
      !finitePoint(p) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) throw new Error("需要画面内的四个指板标定点");
    if (!Number.isInteger(firstFret) || !Number.isInteger(lastFret) || firstFret < 0 ||
      lastFret > 24 || lastFret <= firstFret) throw new Error("结束品必须大于起始边界品，范围 0 到 24");
    if (!Array.isArray(tuning) || tuning.length !== 6 || tuning.some((midi) => !Number.isInteger(midi) || midi < 0 || midi + lastFret > 127)) {
      throw new Error("需要六根弦的有效空弦 MIDI 音高");
    }
    let orientation = 0, area = 0;
    corners.forEach((p, i) => {
      const q = corners[(i + 1) % 4], r = corners[(i + 2) % 4];
      const cross = (q.x - p.x) * (r.y - q.y) - (q.y - p.y) * (r.x - q.x);
      if (Math.abs(cross) < 1e-5 || (orientation && Math.sign(cross) !== orientation)) throw new Error("四点需依次围成指板，不能交叉或共线");
      orientation = Math.sign(cross);
      area += p.x * q.y - q.x * p.y;
    });
    if (Math.abs(area) < .004) throw new Error("指板区域过小，请靠近取景后标定");
    const rectangle = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }];
    const start = 1 - 2 ** (-firstFret / 12), end = 1 - 2 ** (-lastFret / 12);
    const boundaries = [];
    for (let fret = firstFret; fret <= lastFret; fret += 1) {
      boundaries.push({ fret, x: ((1 - 2 ** (-fret / 12)) - start) / (end - start) });
    }
    return { version: 1, corners: corners.map((p) => ({ x: p.x, y: p.y })), firstFret, lastFret,
      tuning: [...tuning], boundaries, toBoard: homography(corners, rectangle),
      toImage: homography(rectangle, corners) };
  }

  function locateTip(calibration, tip) {
    if (!calibration || !finitePoint(tip)) return [];
    const p = transform(calibration.toBoard, tip);
    if (p.x <= 0 || p.x >= 1 || p.y < -.03 || p.y > 1.03) return [];
    const boundaries = calibration.boundaries;
    const index = boundaries.findIndex((boundary) => boundary.x > p.x);
    if (index < 1) return [];
    const previous = boundaries[index - 1], next = boundaries[index];
    const within = (p.x - previous.x) / (next.x - previous.x);
    // A fingertip on a fret wire cannot establish which fret was pressed.
    const ambiguousFret = within < .07 || within > .93;
    const candidates = [];
    for (let string = 0; string < 6; string += 1) {
      const distance = Math.abs(p.y * 5 - string);
      if (distance > .55) continue;
      const midi = calibration.tuning[string] + next.fret;
      candidates.push({ string: 6 - string, fret: next.fret, midi_note: midi,
        note_name: noteName(midi), distance: Number(distance.toFixed(4)),
        ambiguous: ambiguousFret || distance > .35, x: tip.x, y: tip.y,
        evidence: "fingertip_projection", pressed: null, sounding: null });
    }
    return candidates;
  }

  function mapHands(calibration, hands, minimumScore = .65) {
    if (!calibration) return { status: "uncalibrated", positions: [], hands: hands || [] };
    const positions = [];
    for (const [handIndex, hand] of (hands || []).entries()) {
      if (!Number.isFinite(hand.score) || hand.score < minimumScore || !Array.isArray(hand.landmarks) || hand.landmarks.length !== 21) continue;
      for (const [name, index] of [["index", 8], ["middle", 12], ["ring", 16], ["pinky", 20]]) {
        for (const position of locateTip(calibration, hand.landmarks[index])) {
          positions.push({ ...position, finger: name, hand_index: handIndex,
            quality_gate_score: hand.score, score_source: hand.score_source || "provided_by_input" });
        }
      }
    }
    return { status: positions.length ? "candidates" : "no_fingertips_on_board", positions, hands: hands || [] };
  }

  function createObservedGeometry(result) {
    if (result?.status !== "tracking" || !Array.isArray(result.strings) || result.strings.length !== 6 ||
        !Array.isArray(result.frets) || result.frets.length < 5 || result.frets.length > 25) {
      throw new Error("未找到可靠弦与品丝结构");
    }
    const base = createCalibration(result.corners);
    const tracked = result.stage === "tracked_lattice" && Number.isFinite(result.lattice_age_ms) &&
      result.lattice_age_ms >= 0 && result.lattice_age_ms <= 600 &&
      Number.isFinite(result.flow?.inlier_ratio) && result.flow.inlier_ratio >= .7;
    const estimatedSources = new Set([
      "live_neck_width_six_equal_lanes", "tracked_string_lattice",
      "tracked_fret_lattice"
    ]);
    const lines = (items, allowEstimated = false) => items.map(item => {
      const estimated = allowEstimated && item.estimated === true &&
        estimatedSources.has(item.source);
      if (!(item.observed === true || estimated ||
            item.observed === false && tracked) || !finitePoint(item.start) || !finitePoint(item.end) ||
          [item.start, item.end].some(p => p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) {
        throw new Error("线条缺少图像支持");
      }
      return { start: transform(base.toBoard, item.start), end: transform(base.toBoard, item.end),
        observed: item.observed === true, estimated };
    });
    const strings = lines(result.strings, true), frets = lines(result.frets, true);
    const mid = line => ({ x: (line.start.x + line.end.x) / 2, y: (line.start.y + line.end.y) / 2 });
    strings.sort((a, b) => mid(a).y - mid(b).y);
    frets.sort((a, b) => mid(a).x - mid(b).x);
    return { ...base, strings, frets, absolute_frets_known: false, tracked };
  }

  function mapObservedHands(geometry, hands, minimumScore = .65) {
    if (!geometry) return { status: "unlocalized", positions: [], hands: hands || [] };
    const positions = [];
    const at = (line, axis, coordinate) => {
      const other = axis === "x" ? "y" : "x";
      const span = line.end[other] - line.start[other];
      if (Math.abs(span) < 1e-6) return NaN;
      return line.start[axis] + (coordinate - line.start[other]) *
        (line.end[axis] - line.start[axis]) / span;
    };
    for (const [handIndex, hand] of (hands || []).entries()) {
      if (!Number.isFinite(hand.score) || hand.score < minimumScore || hand.landmarks?.length !== 21) continue;
      for (const [finger, index] of [["index", 8], ["middle", 12], ["ring", 16], ["pinky", 20]]) {
        const tip = hand.landmarks[index];
        if (!finitePoint(tip)) continue;
        const p = transform(geometry.toBoard, tip);
        if (p.x <= 0 || p.x >= 1 || p.y < -.03 || p.y > 1.03) continue;
        const boundaries = geometry.frets.map(line => at(line, "x", p.y));
        const next = boundaries.findIndex(x => x > p.x);
        if (next < 1) continue;
        const within = (p.x - boundaries[next - 1]) / (boundaries[next] - boundaries[next - 1]);
        const ys = geometry.strings.map(line => at(line, "y", p.x));
        const spacing = (ys[5] - ys[0]) / 5;
        if (!(spacing > .02)) continue;
        ys.forEach((y, string) => {
          const distance = Math.abs(p.y - y) / spacing;
          if (distance <= .55) positions.push({ x: tip.x, y: tip.y, finger, hand_index: handIndex,
            string_line: string + 1, fret_interval: next, relative: true,
            ambiguous: distance > .35 || within < .07 || within > .93,
            pressed: null, sounding: null, evidence: geometry.tracked ? "tracked_lines_projection" : "observed_lines_projection" });
        });
      }
    }
    return { status: positions.length ? "relative_positions" : "no_fingertips_on_board", positions, hands: hands || [] };
  }

  function compareEvidence(vision, audio, now, maximumGapMs = 750) {
    const empty = (status) => ({ status, matches: [], unmatched: [], synchronized: false,
      limitation: "位置投影不证明压弦或发声；时间采用浏览器接收时刻，尚无硬件同步。" });
    if (!vision || !Number.isFinite(vision.receivedAt) || now - vision.receivedAt > 500 || vision.receivedAt > now) return empty("vision_unavailable");
    if (["video_file", "board_video_file"].includes(vision.source)) return empty("recording_not_synchronized");
    if (vision.source !== "camera" && vision.source !== "board_camera") return empty("demo_only");
    if (!audio || audio.stale || audio.status !== "recognized" || audio.testMode !== 0 ||
      !Number.isFinite(audio.receivedAt) || audio.receivedAt > now ||
      (audio.age_seconds !== undefined && (!Number.isFinite(audio.age_seconds) || audio.age_seconds < 0 ||
        !Number.isFinite(audio.ageReportedAt ?? audio.receivedAt) || (audio.ageReportedAt ?? audio.receivedAt) > now ||
        audio.age_seconds * 1000 + now - (audio.ageReportedAt ?? audio.receivedAt) > maximumGapMs)) ||
      now - audio.receivedAt > maximumGapMs || Math.abs(audio.receivedAt - vision.receivedAt) > maximumGapMs) return empty("audio_unavailable");
    const positions = vision.positions || [];
    const notes = audio.notes || [];
    const matches = [], unmatched = [];
    for (const position of positions) {
      if (position.ambiguous || position.relative || !Number.isInteger(position.midi_note)) continue;
      const note = notes.find((item) => !item.ambiguous && Number.isInteger(item.midi) && item.midi === position.midi_note);
      if (note) matches.push(position);
      else unmatched.push(position);
    }
    return { ...empty(matches.length ? "supported_positions" : positions.length ? "no_matching_pitch" : "no_visual_positions"),
      matches, unmatched, synchronized: false, reception_gap_ms: Math.abs(audio.receivedAt - vision.receivedAt) };
  }

  root.Fretboard = { TUNING, createCalibration, transform, locateTip, mapHands, compareEvidence,
    createObservedGeometry, mapObservedHands, noteName };
})(typeof globalThis !== "undefined" ? globalThis : this);
