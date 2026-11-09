/* Standard-tuning references and conservative, reception-time evidence fusion. */
(function (root) {
  "use strict";

  const TUNING = Object.freeze([40, 45, 50, 55, 59, 64]); // Strings 6 through 1.
  const NAMES = Object.freeze(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]);
  const NATURAL = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
  const CHORD_TYPES = Object.freeze(Object.fromEntries(Object.entries({
    major: [0, 4, 7], minor: [0, 3, 7], dominant_seventh: [0, 4, 7, 10],
    major_seventh: [0, 4, 7, 11], minor_seventh: [0, 3, 7, 10]
  }).map(([key, values]) => [key, Object.freeze(values)])));
  const SUFFIX = { major: "", minor: "m", dominant_seventh: "7", major_seventh: "maj7", minor_seventh: "m7" };
  const MODES = Object.freeze(Object.fromEntries(Object.entries({
    ionian: [0, 2, 4, 5, 7, 9, 11], dorian: [0, 2, 3, 5, 7, 9, 10],
    phrygian: [0, 1, 3, 5, 7, 8, 10], lydian: [0, 2, 4, 6, 7, 9, 11],
    mixolydian: [0, 2, 4, 5, 7, 9, 10], aeolian: [0, 2, 3, 5, 7, 8, 10],
    locrian: [0, 1, 3, 5, 6, 8, 10], major_pentatonic: [0, 2, 4, 7, 9],
    minor_pentatonic: [0, 3, 5, 7, 10]
  }).map(([key, values]) => [key, Object.freeze(values)])));
  const LIMITATION = "Fingertip projections and matching audio support candidate positions; they do not prove a pressed or sounding string. Unobserved strings remain unknown. Reception times are not hardware synchronization.";
  const mod12 = value => ((value % 12) + 12) % 12;
  const noteName = midi => `${NAMES[mod12(midi)]}${Math.floor(midi / 12) - 1}`;
  const escapeXml = value => String(value).replace(/[&<>"']/g, character =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" })[character]);

  function parseTonic(value) {
    if (typeof value !== "string") return null;
    const match = /^([A-Ga-g])([#b\u266f\u266d]?)$/.exec(value.trim());
    if (!match) return null;
    const letter = match[1].toUpperCase();
    const accidental = match[2].replace(/\u266f/g, "#").replace(/\u266d/g, "b");
    return { name: letter + accidental, letter,
      pitchClass: mod12(NATURAL[letter] + (accidental === "#" ? 1 : accidental === "b" ? -1 : 0)) };
  }

  function parseChordLabel(value) {
    if (typeof value !== "string") return null;
    const parts = value.trim().split("/");
    if (parts.length > 2) return null;
    const match = /^([A-Ga-g][#b\u266f\u266d]?)(.*)$/.exec(parts[0]);
    if (!match) return null;
    const tonic = parseTonic(match[1]);
    const token = match[2].trim().replace(/[\s_-]+/g, "");
    const aliases = {
      "": "major", maj: "major", major: "major", m: "minor", min: "minor", minor: "minor",
      "7": "dominant_seventh", dominant7: "dominant_seventh", dominantseventh: "dominant_seventh",
      maj7: "major_seventh", major7: "major_seventh", majorseventh: "major_seventh",
      m7: "minor_seventh", min7: "minor_seventh", minor7: "minor_seventh", minorseventh: "minor_seventh"
    };
    const key = token === "M" ? "maj" : token === "M7" ? "maj7" : token.toLowerCase();
    if (!Object.hasOwn(aliases, key)) return null;
    const quality = aliases[key];
    const bass = parts.length === 2 ? parseTonic(parts[1]) : tonic;
    if (!bass) return null;
    return { root: tonic.name, rootPc: tonic.pitchClass, quality, bass: bass.name,
      bassPc: bass.pitchClass, label: tonic.name + SUFFIX[quality] +
        (bass.pitchClass !== tonic.pitchClass ? `/${bass.name}` : ""),
      intervals: [...CHORD_TYPES[quality]],
      pitchClasses: CHORD_TYPES[quality].map(interval => mod12(tonic.pitchClass + interval)) };
  }

  // null means intentionally muted in a reference only, never an observed mute.
  const OPEN = [
    ["C", [null, 3, 2, 0, 1, 0], [null, 3, 2, 0, 1, 0]],
    ["D", [null, null, 0, 2, 3, 2], [null, null, 0, 1, 3, 2]],
    ["E", [0, 2, 2, 1, 0, 0], [0, 2, 3, 1, 0, 0]],
    ["G", [3, 2, 0, 0, 0, 3], [2, 1, 0, 0, 0, 3]],
    ["A", [null, 0, 2, 2, 2, 0], [null, 0, 1, 2, 3, 0]],
    ["Am", [null, 0, 2, 2, 1, 0], [null, 0, 2, 3, 1, 0]],
    ["Dm", [null, null, 0, 2, 3, 1], [null, null, 0, 2, 3, 1]],
    ["Em", [0, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0]],
    ["C7", [null, 3, 2, 3, 1, 0], [null, 3, 2, 4, 1, 0], [7]],
    ["D7", [null, null, 0, 2, 1, 2], [null, null, 0, 2, 1, 3]],
    ["E7", [0, 2, 0, 1, 0, 0], [0, 2, 0, 1, 0, 0]],
    ["G7", [3, 2, 0, 0, 0, 1], [3, 2, 0, 0, 0, 1]],
    ["A7", [null, 0, 2, 0, 2, 0], [null, 0, 2, 0, 3, 0]],
    ["B7", [null, 2, 1, 2, 0, 2], [null, 2, 1, 3, 0, 4]],
    ["Cmaj7", [null, 3, 2, 0, 0, 0], [null, 3, 2, 0, 0, 0]],
    ["Dmaj7", [null, null, 0, 2, 2, 2], [null, null, 0, 1, 1, 1], [],
      [{ fret: 2, fromString: 3, toString: 1, finger: 1 }]],
    ["Emaj7", [0, 2, 1, 1, 0, 0], [0, 3, 1, 2, 0, 0]],
    ["Fmaj7", [null, null, 3, 2, 1, 0], [null, null, 3, 2, 1, 0]],
    ["Gmaj7", [3, 2, 0, 0, 0, 2], [3, 1, 0, 0, 0, 2]],
    ["Amaj7", [null, 0, 2, 1, 2, 0], [null, 0, 2, 1, 3, 0]],
    ["Am7", [null, 0, 2, 0, 1, 0], [null, 0, 2, 0, 1, 0]],
    ["Dm7", [null, null, 0, 2, 1, 1], [null, null, 0, 2, 1, 1], [],
      [{ fret: 1, fromString: 2, toString: 1, finger: 1 }]],
    ["Em7", [0, 2, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]]
  ];
  const SHAPES = {
    E: { root: 4, fromString: 6, frets: {
      major: [0, 2, 2, 1, 0, 0], minor: [0, 2, 2, 0, 0, 0],
      dominant_seventh: [0, 2, 0, 1, 0, 0], major_seventh: [0, 2, 1, 1, 0, 0],
      minor_seventh: [0, 2, 0, 0, 0, 0]
    } },
    A: { root: 9, fromString: 5, frets: {
      major: [null, 0, 2, 2, 2, 0], minor: [null, 0, 2, 2, 1, 0],
      dominant_seventh: [null, 0, 2, 0, 2, 0], major_seventh: [null, 0, 2, 1, 2, 0],
      minor_seventh: [null, 0, 2, 0, 1, 0]
    } }
  };

  function reference(chord, frets, options) {
    const strings = frets.map((fret, index) => ({ string: 6 - index, fret,
      state: fret === null ? "muted" : fret === 0 ? "open" : "fretted",
      midi_note: fret === null ? null : TUNING[index] + fret,
      note_name: fret === null ? null : noteName(TUNING[index] + fret),
      finger: options.fingers?.[index] ?? null, evidence: "reference", pressed: null, sounding: null }));
    return { kind: "reference", label: chord.label, root: chord.root, rootPc: chord.rootPc,
      quality: chord.quality, intervals: [...chord.intervals], pitchClasses: [...chord.pitchClasses],
      source: options.source, shape: options.shape || "open", tuning: [...TUNING],
      frets: [...frets], strings, positions: strings.filter(position => position.fret !== null),
      baseFret: Math.min(...frets.filter(fret => fret > 0)),
      fingers: strings.map(position => position.finger),
      barres: (options.barres || []).map(barre => ({ ...barre })),
      omittedIntervals: [...(options.omittedIntervals || [])],
      limitation: "Reference voicing only; this does not describe the observed hand or prove a pressed string." };
  }

  function getChordCatalog() {
    return OPEN.map(([label, frets, fingers, omittedIntervals, barres]) =>
      reference(parseChordLabel(label), frets, { fingers, omittedIntervals, barres, source: "verified_open_catalog" }));
  }

  function fretRange(options, defaultLast = 12) {
    const firstFret = options.firstFret ?? 0, lastFret = options.lastFret ?? defaultLast;
    if (!Number.isInteger(firstFret) || !Number.isInteger(lastFret) || firstFret < 0 ||
        lastFret > 24 || lastFret < firstFret) throw new RangeError("Frets must be integers from 0 through 24 in ascending order");
    return { firstFret, lastFret };
  }

  function getChordVoicings(label, options = {}) {
    const chord = parseChordLabel(label);
    if (!chord) return [];
    const { lastFret } = fretRange({ lastFret: options.maxFret ?? 24 }, 24);
    if (options.shape !== undefined && !["open", "E", "A"].includes(options.shape)) throw new RangeError("Unknown chord shape");
    const voicings = options.shape && options.shape !== "open" ? [] : getChordCatalog()
      .filter(item => item.rootPc === chord.rootPc && item.quality === chord.quality)
      .map(item => ({ ...item, root: chord.root, label: chord.label }));
    for (const [shape, template] of Object.entries(SHAPES)) {
      if (options.shape && options.shape !== shape) continue;
      const offset = mod12(chord.rootPc - template.root) || 12;
      for (let base = offset; base <= lastFret; base += 12) {
        const frets = template.frets[chord.quality].map(fret => fret === null ? null : base + fret);
        if (Math.max(...frets) > lastFret) continue;
        voicings.push(reference(chord, frets, { source: `generated_${shape}_barre`, shape,
          barres: [{ fret: base, fromString: template.fromString, toString: 1, finger: 1 }] }));
      }
    }
    // Slash basses are never silently replaced by a root-position reference.
    return voicings.filter(item => Math.max(...item.frets) <= lastFret &&
      mod12(Math.min(...item.positions.map(position => position.midi_note))) === chord.bassPc)
      .sort((a, b) => Number(a.shape !== "open") - Number(b.shape !== "open") || a.baseFret - b.baseFret);
  }

  function chooseReference(label, options = {}) {
    return getChordVoicings(label, options)[0] || null;
  }

  function validTuning(tuning, lastFret = 24) {
    if (!Array.isArray(tuning) || tuning.length !== 6 || tuning.some(midi =>
      !Number.isInteger(midi) || midi < 0 || midi + lastFret > 127)) throw new RangeError("Expected six valid open-string MIDI pitches, strings 6 through 1");
    return [...tuning];
  }

  function evidenceGate(vision, audio, now, options) {
    const visionMaxAgeMs = options.visionMaxAgeMs ?? 500;
    const audioMaxAgeMs = options.audioMaxAgeMs ?? 750;
    const maximumGapMs = options.maximumGapMs ?? 750;
    if ([visionMaxAgeMs, audioMaxAgeMs, maximumGapMs].some(value => !Number.isFinite(value) || value < 0)) {
      throw new RangeError("Freshness windows must be finite nonnegative milliseconds");
    }
    const current = (frame, limit) => frame && !frame.stale && Number.isFinite(now) && now >= 0 &&
      Number.isFinite(frame.receivedAt) && frame.receivedAt >= 0 && frame.receivedAt <= now &&
      now - frame.receivedAt <= limit && (frame.age_seconds === undefined ||
        (Number.isFinite(frame.age_seconds) && frame.age_seconds >= 0 &&
         Number.isFinite(frame.ageReportedAt ?? frame.receivedAt) &&
         (frame.ageReportedAt ?? frame.receivedAt) <= now &&
         frame.age_seconds * 1000 + now - (frame.ageReportedAt ?? frame.receivedAt) <= limit));
    if (!current(vision, visionMaxAgeMs) || (vision.status !== undefined &&
        !["candidates", "relative_positions", "no_fingertips_on_board", "tracking"].includes(vision.status))) {
      return { visionValid: false, audioValid: false, status: "vision_unavailable" };
    }
    if (["video_file", "board_video_file"].includes(vision.source) || vision.recording === true || vision.recorded === true) {
      return { visionValid: false, audioValid: false, status: "recording_not_synchronized" };
    }
    if (!["camera", "board_camera"].includes(vision.source) || vision.synthetic === true ||
        vision.demo === true || vision.provenance?.synthetic === true) {
      return { visionValid: false, audioValid: false, status: "demo_only" };
    }
    const mode = audio?.testMode ?? audio?.fft?.analysis_test_mode;
    const sourceValid = audio?.source === undefined ||
      ["fpga", "hardware", "microphone", "browser_microphone", "browser_mic", "mic", "audio_input"].includes(audio.source);
    const audioValid = Boolean(current(audio, audioMaxAgeMs) && sourceValid && audio.status === "recognized" &&
      mode === 0 && (audio.fft?.analysis_test_mode === undefined || audio.fft.analysis_test_mode === 0) &&
      !audio.synthetic && !audio.demo && !audio.recording && !audio.recorded && !audio.provenance?.synthetic &&
      Array.isArray(audio.notes) && Math.abs(audio.receivedAt - vision.receivedAt) <= maximumGapMs);
    return { visionValid: true, audioValid, status: audioValid ? "ready" : "audio_unavailable",
      reception_gap_ms: audioValid ? Math.abs(audio.receivedAt - vision.receivedAt) : null };
  }

  // Inputs use FretVision.snapshot() and normalized FretAudioSnapshot() fields.
  // Raw FPGA notes (midi_note/octave_ambiguous) are accepted with an explicit test mode.
  function fuseEvidence(vision, audio, now, options = {}) {
    const tuning = validTuning(options.tuning ?? TUNING);
    const gate = evidenceGate(vision, audio, now, options);
    const result = { kind: "observed", label: "Observed projections", status: gate.status,
      positions: [], relativePositions: [], matches: [], unmatched: [], rejectedPositions: [],
      strings: TUNING.map((_, index) => ({ string: 6 - index, state: "unknown", fret: null,
        midi_note: null, pressed: null, sounding: null, evidence: "none", positions: [] })),
      synchronized: false, absoluteFretsKnown: false, timingBasis: "browser_reception",
      audioAvailable: gate.audioValid, reception_gap_ms: gate.reception_gap_ms ?? null, limitation: LIMITATION };
    if (!gate.visionValid) return result;
    const midiNotes = new Set((gate.audioValid ? audio.notes : []).filter(note => {
      if (!note || note.ambiguous || note.octave_ambiguous) return false;
      const midi = note.midi ?? note.midi_note;
      return Number.isInteger(midi) && midi >= 0 && midi <= 127 &&
        (note.midi === undefined || note.midi_note === undefined || note.midi === note.midi_note);
    }).map(note => note.midi ?? note.midi_note));
    for (const position of Array.isArray(vision.positions) ? vision.positions : []) {
      if (!position || typeof position !== "object") continue;
      const identity = { finger: typeof position.finger === "string" ? position.finger : null,
        hand_index: Number.isInteger(position.hand_index) ? position.hand_index : null };
      if (position.relative === true) {
        if (Number.isInteger(position.string_line) && position.string_line >= 1 && position.string_line <= 6 &&
            Number.isInteger(position.fret_interval) && position.fret_interval >= 1 && position.fret_interval <= 24) {
          result.relativePositions.push({ ...identity, string_line: position.string_line,
            fret_interval: position.fret_interval, relative: true, ambiguous: Boolean(position.ambiguous),
            pressed: null, sounding: null, audioSupported: false, evidence: "observed_lines_projection" });
        }
        continue;
      }
      if (!Number.isInteger(position.string) || position.string < 1 || position.string > 6 ||
          !Number.isInteger(position.fret) || position.fret < 1 || position.fret > 24) {
        result.rejectedPositions.push({ reason: "invalid_string_or_fret" });
        continue;
      }
      const midi = tuning[6 - position.string] + position.fret;
      if (position.midi_note !== undefined && position.midi_note !== midi) {
        result.rejectedPositions.push({ string: position.string, fret: position.fret, reason: "inconsistent_midi" });
        continue;
      }
      result.positions.push({ ...identity, string: position.string, fret: position.fret, midi_note: midi,
        note_name: noteName(midi), relative: false, ambiguous: Boolean(position.ambiguous),
        pressed: null, sounding: null, audioSupported: false, evidence: "fingertip_projection" });
    }
    for (const position of result.positions) {
      // One fingertip projected onto multiple cells is still ambiguous even if a caller omitted the flag.
      if (position.finger !== null && result.positions.some(other => other !== position &&
          other.finger === position.finger && other.hand_index === position.hand_index &&
          (other.string !== position.string || other.fret !== position.fret))) position.ambiguous = true;
      position.audioSupported = !position.ambiguous && midiNotes.has(position.midi_note);
      if (position.audioSupported) {
        position.evidence = "audio_supported_projection";
        result.matches.push(position);
      } else result.unmatched.push(position);
      result.strings[6 - position.string].positions.push(position);
    }
    for (const string of result.strings) {
      if (!string.positions.length) continue;
      const frets = new Set(string.positions.map(position => position.fret));
      const ambiguous = frets.size !== 1 || string.positions.some(position => position.ambiguous);
      string.state = ambiguous ? "ambiguous_projection" : "projection";
      string.fret = ambiguous ? null : string.positions[0].fret;
      string.midi_note = ambiguous ? null : string.positions[0].midi_note;
      string.evidence = string.positions.some(position => position.audioSupported) ? "audio_supported_projection" : "fingertip_projection";
    }
    result.absoluteFretsKnown = result.positions.length > 0 && result.relativePositions.length === 0;
    result.status = !gate.audioValid ? gate.status : result.matches.length ? "supported_positions" :
      result.positions.length ? "no_matching_pitch" : result.relativePositions.length ? "relative_positions" : "no_visual_positions";
    return result;
  }

  function getScale(tonicValue, modeValue = "ionian") {
    const tonic = parseTonic(tonicValue);
    if (!tonic || typeof modeValue !== "string") return null;
    let mode = modeValue.trim().toLowerCase().replace(/[\s-]+/g, "_");
    const aliases = { major: "ionian", minor: "aeolian", natural_minor: "aeolian", natural_major: "ionian" };
    if (Object.hasOwn(aliases, mode)) mode = aliases[mode];
    if (!Object.hasOwn(MODES, mode)) return null;
    const intervals = [...MODES[mode]];
    const degrees = mode === "major_pentatonic" ? [1, 2, 3, 5, 6] :
      mode === "minor_pentatonic" ? [1, 3, 4, 5, 7] : [1, 2, 3, 4, 5, 6, 7];
    const letters = "CDEFGAB", major = MODES.ionian;
    const notes = intervals.map((interval, index) => {
      const degree = degrees[index], pitchClass = mod12(tonic.pitchClass + interval);
      const letter = letters[(letters.indexOf(tonic.letter) + degree - 1) % 7];
      const difference = mod12(pitchClass - NATURAL[letter]);
      const accidental = difference > 6 ? "b".repeat(12 - difference) : "#".repeat(difference);
      const alteration = interval - major[degree - 1];
      return { name: letter + accidental, pitchClass, interval, root: interval === 0,
        degree: (alteration < 0 ? "b".repeat(-alteration) : "#".repeat(alteration)) + degree };
    });
    const modeName = mode.replace(/_/g, " ").replace(/^./, character => character.toUpperCase());
    return { kind: "scale_reference", tonic: tonic.name, rootPc: tonic.pitchClass, mode,
      label: `${tonic.name} ${modeName}`, intervals, pitchClasses: notes.map(note => note.pitchClass), notes,
      limitation: "Scale reference only; compatible pitch classes do not establish a key or tonic." };
  }

  function scaleFretmap(tonic, mode = "ionian", options = {}) {
    const scale = getScale(tonic, mode);
    if (!scale) return null;
    const { firstFret, lastFret } = fretRange(options);
    const tuning = validTuning(options.tuning ?? TUNING, lastFret), positions = [];
    tuning.forEach((open, index) => {
      for (let fret = firstFret; fret <= lastFret; fret += 1) {
        const midi = open + fret, note = scale.notes.find(item => item.pitchClass === mod12(midi));
        if (note) positions.push({ string: 6 - index, fret, midi_note: midi, note_name: noteName(midi),
          scale_note: note.name, pitchClass: note.pitchClass, degree: note.degree, root: note.root, evidence: "scale_reference" });
      }
    });
    return { ...scale, kind: "scale_fretmap", firstFret, lastFret, tuning, positions };
  }

  function svgStart(width, height, kind, title, description) {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(title)}" data-kind="${kind}" style="max-width:100%;height:auto" fill="none" color="#303b39" font-family="system-ui, sans-serif"><title>${escapeXml(title)}</title><desc>${escapeXml(description)}</desc>`;
  }
  const textAt = (x, y, content, size = 11, attributes = "") =>
    `<text x="${x}" y="${y}" fill="currentColor" font-size="${size}" ${attributes}>${escapeXml(content)}</text>`;
  const line = (x1, y1, x2, y2, width = 1, attributes = "") =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="currentColor" stroke-width="${width}" ${attributes}/>`;

  function renderRelativeChart(model) {
    const positions = model.relativePositions;
    const first = Math.min(...positions.map(position => position.fret_interval));
    const last = Math.max(...positions.map(position => position.fret_interval));
    const rows = Math.max(4, last - first + 1), top = 80, bottom = top + rows * 30;
    let svg = svgStart(240, bottom + 66, "observed", "Relative projections", LIMITATION);
    svg += textAt(120, 23, "Relative projections", 16, 'text-anchor="middle"');
    svg += textAt(120, 43, "String and fret numbering unknown", 11, 'text-anchor="middle"');
    svg += textAt(120, 63, "Detected lines 1-6", 10, 'text-anchor="middle"');
    for (let row = 0; row <= rows; row += 1) svg += line(45, top + row * 30, 195, top + row * 30);
    for (let row = 0; row < rows; row += 1) svg += textAt(12, top + row * 30 + 19, first + row, 10);
    for (let stringLine = 1; stringLine <= 6; stringLine += 1) {
      const x = 45 + (stringLine - 1) * 30;
      svg += line(x, top, x, bottom) + textAt(x, bottom + 18, stringLine, 11, 'text-anchor="middle"');
    }
    for (const position of positions) {
      const x = 45 + (position.string_line - 1) * 30;
      const y = top + (position.fret_interval - first + .5) * 30;
      svg += `<circle data-line="${position.string_line}" data-interval="${position.fret_interval}" data-evidence="observed_lines_projection" cx="${x}" cy="${y}" r="9" stroke="currentColor" stroke-width="1.6"${position.ambiguous ? ' stroke-dasharray="2 2"' : ""}><title>${escapeXml(`Detected line ${position.string_line}, relative interval ${position.fret_interval}; string, fret and pressure unknown`)}</title></circle>`;
    }
    return svg + textAt(120, bottom + 38, `${positions.length} relative projections; numbering unknown`, 10, 'text-anchor="middle"') +
      textAt(120, bottom + 54, "Intervals only; pressure and sound unknown", 10, 'text-anchor="middle"') + "</svg>";
  }

  function renderChordChart(model) {
    const observed = model?.kind === "observed", referenceModel = model?.kind === "reference";
    if (!observed && !referenceModel) return svgStart(240, 120, "reference", "No reference voicing", "No supported reference was selected") + textAt(120, 60, "No reference voicing", 14, 'text-anchor="middle"') + "</svg>";
    const positions = (model.positions || []).filter(position => Number.isInteger(position.fret) &&
      position.fret > 0 && position.fret <= 24 && Number.isInteger(position.string) && position.string >= 1 && position.string <= 6);
    if (observed && !positions.length && model.relativePositions?.length) return renderRelativeChart(model);
    const min = positions.length ? Math.min(...positions.map(position => position.fret)) : 1;
    const max = positions.length ? Math.max(...positions.map(position => position.fret)) : 1;
    const first = max <= 5 ? 1 : min, rows = Math.max(5, max - first + 1);
    const top = 80, bottom = top + rows * 30, height = bottom + 65;
    const title = observed ? "Observed projections" : model.label;
    let svg = svgStart(240, height, observed ? "observed" : "reference", title, model.limitation || LIMITATION);
    svg += textAt(120, 23, String(title).length > 25 ? String(title).slice(0, 22) + "..." : title, 16, 'text-anchor="middle"');
    svg += textAt(120, 43, observed ? `${model.matches?.length || 0} audio-supported projections` : "Reference voicing", 11, 'text-anchor="middle"');
    const x = string => 45 + (6 - string) * 30;
    const y = fret => top + (fret - first + .5) * 30;
    for (let row = 0; row <= rows; row += 1) svg += line(45, top + row * 30, 195, top + row * 30, row === 0 && first === 1 ? 3 : 1);
    for (let string = 6; string >= 1; string -= 1) {
      svg += line(x(string), top, x(string), bottom, 1 + (string - 1) * .12);
      svg += textAt(x(string), bottom + 18, string, 11, 'text-anchor="middle"');
      const state = observed ? "unknown" : model.strings?.find(item => item.string === string)?.state;
      if (state === "open") svg += `<circle data-state="open" cx="${x(string)}" cy="65" r="5" stroke="currentColor" stroke-width="1.5"/>`;
      else if (state === "muted" || state === "unknown") svg += textAt(x(string), 69, state === "muted" ? "X" : "?", 13, `text-anchor="middle" data-state="${state}"`);
    }
    svg += textAt(19, top + 20, first, 11);
    if (referenceModel) for (const barre of model.barres || []) {
      if (Number.isInteger(barre.fret) && barre.fret >= first && barre.fret <= first + rows - 1 &&
          Number.isInteger(barre.fromString) && barre.fromString <= 6 && barre.fromString >= 1 &&
          Number.isInteger(barre.toString) && barre.toString >= 1 && barre.toString <= barre.fromString) {
        svg += line(x(barre.fromString), y(barre.fret), x(barre.toString), y(barre.fret), 13,
          'stroke-linecap="round" data-evidence="reference_barre"');
      }
    }
    for (const position of positions) {
      const supported = observed && position.audioSupported === true && !position.ambiguous;
      const fill = referenceModel ? "currentColor" : supported ? "#137d67" : "none";
      svg += `<circle data-string="${position.string}" data-fret="${position.fret}" data-evidence="${referenceModel ? "reference" : supported ? "audio_supported_projection" : "fingertip_projection"}" cx="${x(position.string)}" cy="${y(position.fret)}" r="9" fill="${fill}" stroke="${supported ? "#137d67" : "currentColor"}" stroke-width="1.6"${position.ambiguous ? ' stroke-dasharray="2 2"' : ""}><title>${escapeXml(`String ${position.string}, fret ${position.fret}${observed ? supported ? ": audio supports this projection; pressure unknown" : ": projection only; pressure unknown" : ": reference"}`)}</title></circle>`;
      if (referenceModel && Number.isInteger(position.finger) && position.finger >= 1 && position.finger <= 4) {
        svg += `<text x="${x(position.string)}" y="${y(position.fret) + 4}" fill="white" font-size="11" text-anchor="middle">${position.finger}</text>`;
      }
    }
    const relative = observed ? model.relativePositions?.length || 0 : 0;
    svg += textAt(120, bottom + 38, observed ? "Projections; pressure and sound unknown" : "O open / X muted - reference only", 10, 'text-anchor="middle"');
    if (relative) svg += textAt(120, bottom + 54, `${relative} relative projections; numbering unknown`, 10, 'text-anchor="middle"');
    return svg + "</svg>";
  }

  function renderScaleFretmap(map, options = {}) {
    if (!map || map.kind !== "scale_fretmap") return svgStart(640, 100, "scale_reference", "No scale reference", "No scale was selected") + textAt(16, 48, "No scale reference", 14) + "</svg>";
    const { firstFret, lastFret } = fretRange(map), count = lastFret - firstFret + 1;
    const width = Math.max(300, 68 + count * 40), left = 48, cell = (width - left - 12) / count;
    let svg = svgStart(width, 222, "scale_reference", map.label, map.limitation);
    svg += textAt(14, 22, map.label, 16) + textAt(14, 42, "Scale reference - tonic highlighted", 11);
    for (let fret = firstFret; fret <= lastFret; fret += 1) {
      const center = left + (fret - firstFret + .5) * cell;
      svg += textAt(center, 63, fret, 10, 'text-anchor="middle"');
      svg += line(center + cell / 2, 77, center + cell / 2, 187, fret === 0 ? 3 : 1);
    }
    for (let string = 1; string <= 6; string += 1) {
      const y = 77 + (string - 1) * 22;
      svg += textAt(8, y + 4, `${string} ${NAMES[mod12(map.tuning[6 - string])]}`, 10);
      svg += line(left, y, width - 12, y, .8 + string * .1);
    }
    for (const position of map.positions) {
      if (!Number.isInteger(position.fret) || position.fret < firstFret || position.fret > lastFret ||
          !Number.isInteger(position.string) || position.string < 1 || position.string > 6) continue;
      const x = left + (position.fret - firstFret + .5) * cell, y = 77 + (position.string - 1) * 22;
      const label = options.labels === "degrees" ? position.degree : position.scale_note;
      svg += `<circle data-string="${position.string}" data-fret="${position.fret}" data-midi="${escapeXml(position.midi_note)}" data-root="${position.root === true}" cx="${x}" cy="${y}" r="9" fill="${position.root ? "#137d67" : "white"}" stroke="${position.root ? "#137d67" : "currentColor"}"><title>${escapeXml(`${position.scale_note}, ${position.note_name}, string ${position.string}, fret ${position.fret}; scale reference`)}</title></circle>`;
      svg += `<text x="${x}" y="${y + 3}" fill="${position.root ? "white" : "currentColor"}" font-size="9" text-anchor="middle">${escapeXml(label)}</text>`;
    }
    return svg + textAt(14, 212, map.notes.map(note => note.name).join("  "), 11) + "</svg>";
  }

  const api = Object.freeze({ TUNING, CHORD_TYPES, MODES, parseChordLabel, getChordCatalog,
    getChordVoicings, chooseReference, fuseEvidence, getScale, scaleFretmap, renderChordChart, renderScaleFretmap });
  root.GuitarFingering = api;
  if (typeof module === "object" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
