/* Standard six-string chord data, SVG diagrams, and pitch-set candidate search. */
(function (root) {
  "use strict";

  const TUNING = Object.freeze([40, 45, 50, 55, 59, 64]); // Strings 6 through 1.
  const NAMES = Object.freeze(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]);
  const NATURAL = Object.freeze({ C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 });
  // Keep this order aligned with pynq/audio_music.py CHORD_TYPES so an
  // unambiguous audio label keeps the same name in the chart.
  const QUALITIES = Object.freeze(Object.fromEntries(Object.entries({
    major: { suffix: "", intervals: [0, 4, 7] },
    minor: { suffix: "m", intervals: [0, 3, 7] },
    diminished: { suffix: "dim", intervals: [0, 3, 6] },
    augmented: { suffix: "aug", intervals: [0, 4, 8] },
    dominant_seventh: { suffix: "7", intervals: [0, 4, 7, 10] },
    major_seventh: { suffix: "maj7", intervals: [0, 4, 7, 11] },
    minor_seventh: { suffix: "m7", intervals: [0, 3, 7, 10] },
    minor_major_seventh: { suffix: "m(maj7)", intervals: [0, 3, 7, 11] },
    half_diminished_seventh: { suffix: "m7b5", intervals: [0, 3, 6, 10] },
    diminished_seventh: { suffix: "dim7", intervals: [0, 3, 6, 9] },
    suspended_second: { suffix: "sus2", intervals: [0, 2, 7] },
    suspended_fourth: { suffix: "sus4", intervals: [0, 5, 7] },
    dominant_seventh_sus2: { suffix: "7sus2", intervals: [0, 2, 7, 10] },
    dominant_seventh_sus4: { suffix: "7sus4", intervals: [0, 5, 7, 10] },
    power: { suffix: "5", intervals: [0, 7] },
    major_sixth: { suffix: "6", intervals: [0, 4, 7, 9] },
    minor_sixth: { suffix: "m6", intervals: [0, 3, 7, 9] },
    added_ninth: { suffix: "add9", intervals: [0, 2, 4, 7] },
    minor_added_ninth: { suffix: "m(add9)", intervals: [0, 2, 3, 7] },
    dominant_ninth: { suffix: "9", intervals: [0, 2, 4, 7, 10] },
    major_ninth: { suffix: "maj9", intervals: [0, 2, 4, 7, 11] },
    minor_ninth: { suffix: "m9", intervals: [0, 2, 3, 7, 10] },
    added_eleventh: { suffix: "add11", intervals: [0, 4, 5, 7] },
    minor_added_eleventh: { suffix: "m(add11)", intervals: [0, 3, 5, 7] },
    sixth_added_ninth: { suffix: "6/9", intervals: [0, 2, 4, 7, 9] },
    minor_sixth_added_ninth: { suffix: "m6/9", intervals: [0, 2, 3, 7, 9] },
    dominant_seventh_flat_ninth: { suffix: "7b9", intervals: [0, 1, 4, 7, 10] },
    dominant_seventh_sharp_ninth: { suffix: "7#9", intervals: [0, 3, 4, 7, 10] },
    dominant_seventh_flat_fifth: { suffix: "7b5", intervals: [0, 4, 6, 10] },
    dominant_seventh_sharp_fifth: { suffix: "7#5", intervals: [0, 4, 8, 10] },
    dominant_ninth_suspended_fourth: { suffix: "9sus4", intervals: [0, 2, 5, 7, 10] },
    dominant_eleventh: { suffix: "11", intervals: [0, 2, 4, 5, 7, 10] },
    major_eleventh: { suffix: "maj11", intervals: [0, 2, 4, 5, 7, 11] },
    minor_eleventh: { suffix: "m11", intervals: [0, 2, 3, 5, 7, 10] },
    dominant_ninth_sharp_eleventh: { suffix: "9#11", intervals: [0, 2, 4, 6, 7, 10] },
    dominant_seventh_added_thirteenth: { suffix: "7add13", intervals: [0, 4, 7, 9, 10] }
  }).map(([name, item]) => [name, Object.freeze({ suffix: item.suffix,
    intervals: Object.freeze(item.intervals) })])));
  const aliasToken = (value) => String(value ?? "").trim().toLowerCase().replace(/[\s_-]+/g, "");
  const QUALITY_ALIASES = Object.freeze({
    ...Object.fromEntries(Object.entries(QUALITIES).flatMap(([name, definition]) => [
      [aliasToken(name), name],
      [aliasToken(definition.suffix), name]
    ])),
    maj: "major", major: "major",
    min: "minor", minor: "minor",
    dom7: "dominant_seventh", dominant7: "dominant_seventh", dominantseventh: "dominant_seventh",
    major7: "major_seventh", majorseventh: "major_seventh",
    min7: "minor_seventh", minor7: "minor_seventh", minorseventh: "minor_seventh",
    minmaj7: "minor_major_seventh", mmaj7: "minor_major_seventh",
    sus: "suspended_fourth", add: "added_ninth"
  });
  const MUTED = -1;
  const STRINGS = Object.freeze([6, 5, 4, 3, 2, 1]);
  const mod12 = (value) => ((value % 12) + 12) % 12;
  const escapeXml = (value) => String(value).replace(/[&<>"']/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" })[character]);
  const noteName = (midi) => `${NAMES[mod12(midi)]}${Math.floor(midi / 12) - 1}`;
  const sameSet = (left, right) => left.length === right.length &&
    left.every((value, index) => value === right[index]);

  function parseNote(value) {
    if (Number.isInteger(value)) return { name: NAMES[mod12(value)], pitchClass: mod12(value) };
    if (typeof value !== "string") return null;
    const match = /^([A-Ga-g])([#b]?)(?:-?\d+)?$/.exec(value.trim());
    if (!match) return null;
    const letter = match[1].toUpperCase();
    const accidental = match[2].toLowerCase();
    const pitchClass = mod12(NATURAL[letter] + (accidental === "#" ? 1 : accidental === "b" ? -1 : 0));
    return { name: NAMES[pitchClass], pitchClass };
  }

  function normalizeRoot(value) {
    const parsed = parseNote(value);
    if (!parsed) throw new TypeError("Expected a note name or MIDI pitch class for root");
    return parsed;
  }

  function normalizeQuality(value = "major") {
    const original = String(value ?? "").trim();
    if (original === "M") return "major";
    if (original === "M7") return "major_seventh";
    const token = original.toLowerCase().replace(/[\s_-]+/g, "");
    const quality = QUALITY_ALIASES[token];
    if (!quality) throw new RangeError(`Unsupported chord quality: ${original || "empty"}`);
    return quality;
  }

  function parseChordLabel(value) {
    if (typeof value !== "string") return null;
    const text = value.trim().replace(/\u266f/g, "#").replace(/\u266d/g, "b");
    const match = /^([A-Ga-g][#b]?)(.*)$/.exec(text);
    if (!match) return null;
    let qualityText = match[2].trim();
    let bass = null;
    const slash = qualityText.lastIndexOf("/");
    if (slash >= 0) {
      const candidate = parseNote(qualityText.slice(slash + 1));
      if (candidate) {
        bass = candidate;
        qualityText = qualityText.slice(0, slash);
      }
    }
    try {
      const root = normalizeRoot(match[1]);
      const quality = normalizeQuality(qualityText || "major");
      const pitchClasses = QUALITIES[quality].intervals
        .map((interval) => mod12(root.pitchClass + interval));
      return Object.freeze({ root: root.name, rootPc: root.pitchClass, quality,
        bass: bass?.name || root.name, bassPc: bass?.pitchClass ?? root.pitchClass,
        label: root.name + QUALITIES[quality].suffix +
          (bass && bass.pitchClass !== root.pitchClass ? `/${bass.name}` : ""),
        pitchClasses: Object.freeze([...new Set(pitchClasses)].sort((a, b) => a - b)) });
    } catch (_) {
      return null;
    }
  }

  function normalizePitchSet(values) {
    if (!Array.isArray(values) || !values.length) throw new TypeError("Expected a nonempty MIDI pitch-class set");
    const pitches = values.map((value) => {
      const parsed = parseNote(value);
      if (!parsed) throw new TypeError(`Invalid pitch-class value: ${String(value)}`);
      return parsed.pitchClass;
    });
    return Object.freeze([...new Set(pitches)].sort((a, b) => a - b));
  }

  function validateTuning(values) {
    if (!Array.isArray(values) || values.length !== 6 || values.some((midi) =>
      !Number.isInteger(midi) || midi < 0 || midi > 115)) {
      throw new RangeError("Expected six open-string MIDI pitches, strings 6 through 1");
    }
    return Object.freeze([...values]);
  }

  function normalizeFret(value) {
    if (value === null || value === MUTED || value === "x" || value === "X") return null;
    if (!Number.isInteger(value) || value < 0 || value > 24) {
      throw new RangeError("Each fret must be muted, 0, or an integer from 1 to 24");
    }
    return value;
  }

  function normalizeFrets(values) {
    if (!Array.isArray(values) || values.length !== 6) throw new RangeError("Expected exactly six string frets");
    return Object.freeze(values.map(normalizeFret));
  }

  function normalizeBarres(values, frets) {
    if (!Array.isArray(values)) throw new TypeError("Barres must be an array");
    return Object.freeze(values.map((item) => {
      const fret = Number(item.fret);
      const fromString = Number(item.fromString ?? item.from);
      const toString = Number(item.toString ?? item.to);
      const finger = item.finger === undefined ? 1 : Number(item.finger);
      if (!Number.isInteger(fret) || fret < 1 || fret > 24 ||
          !Number.isInteger(fromString) || !Number.isInteger(toString) ||
          fromString < 1 || fromString > 6 || toString < 1 || toString > fromString ||
          !Number.isInteger(finger) || finger < 1 || finger > 4) {
        throw new RangeError("Invalid barre; expected fret, fromString >= toString, and finger");
      }
      for (let string = toString; string <= fromString; string += 1) {
        const pressed = frets[6 - string];
        if (pressed === 0 || pressed === null || pressed < fret) {
          throw new RangeError(`Barre cannot cross open, muted, or lower-fret string ${string}`);
        }
      }
      return Object.freeze({ fret, fromString, toString, finger });
    }));
  }

  function detectBarre(frets) {
    const fretsByNumber = new Map();
    frets.forEach((fret, index) => {
      if (fret > 0) fretsByNumber.set(fret, [...(fretsByNumber.get(fret) || []), index]);
    });
    for (const fret of [...fretsByNumber.keys()].sort((a, b) => a - b)) {
      const indices = fretsByNumber.get(fret);
      if (indices.length < 2) continue;
      const first = Math.min(...indices), last = Math.max(...indices);
      const crossed = frets.slice(first, last + 1);
      if (crossed.some((value) => value === null || value === 0 || value < fret)) continue;
      return [{ fret, fromString: 6 - first, toString: 6 - last, finger: 1 }];
    }
    return [];
  }

  function assignFingers(frets, barres) {
    const fingers = frets.map(() => null);
    const barre = barres[0];
    if (barre) {
      for (let string = barre.toString; string <= barre.fromString; string += 1) {
        if (frets[6 - string] === barre.fret) fingers[6 - string] = barre.finger;
      }
    }
    const remaining = [];
    frets.forEach((fret, index) => {
      if (fret > 0 && fingers[index] === null) remaining.push({ fret, index });
    });
    remaining.sort((left, right) => left.fret - right.fret || right.index - left.index);
    const firstFinger = barre ? 2 : 1;
    if (remaining.length + (barre ? 1 : 0) > 4) return null;
    remaining.forEach((item, index) => { fingers[item.index] = firstFinger + index; });
    return Object.freeze(fingers);
  }

  function normalizeFingers(values, frets) {
    if (!Array.isArray(values) || values.length !== 6) throw new RangeError("Expected six finger values");
    return Object.freeze(values.map((value, index) => {
      if (frets[index] === null || frets[index] === 0) return null;
      if (value === null || value === undefined) return null;
      if (!Number.isInteger(value) || value < 1 || value > 4) throw new RangeError("Finger numbers must be 1 through 4");
      return value;
    }));
  }

  function buildVoicing(spec, tuningOverride) {
    const tuning = tuningOverride || validateTuning(spec.tuning || TUNING);
    const rootNote = normalizeRoot(spec.root);
    const quality = normalizeQuality(spec.quality);
    const requestedBass = spec.bass === undefined ? null : normalizeRoot(spec.bass);
    const frets = normalizeFrets(spec.frets);
    const strings = frets.map((fret, index) => {
      const string = 6 - index;
      const midi = fret === null ? null : tuning[index] + fret;
      if (midi !== null && midi > 127) throw new RangeError("Voicing exceeds MIDI note range");
      return Object.freeze({ string, fret, state: fret === null ? "muted" : fret === 0 ? "open" : "fretted",
        midi, note: midi === null ? null : noteName(midi), pitchClass: midi === null ? null : mod12(midi) });
    });
    const detectedBarres = detectBarre(frets);
    const barres = spec.barres === undefined ? Object.freeze(detectedBarres) : normalizeBarres(spec.barres, frets);
    const fingers = spec.fingers === undefined ? assignFingers(frets, barres) : normalizeFingers(spec.fingers, frets);
    if (!fingers) throw new RangeError("Voicing requires more than four fingers");
    const fretted = frets.filter((fret) => fret > 0);
    const minFret = fretted.length ? Math.min(...fretted) : 1;
    const maxFret = fretted.length ? Math.max(...fretted) : 1;
    const startFret = spec.startFret === undefined ? (maxFret <= 5 ? 1 : minFret) : Number(spec.startFret);
    if (!Number.isInteger(startFret) || startFret < 1 || startFret > 24) throw new RangeError("Start fret must be 1 through 24");
    const sounding = strings.filter((string) => string.fret !== null);
    const pitchClasses = Object.freeze([...new Set(sounding.map((string) => string.pitchClass))].sort((a, b) => a - b));
    const voicing = String(spec.voicing || "custom");
    const bassString = sounding.reduce((lowest, string) =>
      lowest === null || string.midi < lowest.midi ? string : lowest, null);
    const bassPc = bassString?.pitchClass ?? null;
    const bassName = requestedBass?.name || (bassPc === null ? null : NAMES[bassPc]);
    const label = spec.label || rootNote.name + QUALITIES[quality].suffix +
      (requestedBass && requestedBass.pitchClass !== rootNote.pitchClass ? `/${requestedBass.name}` : "");
    return Object.freeze({
      id: spec.id || `${label}-${voicing}`.replace(/\s+/g, "-").toLowerCase(),
      label, root: rootNote.name, rootPc: rootNote.pitchClass, quality,
      qualityLabel: QUALITIES[quality].suffix, voicing, tuning, frets, fingers, barres,
      startFret, span: maxFret - minFret, strings,
      openStrings: Object.freeze(strings.filter((string) => string.state === "open").map((string) => string.string)),
      mutedStrings: Object.freeze(strings.filter((string) => string.state === "muted").map((string) => string.string)),
      frettedStrings: Object.freeze(strings.filter((string) => string.state === "fretted").map((string) => string.string)),
      noteNames: Object.freeze(strings.map((string) => string.note || "X")),
      pitchClasses, bass: bassName, bassPc,
      bassString: bassString?.string ?? null,
      source: spec.source || "data", score: spec.score
    });
  }

  const C = Object.freeze({
    major: Object.freeze({
      open: buildVoicing({ id: "C-major-open", root: "C", quality: "major", voicing: "open",
        frets: [null, 3, 2, 0, 1, 0], fingers: [null, 3, 2, null, 1, null] }),
      barre8: buildVoicing({ id: "C-major-barre8", root: "C", quality: "major", voicing: "barre8",
        frets: [8, 10, 10, 9, 8, 8], fingers: [1, 3, 4, 2, 1, 1],
        barres: [{ fret: 8, fromString: 6, toString: 1, finger: 1 }], startFret: 8 })
    }),
    minor: Object.freeze({
      barre3: buildVoicing({ id: "C-minor-barre3", root: "C", quality: "minor", voicing: "barre3",
        frets: [null, 3, 5, 5, 4, 3], fingers: [null, 1, 3, 4, 2, 1],
        barres: [{ fret: 3, fromString: 5, toString: 1, finger: 1 }], startFret: 3 })
    }),
    dominant_seventh: Object.freeze({
      open: buildVoicing({ id: "C-dominant7-open", root: "C", quality: "dominant7", voicing: "open",
        frets: [null, 3, 2, 3, 1, 0], fingers: [null, 3, 2, 4, 1, null] })
    }),
    major_seventh: Object.freeze({
      open: buildVoicing({ id: "C-major7-open", root: "C", quality: "major7", voicing: "open",
        frets: [null, 3, 2, 0, 0, 0] })
    }),
    minor_seventh: Object.freeze({
      barre3: buildVoicing({ id: "C-minor7-barre3", root: "C", quality: "minor7", voicing: "barre3",
        frets: [null, 3, 5, 3, 4, 3], fingers: [null, 1, 3, 1, 2, 1],
        barres: [{ fret: 3, fromString: 5, toString: 1, finger: 1 }], startFret: 3 })
    }),
    suspended_second: Object.freeze({
      open: buildVoicing({ id: "C-sus2-open", root: "C", quality: "sus2", voicing: "open",
        frets: [null, 3, 0, 0, 3, 3], fingers: [null, 3, null, null, 1, 2] })
    }),
    suspended_fourth: Object.freeze({
      open: buildVoicing({ id: "C-sus4-open", root: "C", quality: "sus4", voicing: "open",
        frets: [null, 3, 3, 0, 1, 1], fingers: [null, 3, 4, null, 1, 2] })
    }),
    added_ninth: Object.freeze({
      open: buildVoicing({ id: "C-add9-open", root: "C", quality: "add9", voicing: "open",
        frets: [null, 3, 2, 0, 3, 3], fingers: [null, 3, 2, null, 1, 4] })
    })
  });
  const CHORD_LIBRARY = Object.freeze({ C });

  function getVoicings(root, quality = "major") {
    const rootNote = normalizeRoot(root);
    const qualityName = normalizeQuality(quality);
    const voicings = CHORD_LIBRARY[rootNote.name]?.[qualityName];
    return voicings ? Object.freeze(Object.values(voicings)) : Object.freeze([]);
  }

  function getChord(root, quality = "major", voicing) {
    const voicings = getVoicings(root, quality);
    if (!voicings.length) return null;
    if (voicing === undefined || voicing === null || voicing === "") return voicings[0];
    return voicings.find((item) => item.voicing === String(voicing) || item.id === String(voicing)) || null;
  }

  function identifyChord(pitchClasses, preferredRootPc) {
    const pitches = normalizePitchSet(pitchClasses);
    const roots = preferredRootPc === undefined ? Array.from({ length: 12 }, (_, index) => index) :
      [normalizeRoot(preferredRootPc).pitchClass];
    for (const rootPc of roots) {
      for (const [quality, definition] of Object.entries(QUALITIES)) {
        const expected = [...new Set(definition.intervals.map((interval) => mod12(rootPc + interval)))]
          .sort((a, b) => a - b);
        if (sameSet(expected, pitches)) return { rootPc, root: NAMES[rootPc], quality,
          label: NAMES[rootPc] + definition.suffix };
      }
    }
    const rootPc = roots[0];
    return { rootPc, root: NAMES[rootPc], quality: "major", label: NAMES[rootPc], fallback: true };
  }

  function scoreCandidate(candidate, options = {}) {
    let score = candidate.mutedStrings.length * 8 + (candidate.startFret - 1) * .8 +
      candidate.span * 1.5 + candidate.frettedStrings.length * .15 -
      candidate.openStrings.length * 3;
    if (candidate.bassPc !== candidate.rootPc) score += 4;
    if (candidate.barres.length) score -= options.preferBarre ? 2.5 : .5;
    if (candidate.source === "library") score -= 20;
    return Number(score.toFixed(4));
  }

  function findCandidates(pitchClasses, tuning = TUNING, options = {}) {
    const pitches = normalizePitchSet(pitchClasses);
    const openStrings = validateTuning(tuning);
    const maxFret = Number(options.maxFret ?? 12);
    const maxSpan = Number(options.maxSpan ?? 4);
    const maxMuted = Number(options.maxMuted ?? 2);
    const minStrings = Number(options.minStrings ?? 4);
    const maxOpenFret = Number(options.maxOpenFret ?? 5);
    const maxResults = Number(options.maxResults ?? 200);
    if (![maxFret, maxSpan, maxMuted, minStrings, maxOpenFret, maxResults].every((value) =>
      Number.isInteger(value) && value >= 0) || maxFret > 24) {
      throw new RangeError("Candidate limits must be nonnegative integers; maxFret cannot exceed 24");
    }
    if (minStrings < 1 || minStrings > 6) throw new RangeError("minStrings must be 1 through 6");
    let identity = identifyChord(pitches, options.rootPc);
    if (options.quality !== undefined) {
      const quality = normalizeQuality(options.quality);
      const rootPc = options.rootPc === undefined ? identity.rootPc : normalizeRoot(options.rootPc).pitchClass;
      const expected = [...new Set(QUALITIES[quality].intervals.map((interval) => mod12(rootPc + interval)))]
        .sort((a, b) => a - b);
      if (!sameSet(expected, pitches)) return Object.freeze([]);
      identity = { rootPc, root: NAMES[rootPc], quality,
        label: NAMES[rootPc] + QUALITIES[quality].suffix };
    }
    const requestedBassPc = options.bassPc === undefined ? null : normalizeRoot(options.bassPc).pitchClass;
    const baseLabel = identity.label;
    const choices = openStrings.map((midi) => {
      const available = [{ fret: null, pitchClass: null }];
      for (let fret = 0; fret <= maxFret; fret += 1) {
        const pitchClass = mod12(midi + fret);
        if (pitches.includes(pitchClass)) available.push({ fret, pitchClass });
      }
      return available;
    });
    const found = new Map();
    const frets = Array(6);
    function visit(index, muted, sounding) {
      if (index === 6) {
        if (muted > maxMuted || sounding < minStrings) return;
        const played = frets.map(normalizeFret);
        const positives = played.filter((fret) => fret > 0);
        if (positives.length && Math.max(...positives) - Math.min(...positives) > maxSpan) return;
        if (positives.length && played.includes(0) && Math.max(...positives) > maxOpenFret) return;
        const actual = [...new Set(played.map((fret, string) => fret === null ? null : mod12(openStrings[string] + fret))
          .filter((value) => value !== null))].sort((a, b) => a - b);
        if (!sameSet(actual, pitches)) return;
        const key = played.map((fret) => fret === null ? "x" : fret).join("-");
        if (found.has(key)) return;
        let preliminary;
        try {
          preliminary = buildVoicing({ root: identity.root, quality: identity.quality,
            voicing: `candidate-${key}`, frets: played, tuning: openStrings, source: "generated",
            bass: requestedBassPc === null ? undefined : NAMES[requestedBassPc],
            label: baseLabel + (requestedBassPc !== null && requestedBassPc !== identity.rootPc ?
              `/${NAMES[requestedBassPc]}` : "") }, openStrings);
        } catch (error) {
          return;
        }
        if (requestedBassPc !== null && preliminary.bassPc !== requestedBassPc) return;
        found.set(key, Object.freeze({ ...preliminary, id: `candidate-${key}`,
          score: scoreCandidate(preliminary, options) }));
        return;
      }
      for (const choice of choices[index]) {
        frets[index] = choice.fret;
        visit(index + 1, muted + (choice.fret === null ? 1 : 0), sounding + (choice.fret === null ? 0 : 1));
      }
    }
    visit(0, 0, 0);
    if (sameSet(openStrings, TUNING)) {
      for (const qualityVoicings of Object.values(CHORD_LIBRARY.C)) {
        for (const voicing of Object.values(qualityVoicings)) {
          const highestFret = Math.max(voicing.startFret,
            ...voicing.frets.filter((fret) => fret !== null));
          if (!sameSet(voicing.pitchClasses, pitches) || voicing.mutedStrings.length > maxMuted ||
              highestFret > maxFret ||
              requestedBassPc !== null && voicing.bassPc !== requestedBassPc) continue;
          const key = voicing.frets.map((fret) => fret === null ? "x" : fret).join("-");
          const candidate = { ...voicing, source: "library",
            label: baseLabel + (requestedBassPc !== null && requestedBassPc !== identity.rootPc ?
              `/${NAMES[requestedBassPc]}` : "") };
          found.set(key, Object.freeze({ ...candidate, score: scoreCandidate(candidate, options) }));
        }
      }
    }
    const result = [...found.values()].sort((left, right) => left.score - right.score ||
      left.mutedStrings.length - right.mutedStrings.length || left.startFret - right.startFret ||
      left.id.localeCompare(right.id));
    return Object.freeze(result.slice(0, maxResults || result.length));
  }

  function selectCandidate(candidates, options = {}) {
    if (!Array.isArray(candidates)) throw new TypeError("Expected an array of voicing candidates");
    const filtered = candidates.filter((candidate) => {
      if (!candidate || !Array.isArray(candidate.frets)) return false;
      if (options.requireBarre && !candidate.barres.length) return false;
      if (options.requireRootInBass && candidate.bassPc !== candidate.rootPc) return false;
      if (options.maxMuted !== undefined && candidate.mutedStrings.length > options.maxMuted) return false;
      if (options.minStartFret !== undefined && candidate.startFret < options.minStartFret) return false;
      if (options.maxStartFret !== undefined && candidate.startFret > options.maxStartFret) return false;
      return true;
    });
    filtered.sort((left, right) => scoreCandidate(left, options) - scoreCandidate(right, options) ||
      left.mutedStrings.length - right.mutedStrings.length || left.startFret - right.startFret ||
      left.id.localeCompare(right.id));
    return filtered[0] || null;
  }

  function svgStart(width, height, attributes) {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" ` +
      `viewBox="0 0 ${width} ${height}" role="img" data-kind="chord-diagram" ${attributes} ` +
      'style="max-width:100%;height:auto" fill="none" color="#263330" ' +
      'font-family="system-ui,sans-serif">';
  }

  const textAt = (x, y, content, size = 11, attributes = "") =>
    `<text x="${x}" y="${y}" fill="currentColor" font-size="${size}" ${attributes}>${escapeXml(content)}</text>`;
  const line = (x1, y1, x2, y2, width = 1, attributes = "") =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="currentColor" stroke-width="${width}" ${attributes}/>`;

  function emptyChart(message) {
    return svgStart(260, 120, 'aria-label="Chord diagram unavailable" data-error="unavailable"') +
      `<title>Chord diagram unavailable</title><desc>${escapeXml(message)}</desc>` +
      textAt(130, 62, message, 13, 'text-anchor="middle"') + "</svg>";
  }

  function renderChordSvg(input, options = {}) {
    let voicing = input;
    if (!input || !Array.isArray(input.frets)) {
      voicing = getChord(input?.root, input?.quality, input?.voicing);
    } else if (!input.strings || !input.noteNames) {
      try {
        voicing = buildVoicing(input);
      } catch (error) {
        return emptyChart(error.message);
      }
    }
    if (!voicing) return emptyChart("Chord voicing not found");
    const width = Number(options.width ?? 260), left = 61, gap = 27;
    const top = 78, rowHeight = 28;
    const highest = voicing.frets.reduce((maximum, fret) => Math.max(maximum, fret || 0), voicing.startFret);
    const rows = Math.max(5, highest - voicing.startFret + 1);
    const bottom = top + rows * rowHeight, height = bottom + 42;
    const x = (string) => left + (6 - string) * gap;
    const y = (fret) => top + (fret - voicing.startFret + .5) * rowHeight;
    const title = `${voicing.label} ${voicing.quality === "major" ? "major" : voicing.quality}`;
    const description = `${title} ${voicing.voicing} voicing. O is open and X is muted.`;
    let svg = svgStart(width, height, `aria-label="${escapeXml(title)}" data-chord="${escapeXml(voicing.label)}" ` +
      `data-quality="${escapeXml(voicing.quality)}" data-voicing="${escapeXml(voicing.voicing)}" ` +
      `data-start-fret="${voicing.startFret}"`);
    svg += `<title>${escapeXml(title)}</title><desc>${escapeXml(description)}</desc>`;
    svg += textAt(width / 2, 23, voicing.label, 18, 'text-anchor="middle" font-weight="600"');
    svg += textAt(width / 2, 43, voicing.voicing, 11, 'text-anchor="middle" opacity=".72"');
    for (let string = 6; string >= 1; string -= 1) {
      svg += line(x(string), top, x(string), bottom, string === 6 || string === 1 ? 1.7 : 1,
        `data-role="string" data-string="${string}"`);
    }
    svg += line(left, top, x(1), top, voicing.startFret === 1 ? 5 : 1.7,
      voicing.startFret === 1 ? 'data-role="nut"' : `data-role="fret-line" data-fret="${voicing.startFret}"`);
    for (let row = 1; row <= rows; row += 1) {
      const fret = voicing.startFret + row;
      svg += line(left, top + row * rowHeight, x(1), top + row * rowHeight, 1,
        `data-role="fret-line" data-fret="${fret}"`);
    }
    if (voicing.startFret > 1) {
      svg += textAt(left - 19, top + rowHeight / 2 + 4, `${voicing.startFret}fr`, 12,
        `text-anchor="end" font-weight="600" data-role="start-fret" data-fret="${voicing.startFret}"`);
    }
    for (const string of voicing.strings) {
      if (string.state === "muted") {
        svg += textAt(x(string.string), top - 12, "X", 14,
          `text-anchor="middle" data-role="muted-string" data-string="${string.string}"`);
      } else if (string.state === "open") {
        svg += `<circle data-role="open-string" data-string="${string.string}" cx="${x(string.string)}" ` +
          `cy="${top - 18}" r="5.5" stroke="currentColor" stroke-width="1.7"/>`;
      }
    }
    for (const barre of voicing.barres) {
      svg += line(x(barre.fromString), y(barre.fret), x(barre.toString), y(barre.fret), 13,
        `stroke-linecap="round" data-role="barre" data-fret="${barre.fret}" ` +
        `data-from-string="${barre.fromString}" data-to-string="${barre.toString}" data-finger="${barre.finger}"`);
    }
    voicing.strings.forEach((string, index) => {
      if (string.state !== "fretted") return;
      const barred = voicing.barres.some((barre) => barre.fret === string.fret &&
        string.string <= barre.fromString && string.string >= barre.toString);
      svg += `<circle data-role="dot" data-string="${string.string}" data-fret="${string.fret}" ` +
        `data-finger="${voicing.fingers[index] ?? ""}"${barred ? ' data-barred="true"' : ""} ` +
        `cx="${x(string.string)}" cy="${y(string.fret)}" r="9" fill="currentColor"/>`;
      if (voicing.fingers[index] !== null) {
        svg += `<text x="${x(string.string)}" y="${y(string.fret) + 4}" fill="white" ` +
          `font-size="11" font-weight="600" text-anchor="middle" data-role="finger" ` +
          `data-string="${string.string}">${voicing.fingers[index]}</text>`;
      }
    });
    voicing.strings.forEach((string) => {
      svg += textAt(x(string.string), bottom + 23, string.note || "X", 10,
        `text-anchor="middle" data-role="note-name" data-string="${string.string}" ` +
        `data-note="${escapeXml(string.note || "")}"`);
    });
    return svg + "</svg>";
  }

  const api = Object.freeze({
    TUNING, NAMES, QUALITIES, CHORD_LIBRARY, STRINGS,
    normalizeRoot, normalizeQuality, parseChordLabel, normalizePitchClasses: normalizePitchSet,
    noteName, identifyChord, getVoicings, getChord, findCandidates,
    selectCandidate, renderChordSvg
  });
  root.ChordLibrary = api;
  if (typeof module === "object" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
