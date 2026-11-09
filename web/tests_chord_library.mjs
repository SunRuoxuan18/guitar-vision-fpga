import assert from "node:assert/strict";
import test from "node:test";
import "./chord_library.js";

const { TUNING, NAMES, QUALITIES, getChord, getVoicings, findCandidates, selectCandidate,
  identifyChord, normalizePitchClasses, parseChordLabel, renderChordSvg } = globalThis.ChordLibrary;
const tags = (svg) => svg.match(/<[^>]+>/g) || [];
const role = (svg, name) => tags(svg).filter((tag) => tag.includes(`data-role="${name}"`));
const valueOf = (tag, name) => new RegExp(`${name}="([^"]*)"`).exec(tag)?.[1];

test("C library covers major, minor, seventh, suspended, and added qualities", () => {
  assert.deepEqual(getVoicings("C", "major").map((item) => item.voicing), ["open", "barre8"]);
  const pitchClasses = { major: [0, 4, 7], minor: [0, 3, 7],
    dominant_seventh: [0, 4, 7, 10], major_seventh: [0, 4, 7, 11],
    minor_seventh: [0, 3, 7, 10], suspended_second: [0, 2, 7],
    suspended_fourth: [0, 5, 7], added_ninth: [0, 2, 4, 7] };
  for (const quality of Object.keys(pitchClasses)) {
    const chord = getChord("C", quality);
    assert.ok(chord, `missing C ${quality}`);
    assert.equal(chord.root, "C");
    assert.equal(chord.frets.length, 6);
    assert.equal(chord.strings.length, 6);
    assert.ok(chord.pitchClasses.every((pitchClass) => pitchClasses[quality].includes(pitchClass)));
    assert.ok(chord.pitchClasses.includes(0));
    assert.equal(chord.label, "C" + QUALITIES[quality].suffix);
  }
  assert.equal(getChord("C", "7", "open").quality, "dominant_seventh");
  assert.equal(getChord("C", "add", "open").quality, "added_ninth");
  assert.equal(getChord("C", "m7", "missing"), null);
});

test("SVG has six strings, chord states, accessible labels, and note names", () => {
  const chord = getChord("C", "major", "open");
  const svg = renderChordSvg(chord);
  assert.match(svg, /^<svg xmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
  assert.match(svg, /role="img" data-kind="chord-diagram"/);
  assert.match(svg, /<title>C major<\/title>/);
  assert.match(svg, /<desc>C major open voicing\. O is open and X is muted\.<\/desc>/);
  assert.equal(role(svg, "string").length, 6);
  assert.equal(role(svg, "nut").length, 1);
  assert.equal(role(svg, "muted-string").length, 1);
  assert.equal(role(svg, "open-string").length, 2);
  assert.equal(role(svg, "note-name").length, 6);
  assert.equal(role(svg, "finger").length, 3);
  assert.ok(svg.includes('data-string="6" data-note=""'));
  assert.ok(svg.includes('data-note="C3"'));
  assert.ok(svg.includes('data-note="E4"'));
  assert.ok(svg.endsWith("</svg>"));
});

test("high-position chart renders start fret, barre, finger numbers, and bass notes", () => {
  const chord = getChord("C", "major", "barre8");
  const svg = renderChordSvg({ root: "C", quality: "major", voicing: "barre8" });
  assert.equal(chord.startFret, 8);
  assert.equal(role(svg, "nut").length, 0);
  const start = role(svg, "start-fret")[0];
  assert.equal(valueOf(start, "data-fret"), "8");
  assert.ok(svg.includes('data-role="start-fret" data-fret="8">8fr</text>'));
  const barre = role(svg, "barre")[0];
  assert.equal(valueOf(barre, "data-fret"), "8");
  assert.equal(valueOf(barre, "data-from-string"), "6");
  assert.equal(valueOf(barre, "data-to-string"), "1");
  assert.equal(valueOf(barre, "data-finger"), "1");
  assert.equal(role(svg, "dot").length, 6);
  assert.ok(role(svg, "dot").every((tag) => tag.includes('data-barred="true"') ||
    valueOf(tag, "data-fret") !== "8"));
  assert.match(svg, /data-role="finger"[^>]*>[1-4]<\/text>/);
  assert.ok(role(svg, "note-name").some((tag) => valueOf(tag, "data-note") === "C5"));
});

test("pitch sets and tunings produce selectable candidates", () => {
  assert.deepEqual(normalizePitchClasses([48, 52, 55]), [0, 4, 7]);
  assert.deepEqual(identifyChord(["C4", "E4", "G4"]), {
    rootPc: 0, root: "C", quality: "major", label: "C"
  });
  const openCandidates = findCandidates([0, 4, 7], TUNING,
    { maxFret: 8, maxMuted: 2, maxResults: 100 });
  assert.ok(openCandidates.length > 1);
  assert.ok(openCandidates.every((candidate) => candidate.pitchClasses.join(",") === "0,4,7"));
  const selected = selectCandidate(openCandidates, { maxMuted: 1, requireRootInBass: true });
  assert.deepEqual(selected.frets, [null, 3, 2, 0, 1, 0]);
  assert.equal(selected.source, "library");

  const dCandidates = findCandidates([2, 6, 9], TUNING,
    { maxFret: 9, maxMuted: 2, maxResults: 100 });
  const dBarre = selectCandidate(dCandidates,
    { requireBarre: true, minStartFret: 5, requireRootInBass: true });
  assert.deepEqual(dBarre.frets, [null, 5, 7, 7, 7, 5]);
  assert.equal(dBarre.barres[0].fret, 5);

  const dropD = [38, 45, 50, 55, 59, 64];
  const altered = findCandidates([2, 6, 9], dropD,
    { maxFret: 5, maxMuted: 2, maxResults: 50 });
  assert.ok(altered.length > 0);
  assert.ok(altered.every((candidate) => candidate.tuning[0] === 38));
  assert.ok(altered.some((candidate) => candidate.strings.some((string) =>
    string.string === 6 && string.midi === 38 + string.fret)));
});

test("slash basses and compound suffixes keep the requested low note", () => {
  const slash = parseChordLabel("C/E");
  assert.equal(slash.quality, "major");
  assert.equal(slash.bassPc, 4);
  assert.equal(slash.label, "C/E");
  const compound = parseChordLabel("C6/9");
  assert.equal(compound.quality, "sixth_added_ninth");
  assert.equal(compound.bassPc, 0);
  assert.equal(compound.label, "C6/9");
  const candidates = findCandidates(slash.pitchClasses, TUNING, {
    rootPc: slash.rootPc, quality: slash.quality, bassPc: slash.bassPc,
    maxFret: 12, maxSpan: 5, maxMuted: 2, minStrings: 4, maxResults: 20
  });
  assert.ok(candidates.length);
  assert.ok(candidates.every(candidate => candidate.bassPc === 4));
  assert.match(renderChordSvg(candidates[0]), /data-chord="C\/E"/);
});

test("all supported roots and qualities can produce a playable diagram", () => {
  const roots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
  for (const rootPc of roots) {
    for (const [quality, definition] of Object.entries(QUALITIES)) {
      const pitchClasses = definition.intervals.map(interval => (rootPc + interval) % 12);
      const candidates = findCandidates(pitchClasses, TUNING,
        { rootPc, maxFret: 12, maxSpan: 5, maxMuted: 2, minStrings: 4,
          maxOpenFret: 5, maxResults: 5 });
      assert.ok(candidates.length, `missing ${NAMES[rootPc]} ${quality}`);
      const svg = renderChordSvg(candidates[0]);
      assert.match(svg, /data-kind="chord-diagram"/);
      assert.equal(role(svg, "note-name").length, 6);
    }
  }
});
