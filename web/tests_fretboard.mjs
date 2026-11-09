import assert from "node:assert/strict";
import test from "node:test";
import "./fretboard.js";
const { createCalibration, transform, locateTip, mapHands, compareEvidence } = globalThis.Fretboard;
const corners = [{ x: .1, y: .2 }, { x: .9, y: .3 }, { x: .8, y: .8 }, { x: .15, y: .7 }];
const board = createCalibration(corners);

test("tracked lines need bounded age and accepted flow, and retain their provenance", () => {
  const toImage = (x,y) => transform(board.toImage,{x,y});
  const result = { status:"tracking", stage:"tracked_lattice", lattice_age_ms:100,
    flow:{inlier_ratio:.9}, corners,
    strings:Array.from({length:6},(_,i)=>({observed:false,start:toImage(0,i/5),end:toImage(1,i/5)})),
    frets:[0,.3,.56,.8,1].map(x=>({observed:false,start:toImage(x,0),end:toImage(x,1)})) };
  assert.equal(Fretboard.createObservedGeometry(result).tracked,true);
  for (const invalid of [{lattice_age_ms:601},{lattice_age_ms:-1},{stage:"lattice"},{flow:{inlier_ratio:.2}}])
    assert.throws(()=>Fretboard.createObservedGeometry({...result,...invalid}));
});

test("explicitly projected string and fret estimates remain displayable", () => {
  const toImage = (x, y) => transform(board.toImage, { x, y });
  const projected = (index, end) => ({ observed:false, estimated:true, projected:true,
    source:index < 6 ? "tracked_string_lattice" : "tracked_fret_lattice", start:end.start, end:end.end });
  const result = { status:"tracking", stage:"tracked_lattice", lattice_age_ms:1200, corners,
    flow:{inlier_ratio:.9},
    strings:Array.from({length:6},(_,i)=>projected(i,{
      start:toImage(0,i/5), end:toImage(1,i/5) })),
    frets:[0,.3,.56,.8,1].map((x,i)=>projected(6+i,{
      start:toImage(x,0), end:toImage(x,1) })) };
  const geometry = Fretboard.createObservedGeometry(result);
  assert.equal(geometry.tracked,false);
  assert.ok(geometry.strings.every(line=>line.estimated));
  assert.ok(geometry.frets.every(line=>line.estimated));
});

test("observed strings and frets map relative positions without invented MIDI or numbering", () => {
  const toImage = (x, y) => transform(board.toImage, { x, y });
  const observation = { status: "tracking", corners,
    strings: Array.from({ length: 6 }, (_, index) => ({ observed: true,
      start: toImage(0, index / 5), end: toImage(1, index / 5) })),
    frets: [0, .3, .56, .8, 1].map(x => ({ observed: true, start: toImage(x, 0), end: toImage(x, 1) })) };
  const geometry = Fretboard.createObservedGeometry(observation);
  const landmarks = Array.from({ length: 21 }, () => ({ x: -1, y: -1 }));
  landmarks[8] = toImage(.42, .6);
  const result = Fretboard.mapObservedHands(geometry, [{ score: .95, landmarks }]);
  assert.equal(result.positions.length, 1);
  const position = result.positions[0];
  assert.equal(position.string_line, 4);
  assert.equal(position.fret_interval, 2);
  assert.equal(position.relative, true);
  assert.equal(position.midi_note, undefined);
  assert.equal(compareEvidence({ ...result, receivedAt: 10, source: "camera" },
    { status: "recognized", receivedAt: 10, testMode: 0, notes: [{ midi: 60 }] }, 10).matches.length, 0);
  assert.throws(() => Fretboard.createObservedGeometry({ ...observation, strings: observation.strings.slice(1) }));
  assert.throws(() => Fretboard.createObservedGeometry({ ...observation,
    frets: observation.frets.map(item => ({ ...item, observed: false })) }));
});
function tip(fret, string) {
  const a = board.boundaries[fret - 1].x, b = board.boundaries[fret].x;
  return transform(board.toImage, { x: (a + b) / 2, y: (6 - string) / 5 });
}
test("perspective maps both orientations and nonuniform fret spacing", () => {
  for (const ordered of [corners, [...corners].reverse()]) {
    const c = createCalibration(ordered);
    for (const p of [{ x: .2, y: .1 }, { x: .7, y: .8 }]) {
      const actual = transform(c.toBoard, transform(c.toImage, p));
      assert.ok(Math.abs(actual.x - p.x) < 1e-9);
      assert.ok(Math.abs(actual.y - p.y) < 1e-9);
    }
  }
  assert.ok(board.boundaries[1].x > 1 / 12);
  assert.ok(board.boundaries[1].x > 1 - board.boundaries[11].x);
});
test("six strings and frets map to actual MIDI, including visible partial neck", () => {
  for (let string = 1; string <= 6; string++) {
    for (const fret of [1, 3, 12]) {
      const positions = locateTip(board, tip(fret, string));
      assert.equal(positions.length, 1);
      assert.equal(positions[0].string, string);
      assert.equal(positions[0].fret, fret);
      assert.equal(positions[0].midi_note, board.tuning[6 - string] + fret);
      assert.equal(positions[0].pressed, null);
    }
  }
  const partial = createCalibration(corners, 3, 8);
  const p = transform(partial.toImage, { x: partial.boundaries[1].x / 2, y: 0 });
  assert.equal(locateTip(partial, p)[0].fret, 4);
});
test("outside neck, fret wires and between strings remain unsupported or ambiguous", () => {
  assert.deepEqual(locateTip(board, { x: 0, y: 0 }), []);
  const wire = transform(board.toImage, { x: board.boundaries[3].x - .00001, y: 0 });
  assert.equal(locateTip(board, wire)[0].ambiguous, true);
  const between = transform(board.toImage, { x: .3, y: .1 });
  const positions = locateTip(board, between);
  assert.equal(positions.length, 2);
  assert.ok(positions.every((p) => p.ambiguous));
});
test("bad calibration is rejected, thumb and weak hand are not pressing evidence", () => {
  for (const pts of [[...corners.slice(0, 3)], [corners[0], corners[2], corners[1], corners[3]], corners.map(() => ({ x: .5, y: .5 }))]) {
    assert.throws(() => createCalibration(pts));
  }
  assert.throws(() => createCalibration(corners, 4, 4));
  const landmarks = Array.from({ length: 21 }, () => ({ x: 0, y: 0 }));
  landmarks[4] = tip(3, 6);
  assert.equal(mapHands(board, [{ score: .99, landmarks }]).positions.length, 0);
  landmarks[8] = tip(3, 6);
  assert.equal(mapHands(board, [{ score: .3, landmarks }]).positions.length, 0);
  assert.equal(mapHands(board, [{ score: .99, landmarks }]).positions[0].note_name, "G2");
});
test("fusion requires actual current audio, exact pitch, no synthetic or diagnostic claim", () => {
  const position = locateTip(board, tip(3, 6))[0];
  const vision = { source: "camera", receivedAt: 100, positions: [position] };
  const audio = { status: "recognized", receivedAt: 100, testMode: 0, notes: [{ midi: 43 }] };
  assert.equal(compareEvidence(vision, audio, 120).matches.length, 1);
  assert.equal(compareEvidence(vision, audio, 120).synchronized, false);
  assert.equal(compareEvidence(vision, { ...audio, testMode: 2 }, 120).status, "audio_unavailable");
  assert.equal(compareEvidence({ ...vision, source: "synthetic" }, audio, 120).status, "demo_only");
  assert.equal(compareEvidence({ ...vision, source: "video_file" }, audio, 120).status, "recording_not_synchronized");
  assert.equal(compareEvidence({ ...vision, receivedAt: NaN }, audio, 120).status, "vision_unavailable");
  assert.equal(compareEvidence(vision, { ...audio, receivedAt: 130 }, 120).status, "audio_unavailable");
  assert.equal(compareEvidence(vision, { ...audio, notes: [{ midi: 55 }] }, 120).matches.length, 0);
  assert.equal(compareEvidence(vision, { ...audio, notes: [{ midi: 43, ambiguous: true }] }, 120).matches.length, 0);
  assert.equal(compareEvidence({ ...vision, source: "board_camera" }, audio, 120).matches.length, 1);
  assert.equal(compareEvidence({ ...vision, source: "board_synthetic" }, audio, 120).status, "demo_only");
  assert.equal(compareEvidence(vision, audio, 1000).status, "vision_unavailable");
  assert.equal(compareEvidence(vision, { ...audio, stale: true }, 120).matches.length, 0);
  assert.equal(compareEvidence(vision, { ...audio, age_seconds: 2.5 }, 120).matches.length, 0);
  assert.equal(compareEvidence(vision, { ...audio, age_seconds: .74 }, 120).matches.length, 0);
  assert.equal(compareEvidence(vision, { ...audio, age_seconds: .7 }, 120).matches.length, 1);
});
