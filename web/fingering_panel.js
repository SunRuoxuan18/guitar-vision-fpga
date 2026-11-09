/* Present observations separately from musical reference shapes. */
(function () {
  "use strict";
  const F = window.GuitarFingering;
  const C = window.ChordLibrary;
  if (!F || !C) return;
  const $ = id => document.getElementById(id);
  let signature = null, snapshot = null;
  const candidateCache = new Map();
  const clear = id => { $(id).replaceChildren(); };

  function currentPitchClasses(audio, label) {
    const heard = [...new Set((audio?.notes || [])
      .filter(note => Number.isInteger(note.midi) && note.midi >= 0 && note.midi <= 127)
      .map(note => ((note.midi % 12) + 12) % 12))].sort((a, b) => a - b);
    if (heard.length >= 2) return heard;
    const parsed = C.parseChordLabel(label);
    return parsed ? [...parsed.pitchClasses].sort((a, b) => a - b) : [];
  }

  function generatedCandidates(pitchClasses, chord, maxFret) {
    const key = `${pitchClasses.join(",")}|${chord?.rootPc ?? "auto"}|` +
      `${chord?.quality ?? "auto"}|${chord?.bassPc ?? "auto"}|${maxFret}`;
    if (!candidateCache.has(key)) {
      candidateCache.set(key, C.findCandidates(pitchClasses, C.TUNING, {
        rootPc: chord?.rootPc, quality: chord?.quality, bassPc: chord?.bassPc,
        maxFret, maxSpan: 5, maxMuted: 2, minStrings: 4, maxOpenFret: 5, maxResults: 200
      }));
      if (candidateCache.size > 32) candidateCache.delete(candidateCache.keys().next().value);
    }
    return candidateCache.get(key);
  }

  function rankCandidates(candidates, observed) {
    const absolute = observed.absoluteFretsKnown
      ? observed.positions.filter(position => !position.ambiguous) : [];
    if (!absolute.length) {
      return { candidate: C.selectCandidate(candidates, {
        maxMuted: 2, requireRootInBass: true
      }) || candidates[0] || null, exact: 0, distance: null };
    }
    const observedStart = Math.min(...absolute.map(position => position.fret));
    const ranked = candidates.map(candidate => {
      let exact = 0, audioSupported = 0, weightedExact = 0, distance = 0, covered = 0;
      for (const position of absolute) {
        const fret = candidate.frets[6 - position.string];
        if (fret === position.fret) {
          exact += 1;
          audioSupported += position.audioSupported ? 1 : 0;
          weightedExact += position.audioSupported ? 5 : 2;
          covered += 1;
        } else if (fret === null) {
          distance += position.audioSupported ? 6 : 3;
        } else {
          distance += Math.min(6, Math.abs(fret - position.fret));
        }
      }
      const startDistance = Math.abs(candidate.startFret - observedStart);
      const score = -20 * weightedExact + 4 * (absolute.length - covered) + 2.5 * distance +
        1.5 * startDistance + (candidate.score || 0) * .08;
      return { candidate, exact, audioSupported, weightedExact, distance, score };
    }).sort((left, right) => left.score - right.score ||
      right.exact - left.exact || left.candidate.id.localeCompare(right.candidate.id));
    return ranked[0] || { candidate: candidates[0] || null, exact: 0, distance: null };
  }

  function chooseReference(audio, label, observed) {
    if (!label) return { candidate: null, exact: 0, distance: null, candidates: [] };
    const parsed = C.parseChordLabel(label);
    const pitchClasses = currentPitchClasses(audio, label);
    if (!pitchClasses.length) return { candidate: null, exact: 0, distance: null, candidates: [] };
    const absoluteFrets = observed.absoluteFretsKnown
      ? observed.positions.filter(position => !position.ambiguous).map(position => position.fret) : [];
    const maxFret = Math.max(12, Math.min(24, (absoluteFrets.length ? Math.max(...absoluteFrets) : 0) + 5));
    const candidates = generatedCandidates(pitchClasses, parsed, maxFret);
    return { ...rankCandidates(candidates, observed), candidates };
  }

  function update() {
    const audio = window.FretAudioSnapshot?.();
    const vision = window.FretVision?.snapshot();
    const now = performance.now();
    const observed = F.fuseEvidence(vision, audio, now);
    const currentAudio = audio && !audio.stale && audio.testMode === 0 &&
      Number.isFinite(audio.receivedAt) && now >= audio.receivedAt && now - audio.receivedAt <= 1000 &&
      (audio.age_seconds === undefined || (Number.isFinite(audio.age_seconds) && audio.age_seconds >= 0 &&
        Number.isFinite(audio.ageReportedAt ?? audio.receivedAt) && now >= (audio.ageReportedAt ?? audio.receivedAt) &&
        audio.age_seconds * 1000 + now - (audio.ageReportedAt ?? audio.receivedAt) <= 1000)) &&
      ["recognized", "unknown"].includes(audio.status);
    const label = currentAudio ? audio.chordLabel : null;
    const selection = chooseReference(currentAudio ? audio : null, label, observed);
    const reference = selection.candidate;
    const candidates = currentAudio && audio.status === "recognized" ? audio.scaleCandidates || [] : [];
    const scale = candidates.map(candidate => F.scaleFretmap(candidate.tonic, candidate.scale,
      { firstFret: 0, lastFret: 12 })).find(Boolean) || null;
    snapshot = { observed, reference, scale, alternatives: selection.candidates.length,
      exactVisualMatches: selection.exact, source: "reception_time_fusion" };
    const next = JSON.stringify({ observed: { status: observed.status, positions: observed.positions,
      relativePositions: observed.relativePositions }, label, reference: reference?.frets,
      referenceStart: reference?.startFret, scale: scale?.label,
      held: currentAudio && audio.heldStatus, candidateCount: candidates.length });
    if (next === signature) return;
    signature = next;
    const count = observed.positions.length + observed.relativePositions.length;
    if (count) $("observedFingering").innerHTML = F.renderChordChart(observed);
    else clear("observedFingering");
    $("observedFingeringNote").textContent = observed.matches.length
      ? `${observed.matches.length} 个位置获音频支持；未观察的弦保留 ?，压弦仍需确认。`
      : observed.relativePositions.length ? "显示相对弦线与品格，绝对弦号和品号尚未确定。"
      : count ? "已画出指尖投影；暂未获得相同音高支持。"
      : ({ demo_only: "演示数据不作为本次演奏的证据。", recording_not_synchronized: "录像与实时音频未同步。",
        vision_unavailable: "等待可靠的指板位置与指尖。" }[observed.status] || "尚无可靠指尖位置。");
    if (reference) $("referenceFingering").innerHTML = C.renderChordSvg(reference);
    else clear("referenceFingering");
    $("referenceFingeringNote").textContent = reference
      ? `${reference.label} · 起始 ${reference.startFret} 品 · ${selection.candidates.length} 个弦品候选` +
        `${selection.exact ? `，${selection.exact} 个观察位置与此指型一致` : ""}` +
        `${selection.audioSupported ? `（${selection.audioSupported} 处同时获音频支持）` : ""}。` +
        `${audio.heldStatus ? "延音或换和弦确认中。" : ""}图示为选出的参考指型，不代表每个压弦都已确认。`
      : label ? `${label} 暂无此类型的参考图；音频识别结果仍保留。` : "等待和弦识别。";
    if (scale) $("scaleFingering").innerHTML = F.renderScaleFretmap(scale);
    else clear("scaleFingering");
    $("scaleFingeringNote").textContent = scale ? `${scale.label} · ${candidates.length} 个调式候选中显示一个，主音尚未确定。`
      : "等待连续音符；单个和弦不能唯一确定调性。";
    $("fingeringState").textContent = observed.matches.length ? "音视频支持部分位置"
      : count ? "视觉位置候选" : reference ? "仅有音频参考" : "等待音视频";
  }
  window.FretFingering = { update, snapshot: () => snapshot };
  update();
  const timer = window.setInterval(update, 100);
  window.addEventListener("beforeunload", () => window.clearInterval(timer));
})();
