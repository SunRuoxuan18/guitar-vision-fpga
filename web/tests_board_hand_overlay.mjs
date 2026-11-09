import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { mkdtemp } from "node:fs/promises";

const { chromium } = createRequire(import.meta.url)(join(process.env.USERPROFILE,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
const base = process.argv[2] || "http://127.0.0.1:8001/";
const browser = await chromium.launch({ headless: true, channel: "msedge" });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const jpeg = await page.evaluate(() => {
    const c = document.createElement("canvas"); c.width = 640; c.height = 480;
    const x = c.getContext("2d"); x.fillStyle = "#363c40"; x.fillRect(0, 0, 640, 480);
    x.fillStyle = "#758078"; x.fillRect(240, 120, 300, 300);
    return c.toDataURL("image/jpeg", .9).split(",")[1];
  });
  let cameraSequence = 0, processRequests = 0, legacyPolls = 0, legacySearches = 0;
  let noteMode = null, epoch = 1, lastOptions;
  await page.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.pathname === "/vision/frame.jpg") {
      if (url.searchParams.has('auto')) legacySearches++;
      cameraSequence++;
      await route.fulfill({ contentType: "image/jpeg", body: Buffer.from(jpeg, "base64"), headers: {
        "X-Vision-Sequence": String(cameraSequence), "X-Vision-Captured-Ms": String(performance.now()),
        "X-Vision-Source": "camera", "X-Vision-Stream": "fixture-c922" } });
      return;
    }
    if (["/api/vision/pc/status", "/api/vision/pl/status"].includes(url.pathname)) {
      if (url.pathname === "/api/vision/pl/status") legacyPolls++;
      await route.fulfill({ json: { link_speed_mbps: 1000 } }); return;
    }
    if (url.pathname === "/api/vision/pc/process") {
      processRequests++;
      const headers = route.request().headers(), options = JSON.parse(headers["x-vision-options"]);
      lastOptions = options;
      const points = Object.fromEntries(["index", "middle", "ring", "little"].map((name, i) =>
        [name, { x: 325 + i * 24, y: 220 + i * 34, raw_score: 1.2 }]));
      const analysis = { stream_id: options.stream, generation: options.generation,
        sequence: options.sequence, processing_location: "ps_linux", status: "tracking", roi_status: "tracking",
        roi_corners: [{x:.3,y:.3},{x:.75,y:.3},{x:.75,y:.8},{x:.3,y:.8}],
        corners: [{x:.3,y:.3},{x:.75,y:.3},{x:.75,y:.8},{x:.3,y:.8}],
        strings: Array.from({length:6},(_,i)=>({start:{x:.3,y:.35+i*.07},end:{x:.75,y:.35+i*.07},observed:true})),
        frets: Array.from({length:6},(_,i)=>({start:{x:.32+i*.08,y:.3},end:{x:.32+i*.08,y:.8},observed:true})),
        stage: "tracked_lattice", lattice_epoch: epoch, lattice_age_ms: 20, flow: {inlier_ratio:1},
        hand_region: { mode: "neck_local_lightweight", status: "tracking", stream_id: options.stream,
          generation: options.generation, sequence: options.sequence,
          regions: [{ corners: [{x:.48,y:.36},{x:.79,y:.36},{x:.79,y:.83},{x:.48,y:.83}] }] },
        fingertip_tracking: {status:"tracking",stream_id:options.stream,generation:options.generation,
          sequence:options.sequence,source_sequence:Math.max(0,options.sequence-1),
          keypoint_source:"optical_flow",method:"pl_pyramidal_lk_fb_patch",
          detector_age_ms:730,processing_ms:32,points},
        fingertip_neural: { status: "ok", current_source: {stream_id:options.stream,
          generation:options.generation, sequence:options.sequence}, max_age_ms:1800,
          last_result: { stream_id:options.stream, generation:options.generation,
            sequence:Math.max(0, options.sequence - (processRequests > 1 ? 1 : 0)),
            roi:[300,190,150,190], age_ms:20, inference_ms:730, points } } };
      if (noteMode) analysis.fingertip_notes = {
        status: noteMode, source_sequence: noteMode === "historical" ? options.sequence-1 : options.sequence,
        stream_id: options.stream, generation: options.generation, age_ms: 2300,
        projection: {status: "note_candidates", processing_location: "ps_linux", positions: [
          {finger: "index", x:.55,y:.5,relative:false,string:5,fret:2,midi_note:47,note_name:"B2",
            ambiguous:false,pressed:null,sounding:null}]}};
      await route.fulfill({ contentType: options.coordinates_only ? "application/json" : "application/x-fretlens-planes",
        body: options.coordinates_only ? Buffer.from("{}") : Buffer.alloc(320*240*2, 90), headers: {
        "X-Vision-Backend":"pl", "X-Vision-Source":"pc_camera", "X-Vision-Token":headers["x-vision-token"],
        "X-Vision-Processing-Ms":"760", "X-Vision-Fretboard":JSON.stringify(analysis) } });
      return;
    }
    if (url.origin === new URL(base).origin) return route.continue();
    await route.abort();
  });
  await page.goto(new URL("/?camera=board", base).href);
  await page.waitForFunction(() => window.FretVision.metrics().pl_updates_per_second > 0);
  await page.waitForFunction(() => {
    const canvas = document.querySelector("#fretOverlay"), data = canvas.getContext("2d").getImageData(0,0,canvas.width,canvas.height).data;
    let red=0, yellow=0, mint=0, blue=0, neck=0;
    for(let i=0;i<data.length;i+=4){
      if(data[i]>220&&data[i+1]<90&&data[i+2]<90) red++;
      if(data[i]>220&&data[i+1]>170&&data[i+2]<120) yellow++;
      if(data[i]<120&&data[i+1]>180&&data[i+2]>140) mint++;
      if(data[i]<150&&data[i+1]>150&&data[i+2]>200) blue++;
      if(data[i]>85&&data[i]<120&&data[i+1]>195&&data[i+2]>190) neck++;
    }
    return red>100&&yellow>8&&mint>8&&blue>8&&neck>100;
  }, null, { timeout: 8000 });
  assert.ok(processRequests > 0, "board camera frames reach the PL vision bridge");
  assert.equal(legacyPolls, 0, "board camera does not poll obsolete 8083 preview endpoint");
  assert.equal(legacySearches, 0, "live camera does not run a duplicate whole-frame search");
  const region = await page.evaluate(() => window.FretVision.trackingSnapshot().display_region);
  assert.deepEqual(region, [{x:.48,y:.36},{x:.79,y:.36},{x:.79,y:.83},{x:.48,y:.83}]);
  await page.evaluate(() => {
    document.querySelector("#debugDetails").open = true;
    document.querySelector("#calibrationSettings").open = true;
  });
  await page.click("#applyNoteCalibration");
  await page.waitForFunction(() => document.querySelector("#noteCalibrationStatus").textContent.includes("已应用"));
  await page.waitForTimeout(800);
  assert.equal(lastOptions.note_calibration.first_boundary_fret, 0);
  assert.equal(lastOptions.note_calibration.lattice_epoch, 1);
  noteMode = "historical";
  await page.waitForFunction(() => document.querySelector("#visualEvidence").textContent.includes("历史源帧") &&
    document.querySelector("#visualEvidence").textContent.includes("B2"));
  noteMode = "current";
  await page.waitForFunction(() => document.querySelector("#visualEvidence").textContent.includes("B2") &&
    !document.querySelector("#visualEvidence").textContent.includes("历史源帧"));
  epoch = 2;
  await page.waitForFunction(() => document.querySelector("#noteCalibrationStatus").textContent.includes("重新确认"));
  await page.waitForTimeout(800);
  assert.equal(lastOptions.note_calibration, undefined, "grid change invalidates musical calibration");
  const screenshots = await mkdtemp(join(tmpdir(), "fingertip-notes-"));
  await page.screenshot({path:join(screenshots,"desktop.png"), fullPage:true});
  await page.setViewportSize({width:360,height:800});
  await page.locator("#noteCalibrationFields").scrollIntoViewIfNeeded();
  await page.screenshot({path:join(screenshots,"mobile.png"), fullPage:true});
  const overflow = await page.locator("#noteCalibrationFields").evaluate(element =>
    element.scrollWidth > element.clientWidth+1);
  assert.equal(overflow, false, "musical calibration fields fit mobile width");
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ board_camera_submissions:processRequests, legacy_8083_polls:legacyPolls,
    hand_region:region, visible_colors:{green_neck:true,red_roi:true,index:true,middle:true,ring:true,little:true},
    page_errors:errors, screenshots }, null, 2));
} finally { await browser.close(); }
