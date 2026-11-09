import assert from "node:assert/strict";
import { mkdir, readFile, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const { chromium } = createRequire(import.meta.url)(join(process.env.USERPROFILE,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
const screenshotDirectory = join(process.env.USERPROFILE, "Documents/Codex/neural_accel_20261027");
const source = await readFile(join(root, "vision.js"), "utf8");
assert.match(source, /function currentFingertips\(\)/);
assert.match(source, /function displayFrame\(\)/);
assert.match(source, /\}\)\(\);\s*$/);

// Expose fixture state inside the existing closure without replacing any
// production function. Different pixels distinguish accepted and raw captures.
const fixtureSource = source.replace(/\}\)\(\);\s*$/, `
  window.fingertipTrackingFixture = {
    reset(source = "board") {
      stop();
      const now = performance.now();
      Object.assign(V, {source, shape:{width:640,height:480}, clicks:null, calibration:null,
        tracking:null, result:null, handObservation:null, detect:false, boardOrigin:"camera",
        boardSequence:105, boardReceivedAt:now, boardCapturedMs:12345,
        boardProcessedAt:now, boardStream:"tracking-fixture", boardError:null,
        lastOverlayAt:-Infinity,lastOverlayKey:null,processedToken:null});
      Object.assign(PC, {stream:"tracking-fixture",generation:7,sequence:101,matchedAt:now});
      PL.frame = {source:source === "board" ? "board_camera" : "pc_camera",session:V.session,
        sequence:101,started:now-20,expiresAt:now+1800,processingMs:18,roundtripMs:25,fps:4.2};
      PL.previewAt = now;
      for (const canvas of [boardFrame,pcMatched]) {canvas.width=640;canvas.height=480;}
      boardFrame.getContext("2d").fillStyle="rgb(190,45,35)";
      boardFrame.getContext("2d").fillRect(0,0,640,480);
      pcMatched.getContext("2d").fillStyle="rgb(32,70,90)";
      pcMatched.getContext("2d").fillRect(0,0,640,480);
      this.matchedColor=[32,70,90,255];
      Object.defineProperty(video,"readyState",{value:4,configurable:true});
      Object.defineProperty(video,"videoWidth",{value:640,configurable:true});
      Object.defineProperty(video,"videoHeight",{value:480,configurable:true});
      $("pcPlEnabled").checked=true; $("autoFretboard").checked=true;
      $("visionView").value="color"; $("focusFretboard").checked=false;
      $("preprocessDetails").open=false;
      stage.dataset.mode="live";stage.dataset.visionSource=source;
      $("visionStatus").textContent="Offline fixture: accepted frame #101; raw capture #105";
      $("overlayState").textContent="Four-tip fixture";
      overlay.width=960;overlay.height=720;
      V.automatic = {stream_id:PC.stream,generation:7,sequence:101,status:"searching",
        roi_status:"searching",fingertip_neural:{status:"stale",points:null,last_result:{
          stream_id:PC.stream,generation:7,sequence:47,
          points:{index:{x:600,y:420,raw_score:2}}}},
        fingertip_tracking:{stream_id:PC.stream,generation:7,sequence:101,source_sequence:47,
          status:"tracking",keypoint_source:"optical_flow",method:"pl_pyramidal_lk_fb_patch",
          detector_age_ms:900,processing_ms:12,
          points:Object.fromEntries(fingertipNames.map((name,i)=>[name,
            {x:128+i*128,y:144+i*48,raw_score:-3+i}]))}};
      this.draw();
    },
    draw() {
      window.trackingArcs=[];window.trackingTexts=[];
      V.lastOverlayAt=-Infinity;V.lastOverlayKey=null;
      render();
      return this.read();
    },
    mutate(kind) {
      const analysis=V.automatic, tracked=analysis.fingertip_tracking;
      if(kind==="lost" || kind==="pending") tracked.status=kind;
      else if(kind==="stream") tracked.stream_id="other-stream";
      else if(kind==="sequence") tracked.sequence=100;
      else if(kind==="generation") tracked.generation=8;
      else if(kind==="analysis_stream") analysis.stream_id="other-stream";
      else if(kind==="analysis_sequence") analysis.sequence=100;
      else if(kind==="analysis_generation") analysis.generation=8;
      else if(kind==="current_stream") PC.stream="other-stream";
      else if(kind==="current_sequence") PC.sequence=105;
      else if(kind==="current_generation") PC.generation=8;
      else if(kind==="timeout") PL.frame.expiresAt=performance.now()-1;
      else if(kind==="processed_timeout") V.boardProcessedAt=performance.now()-1900;
      else if(kind==="age_expired") tracked.detector_age_ms=4001;
      else if(kind==="age_elapsed") {tracked.detector_age_ms=3950;PC.matchedAt=performance.now()-100;}
      else if(kind==="age_negative") tracked.detector_age_ms=-1;
      else if(kind==="age_nonfinite") tracked.detector_age_ms=NaN;
      else if(kind==="source_future") tracked.source_sequence=102;
      else if(kind==="source_negative") tracked.source_sequence=-1;
      else if(kind==="unknown_source") tracked.keypoint_source="affine";
      else if(kind==="empty") tracked.points={};
      else if(kind==="neural") {tracked.source_sequence=101;tracked.keypoint_source="neural";}
      else if(kind==="invalid_points") {
        tracked.points.index.x=NaN;tracked.points.middle.x=640;tracked.points.ring.y=-1;
      } else if(kind==="invisible") {
        tracked.points.middle.visible=false;tracked.points.little.visible=false;
      } else throw new Error("Unknown fixture mutation: "+kind);
      return this.draw();
    },
    partial(names) {
      const points=V.automatic.fingertip_tracking.points;
      V.automatic.fingertip_tracking.points=Object.fromEntries(names.map(name=>[name,points[name]]));
      return this.draw();
    },
    nextCapture() {
      const now=performance.now();
      PC.sequence=105;PC.matchedAt=now;PL.frame.sequence=105;
      PL.frame.started=now-20;PL.frame.expiresAt=now+1800;
      V.boardProcessedAt=now;V.boardReceivedAt=now;V.boardSequence=109;
      V.automatic.sequence=105;V.automatic.fingertip_tracking.sequence=105;
      pcMatched.getContext("2d").fillStyle="rgb(90,55,32)";
      pcMatched.getContext("2d").fillRect(0,0,640,480);
      this.matchedColor=[90,55,32,255];
      $("visionStatus").textContent="Offline fixture: accepted frame #105; raw capture #109";
      return this.draw();
    },
    read() {
      const context=overlay.getContext("2d"),box=frameBox();
      const at=(x,y)=>[...context.getImageData(Math.round(x),Math.round(y),1,1).data];
      const samples=fingertipNames.map((name,i)=>{
        const p=projectionPoint({x:(128+i*128)/640,y:(144+i*48)/480},box);
        return {name,x:p.x,y:p.y,pixel:at(p.x,p.y)};
      });
      return {current:currentFingertips(),matched:displayFrame()===pcMatched,video:displayFrame()===video,
        raw:displayFrame()===boardFrame,matchedColor:this.matchedColor,
        background:at(box.x+box.width*.86,box.y+box.height*.86),
        rawPixel:[...boardFrame.getContext("2d").getImageData(20,20,1,1).data],
        samples,arcs:window.trackingArcs,texts:window.trackingTexts,
        box,sequence:PC.sequence,rawSequence:V.boardSequence};
    }
  };
})();`);
assert.notEqual(fixtureSource, source, "Closure-local hooks must be injected");

const files = new Set(["index.html", "styles.css", "fretboard.js", "fingering.js",
  "chord_library.js", "app.js", "vision.js", "fingering_panel.js"]);
const staticFiles = new Map(await Promise.all([...files].map(async file =>
  [file, file === "vision.js" ? fixtureSource : await readFile(join(root, file))])));
const colors = {index:[255,75,75,255],middle:[255,219,88,255],
  ring:[81,229,186,255],little:[122,191,255,255]};
const reports = [], screenshots = [], requests = [], errors = [];
await mkdir(screenshotDirectory, {recursive:true});
const browser = await chromium.launch({channel:"msedge",headless:true});

function assertDrawn(value, names, description, preview = "raw") {
  assert.deepEqual(Object.keys(value.current?.points || {}), names, description);
  assert.equal(value.arcs.length, names.length, description);
  if (preview === "raw") {
    assert.equal(value.raw, true, "Board color preview must stay on the live source frame");
    assert.deepEqual(value.background, value.rawPixel, "Board preview must render the live source frame");
    assert.notDeepEqual(value.background, value.matchedColor,
      "Accepted processing pixels must not replace the live board preview");
  } else if (preview === "camera_live") {
    assert.equal(value.video, true, "PC color preview must stay on the live camera element");
    assert.equal(value.matched, false, "Recognition result must not replace the live camera");
    assert.equal(value.background[3], 0, "Camera video must remain behind a transparent annotation overlay");
  } else {
    assert.equal(value.matched, true, "Processed preview must use the accepted capture");
    assert.deepEqual(value.background, value.matchedColor, "Rendered pixels must match the accepted capture");
    assert.notDeepEqual(value.background, value.rawPixel, "A newer raw frame must not replace matched pixels");
  }
  for (const sample of value.samples) {
    const background = preview === "raw" ? value.rawPixel : value.matchedColor;
    assert.deepEqual(sample.pixel, names.includes(sample.name) ? colors[sample.name] : background,
      `${description}: ${sample.name} pixel`);
    if (names.includes(sample.name)) {
      assert.ok(value.arcs.some(([x,y,r])=>Math.abs(x-sample.x)<.01&&Math.abs(y-sample.y)<.01&&r===7),
        `${description}: ${sample.name} arc maps from 640x480 pixels`);
      assert.ok(value.texts.includes(String(Object.keys(colors).indexOf(sample.name)+1)),
        `${description}: labels preserve finger identity`);
    }
  }
}

function assertHidden(value, description) {
  assert.equal(value.current, null, description);
  assert.deepEqual(value.arcs, [], `${description}: no stale arcs`);
  assert.equal(value.texts.some(text=>text.startsWith("PL ")),false, `${description}: no stale tracking label`);
  for (const sample of value.samples) {
    assert.deepEqual(sample.pixel, value.background, `${description}: old ${sample.name} pixel erased`);
  }
}

try {
  for (const {width,height} of [{width:1280,height:900},{width:390,height:844},{width:360,height:800}]) {
    const context = await browser.newContext({viewport:{width,height},deviceScaleFactor:1,
      isMobile:width<600,hasTouch:width<600});
    const page = await context.newPage();
    page.on("pageerror", error=>errors.push({width,message:error.message}));
    await page.addInitScript(() => {
      window.trackingArcs=[];window.trackingTexts=[];
      const arc=CanvasRenderingContext2D.prototype.arc;
      CanvasRenderingContext2D.prototype.arc=function(...args){
        if(this.canvas.id==="fretOverlay")window.trackingArcs.push(args);
        return arc.apply(this,args);
      };
      const fillText=CanvasRenderingContext2D.prototype.fillText;
      CanvasRenderingContext2D.prototype.fillText=function(...args){
        if(this.canvas.id==="fretOverlay")window.trackingTexts.push(String(args[0]));
        return fillText.apply(this,args);
      };
      navigator.mediaDevices.getUserMedia=async()=>{throw new Error("Camera forbidden in offline fixture");};
    });
    await page.route("**/*", async route => {
      const url=new URL(route.request().url());
      requests.push({host:url.hostname,path:url.pathname});
      const file=url.pathname==="/"?"index.html":url.pathname.slice(1);
      if(url.origin!=="http://127.0.0.1:48792"||!files.has(file)) {
        await route.fulfill({status:503,body:"Offline fixture: no board or service access"});return;
      }
      await route.fulfill({body:staticFiles.get(file),contentType:({".html":"text/html; charset=utf-8",
        ".js":"text/javascript; charset=utf-8",".css":"text/css; charset=utf-8"})[extname(file)]});
    });
    await page.goto("http://127.0.0.1:48792/");
    await page.waitForFunction(()=>window.fingertipTrackingFixture);
    const reset=async(source="board")=>{
      await page.evaluate(source=>window.fingertipTrackingFixture.reset(source),source);
      return page.evaluate(()=>window.fingertipTrackingFixture.read());
    };
    const mutate=kind=>page.evaluate(kind=>window.fingertipTrackingFixture.mutate(kind),kind);
    const check=async(name,run)=>{await reset();await run();reports.push({width,check:name});};
    const saveScreenshot=async(state)=>{
      await page.locator("#cameraStage").scrollIntoViewIfNeeded();
      const layout=await page.locator("#cameraStage").evaluate(stage=>{
        const rect=stage.getBoundingClientRect();
        return {overflow:document.documentElement.scrollWidth>innerWidth,
          left:rect.left,right:rect.right,width:rect.width,height:rect.height,viewport:innerWidth};
      });
      assert.equal(layout.overflow,false,JSON.stringify(layout));
      assert.ok(layout.left>=0&&layout.right<=layout.viewport+1,JSON.stringify(layout));
      assert.ok(Math.abs(layout.width/layout.height-4/3)<.01,JSON.stringify(layout));
      const path=join(screenshotDirectory,`fingertip-tracking-${width<600?"mobile":"desktop"}-${width}-${state}.png`);
      await page.screenshot({path});
      const {size}=await stat(path);assert.ok(size>5000,"Screenshot must be nonblank");
      screenshots.push({width,state,path,bytes:size});
    };

    await check("board matched-frame pixels and four individually colored tips",async()=>{
      const value=await page.evaluate(()=>window.fingertipTrackingFixture.read());
      assertDrawn(value,Object.keys(colors),"four tips");
      assert.equal(value.sequence,101);assert.equal(value.rawSequence,105);
      assert.equal(value.current.source_sequence,47);
      assert.equal(value.current.keypoint_source,"optical_flow");
      await saveScreenshot("four");
    });
    await check("partial tips preserve names, labels and colors without filling missing fingers",async()=>{
      const value=await page.evaluate(()=>window.fingertipTrackingFixture.partial(["index","ring"]));
      assertDrawn(value,["index","ring"],"partial tips");
      assert.equal(value.texts.includes("2"),false);assert.equal(value.texts.includes("4"),false);
      await saveScreenshot("partial");
    });
    for (const kind of ["lost","pending","stream","sequence","generation",
      "analysis_stream","analysis_sequence","analysis_generation","current_stream",
      "current_sequence","current_generation","timeout","processed_timeout","age_expired",
      "age_elapsed","age_negative","age_nonfinite","source_future","source_negative","unknown_source","empty"]) {
      await check(`${kind} clears previously drawn points`,async()=>{
        assertHidden(await mutate(kind),kind);
      });
    }
    await check("invisible points are omitted individually",async()=>{
      assertDrawn(await mutate("invisible"),["index","ring"],"visible=false");
    });
    await check("nonfinite and out-of-frame points are omitted individually",async()=>{
      assertDrawn(await mutate("invalid_points"),["little"],"invalid coordinates");
    });
    await check("same-frame neural points retain neural provenance",async()=>{
      const value=await mutate("neural");assertDrawn(value,Object.keys(colors),"neural points");
      assert.equal(value.current.keypoint_source,"neural");assert.equal(value.current.source_sequence,101);
      assert.ok(value.texts.some(text=>text.includes("\u795e\u7ecf\u7f51\u7edc")));
      assert.equal(value.texts.some(text=>text.includes("\u6307\u5c16\u8ddf\u8e2a")),false);
    });
    await check("accepted board sequence jumps keep live pixels and preserve NN source",async()=>{
      const value=await page.evaluate(()=>window.fingertipTrackingFixture.nextCapture());
      assertDrawn(value,Object.keys(colors),"capture 101 -> 105");
      assert.equal(value.current.sequence,105);assert.equal(value.current.source_sequence,47);
      assert.equal(value.rawSequence,109);assert.deepEqual(value.background,value.rawPixel);
    });
    const pc=await reset("camera");assertDrawn(pc,Object.keys(colors),"PC live camera","camera_live");
    assertHidden(await mutate("timeout"),"PC timeout");
    reports.push({width,check:"PC camera shares matched display and timeout gates"});
    await context.close();
  }
  assert.deepEqual(errors,[],"No browser JavaScript errors");
  assert.equal(requests.some(({path})=>path.startsWith("/api/vision/")||path==="/vision/frame.jpg"),false,
    "Closure fixtures must not start camera or board polling");
  console.log(JSON.stringify({passed:reports.length,checks:reports,screenshots,page_errors:errors,
    server_started:false,board_access:false,production_source_modified:false},null,2));
} finally {
  await browser.close();
}
