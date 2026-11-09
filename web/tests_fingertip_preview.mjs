import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const { chromium } = createRequire(import.meta.url)(join(process.env.USERPROFILE,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
const source = await readFile(join(root, "vision.js"), "utf8");
const fixtureSource = source.replace('lastDone = now; PL.status = boardSource ? "C922 → PS/PL" : "电脑采集 → PL";',
  'lastDone = now; PL.status = boardSource ? "C922 → PS/PL" : "电脑采集 → PL"; if (window.freezePreviewPolling) PL.generation += 1;')
  .replace(/\}\)\(\);\s*$/, `
  const jpegCanvas = document.createElement("canvas");
  jpegCanvas.width = jpegCanvas.height = 208;
  jpegCanvas.getContext("2d").fillStyle = "rgb(32,70,90)";
  jpegCanvas.getContext("2d").fillRect(0, 0, 208, 208);
  const jpeg = jpegCanvas.toDataURL("image/jpeg").split(",")[1];
  window.fingertipFixture = {
    reset() {
      clearFingertipPreview(); V.source = "camera"; V.handObservation = null; V.result = null;
      PC.generation = 7; PL.frame = {fps:24}; $("pcPlEnabled").checked = true;
      overlay.getContext("2d").fillStyle = "rgb(20,30,40)";
      overlay.getContext("2d").fillRect(0, 0, overlay.width, overlay.height);
      window.previewArcs = []; window.previewImageCount = 0;
    },
    snapshot(sequence = 11, sourceSequence = 10) {
      return {status: "ok", stream_id: null, generation: null, sequence: null,
        current_source: {stream_id:"fixture-stream", generation:7, sequence},
        max_age_ms:250, running:false, pending:false, dropped:0,
        last_result: {stream_id:"fixture-stream", generation:7, sequence:sourceSequence,
          roi:[100,50,200,100], age_ms:80, inference_ms:73.5, completed_monotonic:12,
          points: {index:{x:125,y:75,raw_score:1.25}, middle:{x:175,y:85,raw_score:0.1},
            ring:{x:225,y:100,raw_score:-0.5}, little:{x:275,y:125,raw_score:2.5}},
          preview:{jpeg_base64:jpeg, mime_type:"image/jpeg", width:208, height:208}}};
    },
    update(snapshot, sequence = 11, stream = "fixture-stream", generation = 7) {
      updateFingertipPreview(snapshot, stream, generation, sequence, performance.now()-20);
    },
    render(elapsed = 0) {renderFingertipPreview(performance.now()+elapsed);},
    ageBy(ms) {fingertipPreview.ageAt -= ms;},
    rateBy(ms) {fingertipPreview.completions[0] -= ms;},
    trackingFps(value) {PL.frame.fps=value;},
    generation(value) {PC.generation = value;},
    clear: clearPl,
    read() {
      return {state:$("fingertipPreview").dataset.state,
        hidden:$("fingertipPreviewFigure").hidden, key:$("fingertipPreviewCanvas").dataset.resultKey,
        status:$("fingertipPreviewStatus").textContent, age:$("fingertipPreviewAge").textContent,
        graph:$("fingertipPreviewGraph").textContent, inference:$("fingertipPreviewInference").textContent,
        rate:$("fingertipPreviewRate").textContent, tracking:$("fingertipPreviewTracking").textContent,
        note:$("fingertipPreviewNote").textContent, points:$("fingertipPreviewPoints").textContent,
        completions:fingertipPreview.completions.length, images:window.previewImageCount,
        livePixel:[...overlay.getContext("2d").getImageData(50,50,1,1).data],
        roiPixel:[...$("fingertipPreviewCanvas").getContext("2d").getImageData(10,10,1,1).data],
        arcs:window.previewArcs, hand:V.handObservation, result:V.result};
    }
  };
})();`);
assert.notEqual(fixtureSource, source, "Fixture exposes only closure-local test hooks");

const browser = await chromium.launch({ channel: "msedge", headless: true });
const reports = [];
export const previewScreenshots = [];
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [], requests = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    window.previewArcs = []; window.previewImageCount = 0;
    window.holdPreviewImages = false; window.pendingPreviewImages = [];
    const arc = CanvasRenderingContext2D.prototype.arc;
    CanvasRenderingContext2D.prototype.arc = function (...args) {
      if (this.canvas.id === "fingertipPreviewCanvas") window.previewArcs.push(args);
      return arc.apply(this, args);
    };
    const NativeImage = window.Image;
    const descriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
    window.Image = function (...args) {
      const image = new NativeImage(...args); window.previewImageCount++;
      Object.defineProperty(image, "src", { get() {return descriptor.get.call(image);},
        set(value) {
          const assign = () => descriptor.set.call(image, value);
          if (window.holdPreviewImages) window.pendingPreviewImages.push(assign); else assign();
        }});
      return image;
    };
    navigator.mediaDevices.getUserMedia = async () => {throw new Error("Camera forbidden in fixture");};
  });
  // All resources are fulfilled from disk in-process. No server or board is contacted.
  await page.route("**/*", async route => {
    const url = new URL(route.request().url()); requests.push(url.pathname);
    if (url.hostname !== "127.0.0.1" || url.pathname.startsWith("/api/")) {
      await route.fulfill({ status: 503, body: "Offline fixture" }); return;
    }
    const files = new Set(["index.html", "styles.css", "fretboard.js", "fingering.js",
      "chord_library.js", "app.js", "vision.js", "fingering_panel.js"]);
    const file = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
    if (!files.has(file)) {await route.fulfill({ status:404 }); return;}
    const body = file === "vision.js" ? fixtureSource : await readFile(join(root, file));
    await route.fulfill({ body, contentType: ({".html":"text/html; charset=utf-8",
      ".js":"text/javascript; charset=utf-8", ".css":"text/css; charset=utf-8"})[extname(file)] });
  });
  await page.goto("http://127.0.0.1:48791/");
  await page.waitForFunction(() => window.fingertipFixture);
  const reset = () => page.evaluate(() => window.fingertipFixture.reset());
  const read = () => page.evaluate(() => window.fingertipFixture.read());
  const rendered = () => page.waitForFunction(() => !document.querySelector("#fingertipPreviewFigure").hidden);
  const check = async (name, run) => {await reset(); await run(); reports.push(name);};

  await check("pending identity comes from current_source, not null top-level fields", async () => {
    await page.evaluate(() => {
      const f=window.fingertipFixture, snapshot=f.snapshot();
      snapshot.last_result=null; snapshot.status="pending"; snapshot.running=true; f.update(snapshot);
    });
    const value=await read(); assert.equal(value.state,"running"); assert.equal(value.hidden,true);
    assert.equal(value.status,"PL 推理中");
  });
  await check("idle clears image without claiming an inference", async () => {
    await page.evaluate(() => {
      const f=window.fingertipFixture, snapshot=f.snapshot(); snapshot.last_result=null;
      snapshot.status="idle"; f.update(snapshot);
    });
    assert.equal((await read()).state,"waiting"); assert.equal((await read()).completions,0);
  });
  await check("worker sequence mismatch is labeled historical while current_source gates identity", async () => {
    await page.evaluate(() => {
      const f=window.fingertipFixture,s=f.snapshot(11,10); s.status="sequence_mismatch";
      s.last_result.stale=true; s.last_result.stale_reasons=["sequence"]; f.update(s);
    }); await rendered();
    const value=await read(); assert.equal(value.state,"stale"); assert.match(value.status,/仅历史参考/);
  });
  for (const status of ["stopping", "stopped"]) {
    await check(`worker ${status} status remains visible without claiming inference`, async () => {
      await page.evaluate(status => {
        const f=window.fingertipFixture,s=f.snapshot(); s.last_result=null; s.status=status; f.update(s);
      },status);
      assert.equal((await read()).state,status); assert.equal((await read()).hidden,true);
    });
  }
  await check("source JPEG and exactly four full-pixel predictions stay off live overlay", async () => {
    await page.evaluate(() => {const f=window.fingertipFixture; f.update(f.snapshot());});
    await rendered();
    const value=await read();
    assert.equal(value.state,"stale"); assert.match(value.status,/仅历史参考/);
    assert.deepEqual(value.livePixel,[20,30,40,255]); assert.equal(value.hand,null); assert.equal(value.result,null);
    assert.equal(value.arcs.length,4); assert.deepEqual(value.arcs.map(a=>a.slice(0,2)),[[26,52],[78,72.8],[130,104],[182,156]]);
    assert.ok(Math.abs(value.roiPixel[0]-32)<4 && Math.abs(value.roiPixel[2]-90)<4);
    assert.match(value.points,/1\.250/); assert.match(value.points,/-0\.500/);
    assert.match(value.note,/原始峰值非置信度/); assert.match(value.note,/非完整采集延迟/);
    assert.match(value.age,/板端 ROI ≤\d+ ms/); assert.equal(value.graph,"未提供");
    assert.equal(value.inference,"73.5 ms"); assert.equal(value.tracking,"24.0 Hz");
    const dotColors=await page.evaluate(() => {
      const ctx=document.querySelector("#fingertipPreviewCanvas").getContext("2d");
      return [[26,52],[78,73],[130,104],[182,156]].map(([x,y])=>[...ctx.getImageData(x,y,1,1).data]);
    });
    assert.deepEqual(dotColors,[[255,75,75,255],[255,219,88,255],[81,229,186,255],[122,191,255,255]]);
  });
  await check("duplicate snapshots do not decode or count results twice", async () => {
    await page.evaluate(() => {const f=window.fingertipFixture; f.update(f.snapshot());}); await rendered();
    await page.evaluate(() => {const f=window.fingertipFixture; f.update(f.snapshot()); f.update(f.snapshot());});
    const value=await read(); assert.equal(value.images,1); assert.equal(value.completions,1); assert.equal(value.arcs.length,4);
  });
  await check("NN receive update rate is separate from tracking FPS", async () => {
    await page.evaluate(() => {
      const f=window.fingertipFixture; f.update(f.snapshot()); f.rateBy(500);
      f.update(f.snapshot(12,11),12);
    }); await rendered();
    const value=await read(); assert.equal(value.completions,2); assert.match(value.rate,/Hz · 实收/);
    assert.ok(parseFloat(value.rate)>1 && parseFloat(value.rate)<3); assert.equal(value.tracking,"24.0 Hz");
  });
  await check("production replay metrics remain separate in the UI", async () => {
    await page.evaluate(() => {
      const f=window.fingertipFixture,s=f.snapshot();
      s.last_result.inference_ms=1339.5870730000752; f.trackingFps(4.041237113402062); f.update(s);
    }); await rendered();
    const value=await read();
    assert.equal(value.inference,"1339.6 ms"); assert.equal(value.tracking,"4.0 Hz");
    assert.equal(value.graph,"未提供");
  });
  for (const kind of ["stream", "generation", "current_stream", "current_generation", "current_sequence", "future", "nan", "roi"]) {
    await check(`reject ${kind} without displaying its pixels`, async () => {
      await page.evaluate(kind => {
        const f=window.fingertipFixture, s=f.snapshot();
        if(kind==="stream") s.last_result.stream_id="other";
        if(kind==="generation") s.last_result.generation=8;
        if(kind==="current_stream") s.current_source.stream_id="other";
        if(kind==="current_generation") s.current_source.generation=8;
        if(kind==="current_sequence") s.current_source.sequence=9;
        if(kind==="future") s.last_result.sequence=12;
        if(kind==="nan") s.last_result.points.index.x=NaN;
        if(kind==="roi") s.last_result.roi[2]=0;
        f.update(s);
      },kind);
      const value=await read(); assert.equal(value.hidden,true); assert.equal(value.key,undefined);
      assert.equal(value.images,0); assert.deepEqual(value.livePixel,[20,30,40,255]);
    });
  }
  await check("older result cannot replace a newer completion", async () => {
    await page.evaluate(() => {const f=window.fingertipFixture; f.update(f.snapshot(12,11),12);}); await rendered();
    await page.evaluate(() => {const f=window.fingertipFixture; f.update(f.snapshot(13,10),13);});
    assert.equal((await read()).hidden,true);
  });
  await check("age expiry remains explicitly historical while tracking is alive", async () => {
    await page.evaluate(() => {const f=window.fingertipFixture; f.update(f.snapshot(11,11));}); await rendered();
    assert.equal((await read()).state,"ready");
    await page.evaluate(() => {const f=window.fingertipFixture; f.ageBy(350); f.render();});
    const value=await read(); assert.equal(value.state,"stale"); assert.equal(value.hidden,false);
    assert.match(value.status,/仅历史参考/);
  });
  for (const kind of ["generation", "timeout", "disconnect", "stop", "disabled", "fault"]) {
    await check(`${kind} clears history and invalidates image`, async () => {
      await page.evaluate(() => {const f=window.fingertipFixture; f.update(f.snapshot());}); await rendered();
      await page.evaluate(kind => {
        const f=window.fingertipFixture;
        if(kind==="generation") {f.generation(8); f.render();}
        if(kind==="timeout") f.render(1100);
        if(kind==="disconnect") f.clear("fixture disconnected");
        if(kind==="stop") window.FretVision.stop();
        if(kind==="disabled") {document.querySelector("#pcPlEnabled").checked=false; f.render();}
        if(kind==="fault") {const s=f.snapshot(); s.status="fault"; f.update(s);}
      },kind);
      const value=await read(); assert.equal(value.hidden,true); assert.equal(value.key,undefined);
      assert.deepEqual(value.roiPixel,[0,0,0,0]); assert.equal(value.rate,"--");
    });
  }
  await check("delayed JPEG decode cannot restore a disconnected preview", async () => {
    await page.evaluate(() => {
      const f=window.fingertipFixture; window.holdPreviewImages=true;
      f.update(f.snapshot()); f.clear("fixture disconnected"); window.holdPreviewImages=false;
      window.pendingPreviewImages.splice(0).forEach(assign=>assign());
    });
    await page.waitForTimeout(80); assert.equal((await read()).hidden,true); assert.equal((await read()).key,undefined);
  });
  await check("bad JPEG is reported without modifying video", async () => {
    await page.evaluate(() => {
      const f=window.fingertipFixture, s=f.snapshot(); s.last_result.preview.jpeg_base64="AAAA"; f.update(s);
    });
    await page.waitForFunction(() => document.querySelector("#fingertipPreviewNote").textContent.includes("源图不可用"));
    const value=await read(); assert.equal(value.hidden,true); assert.deepEqual(value.livePixel,[20,30,40,255]);
  });
  await check("missing JPEG is explicit, not replaced by current ROI", async () => {
    await page.evaluate(() => {const f=window.fingertipFixture,s=f.snapshot(); delete s.last_result.preview; f.update(s);});
    const value=await read(); assert.equal(value.hidden,true); assert.match(value.note,/源图未提供/);
  });
  for (const width of [1280,390,360]) {
    await page.setViewportSize({width,height:900});
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await reset();
    await page.evaluate(() => {const f=window.fingertipFixture; f.update(f.snapshot());}); await rendered();
    const layout=await page.locator("#fingertipPreview").evaluate(panel => {
      const figure=panel.querySelector("figure").getBoundingClientRect(), details=panel.querySelector(".fingertip-preview-details").getBoundingClientRect();
      const cells=[...panel.querySelectorAll("dl > div")].map(node=>node.getBoundingClientRect());
      return {overflow:document.documentElement.scrollWidth>innerWidth,
        overlap:figure.right>details.left+1 && figure.bottom>details.top+1,
        cellsOverlap:cells.some((a,i)=>cells.slice(i+1).some(b=>a.left<b.right-1 && a.right>b.left+1 && a.top<b.bottom-1 && a.bottom>b.top+1)),
        panelWidth:panel.getBoundingClientRect().width, imageWidth:figure.width};
    });
    assert.equal(layout.overflow,false,JSON.stringify({width,...layout})); assert.equal(layout.overlap,false);
    assert.equal(layout.cellsOverlap,false); assert.ok(layout.imageWidth>=150 && layout.imageWidth<=160,JSON.stringify({width,...layout,preview:await read()}));
    const screenshot=await page.locator("#fingertipPreview").screenshot(); assert.ok(screenshot.length>5000);
    previewScreenshots.push({width,bytes:screenshot});
    reports.push(`responsive nonblank preview ${width}px`);
  }
  assert.deepEqual(errors,[]); assert.equal(requests.some(path=>path.startsWith("/api/")),false);

  await check("accepted camera response forwards history without changing synchronized capture or boxes", async () => {
    let releaseReply, responseOptions;
    await page.route("**/api/vision/pc/status", route=>route.fulfill({json:{link_speed_mbps:1000}}));
    await page.route("**/api/vision/pc/process", async route => {
      const options=JSON.parse(route.request().headers()["x-vision-options"]); responseOptions=options;
      await new Promise(resolve=>{releaseReply=resolve;});
      const neural=await page.evaluate(options => {
        const s=window.fingertipFixture.snapshot(options.sequence,options.sequence);
        s.current_source={stream_id:options.stream,generation:options.generation,sequence:options.sequence};
        Object.assign(s.last_result,s.current_source); return s;
      },options);
      const corners=[{x:.4,y:.35},{x:.8,y:.35},{x:.8,y:.65},{x:.4,y:.65}];
      const handCorners=[{x:.55,y:.3},{x:.75,y:.3},{x:.75,y:.7},{x:.55,y:.7}];
      const analysis={stream_id:options.stream,sequence:options.sequence,generation:options.generation,
        processing_location:"ps_linux",status:"searching",roi_status:"tracking",roi_corners:corners,
        hand_region:{mode:"neck_local_lightweight",status:"tracking",stream_id:options.stream,
          sequence:options.sequence,generation:options.generation,regions:[{corners:handCorners}]},fingertip_neural:neural};
      await route.fulfill({contentType:options.coordinates_only ? "application/json" : "application/x-fretlens-planes",
        body:options.coordinates_only ? Buffer.from("{}") : Buffer.alloc(320*240*2,80),headers:{
        "X-Vision-Backend":"pl","X-Vision-Source":"pc_camera","X-Vision-Token":route.request().headers()["x-vision-token"],
        "X-Vision-Processing-Ms":"80","X-Vision-Fretboard":JSON.stringify(analysis)}});
    });
    await page.evaluate(async () => {
      window.FretVision.stop();
      window.freezePreviewPolling=true;
      document.querySelector("#autoFretboard").checked=true;
      document.querySelector("#pcPlEnabled").checked=true;
      document.querySelector("#visionView").value="color";
      const capture=document.createElement("canvas"); capture.width=640; capture.height=480;
      const ctx=capture.getContext("2d");
      window.paintPreviewCapture=color=>{ctx.fillStyle=color;ctx.fillRect(0,0,640,480);};
      window.paintPreviewCapture("rgb(210,40,30)");
      window.previewCaptureStream=capture.captureStream(0);
      const video=document.querySelector("#cameraVideo");video.srcObject=window.previewCaptureStream;
      await video.play();window.FretVision.attachCamera(window.previewCaptureStream);
    });
    await page.waitForFunction(()=>window.FretVision.metrics().pl_status!=="not-used");
    for(let n=0;!releaseReply&&n<100;n++) await new Promise(resolve=>setTimeout(resolve,10));
    assert.ok(releaseReply,"Fixture received camera frame");
    await page.evaluate(()=>{
      window.paintPreviewCapture("rgb(30,40,210)");window.previewCaptureStream.getVideoTracks()[0].requestFrame();
    });
    releaseReply();
    await rendered();
    await page.waitForFunction(()=>window.FretVision.trackingSnapshot().display_region);
    const value=await page.evaluate(()=>{
      const canvas=document.querySelector("#fretOverlay"),ctx=canvas.getContext("2d");
      const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
      let red=0,green=0;
      for(let i=0;i<pixels.length;i+=4){
        if(pixels[i]>220&&pixels[i+1]<70&&pixels[i+2]<70)red++;
        if(pixels[i]<140&&pixels[i+1]>180&&pixels[i+2]>170)green++;
      }
      const width=Math.min(canvas.width,canvas.height*4/3),height=width*3/4;
      return {pixel:[...ctx.getImageData((canvas.width-width)/2+width*.1,(canvas.height-height)/2+height*.8,1,1).data],red,green,
        region:window.FretVision.trackingSnapshot().display_region,preview:window.fingertipFixture.read()};
    });
    assert.ok(value.pixel[2]>180&&value.pixel[0]<70,JSON.stringify(value.pixel));
    assert.ok(value.red>100,"Existing red region remains visible");
    assert.ok(value.green>100,"Existing green neck region remains visible");
    assert.equal(value.region.length,4); assert.equal(value.preview.arcs.length,4);
    assert.equal(value.preview.key,JSON.stringify([responseOptions.stream,responseOptions.generation,responseOptions.sequence]));
    assert.equal(value.preview.hand,null); assert.equal(value.preview.result,null);
    await page.evaluate(()=>{window.FretVision.stop();window.previewCaptureStream.getTracks().forEach(track=>track.stop());});
    assert.equal((await read()).hidden,true);
  });
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:reports.length,checks:reports,page_errors:errors,
    server_started:false,board_access:false},null,2));
} finally {await browser.close();}
