import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { join } from "node:path";
import { mkdir } from "node:fs/promises";

const { chromium } = createRequire(import.meta.url)(join(process.env.USERPROFILE,
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"));
const base = "http://127.0.0.1:8001/";
const browser = await chromium.launch({headless:true, channel:"msedge"});
try {
  const page = await browser.newPage({viewport:{width:1280,height:900}});
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  const jpeg = await page.evaluate(() => {
    const c=document.createElement("canvas"); c.width=640; c.height=480;
    c.getContext("2d").fillRect(0,0,640,480);
    return c.toDataURL("image/jpeg").split(",")[1];
  });
  let sequence=0, lost=false, stale=false;
  await page.route("**/vision/stream.mjpeg*", route=>route.abort());
  await page.route("**/vision/frame.jpg*", route=>route.fulfill({contentType:"image/jpeg",
    body:Buffer.from(jpeg,"base64"),headers:{"X-Vision-Sequence":String(++sequence), "X-Vision-Captured-Ms":String(performance.now()),
      "X-Vision-Stream":"string-fixture","X-Vision-Source":"camera"}}));
  await page.route("**/api/vision/pc/status", route=>route.fulfill({json:{link_speed_mbps:1000}}));
  await page.route("**/api/vision/pc/process", async route=>{
    const headers=route.request().headers(), o=JSON.parse(headers["x-vision-options"]);
    const line=y=>({start:{x:.1,y:.3+y*.35},end:{x:.9,y:.3+y*.35},observed:false});
    const corners=[{x:.1,y:.3},{x:.9,y:.3},{x:.9,y:.65},{x:.1,y:.65}];
    const analysis={stream_id:o.stream,generation:o.generation,sequence:o.sequence,
      processing_location:"ps_linux",status:lost||stale?"searching":"tracking",
      reason:stale?"flow_projected":lost?"no_supported_lattice":null,
      roi_status:lost?"lost":"tracking",roi_corners:lost?null:corners,strings:[],frets:[],
      estimated_strings:lost?null:{status:"estimated",stream_id:o.stream,generation:o.generation,
        sequence:o.sequence,strings:Array.from({length:6},(_,i)=>line((i+.5)/6)),
        boundaries:Array.from({length:7},(_,i)=>line(i/6))}};
    if (stale) Object.assign(analysis, {
      status:"tracking",reason:null,strings:[],frets:[],lattice_age_ms:1200,lattice_ttl_ms:850
    });
    await route.fulfill({contentType:"application/json",body:"{}",headers:{
      "X-Vision-Backend":"pl","X-Vision-Source":"pc_camera","X-Vision-Token":headers["x-vision-token"],
      "X-Vision-Processing-Ms":"30","X-Vision-Fretboard":JSON.stringify(analysis)}});
  });
  await page.goto(new URL("?camera=board&v=estimated-strings-test",base).href);
  if (!await page.evaluate(()=>window.FretVision?.active)) {
    await page.locator("#debugDetails").evaluate(node=>{node.open=true;});
    await page.locator("#boardVisionButton").click();
  }
  const yellowCount=()=>page.locator("#fretOverlay").evaluate(canvas=>{
    const d=canvas.getContext("2d").getImageData(0,0,canvas.width,canvas.height).data;
    let n=0; for(let i=0;i<d.length;i+=4) if(d[i]>220&&d[i+1]>170&&d[i+2]<120)n++;
    return n;
  });
  await page.waitForFunction(()=>window.FretVision?.trackingSnapshot().automatic?.estimated_strings?.strings.length===6);
  await page.waitForFunction(()=>document.querySelector("#neckTrackingStatus").textContent.includes("六弦位置估计"));
  const out=join(process.cwd(),"diagnostics/string_geometry_20261029");
  await mkdir(out,{recursive:true});
  const counts=[];
  for(const width of [1280,390]) {
    await page.setViewportSize({width,height:900});
    await page.waitForTimeout(350);
    const pixels=await yellowCount();
    assert.ok(pixels>200,`estimated strings are visible at ${width}px: ${pixels}`);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
    await page.screenshot({path:join(out,`estimated-${width}.png`),fullPage:true});
    counts.push({width,yellow_pixels:pixels});
  }
  stale=true;
  await page.waitForFunction(()=>{
    const automatic=window.FretVision.trackingSnapshot().automatic;
    return automatic?.reason==="stale_lattice_geometry" &&
      automatic.roi_status==="tracking" && automatic.strings.length===0 &&
      automatic.frets.length===0 && automatic.estimated_strings===null;
  });
  assert.equal(await yellowCount(),0,"stale lattice is not drawn");
  assert.equal(await page.evaluate(()=>window.FretVision.trackingSnapshot().display_corners?.length),
    4,"current ROI remains visible after stale lattice is cleared");
  stale=false;
  await page.waitForFunction(()=>window.FretVision.trackingSnapshot().automatic?.estimated_strings?.strings.length===6);
  lost=true;
  await page.waitForFunction(()=>window.FretVision.trackingSnapshot().automatic?.roi_status==="lost");
  await page.waitForTimeout(200);
  assert.equal(await yellowCount(),0,"loss clears estimated strings");
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({counts,loss_clears:true,errors}));
} finally { await browser.close(); }
