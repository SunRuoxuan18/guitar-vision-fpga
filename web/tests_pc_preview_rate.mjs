import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {dirname, join, extname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';

const root = dirname(fileURLToPath(import.meta.url));
const {chromium} = createRequire(import.meta.url)(join(process.env.USERPROFILE,
  '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const baseline = process.argv.includes('--baseline');
const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://localhost').pathname;
    const target=baseline && path==='/vision.js' ?
      join(root,'../output/releases/fretlens_v22_20261023/source/web/vision.js') :
      join(root, path === '/' ? 'index.html' : path);
    const data = await readFile(target);
    res.writeHead(200, {'Content-Type': ({'.js':'text/javascript','.css':'text/css','.html':'text/html'})[extname(path === '/' ? 'index.html' : path)] || 'application/octet-stream'});
    res.end(data);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({channel:'msedge', headless:true});
const reports = [];
try {
  await mkdir(join(root, '../diagnostics/pc_preview_rate'), {recursive:true});
  for (const width of [1280, 390]) {
    const page = await browser.newPage({viewport:{width,height:900}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
      window.previewVideoCopies=0;
      const draw=CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage=function(source,...args) {
        if(this.canvas.id==='fretOverlay' && source.id==='cameraVideo') window.previewVideoCopies++;
        return draw.call(this,source,...args);
      };
      navigator.mediaDevices.getUserMedia = async () => {
        const canvas = document.createElement('canvas'); canvas.width=640; canvas.height=480;
        const ctx=canvas.getContext('2d'); let n=0;
        const paint=()=>{ ctx.fillStyle=`rgb(${50+(n++%180)},30,150)`; ctx.fillRect(0,0,640,480); };
        paint(); window.previewFixtureTimer=setInterval(paint, 1000/30);
        return canvas.captureStream(30);
      };
    });
    let active=0, maxActive=0, responses=0, reject=false, analysisDelay=180;
    let slowRequestStarted=null;
    await page.route('**/api/vision/pc/status', route=>route.fulfill({json:{link_speed_mbps:1000}}));
    await page.route('**/api/vision/pc/process', async route => {
      active++; maxActive=Math.max(maxActive,active);
      try {
        const options=JSON.parse(route.request().headers()['x-vision-options']);
        if (analysisDelay >= 1200) slowRequestStarted?.();
        await new Promise(resolve=>setTimeout(resolve,analysisDelay));
        if (reject) { await route.fulfill({status:503,body:'fixture unavailable'}); return; }
        const corners=[{x:.4,y:.35},{x:.8,y:.35},{x:.8,y:.65},{x:.4,y:.65}];
        const analysis={stream_id:options.stream,sequence:options.sequence,generation:options.generation,
          processing_location:'ps_linux',status:'searching',roi_status:'tracking',roi_corners:corners,
          hand_region:{mode:'neck_local_lightweight',status:'tracking',stream_id:options.stream,
            sequence:options.sequence,generation:options.generation,regions:[{corners}]}};
        const planes=Buffer.alloc(320*240*2); planes.fill(100,0,320*240);
        await route.fulfill({status:200,
          contentType:options.coordinates_only?'application/json':'application/x-fretlens-planes',
          body:options.coordinates_only?Buffer.from('{}'):planes,
          headers:{'X-Vision-Backend':'pl','X-Vision-Source':'pc_camera',
            'X-Vision-Token':route.request().headers()['x-vision-token'],
            'X-Vision-Processing-Ms':'170','X-Vision-Fretboard':JSON.stringify(analysis)}});
        responses++;
      } finally {active--;}
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/?camera=pc`);
    await page.waitForFunction(()=>window.FretVision?.trackingSnapshot().display_region);
    const measure=()=>page.evaluate(async()=>{
      const signatures=new Set(); const before=performance.now(), copies=window.previewVideoCopies;
      const sample=document.createElement('canvas'); sample.width=640;sample.height=480;
      const output=sample.getContext('2d',{willReadFrequently:true});
      for(let n=0;n<60;n++) {
        const c=document.querySelector('#fretOverlay'),ctx=c.getContext('2d');
        // Top of image: outside the delayed PL region and status overlays.
        const w=Math.min(c.width,c.height*4/3),h=w*3/4;
        // Sample the composited picture, including the native video under the transparent overlay.
        output.clearRect(0,0,640,480);
        output.drawImage(document.querySelector('#cameraVideo'),0,0,640,480);
        output.drawImage(c,(c.width-w)/2,(c.height-h)/2,w,h,0,0,640,480);
        signatures.add([...output.getImageData(64,48,1,1).data].join(','));
        await new Promise(resolve=>setTimeout(resolve,33));
      }
      return {unique_frames:signatures.size,pixel_fps:signatures.size*1000/(performance.now()-before),
        video_canvas_copies:window.previewVideoCopies-copies,...window.FretVision.metrics()};
    });
    const initialResponses=responses;
    const result=await measure();
    const red=await page.locator('#fretOverlay').evaluate(c=>{
      const data=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      let count=0;for(let i=0;i<data.length;i+=4)if(data[i]>220&&data[i+1]<70&&data[i+2]<70)count++;
      return count;
    });
    assert.ok(red>100,'Board region remains visible');
    assert.equal(maxActive,1,'Never queue camera frames behind the board');
    if(!baseline) assert.ok(result.pixel_fps>20,JSON.stringify(result));
    if(!baseline) assert.equal(result.video_canvas_copies,0,'Color video must bypass full-frame canvas copies');
    await page.screenshot({path:join(root,`../diagnostics/pc_preview_rate/${baseline?'before':'after'}-${width}.png`)});
    analysisDelay=1200;
    const slowStarted=new Promise(resolve=>{slowRequestStarted=resolve;});
    await Promise.race([slowStarted,new Promise((_,fail)=>setTimeout(()=>fail(new Error('slow request not started')),2000))]);
    const slow=await measure();
    if(!baseline) assert.ok(slow.pixel_fps>20,JSON.stringify({analysisDelay,slow}));
    analysisDelay=180; slowRequestStarted=null;
    reject=true;
    await page.waitForTimeout(1300);
    assert.equal(await page.evaluate(()=>window.FretVision.trackingSnapshot().display_region),null);
    const offline=await measure();
    if(!baseline) assert.ok(offline.pixel_fps>20,'Video continues when PL service fails');
    reject=false;
    await page.waitForFunction(()=>window.FretVision.trackingSnapshot().display_region);
    await page.locator('#visionView').selectOption('gray',{force:true});
    await page.waitForTimeout(1400);
    const gray=await measure();
    if(!baseline) assert.ok(gray.display_fps<10,'PL view must report processed FPS, not camera FPS');
    assert.deepEqual(errors,[]);
    reports.push({width,fixture:'30Hz canvas capture, 180ms normal and 1.2s slow simulated results; no recognition benchmark',
      color:result,slow,offline,gray,board_responses:responses-initialResponses,max_parallel:maxActive});
    await page.evaluate(()=>{window.FretVision.stop();clearInterval(window.previewFixtureTimer);
      document.querySelector('#cameraVideo').srcObject.getTracks().forEach(t=>t.stop());});
    await page.close();
  }
  await writeFile(join(root,`../diagnostics/pc_preview_rate/${baseline?'before':'after'}.json`),JSON.stringify(reports,null,2));
  console.log(JSON.stringify(reports,null,2));
} finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
