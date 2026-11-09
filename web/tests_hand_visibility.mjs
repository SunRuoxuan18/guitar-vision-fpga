import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir} from 'node:fs/promises';
import {dirname, join, extname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';

// Isolated UI fixture: recorded board points, no camera or hardware requests.
const root = dirname(fileURLToPath(import.meta.url));
const {chromium} = createRequire(import.meta.url)(join(process.env.USERPROFILE,
  '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const replay = JSON.parse(await readFile(join(root,'../diagnostics/hand_visibility_replay.json'),'utf8'));
const hand = replay.samples.find(s => s.analysis.hand_detection.hands.length).analysis.hand_detection.hands[0];
const server = createServer(async (req,res) => {
  try {
    const path = new URL(req.url,'http://localhost').pathname;
    if (path.includes('..')) throw Error('path');
    const file = join(root,path === '/' ? 'index.html' : path);
    let data = await readFile(file);
    if (path === '/vision.js') data = Buffer.from(data.toString().replace(/\}\)\(\);\s*$/, `
      window.showHandFixture = (hands) => {
        V.source='camera'; V.detect=true; stage.dataset.mode='camera';
        Object.defineProperty(video,'readyState',{value:4,configurable:true});
        PC.sequence=1; PC.matchedAt=performance.now();
        pcMatched.width=640; pcMatched.height=480;
        pcMatched.getContext('2d').fillStyle='#607060';
        pcMatched.getContext('2d').fillRect(0,0,640,480);
        V.handObservation={hands,source:'ps_linux',token:1,frame:pcMatched,
          detection:{inference_location:'ps_pl'}};
        render(); return handRegionCorners();
      };
      window.expireHandFixture = () => {
        const before=frameBox(); PC.matchedAt=performance.now()-2000;
        return {before,after:frameBox(),matched:inputFrame()===pcMatched,
          hands:visibleHands().length};
      };
      window.showRegionFixture = (sequence=1) => {
        window.showHandFixture([]);
        PC.generation=2;
        V.automatic={stream_id:'region-fixture',roi_status:'tracking',
          roi_corners:[{x:.1,y:.35},{x:.9,y:.4},{x:.9,y:.6},{x:.1,y:.55}],
          hand_region:{mode:'neck_local_lightweight',status:'tracking',
            stream_id:'region-fixture',generation:2,sequence,
            regions:[{corners:[{x:.4,y:.37},{x:.6,y:.38},{x:.6,y:.58},{x:.4,y:.57}]}]}};
        render(); return handRegionCorners();
      };
    })();`));
    res.writeHead(200,{'Content-Type':({'.js':'text/javascript','.css':'text/css','.html':'text/html'})[extname(file)] || 'application/octet-stream'});
    res.end(data);
  } catch { res.writeHead(404);res.end(); }
});
await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
const browser = await chromium.launch({channel:'msedge',headless:true});
try {
  await mkdir(join(root,'../diagnostics/hand_visibility'),{recursive:true});
  for (const width of [1280,390]) {
    const page = await browser.newPage({viewport:{width,height:900}});
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.waitForFunction(()=>window.showHandFixture);
    const corners = await page.evaluate(hand=>window.showHandFixture([hand]),hand);
    assert.equal(corners.length,4);
    const reds = await page.evaluate(()=>{
      const c=document.querySelector('#fretOverlay');
      const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      let n=0;for(let i=0;i<d.length;i+=4) if(d[i]>200 && d[i+1]<100 && d[i+2]<100)n++;
      return n;
    });
    assert.ok(reds>100,`Missing red box at width ${width}: ${reds}`);
    await page.screenshot({path:join(root,`../diagnostics/hand_visibility/ui-${width}.png`)});
    const expired=await page.evaluate(()=>window.expireHandFixture());
    assert.deepEqual(expired.after,expired.before,'Expiry must not change aspect ratio');
    assert.equal(expired.matched,true,'Expiry must not jump back to live camera');
    assert.equal(expired.hands,0,'Expired geometry must remain hidden');
    assert.equal(await page.evaluate(()=>window.showHandFixture([])),null);
    assert.equal((await page.evaluate(()=>window.showRegionFixture())).length,4);
    const regionPixels=await page.evaluate(()=>{
      const c=document.querySelector('#fretOverlay');
      const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      let red=0,teal=0;
      for(let i=0;i<d.length;i+=4){
        if(d[i]>200&&d[i+1]<100&&d[i+2]<100)red++;
        if(d[i]<150&&d[i+1]>120&&d[i+2]>100)teal++;
      }
      return {red,teal};
    });
    assert.ok(regionPixels.red>100,'Lightweight region must draw without neural landmarks');
    assert.ok(regionPixels.teal>100,'Neck outline must draw before string confirmation');
    await page.screenshot({path:join(root,`../diagnostics/hand_visibility/region-${width}.png`)});
    assert.equal(await page.evaluate(()=>window.showRegionFixture(0)),null,'Stale region must hide');
    await page.close();
  }
  console.log('PASS: recorded board hand renders red box on desktop/mobile without neck calibration; missing hands clear box');
} finally {await browser.close(); await new Promise(resolve=>server.close(resolve));}
