// Optional browser QA: npm-installed Playwright + Chromium. Serve outputs/laboratory first.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const origin=process.env.OBSERVER_LAB_URL||'http://127.0.0.1:8010';
const oraclePath=process.env.OBSERVER_REPLAY_ORACLES||'work/replay-oracles.json';
function close(a,b,path='root'){if(typeof b==='number'){assert.ok(Math.abs(a-b)<=1e-9*Math.max(1,Math.abs(b)),path+': '+a+' != '+b);return}if(b===null||typeof b!=='object'){assert.deepEqual(a,b,path);return}if(Array.isArray(b)){assert.equal(a.length,b.length,path);b.forEach((v,i)=>close(a[i],v,path+'['+i+']'));return}for(const k of Object.keys(b))close(a[k],b[k],path+'.'+k)}
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',args:['--no-sandbox','--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader']});
 try{
 const page=await browser.newPage({viewport:{width:1480,height:1100},acceptDownloads:true}),errors=[],external=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',q=>{if(!q.url().startsWith(origin)&&!q.url().startsWith('data:')&&!q.url().startsWith('blob:'))external.push(q.url())});
 const oracles=JSON.parse(fs.readFileSync(oraclePath,'utf8'));let compared=0;
 for(const [name,items]of Object.entries(oracles)){
  await page.goto(origin+'/'+name+'/inspector.html');
  await page.waitForFunction(()=>window.observerLab,null,{timeout:60000});
  for(const item of items){
   await page.evaluate(t=>observerLab.setTime(t),item.time);
   const actual=await page.evaluate(()=>{const {depth,...snapshot}=observerLab.snapshot;return{snapshot,depthFrame:depth?.frame_id||null}});
   close(actual.snapshot,item.expected,name+' at '+item.time);assert.equal(actual.depthFrame,item.depth_frame_id);compared++;
  }
  if(name==='real-no-depth'){assert.match(await page.locator('#depth-caption').innerText(),/disabled|unavailable/i);assert.equal(await page.evaluate(()=>observerLab.geometry.count),0)}
  if(name==='mock-saccade-timeout'){assert.match(await page.locator('#decision-reason').innerText(),/Actual termination/);assert.match(await page.locator('#decision-metrics').innerText(),/0.010/)}
  if(name==='mock-empty')assert.equal(await page.evaluate(()=>observerLab.geometry.count),0);
  if(name==='mock-independent-scales'){
   await page.selectOption('#geometry-scope','separate');
   assert.ok(await page.evaluate(()=>observerLab.geometry.frames.length)>=2);
   assert.equal(await page.evaluate(()=>observerLab.geometry.count),await page.evaluate(()=>observerLab.displayMemory.objects.reduce((n,o)=>n+o.reconstruction.observation_history.filter(h=>h.front_corners).length,0)));
  }
 }
 await page.goto(origin+'/mock-many/inspector.html');await page.waitForFunction(()=>window.observerLab,null,{timeout:60000});
 assert.equal(await page.evaluate(()=>observerLab.snapshot.memory.objects.length),0);assert.equal(await page.evaluate(()=>observerLab.geometry.count),0);
 await page.click('#next');assert.equal(await page.evaluate(()=>observerLab.geometry.count),3);
 await page.selectOption('#object-select','object-2');assert.equal(await page.locator('#object-title').innerText(),'object-2');
 // Click a visible recorded support plane using the renderer's actual projection.
 const target=await page.evaluate(()=>{
  three.scene.updateMatrixWorld(true);const bounds=three.renderer.domElement.getBoundingClientRect();
  for(const mesh of three.picks){if(mesh.userData.id===observerLab.state.selected)continue;mesh.geometry.computeBoundingBox();const p=mesh.geometry.boundingBox.getCenter(new THREE.Vector3()).applyMatrix4(mesh.matrixWorld).project(three.camera);const pointer=new THREE.Vector2(p.x,p.y),ray=new THREE.Raycaster();ray.setFromCamera(pointer,three.camera);const hit=ray.intersectObjects(three.picks)[0];if(hit?.object.userData.id===mesh.userData.id)return{id:mesh.userData.id,x:bounds.left+(p.x+1)*bounds.width/2,y:bounds.top+(1-p.y)*bounds.height/2}}
  return null;
 });
 assert.ok(target,'A visible selectable plane must exist');await page.mouse.click(target.x,target.y);assert.equal(await page.locator('#object-title').innerText(),target.id);
 // Rendered triangle coordinates are exactly the source corner convention, to GPU precision.
 const rendered=await page.evaluate(()=>{const mesh=three.picks[0],o=observerLab.displayMemory.objects.find(o=>o.id===mesh.userData.id),h=o.reconstruction.observation_history.find(h=>h.frame_id===mesh.userData.frame_id);return{actual:Array.from(mesh.geometry.attributes.position.array),expected:[0,1,2,0,2,3].flatMap(i=>[h.front_corners[i][0],-h.front_corners[i][1],-h.front_corners[i][2]])}});
 rendered.expected.forEach((v,i)=>assert.ok(Math.abs(v-rendered.actual[i])<1e-6));
 await page.selectOption('#depth-channel','quality');
 const depth=await page.evaluate(()=>observerLab.snapshot.depth.arrays.quality.values[20][20]);
 await page.locator('#depth').hover({position:{x:25,y:25}});assert.match(await page.locator('#depth-pixel').innerText(),/quality/);
 const displayed=await page.evaluate(()=>{const c=document.querySelector('#depth');return c.getContext('2d').getImageData(20,20,1,1).data[0]});
 assert.equal(displayed,Math.round(depth*255));
 await page.selectOption('#depth-channel','valid');
 const invalid=await page.evaluate(()=>document.querySelector('#depth').getContext('2d').getImageData(0,0,1,1).data[0]);assert.equal(invalid,0);
 await page.selectOption('#priority-layer','estimated_reconstruction_gain');assert.match(await page.locator('#priority-caption').innerText(),/not added/);
 await page.selectOption('#priority-layer','grouping');await page.selectOption('#normalization','frame');assert.match(await page.locator('#color-range').innerText(),/frame normalized/);
 await page.selectOption('#memory-view','before');assert.equal(await page.evaluate(()=>observerLab.displayMemory.objects.length),0);assert.equal(await page.evaluate(()=>observerLab.geometry.count),0);
 await page.selectOption('#memory-view','after');assert.equal(await page.evaluate(()=>observerLab.displayMemory.objects.length),3);
 await page.selectOption('#memory-view','current');await page.check('#show-reliability');
 await page.check('#step-mode');await page.locator('#timeline').fill('0');assert.equal(await page.evaluate(()=>observerLab.snapshot.committed_index),-1);await page.locator('#timeline').fill('1');assert.equal(await page.evaluate(()=>observerLab.snapshot.committed_index),0);
 await page.uncheck('#step-mode');await page.selectOption('#speed','2');await page.click('#play');await page.waitForFunction(()=>observerLab.state.t>600);await page.evaluate(()=>{if(observerLab.state.playing)document.querySelector('#play').click()});assert.equal(await page.evaluate(()=>observerLab.state.playing),false);
 const downloadPromise=page.waitForEvent('download');await page.click('#download-depth');assert.equal((await downloadPromise).suggestedFilename(),'depth-observation.json');
 await page.goto(origin+'/real-depth/inspector.html');await page.waitForFunction(()=>window.observerLab,null,{timeout:60000});
 await page.evaluate(()=>observerLab.setTime(observerLab.run.fixations[2].offset_ms));await page.selectOption('#geometry-scope','latest');await page.click('#reset-camera');
 await page.waitForTimeout(200);await page.screenshot({path:'outputs/laboratory/desktop.png',fullPage:true});
 const memoryBefore=await page.evaluate(()=>JSON.stringify(observerLab.displayMemory));
 const box=await page.locator('#memory3d').boundingBox();await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2+75,box.y+box.height/2+20,{steps:8});await page.mouse.up();await page.mouse.wheel(0,-100);await page.waitForTimeout(100);
 assert.equal(await page.evaluate(()=>JSON.stringify(observerLab.displayMemory)),memoryBefore);
 await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 await page.screenshot({path:'outputs/laboratory/mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
 console.log(JSON.stringify({pass:true,causal_snapshots_compared:compared,checks:['Python/JavaScript clocks including saccades and terminal states','no future depth/objects/geometry','depth disabled, empty, one/many, inconsistent scales, early timeout','actual 3D selection and recorded corner coordinates','raw depth/quality/validity values','priority diagnostics and grouping','before/after comparison','step and time controls','playback speed and depth download','orbit/zoom leaves memory unchanged','mobile layout','no external/CDN requests or JavaScript errors']},null,2));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
