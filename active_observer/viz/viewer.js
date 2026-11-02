'use strict';
const $=id=>document.getElementById(id);
const state={t:0,playing:false,selected:null,priorityPoint:null,depthPoint:null,images:new Map(),geometryKey:null};
let bundle,r,three=null;
const clone=value=>structuredClone(value);
const fmt=(value,n=3)=>value===null||value===undefined?'unavailable':Number(value).toFixed(n);
const pretty=value=>JSON.stringify(value,null,2);
function options(id,entries,keep){const previous=keep===undefined?$(id).value:keep;$(id).replaceChildren(...entries.map(entry=>{const o=document.createElement('option');o.value=String(entry[0]);o.textContent=entry[1];return o}));if(entries.some(e=>String(e[0])===String(previous)))$(id).value=previous}
function details(id,pairs){const nodes=[];for(const pair of pairs){const a=document.createElement('dt'),b=document.createElement('dd');a.textContent=pair[0];b.textContent=String(pair[1]);nodes.push(a,b)}$(id).replaceChildren(...nodes)}
function picture(src){if(!state.images.has(src)){const image=new Image();image.onload=()=>render();image.src=src;state.images.set(src,image)}return state.images.get(src)}
function ctx(id,width=r.width,height=r.height){const c=$(id);if(c.width!==width||c.height!==height){c.width=width;c.height=height}c.style.aspectRatio=String(width)+'/'+String(height);const context=c.getContext('2d');context.clearRect(0,0,width,height);return context}
function snapTime(value){let t=Math.max(0,Math.min(r.duration_ms,value));if(Math.abs(t-r.duration_ms)<1e-6)t=r.duration_ms;for(const f of r.fixations)if(Math.abs(t-f.offset_ms)<1e-6)t=f.offset_ms;return t}
function snapshot(time){
 const t=snapTime(time);let i=0,committed=-1;
 for(let j=0;j<r.fixations.length;j++){const f=r.fixations[j];if(f.onset_ms<=t)i=j;if(f.offset_ms<=t)committed=j}
 const f=r.fixations[i],done=f.offset_ms<=t,memory=clone(done?f.memory_after:f.memory_before);
 let gaze=f.gaze.slice(),phase='fixation',sensoryAvailable=true;
 if(done){phase='between_fixations';const dt=Math.min(Math.max(0,t-f.offset_ms),f.saccade_duration_ms||0);memory.time_ms+=dt;
  for(const object of memory.objects){object.confidence*=Math.exp(-dt/r.config.memory_decay_ms);const u=object.perceptual_uncertainty;u.identity=Math.min(1,u.identity+.02*dt/r.config.memory_decay_ms)}
  if((f.saccade_duration_ms||0)>0&&t<f.offset_ms+f.saccade_duration_ms){phase='saccade';sensoryAvailable=false;const destination=f.saccade_to||(f.selected_next?[f.selected_next.x,f.selected_next.y]:gaze),fraction=dt/Math.max(1e-12,f.saccade_planned_ms||f.saccade_duration_ms);gaze=gaze.map((a,j)=>a+(destination[j]-a)*fraction)}
 }
 const terminal=t>=r.duration_ms;
 if(terminal&&f.termination_after_saccade){sensoryAvailable=false;phase='saccade';const destination=f.saccade_to||gaze,fraction=(f.saccade_duration_ms||0)/Math.max(1e-12,f.saccade_planned_ms||1);gaze=f.gaze.map((a,j)=>a+(destination[j]-a)*fraction)}
 const last=committed>=0?r.fixations[committed]:null;
 return {elapsed_ms:t,fixation_index:i,committed_index:committed,phase:terminal?'terminal':phase,sensory_available:sensoryAvailable,gaze,memory,stopping:last?last.stopping:null,next_choice:last?last.selected_next:null,depth:last?last.depth:null,termination:terminal?r.termination:null};
}
function selectedMemory(view){
 const mode=$('memory-view').value;
 if(view.committed_index<0||mode==='current')return {memory:view.memory,caption:'Current causal memory at '+fmt(view.memory.time_ms/1000,3)+' s'};
 const f=r.fixations[view.committed_index];return {memory:mode==='before'?f.memory_before:f.memory_after,caption:(mode==='before'?'Before':'After')+' fixation '+f.step+' commit; memory time '+fmt((mode==='before'?f.onset_ms:f.offset_ms)/1000,3)+' s. Other panels follow the current playback clock.'};
}
function pause(){state.playing=false;cancelAnimationFrame(state.raf);$('play').textContent='Play'}
function setTime(t){state.t=snapTime(t);state.priorityPoint=null;state.depthPoint=null;render()}
function marker(context,point,label,color,filled=true){context.beginPath();context.arc(point[0],point[1],4,0,Math.PI*2);context.strokeStyle=color;context.fillStyle=color;context.lineWidth=1.4;filled?context.fill():context.stroke();context.font='11px system-ui';context.fillText(label,point[0]+6,point[1]-5)}
function drawScene(view,memory){
 const context=ctx('scene'),image=picture(bundle.original);if(image.complete&&image.naturalWidth)context.drawImage(image,0,0,r.width,r.height);
 context.beginPath();context.strokeStyle='#a2d4c5';context.lineWidth=1;
 for(let j=0;j<=view.fixation_index;j++){const point=r.fixations[j].gaze;j?context.lineTo(...point):context.moveTo(...point)}if(!view.sensory_available)context.lineTo(...view.gaze);context.stroke();
 for(let j=0;j<=view.fixation_index;j++)marker(context,r.fixations[j].gaze,String(j),j===view.fixation_index?'#f0cb90':'#9bcbbb',j!==0);
 marker(context,view.gaze,'gaze','#ffffff',false);
 const f=r.fixations[view.fixation_index];context.beginPath();context.arc(...f.gaze,f.radius_px,0,Math.PI*2);context.strokeStyle='#fff9';context.stroke();
 if(view.next_choice)marker(context,[view.next_choice.x,view.next_choice.y],'chosen next','#e6c494',false);
 if($('show-boxes').checked){for(const object of memory.objects){const b=object.bbox;context.setLineDash(object.last_seen+1e-6<r.fixations[Math.max(0,view.committed_index)].offset_ms?[4,3]:[]);context.strokeStyle=object.id===state.selected?'#ffe2aa':'#94cfc0';context.lineWidth=object.id===state.selected?2:1;context.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);context.fillStyle=context.strokeStyle;context.font='10px system-ui';context.fillText(object.id,b[0]+2,Math.max(10,b[1]-3))}context.setLineDash([])}
 $('scene-caption').textContent='0 = externally supplied initial gaze. '+(view.sensory_available?'Current fixation '+f.step+'.':'Gaze is moving during a saccade; no new sensory evidence.')+' Click a known object box to inspect it.';
}
function drawSensor(view){
 const f=r.fixations[view.fixation_index],context=ctx('sensor'),image=picture(bundle.frames[view.fixation_index]);
 if(image.complete&&image.naturalWidth)context.drawImage(image,0,0);
 if($('show-reliability').checked){const rel=picture(bundle.reliability[view.fixation_index]);if(rel.complete&&rel.naturalWidth){context.globalAlpha=.55;context.drawImage(rel,0,0);context.globalAlpha=1}}
 context.beginPath();context.arc(...f.gaze,f.radius_px,0,Math.PI*2);context.strokeStyle='#fff';context.lineWidth=1.3;context.stroke();
 $('sensory-caption').textContent=(view.sensory_available?'Actual input for fixation '+f.step+'. ':'No new input during saccade: retained fixation '+f.step+' frame. ')+($('show-reliability').checked?'Reliability overlay enabled (heuristic, not calibrated).':'Central full RGB; peripheral bandwidth '+r.config.peripheral_width+' pixels wide.');
 $('input-hashes').textContent=pretty({sensory_sha256:f.sensory_sha256,object_input_sha256:f.object_input_sha256||f.sensory_sha256,depth_input_sha256:view.committed_index>=view.fixation_index?f.depth_input_sha256:'not committed yet'});
}
function channel(record,key){return record&&record.available&&record.arrays?record.arrays[key].values:null}
function ramp(v){v=Math.max(0,Math.min(1,v));return [Math.round(56+190*v),Math.round(74+126*v),Math.round(128-65*v)]}
function drawRaster(context,array,valid,low,high,grayscale=false){
 const h=array.length,w=array[0].length,small=document.createElement('canvas');small.width=w;small.height=h;
 const sc=small.getContext('2d'),data=sc.createImageData(w,h);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const value=Number(array[y][x]),v=(value-low)/(high-low||1),rgb=valid&&!valid[y][x]?[27,33,45]:grayscale?[Math.round(v*255),Math.round(v*255),Math.round(v*255)]:ramp(v);const index=(y*w+x)*4;data.data[index]=rgb[0];data.data[index+1]=rgb[1];data.data[index+2]=rgb[2];data.data[index+3]=255}
 sc.putImageData(data,0,0);context.imageSmoothingEnabled=false;context.drawImage(small,0,0,context.canvas.width,context.canvas.height);
}
function drawDepth(view){
 const context=ctx('depth'),record=view.depth,key=$('depth-channel').value,array=channel(record,key);
 $('download-depth').disabled=!array;
 if(!array){context.fillStyle='#a5b7ca';context.font='14px system-ui';context.fillText(view.committed_index<0?'No depth committed yet':'Depth unavailable',20,35);$('depth-caption').textContent=view.committed_index<0?'No future depth result is exposed before fixation offset.':record&&record.reason?record.reason:'This run has no recorded depth arrays.';$('depth-provenance').textContent='Unavailable';$('depth-pixel').textContent='';return}
 drawRaster(context,array,key==='valid'?null:channel(record,'valid'),0,1,key!=='inverse_depth');
 $('depth-caption').textContent='Fixation '+r.fixations[view.committed_index].step+' · '+record.frame_id+' · fixed display range [0, 1]. Independent normalized inverse depth, not a global metric map. Dark invalid pixels carry no depth evidence.';
 $('depth-provenance').textContent=pretty({input_sha256:record.input_sha256,array_hashes:record.array_hashes,coordinate_contract:record.coordinate_contract,metadata:record.metadata});
 if(state.depthPoint){const x=state.depthPoint[0],y=state.depthPoint[1];$('depth-pixel').textContent='Pixel ('+x+', '+y+'): '+key+' = '+String(array[y][x])+'; valid = '+String(record.arrays.valid.values[y][x])}
 else $('depth-pixel').textContent='Hover to inspect exact stored values. Quality is heuristic, not a probability.';
}
const fixedRanges={total:[-1,2],salience:[0,.55],peripheral_structure:[0,.25],grouping:[0,.2],novelty:[0,.18],perceptual_information_need:[0,.12],reconstruction_information_need:[0,.35],estimated_reconstruction_gain:[0,1.25],inhibition_of_return:[-.65,0],movement_cost:[-.18,0],coverage:[0,1]};
function drawPriority(view){
 const context=ctx('priority'),f=view.committed_index>=0?r.fixations[view.committed_index]:null;
 if(!f){$('priority-caption').textContent='Next priority is unavailable until the current observation commits.';$('choice').textContent='No committed decision';$('priority-pixel').textContent='';$('color-range').textContent='';return}
 const q=f.priority,key=$('priority-layer').value;
 const array=key==='total'?q.total:key==='coverage'?q.coverage:q.components[key]||(q.diagnostics||{})[key];
 if(!array)return;
 let range=fixedRanges[key]||[0,1];if($('normalization').value==='frame'){const values=array.flat();range=[Math.min(...values),Math.max(...values)]}
 drawRaster(context,array,q.valid,range[0],range[1]);context.imageSmoothingEnabled=true;
 if(f.selected_next)marker(context,[f.selected_next.x,f.selected_next.y],'next','#fff',false);
 $('color-range').textContent=fmt(range[0],2)+' … '+fmt(range[1],2)+($('normalization').value==='frame'?' · frame normalized':' · fixed');
 $('priority-caption').textContent='Map committed at '+fmt(f.offset_ms/1000,3)+' s. '+(key==='estimated_reconstruction_gain'?'Derived gain diagnostic; not added to priority again. ':'Signed contribution maps sum to total; map arrays are rounded to four decimals. ')+(view.fixation_index>view.committed_index?'This previous map selected the current gaze; the next decision is pending.':'');
 $('choice').textContent=f.selected_next?pretty({selected_next:[f.selected_next.x,f.selected_next.y],softmax_choice_mass:f.selected_next.selection_probability,temperature:f.selected_next.temperature,actual_signed_components:f.selected_next.components,utility:f.selected_next.utility}):'Observer stopped; no next gaze.';
 const point=state.priorityPoint||(f.selected_next?f.selected_next.field_index:[0,0]),x=point[0],y=point[1],values={sample:[x,y],total:q.total[y][x]};
 for(const [name,a]of Object.entries(q.components))values[name]=a[y][x];values.estimated_reconstruction_gain=q.diagnostics?.estimated_reconstruction_gain?.[y]?.[x]??null;$('priority-pixel').textContent=pretty(values);
}
function objectInspector(memory,caption){
 const objects=memory.objects;if(!objects.some(o=>o.id===state.selected))state.selected=objects[0]?.id||null;
 options('object-select',objects.map(o=>[o.id,o.id+' · '+Object.entries(o.probabilities).sort((a,b)=>b[1]-a[1])[0][0]]),state.selected);
 const object=objects.find(o=>o.id===state.selected);$('object-title').textContent=object?object.id:'No committed object memory';
 $('memory-caption').textContent=caption;
 if(!object){details('object-summary',[['Individual objects',0],['Geometry','Unavailable until centrally acquired evidence commits']]);for(const id of ['object-evidence','object-history','geometry-history'])$(id).textContent='Unavailable';return}
 const g=object.reconstruction;
 details('object-summary',[['Stable ID',object.id],['BBox (xyxy)',object.bbox.map(x=>fmt(x,1)).join(', ')],['Observations',object.observation_count],['Last seen',fmt(object.last_seen/1000,3)+' s'],['Class confidence index',fmt(object.confidence)],['Perceptual identity uncertainty',fmt(object.perceptual_uncertainty.identity)],['2D localization uncertainty',fmt(object.perceptual_uncertainty.localization_2d)],['Latest relative Z proxy',fmt(g.relative_depth)],['Shared scale ID',g.scale_id||'Unavailable'],['Latest observation frame',g.frame_id||'Unavailable'],['Depth uncertainty',fmt(g.depth_uncertainty)],['Width / height / thickness',g.dimensions.map(v=>fmt(v)).join(' / ')],['Orientation','Unknown'],['Geometry frames',g.observation_history.length]]);
 $('object-evidence').textContent=pretty({class_evidence:object.probabilities,perceptual_uncertainty:object.perceptual_uncertainty,dimension_uncertainty:g.dimension_uncertainty});$('object-history').textContent=pretty(object.history);$('geometry-history').textContent=pretty(g.observation_history);
}
function drawEnsemble(memory){
 const e=memory.ensembles.at(-1)?.summary;
 if(!e){options('ensemble-region',[['global','No committed summary']]);details('ensemble-summary',[['Evidence','Unavailable']]);$('ensemble-json').textContent='Unavailable';return}
 const regions=e.local_regions||[];options('ensemble-region',[['global','Global coarse scene summary'],...regions.map((v,i)=>[String(i),'Local structural group '+(i+1)])]);
 const selected=$('ensemble-region').value==='global'?e:regions[+$('ensemble-region').value]||e;
 details('ensemble-summary',[['Region xyxy',selected.region.map(v=>fmt(v,1)).join(', ')],['Mean RGB',selected.mean_rgb.map(v=>fmt(v,2)).join(', ')],['RGB standard deviation',selected.rgb_std?selected.rgb_std.map(v=>fmt(v,2)).join(', '):'Unavailable locally'],['Edge density',fmt(selected.edge_density)],['Mean sensory reliability',fmt(selected.mean_reliability)],['Structural group count',selected.structural_group_count],['Object count','Unavailable'],['Size distribution','Unavailable'],['Member objects','No supported membership inference']]);
 $('ensemble-json').textContent=pretty(selected);
}
function decisions(view){
 const stop=view.termination?{...(view.stopping||{}),...view.termination,metrics_source:'last committed assessment',actual_termination:view.termination}:view.stopping;
 if(!stop){$('decision-label').textContent='Acquiring first observation';$('decision-reason').textContent='The first assessment has not committed.';details('decision-metrics',[]);$('decision').textContent='Unavailable';return}
 const names={administrative_timeout:'Administrative timeout',low_estimated_benefit:'Low estimated benefit',reconstruction_unresolved:'Reconstruction unresolved'};
 $('decision-label').textContent=stop.outcome?names[stop.outcome]||stop.outcome:'Continue observing';
 $('decision-reason').textContent=(view.termination?'Actual termination; metrics are from the last committed assessment. ':'Last committed assessment. ')+stop.reason;
 details('decision-metrics',[['Predicted gain',fmt(stop.estimated_information_gain)],['Expected observation value',fmt(stop.expected_observation_value)],['Threshold',fmt(stop.threshold)],['Observation cost',fmt(stop.observation_cost)],['Coverage',fmt(stop.coverage)],['Low-value streak',String(stop.low_value_streak??'unavailable')+' / '+String(stop.persistence_required??'unavailable')],['Actual 3D improvement','null — no reference 3D evidence'],['Reconstruction validated','false']]);
 $('decision').textContent=pretty(stop);
}
function plot(id,series,limit,color,empty){
 const context=ctx(id,600,150);context.strokeStyle='#42516a';context.beginPath();context.moveTo(35,10);context.lineTo(35,125);context.lineTo(590,125);context.stroke();
 context.fillStyle='#a6b7cc';context.font='11px system-ui';context.fillText(String(limit),3,15);context.fillText('0',15,125);
 if(!series.some(p=>p[1]!==null)){context.fillText(empty,55,65);return}
 let open=false;context.beginPath();context.strokeStyle=color;context.lineWidth=2;
 for(const [time,value]of series){if(value===null){open=false;continue}const x=35+time/r.duration_ms*550,y=125-Math.min(1,value/limit)*110;if(open)context.lineTo(x,y);else context.moveTo(x,y);open=true}context.stroke();
 for(const [time,value]of series)if(value!==null){context.beginPath();context.arc(35+time/r.duration_ms*550,125-Math.min(1,value/limit)*110,2,0,Math.PI*2);context.fillStyle=color;context.fill()}
}
function progress(view){
 const committed=r.fixations.slice(0,view.committed_index+1);
 plot('predicted-plot',committed.map(f=>[f.offset_ms,f.predicted_information_gain]),1.5,'#d5bd84','No committed prediction yet');
 plot('measured-plot',committed.map(f=>{const values=f.measured_proxies.matched_observations;return[f.offset_ms,values.length?values.reduce((s,v)=>s+v.normalized_centroid_displacement,0)/values.length:null]}),.15,'#8bcbbb','No matched-observation measurements yet');
}
function sourceVector(point){return new THREE.Vector3(point[0],-point[1],-point[2])}
function line(group,points,color,opacity=1){const geometry=new THREE.BufferGeometry().setFromPoints(points.map(sourceVector)),material=new THREE.LineBasicMaterial({color,transparent:opacity<1,opacity});const object=new THREE.Line(geometry,material);group.add(object);return object}
function label(group,message,point,size=5){const c=document.createElement('canvas');c.width=512;c.height=72;const x=c.getContext('2d');x.fillStyle='#cad9e9';x.font='24px system-ui';x.fillText(message,4,42);const texture=new THREE.CanvasTexture(c),material=new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false});const sprite=new THREE.Sprite(material);sprite.position.copy(sourceVector(point));sprite.scale.set(size,size*.14,1);group.add(sprite)}
function cameraGuide(group,frame){
 const c=r.camera,z=1.5,corner=(x,y)=>[(x-c.cx)*z/c.fx,(y-c.cy)*z/c.fy,z],a=corner(0,0),b=corner(r.width,0),d=corner(0,r.height),e=corner(r.width,r.height);
 line(group,[a,b,e,d,a],0x536b85,.6);for(const p of [a,b,e,d])line(group,[[0,0,0],p],0x536b85,.6);
 line(group,[[0,0,0],[.5,0,0]],0xa5bcce);line(group,[[0,0,0],[0,.5,0]],0xa5bcce);line(group,[[0,0,0],[0,0,.5]],0xa5bcce);
 label(group,frame,[0,-.45,0]);
}
function fitCamera(context=false){
 if(!three)return;three.scene.updateMatrixWorld(true);const box=new THREE.Box3();if(!context&&three.picks.length){for(const object of three.picks)box.expandByObject(object)}else box.setFromObject(three.world);
 if(box.isEmpty()){three.camera.position.set(4,3,7);three.controls.target.set(0,0,-2)}
 else{const size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());const direction=three.camera.position.clone().sub(three.controls.target).normalize();if(!direction.length())direction.set(1,1,1).normalize();const tangent=Math.tan(three.camera.fov*Math.PI/360);const distance=Math.max(.6,size.y/(2*tangent),size.x/(2*tangent*three.camera.aspect),size.z)*1.8;three.controls.target.copy(center);three.camera.position.copy(center).addScaledVector(direction,distance)}
 three.controls.update();
}
function init3D(){
 const holder=$('memory3d');
 try{
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x101925);holder.append(renderer.domElement);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(48,1,.01,2000);camera.position.set(6,4,9);
  const controls=new THREE.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
  const world=new THREE.Group();scene.add(world);three={renderer,scene,camera,controls,world,picks:[],frames:[],geometryCount:0};
  const resize=()=>{const w=holder.clientWidth,h=holder.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()};new ResizeObserver(resize).observe(holder);resize();
  let down=null;renderer.domElement.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY]});
  renderer.domElement.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;const box=renderer.domElement.getBoundingClientRect(),pointer=new THREE.Vector2((e.clientX-box.left)/box.width*2-1,-(e.clientY-box.top)/box.height*2+1),ray=new THREE.Raycaster();ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(three.picks)[0];if(hit){state.selected=hit.object.userData.id;state.geometryKey=null;render()}});
  function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}animate();
 }catch(error){holder.textContent='3D rendering unavailable in this browser: '+error.message+'. Exact geometry records remain inspectable below.';$('geometry-status').textContent='WebGL unavailable; no replacement geometry is fabricated.'}
}
function render3D(view,memory){
 if(!three)return;
 const key=String(view.committed_index)+'|'+$('memory-view').value+'|'+$('geometry-scope').value+'|'+state.selected;
 if(key===state.geometryKey)return;state.geometryKey=key;
 three.world.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material){if(o.material.map)o.material.map.dispose();o.material.dispose()}});three.scene.remove(three.world);three.world=new THREE.Group();three.scene.add(three.world);three.picks=[];
 let hypotheses=[];
 for(const object of memory.objects)for(const h of object.reconstruction.observation_history||[])if(h.front_corners&&h.frame_id)hypotheses.push({object,h});
 let ids=[...new Set(hypotheses.map(x=>x.h.frame_id))].sort();
 if($('geometry-scope').value==='latest'&&ids.length){const selectedFrames=hypotheses.filter(x=>x.object.id===state.selected).map(x=>x.h.frame_id).sort();ids=(selectedFrames.length?selectedFrames:ids).slice(-1);hypotheses=hypotheses.filter(x=>x.h.frame_id===ids[0])}
 const oldFrames=three.frames.join('|');three.frames=ids;three.geometryCount=hypotheses.length;
 const groups=new Map();
 ids.forEach((id,i)=>{const group=new THREE.Group();group.position.set((i%3)*12,Math.floor(i/3)*-9,0);three.world.add(group);groups.set(id,group);cameraGuide(group,id)});
 if(!ids.length)cameraGuide(three.world,'Input camera · no committed geometry');
 for(const {object,h}of hypotheses){
  const group=groups.get(h.frame_id),selected=object.id===state.selected,color=selected?0xf3d59e:0x83bdc4;
  line(group,[...h.front_corners,h.front_corners[0]],color,selected?1:.35+.45*(1-h.depth_uncertainty));
  const positions=[];for(const index of [0,1,2,0,2,3]){const p=sourceVector(h.front_corners[index]);positions.push(p.x,p.y,p.z)}
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  const pick=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({side:THREE.DoubleSide,transparent:true,opacity:0,depthWrite:false}));pick.userData={id:object.id,frame_id:h.frame_id};group.add(pick);three.picks.push(pick);
  if(selected)label(group,object.id+' · unknown thickness',[h.center[0],h.center[1]-h.dimensions[1]/2-.04,h.center[2]],Math.max(.5,h.dimensions[0]*2));
 }
 $('geometry-status').textContent=ids.length?'Frame(s): '+ids.join(', ')+' · '+hypotheses.length+' recorded partial rectangles in '+ids.length+' separate observation scale(s). Source coordinates: x right, y down, Z forward; display flips y and Z. Shared metric scale: unavailable.':r.capabilities.gaze_dependent_depth?'Depth provider is present, but this memory snapshot contains no central object geometry.':'Depth is disabled: no object positions or extents are invented.';
 if(!state.hadGeometry||oldFrames!==ids.join('|')||state.fitRequested){fitCamera();state.fitRequested=false}state.hadGeometry=ids.length>0;
}
function render(){
 if(!r)return;state.t=snapTime(state.t);const view=snapshot(state.t),selection=selectedMemory(view),memory=selection.memory;
 const stepMode=$('step-mode').checked;$('timeline').max=stepMode?r.fixations.length:r.duration_ms;$('timeline').step=stepMode?'1':'any';$('timeline').value=stepMode?view.committed_index+1:state.t;
 $('elapsed').textContent=fmt(state.t/1000,2)+' s';
 $('status').textContent='Fixation '+view.fixation_index+' · '+view.phase+' · '+(view.fixation_index>view.committed_index?'new evidence not yet committed':'observation committed')+' · memory '+fmt(memory.time_ms/1000,3)+' s';
 $('memory-view').options[1].disabled=view.committed_index<0;$('memory-view').options[2].disabled=view.committed_index<0;
 objectInspector(memory,selection.caption);drawScene(view,memory);drawSensor(view);drawDepth(view);drawPriority(view);drawEnsemble(memory);decisions(view);progress(view);render3D(view,memory);
 window.observerLab={state,snapshot:view,displayMemory:memory,geometry:three?{frames:three.frames.slice(),count:three.geometryCount}:null,run:r,setTime:t=>{pause();setTime(t)},selectObject:id=>{state.selected=id;state.geometryKey=null;render()}};
}
function step(direction){pause();const offsets=r.fixations.map(f=>f.offset_ms),next=direction>0?offsets.find(t=>t>state.t+1):offsets.slice().reverse().find(t=>t<state.t-1);setTime(next===undefined?(direction>0?r.duration_ms:0):next)}
function saveJSON(value,name){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([pretty(value)],{type:'application/json'}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function pixel(canvas,event,width,height){const b=canvas.getBoundingClientRect();return[Math.max(0,Math.min(width-1,Math.floor((event.clientX-b.left)/b.width*width))),Math.max(0,Math.min(height-1,Math.floor((event.clientY-b.top)/b.height*height)))]}
$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('timeline').oninput=()=>{pause();const v=+$('timeline').value;setTime($('step-mode').checked?(v===0?0:r.fixations[v-1].offset_ms):v)};
$('step-mode').onchange=render;
$('play').onclick=()=>{if(state.playing){pause();return}if(state.t>=r.duration_ms)state.t=0;state.playing=true;$('play').textContent='Pause';let last=performance.now();function tick(now){if(!state.playing)return;setTime(state.t+(now-last)*+$('speed').value);last=now;if(state.t>=r.duration_ms)pause();else state.raf=requestAnimationFrame(tick)}state.raf=requestAnimationFrame(tick)};
for(const id of ['show-boxes','show-reliability','depth-channel','priority-layer','normalization','ensemble-region'])$(id).onchange=render;
$('memory-view').onchange=()=>{state.geometryKey=null;state.fitRequested=true;render()};
$('geometry-scope').onchange=()=>{state.geometryKey=null;state.fitRequested=true;render()};
$('object-select').onchange=()=>{state.selected=$('object-select').value;state.geometryKey=null;render()};
$('reset-camera').onclick=()=>fitCamera(false);$('show-cameras').onclick=()=>fitCamera(true);
$('scene').onclick=e=>{const view=snapshot(state.t),memory=selectedMemory(view).memory,p=pixel(e.target,e,r.width,r.height),objects=memory.objects.filter(o=>p[0]>=o.bbox[0]&&p[0]<o.bbox[2]&&p[1]>=o.bbox[1]&&p[1]<o.bbox[3]);if(objects.length){objects.sort((a,b)=>(a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1])-(b.bbox[2]-b.bbox[0])*(b.bbox[3]-b.bbox[1]));state.selected=objects[0].id;state.geometryKey=null;render()}};
$('priority').onclick=e=>{const view=snapshot(state.t);if(view.committed_index<0)return;const q=r.fixations[view.committed_index].priority;state.priorityPoint=pixel(e.target,e,q.width,q.height);render()};
$('depth').onmousemove=e=>{if(!r)return;state.depthPoint=pixel(e.target,e,r.width,r.height);drawDepth(snapshot(state.t))};
$('download-depth').onclick=()=>saveJSON(snapshot(state.t).depth,'depth-observation.json');
$('download-run').onclick=()=>saveJSON(r,'recorded-observer-run.json');
async function init(){
 try{
  const bytes=Uint8Array.from(atob(OBSERVER_PAYLOAD),c=>c.charCodeAt(0));
  bundle=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();r=bundle.result;
  const f=r.fixations[0];if(!f)throw Error('Recording has no fixations');
  const maps=[['total','Total priority'],...Object.keys(f.priority.components).map(k=>[k,k.replaceAll('_',' ')]),...Object.keys(f.priority.diagnostics||{}).map(k=>[k,k.replaceAll('_',' ')+' (diagnostic)']),['coverage','Coverage']];
  options('priority-layer',maps,'total');
  const objects=r.providers?.objects?.provider||'recorded provider',depth=r.providers?.depth?.runtime||(r.capabilities.gaze_dependent_depth?'recorded depth provider':'disabled');
  $('provenance').textContent='Schema '+r.schema_version+' · '+r.image_id+' · seed '+r.config.seed+' · objects: '+objects+' · depth: '+depth+'. No browser inference.';
  if(r.example_kind&&r.example_kind.toLowerCase().includes('mock')){$('mock-warning').hidden=false;$('mock-warning').textContent=r.example_kind+' — diagnostic fixture on an RGB photograph; no pretrained performance is claimed.'}
  init3D();render();
 }catch(error){$('status').textContent='Could not open the recording: '+error.message;console.error(error)}
}
init();
