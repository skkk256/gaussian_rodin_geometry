import * as THREE from 'three';
import {OverlayViewer} from './overlay.js?v=shared-geometry-demand-v1';
const $=id=>document.getElementById(id),percent=value=>(value*100).toFixed(2)+'%';
async function json(url){const response=await fetch(url);if(!response.ok)throw Error(`${response.status}: ${url}`);return response.json();}
function fail(error){console.error(error);$('error').textContent=error.message;document.body.dataset.ready='error';$('loading').hidden=true;$('object').disabled=false;}
const presets={default:['original','final'],filtered:['filtered','final'],stages:['inverse','bbox','final'],points:['fps','condition'],rodin:['raw','condition'],all:['original','filtered','fps','condition','raw','inverse','bbox','final']};
const viewer=new OverlayViewer($('viewport'),(pending,visible)=>{
  $('loading').hidden=pending===0;$('object').disabled=pending>0;
  document.querySelectorAll('[data-preset]').forEach(button=>button.disabled=pending>0);
  document.body.dataset.ready=pending?'loading':'true';$('visible').textContent=visible.join(' + ');
});
let rows,current,report;
const midpoint=bounds=>new THREE.Vector3(...bounds[0]).add(new THREE.Vector3(...bounds[1])).multiplyScalar(.5);
const resultUrl=file=>current.base+file+'?v='+current.renderVersion;
function table(){
  $('cases').replaceChildren();
  for(const row of rows){const tr=document.createElement('tr');tr.dataset.case=row.case;tr.className=row.case===current?.case?'selected':'';
    const values=row.validation;
    for(const text of [row.case,...['inverse','bbox','final'].map(key=>values?percent(values[key].meanIoU):'—'),row.fit?`${row.fit.matches} / ${row.fit.inliers}`:'—',row.totalSeconds?row.totalSeconds.toFixed(1)+' s':'—',row.status==='completed'?(row.finalStage==='bbox'?'bbox（匹配不足）':row.accepted?'DINO 完成':'DINO · 需查看'):row.status]){const td=document.createElement('td');td.textContent=text;tr.append(td);}
    tr.onclick=()=>{if(!viewer.pending)choose(row.case).catch(fail);};$('cases').append(tr);
  }
}
function images(){
  if(!$('projections').open)return;const index=$('view').value;
  for(const name of ['gaussian','inverse','bbox','final'])$(name+'-image').src=resultUrl(`${name}-${index}.png`);
}
function layerControls(){
  $('layers').replaceChildren();
  for(const layer of viewer.layers){const panel=document.createElement('div');panel.className='layer';
    panel.innerHTML=`<label><input id="${layer.id}-visible" type="checkbox" ${layer.visible?'checked':''}>${layer.label}</label><small id="${layer.id}-value">${layer.opacity.toFixed(2)}</small><input id="${layer.id}-opacity" aria-label="${layer.label}透明度" type="range" min="0" max="1" step=".01" value="${layer.opacity}">`;
    panel.querySelector('input[type=checkbox]').onchange=async event=>{layer.visible=event.target.checked;try{await viewer.setVisible(viewer.layers.filter(l=>l.visible).map(l=>l.id));}catch(error){fail(error);}};
    panel.querySelector('input[type=range]').oninput=event=>{layer.opacity=Number(event.target.value);$(layer.id+'-value').textContent=layer.opacity.toFixed(2);viewer.style(layer);};$('layers').append(panel);
  }
}
async function choose(id){
  document.body.dataset.ready='loading';$('loading').hidden=false;$('error').textContent='';
  current=rows.find(row=>row.case===id);$('object').value=id;history.replaceState(null,'','?'+new URLSearchParams({oid:id}));table();
  if(current.status!=='completed'){viewer.reset();report=null;$('quality').textContent=current.error||'该物体尚未完成';for(const id of ['metrics','layers','downloads','matches'])$(id).replaceChildren();$('report').textContent='';$('condition-info').textContent='';$('projections').open=false;$('matches-panel').open=false;$('loading').hidden=true;$('object').disabled=false;document.body.dataset.ready='true';return;}
  report=await json(resultUrl('result.json'));
  const worldShift=midpoint(report.fpsWorldBounds).negate(),conditionShift=midpoint(report.conditionBounds).negate();
  const worldDisplay=new THREE.Matrix4().makeTranslation(...worldShift),conditionDisplay=new THREE.Matrix4().makeTranslation(...conditionShift);
  const worldBounds=report.fpsWorldBounds.map(point=>point.map((value,i)=>value+worldShift.getComponent(i)));
  const definitions=[
    {id:'original',label:'原始高斯',kind:'gs',url:current.base+'original-gaussian.ply',bounds:report.fpsWorldBounds,display:worldDisplay,opacity:.8},
    {id:'filtered',label:'过滤后高斯',kind:'gs',url:current.base+'filtered-gaussian.ply',bounds:report.fpsWorldBounds,display:worldDisplay,opacity:.8},
    {id:'fps',label:'过滤高斯采样 FPS · 未归一化',kind:'points',url:current.base+'pointcloud-fps-world.json',display:worldDisplay,color:'#55aaff',opacity:1},
    {id:'condition',label:'实际上传 condition FPS',kind:'points',url:current.base+'pointcloud-condition.json',display:conditionDisplay,color:'#f477e6',opacity:1},
    {id:'raw',label:'Rodin 原始结果',kind:'mesh',matrix:report.stages.raw,display:conditionDisplay,color:'#dbe2ea',opacity:.5},
    {id:'inverse',label:'Rodin 应用逆变换',kind:'mesh',matrix:report.stages.inverse,display:worldDisplay,color:'#a4e783',opacity:.6},
    {id:'bbox',label:'逆变换 + FPS bbox',kind:'mesh',matrix:report.stages.bbox,display:worldDisplay,color:'#ffad53',opacity:.6},
    {id:'final',label:report.finalStage==='bbox'?'最终结果 · bbox（匹配不足）':'最终八视角配准',kind:'mesh',matrix:report.stages.final,display:worldDisplay,color:'#b294ff',opacity:.7}
  ];
  for(const layer of definitions)layer.visible=presets.default.includes(layer.id);
  $('quality').className=report.diagnosticCheck.accepted?'good':'warn';
  $('quality').textContent=report.finalStage==='bbox'?`${id} · 最终使用 bbox。${report.dino.reason}`:`${id} · ${report.fit.matches} 对匹配 / ${report.fit.inliers} 对内点 · ${report.totalSeconds.toFixed(1)} 秒。`+(report.diagnosticCheck.accepted?'记录：完成。':'记录：'+report.diagnosticCheck.reasons.join('；'));
  $('final-caption').textContent=report.finalStage==='bbox'?'最终结果 · bbox（匹配不足）':'最终八视角配准';
  $('metrics').innerHTML=`<table><tr><th>指标</th><th>仅逆变换</th><th>FPS bbox</th><th>最终八视角</th></tr><tr><td>平均轮廓 IoU</td>${['inverse','bbox','final'].map(key=>`<td>${percent(report.validation[key].meanIoU)}</td>`).join('')}</tr><tr><td>最差轮廓 IoU</td>${['inverse','bbox','final'].map(key=>`<td>${percent(report.validation[key].worstIoU)}</td>`).join('')}</tr></table>`;
  $('condition-info').textContent=JSON.stringify({pointCount:4096,sameFilteredGaussianSamples:true,conditionBounds:report.conditionBounds,fpsWorldBounds:report.fpsWorldBounds,worldDisplayTranslation:worldShift.toArray(),conditionDisplayTranslation:conditionShift.toArray()},null,2);
  $('report').textContent=JSON.stringify(report,null,2);
  $('view').replaceChildren();for(const index of report.fitViews)$('view').add(new Option(`${index<8?'+30°':'−30°'} · 视角 ${index}`,index));
  $('projections').open=false;$('matches-panel').open=false;$('matches').replaceChildren();
  for(const name of ['gaussian','inverse','bbox','final'])$(name+'-image').removeAttribute('src');
  $('downloads').replaceChildren();for(const [label,file] of [[report.finalStage==='bbox'?'最终 GLB（bbox）':'最终 GLB','final.glb'],['bbox GLB','bbox.glb'],['完整指标和矩阵','result.json'],['未归一化 FPS','pointcloud-fps-world.json'],['八视角对应点','matches.json']]){const a=document.createElement('a');a.textContent=label;a.href=resultUrl(file);$('downloads').append(a);}const conditionLink=document.createElement('a');conditionLink.textContent='实际 condition FPS';conditionLink.href=current.base+'pointcloud-condition.json';$('downloads').append(conditionLink);
  viewer.material=$('material').value;
  await viewer.setCase(definitions,current.base+'model.glb',worldBounds,new THREE.Matrix4().fromArray(report.frame.pointcloudToWorld.matrix));
  layerControls();
}
$('object').onchange=()=>choose($('object').value).catch(fail);
for(const button of document.querySelectorAll('[data-preset]'))button.onclick=async()=>{try{await viewer.setVisible(presets[button.dataset.preset]);for(const layer of viewer.layers)$(layer.id+'-visible').checked=layer.visible;}catch(error){fail(error);}};
$('material').onchange=()=>viewer.setMaterial($('material').value).catch(fail);
for(const [id,property] of [['point-size','pointSize'],['point-xray','pointXray'],['wireframe','wireframe']])$(id).oninput=()=>{viewer[property]=$(id).type==='checkbox'?$(id).checked:Number($(id).value);for(const layer of viewer.layers)viewer.style(layer);};
$('camera-buttons').onpointerdown=event=>event.stopPropagation();for(const button of document.querySelectorAll('[data-view]'))button.onclick=()=>viewer.view(button.dataset.view);
$('fit').onclick=()=>viewer.frame();$('fullscreen').onclick=()=>document.fullscreenElement?document.exitFullscreen():$('viewport').requestFullscreen();
$('view').onchange=images;$('projections').ontoggle=images;
$('matches-panel').ontoggle=()=>{
  if(!$('matches-panel').open||$('matches').children.length||!report)return;
  if(report.finalStage==='bbox'){const note=document.createElement('p');note.textContent='最终使用 bbox。'+report.dino.reason;$('matches').append(note);}
  if(!report.fit)return;
  for(const [ordinal,index] of report.fitViews.entries()){
    const figure=document.createElement('figure'),url=resultUrl(`matches-${index}.png`);
    figure.innerHTML=`<figcaption>视角 ${index} · ${report.fit.matchesPerView[ordinal]} 对候选 / ${report.fit.inliersPerView[ordinal]} 对内点</figcaption><a target="_blank" href="${url}"><img loading="lazy" decoding="async" src="${url}" alt="视角 ${index} 对应点"></a>`;$('matches').append(figure);
  }
};
window.overlayState=()=>({case:current?.case,...viewer.state()});window.overlayRender=viewer.requestRender;
(async()=>{rows=await json('manifest.json');for(const row of rows)$('object').add(new Option(row.case,row.case));$('batch-status').textContent=`${rows.filter(r=>r.status==='completed').length} / ${rows.length} 可显示 · ${rows.filter(r=>r.finalStage==='bbox').length} 个使用 bbox`;await choose(new URLSearchParams(location.search).get('oid')||rows[0].case);})().catch(fail);
