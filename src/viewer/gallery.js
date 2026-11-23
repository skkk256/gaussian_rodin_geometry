const $ = id => document.getElementById(id);
const names = {
  pipeline_initialization:'管线初始化', read_and_prefilter:'读取 / 属性预过滤', sam3d_orientation:'SAM3D 五视角朝向',
  gaussian_render:'原高斯四视图', fal_refine:'Fal 提交 / 等待 / 下载', foreground_filter:'前景分割 / 过滤',
  filtered_gaussian_render:'过滤后高斯四视图', sampling_fps_normalization:'体积采样 / FPS / 归一化',
  inverse_transform:'完整逆变换 / 校验', rodin_generate:'Rodin 提交 / 等待 / 下载',
  bbox_registration:'bbox 配准', write_transform:'保存最终变换', export:'结果导出', mesh_render:'对齐后 Mesh 四视图',
  preprocess_sam3d:'前处理 + SAM3D 投票', fal:'Fal + 前景过滤 / 点云准备', rodin:'Rodin 生成', postprocess:'后处理',
  fal_nano_retry:'Nano Banana 重试 / 等待 / 下载', foreground_filter_nano:'Nano 前景分割 / 过滤',
  wait_gpu:'等待 GPU（不计处理耗时）', wait_segmentation:'等待前景分割器（不计处理耗时）'
};
const statuses = {completed:'完成', cached:'复用缓存', skipped:'未启用', failed:'失败', interrupted:'中断', running:'运行中', queued:'排队中'};
const seconds = value => value < 60 ? `${value.toFixed(2)} 秒` : `${Math.floor(value / 60)} 分 ${(value % 60).toFixed(1)} 秒`;
let catalog;
async function read(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data;
}
function cell(row, value, tag = 'td') { const node = document.createElement(tag); node.textContent = value; row.append(node); }
const processingSeconds = run => run.overheadSeconds + run.stages.reduce((total, stage) =>
  total + ('details' in stage ? stage.processingSeconds + stage.overheadSeconds : stage.kind === 'queue' ? 0 : stage.seconds), 0);
function showTiming(article, data) {
  const attempt = data.attempts.at(-1);
  article.querySelector('.duration').textContent = `${attempt.kind === 'mesh_backfill' ? '补图' : '本次'}处理 ${seconds(processingSeconds(attempt))} · 累计处理 ${seconds(data.attempts.reduce((total, run) => total + processingSeconds(run), 0))} · 不含本地排队`;
  const container = article.querySelector('.timing');
  container.replaceChildren();
  for (const [index, run] of data.attempts.entries()) {
    const label = document.createElement('p');
    label.textContent = `${run.kind === 'mesh_backfill' ? '补图' : '尝试'} ${index + 1} · ${statuses[run.status]} · 处理 ${seconds(processingSeconds(run))}`;
    const table = document.createElement('table');
    for (const stage of run.stages) {
      const row = table.insertRow();
      cell(row, names[stage.name]);
      cell(row, seconds('details' in stage ? stage.processingSeconds + stage.overheadSeconds : stage.seconds));
      cell(row, statuses[stage.status]);
      if ('details' in stage) {
        const breakdown = table.insertRow();
        cell(breakdown, '处理 / 本地排队（不计入） / 其他开销');
        cell(breakdown, `${seconds(stage.processingSeconds)} / ${seconds(stage.queueSeconds)} / ${seconds(stage.overheadSeconds)}`);
        cell(breakdown, '');
        for (const detail of stage.details) {
          const row = table.insertRow(); row.className = 'timing-detail';
          cell(row, names[detail.name]); cell(row, seconds(detail.seconds)); cell(row, statuses[detail.status]);
        }
      }
    }
    const row = table.insertRow(); cell(row, '调度 / 校验 / 文件操作等开销'); cell(row, seconds(run.overheadSeconds)); cell(row, '');
    container.append(label, table);
  }
}
async function loadVersion(article, item) {
  const version = article.querySelector('.version').value;
  const path = [version, item.scene_id, item.object_id].map(encodeURIComponent).join('/');
  const base = `/result/${path}/`;
  article.querySelector('.compare').href = `/compare.html?${new URLSearchParams({scene:item.scene_id, object:item.object_id, version})}`;
  const images = article.querySelector('.images'); images.replaceChildren();
  for (const [name, label] of [['gaussian.png','原始高斯'],['fal_refine.png','Fal 清理补全'],['gaussian_filter.png','过滤后高斯'],['rodin.png','对齐后 Rodin Mesh']]) {
    const figure = document.createElement('figure'), caption = document.createElement('figcaption');
    caption.textContent = label;
    const link = document.createElement('a'); link.href = base + name; link.target = '_blank';
    const image = document.createElement('img'); image.src = base + name; image.alt = label; image.loading = 'lazy';
    link.append(image); figure.append(caption, link); images.append(figure);
  }
  const downloads = article.querySelector('.downloads'); downloads.replaceChildren();
  for (const [name, label] of [['rodin.glb','原始 Rodin'],['transform.json','最终变换'],['gaussian_filter.ply','过滤后高斯 PLY'],['timing.json','耗时 JSON']]) {
    const link = document.createElement('a'); link.href = base + name; link.textContent = label; downloads.append(link);
  }
  showTiming(article, await read(base + 'timing.json'));
}
async function refresh() {
  catalog = await read('/api/objects');
  $('summary').textContent = `${catalog.objects.length} 个物体 · 原始模型 + 最终变换`;
  $('objects').replaceChildren();
  for (const item of catalog.objects) {
    const article = $('object-template').content.firstElementChild.cloneNode(true);
    article.dataset.search = `${item.scene_id}/${item.object_id}`.toLowerCase();
    article.querySelector('.scene').textContent = item.scene_id;
    article.querySelector('h2').textContent = item.object_id;
    const version = article.querySelector('.version');
    item.versions.forEach(v => version.add(new Option(v.label, v.id)));
    version.value = item.versions.at(-1).id;
    const report = error => { const node = article.querySelector('.object-error'); node.hidden = false; node.textContent = error.message; };
    version.onchange = () => loadVersion(article, item).catch(report);
    const button = article.querySelector('.retry'); button.disabled = !item.canRetry;
    button.title = item.canRetry ? '提交一次新的生成请求，保留首次结果' : '重跑需要原工作目录和凭据文件';
    button.onclick = async () => {
      button.disabled = true;
      try {
        await read('/api/retry', {method:'POST', headers:{'Content-Type':'application/json','X-Retry-Token':catalog.token},
          body:JSON.stringify({scene_id:item.scene_id,object_id:item.object_id,version:version.value,mode:article.querySelector('.mode').value})});
        await refreshJobs();
      } catch (error) { report(error); }
      finally { button.disabled = false; }
    };
    $('objects').append(article);
    await loadVersion(article, item).catch(report);
  }
  search(); showJobs(catalog.jobs); $('error').hidden = true;
}
function showJobs(jobs) {
  $('jobs').replaceChildren();
  for (const job of jobs) {
    const node = document.createElement('div'); node.className = 'job';
    node.textContent = `${job.scene_id} / ${job.object_id} · ${job.mode === 'rodin' ? 'Rodin' : 'Fal + Rodin'} · ${statuses[job.status]}`;
    const log = document.createElement('a'); log.href = `/api/jobs/${encodeURIComponent(job.id)}/log`; log.textContent = '运行日志'; log.target = '_blank';
    node.append(log); $('jobs').append(node);
  }
}
async function refreshJobs() {
  const data = await read('/api/objects'); showJobs(data.jobs);
  if (data.jobs.some(j => j.status === 'completed' && !catalog.jobs.some(old => old.id === j.id && old.status === 'completed'))) await refresh();
  else catalog.jobs = data.jobs;
}
function search() { for (const node of $('objects').children) node.hidden = !node.dataset.search.includes($('search').value.toLowerCase()); }
function fail(error) { $('error').hidden = false; $('error').textContent = error.message; }
$('search').oninput = search;
$('refresh').onclick = () => refresh().catch(fail);
refresh().catch(fail);
setInterval(() => refreshJobs().catch(fail), 5000);
