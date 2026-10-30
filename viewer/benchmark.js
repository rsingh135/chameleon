/* Local scene matrix. Every job uses one frozen method digest and target list. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {scenes: [], reviewed: [], method: null, report: null, busy: false, stop: false, urls: [],
    renderedReport: null, sceneViews: new Map(), runSources: new Map(), jobExports: new Map(), reportExport: null, stale: false, disposed: false};
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  function error(message = '') {
    $('batchError').hidden = !message;
    $('batchError').textContent = message;
  }
  function connection(online) {
    $('connection').classList.toggle('online', online);
    $('connection').textContent = online ? 'Engine connected' : 'Engine unavailable';
  }
  function validateRequestPolicy(policy) {
    if (!policy || typeof policy !== 'object' || Array.isArray(policy)
      || !['studio_timeout_ms', 'benchmark_timeout_ms'].every(key => Number.isInteger(policy[key]) && policy[key] > 0 && policy[key] <= 1500000)
      || typeof policy.model_configured !== 'boolean') throw Object.assign(new Error('The engine returned an invalid resize runtime policy. Reconnect before testing; no resize job was submitted.'), {reconnect: true});
    return policy;
  }
  function setMethod(method) {
    if (method?.id !== 'deterministic-current' || !/^[a-f0-9]{64}$/.test(method?.digest || '')) throw Object.assign(new Error('The engine returned an invalid method identity. Reconnect before testing.'), {reconnect: true});
    if (method.options?.request_policy !== undefined) validateRequestPolicy(method.options.request_policy);
    if (state.report && state.report.method.digest !== method.digest) dirty();
    state.method = method;
    const plannerRequested = method.analyzer?.enabled === true;
    const criticRequested = method.critic?.enabled === true;
    $('methodName').textContent = plannerRequested ? 'Current engine · local visual planner requested' : method.id;
    $('methodPolicy').textContent = plannerRequested
      ? `A local visual planner is configured. Result traces report whether it ran, changed the layout, or used a fallback. ${criticRequested ? 'A learned critic is configured.' : 'The learned critic is off.'} Visual quality remains unreviewed. No model is downloaded or trained here.`
      : criticRequested
        ? 'Current deterministic engine. Vision analyzer is off; a learned critic is configured. Visual quality remains unreviewed. No model is downloaded or trained here.'
        : 'Current deterministic engine. Vision analyzer and learned critic are off. No model is downloaded or trained here.';
    $('methodDigest').textContent = `Code ${method.digest.slice(0, 16)}…`;
    $('retryConnection').hidden = true;
  }
  async function reconnect() {
    if (state.busy || $('retryConnection').disabled) return;
    $('retryConnection').disabled = true;
    setBusy(true);
    error();
    try {
      setMethod(await request('/api/benchmark/method'));
      $('runBatch').disabled = !state.scenes.length;
      $('batchStatus').textContent = state.scenes.length ? 'Engine ready. Your queued scenes are preserved; run when ready.' : 'Engine ready. Add scenes to start.';
    } catch (failure) {
      error(failure.message);
      $('retryConnection').hidden = false;
    } finally { $('retryConnection').disabled = false; setBusy(false); }
  }
  async function request(path, body) {
    const controller = new AbortController();
    const declared = state.method?.options?.request_policy;
    const budget = body ? (declared === undefined ? 120000 : validateRequestPolicy(declared).benchmark_timeout_ms) : 10000;
    const timeout = setTimeout(() => controller.abort(), budget);
    const timeoutError = () => new Error(body
      ? 'Job timed out. The local engine may still be finishing it. No further jobs were submitted.'
      : 'Engine metadata request timed out after 10 seconds. Reconnect and retry; this request submitted no resize job.');
    try {
      let response;
      try {
        response = await fetch(path, {signal: controller.signal, cache: 'no-store', ...(body ? {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
        } : {})});
        if (controller.signal.aborted) throw Object.assign(new Error('Request expired.'), {name: 'AbortError'});
      } catch (failure) {
        if (failure.name === 'AbortError') throw timeoutError();
        connection(false);
        throw Object.assign(new Error('Cannot reach the local engine. Start the server, then reconnect.'), {reconnect: true});
      }
      connection(true);
      let result;
      try { result = await response.json(); }
      catch (failure) { if (failure.name === 'AbortError') throw timeoutError(); throw new Error('The engine returned invalid JSON. Completed jobs are preserved; retry the run.'); }
      if (controller.signal.aborted) throw timeoutError();
      if (!response.ok) throw Object.assign(new Error(result?.error || `Request failed (${response.status}).`), {reconnect: response.status >= 500 || response.status === 409});
      return result;
    } finally { clearTimeout(timeout); }
  }
  function releaseUrls() {
    for (const url of state.urls) URL.revokeObjectURL(url);
    state.urls = [];
    state.jobExports.clear();
    state.reportExport = null;
  }
  function blobResource(data, filename, type = 'application/json') {
    const url = URL.createObjectURL(new Blob([type === 'application/json' ? JSON.stringify(data, null, 2) : data], {type}));
    state.urls.push(url);
    return Object.freeze({url, filename});
  }
  function exportLink(text, resource) {
    const link = node('a', text);
    if (resource) { link.href = resource.url; link.download = resource.filename; }
    else { link.setAttribute('aria-disabled', 'true'); link.tabIndex = -1; }
    return link;
  }
  function dirty() {
    $('downloadReport').removeAttribute('href');
    $('downloadReport').setAttribute('aria-disabled', 'true');
    $('downloadReport').tabIndex = -1;
    if (state.report) {
      state.stale = true;
      $('runIdentity').textContent = `${state.report.run_id} · previous run; setup changed`;
      $('batchStatus').textContent = 'Setup changed. Displayed results belong to the previous run; run again to update.';
      for (const link of $('batchResults').querySelectorAll('a[download]')) {
        link.removeAttribute('href');
        link.setAttribute('aria-disabled', 'true');
        link.tabIndex = -1;
      }
      releaseUrls();
      state.runSources.clear();
    }
    $('runBatch').disabled = state.busy || !state.method || !state.scenes.length;
  }
  function setBusy(value) {
    state.busy = value;
    $('batchSettings').disabled = value;
    $('runBatch').disabled = value || !state.method || !state.scenes.length;
  }
  function renderQueue() {
    $('sceneList').replaceChildren();
    state.scenes.forEach((scene, index) => {
      const li = node('li');
      const title = node('div', undefined, 'scene-title');
      title.append(node('strong', scene.label));
      const remove = node('button', 'Remove');
      remove.type = 'button';
      remove.setAttribute('aria-label', `Remove ${scene.label}`);
      remove.addEventListener('click', () => {
        state.scenes.splice(index, 1); dirty(); renderQueue();
        if (!state.report) $('batchStatus').textContent = state.scenes.length ? `${state.scenes.length} scene${state.scenes.length === 1 ? '' : 's'} ready. Choose shared targets, then run.` : 'No scenes queued. Add designs or images to start.';
      });
      title.append(remove);
      const canvas = scene.payload.source.canvas;
      li.append(title, node('p', `${canvas.width} × ${canvas.height} · ${scene.input_kind === 'flat-image' ? 'Flat image · fitting only' : `${scene.payload.source.elements.length} editable element${scene.payload.source.elements.length === 1 ? '' : 's'}`}`, 'field-help'));
      const label = node('label', 'Source family');
      label.htmlFor = `family-${scene.id}`;
      const input = node('input');
      input.id = label.htmlFor;
      input.value = scene.family;
      input.maxLength = 100;
      input.addEventListener('input', () => { scene.family = input.value.trim() || 'unassigned'; dirty(); });
      li.append(label, input);
      $('sceneList').append(li);
    });
    $('runBatch').disabled = state.busy || !state.method || !state.scenes.length;
  }
  function sourcePayload(value) {
    const payload = value?.source ? value : {source: value};
    const source = payload?.source;
    if (!source?.canvas || !Array.isArray(source.elements) || !source.elements.length) throw new Error('A JSON design needs a canvas and editable elements.');
    const {width, height} = source.canvas;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 8192 || height > 8192 || width * height > 20000000) throw new Error('Source dimensions must be 1–8192 px per side and at most 20 MP.');
    if (source.elements.length > 200) throw new Error('A scene can contain at most 200 elements.');
    const ids = new Set();
    for (const element of source.elements) {
      if (!element || typeof element.id !== 'string' || !element.id || element.id.length > 200 || ids.has(element.id) || !['text', 'image', 'shape'].includes(element.type) || !['x', 'y', 'width', 'height'].every(key => typeof element[key] === 'number' && Number.isFinite(element[key])) || Math.abs(element.x) > 32768 || Math.abs(element.y) > 32768 || element.width < 0.001 || element.height < 0.001 || element.width > 8192 || element.height > 8192 || element.width * element.height > 20000000) throw new Error('Scene elements need unique IDs, supported types and finite geometry within the engine limits.');
      ids.add(element.id);
    }
    return payload;
  }
  function targetsFromText(text) {
    const lines = text.trim().split(/\n/).filter(line => line.trim());
    if (!lines.length || lines.length > 4) throw new Error('Enter 1–4 target sizes, one per line.');
    const seen = new Set();
    return lines.map(line => {
      const match = line.trim().match(/^(\d+)\s*[x×]\s*(\d+)$/i);
      if (!match) throw new Error(`Invalid target “${line}”. Use width x height.`);
      const width = Number(match[1]), height = Number(match[2]);
      if (width < 1 || height < 1 || width > 8192 || height > 8192 || width * height > 20000000) throw new Error('Targets must be 1–8192 px per side and at most 20 MP.');
      const key = `${width}x${height}`;
      if (seen.has(key)) throw new Error(`Duplicate target ${key}.`);
      seen.add(key);
      return {width, height};
    });
  }
  function comparisonMode() { return $('comparisonMode').value === 'two-level' ? 'two-level' : 'shared'; }
  function updateComparisonMode() {
    const twoLevel = comparisonMode() === 'two-level';
    $('levelOneHelp').hidden = !twoLevel;
    $('targetSizesLabel').textContent = twoLevel ? 'Level 2 · shared width × height, one per line' : 'Shared width × height, one per line';
    $('targetPlanHelp').textContent = twoLevel
      ? 'Up to 2 shared sizes plus 2 proportional sizes per scene, and 16 unique jobs total. Four scenes × four sizes fits. Repeated sizes share one verified job. Engine layout changes remain visible in diagnostics.'
      : 'Up to 4 targets and 16 total jobs. Each target is applied to every scene. Maximum 8192 px per side and 20 MP.';
  }
  function jobPlan(scenes, targets, mode) {
    if (mode === 'two-level' && targets.length > 2) throw new Error('Two-level comparison allows up to 2 shared targets, plus 50% and 25% proportional targets.');
    const jobs = [];
    for (const scene of scenes) {
      const perScene = [];
      const add = (target, comparison) => {
        let job = perScene.find(item => item.target_canvas.width === target.width && item.target_canvas.height === target.height);
        if (!job) {
          job = {job_key: `${scene.id}-${target.width}x${target.height}`, scene_id: scene.id, target_canvas: {...target}, comparisons: []};
          perScene.push(job);
        }
        job.comparisons.push(comparison);
      };
      if (mode === 'two-level') {
        for (const scale of [0.5, 0.25]) {
          const canvas = scene.payload.source.canvas;
          add({width: Math.max(1, Math.round(canvas.width * scale)), height: Math.max(1, Math.round(canvas.height * scale))}, {level: 'resolution', requested_scale: scale});
        }
      }
      for (const target of targets) add(target, {level: mode === 'two-level' ? 'shared-aspect' : 'shared'});
      if (perScene.length > 4) throw new Error('Each scene is limited to 4 target sizes.');
      jobs.push(...perScene);
    }
    if (jobs.length > 16) throw new Error('A benchmark is limited to 16 jobs. Reduce scenes or targets.');
    return jobs;
  }
  async function browseReviewed() {
    if (state.busy) return;
    error(); setBusy(true);
    try {
      const {examples} = await request('/api/posters');
      if (!Array.isArray(examples)) throw new Error('The reviewed poster catalogue is unavailable. Your scenes are unchanged.');
      const reviewed = examples.filter(item => item?.split === 'reviewed' && item.ready === true);
      const ids = new Set();
      for (const item of reviewed) {
        if (typeof item.id !== 'string' || !/^[a-z0-9][a-z0-9_-]{0,99}$/.test(item.id) || ids.has(item.id) || typeof item.title !== 'string' || !item.title.trim() || item.title.length > 200 || /[\x00-\x1f]/.test(item.title)) throw new Error('Reviewed poster IDs and titles must be valid and unique. Your scenes are unchanged.');
        ids.add(item.id);
      }
      state.reviewed = reviewed;
      $('reviewedPosters').replaceChildren();
      for (const poster of reviewed) {
        const option = node('option', poster.title); option.value = poster.id;
        $('reviewedPosters').append(option);
      }
      $('reviewedPicker').hidden = false;
      $('reviewedStatus').textContent = `${reviewed.length} reviewed posters available. Source families remain for you to declare.`;
    } catch (failure) { error(failure.message); }
    finally { setBusy(false); }
  }
  async function addReviewed(pack = false) {
    if (state.busy) return;
    error(); setBusy(true);
    try {
      const selected = pack
        ? state.reviewed.filter(item => !state.scenes.some(scene => scene.id === `poster-${item.id}`)).slice(0, Math.min(4, 8 - state.scenes.length)).map(item => item.id)
        : [...$('reviewedPosters').selectedOptions].map(option => option.value);
      if (!selected.length || selected.length > 4 || new Set(selected).size !== selected.length) throw new Error('Select between 1 and 4 different reviewed posters.');
      const posters = selected.map(id => state.reviewed.find(item => item.id === id));
      if (posters.some(item => !item)) throw new Error('Only reviewed, ready posters from this catalogue can be added.');
      const added = posters.filter(item => !state.scenes.some(scene => scene.id === `poster-${item.id}`));
      if (state.scenes.length + added.length > 8) throw new Error('Keep at most 8 scenes in one benchmark.');
      const queued = [];
      for (const poster of added) {
        const payload = sourcePayload(await request(`/api/posters/${encodeURIComponent(poster.id)}/bundle`));
        queued.push({id: `poster-${poster.id}`, label: poster.title, family: 'unassigned', input_kind: 'editable-design', library_poster_id: poster.id, payload});
      }
      if (!queued.length) { $('reviewedStatus').textContent = 'Those posters are already queued.'; return; }
      state.scenes.push(...queued); dirty(); renderQueue();
      $('batchStatus').textContent = `Added ${queued.length} reviewed poster${queued.length === 1 ? '' : 's'}. Choose comparison levels and source families, then run.`;
      $('reviewedStatus').textContent = 'Exact editable source bundles loaded. Reserved sources were not loaded.';
    } catch (failure) { error(failure.message); }
    finally { setBusy(false); }
  }
  async function imagePayload(file) {
    if (file.size > 8 * 1024 * 1024) throw new Error(`${file.name}: images must be at most 8 MiB.`);
    const bytes = await file.arrayBuffer();
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(value => value.toString(16).padStart(2, '0')).join('');
    let timer, expired = false;
    const decoding = createImageBitmap(file);
    decoding.then(bitmap => { if (expired) bitmap.close(); }, () => {});
    let image;
    try {
      image = await Promise.race([decoding, new Promise((_, reject) => {
        timer = setTimeout(() => { expired = true; reject(new Error(`${file.name}: image decoding timed out. Try another file.`)); }, 30000);
      })]);
    } finally { clearTimeout(timer); }
    const canvas = {width: image.width, height: image.height};
    image.close();
    if (canvas.width > 8192 || canvas.height > 8192 || canvas.width * canvas.height > 20000000) throw new Error(`${file.name}: image dimensions exceed the 8192 px / 20 MP limit.`);
    const mime = file.type === 'image/jpeg' || /\.jpe?g$/i.test(file.name) ? 'image/jpeg' : 'image/png';
    const asset = `raster-${digest.slice(0, 20)}.${mime === 'image/png' ? 'png' : 'jpg'}`;
    let binary = '';
    const array = new Uint8Array(bytes);
    for (let offset = 0; offset < array.length; offset += 32768) binary += String.fromCharCode(...array.subarray(offset, offset + 32768));
    return {
      source: {canvas, elements: [{id: 'original-image', type: 'image', x: 0, y: 0, ...canvas, asset_id: asset, fit: 'contain', z_index: 0}]},
      document: {snapshot_id: `raster-${digest}`, page_id: 'raster-import', artboard_id: 'original-image'},
      assets: [{asset_id: asset, mime_type: mime, data_base64: btoa(binary)}], fonts: [],
    };
  }
  async function importFiles() {
    const files = [...$('sceneFiles').files];
    if (!files.length) return;
    error();
    setBusy(true);
    try {
      if (state.scenes.length + files.length > 8) throw new Error('Keep at most 8 scenes in one benchmark.');
      const added = [];
      for (const file of files) {
        if (file.size > 24 * 1024 * 1024) throw new Error(`${file.name}: file exceeds 24 MiB.`);
        const raster = /\.(png|jpe?g)$/i.test(file.name);
        const payload = raster ? await imagePayload(file) : sourcePayload(JSON.parse(await file.text()));
        added.push({id: crypto.randomUUID(), label: file.name, family: 'unassigned', input_kind: raster ? 'flat-image' : 'editable-design', payload});
      }
      state.scenes.push(...added);
      dirty();
      renderQueue();
      $('batchStatus').textContent = `Added ${added.length} scene${added.length === 1 ? '' : 's'}. Choose families and shared targets, then run.`;
    } catch (failure) { error(failure.message); }
    finally { $('sceneFiles').value = ''; setBusy(false); }
  }
  function rowFor(report, scene, plan) {
    const target = plan.target_canvas || plan;
    const matches = row => row?.scene_id === scene.id && row.target_canvas.width === target.width && row.target_canvas.height === target.height && (!report.job_plan || row.job_key === plan.job_key);
    return report.results.find(matches) || (matches(report.interrupted_job) ? report.interrupted_job : undefined);
  }
  async function validateResult(row, scene, target) {
    if (row.status === 'error') {
      if (row.result != null || row.contract_passed != null || typeof row.error !== 'string') throw new Error('The engine returned an inconsistent failed job. The run stopped.');
      return;
    }
    const result = row.result, design = result?.design;
    sourcePayload({source: design});
    const sourceIds = scene.payload.source.elements.map(element => element.id).sort();
    const resultIds = design?.elements?.map(element => element?.id).sort();
    if (!design || design.canvas?.width !== target.width || design.canvas?.height !== target.height || JSON.stringify(sourceIds) !== JSON.stringify(resultIds) || !Array.isArray(result?.report?.violations) || typeof row.contract_passed !== 'boolean' || row.contract_passed !== (result.report.violations.length === 0) || !/^[a-f0-9]{64}$/.test(row.source_sha256 || '') || !/^[a-f0-9]{64}$/.test(row.resources_sha256 || '') || !Number.isFinite(row.elapsed_ms) || row.elapsed_ms < 0) throw new Error('The engine returned incomplete or mismatched result data. The run stopped.');
    await Promise.all([
      decodePreview(result.source_png, scene.payload.source.canvas),
      decodePreview(result.output_png, target),
    ]);
  }
  async function decodePreview(url, canvas) {
    if (typeof url !== 'string' || !/^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(url)) throw new Error('The engine returned an invalid PNG preview. The run stopped.');
    // Verify binary encoding before admitting a row, so recovery cannot fail
    // again while exporting or rebuilding previously completed results.
    atob(url.slice(url.indexOf(',') + 1));
    const image = new Image();
    image.src = url;
    let timeout;
    try {
      await Promise.race([image.decode(), new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new Error('Preview decoding timed out. Completed jobs are preserved.')), 30000);
      })]);
      if (image.naturalWidth !== canvas.width || image.naturalHeight !== canvas.height) throw new Error('Preview dimensions do not match the submitted scene and target.');
    } catch (failure) {
      image.src = '';
      throw new Error(`Cannot verify the rendered preview: ${failure.message}`);
    } finally { clearTimeout(timeout); }
  }
  function exportsFor(scene, target, row) {
    if (state.stale || state.disposed) return null;
    const key = row.job_key || `${scene.id}-${target.width}x${target.height}`;
    const cached = state.jobExports.get(key);
    if (cached) return cached;
    if (state.jobExports.size >= 16) throw new Error('The verified export cache exceeded the job limit.');
    const source = state.runSources.get(scene.id);
    if (!source) throw new Error('The frozen original input is unavailable for this result.');
    const basename = `${scene.id}-${target.width}x${target.height}`;
    const portable = ['document', 'assets', 'fonts'].some(field => field in source);
    const exported = portable ? {...source, source: row.result.design, target_canvas: {...target}} : row.result.design;
    const binary = atob(row.result.output_png.split(',')[1]);
    const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
    const pair = Object.freeze({
      json: blobResource(exported, `${basename}.json`),
      png: blobResource(bytes, `${basename}.png`, 'image/png'),
    });
    state.jobExports.set(key, pair);
    return pair;
  }
  function renderResultCell(report, scene, plan, level) {
    const target = plan.target_canvas || plan;
    const row = rowFor(report, scene, plan);
    const cell = node('div', undefined, 'render-cell');
    const uses = (plan.comparisons || []).filter(item => item.level === level);
    const requested = level === 'resolution' ? `${uses.map(item => `${item.requested_scale * 100}%`).join(' / ')} requested · ` : '';
    cell.append(node('p', `${requested}${target.width} × ${target.height}`, 'render-label'));
    const stage = node('div', undefined, 'render-stage');
    if (row?.result) {
      const image = node('img'); image.src = row.result.output_png; image.alt = `${scene.label}, result at ${target.width} by ${target.height}`; stage.append(image);
      cell.append(stage);
      const diagnostic = node('p', `${row.contract_passed ? 'Contract checks passed' : 'Contract needs review'} · ${(row.elapsed_ms / 1000).toFixed(2)} s · visual quality unreviewed`, 'render-notes');
      diagnostic.dataset.outcome = row.contract_passed ? 'pass' : 'review';
      cell.append(diagnostic);
      const smallText = row.result.design.elements.filter(element => element.type === 'text' && Number.isFinite(element.font_size) && element.font_size < 10);
      if (smallText.length) {
        const smallest = Math.min(...smallText.map(element => element.font_size));
        cell.append(node('p', `${smallText.length} text element${smallText.length === 1 ? '' : 's'} below 10 px (smallest ${smallest.toFixed(1)} px). Inspect readability at export size.`, 'render-notes'));
      }
      const visualModel = row.result.report.layout_state?.visual_model;
      if (visualModel?.requested === true) {
        const outcome = visualModel.model_executed !== true
          ? 'Local visual planner was requested but did not run; the engine used its fallback.'
          : visualModel.layout_changed === true
            ? 'Local visual planner ran and influenced the layout.'
            : 'Local visual planner ran; no layout change was recorded.';
        cell.append(node('p', `${outcome} Visual quality is unreviewed.`, 'render-notes'));
      }
      if (report.job_plan) {
        const layout = row.result.report.layout_state || {};
        const actual = [...new Set([layout.name, layout.mode, layout.topology].filter(value => typeof value === 'string' && value))].join(' · ') || 'not reported';
        cell.append(node('p', `Engine layout: ${actual}. Proportional targets are a request, not proof of unchanged composition.`, 'field-help'));
      }
      const links = node('div', undefined, 'result-links');
      const pair = exportsFor(scene, target, row);
      links.append(exportLink('JSON ↓', pair?.json), exportLink('PNG ↓', pair?.png));
      cell.append(links);
      const details = node('details'); details.append(node('summary', 'Diagnostics and resource identity'));
      details.append(node('pre', JSON.stringify({job_key: row.job_key, comparisons: row.comparisons, source_sha256: row.source_sha256, resources_sha256: row.resources_sha256, report: row.result.report, warnings: row.result.warnings}, null, 2)));
      cell.append(details);
    } else {
      stage.append(node('p', row?.error || (report.status === 'running' ? 'Waiting for this job' : 'Not run'))); cell.append(stage);
    }
    return cell;
  }
  function createSceneView(report, scene) {
      const cells = [];
      const appendCell = (parent, plan, level) => {
        const cell = node('div', undefined, 'render-cell');
        parent.append(cell); cells.push({cell, plan, level, row: null, phase: null});
      };
      const article = node('article', undefined, 'scene-result');
      const heading = node('header');
      const title = node('div');
      title.append(node('h3', scene.label), node('p', `${scene.family} · ${scene.input_kind === 'flat-image' ? 'Flat image: whole-image fit only' : 'Editable design'}`));
      heading.append(title);
      article.append(heading);
      const renders = node('div', undefined, report.job_plan ? 'scene-comparison' : 'scene-renders');
      const original = node('div', undefined, 'render-cell');
      original.append(node('p', 'Original', 'render-label'));
      const sourceStage = node('div', undefined, 'render-stage');
      const sourceReceipt = node('p', 'Bound source receipt appears after rendering', 'field-help fingerprint');
      original.append(sourceStage, sourceReceipt);
      renders.append(original);
      if (report.job_plan) {
        const levels = node('div', undefined, 'scene-levels');
        for (const [level, title, description] of [
          ['resolution', 'Level 1 · Proportional resolution', '50% and 25% of the original canvas, rounded to pixels. Compare the actual composition and engine diagnostics.'],
          ['shared-aspect', 'Level 2 · Shared canvas sizes', 'The same target sizes for every poster. Each result starts from the original, not a Level 1 result.'],
        ]) {
          const section = node('section', undefined, 'resize-level');
          const header = node('header'); header.append(node('h4', title), node('p', description)); section.append(header);
          const grid = node('div', undefined, 'scene-renders');
          for (const plan of report.job_plan.filter(item => item.scene_id === scene.id && item.comparisons.some(use => use.level === level))) appendCell(grid, plan, level);
          section.append(grid); levels.append(section);
        }
        renders.append(levels);
      } else for (const target of report.targets) appendCell(renders, target);
      article.append(renders);
      return {article, scene, sourceStage, sourceReceipt, sourceRow: null, phase: null, cells};
  }
  function renderResults(report) {
    if (state.renderedReport !== report) {
      releaseUrls();
      state.sceneViews.clear();
      state.runSources = new Map(state.scenes.map(scene => [scene.id, scene.payload]));
      state.renderedReport = report;
      state.stale = false;
      $('batchResults').replaceChildren();
      for (const scene of report.scenes) {
        const view = createSceneView(report, scene);
        state.sceneViews.set(scene.id, view);
        $('batchResults').append(view.article);
      }
    }
    for (const view of state.sceneViews.values()) {
      const first = report.results.find(row => row.scene_id === view.scene.id && row.result);
      if (view.sourceRow !== first || (!first && view.phase !== report.status)) {
        if (first) {
          const image = node('img'); image.src = first.result.source_png; image.alt = `${view.scene.label}, original scene`;
          view.sourceStage.replaceChildren(image);
          view.sourceReceipt.textContent = `Source ${first.source_sha256.slice(0, 12)}…`;
        } else view.sourceStage.replaceChildren(node('p', report.status === 'running' ? 'Original render pending' : 'Original render unavailable'));
        view.sourceRow = first; view.phase = report.status;
      }
      for (const binding of view.cells) {
        const row = rowFor(report, view.scene, binding.plan);
        if (binding.row === row && (row || binding.phase === report.status)) continue;
        const updated = renderResultCell(report, view.scene, binding.plan, binding.level);
        binding.cell.replaceChildren(...updated.children);
        binding.row = row; binding.phase = report.status;
      }
    }
    const rows = report.results;
    const succeeded = rows.filter(row => row.status === 'ok').length;
    const passed = rows.filter(row => row.contract_passed === true).length;
    $('batchSummary').hidden = false;
    $('batchSummary').replaceChildren(node('strong', `${rows.length} / ${report.job_plan?.length || report.scenes.length * report.targets.length} jobs completed`), node('span', `${succeeded} rendered`), node('span', `${rows.length - succeeded} failed`), node('span', `${passed} contract passes`), node('span', 'Visual approval: unreviewed'));
    if (report.interrupted_job) $('batchSummary').append(node('span', '1 unverified attempt'));
    const table = node('table'); table.append(node('caption', 'Breakdown by declared source family · structural diagnostics only'));
    const head = node('thead'), header = node('tr');
    for (const text of ['Family', 'Scenes', 'Completed jobs', 'Rendered', 'Contract passes']) header.append(node('th', text));
    head.append(header); table.append(head);
    const body = node('tbody');
    for (const family of [...new Set(report.scenes.map(scene => scene.family))]) {
      const familyScenes = report.scenes.filter(scene => scene.family === family), ids = new Set(familyScenes.map(scene => scene.id));
      const familyRows = rows.filter(row => ids.has(row.scene_id));
      const tr = node('tr');
      for (const value of [family, familyScenes.length, familyRows.length, familyRows.filter(row => row.status === 'ok').length, familyRows.filter(row => row.contract_passed === true).length]) tr.append(node('td', String(value)));
      body.append(tr);
    }
    table.append(body); $('familySummary').replaceChildren(table); $('familySummary').hidden = false;
  }
  function reportDownload() {
    if (state.stale || state.disposed) return;
    state.reportExport ||= blobResource(state.report, `${state.report.run_id}.json`);
    $('downloadReport').href = state.reportExport.url; $('downloadReport').download = state.reportExport.filename;
    $('downloadReport').setAttribute('aria-disabled', 'false'); $('downloadReport').removeAttribute('tabindex');
  }
  async function run(event) {
    event.preventDefault();
    if (state.busy) return;
    error();
    let activeJob = null;
    try {
      const targets = targetsFromText($('targetSizes').value);
      if (!state.scenes.length) throw new Error('Add at least one scene.');
      const mode = comparisonMode(), jobs = jobPlan(state.scenes, targets, mode);
      setBusy(true); state.stop = false;
      const method = await request('/api/benchmark/method');
      setMethod(method);
      state.report = {schema_version: 'scene-benchmark-ui-v1', run_id: `benchmark-${new Date().toISOString().replace(/[:.]/g, '-')}`, started_at: new Date().toISOString(), method, evaluation_use: $('evaluationUse').value, status: 'running', visual_review: 'not_reviewed', scenes: state.scenes.map(({payload, ...scene}) => ({...scene})), targets, results: []};
      const report = state.report;
      if (mode === 'two-level') {
        report.schema_version = 'scene-benchmark-ui-v2'; report.comparison_mode = mode;
        report.job_plan = JSON.parse(JSON.stringify(jobs));
        report.scenes.forEach(scene => { scene.source_canvas = {...state.scenes.find(item => item.id === scene.id).payload.source.canvas}; });
      }
      $('downloadReport').removeAttribute('href'); $('downloadReport').setAttribute('aria-disabled', 'true'); $('downloadReport').tabIndex = -1;
      $('runIdentity').textContent = `${report.run_id} · method ${method.digest.slice(0, 12)}…`;
      $('batchProgress').hidden = false; $('batchProgress').max = jobs.length; $('batchProgress').value = 0;
      $('stopBatch').hidden = false; $('stopBatch').disabled = false;
      renderResults(report);
      for (const job of jobs) {
          if (state.stop) break;
          const scene = state.scenes.find(item => item.id === job.scene_id), target = job.target_canvas;
          $('batchStatus').textContent = `Rendering ${report.results.length + 1} / ${$('batchProgress').max}: ${scene.label} at ${target.width} × ${target.height}.`;
          activeJob = {scene_id: scene.id, target_canvas: {...target}, status: 'unverified', ...(report.job_plan ? {job_key: job.job_key, comparisons: job.comparisons} : {})};
          const response = await request('/api/benchmark/run', {method: method.id, expected_method_digest: method.digest, targets: [target], scenes: [scene]});
          const row = response.results?.[0];
          if (response.method?.digest !== method.digest || response.comparable !== true || response.results?.length !== 1 || row.scene_id !== scene.id || row.target_canvas?.width !== target.width || row.target_canvas?.height !== target.height || !['ok', 'error'].includes(row.status)) throw new Error('Method, scene or target identity changed, or the response was inconsistent. The run stopped.');
          await validateResult(row, scene, target);
          report.results.push(report.job_plan ? {...row, job_key: job.job_key, comparisons: job.comparisons} : row);
          activeJob = null;
          $('batchProgress').value = report.results.length;
          renderResults(report);
      }
      report.status = state.stop ? 'stopped' : 'completed'; report.finished_at = new Date().toISOString();
      renderResults(report); reportDownload();
      $('batchStatus').textContent = state.stop ? `Stopped after ${report.results.length} job${report.results.length === 1 ? '' : 's'}. Completed results are available; unrun jobs are marked.` : `Finished ${report.results.length} job${report.results.length === 1 ? '' : 's'} with ${report.job_plan ? 'the same method and recorded two-level plan' : 'the same method and targets'}. Inspect the renders; visual quality has not been graded.`;
    } catch (failure) {
      error(failure.message);
      if (!state.method || failure.reconnect) $('retryConnection').hidden = false;
      if (state.report?.status === 'running') {
        state.report.status = 'interrupted'; state.report.error = failure.message; state.report.finished_at = new Date().toISOString();
        if (activeJob) state.report.interrupted_job = {...activeJob, error: `Attempt could not be verified: ${failure.message}`};
        renderResults(state.report); reportDownload();
        const count = state.report.results.length;
        $('batchStatus').textContent = `Run interrupted. ${count} completed job${count === 1 ? '' : 's'} preserved. ${activeJob ? 'The current attempt could not be verified; later jobs were not submitted.' : 'No further jobs were submitted.'}`;
      }
    } finally { setBusy(false); $('stopBatch').hidden = true; }
  }
  $('sceneFiles').addEventListener('change', importFiles);
  $('loadReviewed').addEventListener('click', browseReviewed);
  $('addReviewed').addEventListener('click', () => addReviewed());
  $('addReviewedPack').addEventListener('click', () => addReviewed(true));
  $('comparisonMode').addEventListener('change', () => { updateComparisonMode(); dirty(); });
  $('twoLevelPreset').addEventListener('click', () => {
    if (state.busy) return;
    $('comparisonMode').value = 'two-level'; $('targetSizes').value = '800x500\n500x800';
    updateComparisonMode(); dirty();
  });
  $('addSamples').addEventListener('click', async () => {
    if (state.busy) return;
    error(); setBusy(true);
    try {
      const {samples} = await request('/api/samples');
      if (!Array.isArray(samples)) throw new Error('The engine returned an invalid sample catalogue. Your queued scenes are unchanged.');
      const ids = new Set();
      for (const sample of samples) {
        if (!sample || typeof sample.id !== 'string' || !sample.id.trim() || sample.id.length > 120 || /[\x00-\x1f]/.test(sample.id) || ids.has(sample.id) || typeof sample.label !== 'string' || !sample.label.trim() || sample.label.length > 200 || /[\x00-\x1f]/.test(sample.label)) throw new Error('Sample catalogue IDs and labels must be valid and IDs must be unique. Your queued scenes are unchanged.');
        ids.add(sample.id);
      }
      const added = samples.filter(sample => !state.scenes.some(scene => scene.id === sample.id)).slice(0, 3);
      if (state.scenes.length + added.length > 8) throw new Error('Remove scenes before adding the bundled pack (8 scene maximum).');
      const queued = added.map(sample => ({id: sample.id, label: sample.label, family: 'bundled-campaign', input_kind: 'editable-design', payload: sourcePayload(sample.task)}));
      state.scenes.push(...queued);
      dirty(); renderQueue(); $('batchStatus').textContent = `${state.scenes.length} scenes ready. Bundled examples are smoke tests from one campaign, not independent unseen families.`;
    } catch (failure) { error(failure.message); } finally { setBusy(false); }
  });
  $('batchForm').addEventListener('submit', run);
  $('retryConnection').addEventListener('click', reconnect);
  $('targetSizes').addEventListener('input', dirty); $('evaluationUse').addEventListener('change', dirty);
  $('stopBatch').addEventListener('click', () => { state.stop = true; $('stopBatch').disabled = true; $('batchStatus').textContent = 'Stopping after the current local job finishes. Completed results will be kept.'; });
  let savedTheme;
  try { savedTheme = localStorage.getItem('chameleon.theme'); } catch { /* Preferences are optional. */ }
  $('theme').value = ['light', 'dark'].includes(savedTheme) ? savedTheme : 'dark';
  document.documentElement.dataset.theme = $('theme').value;
  $('theme').addEventListener('change', () => {
    document.documentElement.dataset.theme = $('theme').value;
    try { localStorage.setItem('chameleon.theme', $('theme').value); } catch { /* Keep the active theme without persistence. */ }
  });
  window.addEventListener('pagehide', event => {
    state.stop = true;
    if (!event.persisted) { state.disposed = true; releaseUrls(); }
  });
  request('/api/benchmark/method').then(method => {
    setMethod(method); dirty();
  }).catch(failure => { error(failure.message); $('methodName').textContent = 'Method unavailable'; $('retryConnection').hidden = false; });
  updateComparisonMode();
})();
