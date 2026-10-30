/* Local scene matrix. Every job uses one frozen method digest and target list. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {scenes: [], method: null, report: null, busy: false, stop: false, urls: []};
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
  function setMethod(method) {
    if (method?.id !== 'deterministic-current' || !/^[a-f0-9]{64}$/.test(method?.digest || '')) throw Object.assign(new Error('The engine returned an invalid method identity. Reconnect before testing.'), {reconnect: true});
    if (state.report && state.report.method.digest !== method.digest) dirty();
    state.method = method;
    $('methodName').textContent = method.id;
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
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      let response;
      try {
        response = await fetch(path, {signal: controller.signal, ...(body ? {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
        } : {})});
      } catch (failure) {
        if (failure.name === 'AbortError') throw new Error('Job timed out. The local engine may still be finishing it. No further jobs were submitted.');
        connection(false);
        throw Object.assign(new Error('Cannot reach the local engine. Start the server, then reconnect.'), {reconnect: true});
      }
      connection(true);
      let result;
      try { result = await response.json(); }
      catch (failure) { throw new Error(failure.name === 'AbortError' ? 'Job timed out while receiving data. No further jobs were submitted.' : 'The engine returned invalid JSON. Completed jobs are preserved; retry the run.'); }
      if (!response.ok) throw Object.assign(new Error(result?.error || `Request failed (${response.status}).`), {reconnect: response.status >= 500 || response.status === 409});
      return result;
    } finally { clearTimeout(timeout); }
  }
  function releaseUrls() {
    for (const url of state.urls) URL.revokeObjectURL(url);
    state.urls = [];
  }
  function blobLink(text, data, filename, type = 'application/json') {
    const url = URL.createObjectURL(new Blob([type === 'application/json' ? JSON.stringify(data, null, 2) : data], {type}));
    state.urls.push(url);
    const link = node('a', text);
    link.href = url;
    link.download = filename;
    return link;
  }
  function dirty() {
    $('downloadReport').removeAttribute('href');
    $('downloadReport').setAttribute('aria-disabled', 'true');
    $('downloadReport').tabIndex = -1;
    if (state.report) {
      $('runIdentity').textContent = `${state.report.run_id} · previous run; setup changed`;
      $('batchStatus').textContent = 'Setup changed. Displayed results belong to the previous run; run again to update.';
      for (const link of $('batchResults').querySelectorAll('a[download]')) {
        link.removeAttribute('href');
        link.setAttribute('aria-disabled', 'true');
        link.tabIndex = -1;
      }
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
  function rowFor(report, scene, target) {
    const matches = row => row?.scene_id === scene.id && row.target_canvas.width === target.width && row.target_canvas.height === target.height;
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
  function renderResults(report) {
    releaseUrls();
    $('batchResults').replaceChildren();
    for (const scene of report.scenes) {
      const article = node('article', undefined, 'scene-result');
      const heading = node('header');
      const title = node('div');
      title.append(node('h3', scene.label), node('p', `${scene.family} · ${scene.input_kind === 'flat-image' ? 'Flat image: whole-image fit only' : 'Editable design'}`));
      heading.append(title);
      article.append(heading);
      const renders = node('div', undefined, 'scene-renders');
      const first = report.results.find(row => row.scene_id === scene.id && row.result);
      const original = node('div', undefined, 'render-cell');
      original.append(node('p', 'Original', 'render-label'));
      const sourceStage = node('div', undefined, 'render-stage');
      if (first) { const image = node('img'); image.src = first.result.source_png; image.alt = `${scene.label}, original scene`; sourceStage.append(image); }
      else sourceStage.append(node('p', 'Original render pending'));
      original.append(sourceStage, node('p', first ? `Source ${first.source_sha256.slice(0, 12)}…` : 'Bound source receipt appears after rendering', 'field-help fingerprint'));
      renders.append(original);
      for (const target of report.targets) {
        const row = rowFor(report, scene, target);
        const cell = node('div', undefined, 'render-cell');
        cell.append(node('p', `${target.width} × ${target.height}`, 'render-label'));
        const stage = node('div', undefined, 'render-stage');
        if (row?.result) {
          const image = node('img'); image.src = row.result.output_png; image.alt = `${scene.label}, result at ${target.width} by ${target.height}`; stage.append(image);
          cell.append(stage);
          const diagnostic = node('p', `${row.contract_passed ? 'Contract checks passed' : 'Contract needs review'} · ${(row.elapsed_ms / 1000).toFixed(2)} s · visual quality unreviewed`, 'render-notes');
          diagnostic.dataset.outcome = row.contract_passed ? 'pass' : 'review';
          cell.append(diagnostic);
          const links = node('div', undefined, 'result-links');
          const basename = `${scene.id}-${target.width}x${target.height}`;
          const sourcePayload = state.scenes.find(item => item.id === scene.id)?.payload;
          const portable = sourcePayload && ['document', 'assets', 'fonts'].some(key => key in sourcePayload);
          const exported = portable ? {
            ...sourcePayload, source: row.result.design, target_canvas: {...target},
          } : row.result.design;
          links.append(blobLink('JSON ↓', exported, `${basename}.json`));
          const binary = atob(row.result.output_png.split(',')[1]);
          const pngBytes = Uint8Array.from(binary, character => character.charCodeAt(0));
          links.append(blobLink('PNG ↓', pngBytes, `${basename}.png`, 'image/png'));
          cell.append(links);
          const details = node('details'); details.append(node('summary', 'Diagnostics and resource identity'));
          details.append(node('pre', JSON.stringify({source_sha256: row.source_sha256, resources_sha256: row.resources_sha256, report: row.result.report, warnings: row.result.warnings}, null, 2)));
          cell.append(details);
        } else {
          stage.append(node('p', row?.error || (report.status === 'running' ? 'Waiting for this job' : 'Not run'))); cell.append(stage);
        }
        renders.append(cell);
      }
      article.append(renders);
      $('batchResults').append(article);
    }
    const rows = report.results;
    const succeeded = rows.filter(row => row.status === 'ok').length;
    const passed = rows.filter(row => row.contract_passed === true).length;
    $('batchSummary').hidden = false;
    $('batchSummary').replaceChildren(node('strong', `${rows.length} / ${report.scenes.length * report.targets.length} jobs completed`), node('span', `${succeeded} rendered`), node('span', `${rows.length - succeeded} failed`), node('span', `${passed} contract passes`), node('span', 'Visual approval: unreviewed'));
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
    const link = blobLink('Download run report ↓', state.report, `${state.report.run_id}.json`);
    $('downloadReport').href = link.href; $('downloadReport').download = link.download;
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
      if (state.scenes.length * targets.length > 16) throw new Error('A benchmark is limited to 16 jobs. Reduce scenes or targets.');
      setBusy(true); state.stop = false;
      const method = await request('/api/benchmark/method');
      setMethod(method);
      state.report = {schema_version: 'scene-benchmark-ui-v1', run_id: `benchmark-${new Date().toISOString().replace(/[:.]/g, '-')}`, started_at: new Date().toISOString(), method, evaluation_use: $('evaluationUse').value, status: 'running', visual_review: 'not_reviewed', scenes: state.scenes.map(({payload, ...scene}) => ({...scene})), targets, results: []};
      const report = state.report;
      $('downloadReport').removeAttribute('href'); $('downloadReport').setAttribute('aria-disabled', 'true'); $('downloadReport').tabIndex = -1;
      $('runIdentity').textContent = `${report.run_id} · method ${method.digest.slice(0, 12)}…`;
      $('batchProgress').hidden = false; $('batchProgress').max = state.scenes.length * targets.length; $('batchProgress').value = 0;
      $('stopBatch').hidden = false; $('stopBatch').disabled = false;
      renderResults(report);
      outer: for (const scene of state.scenes) {
        for (const target of targets) {
          if (state.stop) break outer;
          $('batchStatus').textContent = `Rendering ${report.results.length + 1} / ${$('batchProgress').max}: ${scene.label} at ${target.width} × ${target.height}.`;
          activeJob = {scene_id: scene.id, target_canvas: {...target}, status: 'unverified'};
          const response = await request('/api/benchmark/run', {method: method.id, expected_method_digest: method.digest, targets: [target], scenes: [scene]});
          const row = response.results?.[0];
          if (response.method?.digest !== method.digest || response.comparable !== true || response.results?.length !== 1 || row.scene_id !== scene.id || row.target_canvas?.width !== target.width || row.target_canvas?.height !== target.height || !['ok', 'error'].includes(row.status)) throw new Error('Method, scene or target identity changed, or the response was inconsistent. The run stopped.');
          await validateResult(row, scene, target);
          report.results.push(row);
          activeJob = null;
          $('batchProgress').value = report.results.length;
          renderResults(report);
        }
      }
      report.status = state.stop ? 'stopped' : 'completed'; report.finished_at = new Date().toISOString();
      renderResults(report); reportDownload();
      $('batchStatus').textContent = state.stop ? `Stopped after ${report.results.length} job${report.results.length === 1 ? '' : 's'}. Completed results are available; unrun jobs are marked.` : `Finished ${report.results.length} job${report.results.length === 1 ? '' : 's'} with the same method and targets. Inspect the renders; visual quality has not been graded.`;
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
    if (!event.persisted) releaseUrls();
  });
  request('/api/benchmark/method').then(method => {
    setMethod(method); dirty();
  }).catch(failure => { error(failure.message); $('methodName').textContent = 'Method unavailable'; $('retryConnection').hidden = false; });
})();
