(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const colors = {text: '#86baff', image: '#e9bc73', shape: '#7dcca4'};
  let connectionRequest = 0, latestConnectionRequest = 0;
  const state = {samples: [], posters: [], targets: [], imported: null, task: null, result: null, selected: null, images: {}, jsonUrl: null, pngUrl: null, busy: false, initializing: false, dirty: true, version: 0, loadVersion: 0, controller: null};
  const hasResources = task => task && Array.isArray(task.assets) && Array.isArray(task.fonts)
    && task.document && typeof task.document === 'object' && !Array.isArray(task.document);
  const elementLabel = el => el.type === 'text' ? el.content.replace(/\s+/g, ' ').trim().slice(0, 64) || el.id : el.name || el.id;
  const webLink = value => {
    try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) && url.hostname ? url.href : null; }
    catch { return null; }
  };

  function errorMessage(error) {
    $('error').textContent = error.message || String(error);
    $('error').hidden = false;
  }
  function setConnection(connected) {
    $('connection').textContent = connected ? 'Engine connected' : 'Engine unavailable';
    $('connection').classList.toggle('online', connected);
  }
  function disableDownloads() {
    for (const id of ['downloadJson', 'downloadPng']) {
      $(id).removeAttribute('href');
      $(id).setAttribute('aria-disabled', 'true');
      $(id).setAttribute('tabindex', '-1');
    }
    if (state.jsonUrl) URL.revokeObjectURL(state.jsonUrl);
    if (state.pngUrl) URL.revokeObjectURL(state.pngUrl);
    state.jsonUrl = null;
    state.pngUrl = null;
  }
  function markDirty() {
    state.version++;
    cancelRequest();
    state.dirty = true;
    disableDownloads();
    $('resultSummary').hidden = $('warnings').hidden = $('reviewPanel').hidden = true;
    if (state.result) {
      const canvas = state.result.design.canvas;
      $('outputSize').textContent = `${canvas.width} × ${canvas.height} · previous result`;
    }
    $('status').textContent = 'Canvas changed. Resize the design to update the preview.';
  }
  function setBusy(value) {
    state.busy = value;
    $('settings').disabled = value;
    $('sample').disabled = value || (!state.samples.length && !state.posters.length && !state.imported);
    $('previews').setAttribute('aria-busy', String(value));
    $('runLabel').textContent = value ? 'Resizing design…' : state.task ? 'Resize design' : 'Retry connection';
    $('run').setAttribute('aria-busy', String(value));
  }
  function cancelRequest() {
    state.controller?.abort();
    state.controller = null;
    setBusy(false);
  }
  function beginRequest() {
    cancelRequest();
    const controller = new AbortController();
    state.controller = controller;
    setBusy(true);
    return controller;
  }
  function finishRequest(controller) {
    if (state.controller !== controller) return;
    state.controller = null;
    controller.abort();
    setBusy(false);
  }
  function validateCanvas(canvas) {
    return canvas && Number.isInteger(canvas.width) && Number.isInteger(canvas.height)
      && canvas.width >= 1 && canvas.height >= 1 && canvas.width <= 8192 && canvas.height <= 8192
      && canvas.width * canvas.height <= 20000000;
  }
  function gentleTarget(canvas) {
    const portrait = canvas.height > canvas.width, landscape = canvas.width > canvas.height;
    return {
      width: Math.max(1, Math.round(canvas.width * (portrait ? 0.9 : 0.95))),
      height: Math.max(1, Math.round(canvas.height * (landscape ? 0.9 : 0.95))),
    };
  }
  function validateRequestPolicy(policy) {
    if (!policy || typeof policy !== 'object' || Array.isArray(policy)
      || !['studio_timeout_ms', 'benchmark_timeout_ms'].every(key => Number.isInteger(policy[key]) && policy[key] > 0 && policy[key] <= 2100000)
      || typeof policy.model_configured !== 'boolean') throw new Error('The engine returned an invalid resize runtime policy. Restart or update the local server, then retry.');
    return policy;
  }
  function validateDesign(design) {
    if (!design || !design.canvas || !Array.isArray(design.elements) || !design.elements.length) throw new Error('A design needs a canvas and a nonempty elements array.');
    if (!validateCanvas(design.canvas)) throw new Error('Canvas dimensions must be whole numbers from 1 to 8192 px and at most 20 million pixels.');
    if (design.elements.length > 200) throw new Error('A design supports at most 200 elements.');
    const ids = new Set();
    for (const element of design.elements) {
      if (!element || typeof element.id !== 'string' || !element.id || ids.has(element.id)) throw new Error('Every element needs a unique, nonempty ID.');
      ids.add(element.id);
      if (!['text', 'image', 'shape'].includes(element.type)) throw new Error(`Element ${element.id} has an unsupported type.`);
      if (!['x', 'y', 'width', 'height'].every(key => Number.isFinite(element[key])) || element.width <= 0 || element.height <= 0) throw new Error(`Element ${element.id} needs finite geometry and positive dimensions.`);
      if (element.type === 'text' && (typeof element.content !== 'string' || !Number.isFinite(element.font_size) || element.font_size <= 0)) throw new Error(`Text element ${element.id} needs content and a positive font size.`);
    }
  }
  function validateSemantics(value, source) {
    if (!value || typeof value !== 'object' || Array.isArray(value)
      || value.source_canvas?.width !== source.canvas.width || value.source_canvas?.height !== source.canvas.height
      || !Array.isArray(value.elements) || value.elements.length !== source.elements.length) return null;
    const ids = new Set(source.elements.map(element => element.id)), seen = new Set();
    for (const row of value.elements) {
      if (!row || !ids.has(row.element_id) || seen.has(row.element_id)
        || !['purpose', 'design_purpose', 'provenance'].every(key => typeof row[key] === 'string')
        || (row.rigid_group !== null && typeof row.rigid_group !== 'string')
        || !Array.isArray(row.anchors) || !row.anchors.every(anchor => anchor && Array.isArray(anchor.edges)
          && anchor.edges.every(edge => ['top', 'right', 'bottom', 'left'].includes(edge))
          && (anchor.horizontal === undefined || ['left', 'center', 'right'].includes(anchor.horizontal)))
        || !Array.isArray(row.relationships) || !row.relationships.every(relation => relation
          && ids.has(relation.source_id) && ids.has(relation.target_id)
          && [relation.source_id, relation.target_id].includes(row.element_id)
          && typeof relation.type === 'string'
          && (relation.origin === undefined || typeof relation.origin === 'string'))) return null;
      seen.add(row.element_id);
    }
    return value;
  }
  function validateResult(value, source, target) {
    if (!value || !value.report || !Array.isArray(value.report.violations) || value.report.violations.some(message => typeof message !== 'string')) throw new Error('The engine returned an incomplete result report. Retry the resize.');
    validateDesign(value.design);
    if (value.design.canvas.width !== target.width || value.design.canvas.height !== target.height) throw new Error('The engine returned a different target canvas. Retry the resize.');
    const ids = new Set(source.elements.map(element => element.id));
    if (value.design.elements.length !== ids.size || value.design.elements.some(element => !ids.has(element.id))) throw new Error('The engine returned elements from a different source. Retry the resize.');
    if (!Number.isFinite(value.elapsed_ms) || value.elapsed_ms < 0) throw new Error('The engine returned invalid timing information. Retry the resize.');
    for (const image of [value.source_png, value.output_png]) {
      if (typeof image !== 'string' || !/^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(image)) throw new Error('The engine returned invalid PNG preview data. Retry the resize.');
      try { atob(image.split(',')[1]); }
      catch { throw new Error('The engine returned invalid PNG preview data. Retry the resize.'); }
    }
    if (value.warnings !== undefined && (!Array.isArray(value.warnings) || value.warnings.some(message => typeof message !== 'string'))) throw new Error('The engine returned invalid warnings. Retry the resize.');
    const report = {...value.report};
    const semantics = validateSemantics(report.design_semantics, source);
    if (semantics) report.design_semantics = semantics;
    else delete report.design_semantics;
    for (const key of ['style', 'relationships', 'proportions', 'focal']) {
      if (report[key] == null) continue;
      if (typeof report[key] !== 'object' || Array.isArray(report[key]) || (report[key].warnings !== undefined && (!Array.isArray(report[key].warnings) || report[key].warnings.some(warning => !warning || typeof warning.message !== 'string')))) throw new Error('The engine returned invalid review diagnostics. Retry the resize.');
      report[key] = {...report[key], warnings: report[key].warnings || []};
    }
    for (const key of ['relations', 'ambiguities']) {
      if (report.relationships?.[key] !== undefined && !Array.isArray(report.relationships[key])) throw new Error('The engine returned invalid relationship diagnostics. Retry the resize.');
    }
    return {...value, report};
  }
  function setTask(task, label, {preserveTarget = false} = {}) {
    validateDesign(task?.source || task);
    if (['assets', 'fonts', 'document'].some(key => key in task) && !hasResources(task)) throw new Error('A portable bundle needs its document, assets array and fonts array.');
    const next = structuredClone(task.source ? task : {source: task});
    const currentTarget = {width: Number($('width').value), height: Number($('height').value)};
    if (preserveTarget && validateCanvas(currentTarget)) next.target_canvas = currentTarget;
    next.target_canvas ||= {width: Number($('width').value) || next.source.canvas.width, height: Number($('height').value) || next.source.canvas.height};
    if (!validateCanvas(next.target_canvas)) throw new Error('The target canvas must be 1–8192 px per side and at most 20 million pixels.');
    cancelRequest();
    state.task = next;
    state.version++;
    state.result = null;
    state.images = {};
    state.selected = null;
    resetInspector();
    state.dirty = true;
    disableDownloads();
    const source = state.task.source;
    const target = state.task.target_canvas || source.canvas;
    $('width').value = target.width;
    $('height').value = target.height;
    syncPreset();
    $('sourceSummary').textContent = `${source.canvas.width} × ${source.canvas.height} px · ${source.elements.length} elements`;
    $('taskName').textContent = label || state.task.task_id || 'Imported design';
    const editable = {...state.task};
    for (const key of ['assets', 'fonts', 'document']) delete editable[key];
    $('taskJson').value = JSON.stringify(editable, null, 2);
    const poster = state.posters.find(item => item.id === state.task.poster_id) || state.task.attribution;
    $('posterCredit').replaceChildren();
    $('posterCredit').hidden = !poster;
    if (poster) {
      const credit = document.createElement('span');
      credit.textContent = `${poster.attribution_text || poster.text || poster.author || 'Imported artwork'} · ${poster.license_id || 'See source license'}. `;
      $('posterCredit').append(credit);
      if (typeof poster.original_url === 'string' && poster.original_url.startsWith('/api/posters/')) {
        const original = document.createElement('a');
        original.href = poster.original_url; original.textContent = 'Editable original';
        $('posterCredit').append(original);
      }
      if (webLink(poster.source_url)) {
        const source = document.createElement('a'); source.href = webLink(poster.source_url);
        source.textContent = ' · Source and attribution'; source.target = '_blank'; source.rel = 'noopener';
        $('posterCredit').append(source);
      }
      if (poster.source_note) {
        const note = document.createElement('p'); note.textContent = poster.source_note;
        $('posterCredit').append(note);
      }
    }
    $('sourceSize').textContent = `${source.canvas.width} × ${source.canvas.height}`;
    $('outputSize').textContent = 'Not resized yet';
    $('sourceCanvas').hidden = $('outputCanvas').hidden = true;
    $('sourceEmpty').hidden = $('outputEmpty').hidden = false;
    $('inspection').hidden = $('resultSummary').hidden = $('warnings').hidden = $('reviewPanel').hidden = true;
    $('error').hidden = true;
    $('status').textContent = 'Design loaded. Choose a format and resize.';
    $('run').disabled = false;
    $('run').formNoValidate = false;
  }
  function syncPreset() {
    const near = state.task && gentleTarget(state.task.source.canvas);
    if (near && near.width === Number($('width').value) && near.height === Number($('height').value)) {
      $('preset').value = 'gentle'; return;
    }
    const match = state.targets.find(t => t.width === Number($('width').value) && t.height === Number($('height').value));
    $('preset').value = match ? match.id : 'custom';
  }
  function syncSourceURL(kind, id) {
    const url = new URL(location.href);
    url.searchParams.delete('poster');
    url.searchParams.delete('sample');
    if (kind && id) url.searchParams.set(kind, id);
    history.replaceState(null, '', url);
  }
  function importTask(task, label) {
    setTask(task, label);
    state.loadVersion++;
    state.imported = {task: structuredClone(state.task), label};
    if (![...$('sample').options].some(option => option.value === 'imported')) $('sample').add(new Option('Imported design', 'imported'));
    $('sample').value = 'imported';
    syncSourceURL();
    resize();
  }
  async function loadPoster(id, {preserveTarget = Boolean(state.task) && $('preset').value !== 'gentle'} = {}) {
    const poster = state.posters.find(item => item.id === id);
    if (!poster?.ready) { errorMessage(new Error('This poster needs import support. See the poster library for its notes.')); return; }
    const loadVersion = ++state.loadVersion;
    markDirty(); $('error').hidden = true;
    const controller = beginRequest();
    const timeout = setTimeout(() => controller.abort(), 60000);
    $('status').textContent = 'Loading poster images and fonts…';
    let loaded = false;
    try {
      const bundle = await requestJSON(poster.bundle_url, {signal: controller.signal});
      if (loadVersion !== state.loadVersion || state.controller !== controller) return;
      bundle.poster_id = id; bundle.task_id = id;
      bundle.attribution ||= {text: poster.attribution_text || poster.author, author: poster.author,
        source_url: poster.source_url, license_id: poster.license_id, license_url: poster.license_url,
        original_sha256: poster.original_sha256, source_note: poster.source_note};
      if (!preserveTarget) bundle.target_canvas = gentleTarget(bundle.source.canvas);
      setTask(bundle, poster.title, {preserveTarget}); syncSourceURL('poster', id); loaded = true;
    } catch (error) {
      if (loadVersion === state.loadVersion && state.controller === controller) {
        errorMessage(error.name === 'AbortError' ? new Error('Poster loading exceeded 60 seconds. Try again.') : error);
        $('status').textContent = 'Poster loading failed. Choose a design or try again.';
      }
    }
    finally { clearTimeout(timeout); finishRequest(controller); }
    if (loaded && loadVersion === state.loadVersion) await resize();
  }
  async function requestJSON(url, options = {}) {
    const request = ++connectionRequest;
    const updateConnection = connected => {
      if (request < latestConnectionRequest || options.signal?.aborted) return;
      latestConnectionRequest = request;
      setConnection(connected);
    };
    let response;
    try { response = await fetch(url, options); }
    catch (error) {
      if (error.name === 'AbortError' || options.signal?.aborted) throw error;
      updateConnection(false);
      throw new Error('Cannot reach the local engine. Start python -m viewer.server, then retry your action. Your loaded design is preserved.');
    }
    // An HTTP error still proves the engine is reachable. Request validation
    // and rendering errors belong in the error message, not an offline badge.
    updateConnection(true);
    let data;
    try { data = await response.json(); }
    catch (error) { if (error.name === 'AbortError') throw error; throw new Error(`The engine returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }
  const loadImage = (url, signal) => new Promise((resolve, reject) => {
    const image = new Image();
    const cleanup = () => signal?.removeEventListener('abort', abort);
    const abort = () => {
      cleanup(); image.onload = image.onerror = null; image.src = '';
      reject(new DOMException('Preview loading was cancelled.', 'AbortError'));
    };
    image.onload = () => { cleanup(); resolve(image); };
    image.onerror = () => { cleanup(); reject(new Error('The engine returned an invalid preview image.')); };
    if (signal?.aborted) { abort(); return; }
    signal?.addEventListener('abort', abort, {once: true});
    image.src = url;
  });

  function drawCanvas(id, design, image) {
    const canvas = $(id);
    const {width, height} = design.canvas;
    const scale = Math.min(1, 1600 / Math.max(width, height));
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.drawImage(image, 0, 0, width, height);
    const show = $('showBoxes').checked;
    for (const el of design.elements) {
      if (!show && el.id !== state.selected) continue;
      ctx.save();
      ctx.translate(el.x + el.width / 2, el.y + el.height / 2);
      ctx.rotate((el.rotation || 0) * Math.PI / 180);
      ctx.strokeStyle = colors[el.type] || '#555';
      ctx.lineWidth = (el.id === state.selected ? 3 : 1.5) / scale;
      ctx.strokeRect(-el.width / 2, -el.height / 2, el.width, el.height);
      ctx.restore();
    }
    canvas.hidden = false;
    canvas.onclick = event => {
      const rect = canvas.getBoundingClientRect();
      const px = (event.clientX - rect.left) * width / rect.width;
      const py = (event.clientY - rect.top) * height / rect.height;
      const elements = [...design.elements].reverse().sort((a, b) => (b.z_index || 0) - (a.z_index || 0));
      const hit = elements.find(el => {
        const angle = -(el.rotation || 0) * Math.PI / 180;
        const dx = px - el.x - el.width / 2, dy = py - el.y - el.height / 2;
        return Math.abs(dx * Math.cos(angle) - dy * Math.sin(angle)) <= el.width / 2 && Math.abs(dx * Math.sin(angle) + dy * Math.cos(angle)) <= el.height / 2;
      });
      if (hit) selectElement(hit.id, id === 'sourceCanvas' ? 'source' : 'result');
    };
  }
  function redraw() {
    if (!state.result) return;
    drawCanvas('sourceCanvas', state.task.source, state.images.source);
    drawCanvas('outputCanvas', state.result.design, state.images.output);
  }
  function selectElement(id, view = $('previews').dataset.view === 'source' ? 'source' : 'result') {
    if (!state.result) return;
    const design = view === 'source' ? state.task.source : state.result.design;
    const el = design.elements.find(item => item.id === id);
    if (!el) return;
    state.selected = id;
    $('propertyTitle').textContent = elementLabel(el);
    $('propertyType').textContent = `${el.type} · ${view === 'source' ? 'Original' : 'Result'}`;
    $('properties').textContent = JSON.stringify(el, null, 2);
    renderSemanticProperties(id);
    $('propertyHint').hidden = true;
    $('geometry').hidden = false;
    $('propertyPanel').open = true;
    for (const [field, id] of [['x', 'propX'], ['y', 'propY'], ['width', 'propWidth'], ['height', 'propHeight']]) $(id).textContent = `${Number(el[field].toFixed(1))} px`;
    for (const button of $('elements').children) button.setAttribute('aria-pressed', String(button.dataset.id === id));
    redraw();
  }
  function renderSemanticProperties(id) {
    const row = state.result?.report.design_semantics?.elements.find(element => element.element_id === id);
    const fields = $('semanticFields');
    fields.replaceChildren();
    $('designSemantics').hidden = !row;
    if (!row) return;
    const words = value => value.replace(/_/g, ' ');
    const field = (label, value) => {
      const item = document.createElement('div'), term = document.createElement('dt'), detail = document.createElement('dd');
      term.textContent = label;
      if (typeof value === 'string') detail.textContent = value;
      else detail.append(value);
      item.append(term, detail); fields.append(item);
    };
    field('Purpose', row.design_purpose);
    field('Role', words(row.purpose));
    field('Evidence', words(row.provenance));
    field('Source anchors', row.anchors.length ? row.anchors.map(anchor =>
      `${anchor.edges.join(' + ')}${anchor.horizontal ? ` · ${anchor.horizontal} aligned` : ''}`
    ).join('; ') : 'None inferred');
    field('Authored group', row.rigid_group || 'None declared');
    if (row.relationships.length) {
      const list = document.createElement('ul');
      for (const relation of row.relationships) {
        const item = document.createElement('li');
        item.textContent = `${words(relation.type)}: ${relation.source_id} → ${relation.target_id}${relation.origin ? ` · ${words(relation.origin)}` : ''}`;
        list.append(item);
      }
      field('Connections', list);
    } else field('Connections', 'None recorded');
  }
  function renderElements() {
    resetInspector();
    const list = $('elements');
    list.replaceChildren();
    const elements = state.result.design.elements;
    $('elementCount').textContent = `${elements.length} preserved`;
    for (const el of elements) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'element-row'; button.dataset.id = el.id; button.setAttribute('aria-pressed', 'false');
      const dot = document.createElement('span'); dot.className = 'dot'; dot.style.background = colors[el.type] || '#555'; dot.setAttribute('aria-hidden', 'true');
      const name = document.createElement('span'); name.className = 'element-id'; name.textContent = elementLabel(el); name.title = el.id;
      const kind = document.createElement('span'); kind.className = 'element-kind'; kind.textContent = el.type;
      button.append(dot, name, kind); button.addEventListener('click', () => selectElement(el.id)); list.append(button);
    }
    $('inspection').hidden = false;
  }
  function resetInspector() {
    $('propertyTitle').textContent = 'Element properties';
    $('propertyType').textContent = '';
    $('properties').textContent = '';
    $('propertyHint').hidden = false;
    $('geometry').hidden = true;
    $('designSemantics').hidden = true;
    $('semanticFields').replaceChildren();
  }
  function enableDownloads() {
    const downloadable = hasResources(state.task) ? {...state.task, source: state.result.design, target_canvas: state.result.design.canvas} : state.result.design;
    state.jsonUrl = URL.createObjectURL(new Blob([JSON.stringify(downloadable, null, 2)], {type: 'application/json'}));
    const pngBytes = Uint8Array.from(atob(state.result.output_png.split(',')[1]), char => char.charCodeAt(0));
    state.pngUrl = URL.createObjectURL(new Blob([pngBytes], {type: 'image/png'}));
    const name = String(state.task.task_id || 'chameleon').replace(/[^a-zA-Z0-9_-]/g, '_');
    const {width, height} = state.result.design.canvas;
    for (const [id, href, ext] of [['downloadJson', state.jsonUrl, 'json'], ['downloadPng', state.pngUrl, 'png']]) {
      $(id).href = href; $(id).download = `${name}_${width}x${height}.${ext}`; $(id).setAttribute('aria-disabled', 'false'); $(id).removeAttribute('tabindex');
    }
  }
  async function resize(event) {
    if (event) event.preventDefault();
    if (state.busy) return;
    const requestedPoster = $('sample').value.startsWith('poster:') ? $('sample').value.slice(7) : null;
    if (requestedPoster && state.task?.poster_id !== requestedPoster) { await loadPoster(requestedPoster); return; }
    if (!state.task) { await init(); return; }
    if (!$('resizeForm').reportValidity()) return;
    const width = Number($('width').value), height = Number($('height').value);
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 8192 || height > 8192 || width * height > 20000000) {
      errorMessage(new Error('Use whole canvas dimensions from 1 to 8192 px and no more than 20 million pixels.')); return;
    }
    const version = state.version;
    $('error').hidden = true;
    disableDownloads();
    const controller = beginRequest();
    $('resultSummary').hidden = $('warnings').hidden = $('reviewPanel').hidden = true;
    $('status').textContent = 'Checking the local resize runtime policy…';
    let phase = 'policy', budget = 60000;
    let timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const portable = hasResources(state.task);
      const payload = portable ? {...state.task, target_canvas: {width, height}} : {source: state.task.source, target_canvas: {width, height}};
      const policy = validateRequestPolicy(await requestJSON('/api/resize/policy', {signal: controller.signal, cache: 'no-store'}));
      if (version !== state.version || state.controller !== controller) return;
      if (controller.signal.aborted) throw new DOMException('Policy request expired.', 'AbortError');
      clearTimeout(timeout);
      phase = 'resize'; budget = policy.studio_timeout_ms;
      timeout = setTimeout(() => controller.abort(), budget);
      $('status').textContent = policy.model_configured ? 'Running the configured local model pipeline and rendering real previews…' : 'Reflowing elements and rendering real previews…';
      const response = await requestJSON(portable ? '/api/document/resize' : '/api/resize', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload), signal: controller.signal});
      if (version !== state.version || state.controller !== controller) return;
      if (controller.signal.aborted) throw new DOMException('Resize request expired.', 'AbortError');
      const result = validateResult(response, payload.source, {width, height});
      const [source, output] = await Promise.all([loadImage(result.source_png, controller.signal), loadImage(result.output_png, controller.signal)]);
      if (version !== state.version || state.controller !== controller) return;
      if (controller.signal.aborted) throw new DOMException('Preview request expired.', 'AbortError');
      if (source.naturalWidth !== payload.source.canvas.width || source.naturalHeight !== payload.source.canvas.height || output.naturalWidth !== width || output.naturalHeight !== height) throw new Error('The preview image dimensions do not match the source and target canvases. Retry the resize.');
      state.result = result; state.images = {source, output}; state.dirty = false; state.selected = null;
      redraw();
      $('sourceEmpty').hidden = $('outputEmpty').hidden = true;
      $('outputSize').textContent = `${width} × ${height}`;
      $('status').textContent = 'Resize complete. Your new composition is ready to review.';
      const style = result.report.style;
      const relationships = result.report.relationships;
      const proportions = result.report.proportions;
      const focal = result.report.focal;
      const uncertain = Boolean(relationships?.ambiguities?.length);
      const review = Boolean(style?.warnings.length || relationships?.warnings.length || proportions?.warnings.length || focal?.warnings.length || uncertain);
      const structure = result.report.violations.length ? 'Structure needs attention' : 'Structure checks passed';
      const composition = uncertain ? 'Composition inference uncertain' : review ? 'Composition needs review' : relationships?.relations.length ? 'No modeled relationship losses' : 'Composition inference limited';
      $('checkSummary').textContent = `${structure} · ${composition}`;
      $('checkSummary').classList.toggle('needs-review', review);
      $('timing').textContent = `${(result.elapsed_ms / 1000).toFixed(2)} s`;
      const texts = result.design.elements.filter(el => el.type === 'text');
      const minFont = texts.length ? Math.min(...texts.map(el => el.font_size)) : null;
      $('fontSummary').textContent = minFont === null ? 'No text elements' : `Smallest text ${minFont.toFixed(1)} px`;
      $('resultSummary').hidden = false;
      const warnings = [...(result.warnings || []), ...result.report.violations, ...(style?.warnings || []).map(warning => `Style: ${warning.message}`), ...(relationships?.warnings || []).map(warning => `Relationship: ${warning.message}`)];
      warnings.push(...(proportions?.warnings || []).map(warning => `Proportions: ${warning.message}`));
      warnings.push(...(focal?.warnings || []).map(warning => `Focal point: ${warning.message}`));
      if (uncertain) warnings.push('Some product/message associations are ambiguous in the source. Review the resized connections.');
      if (minFont !== null && minFont < 10) warnings.push('Some text is smaller than 10 px at this canvas size. Review readability or choose a larger format.');
      $('warnings').replaceChildren();
      const uniqueWarnings = [...new Set(warnings)];
      for (const warning of uniqueWarnings) { const li = document.createElement('li'); li.textContent = warning; $('warnings').append(li); }
      $('warnings').hidden = $('reviewPanel').hidden = !uniqueWarnings.length;
      $('warningCount').textContent = `${uniqueWarnings.length} ${uniqueWarnings.length === 1 ? 'note' : 'notes'}`;
      $('reviewPanel').open = false;
      renderElements(); enableDownloads();
    } catch (error) {
      if (version !== state.version || state.controller !== controller) return;
      const timeoutMessage = phase === 'policy'
        ? 'Resize policy lookup exceeded 10 seconds. No resize was submitted. Try again.'
        : budget === 60000 ? 'The resize exceeded 60 seconds. Try a smaller canvas or a simpler design.'
          : `The resize exceeded the configured ${Math.ceil(budget / 1000)}-second runtime budget. The local engine may still be finishing; retry after it completes.`;
      errorMessage(error.name === 'AbortError' ? new Error(timeoutMessage)
        : phase === 'policy' ? new Error(`Cannot verify the local resize runtime policy. No resize was submitted. ${error.message}`) : error);
      $('status').textContent = 'Resize failed. Adjust the input and try again.';
    } finally { clearTimeout(timeout); finishRequest(controller); }
  }

  $('resizeForm').addEventListener('submit', resize);
  function setView(view) {
    $('previews').dataset.view = view;
    $('originalPanel').hidden = view === 'result';
    $('resizedPanel').hidden = view === 'source';
    for (const button of document.querySelectorAll('.view-switch button')) button.setAttribute('aria-pressed', String(button.dataset.view === view));
    if (state.selected && view !== 'compare') selectElement(state.selected, view);
  }
  for (const button of document.querySelectorAll('.view-switch button')) button.addEventListener('click', () => setView(button.dataset.view));
  setView('result');
  $('theme').addEventListener('change', () => { document.documentElement.dataset.theme = $('theme').value; });
  $('sample').addEventListener('change', () => {
    if ($('sample').value.startsWith('poster:')) { loadPoster($('sample').value.slice(7)); return; }
    const item = $('sample').value === 'imported' ? state.imported : state.samples.find(sample => sample.id === $('sample').value);
    if (item) {
      state.loadVersion++; setTask(item.task, item.label, {preserveTarget: true});
      syncSourceURL($('sample').value === 'imported' ? null : 'sample', $('sample').value);
      resize();
    }
  });
  $('preset').addEventListener('change', () => {
    const target = $('preset').value === 'gentle' && state.task
      ? gentleTarget(state.task.source.canvas) : state.targets.find(item => item.id === $('preset').value);
    if (target) { $('width').value = target.width; $('height').value = target.height; markDirty(); }
  });
  for (const id of ['width', 'height']) $(id).addEventListener('input', () => { syncPreset(); markDirty(); });
  $('showBoxes').addEventListener('change', redraw);
  $('applyJson').addEventListener('click', () => {
    try {
      const edited = JSON.parse($('taskJson').value);
      const task = edited.source ? edited : {source: edited, target_canvas: {
        width: Number($('width').value) || 1080, height: Number($('height').value) || 1080,
      }};
      if (hasResources(state.task) && !['assets', 'fonts', 'document'].some(key => key in task)) {
        for (const key of ['assets', 'fonts', 'document', 'poster_id', 'attribution']) task[key] = state.task[key];
      }
      importTask(task, 'Edited design');
    }
    catch (error) { errorMessage(new Error(`Cannot load JSON. ${error.message}`)); }
  });
  $('taskFile').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    const loadVersion = ++state.loadVersion;
    try {
      if (file.size > 24 * 1024 * 1024) throw new Error('The file must be smaller than 24 MiB.');
      const task = JSON.parse(await file.text());
      if (loadVersion !== state.loadVersion) return;
      if (!hasResources(task) && file.size > 2000000) throw new Error('Plain tasks must be smaller than 2 MB. Larger files need an image/font bundle.');
      importTask(task, file.name);
    }
    catch (error) { if (loadVersion === state.loadVersion) errorMessage(new Error(`Cannot load the task file. ${error.message}`)); }
    finally { if (event.target.files[0] === file) event.target.value = ''; }
  });
  for (const id of ['downloadJson', 'downloadPng']) $(id).addEventListener('click', event => { if ($(id).getAttribute('aria-disabled') === 'true') event.preventDefault(); });
  window.addEventListener('pagehide', event => {
    if (state.busy) {
      cancelRequest();
      $('status').textContent = 'Resize paused when leaving the page. Retry to update the preview.';
    }
    // Cached documents retain their blob URLs when restored with Back/Forward.
    if (!event.persisted) disableDownloads();
  });

  async function init() {
    if (state.initializing) return;
    state.initializing = true;
    const loadVersion = state.loadVersion;
    if (!state.task) { $('run').disabled = true; $('runLabel').textContent = 'Connecting to engine…'; }
    try {
      const [meta, library] = await Promise.all([requestJSON('/api/samples'), requestJSON('/api/posters').catch(() => ({examples: []}))]);
      if (!meta || !Array.isArray(meta.samples) || !Array.isArray(meta.targets)) throw new Error('The engine returned an incomplete design catalogue. Retry connection.');
      state.samples = meta.samples.filter(item => !/^chameleon_sample_level[123](?:_|$)/.test(item.id));
      state.targets = meta.targets; state.posters = Array.isArray(library?.examples) ? library.examples : [];
      $('sample').replaceChildren(...state.samples.map(item => new Option(item.label, item.id)));
      if (state.imported) { $('sample').add(new Option('Imported design', 'imported')); $('sample').value = 'imported'; }
      for (const [split, label] of [['reviewed', 'Open posters · reviewed'], ['reserve', 'Open posters · personal tests']]) {
        const entries = state.posters.filter(item => item.split === split);
        if (!entries.length) continue;
        const group = document.createElement('optgroup'); group.label = label;
        for (const item of entries) {
          const option = new Option(`${item.title}${item.ready ? '' : ' · needs import support'}`, `poster:${item.id}`);
          option.disabled = !item.ready; group.append(option);
        }
        $('sample').append(group);
      }
      $('preset').replaceChildren(new Option('Near original · gentle resize', 'gentle'), new Option('Custom dimensions', 'custom'), ...state.targets.map(item => new Option(`${item.label} · ${item.width} × ${item.height}`, item.id)));
      $('sample').disabled = state.busy;
      $('preset').disabled = $('run').disabled = false;
      if (state.task) { syncPreset(); return; }
      const query = new URLSearchParams(location.search);
      const requestedPoster = query.get('poster');
      const defaultPoster = state.posters.find(item => item.ready && item.split === 'reviewed') || state.posters.find(item => item.ready);
      const poster = state.posters.find(item => item.id === requestedPoster && item.ready)
        || (!state.samples.length ? defaultPoster : null);
      if (loadVersion === state.loadVersion && poster) {
        $('sample').value = `poster:${poster.id}`; await loadPoster(poster.id, {preserveTarget: false}); return;
      }
      if (!state.samples.length) {
        $('sample').replaceChildren(new Option('No posters available', ''));
        $('sample').disabled = !state.imported;
        $('sourceSummary').textContent = 'Import your own design JSON to start.';
        $('taskName').textContent = 'Choose a design';
        $('status').textContent = 'Import a design JSON to start. The poster library is currently unavailable.';
        return;
      }
      const requestedSample = query.get('sample');
      const sample = state.samples.find(item => item.id === requestedSample) || state.samples[Math.min(1, state.samples.length - 1)];
      $('sample').value = sample.id; setTask(sample.task, sample.label);
      if (requestedPoster || requestedSample) syncSourceURL('sample', sample.id);
      await resize();
    } catch (error) {
      if (loadVersion !== state.loadVersion && state.task) return;
      errorMessage(error);
      $('status').textContent = state.task
        ? 'Setup could not finish. Your imported design is preserved; retry Resize design.'
        : 'Setup could not finish. Start the local server if needed, then choose Retry connection.';
    } finally {
      state.initializing = false;
      if (!state.busy) {
        $('run').disabled = false;
        $('run').formNoValidate = !state.task;
        $('runLabel').textContent = state.task ? 'Resize design' : 'Retry connection';
      }
    }
  }
  init();
})();
