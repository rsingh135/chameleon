(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const colors = {text: '#86baff', image: '#e9bc73', shape: '#7dcca4'};
  const state = {samples: [], posters: [], targets: [], imported: null, task: null, result: null, selected: null, images: {}, jsonUrl: null, pngUrl: null, busy: false, dirty: true, version: 0};
  const hasResources = task => task && Array.isArray(task.assets) && Array.isArray(task.fonts) && task.document;

  function errorMessage(error) {
    $('error').textContent = error.message || String(error);
    $('error').hidden = false;
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
    $('previews').setAttribute('aria-busy', String(value));
    $('runLabel').textContent = value ? 'Resizing design…' : 'Resize design';
    $('run').setAttribute('aria-busy', String(value));
  }
  function validateDesign(design) {
    if (!design || !design.canvas || !Array.isArray(design.elements) || !design.elements.length) throw new Error('A design needs a canvas and a nonempty elements array.');
    const {width, height} = design.canvas;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) throw new Error('Source canvas dimensions must be positive whole numbers.');
  }
  function setTask(task, label) {
    validateDesign(task.source || task);
    state.task = structuredClone(task.source ? task : {source: task, target_canvas: {width: Number($('width').value) || 1080, height: Number($('height').value) || 1080}});
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
    const poster = state.posters.find(item => item.id === state.task.poster_id);
    $('posterCredit').replaceChildren();
    $('posterCredit').hidden = !poster;
    if (poster) {
      const credit = document.createElement('span');
      credit.textContent = `${poster.attribution_text || poster.author} · ${poster.license_id}. `;
      $('posterCredit').append(credit);
      const original = document.createElement('a');
      original.href = poster.original_url; original.textContent = 'Editable original';
      $('posterCredit').append(original);
      if (poster.source_url) {
        const source = document.createElement('a'); source.href = poster.source_url;
        source.textContent = ' · Source and attribution'; source.target = '_blank'; source.rel = 'noopener';
        $('posterCredit').append(source);
      }
    }
    $('sourceSize').textContent = `${source.canvas.width} × ${source.canvas.height}`;
    $('outputSize').textContent = 'Not resized yet';
    $('sourceCanvas').hidden = $('outputCanvas').hidden = true;
    $('sourceEmpty').hidden = $('outputEmpty').hidden = false;
    $('inspection').hidden = $('resultSummary').hidden = $('warnings').hidden = $('reviewPanel').hidden = true;
    $('error').hidden = true;
    $('status').textContent = 'Design loaded. Choose a format and resize.';
  }
  function syncPreset() {
    const match = state.targets.find(t => t.width === Number($('width').value) && t.height === Number($('height').value));
    $('preset').value = match ? match.id : 'custom';
  }
  function importTask(task, label) {
    setTask(task, label);
    state.imported = {task: structuredClone(state.task), label};
    if (![...$('sample').options].some(option => option.value === 'imported')) $('sample').add(new Option('Imported design', 'imported'));
    $('sample').value = 'imported';
    resize();
  }
  async function loadPoster(id) {
    const poster = state.posters.find(item => item.id === id);
    if (!poster?.ready) { errorMessage(new Error('This poster needs import support. See the poster library for its notes.')); return; }
    const version = ++state.version;
    markDirty(); setBusy(true); $('error').hidden = true;
    $('status').textContent = 'Loading poster images and fonts…';
    let loaded = false;
    try {
      const bundle = await requestJSON(poster.bundle_url);
      if (version !== state.version) return;
      bundle.poster_id = id; bundle.task_id = id;
      setTask(bundle, poster.title); loaded = true;
    } catch (error) { if (version === state.version) errorMessage(error); }
    finally { setBusy(false); }
    if (loaded) await resize();
  }
  async function requestJSON(url, options = {}) {
    let response;
    try { response = await fetch(url, options); }
    catch (error) { if (error.name === 'AbortError') throw error; throw new Error('Cannot reach the local engine. Start python -m viewer.server and reload this page.'); }
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`The engine returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }
  const loadImage = url => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('The engine returned an invalid preview image.'));
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
      const elements = [...design.elements].sort((a, b) => (b.z_index || 0) - (a.z_index || 0));
      const hit = elements.find(el => {
        const angle = -(el.rotation || 0) * Math.PI / 180;
        const dx = px - el.x - el.width / 2, dy = py - el.y - el.height / 2;
        return Math.abs(dx * Math.cos(angle) - dy * Math.sin(angle)) <= el.width / 2 && Math.abs(dx * Math.sin(angle) + dy * Math.cos(angle)) <= el.height / 2;
      });
      if (hit) selectElement(hit.id);
    };
  }
  function redraw() {
    if (!state.result) return;
    drawCanvas('sourceCanvas', state.task.source, state.images.source);
    drawCanvas('outputCanvas', state.result.design, state.images.output);
  }
  function selectElement(id) {
    if (!state.result) return;
    state.selected = id;
    const el = state.result.design.elements.find(item => item.id === id);
    if (!el) return;
    $('propertyTitle').textContent = el.id;
    $('propertyType').textContent = el.type;
    $('properties').textContent = JSON.stringify(el, null, 2);
    $('propertyHint').hidden = true;
    $('geometry').hidden = false;
    $('propertyPanel').open = true;
    for (const [field, id] of [['x', 'propX'], ['y', 'propY'], ['width', 'propWidth'], ['height', 'propHeight']]) $(id).textContent = `${Number(el[field].toFixed(1))} px`;
    for (const button of $('elements').children) button.setAttribute('aria-pressed', String(button.dataset.id === id));
    redraw();
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
      const name = document.createElement('span'); name.className = 'element-id'; name.textContent = el.id;
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
    if (state.busy || !state.task || !$('resizeForm').reportValidity()) return;
    const width = Number($('width').value), height = Number($('height').value);
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 8192 || height > 8192 || width * height > 20000000) {
      errorMessage(new Error('Use whole canvas dimensions from 1 to 8192 px and no more than 20 million pixels.')); return;
    }
    const version = state.version;
    $('error').hidden = true;
    disableDownloads(); setBusy(true);
    $('resultSummary').hidden = $('warnings').hidden = $('reviewPanel').hidden = true;
    $('status').textContent = 'Reflowing elements and rendering real previews…';
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const payload = hasResources(state.task) ? {...state.task, target_canvas: {width, height}} : {source: state.task.source, target_canvas: {width, height}};
      const result = await requestJSON(hasResources(state.task) ? '/api/document/resize' : '/api/resize', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload), signal: controller.signal});
      const [source, output] = await Promise.all([loadImage(result.source_png), loadImage(result.output_png)]);
      if (version !== state.version) return;
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
      if (version !== state.version) return;
      errorMessage(error.name === 'AbortError' ? new Error('The resize exceeded 60 seconds. Try a smaller canvas or a simpler design.') : error);
      $('status').textContent = 'Resize failed. Adjust the input and try again.';
    } finally { clearTimeout(timeout); setBusy(false); }
  }

  $('resizeForm').addEventListener('submit', resize);
  function setView(view) {
    $('previews').dataset.view = view;
    $('originalPanel').hidden = view === 'result';
    $('resizedPanel').hidden = view === 'source';
    for (const button of document.querySelectorAll('.view-switch button')) button.setAttribute('aria-pressed', String(button.dataset.view === view));
  }
  for (const button of document.querySelectorAll('.view-switch button')) button.addEventListener('click', () => setView(button.dataset.view));
  setView('result');
  function setSetupOpen(open) {
    $('workspace').dataset.setupOpen = String(open);
    $('settingsToggle').setAttribute('aria-expanded', String(open));
    const target = open ? ($('sample').disabled ? $('closeSettings') : $('sample')) : $('settingsToggle');
    requestAnimationFrame(() => {
      if ($('workspace').dataset.setupOpen === String(open)) target.focus();
    });
  }
  $('settingsToggle').addEventListener('click', () => setSetupOpen($('workspace').dataset.setupOpen !== 'true'));
  $('closeSettings').addEventListener('click', () => setSetupOpen(false));
  $('theme').addEventListener('change', () => { document.documentElement.dataset.theme = $('theme').value; });
  $('sample').addEventListener('change', () => {
    if ($('sample').value.startsWith('poster:')) { loadPoster($('sample').value.slice(7)); return; }
    const item = $('sample').value === 'imported' ? state.imported : state.samples.find(sample => sample.id === $('sample').value);
    if (item) { setTask(item.task, item.label); resize(); }
  });
  $('preset').addEventListener('change', () => {
    const target = state.targets.find(item => item.id === $('preset').value);
    if (target) { $('width').value = target.width; $('height').value = target.height; markDirty(); }
  });
  for (const id of ['width', 'height']) $(id).addEventListener('input', () => { syncPreset(); markDirty(); });
  $('showBoxes').addEventListener('change', redraw);
  $('applyJson').addEventListener('click', () => {
    try {
      const edited = JSON.parse($('taskJson').value);
      const task = edited.source ? edited : {source: edited, target_canvas: state.task?.target_canvas};
      if (hasResources(state.task) && !hasResources(task)) {
        for (const key of ['assets', 'fonts', 'document', 'poster_id']) task[key] = state.task[key];
      }
      importTask(task, 'Edited design');
    }
    catch (error) { errorMessage(new Error(`Cannot load JSON. ${error.message}`)); }
  });
  $('taskFile').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 24 * 1024 * 1024) throw new Error('The file must be smaller than 24 MiB.');
      const task = JSON.parse(await file.text());
      if (!hasResources(task) && file.size > 2000000) throw new Error('Plain tasks must be smaller than 2 MB. Larger files need an image/font bundle.');
      importTask(task, file.name);
    }
    catch (error) { errorMessage(new Error(`Cannot load the task file. ${error.message}`)); }
    finally { event.target.value = ''; }
  });
  for (const id of ['downloadJson', 'downloadPng']) $(id).addEventListener('click', event => { if ($(id).getAttribute('aria-disabled') === 'true') event.preventDefault(); });
  window.addEventListener('beforeunload', () => { if (state.jsonUrl) URL.revokeObjectURL(state.jsonUrl); if (state.pngUrl) URL.revokeObjectURL(state.pngUrl); });

  async function init() {
    try {
      const [meta, library] = await Promise.all([requestJSON('/api/samples'), requestJSON('/api/posters').catch(() => ({examples: []}))]);
      state.samples = meta.samples; state.targets = meta.targets; state.posters = library.examples;
      $('sample').replaceChildren(...state.samples.map(item => new Option(item.label, item.id)));
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
      $('preset').replaceChildren(new Option('Custom dimensions', 'custom'), ...state.targets.map(item => new Option(`${item.label} · ${item.width} × ${item.height}`, item.id)));
      $('sample').disabled = $('preset').disabled = $('run').disabled = false;
      $('connection').textContent = 'Engine connected'; $('connection').classList.add('online');
      if (!state.samples.length) throw new Error('No bundled samples were found. Import a task JSON to start.');
      const requestedPoster = new URLSearchParams(location.search).get('poster');
      if (state.posters.some(item => item.id === requestedPoster && item.ready)) {
        $('sample').value = `poster:${requestedPoster}`; await loadPoster(requestedPoster); return;
      }
      const sample = state.samples[Math.min(1, state.samples.length - 1)];
      $('sample').value = sample.id; setTask(sample.task, sample.label); await resize();
    } catch (error) { errorMessage(error); $('connection').textContent = 'Engine unavailable'; $('status').textContent = 'Start the local server, then reload this page.'; }
  }
  init();
})();
