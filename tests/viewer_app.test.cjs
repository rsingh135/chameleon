/* Real app.js behavior with a small DOM/API double; no renderer or browser claims. */
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const code = readFileSync(path.join(__dirname, '../viewer/app.js'), 'utf8');
const workspaceCode = readFileSync(path.join(__dirname, '../viewer/workspace.js'), 'utf8');
const markup = readFileSync(path.join(__dirname, '../viewer/index.html'), 'utf8');
const ids = [...markup.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const sample = id => ({id, label: `Sample ${id}`, task: {
  task_id: id, target_canvas: {width: 100, height: 100},
  source: {canvas: {width: 100, height: 100}, elements: [
    {id: 'paper', type: 'shape', x: 0, y: 0, width: 100, height: 100},
    {id: 'hero', type: 'shape', x: 10, y: 10, width: 20, height: 20},
  ]},
}});
const metadata = () => ({samples: [sample('a'), sample('b')], targets: [
  {id: 'square', label: 'Square', width: 100, height: 100},
]});
const png = label => `data:image/png;base64,${Buffer.from(label).toString('base64')}`;
const rendered = payload => ({
  design: {...structuredClone(payload.source), canvas: {...payload.target_canvas}},
  source_png: png('source'), output_png: png('output'), elapsed_ms: 20,
  warnings: [], report: {violations: []},
});

function application({meta = metadata(), posters = [], workspace = false, storageFailure = false, search = ''} = {}) {
  const environment = {
    requests: [], images: [], imageLoads: true, blobs: new Map(), revoked: [], timers: new Map(),
    resizeHandler: payload => rendered(payload), windowEvents: new Map(), imageSizes: new Map(), observers: [],
  };
  class Element {
    constructor(id = '', tagName = 'DIV') {
      this.id = id; this.tagName = tagName; this.value = ''; this.textContent = '';
      this.hidden = false; this.disabled = false; this.checked = false;
      this.dataset = new Proxy({}, {set: (target, key, value) => {
        const changed = target[key] !== value; target[key] = value;
        if (changed) for (const observer of environment.observers) {
          if (observer.target === this && observer.options.attributes) queueMicrotask(observer.callback);
        }
        return true;
      }});
      this.style = {}; this.children = []; this.attributes = new Map(); this.listeners = new Map();
      const classes = new Set();
      this.classList = {add: name => classes.add(name), contains: name => classes.has(name), toggle(name, force) {
        if (force ?? !classes.has(name)) classes.add(name); else classes.delete(name);
      }};
    }
    get options() { return this.children.flatMap(child => child.tagName === 'OPTGROUP' ? child.children : [child]); }
    get href() { return this.attributes.get('href') || ''; }
    set href(value) { this.attributes.set('href', value); }
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    removeAttribute(name) { this.attributes.delete(name); }
    append(...children) { this.children.push(...children); for (const child of children) child.parent = this; }
    add(child) { this.append(child); }
    replaceChildren(...children) {
      this.children = children;
      if (this.tagName === 'SELECT') this.value = this.options[0]?.value || '';
    }
    addEventListener(type, callback) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(callback);
    }
    fire(type, details = {}) {
      const event = {target: this, preventDefault() {}, ...details};
      return Promise.all((this.listeners.get(type) || []).map(callback => callback(event)));
    }
    reportValidity() { return true; }
    contains(element) { return element === this || this.children.some(child => child.contains(element)); }
    matches(selector) { return selector === ':disabled' && (this.disabled || Boolean(this.parent?.matches(selector))); }
    focus() { if (!this.matches(':disabled') && !this.hidden) environment.document.activeElement = this; }
    click() { if (!this.matches(':disabled')) return this.fire('click'); }
    getClientRects() { return this.hidden || this.visible === false ? [] : [{}]; }
    scrollIntoView() {}
    querySelector(selector) { return this.children.find(child => child.tagName === selector.toUpperCase()) || null; }
    getBoundingClientRect() { return {left: 0, top: 0, width: this.width || 100, height: this.height || 100}; }
    getContext() {
      return {setTransform() {}, drawImage() {}, save() {}, translate() {}, rotate() {}, strokeRect() {}, restore() {}};
    }
  }
  const elements = Object.fromEntries(ids.map(id => [id, new Element(id, ['sample', 'preset', 'theme'].includes(id) ? 'SELECT' : 'DIV')]));
  elements.width.value = elements.height.value = '100';
  elements.theme.value = 'dark';
  elements.sample.disabled = elements.preset.disabled = elements.run.disabled = true;
  const views = ['result', 'compare', 'source'].map(view => {
    const button = new Element(); button.dataset.view = view; return button;
  });
  const actions = ['source', 'canvas', 'compare', 'inspect'].map(action => {
    const button = new Element('', 'BUTTON'); button.dataset.workspaceAction = action; return button;
  });
  if (workspace) {
    elements.controls.append(elements.settings, elements.closeSettings);
    elements.settings.append(...['sample', 'preset', 'run', 'width', 'height'].map(id => elements[id]));
    elements.inspection.append(new Element('', 'DETAILS'));
  }
  class Image {
    set src(value) {
      this.url = value;
      if (!value) return;
      environment.images.push(this);
      const size = environment.imageDimensions || environment.imageSizes.get(value) || {width: 100, height: 100};
      this.naturalWidth = size.width; this.naturalHeight = size.height;
      if (environment.imageLoads) queueMicrotask(() => environment.imageError ? this.onerror?.() : this.onload?.());
    }
  }
  class LocalURL extends URL {
    static createObjectURL(blob) {
      const url = `blob:local-${environment.blobs.size}`;
      environment.blobs.set(url, blob); return url;
    }
    static revokeObjectURL(url) { environment.revoked.push(url); }
  }
  const response = (data, status = 200) => ({ok: status < 400, status, json: async () => data});
  let timerId = 0;
  const context = {
    document: {getElementById: id => elements[id], createElement: tag => new Element('', tag.toUpperCase()),
      querySelectorAll: selector => selector === '.view-switch button' ? views
        : selector === '[data-setup-toggle]' ? [elements.settingsToggle, elements.setupRailToggle]
        : selector === '[data-workspace-action]' ? actions : [],
      querySelector: selector => views.find(button => selector.includes(`data-view="${button.dataset.view}"`)),
      documentElement: {dataset: {}}},
    window: {addEventListener(name, callback) {
      if (!environment.windowEvents.has(name)) environment.windowEvents.set(name, []);
      environment.windowEvents.get(name).push(callback);
    }}, location: {search, href: `http://127.0.0.1:8765/${search}`},
    localStorage: {
      getItem() { if (storageFailure) throw new Error('Storage blocked'); return null; },
      setItem() { if (storageFailure) throw new Error('Storage blocked'); },
    },
    matchMedia: () => ({matches: false}),
    MutationObserver: class {
      constructor(callback) { this.callback = callback; }
      observe(target, options) { this.target = target; this.options = options; environment.observers.push(this); }
    },
    history: {replaceState(_state, _title, url) { context.location.href = String(url); context.location.search = new URL(url).search; }},
    Image, Blob, URL: LocalURL, URLSearchParams, AbortController, DOMException, structuredClone,
    atob, Uint8Array, queueMicrotask,
    Option: class extends Element {
      constructor(label, value) { super('', 'OPTION'); this.textContent = label; this.value = value; }
    },
    setTimeout(callback, delay) { environment.timers.set(++timerId, {callback, delay}); return timerId; },
    clearTimeout(id) { environment.timers.delete(id); },
    async fetch(url, options = {}) {
      if (url === '/api/samples') return response(await (typeof meta === 'function' ? meta() : meta));
      if (url === '/api/posters') return response({examples: posters});
      if (url === '/api/resize' || url === '/api/document/resize') {
        const payload = JSON.parse(options.body);
        const request = {url, payload, signal: options.signal}; environment.requests.push(request);
        // Deliberately allow late server replies after abort: app ownership must
        // protect current state independently of cooperative cancellation.
        const result = await environment.resizeHandler(payload, request);
        if (result?.source_png) environment.imageSizes.set(result.source_png, payload.source.canvas);
        if (result?.output_png) environment.imageSizes.set(result.output_png, payload.target_canvas);
        return response(result, environment.resizeStatus || 200);
      }
      if (environment.posterHandler) return response(await environment.posterHandler(url, options));
      throw new Error(`Unexpected route: ${url}`);
    },
  };
  environment.document = context.document;
  vm.runInNewContext(code, context, {filename: 'viewer/app.js'});
  if (workspace) vm.runInNewContext(workspaceCode, context, {filename: 'viewer/workspace.js'});
  environment.elements = elements;
  environment.views = views;
  environment.actions = actions;
  environment.context = context;
  environment.windowEvent = (name, details = {}) => Promise.all(
    (environment.windowEvents.get(name) || []).map(callback => callback(details))
  );
  environment.flush = async () => { for (let index = 0; index < 3; index++) await new Promise(setImmediate); };
  environment.file = (name, text) => {
    const input = elements.taskFile;
    input.value = name;
    input.files = [{name, size: 500, text: typeof text === 'function' ? text : async () => JSON.stringify(text)}];
    return input.fire('change');
  };
  environment.select = async id => { elements.sample.value = id; await elements.sample.fire('change'); await environment.flush(); };
  environment.timeout = async () => {
    for (const [id, timer] of [...environment.timers]) {
      if (timer.delay === 60000) { environment.timers.delete(id); timer.callback(); }
    }
    await environment.flush();
  };
  return environment;
}

test('canvas hit testing selects the latest painted element when z-index ties', async () => {
  const app = application(); await app.flush();
  app.elements.outputCanvas.onclick({clientX: 15, clientY: 15});
  assert.equal(app.elements.propertyTitle.textContent, 'hero');
});

test('late startup metadata preserves a user import and its resource-aware export', async () => {
  const boot = deferred(); const app = application({meta: boot.promise});
  const task = {...sample('owned').task, assets: [{id: 'owned-image'}], fonts: [{family: 'Owned Font'}], document: {format: 'fixture'}};
  await app.file('owned.json', task); await app.flush();
  boot.resolve(metadata()); await app.flush();
  assert.equal(app.elements.taskName.textContent, 'owned.json');
  assert.equal(app.elements.sample.value, 'imported');
  assert.equal(app.requests.at(-1).url, '/api/document/resize');
  const exported = JSON.parse(await app.blobs.get(app.elements.downloadJson.href).text());
  assert.deepEqual(exported.assets, task.assets);
  assert.deepEqual(exported.fonts, task.fonts);
  assert.deepEqual(exported.document, task.document);
  assert.deepEqual(exported.source.canvas, {width: 100, height: 100});
});

test('an older file read cannot replace a later source choice', async () => {
  const app = application(); await app.flush();
  const oldFile = deferred(); const reading = app.file('old.json', () => oldFile.promise);
  await app.select('a');
  oldFile.resolve(JSON.stringify(sample('old').task)); await reading; await app.flush();
  assert.equal(app.elements.taskName.textContent, 'Sample a');
  assert.equal(app.elements.sample.value, 'a');
  assert.match(app.elements.downloadJson.download, /^a_/);
  assert.equal(app.elements.taskFile.value, '');
});

test('a stale file reader cannot clear a newer file selection', async () => {
  const app = application(); await app.flush();
  const older = deferred(), newer = deferred();
  const first = app.file('older.json', () => older.promise);
  const second = app.file('newer.json', () => newer.promise);
  older.resolve(JSON.stringify(sample('older').task)); await first; await app.flush();
  assert.equal(app.elements.taskFile.value, 'newer.json');
  newer.resolve(JSON.stringify(sample('newer').task)); await second; await app.flush();
  assert.equal(app.elements.taskName.textContent, 'newer.json');
  assert.equal(app.elements.taskFile.value, '');
});

test('superseded resize cannot release a newer request or publish old downloads', async () => {
  const app = application(); await app.flush();
  const first = deferred(), second = deferred();
  let index = 0;
  app.resizeHandler = () => (++index === 1 ? first.promise : second.promise);
  await app.select('a');
  const obsolete = app.requests.at(-1);
  await app.file('new.json', sample('new').task); await app.flush();
  assert.equal(index, 2);
  assert.equal(obsolete.signal.aborted, true);
  first.resolve(rendered(obsolete.payload)); await app.flush();
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'true');
  assert.equal(app.elements.downloadJson.getAttribute('aria-disabled'), 'true');
  second.resolve(rendered(app.requests.at(-1).payload)); await app.flush();
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
  assert.match(app.elements.downloadJson.download, /^new_/);
});

test('target edits invalidate an in-flight result even if its server reply arrives late', async () => {
  const app = application(); await app.flush();
  const delayed = deferred(); app.resizeHandler = () => delayed.promise;
  const resizing = app.elements.resizeForm.fire('submit'); await app.flush();
  const obsolete = app.requests.at(-1);
  app.elements.width.value = '200'; await app.elements.width.fire('input');
  delayed.resolve(rendered(obsolete.payload)); await resizing; await app.flush();
  assert.equal(app.elements.downloadJson.getAttribute('aria-disabled'), 'true');
  assert.equal(app.elements.downloadJson.href, '');
  assert.match(app.elements.outputSize.textContent, /previous result/);
  assert.equal(app.elements.width.value, '200');
});

test('the resize timeout also releases pending image decoding and permits retry', async () => {
  const app = application(); await app.flush();
  app.imageLoads = false;
  const resizing = app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'true');
  await app.timeout(); await resizing;
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
  assert.match(app.elements.error.textContent, /60 seconds/);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  app.imageLoads = true;
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
});

test('a network failure marks the engine unavailable and a successful retry restores it', async () => {
  const app = application(); await app.flush();
  const source = app.elements.taskName.textContent;
  app.resizeHandler = () => { throw new TypeError('Failed to fetch'); };
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.connection.textContent, 'Engine unavailable');
  assert.equal(app.elements.connection.classList.contains('online'), false);
  assert.match(app.elements.error.textContent, /then retry/);
  assert.doesNotMatch(app.elements.error.textContent, /reload/);
  assert.equal(app.elements.taskName.textContent, source);
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  app.resizeHandler = payload => rendered(payload);
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.connection.textContent, 'Engine connected');
  assert.equal(app.elements.connection.classList.contains('online'), true);
  assert.equal(app.elements.error.hidden, true);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
});

test('HTTP request errors keep the engine connected while reporting the server error', async () => {
  const app = application(); await app.flush();
  app.resizeStatus = 500;
  app.resizeHandler = () => ({error: 'Document preparation failed: cache is read-only'});
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.connection.textContent, 'Engine connected');
  assert.equal(app.elements.connection.classList.contains('online'), true);
  assert.match(app.elements.error.textContent, /Document preparation failed/);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
});

test('invalid imports preserve the current source and still allow a retry', async () => {
  const app = application(); await app.flush();
  const previous = app.elements.taskName.textContent;
  await app.file('invalid.json', async () => '{'); await app.flush();
  assert.equal(app.elements.taskName.textContent, previous);
  assert.match(app.elements.error.textContent, /Cannot load the task file/);
  assert.equal(app.elements.run.disabled, false);
  assert.equal(app.elements.taskFile.value, '');
});

test('inspection uses the clicked canvas and follows Original/Result view changes', async () => {
  const app = application(); await app.flush();
  app.resizeHandler = payload => {
    const result = rendered(payload); result.design.elements[1].x = 40; return result;
  };
  await app.elements.resizeForm.fire('submit'); await app.flush();
  app.elements.outputCanvas.onclick({clientX: 45, clientY: 15});
  assert.equal(app.elements.propX.textContent, '40 px');
  assert.match(app.elements.propertyType.textContent, /Result/);
  await app.views.find(button => button.dataset.view === 'source').fire('click');
  assert.equal(app.elements.propX.textContent, '10 px');
  assert.match(app.elements.propertyType.textContent, /Original/);
  await app.views.find(button => button.dataset.view === 'compare').fire('click');
  app.elements.sourceCanvas.onclick({clientX: 15, clientY: 15});
  assert.equal(app.elements.propX.textContent, '10 px');
  assert.match(app.elements.propertyType.textContent, /Original/);
  await app.views.find(button => button.dataset.view === 'result').fire('click');
  assert.equal(app.elements.propX.textContent, '40 px');
  assert.match(app.elements.propertyType.textContent, /Result/);
});

test('pasted plain source keeps the selected target canvas and current exports', async () => {
  const app = application(); await app.flush();
  app.elements.width.value = '200'; await app.elements.width.fire('input');
  app.elements.taskJson.value = JSON.stringify(sample('edited').task.source);
  await app.elements.applyJson.fire('click'); await app.flush();
  assert.equal(Number(app.elements.width.value), 200);
  assert.equal(app.requests.at(-1).payload.target_canvas.width, 200);
  const exported = JSON.parse(await app.blobs.get(app.elements.downloadJson.href).text());
  assert.deepEqual(exported.canvas, {width: 200, height: 100});
});

test('a pasted source-only task keeps the selected target instead of resetting to source size', async () => {
  const app = application(); await app.flush();
  app.elements.width.value = '200'; await app.elements.width.fire('input');
  app.elements.taskJson.value = JSON.stringify({source: sample('edited').task.source});
  await app.elements.applyJson.fire('click'); await app.flush();
  assert.equal(Number(app.elements.width.value), 200);
  assert.deepEqual(app.requests.at(-1).payload.target_canvas, {width: 200, height: 100});
});

test('startup offline can recover through the existing submit action without reloading', async () => {
  let offline = true;
  const app = application({meta: () => {
    if (offline) throw new TypeError('Failed to fetch');
    return metadata();
  }});
  await app.flush();
  assert.equal(app.elements.run.disabled, false);
  assert.equal(app.elements.run.formNoValidate, true);
  assert.match(app.elements.runLabel.textContent, /Retry/);
  offline = false;
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.connection.textContent, 'Engine connected');
  assert.equal(app.elements.taskName.textContent, 'Sample b');
  assert.equal(app.elements.run.formNoValidate, false);
  assert.equal(app.elements.downloadJson.getAttribute('aria-disabled'), 'false');
});

test('a malformed successful response cannot replace the previous valid result', async () => {
  const app = application(); await app.flush();
  app.resizeHandler = payload => {
    const result = rendered(payload);
    result.design.elements[1].x = 75;
    delete result.report.violations;
    return result;
  };
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.error.hidden, false);
  assert.equal(app.elements.downloadJson.getAttribute('aria-disabled'), 'true');
  app.elements.outputCanvas.onclick({clientX: 15, clientY: 15});
  assert.equal(app.elements.propertyTitle.textContent, 'hero');
  assert.equal(app.elements.propX.textContent, '10 px');
});

test('an unrelated target response is rejected before drawing or enabling exports', async () => {
  const app = application(); await app.flush();
  app.resizeHandler = payload => {
    const result = rendered(payload); result.design.canvas.width = 999; return result;
  };
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.error.hidden, false);
  assert.equal(app.elements.outputCanvas.width, 100);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
});

test('malformed PNG data is rejected before publishing its result', async () => {
  const app = application(); await app.flush();
  app.resizeHandler = payload => {
    const result = rendered(payload); result.output_png = 'data:image/png;base64,%%%';
    result.design.elements[1].x = 75; return result;
  };
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.error.hidden, false);
  app.elements.outputCanvas.onclick({clientX: 15, clientY: 15});
  assert.equal(app.elements.propertyTitle.textContent, 'hero');
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
});

test('malformed element imports retain the current valid source and exports', async () => {
  const app = application(); await app.flush();
  const task = sample('bad').task; delete task.source.elements[1].width;
  await app.file('bad.json', task); await app.flush();
  assert.equal(app.elements.taskName.textContent, 'Sample b');
  assert.match(app.elements.error.textContent, /Cannot load/);
  assert.match(app.elements.downloadJson.download, /^b_/);
});

test('browser history caching preserves valid blob exports and leaving permanently releases them', async () => {
  const app = application(); await app.flush();
  const exports = [app.elements.downloadJson.href, app.elements.downloadPng.href];
  await app.windowEvent('beforeunload');
  await app.windowEvent('pagehide', {persisted: true});
  await app.windowEvent('pageshow', {persisted: true});
  assert.ok(exports.every(url => !app.revoked.includes(url)));
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
  await app.windowEvent('pagehide', {persisted: false});
  assert.ok(exports.every(url => app.revoked.includes(url)));
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
});

test('a decodable PNG with the wrong pixel dimensions cannot be exported as the requested canvas', async () => {
  const app = application(); await app.flush();
  app.imageDimensions = {width: 17, height: 23};
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.error.hidden, false);
  assert.match(app.elements.error.textContent, /dimensions/);
  assert.equal(app.elements.outputCanvas.width, 100);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
});

test('an actual preview decoding failure preserves prior output and a later retry succeeds', async () => {
  const app = application(); await app.flush();
  app.imageError = true;
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.match(app.elements.error.textContent, /invalid preview image/);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
  app.imageError = false;
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.error.hidden, true);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
});

test('late startup failure cannot overwrite a successful import or its connection status', async () => {
  const boot = deferred(), app = application({meta: boot.promise});
  await app.file('owned.json', sample('owned').task); await app.flush();
  assert.equal(app.elements.connection.textContent, 'Engine connected');
  boot.reject(new TypeError('Failed to fetch')); await app.flush();
  assert.equal(app.elements.connection.textContent, 'Engine connected');
  assert.equal(app.elements.error.hidden, true);
  assert.match(app.elements.status.textContent, /Resize complete/);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
});

test('partial pasted resources are rejected without silently substituting the previous bundle', async () => {
  const app = application(); await app.flush();
  const owned = {...sample('owned').task, assets: [{asset_id: 'owned.png'}], fonts: [], document: {snapshot_id: 'owned'}};
  await app.file('owned.json', owned); await app.flush();
  app.elements.taskJson.value = JSON.stringify({...sample('partial').task, assets: [{asset_id: 'different.png'}]});
  await app.elements.applyJson.fire('click'); await app.flush();
  assert.equal(app.elements.taskName.textContent, 'owned.json');
  assert.match(app.elements.error.textContent, /portable bundle/);
  const exported = JSON.parse(await app.blobs.get(app.elements.downloadJson.href).text());
  assert.deepEqual(exported.assets, owned.assets);
});

test('leaving during a request cancels ownership so a late response cannot publish after return', async () => {
  const app = application(); await app.flush();
  const pending = deferred(); app.resizeHandler = () => pending.promise;
  const running = app.elements.resizeForm.fire('submit'); await app.flush();
  const request = app.requests.at(-1);
  await app.windowEvent('pagehide', {persisted: true});
  assert.equal(request.signal.aborted, true);
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
  pending.resolve(rendered(request.payload)); await running; await app.flush();
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  assert.match(app.elements.status.textContent, /paused/);
});

test('rapid canvas changes and duplicate submits publish only the last requested dimensions', async () => {
  const app = application(); await app.flush();
  const pending = [];
  app.resizeHandler = payload => { const wait = deferred(); pending.push({wait, payload}); return wait.promise; };
  const running = [];
  for (const width of [120, 140, 160, 180, 200]) {
    app.elements.width.value = String(width); await app.elements.width.fire('input');
    running.push(app.elements.resizeForm.fire('submit'));
    await app.elements.resizeForm.fire('submit'); // Same active request must not submit twice.
    await app.flush();
  }
  assert.equal(pending.length, 5);
  for (const {wait, payload} of pending.slice(0, -1).reverse()) wait.resolve(rendered(payload));
  await app.flush();
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'true');
  pending.at(-1).wait.resolve(rendered(pending.at(-1).payload));
  await Promise.all(running); await app.flush();
  assert.equal(app.elements.outputCanvas.width, 200);
  const exported = JSON.parse(await app.blobs.get(app.elements.downloadJson.href).text());
  assert.deepEqual(exported.canvas, {width: 200, height: 100});
});

test('workspace shortcuts focus the panel close control while the target fieldset is disabled', async () => {
  const app = application({workspace: true}); await app.flush();
  app.imageLoads = false;
  const resizing = app.elements.resizeForm.fire('submit'); await app.flush();
  app.elements.setupRailToggle.focus();
  await app.actions.find(button => button.dataset.workspaceAction === 'source').fire('click');
  assert.equal(app.document.activeElement, app.elements.closeSettings);
  await app.timeout(); await resizing;
});

test('workspace supports blocked storage, Escape focus return, and history view restoration without rerendering', async () => {
  const app = application({workspace: true, storageFailure: true, search: '?view=compare'}); await app.flush();
  assert.equal(app.elements.previews.dataset.view, 'compare');
  assert.equal(app.document.documentElement.dataset.workspace, 'review');
  app.elements.width.focus(); await app.elements.controls.fire('keydown', {key: 'Escape'});
  assert.equal(app.elements.controls.hidden, true);
  assert.equal(app.document.activeElement, app.elements.settingsToggle);
  assert.equal(app.elements.settingsToggle.getAttribute('aria-expanded'), 'false');
  await app.elements.settingsToggle.fire('click');
  assert.equal(app.elements.controls.hidden, false);
  assert.equal(app.document.activeElement, app.elements.sample);
  const count = app.requests.length;
  app.context.location.search = '?view=source';
  await app.windowEvent('popstate'); await app.flush();
  assert.equal(app.elements.previews.dataset.view, 'source');
  assert.equal(app.elements.originalPanel.hidden, false);
  assert.equal(app.elements.resizedPanel.hidden, true);
  assert.equal(app.document.documentElement.dataset.workspace, 'studio');
  assert.equal(app.requests.length, count);
  app.elements.theme.value = 'light'; await app.elements.theme.fire('change');
  assert.equal(app.document.documentElement.dataset.theme, 'light');
});

test('malformed diagnostic entries are rejected before they can partially publish a result', async () => {
  const app = application(); await app.flush();
  for (const report of [
    {violations: [null]},
    {violations: [], style: {warnings: [null]}},
    {violations: [], relationships: {warnings: [{message: 'valid'}], relations: 'not-an-array'}},
  ]) {
    app.resizeHandler = payload => {
      const result = rendered(payload); result.design.elements[1].x = 75; result.report = report; return result;
    };
    await app.elements.resizeForm.fire('submit'); await app.flush();
    assert.equal(app.elements.error.hidden, false);
    app.elements.outputCanvas.onclick({clientX: 15, clientY: 15});
    assert.equal(app.elements.propertyTitle.textContent, 'hero');
    assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  }
});

test('optional diagnostic blocks without warnings remain usable and preserve the full report', async () => {
  const app = application(); await app.flush();
  app.resizeHandler = payload => {
    const result = rendered(payload);
    result.report.style = {profile: 'owned-profile'};
    result.report.relationships = {relations: [], ambiguities: []};
    return result;
  };
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.error.hidden, true);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
  assert.match(app.elements.checkSummary.textContent, /Structure checks passed/);
});

test('invalid portable document metadata cannot discard the current valid source or exports', async () => {
  const app = application(); await app.flush();
  const priorRequests = app.requests.length;
  for (const document of ['broken', true, []]) {
    const malformed = {...sample('bad-bundle').task, document, assets: [], fonts: []};
    await app.file('bad-bundle.json', malformed); await app.flush();
    assert.equal(app.elements.taskName.textContent, 'Sample b');
    assert.match(app.elements.error.textContent, /portable bundle/);
    assert.equal(app.requests.length, priorRequests);
    assert.match(app.elements.downloadJson.download, /^b_/);
  }
});

test('retry after a failed poster load fetches the selected poster instead of resizing the previous source', async () => {
  const app = application({posters: [{id: 'new', title: 'New poster', ready: true, bundle_url: '/api/poster-new'}]});
  await app.flush();
  app.posterHandler = () => { throw new TypeError('Failed to fetch'); };
  await app.select('poster:new');
  assert.equal(app.elements.sample.value, 'poster:new');
  assert.equal(app.elements.taskName.textContent, 'Sample b');
  assert.match(app.elements.error.textContent, /Cannot reach/);
  app.posterHandler = () => ({...sample('new').task, assets: [], fonts: [], document: {snapshot_id: 'new'}});
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.taskName.textContent, 'New poster');
  assert.equal(app.requests.at(-1).url, '/api/document/resize');
  const exported = JSON.parse(await app.blobs.get(app.elements.downloadJson.href).text());
  assert.equal(exported.poster_id, 'new');
});

test('rapid portable/plain source switches keep each export bound to the newest exact resource bundle', async () => {
  const app = application(); await app.flush();
  const bundle = (id, bytes) => ({...sample(id).task,
    document: {snapshot_id: id}, assets: [{asset_id: 'shared.png', data_base64: bytes}], fonts: []});
  await app.file('first.json', bundle('first', 'Zmlyc3Q=')); await app.flush();
  const obsolete = deferred(); const oldRead = app.file('obsolete.json', () => obsolete.promise);
  await app.select('a');
  assert.equal(app.requests.at(-1).url, '/api/resize');
  assert.equal('assets' in app.requests.at(-1).payload, false);
  await app.file('latest.json', bundle('latest', 'bGF0ZXN0')); await app.flush();
  obsolete.resolve(JSON.stringify(bundle('obsolete', 'b2xk'))); await oldRead; await app.flush();
  assert.equal(app.elements.taskName.textContent, 'latest.json');
  const exported = JSON.parse(await app.blobs.get(app.elements.downloadJson.href).text());
  assert.equal(exported.document.snapshot_id, 'latest');
  assert.equal(exported.assets[0].data_base64, 'bGF0ZXN0');
  assert.equal(app.requests.at(-1).url, '/api/document/resize');
});

test('navigation cancels a pending poster import and retry after return loads only the current bundle', async () => {
  const app = application({posters: [{id: 'new', title: 'New poster', ready: true, bundle_url: '/api/poster-new'}]});
  await app.flush();
  const pending = deferred(); let posterSignal;
  app.posterHandler = (_url, options) => { posterSignal = options.signal; return pending.promise; };
  await app.select('poster:new');
  await app.windowEvent('pagehide', {persisted: true});
  assert.equal(posterSignal.aborted, true);
  pending.resolve({...sample('obsolete').task, document: {snapshot_id: 'obsolete'}, assets: [], fonts: []});
  await app.flush();
  assert.equal(app.elements.taskName.textContent, 'Sample b');
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  app.posterHandler = () => ({...sample('current').task, document: {snapshot_id: 'current'}, assets: [], fonts: []});
  await app.windowEvent('pageshow', {persisted: true});
  await app.elements.resizeForm.fire('submit'); await app.flush();
  const exported = JSON.parse(await app.blobs.get(app.elements.downloadJson.href).text());
  assert.equal(exported.poster_id, 'new');
  assert.equal(exported.document.snapshot_id, 'current');
  assert.equal(app.elements.taskName.textContent, 'New poster');
});
