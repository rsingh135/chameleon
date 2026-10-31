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
const semanticReport = source => ({
  source_canvas: {...source.canvas},
  elements: source.elements.map((element, index) => ({
    element_id: element.id,
    purpose: index ? 'corner_brand_mark' : 'background_layer',
    design_purpose: index ? 'Keep the emblem attached to its source corner.' : 'Provide the design background.',
    provenance: index ? 'geometry_inferred' : 'authored_metadata',
    rigid_group: index ? 'brand-cluster' : null,
    anchors: index ? [{edges: ['top', 'right'], horizontal: 'right'}] : [],
    relationships: index ? [{source_id: element.id, target_id: source.elements[0].id,
      type: 'overlays_background', origin: 'source_geometry'}] : [],
    confidence: 0.973,
  })),
});
const renderedWithSemantics = payload => {
  const result = rendered(payload);
  result.report.design_semantics = semanticReport(payload.source);
  return result;
};
const elementText = element => element.textContent + element.children.map(elementText).join('');
const semanticFields = app => Object.fromEntries(app.elements.semanticFields.children.map(field => [
  field.children[0].textContent, elementText(field.children[1]),
]));

function application({meta = metadata(), posters = [], workspace = false, storageFailure = false, search = ''} = {}) {
  const environment = {
    requests: [], policyRequests: [], images: [], imageLoads: true, blobs: new Map(), revoked: [], timers: new Map(),
    resizeHandler: payload => rendered(payload), windowEvents: new Map(), imageSizes: new Map(), observers: [],
    policyHandler: () => ({studio_timeout_ms: 60000, benchmark_timeout_ms: 120000, model_configured: false}),
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
    elements.settings.append(...['preset', 'run', 'width', 'height'].map(id => elements[id]));
    elements.designPicker.append(elements.sample);
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
      if (url === '/api/resize/policy') {
        const request = {url, signal: options.signal, cache: options.cache};
        environment.policyRequests.push(request);
        return response(await environment.policyHandler(request), environment.policyStatus || 200);
      }
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
  environment.timeout = async (delay = 60000) => {
    for (const [id, timer] of [...environment.timers]) {
      if (timer.delay === delay) { environment.timers.delete(id); timer.callback(); }
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

test('every Studio POST obtains a fresh uncached policy and retains the default 60-second timer', async () => {
  const app = application(); await app.flush();
  const calls = app.requests.length, lookups = app.policyRequests.length;
  const pending = deferred(); app.resizeHandler = () => pending.promise;
  const running = app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.policyRequests.length, lookups + 1);
  assert.equal(app.requests.length, calls + 1);
  assert.equal(app.policyRequests.at(-1).cache, 'no-store');
  assert.equal(app.policyRequests.at(-1).signal, app.requests.at(-1).signal);
  assert.ok([...app.timers.values()].some(timer => timer.delay === 60000));
  assert.ok(![...app.timers.values()].some(timer => timer.delay === 10000));
  assert.deepEqual(Object.keys(app.requests.at(-1).payload).sort(), ['source', 'target_canvas']);
  pending.resolve(rendered(app.requests.at(-1).payload)); await running;
  assert.equal(app.timers.size, 0);
});

test('configured Studio model budgets exceed 240 seconds without an old 60-second abort', async () => {
  const app = application(); await app.flush();
  app.policyHandler = () => ({studio_timeout_ms: 630000, benchmark_timeout_ms: 630000, model_configured: true});
  const pending = deferred(); app.resizeHandler = () => pending.promise;
  const running = app.elements.resizeForm.fire('submit'); await app.flush();
  assert.ok([...app.timers.values()].some(timer => timer.delay === 630000));
  assert.ok(![...app.timers.values()].some(timer => timer.delay === 60000));
  assert.match(app.elements.status.textContent, /configured local model pipeline/);
  await app.timeout(60000);
  assert.equal(app.requests.at(-1).signal.aborted, false);
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'true');
  pending.resolve(rendered(app.requests.at(-1).payload)); await running;
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
  assert.equal(app.timers.size, 0);
});

for (const budget of [1950000, 2100000]) test(`Studio accepts the bounded three-role model budget ${budget}`, async () => {
  const app = application(); await app.flush();
  app.policyHandler = () => ({studio_timeout_ms: budget, benchmark_timeout_ms: budget, model_configured: true});
  const count = app.requests.length;
  const pending = deferred(); app.resizeHandler = () => pending.promise;
  const running = app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.requests.length, count + 1);
  const request = app.requests.at(-1);
  assert.ok([...app.timers.values()].some(timer => timer.delay === budget));
  await app.timeout(1500000);
  assert.equal(request.signal.aborted, false);
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'true');
  pending.resolve(rendered(request.payload)); await running;
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
  assert.equal(app.timers.size, 0);
});

test('the three-role Studio deadline aborts and rejects a late result', async () => {
  const app = application(); await app.flush();
  app.policyHandler = () => ({studio_timeout_ms: 1950000, benchmark_timeout_ms: 1950000, model_configured: true});
  const pending = deferred(); app.resizeHandler = () => pending.promise;
  const running = app.elements.resizeForm.fire('submit'); await app.flush();
  const request = app.requests.at(-1);
  await app.timeout(1950000);
  assert.equal(request.signal.aborted, true);
  pending.resolve(rendered(request.payload)); await running;
  assert.match(app.elements.error.textContent, /configured 1950-second runtime budget/);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
  assert.equal(app.timers.size, 0);
});

test('the configured Studio deadline rejects a noncooperative late response without publishing it', async () => {
  const app = application(); await app.flush();
  app.policyHandler = () => ({studio_timeout_ms: 630000, benchmark_timeout_ms: 630000, model_configured: true});
  const pending = deferred(); app.resizeHandler = () => pending.promise;
  const running = app.elements.resizeForm.fire('submit'); await app.flush();
  const request = app.requests.at(-1);
  await app.timeout(630000);
  assert.equal(request.signal.aborted, true);
  pending.resolve(rendered(request.payload)); await running;
  assert.match(app.elements.error.textContent, /configured 630-second runtime budget/);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
  assert.equal(app.timers.size, 0);
});

test('a deadline abort at the resolved-preview boundary cannot publish downloads', async () => {
  const app = application(); await app.flush(); app.imageLoads = false;
  const running = app.elements.resizeForm.fire('submit'); await app.flush();
  for (const image of app.images.slice(-2)) image.onload();
  await app.timeout(60000); await running;
  assert.match(app.elements.error.textContent, /60 seconds/);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
});

for (const policy of [null,
  {studio_timeout_ms: 0, benchmark_timeout_ms: 120000, model_configured: false},
  {studio_timeout_ms: true, benchmark_timeout_ms: 120000, model_configured: false},
  {studio_timeout_ms: 60000.5, benchmark_timeout_ms: 120000, model_configured: false},
  {studio_timeout_ms: 2100001, benchmark_timeout_ms: 120000, model_configured: true},
  {studio_timeout_ms: 1950000, benchmark_timeout_ms: 2100001, model_configured: true},
  {studio_timeout_ms: 60000, benchmark_timeout_ms: 120000, model_configured: 'true'},
]) test(`invalid Studio policy fails before expensive submission: ${JSON.stringify(policy)}`, async () => {
  const app = application(); await app.flush();
  const count = app.requests.length, name = app.elements.taskName.textContent;
  app.policyHandler = () => policy;
  await app.elements.resizeForm.fire('submit');
  assert.equal(app.requests.length, count);
  assert.match(app.elements.error.textContent, /runtime policy.*No resize was submitted|No resize was submitted.*runtime policy/);
  assert.equal(app.elements.taskName.textContent, name);
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  assert.equal(app.timers.size, 0);
});

test('an unavailable policy is not replaced with a silent default Studio deadline', async () => {
  const app = application(); await app.flush(); const count = app.requests.length;
  app.policyStatus = 404; app.policyHandler = () => ({error: 'Runtime policy endpoint is unavailable'});
  await app.elements.resizeForm.fire('submit');
  assert.equal(app.requests.length, count);
  assert.match(app.elements.error.textContent, /No resize was submitted/);
  assert.match(app.elements.error.textContent, /endpoint is unavailable/);
});

test('a late policy response after its 10-second timeout cannot submit a Studio POST', async () => {
  const app = application(); await app.flush(); const count = app.requests.length;
  const policy = deferred(); app.policyHandler = () => policy.promise;
  const running = app.elements.resizeForm.fire('submit'); await app.flush();
  assert.ok([...app.timers.values()].some(timer => timer.delay === 10000));
  await app.timeout(10000);
  policy.resolve({studio_timeout_ms: 60000, benchmark_timeout_ms: 120000, model_configured: false}); await running;
  assert.equal(app.requests.length, count);
  assert.match(app.elements.error.textContent, /lookup exceeded 10 seconds/);
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
});

for (const action of ['canvas', 'navigation']) test(`a late policy after ${action} cannot submit stale Studio work`, async () => {
  const app = application(); await app.flush(); const count = app.requests.length;
  const policy = deferred(); app.policyHandler = () => policy.promise;
  const running = app.elements.resizeForm.fire('submit'); await app.flush();
  const old = app.policyRequests.at(-1);
  if (action === 'canvas') { app.elements.width.value = '200'; await app.elements.width.fire('input'); }
  else await app.windowEvent('pagehide', {persisted: true});
  policy.resolve({studio_timeout_ms: 60000, benchmark_timeout_ms: 120000, model_configured: false}); await running;
  assert.equal(old.signal.aborted, true);
  assert.equal(app.requests.length, count);
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'true');
  assert.match(app.elements.status.textContent, action === 'canvas' ? /Canvas changed/ : /paused/);
});

test('an obsolete source policy cannot submit or release a newer resource-bound resize', async () => {
  const app = application(); await app.flush(); const before = app.requests.length;
  const oldPolicy = deferred(), freshResult = deferred(); let count = 0;
  app.policyHandler = () => ++count === 1 ? oldPolicy.promise : {studio_timeout_ms: 60000, benchmark_timeout_ms: 120000, model_configured: false};
  const old = app.elements.resizeForm.fire('submit'); await app.flush();
  app.resizeHandler = () => freshResult.promise;
  const owned = {...sample('fresh').task, assets: [{asset_id: 'original.png'}], fonts: [], document: {snapshot_id: 'owned'}};
  await app.file('fresh.json', owned); await app.flush();
  const latest = app.requests.at(-1);
  oldPolicy.resolve({studio_timeout_ms: 60000, benchmark_timeout_ms: 120000, model_configured: false}); await old;
  assert.equal(app.requests.length, before + 1);
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'true');
  assert.deepEqual(latest.payload.assets, owned.assets);
  assert.equal(latest.url, '/api/document/resize');
  freshResult.resolve(rendered(latest.payload)); await app.flush();
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'false');
  assert.match(app.elements.downloadJson.download, /^fresh_/);
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

test('semantic inspection follows the same source element across Original and Result geometry', async () => {
  const app = application(); await app.flush();
  app.resizeHandler = payload => {
    const result = renderedWithSemantics(payload); result.design.elements[1].x = 40; return result;
  };
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.designSemantics.hidden, true);
  app.elements.outputCanvas.onclick({clientX: 45, clientY: 15});
  const expected = {
    Purpose: 'Keep the emblem attached to its source corner.',
    Role: 'corner brand mark',
    Evidence: 'geometry inferred',
    'Source anchors': 'top + right · right aligned',
    'Authored group': 'brand-cluster',
    Connections: 'overlays background: hero → paper · source geometry',
  };
  assert.equal(app.elements.propertyTitle.textContent, 'hero');
  assert.equal(app.elements.propX.textContent, '40 px');
  assert.equal(app.elements.designSemantics.hidden, false);
  assert.deepEqual(semanticFields(app), expected);
  assert.doesNotMatch(elementText(app.elements.semanticFields), /confidence|0\.973/i);

  await app.views.find(button => button.dataset.view === 'source').fire('click');
  assert.equal(app.elements.propX.textContent, '10 px');
  assert.match(app.elements.propertyType.textContent, /Original/);
  assert.deepEqual(semanticFields(app), expected);
  await app.views.find(button => button.dataset.view === 'compare').fire('click');
  app.elements.sourceCanvas.onclick({clientX: 15, clientY: 15});
  assert.deepEqual(semanticFields(app), expected);
  await app.views.find(button => button.dataset.view === 'result').fire('click');
  assert.equal(app.elements.propX.textContent, '40 px');
  assert.match(app.elements.propertyType.textContent, /Result/);
  assert.deepEqual(semanticFields(app), expected);
});

test('a missing optional semantic report keeps inspection and current exports available', async () => {
  const app = application(); await app.flush();
  app.elements.outputCanvas.onclick({clientX: 15, clientY: 15});
  assert.equal(app.elements.propertyTitle.textContent, 'hero');
  assert.equal(app.elements.designSemantics.hidden, true);
  assert.equal(app.elements.semanticFields.children.length, 0);
  assert.equal(app.elements.error.hidden, true);
  assert.equal(app.elements.downloadJson.getAttribute('aria-disabled'), 'false');
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
  const exported = JSON.parse(await app.blobs.get(app.elements.downloadJson.href).text());
  assert.deepEqual(exported.elements.map(element => element.id), ['paper', 'hero']);
});

for (const [name, corrupt] of [
  ['a mismatched source canvas', report => { report.source_canvas.width += 1; }],
  ['incomplete source coverage', report => { report.elements.pop(); }],
  ['duplicate element IDs', report => { report.elements[1].element_id = 'paper'; }],
  ['an unknown element ID', report => { report.elements[1].element_id = 'unrelated'; }],
  ['an unknown connection ID', report => { report.elements[1].relationships[0].target_id = 'unrelated'; }],
  ['a connection unrelated to its row', report => { report.elements[1].relationships[0].source_id = 'paper'; }],
  ['a non-text purpose', report => { report.elements[1].design_purpose = {text: 'invalid'}; }],
  ['an invalid source anchor', report => { report.elements[1].anchors[0].edges = ['diagonal']; }],
]) test(`malformed optional semantics are discarded for ${name} while resize and export succeed`, async () => {
  const app = application(); await app.flush();
  app.resizeHandler = payload => {
    const result = renderedWithSemantics(payload); corrupt(result.report.design_semantics); return result;
  };
  await app.elements.resizeForm.fire('submit'); await app.flush();
  app.elements.outputCanvas.onclick({clientX: 15, clientY: 15});
  assert.equal(app.elements.propertyTitle.textContent, 'hero');
  assert.equal(app.elements.designSemantics.hidden, true);
  assert.equal(app.elements.semanticFields.children.length, 0);
  assert.equal(app.elements.error.hidden, true);
  assert.equal(app.elements.downloadJson.getAttribute('aria-disabled'), 'false');
  assert.equal(app.elements.downloadPng.getAttribute('aria-disabled'), 'false');
  const exported = JSON.parse(await app.blobs.get(app.elements.downloadJson.href).text());
  assert.deepEqual(exported.canvas, {width: 100, height: 100});
});

test('semantic descriptions and connections render untrusted markup as literal text', async () => {
  const app = application(); await app.flush();
  const injected = '<img src=x onerror="alert(1)">';
  app.resizeHandler = payload => {
    const result = renderedWithSemantics(payload);
    const row = result.report.design_semantics.elements[1];
    row.design_purpose = row.purpose = row.provenance = row.rigid_group = injected;
    row.relationships[0].type = row.relationships[0].origin = injected;
    return result;
  };
  await app.elements.resizeForm.fire('submit'); await app.flush();
  app.elements.outputCanvas.onclick({clientX: 15, clientY: 15});
  assert.equal(app.elements.designSemantics.hidden, false);
  const fields = semanticFields(app);
  for (const name of ['Purpose', 'Role', 'Evidence', 'Authored group']) assert.equal(fields[name], injected);
  assert.equal(fields.Connections, `${injected}: hero → paper · ${injected}`);
  for (const field of app.elements.semanticFields.children) {
    const detail = field.children[1];
    if (field.children[0].textContent === 'Connections') {
      assert.equal(detail.children[0].tagName, 'UL');
      assert.equal(detail.children[0].children[0].tagName, 'LI');
      assert.equal(detail.children[0].children[0].children.length, 0);
    } else assert.equal(detail.children.length, 0);
  }
});

test('a new source clears semantic inspection and an older response cannot replace its current panel', async () => {
  const app = application(); await app.flush();
  app.resizeHandler = renderedWithSemantics;
  await app.elements.resizeForm.fire('submit'); await app.flush();
  app.elements.outputCanvas.onclick({clientX: 15, clientY: 15});
  assert.equal(app.elements.designSemantics.hidden, false);
  const oldResult = deferred(), freshResult = deferred(); let count = 0;
  app.resizeHandler = () => ++count === 1 ? oldResult.promise : freshResult.promise;
  const oldRun = app.elements.resizeForm.fire('submit'); await app.flush();
  const oldRequest = app.requests.at(-1);
  const task = sample('fresh').task; task.source.elements[1].id = 'fresh-hero';
  await app.file('fresh.json', task); await app.flush();
  const freshRequest = app.requests.at(-1);
  assert.equal(oldRequest.signal.aborted, true);
  assert.equal(app.elements.designSemantics.hidden, true);
  assert.equal(app.elements.semanticFields.children.length, 0);

  const fresh = renderedWithSemantics(freshRequest.payload);
  fresh.report.design_semantics.elements[1].design_purpose = 'Current imported emblem.';
  freshResult.resolve(fresh); await app.flush();
  app.elements.outputCanvas.onclick({clientX: 15, clientY: 15});
  assert.equal(app.elements.propertyTitle.textContent, 'fresh-hero');
  assert.equal(semanticFields(app).Purpose, 'Current imported emblem.');
  oldResult.resolve(renderedWithSemantics(oldRequest.payload)); await oldRun; await app.flush();
  assert.equal(app.elements.taskName.textContent, 'fresh.json');
  assert.equal(app.elements.designSemantics.hidden, false);
  assert.equal(app.elements.propertyTitle.textContent, 'fresh-hero');
  assert.equal(semanticFields(app).Purpose, 'Current imported emblem.');
  assert.match(semanticFields(app).Connections, /fresh-hero → paper/);
  assert.equal(app.elements.downloadJson.getAttribute('aria-disabled'), 'false');
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

test('the canvas shortcut focuses the panel close control while the target fieldset is disabled', async () => {
  const app = application({workspace: true}); await app.flush();
  app.imageLoads = false;
  const resizing = app.elements.resizeForm.fire('submit'); await app.flush();
  app.elements.setupRailToggle.focus();
  await app.actions.find(button => button.dataset.workspaceAction === 'canvas').fire('click');
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
  assert.equal(app.document.activeElement, app.elements.preset);
  assert.equal(app.elements.controls.contains(app.document.activeElement), true);
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

test('the visible design picker preserves target and Compare when Setup is folded', async () => {
  const app = application({workspace: true, posters: [
    {id: 'new', title: 'New poster', split: 'reviewed', ready: true, bundle_url: '/api/poster-new'},
  ]});
  await app.flush();
  await app.views.find(button => button.dataset.view === 'compare').fire('click');
  await app.elements.closeSettings.fire('click');
  assert.equal(app.elements.workspace.dataset.setupOpen, 'false');
  assert.equal(app.elements.controls.contains(app.elements.sample), false);
  assert.equal(app.elements.designPicker.contains(app.elements.sample), true);
  app.elements.width.value = '240'; app.elements.height.value = '160';
  await app.elements.width.fire('input');
  await app.select('a');
  assert.deepEqual(app.requests.at(-1).payload.target_canvas, {width: 240, height: 160});
  app.posterHandler = () => ({...sample('new').task, assets: [], fonts: [], document: {snapshot_id: 'new'}});
  await app.select('poster:new'); await app.flush();
  assert.equal(app.elements.taskName.textContent, 'New poster');
  assert.equal(app.elements.previews.dataset.view, 'compare');
  assert.equal(app.elements.workspace.dataset.setupOpen, 'false');
  assert.deepEqual(app.requests.at(-1).payload.target_canvas, {width: 240, height: 160});
  assert.equal(app.requests.at(-1).payload.poster_id, 'new');
  assert.match(app.elements.downloadPng.download, /^new_240x160/);
  await app.select('b');
  assert.equal(app.elements.taskName.textContent, 'Sample b');
  assert.equal(app.elements.previews.dataset.view, 'compare');
  assert.deepEqual(app.requests.at(-1).payload.target_canvas, {width: 240, height: 160});
  assert.match(app.elements.downloadPng.download, /^b_240x160/);
});

test('the standalone picker stays disabled until its pending resize settles', async () => {
  const app = application({workspace: true}); await app.flush();
  const pending = deferred(); app.resizeHandler = () => pending.promise;
  const resizing = app.elements.resizeForm.fire('submit'); await app.flush();
  assert.equal(app.elements.sample.disabled, true);
  pending.resolve(rendered(app.requests.at(-1).payload));
  await resizing; await app.flush();
  assert.equal(app.elements.sample.disabled, false);
});

test('late startup metadata cannot unlock the standalone picker during an imported resize', async () => {
  const boot = deferred();
  const app = application({meta: boot.promise, workspace: true});
  const pending = deferred(); app.resizeHandler = () => pending.promise;
  await app.file('owned.json', sample('owned').task); await app.flush();
  assert.equal(app.elements.sample.disabled, true);
  boot.resolve(metadata()); await app.flush();
  assert.equal(app.elements.run.getAttribute('aria-busy'), 'true');
  assert.equal(app.elements.sample.disabled, true);
  pending.resolve(rendered(app.requests.at(-1).payload)); await app.flush();
  assert.equal(app.elements.sample.disabled, false);
  assert.equal(app.elements.taskName.textContent, 'owned.json');
  assert.match(app.elements.downloadJson.download, /^owned_/);
});

test('the Source shortcut focuses the visible picker while leaving folded Setup folded', async () => {
  const app = application({workspace: true}); await app.flush();
  await app.elements.closeSettings.fire('click');
  await app.actions.find(button => button.dataset.workspaceAction === 'source').fire('click');
  assert.equal(app.elements.controls.hidden, true);
  assert.equal(app.elements.workspace.dataset.setupOpen, 'false');
  assert.equal(app.document.activeElement, app.elements.sample);
});

test('the Source shortcut retains reachable focus when its picker is busy', async () => {
  const app = application({workspace: true}); await app.flush();
  await app.elements.closeSettings.fire('click');
  app.imageLoads = false;
  const resizing = app.elements.resizeForm.fire('submit'); await app.flush();
  const sourceShortcut = app.actions.find(button => button.dataset.workspaceAction === 'source');
  sourceShortcut.focus(); await sourceShortcut.fire('click');
  assert.equal(app.elements.controls.hidden, true);
  assert.equal(app.elements.sample.disabled, true);
  assert.equal(app.document.activeElement, sourceShortcut);
  await app.timeout(); await resizing;
});

test('retrying a failed picker poster keeps the current custom target', async () => {
  const app = application({posters: [
    {id: 'new', title: 'New poster', split: 'reviewed', ready: true, bundle_url: '/api/poster-new'},
  ]});
  await app.flush();
  app.elements.width.value = '320'; app.elements.height.value = '180';
  await app.elements.width.fire('input');
  app.posterHandler = () => { throw new TypeError('Failed to fetch'); };
  await app.select('poster:new');
  app.posterHandler = () => ({...sample('new').task, assets: [], fonts: [], document: {snapshot_id: 'new'}});
  await app.elements.resizeForm.fire('submit'); await app.flush();
  assert.deepEqual(app.requests.at(-1).payload.target_canvas, {width: 320, height: 180});
  assert.match(app.elements.downloadJson.download, /^new_320x180/);
});

test('source picker URL tracks the successful source and preserves comparison query context', async () => {
  const app = application({workspace: true, search: '?poster=first&view=compare&width=800&height=500', posters: [
    {id: 'first', title: 'First poster', split: 'reviewed', ready: true, bundle_url: '/api/poster-first'},
    {id: 'second', title: 'Second poster', split: 'reviewed', ready: true, bundle_url: '/api/poster-second'},
  ]});
  app.posterHandler = url => ({...sample(url.endsWith('first') ? 'first' : 'second').task,
    assets: [], fonts: [], document: {snapshot_id: url}});
  await app.flush();
  await app.select('poster:second');
  let query = new URL(app.context.location.href).searchParams;
  assert.equal(query.get('poster'), 'second');
  assert.equal(query.get('sample'), null);
  assert.equal(query.get('view'), 'compare');
  assert.equal(query.get('width'), '800');
  assert.equal(query.get('height'), '500');
  await app.select('a');
  query = new URL(app.context.location.href).searchParams;
  assert.equal(query.get('poster'), null);
  assert.equal(query.get('sample'), 'a');
  assert.equal(query.get('view'), 'compare');
  await app.file('owned.json', sample('owned').task); await app.flush();
  query = new URL(app.context.location.href).searchParams;
  assert.equal(query.get('poster'), null);
  assert.equal(query.get('sample'), null);
  assert.equal(query.get('view'), 'compare');
});

test('refreshing a selected bundled design restores its exact source rather than the default sample', async () => {
  const meta = metadata(); meta.samples[0].task.source.elements[1].id = 'only-in-a';
  const app = application({meta, workspace: true, search: '?sample=a&view=compare'});
  await app.flush();
  assert.equal(app.elements.sample.value, 'a');
  assert.equal(app.elements.taskName.textContent, 'Sample a');
  assert.equal(app.elements.previews.dataset.view, 'compare');
  assert.equal(app.requests.at(-1).payload.source.elements[1].id, 'only-in-a');
  assert.match(app.elements.downloadPng.download, /^a_/);
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
