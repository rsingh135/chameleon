/* Execute the real benchmark UI against a minimal DOM/API double.
 * These checks cover orchestration and identity; they make no rendering claims.
 */
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const code = readFileSync(path.join(__dirname, '../viewer/benchmark.js'), 'utf8');
const markup = readFileSync(path.join(__dirname, '../viewer/benchmark.html'), 'utf8');
const ids = [...markup.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
const deferred = () => {
  let resolve;
  const promise = new Promise(yes => { resolve = yes; });
  return {promise, resolve};
};
const method = {id: 'deterministic-current', digest: 'a'.repeat(64)};
const source = () => ({canvas: {width: 100, height: 100}, elements: [
  {id: 'paper', type: 'shape', x: 0, y: 0, width: 100, height: 100},
]});
const sample = (id, target) => ({id, label: `Scene ${id}`, task: {
  source: source(), target_canvas: target || {width: 400, height: 500},
}});
const rowFor = body => {
  const scene = body.scenes[0], target = body.targets[0];
  return {scene_id: scene.id, target_canvas: {...target}, source_sha256: 'b'.repeat(64),
    resources_sha256: 'c'.repeat(64), status: 'ok', contract_passed: true, elapsed_ms: 1,
    result: {source_png: 'data:image/png;base64,c291cmNl', output_png: 'data:image/png;base64,b3V0cHV0',
      design: {...scene.payload.source, canvas: {...target}}, report: {violations: []}, warnings: []}};
};
const responseFor = body => ({method, comparable: true, results: [rowFor(body)]});

function application(settings = {}) {
  const app = {requests: [], posterRequests: [], blobs: new Map(), blobCreations: [], revoked: [], images: new Map(), timers: new Map(), handler: body => responseFor(body)};
  const windowListeners = new Map();
  let methodRequests = 0;
  class Element {
    constructor(tag = 'div', id = '') {
      this.tagName = tag.toUpperCase(); this.id = id; this.value = ''; this.children = [];
      this.attributes = new Map(); this.listeners = new Map(); this.dataset = {}; this.textContent = '';
      this.hidden = false; this.disabled = false;
      const classes = new Set();
      this.classList = {toggle(name, force) {
        if (force ?? !classes.has(name)) classes.add(name); else classes.delete(name);
      }};
    }
    set href(value) { this.attributes.set('href', value); }
    get href() { return this.attributes.get('href') || ''; }
    set download(value) { this.attributes.set('download', value); }
    get download() { return this.attributes.get('download') || ''; }
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    removeAttribute(name) { this.attributes.delete(name); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    querySelectorAll(selector) {
      const all = this.children.flatMap(child => [child, ...child.querySelectorAll(selector)]);
      return all.filter(child => selector === 'a[download]' && child.tagName === 'A' && child.download);
    }
    addEventListener(name, callback) {
      if (!this.listeners.has(name)) this.listeners.set(name, []);
      this.listeners.get(name).push(callback);
    }
    fire(name) { return Promise.all((this.listeners.get(name) || []).map(callback => callback({preventDefault() {}}))); }
  }
  const elements = Object.fromEntries(ids.map(id => [id, new Element('div', id)]));
  elements.targetSizes.value = '100x100\n300x200';
  elements.evaluationUse.value = 'exploration';
  let nextUrl = 0, nextScene = 0, nextTimer = 0;
  class LocalURL extends URL {
    static createObjectURL(blob) { const url = `blob:${++nextUrl}`; app.blobs.set(url, blob); return url; }
    static revokeObjectURL(url) { app.revoked.push(url); }
  }
  const context = {
    document: {getElementById: id => elements[id], createElement: tag => new Element(tag), documentElement: {dataset: {}}},
    window: {addEventListener(name, handler) { windowListeners.set(name, handler); }},
    localStorage: {getItem() { if (settings.storageFailure) throw new Error('Storage unavailable'); return null; }, setItem() { if (settings.storageFailure) throw new Error('Storage unavailable'); }},
    URL: LocalURL, Blob: class extends Blob {
      constructor(parts, options) { super(parts, options); app.blobCreations.push({type: this.type, size: this.size}); }
    }, AbortController, Date, Uint8Array, atob, btoa,
    Image: class {
      set src(value) { this.url = value; }
      async decode() {
        if (settings.decodeFailure) throw new Error('Image is corrupt');
        if (settings.decodeStall) return new Promise(() => {});
        const size = app.images.get(this.url);
        this.naturalWidth = settings.wrongImageSize ? size?.width + 1 : size?.width;
        this.naturalHeight = size?.height;
      }
    },
    crypto: {randomUUID: () => `upload-${++nextScene}`, subtle: {digest: async () => new ArrayBuffer(32)}},
    createImageBitmap() { app.bitmapCalled = true; return settings.bitmapStall ? new Promise(() => {}) : Promise.resolve({width: 240, height: 160, close() {}}); },
    setTimeout(callback, milliseconds) { const id = ++nextTimer; app.timers.set(id, {callback, milliseconds}); return id; },
    clearTimeout(id) { app.timers.delete(id); },
    async fetch(url, options = {}) {
      if (url === '/api/benchmark/method') {
        if (settings.failInitialMethod && ++methodRequests === 1) throw new TypeError('Offline');
        if (app.methodHandler) return {ok: true, json: async () => app.methodHandler()};
        if (settings.configuredMethod) return {ok: true, json: async () => settings.configuredMethod};
        return {ok: true, json: async () => method};
      }
      if (url === '/api/samples') return {ok: true, json: async () => ({samples: [sample('one'), sample('two', {width: 555, height: 777})]})};
      if (url === '/api/posters') return {ok: true, json: async () => ({examples: settings.posters || [
        {id: 'reviewed-one', title: 'Reviewed One', split: 'reviewed', ready: true},
        {id: 'reviewed-two', title: 'Reviewed Two', split: 'reviewed', ready: true},
        {id: 'reserved', title: 'Reserved source', split: 'reserve', ready: true},
        {id: 'unprepared', title: 'Unprepared source', split: 'reviewed', ready: false},
      ]})};
      if (url.startsWith('/api/posters/') && url.endsWith('/bundle')) {
        const id = url.split('/')[3]; app.posterRequests.push(id);
        return {ok: true, json: async () => settings.posterBundle ? settings.posterBundle(id) : sample(id).task};
      }
      assert.equal(url, '/api/benchmark/run');
      const body = JSON.parse(options.body); app.requests.push(body);
      return {ok: true, json: async () => {
        const response = await app.handler(body);
        for (const row of response.results || []) {
          if (row.result) {
            app.images.set(row.result.source_png, body.scenes[0].payload.source.canvas);
            app.images.set(row.result.output_png, row.result.design.canvas);
          }
        }
        return response;
      }};
    },
  };
  vm.runInNewContext(code, context, {filename: 'viewer/benchmark.js'});
  app.elements = elements;
  app.flush = async () => { for (let i = 0; i < 3; i++) await new Promise(setImmediate); };
  app.addSamples = () => elements.addSamples.fire('click');
  app.browseReviewed = () => elements.loadReviewed.fire('click');
  app.addReviewed = ids => {
    elements.reviewedPosters.selectedOptions = ids.map(value => ({value}));
    return elements.addReviewed.fire('click');
  };
  app.import = files => {
    elements.sceneFiles.files = files.map(([name, text]) => ({name, size: 100, text: async () => text}));
    return elements.sceneFiles.fire('change');
  };
  app.run = () => elements.batchForm.fire('submit');
  app.importRaster = () => {
    elements.sceneFiles.files = [{name: 'scene.png', size: 100, type: 'image/png', arrayBuffer: async () => new ArrayBuffer(1)}];
    return elements.sceneFiles.fire('change');
  };
  app.windowEvent = (name, event) => windowListeners.get(name)?.(event);
  app.timeout = milliseconds => {
    for (const [id, timer] of [...app.timers]) if (timer.milliseconds === milliseconds) { app.timers.delete(id); timer.callback(); }
  };
  app.report = async () => JSON.parse(await app.blobs.get(elements.downloadReport.href).text());
  return app;
}

test('every scene uses the same explicit target list and frozen method digest', async () => {
  const app = application(); await app.flush(); await app.addSamples(); await app.run();
  assert.equal(app.requests.length, 4);
  assert.deepEqual(app.requests.map(job => [job.scenes[0].id, job.targets[0]]), [
    ['one', {width: 100, height: 100}], ['one', {width: 300, height: 200}],
    ['two', {width: 100, height: 100}], ['two', {width: 300, height: 200}],
  ]);
  assert.ok(app.requests.every(job => job.expected_method_digest === method.digest));
  const report = await app.report();
  assert.equal(report.status, 'completed');
  assert.equal(report.results.length, 4);
  assert.deepEqual(report.targets, [{width: 100, height: 100}, {width: 300, height: 200}]);
  assert.equal(report.visual_review, 'not_reviewed');
  assert.match(app.elements.batchStatus.textContent, /same method and targets/);
  assert.equal(app.elements.methodName.textContent, 'deterministic-current');
  assert.match(app.elements.methodPolicy.textContent, /Vision analyzer and learned critic are off/);
});

for (const [label, model_executed, layout_changed, message] of [
  ['influenced', true, true, 'ran and influenced the layout'],
  ['fallback', false, false, 'was requested but did not run; the engine used its fallback'],
  ['unchanged', true, false, 'ran; no layout change was recorded'],
]) test(`configured visual planner reports ${label} without claiming better quality`, async () => {
  const configuredMethod = {...method, analyzer: {enabled: true}, critic: {enabled: label === 'influenced'}};
  const app = application({configuredMethod}); await app.flush(); await app.addSamples();
  app.elements.targetSizes.value = '100x100';
  assert.equal(app.elements.methodName.textContent, 'Current engine · local visual planner requested');
  assert.match(app.elements.methodPolicy.textContent, /Result traces report whether it ran/);
  assert.match(app.elements.methodPolicy.textContent, label === 'influenced' ? /A learned critic is configured/ : /The learned critic is off/);
  const pending = deferred();
  const reply = body => {
    const result = responseFor(body); result.method = configuredMethod;
    result.results[0].result.report.layout_state = {visual_model: {requested: true, model_executed, layout_changed}};
    return result;
  };
  app.handler = body => app.requests.length === 2 ? pending.promise : reply(body);
  const running = app.run(); await app.flush();
  const walk = element => [element, ...element.children.flatMap(walk)];
  const note = walk(app.elements.batchResults).find(element => element.textContent.includes(`Local visual planner ${message}`));
  assert.ok(note);
  assert.match(note.textContent, /Visual quality is unreviewed/);
  assert.doesNotMatch(note.textContent, /improved|better|approved/);
  pending.resolve(reply(app.requests[1])); await running;
  assert.ok(walk(app.elements.batchResults).includes(note), 'Later completions must preserve the admitted trace note');
  assert.ok(app.requests.every(request => request.method === 'deterministic-current' && request.expected_method_digest === method.digest));
  assert.deepEqual((await app.report()).results[0].result.report.layout_state.visual_model, {requested: true, model_executed, layout_changed});
});

test('critic-only configuration keeps the legacy method ID without claiming the critic is off', async () => {
  const app = application({configuredMethod: {...method, analyzer: {enabled: false}, critic: {enabled: true}}});
  await app.flush();
  assert.equal(app.elements.methodName.textContent, 'deterministic-current');
  assert.match(app.elements.methodPolicy.textContent, /Vision analyzer is off; a learned critic is configured/);
});

test('legacy batch methods keep the 120-second job timer after a short metadata lookup', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  const pending = deferred(); app.handler = () => pending.promise;
  const running = app.run(); await app.flush();
  assert.ok([...app.timers.values()].some(timer => timer.milliseconds === 120000));
  assert.ok(![...app.timers.values()].some(timer => timer.milliseconds === 10000));
  await app.elements.stopBatch.fire('click'); pending.resolve(responseFor(app.requests[0])); await running;
});

test('a fresh configured batch method supplies its long POST budget without entering payloads', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  const configured = {...method, analyzer: {enabled: true}, options: {request_policy: {studio_timeout_ms: 630000, benchmark_timeout_ms: 630000, model_configured: true}}};
  app.methodHandler = () => configured;
  const pending = deferred(); app.handler = () => pending.promise;
  const running = app.run(); await app.flush();
  assert.ok([...app.timers.values()].some(timer => timer.milliseconds === 630000));
  assert.ok(![...app.timers.values()].some(timer => timer.milliseconds === 120000));
  app.timeout(120000);
  assert.equal(app.elements.batchSettings.disabled, true);
  assert.deepEqual(Object.keys(app.requests[0]).sort(), ['expected_method_digest', 'method', 'scenes', 'targets']);
  await app.elements.stopBatch.fire('click');
  pending.resolve({...responseFor(app.requests[0]), method: configured}); await running;
  assert.equal((await app.report()).method.options.request_policy.benchmark_timeout_ms, 630000);
});

test('a configured batch deadline rejects late response data and submits no further jobs', async () => {
  const configuredMethod = {...method, options: {request_policy: {studio_timeout_ms: 630000, benchmark_timeout_ms: 630000, model_configured: true}}};
  const app = application({configuredMethod}); await app.flush(); await app.addSamples();
  const pending = deferred(); app.handler = () => pending.promise;
  const running = app.run(); await app.flush();
  app.timeout(630000); pending.resolve({...responseFor(app.requests[0]), method: configuredMethod}); await running;
  assert.equal(app.requests.length, 1);
  const report = await app.report();
  assert.equal(report.status, 'interrupted');
  assert.equal(report.results.length, 0);
  assert.match(app.elements.batchError.textContent, /Job timed out/);
});

for (const policy of [null,
  {studio_timeout_ms: 60000, benchmark_timeout_ms: 0, model_configured: false},
  {studio_timeout_ms: 60000, benchmark_timeout_ms: true, model_configured: false},
  {studio_timeout_ms: 60000, benchmark_timeout_ms: 120000.5, model_configured: false},
  {studio_timeout_ms: 60000, benchmark_timeout_ms: 1500001, model_configured: true},
  {studio_timeout_ms: 60000, benchmark_timeout_ms: 120000, model_configured: 1},
]) test(`malformed batch method policy rejects before POST: ${JSON.stringify(policy)}`, async () => {
  const app = application(); await app.flush(); await app.addSamples();
  app.methodHandler = () => ({...method, options: {request_policy: policy}});
  await app.run();
  assert.equal(app.requests.length, 0);
  assert.match(app.elements.batchError.textContent, /invalid resize runtime policy/);
  assert.equal(app.elements.batchSettings.disabled, false);
  assert.equal(app.elements.retryConnection.hidden, false);
});

test('a late batch metadata response after 10 seconds cannot start an expensive job', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  const pending = deferred(); app.methodHandler = () => pending.promise;
  const running = app.run(); await app.flush();
  assert.ok([...app.timers.values()].some(timer => timer.milliseconds === 10000));
  app.timeout(10000); pending.resolve(method); await running;
  assert.equal(app.requests.length, 0);
  assert.match(app.elements.batchError.textContent, /metadata request timed out after 10 seconds/);
  assert.equal(app.elements.batchSettings.disabled, false);
});

test('completed exports grow linearly and completed DOM stays alive as later jobs finish', async () => {
  const app = application(); await app.flush();
  const task = {...sample('portable').task, assets: [{asset_id: 'exact.png', data_base64: 'ZXhhY3Q='}],
    fonts: [{family: 'Owned', data_base64: 'Zm9udA=='}], document: {snapshot_id: 'original'}};
  await app.import([['portable.json', JSON.stringify(task)]]);
  app.elements.targetSizes.value = '100x100\n200x100\n300x100\n400x100';
  const second = deferred(), third = deferred();
  app.handler = body => app.requests.length === 2 ? second.promise : app.requests.length === 3 ? third.promise : responseFor(body);
  const running = app.run(); await app.flush();
  assert.equal(app.requests.length, 2);
  assert.equal(app.blobCreations.length, 2);
  const walk = element => [element, ...element.children.flatMap(walk)];
  const firstDetails = walk(app.elements.batchResults).find(element => element.tagName === 'DETAILS');
  firstDetails.open = true;
  const firstSource = walk(app.elements.batchResults).find(element => element.tagName === 'IMG');
  const firstLinks = app.elements.batchResults.querySelectorAll('a[download]');
  second.resolve(responseFor(app.requests[1])); await app.flush();
  assert.equal(app.requests.length, 3);
  assert.equal(app.blobCreations.length, 4, 'Each admitted job owns one JSON/PNG pair, with no reserialization of prior jobs');
  assert.ok(walk(app.elements.batchResults).includes(firstDetails));
  assert.equal(firstDetails.open, true);
  assert.ok(walk(app.elements.batchResults).includes(firstSource));
  assert.ok(firstLinks.every(link => !app.revoked.includes(link.href)));
  third.resolve(responseFor(app.requests[2])); await running;
  assert.equal(app.blobCreations.length, 9, 'Four pairs plus one final report');
  assert.equal(app.revoked.length, 0);
  assert.ok(app.requests.every(request => JSON.stringify(request.scenes[0].payload) === JSON.stringify(task)));
});

test('duplicate two-level uses share exactly one export pair per verified job', async () => {
  const app = application(); await app.flush();
  await app.import([['same-size.json', JSON.stringify(source())]]);
  app.elements.comparisonMode.value = 'two-level'; app.elements.targetSizes.value = '50x50\n25x25';
  await app.run();
  assert.equal(app.requests.length, 2);
  const links = app.elements.batchResults.querySelectorAll('a[download]');
  const png = links.filter(link => link.textContent === 'PNG ↓'), json = links.filter(link => link.textContent === 'JSON ↓');
  assert.equal(png.length, 4); assert.equal(json.length, 4);
  assert.equal(new Set(png.map(link => link.href)).size, 2);
  assert.equal(new Set(json.map(link => link.href)).size, 2);
  assert.equal(app.blobCreations.length, 5);
  assert.equal(app.revoked.length, 0);
});

test('the 16-job matrix owns 16 verified export pairs and one final report', async () => {
  const app = application(); await app.flush();
  const tasks = Array.from({length: 4}, (_, index) => ({source: source(),
    assets: [{asset_id: `asset-${index}.png`, data_base64: `exact-${index}`}],
    fonts: [{family: `Font ${index}`, data_base64: `font-${index}`}], document: {snapshot_id: `source-${index}`}}));
  await app.import(tasks.map((task, index) => [`source-${index}.json`, JSON.stringify(task)]));
  app.elements.targetSizes.value = '80x50\n50x80\n60x60\n90x60'; await app.run();
  assert.equal(app.requests.length, 16);
  assert.equal(app.blobCreations.length, 33);
  assert.equal(app.blobCreations.filter(item => item.type === 'image/png').length, 16);
  assert.equal(app.revoked.length, 0);
  const links = app.elements.batchResults.querySelectorAll('a[download]').filter(link => link.textContent === 'JSON ↓');
  assert.equal(links.length, 16);
  for (let index = 0; index < links.length; index++) {
    const exported = JSON.parse(await app.blobs.get(links[index].href).text());
    const original = tasks[Math.floor(index / 4)];
    assert.deepEqual(exported.assets, original.assets);
    assert.deepEqual(exported.fonts, original.fonts);
    assert.deepEqual(exported.document, original.document);
    assert.deepEqual(app.requests[index].scenes[0].payload, original);
  }
});

test('dirty cleanup revokes each URL once and a late status paint cannot restore exports', async () => {
  const app = application(); await app.flush();
  await app.import([['one.json', JSON.stringify(source())]]);
  const pending = deferred();
  app.handler = body => app.requests.length === 2 ? pending.promise : responseFor(body);
  const running = app.run(); await app.flush();
  assert.equal(app.blobCreations.length, 2);
  const owned = [...app.blobs.keys()];
  app.elements.targetSizes.value = '200x200'; await app.elements.targetSizes.fire('input');
  await app.elements.targetSizes.fire('input');
  assert.deepEqual(app.revoked, owned);
  pending.resolve(responseFor(app.requests[1])); await running;
  assert.equal(app.blobCreations.length, 2);
  assert.equal(app.elements.downloadReport.href, '');
  const walk = element => [element, ...element.children.flatMap(walk)];
  const links = walk(app.elements.batchResults).filter(element => element.tagName === 'A');
  assert.equal(links.length, 4);
  assert.ok(links.every(link => !link.href && link.getAttribute('aria-disabled') === 'true'));
  assert.deepEqual(app.revoked, owned);
});

test('a new report drops old cached nodes and URLs even when scene IDs and targets repeat', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  app.elements.targetSizes.value = '100x100'; await app.run();
  const oldArticles = [...app.elements.batchResults.children], oldUrls = [...app.blobs.keys()];
  assert.equal(app.blobCreations.length, 5);
  app.handler = body => {
    const result = responseFor(body);
    result.results[0].result.output_png = 'data:image/png;base64,bmV3LW5hdGl2ZS1ieXRlcw==';
    return result;
  };
  await app.run();
  assert.equal(app.blobCreations.length, 10);
  assert.deepEqual(app.revoked, oldUrls);
  assert.ok(app.elements.batchResults.children.every(article => !oldArticles.includes(article)));
  const png = app.elements.batchResults.querySelectorAll('a[download]').filter(link => link.textContent === 'PNG ↓');
  for (const link of png) {
    assert.ok(!oldUrls.includes(link.href));
    assert.equal(Buffer.from(await app.blobs.get(link.href).arrayBuffer()).toString(), 'new-native-bytes');
  }
});

test('source placeholders update independently after the first verified row and a stopped run', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  const pending = deferred(); app.handler = () => pending.promise;
  const running = app.run(); await app.flush();
  const stages = app.elements.batchResults.children.map(article => article.children[1].children[0].children[1]);
  assert.ok(stages.every(stage => stage.children[0].textContent === 'Original render pending'));
  await app.elements.stopBatch.fire('click');
  pending.resolve(responseFor(app.requests[0])); await running;
  assert.equal(stages[0].children[0].tagName, 'IMG');
  assert.equal(stages[0].children[0].src, 'data:image/png;base64,c291cmNl');
  assert.equal(stages[1].children[0].textContent, 'Original render unavailable');
  assert.equal(app.blobCreations.length, 3);
});

test('leaving an uncached page cannot create late export URLs after cleanup', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  const pending = deferred(); app.handler = () => pending.promise;
  const running = app.run(); await app.flush();
  app.windowEvent('pagehide', {persisted: false});
  pending.resolve(responseFor(app.requests[0])); await running;
  assert.equal(app.requests.length, 1);
  assert.equal(app.blobCreations.length, 0);
  assert.equal(app.elements.downloadReport.href, '');
});

test('reviewed picker excludes reserved/unprepared posters and loads exact original bundles', async () => {
  const task = {...sample('poster').task, assets: [{asset_id: 'exact.png', data_base64: 'eGFjdA=='}], fonts: [], document: {snapshot_id: 'original'}};
  const app = application({posterBundle: () => task}); await app.flush();
  await app.browseReviewed();
  assert.deepEqual(app.elements.reviewedPosters.children.map(option => option.value), ['reviewed-one', 'reviewed-two']);
  assert.equal(app.elements.reviewedPicker.hidden, false);
  await app.addReviewed(['reviewed-two']);
  app.elements.targetSizes.value = '300x200'; await app.run();
  assert.deepEqual(app.posterRequests, ['reviewed-two']);
  assert.equal(JSON.stringify(app.requests[0].scenes[0].payload), JSON.stringify(task));
  assert.equal(app.requests[0].scenes[0].family, 'unassigned');
  assert.equal(app.requests[0].scenes[0].library_poster_id, 'reviewed-two');
  const count = app.posterRequests.length;
  await app.addReviewed(['reserved']);
  assert.equal(app.posterRequests.length, count);
  assert.match(app.elements.batchError.textContent, /Only reviewed/);
});

test('reviewed multi-add is atomic when a later bundle is invalid', async () => {
  const app = application({posterBundle: id => id === 'reviewed-two' ? {} : sample(id).task});
  await app.flush(); await app.addSamples();
  app.elements.targetSizes.value = '100x100'; await app.run();
  const report = app.elements.downloadReport.href;
  await app.browseReviewed(); await app.addReviewed(['reviewed-one', 'reviewed-two']);
  assert.equal(app.elements.sceneList.children.length, 2);
  assert.equal(app.elements.downloadReport.href, report);
  assert.equal(app.elements.batchSettings.disabled, false);
  assert.equal(app.elements.batchError.hidden, false);
});

test('reviewed pack adds at most four and repeated additions never duplicate scene IDs', async () => {
  const posters = Array.from({length: 6}, (_, index) => ({id: `poster-${index}`, title: `Poster ${index}`, split: 'reviewed', ready: true}));
  const app = application({posters}); await app.flush(); await app.browseReviewed();
  await app.elements.addReviewedPack.fire('click');
  assert.equal(app.elements.sceneList.children.length, 4);
  assert.equal(app.posterRequests.length, 4);
  await app.addReviewed(['poster-0']);
  assert.equal(app.posterRequests.length, 4);
  assert.equal(app.elements.sceneList.children.length, 4);
  await app.addReviewed(posters.slice(0, 5).map(poster => poster.id));
  assert.match(app.elements.batchError.textContent, /between 1 and 4/);
  assert.equal(app.elements.sceneList.children.length, 4);
});

test('two-level plan uses per-original rounded targets and never cascades resized output', async () => {
  const app = application(); await app.flush();
  const first = source(), second = source();
  first.canvas = {width: 101, height: 199}; second.canvas = {width: 200, height: 100};
  await app.import([['first.json', JSON.stringify(first)], ['second.json', JSON.stringify(second)]]);
  await app.elements.twoLevelPreset.fire('click'); await app.run();
  assert.deepEqual(app.requests.map(request => request.targets[0]), [
    {width: 51, height: 100}, {width: 25, height: 50}, {width: 800, height: 500}, {width: 500, height: 800},
    {width: 100, height: 50}, {width: 50, height: 25}, {width: 800, height: 500}, {width: 500, height: 800},
  ]);
  assert.ok(app.requests.slice(0, 4).every(request => JSON.stringify(request.scenes[0].payload.source) === JSON.stringify(first)));
  assert.ok(app.requests.slice(4).every(request => JSON.stringify(request.scenes[0].payload.source) === JSON.stringify(second)));
  assert.ok(app.requests.every(request => request.expected_method_digest === method.digest));
  const report = await app.report();
  assert.equal(report.schema_version, 'scene-benchmark-ui-v2');
  assert.equal(report.comparison_mode, 'two-level');
  assert.equal(report.job_plan.length, 8);
  assert.equal(report.results.length, 8);
  assert.deepEqual(report.results[0].comparisons, [{level: 'resolution', requested_scale: 0.5}]);
  assert.equal(report.results[0].job_key, report.job_plan[0].job_key);
  assert.equal(report.job_plan[2].comparisons[0].level, 'shared-aspect');
  assert.match(app.elements.batchStatus.textContent, /same method and recorded two-level plan/);
  const text = element => [element.textContent, ...element.children.flatMap(child => text(child))].join(' ');
  assert.match(text(app.elements.batchResults), /Level 1 · Proportional resolution/);
  assert.match(text(app.elements.batchResults), /Level 2 · Shared canvas sizes/);
  assert.match(text(app.elements.batchResults), /Engine layout: not reported/);
});

test('rounded proportional duplicates and shared-size overlap reuse one verified job', async () => {
  const app = application(); await app.flush();
  const tiny = source(); tiny.canvas = {width: 1, height: 1};
  await app.import([['tiny.json', JSON.stringify(tiny)]]);
  app.elements.comparisonMode.value = 'two-level';
  app.elements.targetSizes.value = '1x1\n2x1'; await app.run();
  assert.equal(app.requests.length, 2);
  const report = await app.report();
  assert.equal(report.job_plan[0].comparisons.length, 3);
  assert.deepEqual(report.job_plan[0].comparisons, [
    {level: 'resolution', requested_scale: 0.5}, {level: 'resolution', requested_scale: 0.25}, {level: 'shared-aspect'},
  ]);
  assert.equal(report.results.length, 2);
  assert.equal(app.elements.batchProgress.max, 2);
});

test('two-level job and shared-target limits reject before any engine job', async () => {
  const app = application(); await app.flush();
  await app.import(Array.from({length: 5}, (_, index) => [`scene-${index}.json`, JSON.stringify(source())]));
  await app.elements.twoLevelPreset.fire('click'); await app.run();
  assert.equal(app.requests.length, 0);
  assert.match(app.elements.batchError.textContent, /16 jobs/);
  app.elements.targetSizes.value = '80x50\n50x80\n60x60'; await app.run();
  assert.equal(app.requests.length, 0);
  assert.match(app.elements.batchError.textContent, /up to 2 shared targets/);
});

test('changing comparison mode invalidates exports and shared-only reports remain v1', async () => {
  const app = application(); await app.flush(); await app.addSamples(); await app.run();
  assert.equal((await app.report()).schema_version, 'scene-benchmark-ui-v1');
  const links = app.elements.batchResults.querySelectorAll('a[download]');
  app.elements.comparisonMode.value = 'two-level'; await app.elements.comparisonMode.fire('change');
  assert.equal(app.elements.downloadReport.href, '');
  assert.ok(links.every(link => !link.href));
  assert.match(app.elements.runIdentity.textContent, /previous run/);
});

test('two-level cells expose actual engine topology and small text without modifying reports', async () => {
  const app = application(); await app.flush();
  const original = source();
  original.elements = [{id: 'label', type: 'text', x: 0, y: 0, width: 100, height: 50, font_size: 16, content: 'Same source'}];
  await app.import([['type.json', JSON.stringify(original)]]);
  await app.elements.twoLevelPreset.fire('click');
  app.handler = body => {
    const response = responseFor(body);
    response.results[0].result.design = {...body.scenes[0].payload.source, canvas: body.targets[0],
      elements: body.scenes[0].payload.source.elements.map(element => ({...element, font_size: 8}))};
    response.results[0].result.report.layout_state = {name: 'source-guide', topology: 'rows'};
    return response;
  };
  await app.run();
  const text = element => [element.textContent, ...element.children.flatMap(child => text(child))].join(' ');
  assert.match(text(app.elements.batchResults), /below 10 px \(smallest 8.0 px\)/);
  assert.match(text(app.elements.batchResults), /Engine layout: source-guide · rows/);
  const report = await app.report();
  assert.deepEqual(report.results[0].result.report, {violations: [], layout_state: {name: 'source-guide', topology: 'rows'}});
  assert.equal(report.results[0].comparisons[0].requested_scale, 0.5);
});

test('reviewed picker is a labeled multi-select hidden until catalogue loading succeeds', () => {
  assert.match(markup, /id="reviewedPicker"[^>]*hidden/);
  assert.match(markup, /<label for="reviewedPosters">/);
  assert.match(markup, /<select id="reviewedPosters"[^>]*multiple/);
  assert.match(markup, /id="reviewedHelp"/);
});

test('stop waits for the current job and exports completed work without submitting another', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  const pending = deferred(); app.handler = () => pending.promise;
  const running = app.run(); await app.flush();
  assert.equal(app.requests.length, 1);
  await app.elements.stopBatch.fire('click');
  assert.equal(app.elements.stopBatch.disabled, true);
  pending.resolve(responseFor(app.requests[0])); await running;
  assert.equal(app.requests.length, 1);
  assert.equal(app.elements.batchSettings.disabled, false);
  const report = await app.report();
  assert.equal(report.status, 'stopped');
  assert.equal(report.results.length, 1);
  assert.match(app.elements.batchStatus.textContent, /Stopped after 1 job\./);
});

test('changing targets invalidates report and per-result exports while marking previous results', async () => {
  const app = application(); await app.flush(); await app.addSamples(); await app.run();
  const links = app.elements.batchResults.querySelectorAll('a[download]');
  assert.ok(links.length > 0 && links.every(link => link.href));
  app.elements.targetSizes.value = '200x200'; await app.elements.targetSizes.fire('input');
  assert.equal(app.elements.downloadReport.href, '');
  assert.equal(app.elements.downloadReport.getAttribute('aria-disabled'), 'true');
  assert.ok(links.every(link => !link.href && link.getAttribute('aria-disabled') === 'true'));
  assert.match(app.elements.runIdentity.textContent, /previous run; setup changed/);
  assert.match(app.elements.batchStatus.textContent, /previous run/);
});

test('a multi-file import is atomic when a later JSON file is invalid', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  await app.import([['valid.json', JSON.stringify(sample('extra').task)], ['bad.json', '{']]);
  assert.equal(app.elements.sceneList.children.length, 2);
  assert.equal(app.elements.batchError.hidden, false);
  assert.equal(app.elements.batchSettings.disabled, false);
  assert.equal(app.elements.sceneFiles.value, '');
  await app.run();
  assert.deepEqual([...new Set(app.requests.map(job => job.scenes[0].id))], ['one', 'two']);
});

test('bundle JSON exports preserve the exact assets, fonts and document with resized native elements', async () => {
  const app = application(); await app.flush();
  const task = {source: source(), assets: [{asset_id: 'logo.png', data_base64: 'bG9nbw=='}],
    fonts: [{family: 'Owned font', data_base64: 'Zm9udA=='}], document: {snapshot_id: 'owned'}};
  app.elements.targetSizes.value = '300x200';
  await app.import([['owned.json', JSON.stringify(task)]]); await app.run();
  const link = app.elements.batchResults.querySelectorAll('a[download]').find(item => item.textContent === 'JSON ↓');
  const output = JSON.parse(await app.blobs.get(link.href).text());
  assert.deepEqual(output.assets, task.assets);
  assert.deepEqual(output.fonts, task.fonts);
  assert.deepEqual(output.document, task.document);
  assert.deepEqual(output.source.canvas, {width: 300, height: 200});
  assert.deepEqual(output.target_canvas, {width: 300, height: 200});
});

test('PNG exports use downloadable image blobs containing the exact decoded render bytes', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  // Include zero and high bytes to detect accidental text encoding of binary.
  const bytes = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10, 0, 128, 255]);
  app.handler = body => {
    const response = responseFor(body);
    response.results[0].result.output_png = `data:image/png;base64,${bytes.toString('base64')}`;
    return response;
  };
  await app.run();
  const links = app.elements.batchResults.querySelectorAll('a[download]')
    .filter(item => item.textContent === 'PNG ↓');
  assert.equal(links.length, 4);
  for (const link of links) {
    assert.match(link.href, /^blob:/);
    assert.match(link.download, /-(100x100|300x200)\.png$/);
    const blob = app.blobs.get(link.href);
    assert.equal(blob.type, 'image/png');
    assert.deepEqual(Buffer.from(await blob.arrayBuffer()), bytes);
  }
});

test('an inconsistent scene identity interrupts the run before a result is admitted', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  app.handler = body => {
    const response = responseFor(body); response.results[0].scene_id = 'unrelated'; return response;
  };
  await app.run();
  assert.equal(app.requests.length, 1);
  const report = await app.report();
  assert.equal(report.status, 'interrupted');
  assert.equal(report.results.length, 0);
  assert.equal(app.elements.batchError.hidden, false);
});

test('a later mismatched target preserves earlier completed work in an interrupted report', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  let count = 0;
  app.handler = body => {
    const response = responseFor(body);
    if (++count === 2) response.results[0].target_canvas.width++;
    return response;
  };
  await app.run();
  assert.equal(app.requests.length, 2);
  const report = await app.report();
  assert.equal(report.status, 'interrupted');
  assert.equal(report.results.length, 1);
  assert.deepEqual(report.results[0].target_canvas, {width: 100, height: 100});
  assert.equal(app.elements.batchSettings.disabled, false);
  assert.equal(app.elements.downloadReport.getAttribute('aria-disabled'), 'false');
  assert.equal(app.blobCreations.length, 3, 'The earlier verified pair survives; the invalid attempt owns no exports');
});

test('a failed startup method can reconnect without losing the queued scenes', async () => {
  const app = application({failInitialMethod: true}); await app.flush(); await app.addSamples();
  assert.equal(app.elements.runBatch.disabled, true);
  assert.ok(app.elements.retryConnection, 'Startup failure must offer an in-page reconnect action');
  await app.elements.retryConnection.fire('click');
  assert.equal(app.elements.runBatch.disabled, false);
  assert.equal(app.elements.sceneList.children.length, 2);
  await app.run();
  assert.equal((await app.report()).status, 'completed');
});

test('a partial success response cannot count as a rendered or exportable result', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  app.handler = body => { const response = responseFor(body); response.results[0].result = null; return response; };
  await app.run();
  const report = await app.report();
  assert.equal(report.status, 'interrupted');
  assert.equal(report.results.length, 0);
  assert.equal(app.elements.batchSettings.disabled, false);
  assert.equal(app.blobCreations.length, 1, 'Only the interrupted report is exportable');
  assert.equal(app.blobCreations[0].type, 'application/json');
});

test('a malformed PNG response does not poison recovery or erase completed jobs', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  let calls = 0;
  app.handler = body => { const response = responseFor(body); if (++calls === 2) response.results[0].result.output_png = 'broken'; return response; };
  await assert.doesNotReject(app.run());
  const report = await app.report();
  assert.equal(report.status, 'interrupted');
  assert.equal(report.results.length, 1);
  assert.equal(app.elements.batchSettings.disabled, false);
});

test('JSON result geometry must match the submitted target before admission', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  app.handler = body => { const response = responseFor(body); response.results[0].result.design.canvas.width++; return response; };
  await app.run();
  assert.equal((await app.report()).results.length, 0);
  assert.equal((await app.report()).status, 'interrupted');
});

test('browser history caching retains live export blobs', async () => {
  const app = application(); await app.flush(); await app.addSamples(); await app.run();
  app.revoked.length = 0;
  app.windowEvent('pagehide', {persisted: true});
  assert.equal(app.revoked.length, 0);
  app.windowEvent('pagehide', {persisted: false});
  assert.ok(app.revoked.length > 0);
});

test('unavailable preference storage cannot prevent local testing', async () => {
  let app;
  assert.doesNotThrow(() => { app = application({storageFailure: true}); });
  await app.flush(); await app.addSamples(); await app.run();
  assert.equal((await app.report()).status, 'completed');
});

for (const settings of [{decodeFailure: true}, {wrongImageSize: true}]) {
  test(`unverified browser image decoding cannot count a rendered result: ${JSON.stringify(settings)}`, async () => {
    const app = application(settings); await app.flush(); await app.addSamples(); await app.run();
    assert.equal((await app.report()).results.length, 0);
    assert.equal((await app.report()).status, 'interrupted');
    assert.equal(app.elements.batchSettings.disabled, false);
  });
}

test('a stalled preview decode has bounded recovery and releases controls', async () => {
  const app = application({decodeStall: true}); await app.flush(); await app.addSamples();
  const pending = app.run(); await app.flush();
  app.timeout(30000); await pending;
  assert.equal((await app.report()).status, 'interrupted');
  assert.equal(app.elements.batchSettings.disabled, false);
  assert.equal(app.timers.size, 0);
});

test('leaving a cached page stops queued jobs after the active one', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  const pending = deferred(); app.handler = () => pending.promise;
  const running = app.run(); await app.flush();
  app.windowEvent('pagehide', {persisted: true});
  pending.resolve(responseFor(app.requests[0])); await running;
  assert.equal(app.requests.length, 1);
  assert.equal((await app.report()).status, 'stopped');
});

test('invalid source geometry is rejected atomically before it enters the queue', async () => {
  const app = application(); await app.flush();
  const invalid = source(); invalid.canvas.width = 100000000;
  await app.import([['valid.json', JSON.stringify(source())], ['invalid.json', JSON.stringify(invalid)]]);
  assert.equal(app.elements.sceneList.children.length, 0);
  assert.equal(app.elements.runBatch.disabled, true);
  assert.match(app.elements.batchError.textContent, /dimensions/);
});

test('seeded repeated target edits, runs and exports retain exact matrix identity', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  let seed = 23479;
  for (let round = 0; round < 40; round++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const width = 100 + seed % 450, height = 100 + (seed >>> 12) % 450;
    app.elements.targetSizes.value = `${width}x${height}`;
    await app.elements.targetSizes.fire('input');
    assert.equal(app.elements.downloadReport.href, '');
    await app.run();
    const report = await app.report();
    assert.equal(report.status, 'completed');
    assert.equal(report.results.length, 2);
    assert.ok(report.results.every(row => row.target_canvas.width === width && row.result.design.canvas.height === height));
    const links = app.elements.batchResults.querySelectorAll('a[download]');
    assert.ok(links.every(link => link.href && !app.revoked.includes(link.href)));
    assert.equal(app.timers.size, 0);
  }
  assert.equal(app.requests.length, 80);
});

test('a stalled raster import releases controls after a bounded decode timeout', async () => {
  const app = application({bitmapStall: true}); await app.flush();
  const importing = app.importRaster(); await app.flush();
  assert.equal(app.bitmapCalled, true);
  assert.ok([...app.timers.values()].some(timer => timer.milliseconds === 30000), 'Raster decoding must have a deadline');
  app.timeout(30000); await importing;
  assert.equal(app.elements.batchSettings.disabled, false);
  assert.equal(app.elements.sceneList.children.length, 0);
  assert.match(app.elements.batchError.textContent, /timed out/i);
});

test('a valid-looking response with broken element geometry is never exported', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  app.handler = body => {
    const response = responseFor(body);
    response.results[0].result.design.elements = response.results[0].result.design.elements.map(el => ({...el, width: null}));
    return response;
  };
  await app.run();
  assert.equal((await app.report()).results.length, 0);
  assert.equal((await app.report()).status, 'interrupted');
});

test('out-of-limit element geometry is rejected before joining the source queue', async () => {
  const app = application(); await app.flush();
  const invalid = source(); invalid.elements[0].width = 100000000;
  await app.import([['invalid.json', JSON.stringify(invalid)]]);
  assert.equal(app.elements.sceneList.children.length, 0);
  assert.equal(app.elements.runBatch.disabled, true);
});

test('invalid target input reports the input problem without suggesting reconnect', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  app.elements.targetSizes.value = '120x80\n120x80';
  await app.run();
  assert.equal(app.requests.length, 0);
  assert.equal(app.elements.retryConnection.hidden, true);
  assert.match(app.elements.batchError.textContent, /Duplicate target/);
});

test('a truncated JSON response preserves earlier jobs and gives an actionable error', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  let calls = 0;
  app.handler = body => { if (++calls === 2) throw new SyntaxError('Unexpected end of JSON input'); return responseFor(body); };
  await app.run();
  const report = await app.report();
  assert.equal(report.results.length, 1);
  assert.equal(report.status, 'interrupted');
  assert.match(app.elements.batchError.textContent, /invalid JSON.*retry/i);
  assert.match(app.elements.batchStatus.textContent, /interrupted.*1 completed job preserved/i);
  assert.equal(report.interrupted_job.status, 'unverified');
  assert.deepEqual(report.interrupted_job.target_canvas, {width: 300, height: 200});
  assert.match(report.interrupted_job.error, /could not be verified.*invalid JSON/i);
});
