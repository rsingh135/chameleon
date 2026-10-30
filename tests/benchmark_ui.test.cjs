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
  const app = {requests: [], blobs: new Map(), revoked: [], images: new Map(), timers: new Map(), handler: body => responseFor(body)};
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
    URL: LocalURL, Blob, AbortController, Date, Uint8Array, atob, btoa,
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
        return {ok: true, json: async () => method};
      }
      if (url === '/api/samples') return {ok: true, json: async () => ({samples: [sample('one'), sample('two', {width: 555, height: 777})]})};
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
