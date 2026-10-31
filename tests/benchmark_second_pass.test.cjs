/* Read-only second-pass orchestration checks using the established DOM fixture.
 * The production script is unmodified; only fixture API responses are varied.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const fixturePath = path.join(__dirname, 'benchmark_ui.test.cjs');
const existing = fs.readFileSync(fixturePath, 'utf8');
const marker = existing.indexOf("\ntest('every scene uses");
assert.ok(marker > 0, 'Update second-pass fixture extraction after reorganizing the shared fixture');
let fixture = existing.slice(0, marker);
const sampleRoute = "if (url === '/api/samples') return {ok: true, json: async () => ({samples: [sample('one'), sample('two', {width: 555, height: 777})]})};";
assert.ok(fixture.includes(sampleRoute), 'Update second-pass sample-response hook');
fixture = fixture.replace(sampleRoute, "if (url === '/api/samples') return {ok: true, json: async () => ({samples: settings.samplesProvider ? await settings.samplesProvider() : [sample('one'), sample('two')]})};");
const methodRoute = 'return {ok: true, json: async () => method};';
assert.ok(fixture.includes(methodRoute), 'Update second-pass method-response hook');
fixture = fixture.replace(methodRoute, 'const reply = settings.methodProvider ? await settings.methodProvider() : method; return {ok: true, json: async () => reply};');
const exported = { exports: {} };
vm.runInNewContext(fixture + '\nmodule.exports = {application, source, sample, method, responseFor};', {
  require, module: exported, __dirname, URL, Blob, AbortController, Date, Uint8Array,
  atob, btoa, ArrayBuffer, setImmediate,
}, { filename: 'benchmark-second-pass-fixture.cjs' });
const { application, source, sample, method, responseFor } = exported.exports;
const remove = async (app, index) => app.elements.sceneList.children[index].children[0].children[1].fire('click');
const jsonExports = app => app.elements.batchResults.querySelectorAll('a[download]').filter(link => link.textContent === 'JSON ↓');

test('removing queued scenes updates the ready count before the first run', async () => {
  const app = application(); await app.flush(); await app.addSamples();
  await remove(app, 0);
  assert.equal(app.elements.sceneList.children.length, 1);
  assert.match(app.elements.batchStatus.textContent, /1 scene ready/);
  await remove(app, 0);
  assert.equal(app.elements.sceneList.children.length, 0);
  assert.match(app.elements.batchStatus.textContent, /No scenes queued/);
  assert.equal(app.elements.runBatch.disabled, true);
});

test('catalogue addition is atomic when a later bundled sample is malformed', async () => {
  const valid = sample('catalogue-one'), invalid = sample('catalogue-two');
  invalid.task.source.canvas.width = 1000000;
  const app = application({ samplesProvider: () => [valid, invalid] });
  await app.flush();
  await app.import([['existing.json', JSON.stringify(source())]]);
  app.elements.targetSizes.value = '100x100';
  await app.run();
  const oldReport = app.elements.downloadReport.href;
  await app.addSamples();
  assert.equal(app.elements.batchError.hidden, false);
  assert.equal(app.elements.sceneList.children.length, 1);
  assert.equal(app.elements.downloadReport.href, oldReport);
  const requestCount = app.requests.length;
  await app.run();
  assert.deepEqual(Array.from(app.requests.slice(requestCount), job => job.scenes[0].id), ['upload-1'],
    'A rejected catalogue batch must not add an invisible earlier scene to later runs');
});

test('duplicate catalogue IDs cannot produce ambiguous same-ID scene rows', async () => {
  const first = sample('same'), second = sample('same');
  second.task.source.elements[0].id = 'other-paper';
  const app = application({ samplesProvider: () => [first, second] });
  await app.flush(); await app.addSamples();
  assert.equal(app.elements.sceneList.children.length, 0);
  assert.equal(app.elements.batchError.hidden, false);
  assert.equal(app.elements.runBatch.disabled, true);
});

for (const [field, value] of [['id', undefined], ['id', ''], ['label', { invalid: true }]]) {
  test(`invalid catalogue ${field} is rejected before changing the queued scenes: ${JSON.stringify(value)}`, async () => {
    const entry = sample('invalid-metadata'); entry[field] = value;
    const app = application({ samplesProvider: () => [entry] });
    await app.flush(); await app.addSamples();
    assert.equal(app.elements.sceneList.children.length, 0);
    assert.equal(app.elements.batchError.hidden, false);
    assert.equal(app.elements.runBatch.disabled, true);
  });
}

test('a failed first catalogue scene preserves the existing queued input and report', async () => {
  const app = application({ samplesProvider: () => [{ id: 'broken', label: 'Broken', task: {} }] });
  await app.flush(); await app.import([['existing.json', JSON.stringify(source())]]);
  app.elements.targetSizes.value = '100x100'; await app.run();
  const report = app.elements.downloadReport.href;
  await app.addSamples();
  assert.equal(app.elements.sceneList.children.length, 1);
  assert.equal(app.elements.downloadReport.href, report);
  assert.equal(app.elements.batchSettings.disabled, false);
});

test('removing and readding an ID invalidates old exports and uses its new exact resources', async () => {
  const old = sample('same'), replacement = sample('same');
  old.task.assets = [{ asset_id: 'old.png', data_base64: 'b2xk' }];
  replacement.task.assets = [{ asset_id: 'new.png', data_base64: 'bmV3' }];
  let current = old;
  const app = application({ samplesProvider: () => [current] });
  await app.flush(); await app.addSamples();
  app.elements.targetSizes.value = '100x100'; await app.run();
  const oldLinks = jsonExports(app);
  await remove(app, 0);
  current = replacement; await app.addSamples();
  assert.equal(app.elements.downloadReport.href, '');
  assert.ok(oldLinks.every(link => !link.href));
  assert.match(app.elements.runIdentity.textContent, /previous run/);
  await app.run();
  assert.equal(JSON.stringify(app.requests.at(-1).scenes[0].payload.assets), JSON.stringify(replacement.task.assets));
  const output = JSON.parse(await app.blobs.get(jsonExports(app)[0].href).text());
  assert.equal(JSON.stringify(output.assets), JSON.stringify(replacement.task.assets));
  assert.equal((await app.report()).results.length, 1);
});

test('reconnecting to a changed method invalidates old report exports and freezes the next run', async () => {
  let current = method, offline = false;
  const app = application({ methodProvider: () => {
    if (offline) throw new TypeError('Offline');
    return current;
  } });
  await app.flush(); await app.addSamples();
  app.elements.targetSizes.value = '100x100'; await app.run();
  const oldLinks = jsonExports(app);
  offline = true; await app.run();
  assert.equal(app.elements.retryConnection.hidden, false);
  assert.equal((await app.report()).status, 'completed'); // Previous completed evidence survives.
  offline = false; current = { ...method, digest: 'd'.repeat(64) };
  await app.elements.retryConnection.fire('click');
  assert.equal(app.elements.sceneList.children.length, 2);
  assert.equal(app.elements.downloadReport.href, '');
  assert.ok(oldLinks.every(link => !link.href));
  assert.match(app.elements.runIdentity.textContent, /previous run/);
  app.handler = body => ({ ...responseFor(body), method: current });
  await app.run();
  const report = await app.report();
  assert.equal(report.status, 'completed');
  assert.equal(report.method.digest, current.digest);
  assert.ok(app.requests.slice(-2).every(job => job.expected_method_digest === current.digest));
});
