// Executes the file inspector's real inline script; canvas calls are observed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const markup = fs.readFileSync(process.env.INSPECTOR_MARKUP || path.join(__dirname, '../viewer/inspector.html'), 'utf8');
const script = markup.match(/<script>([\s\S]*?)<\/script>/)[1];
function deferred() {
  let resolve; const promise = new Promise(yes => { resolve = yes; }); return { promise, resolve };
}
function design(id = 'paper', canvas = { width: 200, height: 200 }) {
  return { canvas, elements: [{ id, type: 'shape', x: 0, y: 0, width: 100, height: 100 }] };
}
function application() {
  const env = { images: [], revoked: [], draws: [], timers: new Map() };
  class Element {
    constructor(id) { this.id = id; this.textContent = ''; this.value = ''; this.checked = true; this.listeners = {}; this.children = []; }
    addEventListener(type, callback) { this.listeners[type] = callback; }
    fire(type) { return this.listeners[type]?.({ target: this }); }
    replaceChildren(...children) { this.children = children; this.value = children[0]?.value || ''; }
    getBoundingClientRect() { return { left: 0, top: 0, width: this.width, height: this.height }; }
    getContext() {
      const canvas = this;
      return {
        setTransform() {}, fillRect() {}, save() {}, restore() {}, translate() {}, rotate() {},
        beginPath() {}, ellipse() {}, fill() {}, strokeRect() {}, fillText() {},
        drawImage(image) { env.draws.push({ canvas: canvas.id, image }); },
        measureText(text) { return { width: text.length * 8 }; },
      };
    }
  }
  const ids = [...markup.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  const elements = Object.fromEntries([...ids, 'sourceElement', 'outputElement'].map(id => [id, new Element(id)]));
  elements.importError.hidden = true;
  class Image {
    constructor() { env.images.push(this); }
    set src(value) { this.url = value; }
  }
  let timer = 0;
  vm.runInNewContext(script, {
    document: { getElementById: id => elements[id] }, Image,
    URL: { createObjectURL: () => `blob:${env.images.length}`, revokeObjectURL: url => env.revoked.push(url) },
    Option: class { constructor(text, value) { this.textContent = text; this.value = value; } },
    setTimeout: callback => { env.timers.set(++timer, callback); return timer; },
    clearTimeout: id => env.timers.delete(id),
  });
  env.elements = elements;
  env.file = (id, value, size = 500) => {
    elements[id].value = `selected-${id}`;
    elements[id].files = [{ size, text: typeof value === 'function' ? value : async () => JSON.stringify(value) }];
    return elements[id].fire('change');
  };
  env.png = () => env.file('png', {});
  env.decode = (width, height, index = env.images.length - 1) => {
    const image = env.images[index]; image.naturalWidth = width; image.naturalHeight = height; image.onload();
  };
  return env;
}

test('oversized and duplicate-id designs fail without replacing the last valid source', async () => {
  const app = application(); await app.file('task', design());
  await app.file('task', design('huge', { width: 1000000, height: 2 }));
  assert.equal(app.elements.importError.hidden, false);
  assert.equal(app.elements.srcTitle.textContent, 'Source 200×200');
  const duplicate = design(); duplicate.elements.push({ ...duplicate.elements[0] });
  await app.file('task', duplicate);
  assert.match(app.elements.importError.textContent, /unique/);
  assert.equal(app.elements.srcTitle.textContent, 'Source 200×200');
});

test('invalid paint metadata and excessive files are rejected before drawing or reading', async () => {
  const app = application(); const invalid = design(); invalid.elements[0].opacity = 2;
  await app.file('task', invalid);
  assert.match(app.elements.importError.textContent, /opacity/);
  let read = false;
  await app.file('task', async () => { read = true; return '{}'; }, 25 * 1024 * 1024);
  assert.equal(read, false);
  assert.match(app.elements.importError.textContent, /24 MiB/);
  await app.file('task', () => Promise.resolve('{"canvas":{"width":200,"height":200},"elements":[{"id":"x","type":"shape","x":0,"y":0,"width":10,"height":10,"z_index":1e400}]}'));
  assert.match(app.elements.importError.textContent, /finite/);
});

test('portable output JSON exposes element properties through the keyboard selector', async () => {
  const app = application(); await app.file('task', design('original'));
  const output = design('</script><img src=x onerror=alert(1)>');
  await app.file('output', { source: output, assets: [] });
  assert.equal(app.elements.outputElement.disabled, false);
  app.elements.outputElement.value = output.elements[0].id;
  app.elements.outputElement.fire('change');
  assert.match(app.elements.inspector.textContent, /^Output:/);
  assert.match(app.elements.inspector.textContent, /onerror=alert/); // Literal text, never HTML.
});

test('a mismatched PNG is rejected rather than stretched under output geometry', async () => {
  const app = application(); await app.file('output', design());
  const loading = app.png(); app.decode(300, 200); await loading;
  assert.match(app.elements.importError.textContent, /dimensions do not match/);
  assert.equal(app.draws.length, 0);
  assert.equal(app.revoked.length, 1);
});

test('a new output clears its previous PNG and ignores an older pending image', async () => {
  const app = application(); await app.file('output', design('first'));
  const first = app.png(); app.decode(200, 200); await first;
  assert.equal(app.draws.length, 1);
  const pending = app.png();
  await app.file('output', design('second'));
  assert.equal(app.elements.png.value, '');
  app.decode(200, 200); await pending;
  assert.equal(app.draws.length, 1);
  app.elements.showText.fire('change');
  assert.equal(app.draws.length, 1);
});

test('loading a different task invalidates a pending output and clears old comparisons', async () => {
  const app = application(); await app.file('task', design('first'));
  await app.file('output', design('old-output'));
  const old = deferred(); const loading = app.file('output', () => old.promise);
  await app.file('task', design('second'));
  old.resolve(JSON.stringify(design('stale-output'))); await loading;
  assert.equal(app.elements.outTitle.textContent, 'Output');
  assert.equal(app.elements.outputElement.disabled, true);
  assert.equal(app.elements.out.width, 0);
});

test('rotated hit testing follows the painted box rather than its unrotated bounds', async () => {
  const app = application(); const scene = design('rotated');
  Object.assign(scene.elements[0], { x: 100, y: 50, width: 20, height: 100, rotation: 90 });
  await app.file('task', scene);
  app.elements.src.onmousemove({ clientX: 65, clientY: 100 });
  assert.match(app.elements.inspector.textContent, /rotated/);
  assert.equal(app.elements.sourceElement.value, 'rotated');
  app.elements.src.onmousemove({ clientX: 110, clientY: 55 });
  assert.equal(app.elements.inspector.textContent, '');
});

test('stalled image load times out, revokes its URL, and leaves no bitmap applied', async () => {
  const app = application(); await app.file('output', design());
  const loading = app.png();
  for (const callback of [...app.timers.values()]) callback();
  await loading;
  assert.match(app.elements.importError.textContent, /timed out/);
  assert.equal(app.revoked.length, 1);
  assert.equal(app.draws.length, 0);
});

for (const [filename, query, expected] of [
  ['trial.html', '?variant=B&poster=one%26two', '/?view=compare&poster=one%26two'],
  ['trial-comparison.html', '?poster=one%26two', '/?view=compare&poster=one%26two'],
]) test(`${filename} preserves encoded poster selection when opening comparison`, () => {
  const html = fs.readFileSync(path.join(__dirname, '../viewer', filename), 'utf8');
  const code = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
  let destination;
  vm.runInNewContext(code || '', { URL, URLSearchParams, location: {
    href: `http://127.0.0.1:8765/${filename}${query}`, search: query,
    replace: url => { destination = new URL(url); },
  } });
  assert.equal(destination?.pathname + destination?.search, expected);
});
