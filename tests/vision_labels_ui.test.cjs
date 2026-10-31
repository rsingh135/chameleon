// State regressions execute the real annotation UI script without a web server.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.value = '';
    this.textContent = '';
    this.hidden = false;
    this.disabled = false;
    this.dataset = {};
    this.attributes = {};
    this.children = [];
    this.listeners = {};
    this.decode = async () => {};
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  emit(name) { return this.listeners[name]?.({ preventDefault() {} }); }
  setAttribute(name, value) { this.attributes[name] = value; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) {
    this.children = children;
    if (this.tag === 'select') this.value = children[0]?.value || '';
  }
  focus() { this.focused = true; }
  querySelector() { return this.children[0]; }
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

function start(timers = {}) {
  const elements = new Map();
  const ids = [
    'error', 'reload', 'feedback', 'save', 'addIssue', 'boardA', 'boardB',
    'saveStatus', 'issueCandidate', 'issueElement', 'issueCode', 'issues',
    'loading', 'artboards', 'complete', 'annotator', 'progress', 'poolName',
    'collectionNote', 'completeTitle', 'targetSize', 'sourceImage', 'sourceLink',
    'aImage', 'aLink', 'bImage', 'bLink', 'observations', 'judgmentLegend',
  ];
  ids.forEach(id => elements.set(id, new Element(id.startsWith('issue') && id !== 'issues' ? 'select' : 'div')));
  const decisions = ['A', 'B', 'tie', 'neither', 'abstain'].map(value => {
    const button = new Element('button');
    button.dataset.decision = value;
    return button;
  });
  elements.get('complete').append(new Element('p'));
  elements.get('feedback').querySelectorAll = () => [
    ...decisions, ...['save', 'addIssue', 'issueCandidate', 'issueCode', 'issueElement'].map(id => elements.get(id)),
    ...elements.get('issues').children.flatMap(item => item.children.filter(child => child.tag === 'button')),
  ];
  const requests = [];
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../experiments/vision_critic/static/app.js'), 'utf8'), {
    document: {
      getElementById: id => elements.get(id),
      querySelectorAll: () => decisions,
      createElement: tag => new Element(tag),
      addEventListener() {},
    },
    Option: class { constructor(text, value) { this.textContent = text; this.value = value; } },
    AbortSignal,
    setTimeout: timers.setTimeout || setTimeout,
    clearTimeout: timers.clearTimeout || clearTimeout,
    fetch: (url, options) => {
      const pending = deferred();
      requests.push({ url, options, ...pending });
      return pending.promise;
    },
  });
  const respond = (index, value, ok = true) => requests[index].resolve({ ok, status: ok ? 200 : 500, json: async () => value });
  return { elements, decisions, requests, respond };
}

function status(token = 'csrf') {
  return { annotator: 'automated-ui-test', total: 3, completed: 0, purpose: 'synthetic', issue_codes: ['bad_overlap'], csrf_token: token };
}
function pair(id) {
  return {
    done: false, label_context: { pair_id: id },
    target_canvas: { width: 320, height: 180 },
    source_url: `/${id}-source.png`, a_url: `/${id}-a.png`, b_url: `/${id}-b.png`,
    element_ids: { A: [`${id}-element`], B: [] },
  };
}
async function loaded() {
  const app = start();
  app.respond(0, status()); app.respond(1, pair('first'));
  await settle();
  return app;
}

test('late pair/status response cannot replace newer artwork or submitted context', async () => {
  const app = start();
  app.elements.get('reload').emit('click');
  app.respond(2, status('new-token')); app.respond(3, pair('new'));
  await settle();
  app.respond(0, status('old-token')); app.respond(1, pair('old'));
  await settle();
  assert.equal(app.elements.get('aImage').src, '/new-a.png');
  app.elements.get('issueCandidate').value = 'A';
  app.elements.get('issueCandidate').emit('change');
  assert.equal(app.elements.get('issueElement').children[1].value, 'new-element');
  app.decisions[0].emit('click');
  app.elements.get('feedback').emit('submit');
  const saved = app.requests[4];
  assert.equal(JSON.parse(saved.options.body).pair_id, 'new');
  assert.equal(saved.options.headers['X-CSRF-Token'], 'new-token');
});

test('saving freezes judgment and all observation controls until next pair is ready', async () => {
  const app = await loaded();
  const el = id => app.elements.get(id);
  el('issueCandidate').value = 'A'; el('issueCandidate').emit('change');
  el('issueCode').value = 'bad_overlap'; el('issueElement').value = 'first-element';
  el('addIssue').emit('click'); app.decisions[0].emit('click');
  const remove = el('issues').children[0].children[1];
  const saving = el('feedback').emit('submit');
  for (const id of ['issueCandidate', 'issueCode', 'issueElement', 'addIssue', 'save']) assert.equal(el(id).disabled, true, id);
  assert.equal(remove.disabled, true);
  // Handler guards also protect against an already queued click event.
  remove.emit('click'); app.decisions[1].emit('click'); el('addIssue').emit('click');
  assert.equal(el('issues').children.length, 1);
  assert.equal(app.decisions[0].attributes['aria-pressed'], 'true');
  const submitted = JSON.parse(app.requests[2].options.body);
  assert.equal(submitted.decision, 'A'); assert.equal(submitted.issues.length, 1);
  app.respond(2, { saved: true }); await settle();
  app.respond(3, status()); app.respond(4, pair('next'));
  await saving;
  assert.equal(el('judgmentLegend').focused, true);
  assert.equal(el('save').disabled, true); // New pair never inherits a judgment.
  assert.equal(el('issues').children.length, 0);
});

test('failed save retains judgment and observations for a deliberate retry', async () => {
  const app = await loaded();
  app.decisions[1].emit('click');
  const saving = app.elements.get('feedback').emit('submit');
  app.respond(2, { error: 'Disk unavailable' }, false);
  await saving;
  assert.equal(app.elements.get('error').textContent, 'Disk unavailable');
  assert.equal(app.decisions[1].attributes['aria-pressed'], 'true');
  assert.equal(app.elements.get('save').disabled, false);
  assert.match(app.elements.get('saveStatus').textContent, /not confirmed/);
});

test('a failed artwork decode never enables a judgment or save', async () => {
  const app = start();
  app.elements.get('aImage').decode = async () => { throw new Error('Image unavailable'); };
  app.respond(0, status()); app.respond(1, pair('broken'));
  await settle();
  assert.equal(app.elements.get('error').textContent, 'Image unavailable');
  assert.equal(app.elements.get('artboards').hidden, true);
  app.decisions[0].emit('click');
  assert.equal(app.elements.get('save').disabled, true);
});

test('stalled artwork has bounded failure and a visible retry', async () => {
  const timers = [];
  const app = start({
    setTimeout: (callback, delay) => { timers.push({ callback, delay }); return timers.length; },
    clearTimeout() {},
  });
  app.elements.get('aImage').decode = () => new Promise(() => {});
  app.respond(0, status()); app.respond(1, pair('stalled'));
  await settle();
  assert.equal(timers[1].delay, 30000);
  timers[1].callback();
  await settle();
  assert.match(app.elements.get('error').textContent, /timed out/);
  assert.equal(app.elements.get('reload').hidden, false);
  assert.equal(app.elements.get('save').disabled, true);
});
