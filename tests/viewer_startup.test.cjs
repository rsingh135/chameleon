/* Modal lifecycle checks; artwork and browser rendering are verified separately. */
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const code = readFileSync(path.join(__dirname, '../viewer/startup.js'), 'utf8');

function introduction({played = false, reduced = false, storageBlocked = false, failure = null, hidden = false} = {}) {
  const env = {now: 0, timers: new Map(), frames: new Map(), disposed: 0, draws: [], observed: 0, disconnected: 0};
  let sequence = 0;
  class Target {
    constructor() {
      this.listeners = new Map(); this.hidden = true; this.open = false; this.dataset = {};
      this.textContent = ''; this.isConnected = true;
      this.classes = new Set();
      this.classList = {add: value => this.classes.add(value), remove: value => this.classes.delete(value)};
    }
    addEventListener(type, fn) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type).add(fn);
    }
    removeEventListener(type, fn) { this.listeners.get(type)?.delete(fn); }
    fire(type, details = {}) {
      const event = {preventDefault() { this.defaultPrevented = true; }, ...details};
      for (const fn of [...(this.listeners.get(type) || [])]) fn(event);
      return event;
    }
    focus() { env.document.activeElement = this; }
    showModal() {
      if (failure === 'modal') throw new Error('Modal unavailable');
      this.open = true; env.skip.focus();
    }
    close() { this.open = false; this.fire('close'); }
    listenerCount() { return [...this.listeners.values()].reduce((sum, set) => sum + set.size, 0); }
  }
  env.intro = new Target(); env.skip = new Target(); env.replay = new Target();
  env.caption = new Target(); env.canvas = new Target(); env.motion = new Target();
  env.motion.matches = reduced;
  env.document = Object.assign(new Target(), {hidden, body: new Target()});
  const elements = {startupIntro: env.intro, startupSkip: env.skip, startupCanvas: env.canvas, startupCaption: env.caption};
  env.document.getElementById = id => elements[id];
  env.document.querySelectorAll = () => [env.replay];
  const art = {
    resize() { if (failure === 'resize') throw new Error('Resize unavailable'); },
    draw(seconds) { if (failure === 'draw') throw new Error('Drawing unavailable'); env.draws.push(seconds); },
    dispose() { env.disposed++; if (failure === 'dispose') throw new Error('Disposal unavailable'); },
  };
  env.window = Object.assign(new Target(), {
    matchMedia: () => env.motion,
    ChameleonIntroArt: {create() { if (failure === 'create') throw new Error('Canvas unavailable'); return art; }},
  });
  const storage = new Map(played ? [['chameleon.introPlayed', '1']] : []);
  vm.runInNewContext(code, {
    window: env.window, document: env.document,
    sessionStorage: {
      getItem(key) { if (storageBlocked) throw new Error('Blocked'); return storage.get(key); },
      setItem(key, value) { if (storageBlocked) throw new Error('Blocked'); storage.set(key, value); },
    },
    performance: {now: () => env.now},
    setTimeout(fn, delay) { const id = ++sequence; env.timers.set(id, {fn, at: env.now + delay}); return id; },
    clearTimeout(id) { env.timers.delete(id); },
    requestAnimationFrame(fn) { const id = ++sequence; env.frames.set(id, fn); return id; },
    cancelAnimationFrame(id) { env.frames.delete(id); },
    ResizeObserver: class {
      observe() { env.observed++; }
      disconnect() { env.disconnected++; }
    },
  }, {filename: 'viewer/startup.js'});
  env.frame = now => {
    env.now = now;
    const callbacks = [...env.frames.values()]; env.frames.clear();
    for (const fn of callbacks) fn(now);
  };
  env.tick = now => {
    env.now = now;
    for (const [id, timer] of [...env.timers]) {
      if (timer.at <= now && env.timers.delete(id)) timer.fn();
    }
  };
  env.storage = storage;
  return env;
}

function assertReleased(env) {
  assert.equal(env.intro.hidden, true);
  assert.equal(env.intro.open, false);
  assert.equal(env.document.body.classes.has('startup-active'), false);
  assert.equal(env.frames.size, 0);
  assert.equal(env.timers.size, 0);
  for (const target of [env.window, env.document, env.motion, env.skip, env.intro]) assert.equal(target.listenerCount(), 0);
}

test('initial play remembers the tab session; Skip immediately releases the modal and all work', () => {
  const env = introduction();
  assert.equal(env.intro.open, true);
  assert.equal(env.intro.hidden, false);
  assert.equal(env.storage.get('chameleon.introPlayed'), '1');
  assert.equal(env.document.activeElement, env.skip);
  env.skip.fire('click');
  assertReleased(env);
  assert.equal(env.disposed, 1);
  assert.equal(env.disconnected, 1);
});

test('a completed session skips automatic play but permits replay with focus return', () => {
  const env = introduction({played: true});
  assertReleased(env);
  env.replay.fire('click');
  assert.equal(env.intro.open, true);
  env.replay.fire('click');
  assert.equal(env.observed, 1, 'duplicate starts cannot create multiple runs');
  env.skip.fire('click');
  assertReleased(env);
  assert.equal(env.document.activeElement, env.replay);
});

test('the finite timeline leaves after its last pose and releases at 5.2 seconds', () => {
  const env = introduction();
  env.frame(2000);
  assert.equal(env.intro.dataset.phase, 'notice');
  env.frame(2800);
  assert.equal(env.intro.dataset.phase, 'catch');
  env.frame(3400);
  assert.equal(env.intro.dataset.phase, 'change');
  env.frame(4200);
  assert.equal(env.intro.dataset.phase, 'reveal');
  env.frame(4820);
  assert.equal(env.intro.classes.has('is-leaving'), true);
  env.tick(5200);
  assertReleased(env);
});

test('the independent deadline releases a stalled animation without relying on engine responses', () => {
  const env = introduction();
  env.tick(6000);
  assertReleased(env);
});

test('reduced motion skips automatic and replay animation, and changes dismiss an active run', () => {
  const reduced = introduction({reduced: true});
  reduced.replay.fire('click');
  assertReleased(reduced);
  assert.equal(reduced.draws.length, 0);
  const env = introduction();
  env.motion.matches = true;
  env.motion.fire('change');
  assertReleased(env);
});

test('Escape, native cancellation, and native close all release animation resources', () => {
  for (const method of ['escape', 'cancel', 'close']) {
    const env = introduction();
    if (method === 'escape') assert.equal(env.window.fire('keydown', {key: 'Escape'}).defaultPrevented, true);
    else if (method === 'cancel') assert.equal(env.intro.fire('cancel').defaultPrevented, true);
    else env.intro.close();
    assertReleased(env);
  }
});

test('blocked session storage still permits play, dismissal, and replay', () => {
  const env = introduction({storageBlocked: true});
  assert.equal(env.intro.open, true);
  env.skip.fire('click');
  assertReleased(env);
  env.replay.fire('click');
  assert.equal(env.intro.open, true);
  env.skip.fire('click');
  assertReleased(env);
});

test('renderer and modal failures always reveal the usable studio', () => {
  for (const failure of ['create', 'resize', 'draw', 'modal', 'dispose']) {
    const env = introduction({failure});
    if (failure === 'dispose') env.skip.fire('click');
    assertReleased(env);
  }
});

test('background pages skip startup and hiding or leaving an active page releases it', () => {
  assertReleased(introduction({hidden: true}));
  const hidden = introduction();
  hidden.document.hidden = true;
  hidden.document.fire('visibilitychange');
  assertReleased(hidden);
  const leaving = introduction();
  leaving.window.fire('pagehide');
  assertReleased(leaving);
});

test('resizing redraws the current pose without resetting the timeline', () => {
  const env = introduction();
  env.frame(2300);
  env.window.fire('resize');
  assert.equal(env.draws.at(-1), 2.3);
  assert.equal(env.intro.dataset.phase, 'notice');
  env.skip.fire('click');
  assertReleased(env);
});
