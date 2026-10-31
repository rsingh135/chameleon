/* Optional, finite introduction. Engine loading proceeds independently in app.js. */
(() => {
  'use strict';
  const intro = document.getElementById('startupIntro');
  const canvas = document.getElementById('startupCanvas');
  const skip = document.getElementById('startupSkip');
  const caption = document.getElementById('startupCaption');
  if (!intro || !canvas || !skip || typeof intro.showModal !== 'function') return;

  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const sessionKey = 'chameleon.introPlayed';
  const duration = 5200;
  const fadeDuration = 380;
  let active = null;

  function hasPlayed() {
    try { return sessionStorage.getItem(sessionKey) === '1'; }
    catch { return false; }
  }
  function remember() {
    try { sessionStorage.setItem(sessionKey, '1'); }
    catch { /* Restricted storage must not prevent the workspace from opening. */ }
  }

  function finish() {
    const run = active;
    if (!run) return;
    active = null;
    cancelAnimationFrame(run.frame);
    clearTimeout(run.deadline);
    clearTimeout(run.exitTimer);
    run.observer?.disconnect();
    window.removeEventListener('resize', run.resize);
    window.removeEventListener('keydown', run.keydown);
    window.removeEventListener('pagehide', finish);
    document.removeEventListener('visibilitychange', run.visibility);
    motion.removeEventListener('change', run.motionChange);
    skip.removeEventListener('click', finish);
    intro.removeEventListener('cancel', run.cancel);
    intro.removeEventListener('close', finish);
    // A failed renderer also releases the native modal and keyboard focus.
    try { run.art?.dispose(); }
    catch { /* Cleanup must still reveal the studio if a renderer fails. */ }
    finally {
      if (intro.open) intro.close();
      intro.hidden = true;
      intro.classList.remove('is-leaving');
      document.body.classList.remove('startup-active');
      if (run.opener?.isConnected) run.opener.focus({preventScroll: true});
    }
  }

  function leave() {
    if (!active || active.leaving) return;
    active.leaving = true;
    cancelAnimationFrame(active.frame);
    intro.classList.add('is-leaving');
    active.exitTimer = setTimeout(finish, fadeDuration);
  }

  function play(opener = null) {
    if (active || motion.matches || document.hidden) return;
    const run = {art: null, frame: 0, deadline: 0, exitTimer: 0, observer: null, opener, leaving: false};
    active = run;
    // The hard deadline is independent of animation frames or engine requests.
    run.deadline = setTimeout(finish, duration + 800);
    try {
      intro.dataset.phase = 'walk';
      intro.hidden = false;
      intro.showModal();
      document.body.classList.add('startup-active');
      run.art = window.ChameleonIntroArt.create(canvas);
      run.art.resize();
      run.art.draw(0);
      remember();
      const started = performance.now();
      let previousPhase = 'walk';
      function frame(now) {
        if (active !== run || run.leaving) return;
        try {
          const elapsed = Math.max(0, now - started);
          run.art.draw(elapsed / 1000);
          const phase = elapsed < 1900 ? 'walk' : elapsed < 2600 ? 'notice' : elapsed < 3100 ? 'catch' : elapsed < 4100 ? 'change' : 'reveal';
          if (phase !== previousPhase) {
            intro.dataset.phase = phase;
            caption.textContent = phase === 'notice' ? 'Found a fresh perspective.' : phase === 'catch' ? 'A taste for something different.' : phase === 'change' ? 'A little color changes everything.' : 'Made to change. Ready to create.';
            previousPhase = phase;
          }
          if (elapsed >= duration - fadeDuration) { leave(); return; }
          run.frame = requestAnimationFrame(frame);
        } catch { finish(); }
      }
      caption.textContent = 'A little curiosity goes a long way.';
      run.resize = () => {
        if (active !== run) return;
        try { run.art.resize(); run.art.draw(Math.min(4.8, (performance.now() - started) / 1000)); }
        catch { finish(); }
      };
      run.keydown = event => { if (event.key === 'Escape') { event.preventDefault(); finish(); } };
      run.cancel = event => { event.preventDefault(); finish(); };
      run.visibility = () => { if (document.hidden) finish(); };
      run.motionChange = () => { if (motion.matches) finish(); };
      window.addEventListener('resize', run.resize);
      window.addEventListener('keydown', run.keydown);
      window.addEventListener('pagehide', finish);
      document.addEventListener('visibilitychange', run.visibility);
      motion.addEventListener('change', run.motionChange);
      skip.addEventListener('click', finish);
      intro.addEventListener('cancel', run.cancel);
      intro.addEventListener('close', finish);
      if (typeof ResizeObserver === 'function') {
        run.observer = new ResizeObserver(run.resize);
        run.observer.observe(canvas);
      }
      run.frame = requestAnimationFrame(frame);
    } catch { finish(); }
  }

  for (const button of document.querySelectorAll('[data-startup-replay]')) {
    button.addEventListener('click', () => play(button));
  }
  if (!hasPlayed()) play();
})();
