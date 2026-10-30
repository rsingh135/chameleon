/* Application frame only. app.js owns the document, preview view and exports. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const controls = $('controls');
  const setupToggles = [...document.querySelectorAll('[data-setup-toggle]')];
  let lastSetupToggle = $('settingsToggle');
  let setupOpen;

  function preference(key) {
    try { return localStorage.getItem(`chameleon.${key}`); }
    catch { return null; }
  }
  function remember(key, value) {
    try { localStorage.setItem(`chameleon.${key}`, String(value)); }
    catch { /* The workspace also works when browser storage is unavailable. */ }
  }

  function setSetupOpen(open, {focus = false, persist = false, target = null} = {}) {
    const focusWasInside = controls.contains(document.activeElement);
    setupOpen = open;
    controls.hidden = !open;
    $('workspace').dataset.setupOpen = String(open);
    for (const toggle of setupToggles) {
      toggle.setAttribute('aria-expanded', String(open));
      toggle.setAttribute('aria-label', open ? 'Collapse setup panel' : 'Expand setup panel');
      toggle.title = open ? 'Collapse setup panel' : 'Expand setup panel';
    }
    if (persist) remember('setupOpen', open);
    if (!focus) return;
    if (open) {
      const control = target || $('sample');
      (control.matches(':disabled') ? $('closeSettings') : control).focus();
    } else if (focusWasInside) {
      const opener = lastSetupToggle.getClientRects().length ? lastSetupToggle : $('settingsToggle');
      opener.focus();
    }
  }

  for (const toggle of setupToggles) {
    toggle.addEventListener('click', () => {
      lastSetupToggle = toggle;
      setSetupOpen(!setupOpen, {focus: true, persist: true});
    });
  }
  $('closeSettings').addEventListener('click', () => setSetupOpen(false, {focus: true, persist: true}));
  controls.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    event.preventDefault();
    setSetupOpen(false, {focus: true, persist: true});
  });
  const savedSetup = preference('setupOpen');
  setSetupOpen(savedSetup === null ? !matchMedia('(max-width: 760px)').matches : savedSetup === 'true');

  const savedTheme = preference('theme');
  if (['dark', 'light'].includes(savedTheme)) {
    document.documentElement.dataset.theme = savedTheme;
    $('theme').value = savedTheme;
  }
  $('theme').addEventListener('change', () => remember('theme', $('theme').value));

  // The actual preview view is the source of truth for the surrounding frame.
  // No design state, render request, selection, or download URL changes here.
  function syncView() {
    const view = $('previews').dataset.view;
    const review = view === 'compare';
    document.documentElement.dataset.workspace = review ? 'review' : 'studio';
    $('workspaceLabel').textContent = review ? 'Review desk · Original and result' : 'Docked studio workspace';
    const url = new URL(location.href);
    if (view === 'result') url.searchParams.delete('view');
    else url.searchParams.set('view', view);
    history.replaceState(null, '', url);
  }
  function restoreView() {
    const requested = new URLSearchParams(location.search).get('view');
    const view = ['compare', 'source'].includes(requested) ? requested : 'result';
    document.querySelector(`.view-switch button[data-view="${view}"]`).click();
  }
  restoreView();
  new MutationObserver(syncView).observe($('previews'), {attributes: true, attributeFilter: ['data-view']});
  syncView();
  window.addEventListener('popstate', restoreView);

  for (const button of document.querySelectorAll('[data-workspace-action]')) {
    button.addEventListener('click', () => {
      const action = button.dataset.workspaceAction;
      if (action === 'source' || action === 'canvas') {
        lastSetupToggle = $('setupRailToggle');
        setSetupOpen(true, {focus: true, persist: true, target: $(action === 'source' ? 'sample' : 'preset')});
      } else if (action === 'compare') {
        const compare = document.querySelector('.view-switch button[data-view="compare"]');
        compare.click();
        compare.focus({preventScroll: true});
      } else if (action === 'inspect') {
        if ($('inspection').hidden) {
          setSetupOpen(true, {focus: true, persist: true, target: $('run')});
          return;
        }
        $('inspection').querySelector('details').open = true;
        $('inspection').scrollIntoView({block: 'nearest', behavior: 'instant'});
        $('elements').querySelector('button')?.focus({preventScroll: true});
      }
    });
  }
  const syncSource = () => { $('workspaceSourceName').textContent = $('taskName').textContent; };
  new MutationObserver(syncSource).observe($('taskName'), {childList: true, subtree: true, characterData: true});
  syncSource();
})();
