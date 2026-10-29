/* Opt-in UI experiment. All design, resize, inspection and export state is owned
   by app.js. Switching the frame keeps that state and the actual DOM nodes. */
(() => {
  'use strict';
  const variants = [
    {id: 'A', name: 'Docked studio', theme: 'dark', description: 'Setup · canvas · inspector'},
    {id: 'B', name: 'Review desk', theme: 'light', description: 'Setup ribbon · paired previews'},
    {id: 'C', name: 'Focus canvas', theme: 'dark', description: 'Wide canvas · right setup dock'},
  ];
  const $ = id => document.getElementById(id);
  let current;

  function applyVariant(id, updateURL = true) {
    current = variants.find(item => item.id === id) || variants[0];
    document.documentElement.dataset.variant = current.id;
    document.documentElement.dataset.theme = current.theme;
    $('theme').value = current.theme;
    $('trialName').textContent = current.name;
    $('trialDescription').textContent = current.description;
    $('trialWorkspaceLabel').textContent = current.name + ' workspace';
    document.title = `Chameleon · ${current.name} trial`;
    for (const button of document.querySelectorAll('[data-trial-variant]')) {
      button.setAttribute('aria-pressed', String(button.dataset.trialVariant === current.id));
    }
    if (updateURL) {
      const url = new URL(location.href);
      url.searchParams.set('variant', current.id);
      history.replaceState(null, '', url);
    }
  }

  function cycle(offset) {
    const index = variants.indexOf(current);
    applyVariant(variants[(index + offset + variants.length) % variants.length].id);
  }
  for (const button of document.querySelectorAll('[data-trial-variant]')) {
    button.addEventListener('click', () => applyVariant(button.dataset.trialVariant));
  }
  $('trialPrevious').addEventListener('click', () => cycle(-1));
  $('trialNext').addEventListener('click', () => cycle(1));
  window.addEventListener('popstate', () => applyVariant(new URLSearchParams(location.search).get('variant'), false));
  document.addEventListener('keydown', event => {
    // Native selects, numeric steppers, summaries and sliders own their keys.
    if (event.target.closest('input, textarea, select, summary, [contenteditable], [role="slider"]') || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      cycle(event.key === 'ArrowLeft' ? -1 : 1);
    }
  });

  function focusControl(id) {
    const control = $(id);
    control.scrollIntoView({block: 'nearest', behavior: 'instant'});
    control.focus({preventScroll: true});
  }
  for (const button of document.querySelectorAll('[data-trial-action]')) {
    button.addEventListener('click', () => {
      const action = button.dataset.trialAction;
      if (action === 'source') focusControl('sample');
      if (action === 'canvas') focusControl('preset');
      if (action === 'compare') document.querySelector('[data-view="compare"]').click();
      if (action === 'inspect') {
        if ($('inspection').hidden) { focusControl('run'); return; }
        $('inspection').querySelector('details').open = true;
        $('inspection').scrollIntoView({block: 'nearest', behavior: 'instant'});
        const element = $('elements').querySelector('button');
        if (element) element.focus({preventScroll: true});
      }
    });
  }

  const requested = new URLSearchParams(location.search).get('variant');
  applyVariant(requested, false);
  // Review opens paired previews once; later switches preserve the chosen view.
  if (current.id === 'B') document.querySelector('[data-view="compare"]').click();
  // The miniature document identity follows the real app's source selection.
  const sourceName = $('trialSourceName');
  const syncSource = () => { sourceName.textContent = $('taskName').textContent; };
  new MutationObserver(syncSource).observe($('taskName'), {childList: true, subtree: true, characterData: true});
  syncSource();
})();
