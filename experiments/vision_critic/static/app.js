(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const names = {
    unreadable_text: 'Unreadable text', weak_hierarchy: 'Weak hierarchy',
    detached_action: 'Detached action', broken_reading_order: 'Broken reading order',
    source_style_loss: 'Source style lost', poor_balance: 'Poor balance',
    poor_spacing: 'Poor spacing', weak_product_prominence: 'Weak product prominence',
    bad_overlap: 'Problematic overlap', distorted_ornament: 'Distorted ornament',
  };
  let status, pair, decision = null, issues = [], busy = false, artworkReady = false;
  let generation = 0;

  function failure(error) {
    $('error').textContent = error.message || String(error);
    $('error').hidden = false;
    $('reload').hidden = false;
  }
  async function request(path, options = {}) {
    const response = await fetch(path, {...options, signal: AbortSignal.timeout(30000)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
    return result;
  }
  function controls() {
    for (const control of $('feedback').querySelectorAll('button, select')) control.disabled = busy || !artworkReady;
    $('save').disabled = busy || !artworkReady || !decision || !pair || pair.done;
    for (const button of document.querySelectorAll('[data-decision]')) button.disabled = busy || !artworkReady;
    $('addIssue').disabled = busy || !artworkReady;
    $('save').textContent = busy ? 'Saving…' : 'Save and continue →';
  }
  function choose(value) {
    if (busy || !artworkReady || !pair || pair.done) return;
    decision = value;
    for (const button of document.querySelectorAll('[data-decision]')) button.setAttribute('aria-pressed', String(button.dataset.decision === value));
    $('boardA').dataset.selected = String(value === 'A');
    $('boardB').dataset.selected = String(value === 'B');
    $('saveStatus').textContent = 'Judgment selected. Save when you are ready.';
    controls();
  }
  function refreshElements() {
    const ids = pair?.element_ids?.[$('issueCandidate').value] || [];
    $('issueElement').replaceChildren(new Option('Whole composition', ''), ...ids.map(id => new Option(id, id)));
  }
  function renderIssues() {
    $('issues').replaceChildren();
    issues.forEach((issue, index) => {
      const item = document.createElement('li');
      const text = document.createElement('span');
      text.textContent = `${issue.candidate} · ${names[issue.code] || issue.code}${issue.element_ids.length ? ' · ' + issue.element_ids.join(', ') : ''}`;
      const remove = document.createElement('button');
      remove.type = 'button'; remove.textContent = 'Remove';
      remove.setAttribute('aria-label', 'Remove ' + text.textContent);
      remove.disabled = busy || !artworkReady;
      remove.addEventListener('click', () => { if (busy) return; issues.splice(index, 1); renderIssues(); });
      item.append(text, remove); $('issues').append(item);
    });
  }
  async function image(id, linkId, url) {
    $(linkId).href = url;
    $(id).src = url;
    let timer;
    try {
      await Promise.race([
        $(id).decode(),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error('Artwork loading timed out. Check the local collection, then retry.')), 30000);
        }),
      ]);
    } finally {
      clearTimeout(timer);
    }
  }
  async function loadPair({restoreFocus = false} = {}) {
    const token = ++generation;
    pair = null; decision = null; issues = []; artworkReady = false;
    $('error').hidden = $('reload').hidden = true; $('loading').hidden = false;
    $('loading').textContent = 'Loading the next source and adaptations…';
    $('artboards').hidden = $('feedback').hidden = $('complete').hidden = true;
    for (const button of document.querySelectorAll('[data-decision]')) button.setAttribute('aria-pressed', 'false');
    $('boardA').dataset.selected = $('boardB').dataset.selected = 'false';
    $('issueCandidate').value = $('issueCode').value = '';
    $('observations').open = false; refreshElements(); renderIssues(); controls();
    try {
      const [nextStatus, nextPair] = await Promise.all([request('/api/status'), request('/api/pair')]);
      if (token !== generation) return;
      status = nextStatus; pair = nextPair;
      $('annotator').value = status.annotator;
      $('progress').textContent = `${status.completed} of ${status.total} reviewed`;
      $('poolName').textContent = status.pair_pool === 'all-valid' ? 'All valid options · shadow research' : 'Admitted options';
      if (status.held_out_case_count) $('poolName').textContent += ` · ${status.held_out_case_count} test case${status.held_out_case_count === 1 ? '' : 's'} concealed`;
      const notices = [];
      if (status.purpose === 'synthetic') notices.push('Demo collection · Synthetic artwork and practice feedback. This is not task-training data.');
      if (status.include_test) notices.push('Test review explicitly enabled. Freeze the training recipe before using this feedback.');
      $('collectionNote').hidden = !notices.length;
      $('collectionNote').textContent = notices.join(' ');
      $('issueCode').replaceChildren(new Option('Choose an observation', ''), ...status.issue_codes.map(code => new Option(names[code] || code, code)));
      if (pair.done) {
        $('complete').hidden = false; $('targetSize').textContent = 'Review complete';
        $('completeTitle').textContent = status.total ? 'This review is complete' : 'No reviewable comparisons';
        $('complete').querySelector('p').textContent = status.total
          ? 'Your feedback is saved locally. No more unreviewed pairs remain for this annotator.'
          : 'This collection has no source-verified pairs in the selected pool. Check the collection before reviewing.';
        $('loading').hidden = true;
        if (restoreFocus) $('completeTitle').focus({preventScroll: true});
        return;
      }
      $('targetSize').textContent = `${pair.target_canvas.width} × ${pair.target_canvas.height} px`;
      await Promise.all([
        image('sourceImage', 'sourceLink', pair.source_url),
        image('aImage', 'aLink', pair.a_url),
        image('bImage', 'bLink', pair.b_url),
      ]);
      if (token !== generation) return;
      artworkReady = true;
      $('loading').hidden = true; $('artboards').hidden = $('feedback').hidden = false;
      $('saveStatus').textContent = 'Choose a judgment, then save. Nothing is selected for you.';
      controls();
      if (restoreFocus) $('judgmentLegend')?.focus({preventScroll: true});
    } catch (error) {
      if (token !== generation) return;
      $('loading').textContent = 'The review could not load. Reload this page after checking the local collection.';
      failure(error); controls();
      if (restoreFocus) $('reload').focus({preventScroll: true});
    }
  }
  for (const button of document.querySelectorAll('[data-decision]')) button.addEventListener('click', () => choose(button.dataset.decision));
  $('issueCandidate').addEventListener('change', refreshElements);
  $('reload').addEventListener('click', () => { if (!busy) loadPair(); });
  $('addIssue').addEventListener('click', () => {
    if (busy || !artworkReady || !pair || pair.done) return;
    const candidate = $('issueCandidate').value, code = $('issueCode').value;
    if (!candidate || !code) { $('saveStatus').textContent = 'Choose a candidate and an observation first.'; return; }
    const element = $('issueElement').value;
    const issue = {candidate, code, element_ids: element ? [element] : []};
    if (issues.some(existing => JSON.stringify(existing) === JSON.stringify(issue))) return;
    if (issues.length >= 24) { $('saveStatus').textContent = 'Keep this review to 24 observations.'; return; }
    issues.push(issue); renderIssues();
  });
  $('feedback').addEventListener('submit', async event => {
    event.preventDefault();
    if (busy || !artworkReady || !decision || !pair || pair.done) return;
    busy = true; $('error').hidden = true; controls();
    try {
      const label = {...pair.label_context, decision, issues, created_at: new Date().toISOString()};
      await request('/api/label', {
        method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': status.csrf_token},
        body: JSON.stringify(label),
      });
      busy = false;
      await loadPair({restoreFocus: true});
    } catch (error) {
      busy = false; failure(error);
      $('saveStatus').textContent = 'Save was not confirmed. Your selection remains here; check the message before retrying.';
      controls();
    }
  });
  document.addEventListener('keydown', event => {
    if (event.target.closest('input,select,textarea,[contenteditable]') || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.repeat) return;
    const values = {a:'A', b:'B', t:'tie', n:'neither', x:'abstain'};
    const value = values[event.key.toLowerCase()];
    if (value) { event.preventDefault(); choose(value); }
  });
  loadPair();
})();
