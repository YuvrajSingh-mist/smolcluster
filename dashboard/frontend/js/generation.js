// ════════════════════════════════════════════════════════════════════════════
// Generation panel — form state, curl preview, availability
// ════════════════════════════════════════════════════════════════════════════
function generationFormState() {
  const algoNow = String(_activeAlgo || $('algo-sel').value || 'syncps').toLowerCase();
  const rawRank = Number($('gen-worker-rank').value || 0);
  const useHFDefaults = $('gen-use-hf-defaults').checked;
  const payload = {
    text: $('gen-text').value,
    worker_rank: algoNow === 'syncps' ? 0 : Math.max(0, Math.trunc(Number.isFinite(rawRank) ? rawRank : 0)),
    max_tokens: Number($('gen-max-tokens').value || 128),
    session_id: $('gen-session-id').value,
    use_memory: $('gen-use-memory').checked,
    use_hf_defaults: useHFDefaults,
  };
  if (useHFDefaults) {
    payload.decoding_strategy = $('gen-strategy').value;
    payload.top_p = Number($('gen-top-p').value || 0.9);
    payload.temperature = Number($('gen-temperature').value || 0.7);
  }
  return payload;
}

function syncGenerationDecodingControls() {
  const useHFDefaults = $('gen-use-hf-defaults').checked;
  ['gen-strategy', 'gen-top-p', 'gen-temperature'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.disabled = !useHFDefaults;
  });
}

function selectedNodeInfo() {
  if (!selectedServer) return null;
  return state.running[selectedServer]
    || state.selected[selectedServer]
    || state.discovered[selectedServer]
    || null;
}

function syncGenerationTargetingRules() {
  const algoNow = String(_activeAlgo || $('algo-sel').value || 'syncps').toLowerCase();
  const rankInput = $('gen-worker-rank');
  const rankHint = $('gen-rank-hint');
  const nodeHint = $('gen-node-source');
  if (!rankInput || !rankHint || !nodeHint) return;

  if (algoNow === 'syncps') {
    rankInput.value = '0';
    rankInput.min = '0';
    rankInput.max = '0';
    rankInput.disabled = true;
    rankHint.textContent = 'SyncPS generation is pinned to worker rank 0.';
    nodeHint.textContent = 'Crowned node still drives session defaults, but SyncPS requests always target worker rank 0.';
    return;
  }

  rankInput.disabled = false;
  rankInput.min = '0';
  rankInput.removeAttribute('max');
  if (algoNow === 'classicdp') {
    rankHint.textContent = 'ClassicDP allows generation requests against any worker rank.';
  } else {
    rankHint.textContent = 'Pick the worker rank to target for generation.';
  }
  nodeHint.textContent = 'Worker rank and session ID can follow the crowned node.';
}

function generationAutoFillFromSelectedNode(force = false) {
  const info = selectedNodeInfo();
  if (!info || !selectedServer) {
    syncGenerationTargetingRules();
    if (force) updateGenerationCurlPreview();
    return;
  }

  const rankNum = Number(info.rank);
  const hasRank = Number.isFinite(rankNum);
  const algoNow = String(_activeAlgo || $('algo-sel').value || 'syncps').toLowerCase();
  const algoTag = ['syncps', 'mp', 'classicdp'].includes(algoNow) ? algoNow : 'syncps';

  const rankInput = $('gen-worker-rank');
  const sessionInput = $('gen-session-id');
  const targetRank = algoNow === 'syncps' ? 0 : (hasRank ? Math.max(0, Math.trunc(rankNum)) : 0);

  if (algoNow === 'syncps' || (hasRank && (force || rankInput.dataset.userEdited !== '1'))) {
    rankInput.value = String(targetRank);
  }

  const alias = state.ssh_aliases[selectedServer] || selectedServer;
  const slug = String(alias).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'node';
  const sid = (algoNow === 'syncps' || hasRank)
    ? `${algoTag}-worker-${targetRank}`
    : `${algoTag}-${slug}`;
  if (force || sessionInput.dataset.userEdited !== '1') {
    sessionInput.value = sid;
  }

  syncGenerationTargetingRules();
  updateGenerationCurlPreview();
}

function applyGenerationPreset(index) {
  const p = GENERATION_PRESETS[index];
  if (!p) return;
  $('gen-text').value = p.text;
  updateGenerationCurlPreview();
  $('gen-text').focus();
}

function renderGenerationPresets() {
  const row = $('gen-presets');
  if (!row) return;
  row.innerHTML = GENERATION_PRESETS.map((p, i) =>
    `<button class="gen-chip" type="button" onclick="applyGenerationPreset(${i})">${escHtml(p.label)}</button>`
  ).join('');
}

function persistGenerationForm() {
  uiSave({ gen: generationFormState() });
}

function shellSingleQuote(s) {
  return `'${String(s).replace(/'/g, `'"'"'`)}'`;
}

function generationCurlText() {
  const payload = generationFormState();
  const body = JSON.stringify(payload, null, 2);
  return [
    'curl -N -X POST http://localhost:8080/chat \\',
    '  -H "Content-Type: application/json" \\',
    `  -d ${shellSingleQuote(body)}`,
  ].join('\n');
}

function updateGenerationCurlPreview() {
  syncGenerationTargetingRules();
  syncGenerationDecodingControls();
  persistGenerationForm();
  setText('gen-curl-preview', generationCurlText());
  syncGenerationAvailability();
}

function syncGenerationAvailability() {
  syncGenerationTargetingRules();
  const live = isInferenceRunning() || (inferLocked && dashboardMode === 'infer');
  const canSend = live && !generationInFlight;
  $('gen-send').disabled = !canSend;
  $('gen-stop').disabled = !generationInFlight;
  $('gen-availability').textContent = generationInFlight ? 'streaming' : (live ? 'ready' : 'offline');
  $('gen-availability').className = `gen-status-pill${generationInFlight || live ? ' live' : ''}`;
  if (!generationInFlight) {
    $('gen-stream-pill').textContent = live ? 'ready' : 'offline';
    $('gen-stream-pill').className = `gen-status-pill${live ? ' live' : ''}`;
    $('gen-stream-status').textContent = live
      ? 'Inference is live. Send a prompt and stream tokens here.'
      : 'Launch inference first, then send a /chat request from this panel.';
  }
  $('gen-meta').textContent = generationInFlight
    ? 'Streaming response from /chat…'
    : (live ? 'Using the active inference service on localhost:8080.' : 'Generation is disabled until an inference launcher is running.');
}

function clearGenerationOutput() {
  $('gen-output').innerHTML = '<span class="gen-empty">Output cleared. Send another request to stream a fresh reply.</span>';
  $('gen-raw').innerHTML = '<span class="gen-empty">Raw event stream cleared.</span>';
  if (!generationInFlight) syncGenerationAvailability();
}

function appendGenerationText(text) {
  const out = $('gen-output');
  if (out.querySelector('.gen-empty')) out.textContent = '';
  out.textContent += text;
  out.scrollTop = out.scrollHeight;
}

function appendGenerationRaw(text) {
  const raw = $('gen-raw');
  if (raw.querySelector('.gen-empty')) raw.textContent = '';
  raw.textContent += text;
  raw.scrollTop = raw.scrollHeight;
}

async function copyGenerationCurl() {
  try {
    await navigator.clipboard.writeText(generationCurlText());
    $('gen-stream-pill').textContent = 'copied';
    $('gen-stream-pill').className = 'gen-status-pill live';
    setTimeout(() => { if (!generationInFlight) syncGenerationAvailability(); }, 1200);
  } catch {}
}

function stopGenerationRequest() {
  if (generationAbortController) generationAbortController.abort();
}
