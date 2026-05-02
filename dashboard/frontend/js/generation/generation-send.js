// ════════════════════════════════════════════════════════════════════════════
// Generation — send request + composer init
// ════════════════════════════════════════════════════════════════════════════
async function sendGenerationRequest() {
  if (generationInFlight) return;
  if (!isInferenceRunning()) {
    syncGenerationAvailability();
    return;
  }
  const payload = generationFormState();
  if (!payload.text.trim()) {
    $('gen-stream-pill').textContent = 'empty';
    $('gen-stream-pill').className = 'gen-status-pill error';
    $('gen-stream-status').textContent = 'Prompt is empty. Add text before sending.';
    return;
  }

  setBottomTab('generation');
  persistGenerationForm();
  generationAbortController = new AbortController();
  generationInFlight = true;
  _genStartTime = null;
  _genTokenCount = 0;
  $('gen-output').textContent = '';
  $('gen-raw').textContent = '';
  $('gen-tput-stat').style.display = 'none';
  $('gen-tput-num').textContent = '—';
  $('gen-stream-pill').textContent = 'live';
  $('gen-stream-pill').className = 'gen-status-pill live';
  $('gen-stream-status').textContent = 'Streaming response…';
  syncGenerationAvailability();

  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify(payload),
      signal: generationAbortController.signal,
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    if (!r.body) throw new Error('Missing response body');

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let sawToken = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunkLines = buffer.split(/\r?\n/);
      buffer = chunkLines.pop() || '';
      for (const line of chunkLines) {
        appendGenerationRaw(line + '\n');
        if (!line.startsWith('data:')) continue;
        const payloadText = line.slice(5).trim();
        if (!payloadText) continue;
        try {
          const evt = JSON.parse(payloadText);
          if (typeof evt.token === 'string') {
            appendGenerationText(evt.token);
            sawToken = true;
            if (!_genStartTime) _genStartTime = Date.now();
            _genTokenCount++;
            const elapsed = (Date.now() - _genStartTime) / 1000;
            if (elapsed > 0) {
              $('gen-tput-stat').style.display = 'flex';
              $('gen-tput-num').textContent = Math.round(_genTokenCount / elapsed).toLocaleString();
            }
          }
          if (evt.done) {
            $('gen-stream-pill').textContent = 'done';
            $('gen-stream-pill').className = 'gen-status-pill live';
            $('gen-stream-status').textContent = sawToken ? 'Generation finished.' : 'Request finished without streamed tokens.';
          }
        } catch {
          appendGenerationText(payloadText + '\n');
        }
      }
    }

    if (buffer) appendGenerationRaw(buffer);
    if (!sawToken && !$('gen-output').textContent) {
      $('gen-output').innerHTML = '<span class="gen-empty">No streamed tokens were returned. Check the raw stream below for backend output.</span>';
    }
    if ($('gen-stream-pill').textContent === 'live') {
      $('gen-stream-pill').textContent = 'done';
      $('gen-stream-status').textContent = 'Generation finished.';
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      $('gen-stream-pill').textContent = 'stopped';
      $('gen-stream-pill').className = 'gen-status-pill';
      $('gen-stream-status').textContent = 'Generation stopped.';
    } else {
      $('gen-stream-pill').textContent = 'error';
      $('gen-stream-pill').className = 'gen-status-pill error';
      $('gen-stream-status').textContent = `Generation failed: ${err.message}`;
      appendGenerationRaw(`\n[error] ${err.message}\n`);
    }
  } finally {
    generationInFlight = false;
    generationAbortController = null;
    syncGenerationAvailability();
  }
}

function initGenerationComposer() {
  renderGenerationPresets();
  ['gen-text','gen-worker-rank','gen-max-tokens','gen-strategy','gen-session-id','gen-top-p','gen-temperature','gen-use-memory','gen-use-hf-defaults']
    .forEach(id => {
      const el = $(id);
      if (!el) return;
      el.addEventListener('input', updateGenerationCurlPreview);
      el.addEventListener('change', updateGenerationCurlPreview);
    });
  ['gen-worker-rank','gen-session-id'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.addEventListener('input', () => { el.dataset.userEdited = '1'; });
  });
  $('gen-text').addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      sendGenerationRequest();
    }
  });
  $('algo-sel')?.addEventListener('change', () => {
    syncGenerationTargetingRules();
    generationAutoFillFromSelectedNode(false);
    updateGenerationCurlPreview();
  });
}
