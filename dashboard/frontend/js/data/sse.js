// ════════════════════════════════════════════════════════════════════════════
// SSE — state stream + log stream
// ════════════════════════════════════════════════════════════════════════════
let _dupCleanupDone = false;

/** One-time cleanup: deselect stale duplicate-IP entries from state.selected. */
async function _cleanupDuplicateSelected() {
  if (_dupCleanupDone) return;
  // Wait until discovered has IP info
  if (!Object.values(state.discovered).some(n => n.ip)) return;
  _dupCleanupDone = true;
  const seenIps = new Map();
  // Sort: shorter key (SSH alias) wins
  const keys = Object.keys(state.selected).sort((a, b) => a.length - b.length);
  for (const h of keys) {
    const ip = state.discovered[h]?.ip || '';
    if (!ip) continue;
    if (seenIps.has(ip)) {
      // h is a duplicate — deselect it silently
      await fetch(`/api/nodes/${h}/deselect`, { method: 'POST' }).catch(() => {});
    } else {
      seenIps.set(ip, h);
    }
  }
}

function startSSE() {
  ssEventSource = new EventSource('/api/events');
  ssEventSource.onmessage = e => {
    const d = JSON.parse(e.data);
    state.discovered    = d.nodes.discovered    || {};
    state.selected      = d.nodes.selected      || {};
    state.running       = d.nodes.running       || {};
    state.usernames     = d.nodes.usernames     || {};
    state.ssh_aliases   = d.nodes.ssh_aliases   || {};
    state.node_os       = d.nodes.node_os       || {};
    // null means server explicitly cleared metrics (training stopped).
    // Reset the log-side fallback too so stale values don't linger.
    if (d.training === null) trainingFallbackMetrics = {};
    state.training      = d.training            || {};
    state.connectivity  = d.connectivity        || {};
    state.redis         = d.redis               || {};
    state.token_ts      = d.token_ts            || 0;
    state.token_text    = d.token_text          || '';
    state.grad_ts       = d.grad_ts             || 0;
    // Measure the real interval between gradient exchange events.
    // Clamp to [400ms, 30s] to avoid absurd values on first ping or stale data.
    if (state.grad_ts && state.grad_ts !== _prevGradTs && _prevGradTs > 0) {
      const measured = (state.grad_ts - _prevGradTs) * 1000;
      if (measured > 200 && measured < 30000)
        _gradIntervalMs = _gradIntervalMs * 0.7 + measured * 0.3; // EMA smoothing
    }
    if (state.grad_ts && state.grad_ts !== _prevGradTs) _prevGradTs = state.grad_ts;
    // Measure real interval between token events for inference speed matching.
    if (state.token_ts && state.token_ts !== _prevTokenTs && _prevTokenTs > 0) {
      const measuredTok = (state.token_ts - _prevTokenTs) * 1000;
      if (measuredTok > 50 && measuredTok < 10000)
        _tokenIntervalMs = _tokenIntervalMs * 0.35 + measuredTok * 0.65; // fast-converging EMA
    }
    if (state.token_ts && state.token_ts !== _prevTokenTs) _prevTokenTs = state.token_ts;
    // Override EMA with the real server-measured intervals when available (no estimation).
    if (d.grad_interval_ms != null && d.grad_interval_ms > 100 && d.grad_interval_ms < 60000)
      _gradIntervalMs = d.grad_interval_ms;
    if (d.token_interval_ms != null && d.token_interval_ms > 30 && d.token_interval_ms < 10000)
      _tokenIntervalMs = d.token_interval_ms;

    _cleanupDuplicateSelected();

    const nSig = JSON.stringify([state.discovered, state.selected, state.running, state.usernames, state.node_os]);
    const tSig = JSON.stringify(state.training);
    const cSig = JSON.stringify(state.connectivity);
    const rSig = JSON.stringify(state.redis || {});

    if (nSig !== _prevNSig) { renderHeader(); renderLeft(); updateButtons(); syncLogFilter(); renderMetrics(); _prevNSig = nSig; }
    if (tSig !== _prevTSig) { renderMetrics(); updateButtons(); _prevTSig = tSig; }
    if (cSig !== _prevCSig) { renderConnBar(); renderMetrics(); _prevCSig = cSig; }
    if (rSig !== _prevRSig) { renderHeader(); _prevRSig = rSig; }
  };
  ssEventSource.onerror = () => { ssEventSource.close(); ssEventSource = null; setTimeout(startSSE, 3000); };
}

// SSE — log stream
function startLogs() {
  logsEventSource = new EventSource('/api/logs');
  logsEventSource.onmessage = e => {
    const linesData = JSON.parse(e.data);
    linesData.forEach(appendLog);
  };
  logsEventSource.onerror = () => { logsEventSource.close(); logsEventSource = null; setTimeout(startLogs, 3000); };
}
