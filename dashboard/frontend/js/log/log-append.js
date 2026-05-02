// ════════════════════════════════════════════════════════════════════════════
// Log append + log UI actions
// ════════════════════════════════════════════════════════════════════════════
function appendLog({ hostname, line }) {
  // Persist log entry to Redis-backed buffer
  persistLog({ hostname, line });

  // Fast-path: structured metrics JSON
  const smolM = line.match(/\[SMOL_METRICS\]\s*(\{.+\})/);
  if (smolM) {
    try {
      const m = JSON.parse(smolM[1]);
      for (const [k, v] of Object.entries(m)) {
        if (v !== null && v !== undefined) trainingFallbackMetrics[k] = v;
      }
      trainingFallbackMetrics.algorithm ||= _activeAlgo || $('algo-sel').value;
      renderMetrics();
    } catch(e) {}
    // Fall through so the line is also visible in the log terminal.
  }

  // Transport events (structured + pattern-based)
  _parseTransportEvent(line);
  _checkTransportPatterns(line);

  // Training metrics from log pattern matching
  _parseTrainingLogLine(line);

  // In inference, grab token text from ANY host's log
  const runningVals = Object.values(state.running || {});
  const isInferringNow = runningVals.some(r => r.role === 'inference_launcher' || r.algorithm === 'infer');
  if (isInferringNow) {
    const sseM = line.match(/^data:\s*(\{.+\})\s*$/);
    if (sseM) {
      try {
        const obj = JSON.parse(sseM[1]);
        if (typeof obj.token === 'string') _queueToken(obj.token);
      } catch(e) {}
    } else if (line.trim().length > 0) {
      const m = line.match(/\[Worker\s*\d+\]\s*Token\s+\d+\s*:\s*(.+)$/i)
        || line.match(/\bStreamed\s+token\s+\d+\s*:\s*(.+)$/i)
        || line.match(/\btoken\s+\d+\s*:\s*(.+)$/i);
      if (m) {
        const raw = (m[1] || '').trim().replace(/^repr\(['"]/,'').replace(/['"]\)$/,'').replace(/^['"]+|['"]+$/g, '');
        if (raw.length > 0) _queueToken(raw);
      }
    }
  }

  if (!knownLogHosts.has(hostname)) knownLogHosts.add(hostname);
  ensureLogFilterOption(hostname);

  const visible = logFilterHost === 'all' || logFilterHost === hostname;
  const box = $('logbox');
  const row = document.createElement('div');
  row.className = 'logline';
  row.dataset.host = hostname;
  if (!visible) row.style.display = 'none';
  row.innerHTML = `<span class="loghostname ${hostColor(hostname)}">${logHostLabel(hostname)}</span><span class="logtext">${ansiToHtml(line)}</span>`;
  box.appendChild(row);
  while (box.children.length > 800) box.removeChild(box.firstChild);
  if (autoscroll && visible) box.scrollTop = box.scrollHeight;
}

function applyLogFilter() {
  logFilterHost = $('log-filter').value;
  $('logbox').querySelectorAll('.logline').forEach(row => {
    row.style.display = (logFilterHost === 'all' || row.dataset.host === logFilterHost) ? '' : 'none';
  });
  if (autoscroll) $('logbox').scrollTop = $('logbox').scrollHeight;
}

function clearLogs() {
  $('logbox').innerHTML = '';
  logBuffer = [];
  knownLogHosts.clear();
  const sel = $('log-filter');
  while (sel.children.length > 1) sel.removeChild(sel.lastChild);
  logFilterHost = 'all';
  sel.value = 'all';
  syncLogFilter();  // re-add current nodes immediately
  trainingFallbackMetrics = {};
  uiSave({ logs: [] });
}

function toggleAutoscroll() {
  autoscroll = !autoscroll;
  $('autoscroll-btn').textContent = autoscroll ? 'autoscroll ✓' : 'autoscroll ✗';
}

function initBottomResizer() {
  const right = document.querySelector('.right');
  const bottom = document.querySelector('.bottom');
  const resizer = $('bottom-resizer');
  if (!right || !bottom || !resizer) return;

  const MIN_BOTTOM = 170;
  const MIN_TOPO = 170;
  let dragging = false, startY = 0, startBottom = 0;

  resizer.addEventListener('pointerdown', (e) => {
    dragging = true;
    startY = e.clientY;
    startBottom = bottom.getBoundingClientRect().height;
    right.classList.add('resizing');
    resizer.setPointerCapture(e.pointerId);
    e.preventDefault();
  });

  const onPointerMove = (e) => {
    if (!dragging) return;
    const delta = startY - e.clientY;
    const maxBottom = Math.max(MIN_BOTTOM, right.clientHeight - MIN_TOPO);
    const next = Math.max(MIN_BOTTOM, Math.min(maxBottom, startBottom + delta));
    bottom.style.height = `${Math.round(next)}px`;
  };

  const onPointerUp = () => {
    if (!dragging) return;
    dragging = false;
    right.classList.remove('resizing');
    uiSave({ bottom_height: Math.round(bottom.getBoundingClientRect().height) });
  };

  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);

  resizer.addEventListener('dblclick', () => {
    bottom.style.height = '34vh';
    uiSave({ bottom_height: '' });
  });
}
