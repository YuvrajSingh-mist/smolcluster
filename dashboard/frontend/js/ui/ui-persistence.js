// ════════════════════════════════════════════════════════════════════════════
// UI Persistence — Redis (via /api/ui-state)
// ════════════════════════════════════════════════════════════════════════════
let logBuffer = [];          // in-memory mirror of stored log entries
let _replayingLogs = false;  // suppress re-persisting during replay on boot
let _uiStateCache = {};      // in-memory copy populated at boot
let _persistLogTimer = null; // debounce: avoid a POST per log line

async function _uiLoadRemote() {
  try {
    const r = await fetch('/api/ui-state');
    if (r.ok) _uiStateCache = await r.json();
  } catch {}
  return _uiStateCache;
}

function uiSave(patch) {
  Object.assign(_uiStateCache, patch);
  fetch('/api/ui-state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch)
  }).catch(() => {});
}

function persistLog(entry) {
  logBuffer.push(entry);
  if (logBuffer.length > 600) logBuffer.splice(0, logBuffer.length - 600);
  if (_replayingLogs) return;
  // Debounce: flush at most once every 2 s instead of on every line.
  // During active training batches of 200 lines arrive at once — without
  // this a single SSE tick would fire 200 POST requests.
  clearTimeout(_persistLogTimer);
  _persistLogTimer = setTimeout(() => uiSave({ logs: logBuffer }), 2000);
}

async function loadUI() {
  const s = await _uiLoadRemote();

  // Restore SSH overrides before nodes are rendered
  if (s.ssh && typeof s.ssh === 'object') Object.assign(sshOverrides, s.ssh);

  // Restore dashboard mode
  if (s.mode) {
    dashboardMode = s.mode;
    $('mode-train').classList.toggle('active', dashboardMode === 'train');
    $('mode-infer').classList.toggle('active', dashboardMode === 'infer');
  }
  if (s.bottom_tab) bottomTab = s.bottom_tab === 'generation' ? 'generation' : 'logs';
  syncAlgoMenu(s.algo || undefined);

  if (s.gen && typeof s.gen === 'object') {
    if (typeof s.gen.text === 'string') $('gen-text').value = s.gen.text;
    if (s.gen.worker_rank != null) $('gen-worker-rank').value = s.gen.worker_rank;
    if (s.gen.max_tokens != null) $('gen-max-tokens').value = s.gen.max_tokens;
    if (typeof s.gen.decoding_strategy === 'string') $('gen-strategy').value = s.gen.decoding_strategy;
    if (typeof s.gen.session_id === 'string') $('gen-session-id').value = s.gen.session_id;
    if (s.gen.top_p != null) $('gen-top-p').value = s.gen.top_p;
    if (s.gen.temperature != null) $('gen-temperature').value = s.gen.temperature;
    $('gen-use-memory').checked = s.gen.use_memory !== false;
    $('gen-use-hf-defaults').checked = s.gen.use_hf_defaults === true;
  }

  // Restore setup guide position
  if (s.track) activeSetupTrack = s.track;
  if (s.step != null) activeSetupStep = Number(s.step);

  // Replay persisted logs into the DOM
  if (Array.isArray(s.logs) && s.logs.length) {
    logBuffer = s.logs;
    _replayingLogs = true;
    s.logs.forEach(e => appendLog(e));
    _replayingLogs = false;
  }

  // Navigate to saved view (default: entry)
  const view = s.view || 'entry';
  if (view === 'dashboard') openDashboard();
  else if (view === 'setup') openSetup();
  else backToEntry();

  setBottomTab(bottomTab);
  updateGenerationCurlPreview();
  syncGenerationAvailability();
  // Restore bottom panel height
  if (s.bottom_height) {
    const bottomEl = document.querySelector('.bottom');
    if (bottomEl) bottomEl.style.height = `${Number(s.bottom_height)}px`;
  }
}
