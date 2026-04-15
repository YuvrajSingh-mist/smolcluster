// ════════════════════════════════════════════════════════════════════════════
// State
// ════════════════════════════════════════════════════════════════════════════
let state = {
  discovered:{}, selected:{}, running:{}, usernames:{}, ssh_aliases:{}, training:{}, connectivity:{}, redis:{}
};
const sshOverrides = {};
let _prevNSig='', _prevTSig='', _prevCSig='', _prevRSig='';
let dashboardMode = 'train';
let bottomTab = 'logs';
let ssEventSource = null;  // Global state event source
let logsEventSource = null; // Global logs event source
let generationAbortController = null;
let generationInFlight = false;
let _genStartTime = null, _genTokenCount = 0;
let trainingFallbackMetrics = {};

const TRAIN_ALGOS = [
  { value: 'syncps', label: 'SyncPS — Sync Parameter Server' },
  { value: 'mp', label: 'MP — Model Parallelism' },
  { value: 'classicdp', label: 'ClassicDP — Classic Data Parallelism' },
  { value: 'fsdp', label: 'FSDP — Fully Sharded Data Parallel' },
  { value: 'ep', label: 'EP — Expert Parallelism' },
  { value: 'mp_pipeline', label: 'MP Pipeline — Model Parallelism Pipeline' },
  { value: 'edp', label: 'EDP — Elastic Data Parallelism' },
  { value: 'grpo', label: 'GRPO — Group Relative Policy Optimization' },
];

const INFER_ALGOS = [
  { value: 'syncps', label: 'SyncPS — Sync Parameter Server' },
  { value: 'mp', label: 'MP — Model Parallelism' },
  { value: 'classicdp', label: 'ClassicDP — Classic Data Parallelism' },
];

const GENERATION_PRESETS = [
  { label: 'SyncPS explainer', text: 'Explain parameter server architecture in SyncPS for this cluster in simple terms.' },
  { label: 'Worker role', text: 'What does worker rank 1 do during inference and how does it communicate with the server?' },
  { label: 'Bottlenecks', text: 'List likely bottlenecks in this distributed inference setup and suggest concrete optimizations.' },
  { label: 'Debug checklist', text: 'Give me a practical debugging checklist when token streaming stalls in distributed inference.' },
];

const lines = (...parts) => parts.join('\n');

const setupTracks = {
  mac: [
    {
      title: 'Cable + Static IPs',
      copy: 'Connect Thunderbolt 4 cables in a chain: mini1↔mini2↔mini3. On each Mac go to System Settings → Network → Thunderbolt Bridge → Details → set Configure IPv4 to Manually.',
      command: lines(
        '# Assign static IPs on each Mac:',
        '# mini1:  10.10.0.1 / 255.255.255.0',
        '# mini2:  10.10.0.2 / 255.255.255.0',
        '# mini3:  10.10.0.3 / 255.255.255.0',
        '',
        '# Verify connectivity from mini1:',
        'ifconfig | grep -A5 "bridge\\|thunderbolt"',
        'ping -c 4 10.10.0.2',
        'ping -c 4 10.10.0.3'
      ),
      diagram: 'controller'
    },
    {
      title: 'Node Inventory',
      copy: 'On mini1, copy the nodes.yaml template and fill in the alias, IP, and username for each worker node.',
      command: lines(
        'cp scripts/installations/nodes.yaml.example \\',
        '  ~/.config/smolcluster/nodes.yaml',
        '\${EDITOR:-nano} ~/.config/smolcluster/nodes.yaml',
        '',
        '# nodes.yaml example:',
        '# nodes:',
        '#   - alias: mini2',
        '#     ip: 10.10.0.2',
        '#     user: your_username',
        '#   - alias: mini3',
        '#     ip: 10.10.0.3',
        '#     user: your_username'
      ),
      diagram: 'ssh'
    },
    {
      title: 'SSH Setup + Bootstrap',
      copy: 'Distribute SSH keys to all workers, install smolcluster + deps on each node, then copy your .env with W&B and HF tokens.',
      command: lines(
        '# Distribute keys and write ~/.ssh/config',
        'bash scripts/installations/setup_ssh.sh',
        '',
        '# Install deps + clone repo on each worker',
        'bash scripts/installations/setup.sh',
        '',
        '# Copy .env to workers',
        'awk \'/^[[:space:]]*-[[:space:]]*alias:/ {print $3}\' \\',
        '  ~/.config/smolcluster/nodes.yaml | while read -r node; do',
        '  scp .env "$node:~/Desktop/smolcluster/"',
        'done'
      ),
      diagram: 'keys'
    },
    {
      title: 'Configure Cluster YAMLs',
      copy: 'Update cluster_config_syncps.yaml and cluster_config_inference.yaml with your actual node aliases, IPs, and port.',
      command: lines(
        '# src/smolcluster/configs/cluster_config_syncps.yaml',
        'host_ip:',
        '  mini1: "10.10.0.1"',
        '  mini2: "10.10.0.2"',
        '  mini3: "10.10.0.3"',
        'port: 65432',
        'num_workers: 2',
        'server: mini1',
        'workers:',
        '  - hostname: mini2',
        '    rank: 1',
        '  - hostname: mini3',
        '    rank: 2'
      ),
      diagram: 'dashboard'
    },
    {
      title: 'Smoke Test',
      copy: 'Run a dry run to validate config + SSH, then launch SyncPS inference. Hit the health endpoint to confirm everything is live.',
      command: lines(
        '# Dry run (validates config + SSH, no workers launched)',
        './scripts/inference/launch_inference.sh --algorithm syncps --dry-run',
        '',
        '# Launch',
        './scripts/inference/launch_inference.sh --algorithm syncps',
        '',
        '# Health check',
        'curl http://localhost:8080/health',
        '',
        '# Generate',
        'curl -X POST http://localhost:8080/generate \\',
        '  -H "Content-Type: application/json" \\',
        '  -d \'{"prompt": "Once upon a time", "max_new_tokens": 50}\'',
        '',
        '# Cleanup',
        './scripts/inference/launch_inference.sh --cleanup'
      ),
      diagram: 'launch'
    }
  ],
  jetson: [
    {
      title: 'Enable SSH + Static IP',
      copy: 'On each Jetson, enable SSH and assign a static IP using nmcli. Run the discover script first to find your Ethernet interface name.',
      command: lines(
        '# Enable SSH (on each Jetson)',
        'sudo systemctl enable ssh && sudo systemctl start ssh',
        '',
        '# Find interface name',
        './scripts/installations/discover_network.sh',
        '',
        '# Assign static IP (replace CONNECTION_NAME)',
        'sudo nmcli con mod "<CONNECTION_NAME>" \\',
        '  ipv4.addresses 192.168.50.101/24 \\',
        '  ipv4.method manual',
        'sudo nmcli con up "<CONNECTION_NAME>"',
        'ip addr show'
      ),
      diagram: 'workers'
    },
    {
      title: 'Passwordless sudo',
      copy: 'setup_jetson.sh installs system packages and requires passwordless sudo. Add the NOPASSWD rule on each Jetson before running setup.sh.',
      command: lines(
        '# On each Jetson:',
        'sudo visudo',
        '',
        '# Add this line at the end (replace your_username):',
        '# your_username ALL=(ALL) NOPASSWD:ALL',
        '',
        '# Verify - should not prompt for a password:',
        'sudo whoami'
      ),
      diagram: 'deps'
    },
    {
      title: 'Node Inventory + SSH',
      copy: 'On the controller, fill nodes.yaml with Jetson IPs, then run setup_ssh.sh to distribute keys and setup.sh to bootstrap deps on every Jetson.',
      command: lines(
        'cp scripts/installations/nodes.yaml.example \\',
        '  ~/.config/smolcluster/nodes.yaml',
        '',
        '# nodes:',
        '#   - alias: jetson1',
        '#     ip: 192.168.50.101',
        '#     user: nvidia',
        '#   - alias: jetson2',
        '#     ip: 192.168.50.102',
        '#     user: nvidia',
        '',
        'bash scripts/installations/setup_ssh.sh',
        'bash scripts/installations/setup.sh'
      ),
      diagram: 'keys'
    },
    {
      title: 'Configure Cluster YAMLs',
      copy: 'Update cluster_config_syncps.yaml and cluster_config_inference.yaml with your Jetson IPs and the controller IP on the same subnet.',
      command: lines(
        '# src/smolcluster/configs/cluster_config_syncps.yaml',
        'host_ip:',
        '  mini1:   "192.168.50.100"',
        '  jetson1: "192.168.50.101"',
        '  jetson2: "192.168.50.102"',
        'port: 65432',
        'num_workers: 2',
        'server: mini1',
        'workers:',
        '  - hostname: jetson1',
        '    rank: 1',
        '  - hostname: jetson2',
        '    rank: 2'
      ),
      diagram: 'dashboard'
    },
    {
      title: 'Smoke Test',
      copy: 'Dry run first to validate config + SSH, then launch SyncPS inference across your Jetsons. Check the health endpoint.',
      command: lines(
        '# Dry run',
        './scripts/inference/launch_inference.sh --algorithm syncps --dry-run',
        '',
        '# Launch',
        './scripts/inference/launch_inference.sh --algorithm syncps',
        '',
        '# Health check',
        'curl http://localhost:8080/health',
        '',
        '# Generate',
        'curl -X POST http://localhost:8080/generate \\',
        '  -H "Content-Type: application/json" \\',
        '  -d \'{"prompt": "Once upon a time", "max_new_tokens": 50}\'',
        '',
        '# Cleanup',
        './scripts/inference/launch_inference.sh --cleanup'
      ),
      diagram: 'launch'
    }
  ]
};
let activeSetupTrack = 'mac';
let activeSetupStep = 0;

// ════════════════════════════════════════════════════════════════════════════
// UI Persistence — Redis (via /api/ui-state)
// ════════════════════════════════════════════════════════════════════════════
let logBuffer = [];          // in-memory mirror of stored log entries
let _replayingLogs = false;  // suppress re-persisting during replay on boot
let _uiStateCache = {};      // in-memory copy populated at boot

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
  if (!_replayingLogs) uiSave({ logs: logBuffer });
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

// Log terminal
let autoscroll = true;
let logFilterHost = 'all';
const knownLogHosts = new Set();
const hostColorMap = {};
let nextColor = 0;
const HOST_COLORS = ['hc0','hc1','hc2','hc3','hc4','hc5','hc6','hc7'];

function hostColor(hostname) {
  if (!hostColorMap[hostname]) hostColorMap[hostname] = HOST_COLORS[nextColor++ % HOST_COLORS.length];
  return hostColorMap[hostname];
}

function allKnownLogHostKeys() {
  return new Set([
    ...Object.keys(state.selected || {}),
    ...Object.keys(state.running || {}),
    ...Object.keys(state.discovered || {}),
    ...knownLogHosts,
  ]);
}

function logHostLabel(hostname) {
  const alias = (state.ssh_aliases && state.ssh_aliases[hostname]) || '';
  if (!alias || alias === hostname) return hostname;

  let collision = false;
  for (const other of allKnownLogHostKeys()) {
    if (other === hostname) continue;
    const otherAlias = (state.ssh_aliases && state.ssh_aliases[other]) || '';
    if (other === alias || otherAlias === alias) {
      collision = true;
      break;
    }
  }
  return collision ? `${alias} (${hostname})` : alias;
}

function ensureLogFilterOption(hostname) {
  if (!hostname || hostname === 'all') return;
  let opt = Array.from($('log-filter').options).find(o => o.value === hostname);
  if (!opt) {
    opt = document.createElement('option');
    opt.value = hostname;
    $('log-filter').appendChild(opt);
  }
  const label = logHostLabel(hostname);
  if (opt.textContent !== label) opt.textContent = label;
}

function refreshLogHostLabels() {
  Array.from($('log-filter').options).forEach(opt => {
    if (opt.value !== 'all') opt.textContent = logHostLabel(opt.value);
  });
  $('logbox').querySelectorAll('.logline').forEach(row => {
    const badge = row.querySelector('.loghostname');
    if (badge && row.dataset.host) badge.textContent = logHostLabel(row.dataset.host);
  });
}

// ════════════════════════════════════════════════════════════════════════════
// Entry + Setup views
// ════════════════════════════════════════════════════════════════════════════
function backToEntry() {
  $('entry-view').classList.remove('hidden');
  $('setup-view').classList.add('hidden');
  $('dashboard-shell').classList.add('hidden');
  uiSave({ view: 'entry' });
}

function openSetup() {
  $('entry-view').classList.add('hidden');
  $('setup-view').classList.remove('hidden');
  $('dashboard-shell').classList.add('hidden');
  renderSetup();
  uiSave({ view: 'setup' });
}

function openDashboard() {
  $('entry-view').classList.add('hidden');
  $('setup-view').classList.add('hidden');
  $('dashboard-shell').classList.remove('hidden');
  uiSave({ view: 'dashboard' });
}

function setSetupTrack(track) {
  activeSetupTrack = track === 'jetson' ? 'jetson' : 'mac';
  activeSetupStep = 0;
  $('setup-tab-mac').classList.toggle('active', activeSetupTrack === 'mac');
  $('setup-tab-jetson').classList.toggle('active', activeSetupTrack === 'jetson');
  renderSetup();
  uiSave({ track: activeSetupTrack, step: activeSetupStep });
}

function renderSetup() {
  const steps = setupTracks[activeSetupTrack] || [];
  const list = $('setup-steps');
  list.innerHTML = steps.map((s, i) =>
    `<button class="setup-step ${i === activeSetupStep ? 'active' : ''}" onclick="jumpSetupStep(${i})">${i + 1}. ${s.title}</button>`
  ).join('');
  renderSetupStep();
}

function jumpSetupStep(i) {
  const steps = setupTracks[activeSetupTrack] || [];
  activeSetupStep = Math.max(0, Math.min(steps.length - 1, i));
  renderSetup();
  uiSave({ step: activeSetupStep });
}

function nextSetupStep() {
  jumpSetupStep(activeSetupStep + 1);
}

function prevSetupStep() {
  jumpSetupStep(activeSetupStep - 1);
}

function copyCmd(btn) {
  const txt = btn.previousElementSibling.textContent;
  navigator.clipboard.writeText(txt).then(() => {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1600);
  }).catch(() => {});
}

function renderSetupStep() {
  const steps = setupTracks[activeSetupTrack] || [];
  if (!steps.length) return;
  const s = steps[activeSetupStep];
  setText('setup-kicker', `${activeSetupTrack === 'mac' ? 'Mac Mini (Thunderbolt)' : 'Jetson / Home Router'} • Step ${activeSetupStep + 1}/${steps.length}`);
  setText('setup-title', s.title);
  setText('setup-copy', s.copy);
  setText('setup-command', s.command);
  const pct = Math.round(((activeSetupStep + 1) / steps.length) * 100);
  $('setup-progress-fill').style.width = `${pct}%`;
}

// ════════════════════════════════════════════════════════════════════════════
// SSE — state stream
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

    if (nSig !== _prevNSig) { renderHeader(); renderLeft(); updateButtons(); syncLogFilter(); _prevNSig = nSig; }
    if (tSig !== _prevTSig) { renderMetrics(); updateButtons(); _prevTSig = tSig; }
    if (cSig !== _prevCSig) { renderConnBar(); _prevCSig = cSig; }
    if (rSig !== _prevRSig) { renderHeader(); _prevRSig = rSig; }
  };
  ssEventSource.onerror = () => { ssEventSource.close(); ssEventSource = null; setTimeout(startSSE, 3000); };
}

// SSE — log stream
function startLogs() {
  logsEventSource = new EventSource('/api/logs');
  logsEventSource.onmessage = e => {
    const lines = JSON.parse(e.data);
    lines.forEach(appendLog);
  };
  logsEventSource.onerror = () => { logsEventSource.close(); logsEventSource = null; setTimeout(startLogs, 3000); };
}

// ════════════════════════════════════════════════════════════════════════════
// Render helpers
// ════════════════════════════════════════════════════════════════════════════

/**
 * Deduplicate a set of hostnames by IP address, using state.discovered for
 * IP lookup. When two hostnames resolve to the same IP, keep whichever is
 * in state.running first, then prefer the shorter key (SSH alias).
 * Returns a new object with duplicate keys removed.
 */
function dedupByIp(hostnameMap) {
  const seenIps = new Map(); // ip → kept key
  const result  = {};
  // Sort: running entries first, then by key length (shorter = SSH alias)
  const sorted = Object.keys(hostnameMap).sort((a, b) => {
    const aRun = a in state.running ? 0 : 1;
    const bRun = b in state.running ? 0 : 1;
    if (aRun !== bRun) return aRun - bRun;
    return a.length - b.length;
  });
  for (const h of sorted) {
    const ip = state.discovered[h]?.ip || '';
    if (ip && seenIps.has(ip)) continue; // duplicate — skip
    if (ip) seenIps.set(ip, h);
    result[h] = hostnameMap[h];
  }
  return result;
}

/** Deduplicated discovered nodes (by IP). */
function discoveredDedup() {
  return dedupByIp(state.discovered);
}

/** Deduplicated union of selected + running nodes (by IP). */
function clusterDedup() {
  const merged = {};
  for (const [h,v] of Object.entries(state.selected)) merged[h] = v;
  for (const [h,v] of Object.entries(state.running))  merged[h] = v;
  return dedupByIp(merged);
}

function renderHeader() {
  setText('lbl-disc', `${Object.keys(discoveredDedup()).length} discovered`);
  setText('lbl-clus', `${Object.keys(clusterDedup()).length} in cluster`);
  const redis = state.redis || {};
  const redisStatus = String(redis.status || 'unknown').toLowerCase();
  const redisOps = redis.ops || {};
  const totalOps = Number(redisOps.selected_restore || 0)
    + Number(redisOps.selected_write || 0)
    + Number(redisOps.selected_delete || 0)
    + Number(redisOps.ui_get || 0)
    + Number(redisOps.ui_set || 0)
    + Number(redisOps.events_cache_writes || 0)
    + Number(redisOps.logs_stream_writes || 0);
  const redisLabel = redisStatus === 'connected'
    ? `redis: on (${totalOps} ops)`
    : `redis: ${redisStatus}`;
  setText('lbl-redis', redisLabel);
  const redisPip = $('pip-redis');
  if (redisPip) {
    redisPip.classList.toggle('on', redisStatus === 'connected');
    redisPip.classList.toggle('hot', redisStatus !== 'connected' && redisStatus !== 'unknown');
  }
  const redisLabelEl = $('lbl-redis');
  if (redisLabelEl) {
    const lastTs = Number(redis.last_ts || 0);
    const when = lastTs ? new Date(lastTs * 1000).toLocaleTimeString() : 'n/a';
    const lastAction = redis.last_action || 'none';
    redisLabelEl.title = `status=${redisStatus} | last=${lastAction} @ ${when}`;
  }
}

function currentAlgoOptions() {
  return dashboardMode === 'infer' ? INFER_ALGOS : TRAIN_ALGOS;
}

function syncAlgoMenu(preferredValue) {
  const select = $('algo-sel');
  if (!select) return;
  const current = preferredValue || select.value;
  const options = currentAlgoOptions();
  select.innerHTML = options.map(opt => `<option value="${opt.value}">${opt.label}</option>`).join('');
  const fallback = options[0]?.value || '';
  const next = options.some(opt => opt.value === current) ? current : fallback;
  if (next) select.value = next;
  uiSave({ algo: select.value });
}

function setDashboardMode(mode) {
  dashboardMode = mode === 'infer' ? 'infer' : 'train';
  $('mode-train').classList.toggle('active', dashboardMode === 'train');
  $('mode-infer').classList.toggle('active', dashboardMode === 'infer');
  syncAlgoMenu();
  updateButtons();
  uiSave({ mode: dashboardMode });
}

function updateButtons() {
  const running = Object.keys(state.running).length > 0;
  const hasSel  = Object.keys(state.selected).length > 0;
  $('btn-train').style.display = dashboardMode === 'train' ? '' : 'none';
  $('btn-infer').style.display = dashboardMode === 'infer' ? '' : 'none';
  $('btn-train').disabled = running || !hasSel;
  $('btn-infer').disabled = running || !hasSel;
  $('btn-conn').disabled  = !hasSel && !running;
  $('btn-stop').disabled  = !running;
  syncGenerationAvailability();
}

function isInferenceRunning() {
  return Object.values(state.running || {}).some(r => r.role === 'inference_launcher' || r.algorithm === 'infer');
}

function setBottomTab(tab) {
  bottomTab = tab === 'generation' ? 'generation' : 'logs';
  $('tab-logs').classList.toggle('active', bottomTab === 'logs');
  $('tab-generation').classList.toggle('active', bottomTab === 'generation');
  $('logs-panel').classList.toggle('hidden', bottomTab !== 'logs');
  $('generation-panel').classList.toggle('hidden', bottomTab !== 'generation');
  $('logs-actions').classList.toggle('hidden', bottomTab !== 'logs');
  $('generation-actions').classList.toggle('hidden', bottomTab !== 'generation');
  uiSave({ bottom_tab: bottomTab });
}

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
  // isInferenceRunning() checks state.running, but after a launcher script exits
  // (status: "launched") the entry remains in state.running with role "inference_launcher".
  // Also treat inferLocked=true as "live" so the button enables as soon as the launch
  // API call returns, before the first SSE tick arrives.
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
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || '';
      for (const line of lines) {
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

// ── Left panel ───────────────────────────────────────────────────────────────
function renderLeft() {
  const list  = $('disc-list');
  const nodes = discoveredDedup();
  const keys  = Object.keys(nodes);

  if (!keys.length) {
    if (!list.querySelector('.empty'))
      list.innerHTML = '<div class="empty"><div class="eico">📡</div>Scanning…</div>';
    return;
  }
  list.querySelector('.empty')?.remove();
  list.querySelectorAll('.ncard[data-h]').forEach(c => { if (!nodes[c.dataset.h]) c.remove(); });

  keys.forEach(h => {
    let c = list.querySelector(`.ncard[data-h="${CSS.escape(h)}"]`);
    if (!c) { c = buildCard(h, nodes[h]); list.appendChild(c); }
    syncCard(c, h, nodes[h]);
  });
}

function buildCard(h, n) {
  const aliasLabel = (state.ssh_aliases[h] || n.alias || h);
  const c = document.createElement('div');
  c.className = 'ncard'; c.dataset.h = h;
  c.innerHTML = `
    <div class="nhead">
      <div class="nicon">${nodeIcon({...n, alias: state.ssh_aliases[h] || n.alias || h})}</div>
      <div>
        <div class="nname">${aliasLabel}</div>
        <div class="nmeta">${[n.os,n.os_version,n.machine].filter(s=>s&&s!=='unknown').join(' · ')}</div>
      </div>
    </div>
    <div class="badges"></div>
    <div class="sshr">
      <span class="sshlbl">SSH target:</span>
      <input type="text" class="sshinput" placeholder="detecting…">
    </div>
    <button class="nbtn ncbtn">Add to Cluster</button>
  `;
  c.querySelector('.sshinput').addEventListener('input', function() {
    sshOverrides[h] = this.value.trim();
    uiSave({ ssh: { ...sshOverrides } });
  });
  c.querySelector('.ncbtn').addEventListener('click', () => {
    (h in state.selected || h in state.running) ? removeNode(h, c) : addNode(h, c);
  });
  return c;
}

function syncCard(c, h, n) {
  const isSel = h in state.selected;
  const isRun = h in state.running;

  c.className = `ncard${isRun ? ' running' : isSel ? ' in-cluster' : ''}`;

  const aliasLabel = (state.ssh_aliases[h] || n.alias || h);
  const nameEl = c.querySelector('.nname');
  if (nameEl && nameEl.textContent !== aliasLabel) nameEl.textContent = aliasLabel;

  // Update OS meta if SSH probe returned info (discovered nodes start with empty OS)
  const osInfo = state.node_os[h] || n || {};
  const metaEl = c.querySelector('.nmeta');
  if (metaEl) {
    const osStr = [osInfo.os, osInfo.os_version, osInfo.machine]
      .filter(s => s && s !== 'unknown').join(' · ');
    if (metaEl.textContent !== osStr) metaEl.textContent = osStr;
  }

  const inp   = c.querySelector('.sshinput');
  // Priority: SSH config alias > probed username > pattern guess
  const alias  = state.ssh_aliases[h];
  const probed = state.usernames[h];
  const fill   = alias || probed || '';
  if (fill && !sshOverrides[h] && inp.value !== fill) {
    inp.value = fill; inp.placeholder = fill;
  } else if (!fill && !sshOverrides[h]) {
    inp.placeholder = guessUser(h) || (probed === null ? 'detecting…' : 'user or alias');
  }

  const badges = c.querySelector('.badges');
  if (isRun) {
    const r = state.running[h] || {};
    const label = (r.role === 'inference_launcher' || r.algorithm === 'infer')
      ? 'inferring'
      : 'training';
    badges.innerHTML = `<span class="nbadge active">● ${label}</span>`;
  } else if (isSel) {
    badges.innerHTML = `<span class="nbadge cluster">✓ in cluster</span>`;
  } else {
    badges.innerHTML = `<span class="nbadge avail">available</span>`;
  }

  const btn = c.querySelector('.ncbtn');
  if (!btn.disabled) {
    const inC = isSel || isRun;
    btn.textContent = inC ? 'Remove' : 'Add to Cluster';
    btn.className   = `nbtn${inC ? ' rm' : ''} ncbtn`;
  }
}

// ── Status bar (connectivity / inference) ────────────────────────────────────
function renderConnBar() {
  const bar  = $('status-bar');
  const c    = state.connectivity;
  if (!c || !c.mode) { bar.classList.remove('show'); return; }

  bar.classList.add('show');
  setText('status-msg', c.message || '');
  $('status-spin').style.display = c.status === 'done' || c.status === 'ready' ? 'none' : '';

  if (c.mode === 'connectivity' && c.results) {
    const allHosts = Object.keys(state.selected);
    const chips = $('conn-chips');
    chips.innerHTML = '';
    allHosts.forEach(h => {
      const r   = (c.results || []).find(x => x.hostname === h);
      const cls = !r ? 'wait' : r.status === 'ok' ? 'ok' : 'fail';
      const lbl = !r ? '…' : r.status === 'ok' ? `${r.ms}ms` : r.status;
      const name = state.ssh_aliases[h] || state.discovered[h]?.alias || h;
      chips.innerHTML += `<span class="cchip ${cls}"><span class="cdot"></span>${name}  ${lbl}</span>`;
    });
  } else {
    $('conn-chips').innerHTML = '';
  }
}

// ── Training metrics ──────────────────────────────────────────────────────────
function renderMetrics() {
  const tRaw = state.training || {};
  // Merge: fallback first, then state.training — but skip null/undefined values
  // from the API so they never shadow valid fallback metrics. (Nulls appear when
  // the server writes NaN floats or when a race-condition produces a partial read.)
  const t = { ...trainingFallbackMetrics };
  for (const [k, v] of Object.entries(tRaw)) {
    if (v !== null && v !== undefined) t[k] = v;
  }
  const pickMetricNumber = (obj, keys) => {
    for (const key of keys) {
      const v = obj[key];
      if (v === null || v === undefined) continue;
      const n = Number(v);
      if (Number.isFinite(n)) return n;
    }
    return null;
  };
  const pickMetricText = (obj, keys) => {
    for (const key of keys) {
      const v = obj[key];
      if (v === null || v === undefined) continue;
      const s = String(v).trim();
      if (s) return s;
    }
    return '';
  };
  const fmtTput = (n) => {
    if (!Number.isFinite(n)) return '—';
    if (n >= 1000) return Math.round(n).toLocaleString();
    if (n >= 100) return n.toFixed(1);
    return n.toFixed(2);
  };
  const tputIn = pickMetricNumber(t, [
    'tok_sec_in', 'tokens_per_sec_in', 'tps_in', 'input_tps', 'throughput_in', 'input_throughput',
  ]);
  const tputGeneric = pickMetricNumber(t, ['throughput', 'tok_sec', 'tokens_per_sec', 'tps']);
  let tputOut = pickMetricNumber(t, [
    'tok_sec_out', 'tokens_per_sec_out', 'tps_out', 'output_tps', 'throughput_out', 'output_throughput',
  ]);
  if (tputOut === null) {
    tputOut = tputGeneric;
  }
  const tputInResolved = tputIn !== null ? tputIn : (tputOut !== null ? tputOut : tputGeneric);
  const tputOutResolved = tputOut !== null ? tputOut : (tputIn !== null ? tputIn : tputGeneric);
  const etaText = pickMetricText(t, ['eta_tqdm', 'tqdm_eta', 'eta_remaining', 'eta', 'remaining', 'remaining_time']);
  const runningVals = Object.values(state.running || {});
  const isTrainingRunning = runningVals.some(r => r.role === 'server' || r.role === 'worker' || r.role === 'training_launcher');
  const has = isTrainingRunning && (
    t.step != null || t.loss != null || t.throughput != null || t.grad_norm != null
    || tputInResolved !== null || tputOutResolved !== null || !!etaText
  );
  $('metrics-strip').classList.toggle('show', has);
  if (!has) return;
  setText('m-loss', t.loss       != null ? Number(t.loss).toFixed(4) : '—');
  setText('m-tput-in', fmtTput(tputInResolved));
  setText('m-tput-out', fmtTput(tputOutResolved));
  setText('m-step', t.step       != null ? t.step.toLocaleString()   : '—');
  setText('m-eta', etaText || '—');
  setText('m-gn',   t.grad_norm  != null ? (isNaN(+t.grad_norm) ? String(t.grad_norm) : (+t.grad_norm).toFixed(3)) : '—');
  if (t.lr != null) {
    const lrVal = Number(t.lr);
    setText('m-lr', lrVal < 0.001 ? lrVal.toExponential(2) : lrVal.toPrecision(3));
  } else {
    setText('m-lr', '—');
  }
  const totalSteps = t.total_steps ?? t.max_steps ?? t.steps_total ?? null;
  setText('m-step-sub', totalSteps ? `of ${totalSteps.toLocaleString()}` : '');
  const hasProg = t.step != null && totalSteps;
  $('prog-wrap').style.display = hasProg ? '' : 'none';
  if (hasProg) {
    const pct = Math.min(100, (t.step / totalSteps * 100));
    $('prog-fill').style.width = pct.toFixed(2) + '%';
    setText('prog-lbl', etaText
      ? `Step ${t.step.toLocaleString()} / ${totalSteps.toLocaleString()} · ETA ${etaText}`
      : `Step ${t.step.toLocaleString()} / ${totalSteps.toLocaleString()}`);
    setText('prog-pct', pct < 1 ? pct.toFixed(2) + '%' : Math.round(pct) + '%');
  }
}

// ── Log terminal ─────────────────────────────────────────────────────────────
// Pre-populate the filter dropdown from selected+running nodes (don't wait for log lines)
function syncLogFilter() {
  const allNodes = {...state.selected, ...state.running};
  for (const h of Object.keys(allNodes)) {
    knownLogHosts.add(h);
    ensureLogFilterOption(h);
  }
  refreshLogHostLabels();
}

function appendLog({ hostname, line }) {
  // Persist log entry to Redis-backed buffer
  persistLog({ hostname, line });

  // Fast-path: structured metrics JSON emitted by the training server.
  // Format: "[SMOL_METRICS] {\"step\":50,\"grad_norm\":1.23,...}"
  // This rides the SSH-tailed log stream so metrics reach the dashboard
  // even when the training server runs on a different machine.
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

  const lowerLine = String(line || '').toLowerCase();

  // Optional structured training transport marker.
  // Any algorithm/file can emit:
  //   [SMOL_IO] {"phase":"request"}
  //   [SMOL_IO] {"phase":"response"}
  // and the topology will animate coordinator<->workers with measured RTT.
  const smolIoM = line.match(/\[SMOL_IO\]\s*(\{.+\})/);
  if (smolIoM) {
    try {
      const io = JSON.parse(smolIoM[1]);
      const phase = String(io.phase || io.event || io.type || '').toLowerCase();
      if (phase === 'request' || phase === 'req' || phase === 'outbound' || phase === 'send') {
        _markTrainIoRequestEvent();
      } else if (phase === 'response' || phase === 'resp' || phase === 'inbound' || phase === 'recv' || phase === 'receive') {
        _markTrainIoResponseEvent();
      }
    } catch (e) {}
  }

  // Training transport events (coordinator -> workers request, workers -> coordinator response).
  // This is algorithm-agnostic and driven by live log lines when available.
  const looksLikeRequest =
    (/\brequest\b/i.test(line) && /\bn=\d+/i.test(line) && /vllm|worker/i.test(line))
    || /request n=\d+ comple/i.test(lowerLine)
    || /\b(dispatch|sending|send|submit|submitted)\b.*\b(prompt|rollout|batch|request|rpc)\b/.test(lowerLine)
    || /\b(prompt|rollout|batch|request|rpc)\b.*\b(to|->)\b.*\b(worker|server|rank|node)\b/.test(lowerLine);
  const looksLikeResponse =
    /got\s+\d+\/\d+\s+non-empty\s+completion/i.test(lowerLine)
    || /all workers done\.\s*\d+\s+non-empty/i.test(lowerLine)
    || /received\s+\d+\s+usable/i.test(lowerLine)
    || /\b(received|recv|returned|response|reply)\b.*\b(result|completion|rollout|batch|token|output)\b/.test(lowerLine)
    || /\b(result|completion|rollout|batch|output)\b.*\b(from|<-|back)\b.*\b(worker|server|rank|node)\b/.test(lowerLine);

  if (looksLikeRequest) {
    _markTrainIoRequestEvent();
  }
  if (looksLikeResponse) {
    _markTrainIoResponseEvent();
  }

  const isTrainingLog = /step|loss|tok\/s|gradient norm|grad_norm|\blr\b|it\/s|eta/.test(lowerLine);
  if (isTrainingLog) {
    let dirtyTrainingFallback = false;

    const stepTotalMatch = line.match(/\[step\s+(\d+)\s*\/\s*(\d+)\]/i) || line.match(/step:\s*(\d+)\s*\/\s*(\d+)/i);
    if (stepTotalMatch) {
      trainingFallbackMetrics.step = Number(stepTotalMatch[1]);
      trainingFallbackMetrics.total_steps = Number(stepTotalMatch[2]);
      dirtyTrainingFallback = true;
    } else {
      const stepOnlyMatch = line.match(/\[step\s+(\d+)\]/i) || line.match(/step[:=]\s*(\d+)/i);
      if (stepOnlyMatch) {
        trainingFallbackMetrics.step = Number(stepOnlyMatch[1]);
        dirtyTrainingFallback = true;
      }
    }

    const lossMatch = line.match(/(?:leader\s+loss|training\s+loss|worker\s+\d+\s+loss|last rank mixtral loss|loss)\s*[:=]\s*([0-9]*\.?[0-9]+)/i);
    if (lossMatch) {
      trainingFallbackMetrics.loss = Number(lossMatch[1]);
      dirtyTrainingFallback = true;
    }

    // Match both "Gradient Norm: 1.23" (server log) and "grad_norm=1.23" (tqdm postfix)
    const gradMatch = line.match(/gradient norm(?: before clipping)?\s*[:=]\s*([0-9]*\.?[0-9]+)/i)
                   || line.match(/\bgrad_norm[:=]\s*([0-9]*\.?[0-9]+)/i);
    if (gradMatch) {
      trainingFallbackMetrics.grad_norm = Number(gradMatch[1]);
      dirtyTrainingFallback = true;
    }

    const tputInMatch =
      line.match(/tok\/(?:sec|s)\s*\(in\)\s*[:=]?\s*([0-9]*\.?[0-9]+)/i)
      || line.match(/\bin(?:put)?\s+tok\/(?:sec|s)\s*[:=]?\s*([0-9]*\.?[0-9]+)/i);
    if (tputInMatch) {
      trainingFallbackMetrics.tok_sec_in = Number(tputInMatch[1]);
      dirtyTrainingFallback = true;
    }

    const tputOutMatch =
      line.match(/tok\/(?:sec|s)\s*\(out\)\s*[:=]?\s*([0-9]*\.?[0-9]+)/i)
      || line.match(/\bout(?:put)?\s+tok\/(?:sec|s)\s*[:=]?\s*([0-9]*\.?[0-9]+)/i);
    if (tputOutMatch) {
      trainingFallbackMetrics.tok_sec_out = Number(tputOutMatch[1]);
      dirtyTrainingFallback = true;
    }

    const tputMatch = line.match(/tok\/(?:sec|s)[^0-9]*([0-9]*\.?[0-9]+)/i) || line.match(/([0-9]*\.?[0-9]+)\s*tok\/(?:sec|s)/i);
    if (tputMatch) {
      const _t = Number(tputMatch[1]);
      trainingFallbackMetrics.throughput = _t;
      if (trainingFallbackMetrics.tok_sec_in == null) {
        trainingFallbackMetrics.tok_sec_in = _t;
      }
      if (trainingFallbackMetrics.tok_sec_out == null) {
        trainingFallbackMetrics.tok_sec_out = _t;
      }
      dirtyTrainingFallback = true;
    }

    const tqdmEtaMatch = line.match(/\b\d+%\|[^|]*\|\s*\d+\/\d+\s*\[[^\]]*?<\s*([^,\]\s]+)\s*,/);
    if (tqdmEtaMatch) {
      const etaText = String(tqdmEtaMatch[1] || '').trim();
      if (etaText) {
        trainingFallbackMetrics.eta_tqdm = etaText;
        dirtyTrainingFallback = true;
      }
    }

    const etaMatch =
      line.match(/\beta(?:\s*remaining)?\s*[:=]\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)/i)
      || line.match(/\[[^\]]*?<\s*([^,\]]+)/);
    if (etaMatch) {
      const etaText = String(etaMatch[1]).trim().replace(/^<+/, '');
      if (etaText) {
        // Keep tqdm ETA as preferred source when available.
        if (!trainingFallbackMetrics.eta_tqdm) {
          trainingFallbackMetrics.eta_remaining = etaText;
        }
        dirtyTrainingFallback = true;
      }
    }

    const lrMatch = line.match(/\bLR[:=]\s*([0-9eE+\-.]+)/i);
    if (lrMatch) {
      const lrVal = Number(lrMatch[1]);
      if (isFinite(lrVal)) {
        trainingFallbackMetrics.lr = lrVal;
        dirtyTrainingFallback = true;
      }
    }

    if (dirtyTrainingFallback) {
      trainingFallbackMetrics.algorithm ||= _activeAlgo || $('algo-sel').value;
      renderMetrics();
    }
  }

  // In inference, grab token text from ANY host's log — the token-generating
  // process (worker/server) may differ from selectedServer.
  const runningVals = Object.values(state.running || {});
  const isInferringNow = runningVals.some(r => r.role === 'inference_launcher' || r.algorithm === 'infer');
  if (isInferringNow) {
    // Prefer direct JSON parse of raw SSE token packets logged by the server:
    //   data: {"token": " parameter", "done": false}
    // This preserves leading spaces, newlines, and all escape sequences exactly.
    const sseM = line.match(/^data:\s*(\{.+\})\s*$/);
    if (sseM) {
      try {
        const obj = JSON.parse(sseM[1]);
        if (typeof obj.token === 'string') {
          // Preserve token bytes exactly but reveal in sync with packet animation.
          _queueToken(obj.token);
        }
      } catch(e) {}
    } else if (line.trim().length > 0) {
      // Fallback regex for other non-SSE log-line formats
      const m = line.match(/\[Worker\s*\d+\]\s*Token\s+\d+\s*:\s*(.+)$/i)
        || line.match(/\bStreamed\s+token\s+\d+\s*:\s*(.+)$/i)
        || line.match(/\btoken\s+\d+\s*:\s*(.+)$/i);
      if (m) {
        const raw = (m[1] || '').trim().replace(/^repr\(['"]/,'').replace(/['"]\)$/,'').replace(/^['"]+|['"]+$/g, '');
        if (raw.length > 0) {
          _queueToken(raw);
        }
      }
    }
  }
  if (!knownLogHosts.has(hostname)) {
    knownLogHosts.add(hostname);
  }
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

  let dragging = false;
  let startY = 0;
  let startBottom = 0;

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

// ── Actions ───────────────────────────────────────────────────────────────────
async function addNode(h, card) {
  const btn  = card.querySelector('.ncbtn');
  const user = sshOverrides[h] || state.ssh_aliases[h] || state.usernames[h] || guessUser(h) || '';
  btn.disabled = true; btn.textContent = 'Adding…';
  try {
    // Deselect any already-selected node that maps to the same IP (stale duplicate)
    const myIp = state.discovered[h]?.ip || '';
    if (myIp) {
      for (const sel of Object.keys(state.selected)) {
        if (sel !== h && state.discovered[sel]?.ip === myIp) {
          await fetch(`/api/nodes/${sel}/deselect`, { method: 'POST' }).catch(() => {});
        }
      }
    }
    const r = await fetch(`/api/nodes/${h}/select`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ ssh_user: user }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: 'Unknown error'}));
      console.error('Add node failed:', err.detail);
      btn.textContent = '⚠ Failed';
      return;
    }
    btn.disabled = false; btn.textContent = 'Added';
  } catch (e) {
    console.error('Add node error:', e);
    btn.textContent = '⚠ Error';
  }
}


async function removeNode(h, card) {
  if (!card) return;
  const btn = card.querySelector('.ncbtn');
  btn.disabled = true; btn.textContent = 'Removing…';
  try {
    const r = await fetch(`/api/nodes/${h}/deselect`, {method:'POST'});
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: 'Unknown error'}));
      console.error('Remove node failed:', err.detail);
      btn.textContent = '⚠ Failed';
      return;
    }
    btn.disabled = false; btn.textContent = 'Removed';
  } catch (e) {
    console.error('Remove node error:', e);
    btn.textContent = '⚠ Error';
  }
}

async function startTraining() {
  setDashboardMode('train');
  if (!Object.keys(state.selected).length) { alert('Add nodes first.'); return; }
  clearLogs();
  const algo  = $('algo-sel').value;
  trainingFallbackMetrics = { algorithm: algo };
  $('prog-wrap').style.display = 'none';
  // Reset to canonical default layout for these well-defined archetypes.
  // User can drag nodes after start; positions will persist for that session.
  if (['syncps', 'classicdp', 'grpo'].includes(algo)) _manualNodePos.clear();
  const hosts = Object.keys(state.selected);
  // Use user-picked server (click on node), fall back to lowest-rank node
  const srv = (selectedServer && state.selected[selectedServer])
    ? selectedServer
    : hosts.reduce((a, b) => state.selected[a].rank <= state.selected[b].rank ? a : b);
  selectedServer = srv;
  const r = await fetch('/api/training/launch', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ algorithm: algo, server_hostname: srv }),
  });
  if (r.ok) {
    inferLocked = true;
  } else {
    alert(`Error: ${(await r.json()).detail}`);
  }
}
let selectedServer = null;   // hostname of user-picked server node
let inferLocked    = false;  // true while inference is running — blocks server re-selection

async function startInference() {
  setDashboardMode('infer');
  if (!Object.keys(state.selected).length) { alert('Add nodes first.'); return; }
  clearLogs();
  const algo = $('algo-sel').value;
  // Reset token UI state so stale previous-run tokens do not appear immediately.
  _lastTokenStamp = Number(state.token_ts) || 0;
  _lastTokenRaw = '';
  _lastTokenText = '';
  _pendingTokens = [];
  _tokenDockActive = false;
  // Reset to canonical default layout for these well-defined archetypes.
  if (['syncps', 'classicdp'].includes(algo)) _manualNodePos.clear();

  // Respect user-picked server; fallback to lowest-rank selected node
  const hosts = Object.keys(state.selected);
  let srv = selectedServer;
  if (!srv || !state.selected[srv]) {
    srv = hosts.reduce((a, b) => state.selected[a].rank <= state.selected[b].rank ? a : b);
    selectedServer = srv;
  }
  generationAutoFillFromSelectedNode(true);

  const r = await fetch('/api/inference/launch', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ algorithm: algo, server_hostname: srv }),
  });
  if (r.ok) {
    inferLocked = true;
    syncGenerationAvailability();  // enable Send immediately without waiting for SSE
  } else {
    alert(`Error: ${(await r.json()).detail}`);
  }
}
async function checkConn() {
  const r = await fetch('/api/connectivity/check', { method:'POST' });
  if (!r.ok) alert(`Error: ${(await r.json()).detail}`);
}

function resetTopologyLayout() {
  _manualNodePos.clear();
}

async function stopAll() {
  // Close SSE streams to stop receiving log and state updates
  if (ssEventSource) { ssEventSource.close(); ssEventSource = null; }
  if (logsEventSource) { logsEventSource.close(); logsEventSource = null; }
  stopGenerationRequest();
  
  await fetch('/api/training/stop',  { method:'POST' });
  await fetch('/api/inference/stop', { method:'POST' });
  inferLocked    = false;
  selectedServer = null;
  trainingFallbackMetrics = {};
  clearLogs();
  
  // Restart streams after a brief delay
  setTimeout(() => {
    if (!ssEventSource) startSSE();
    if (!logsEventSource) startLogs();
  }, 500);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function $(id)      { return document.getElementById(id); }
function setText(id, v) { const e=$(id); if(e && e.textContent!==String(v)) e.textContent=v; }
function guessUser(h) { return ''; }
function escHtml(s)   { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function ansiToHtml(s) {
  const text = String(s || '');
  const parts = text.split(/(\x1b\[[0-9;]*m)/g);
  let bold = false;
  let fg = null;
  let out = '';

  const openSpan = () => {
    const classes = [];
    if (bold) classes.push('ansi-bold');
    if (fg !== null) classes.push(`ansi-fg-${fg}`);
    if (!classes.length) return '';
    return `<span class="${classes.join(' ')}">`;
  };

  for (const part of parts) {
    const m = part.match(/^\x1b\[([0-9;]*)m$/);
    if (m) {
      const codes = (m[1] || '0').split(';').filter(Boolean).map(n => Number(n));
      if (!codes.length) codes.push(0);
      for (const code of codes) {
        if (code === 0) {
          bold = false;
          fg = null;
        } else if (code === 1) {
          bold = true;
        } else if (code === 22) {
          bold = false;
        } else if ((code >= 30 && code <= 37) || (code >= 90 && code <= 97)) {
          fg = code;
        } else if (code === 39) {
          fg = null;
        }
      }
      continue;
    }

    if (!part) continue;
    const safe = escHtml(part);
    const opener = openSpan();
    out += opener ? `${opener}${safe}</span>` : safe;
  }

  return out;
}
function nodeIcon(n) {
  const os=((n.os||'').toLowerCase());
  const h=((n.hostname||'').toLowerCase());
  const a=((n.alias||'').toLowerCase());
  const any = h + ' ' + a;
  if(any.includes('ipad')) return '📱';
  if(any.includes('pi')||any.includes('rasp')) return '🍓';
  if(os==='darwin'||any.includes('mac')||any.includes('mini')) return '💻';
  if(os==='windows'||os==='win32'||any.includes('win')) return '🖥️';
  if(os==='linux'||os==='ubuntu'||os==='debian'||os==='fedora') return '🐧';
  if(any.includes('jetson')||any.includes('nano')||any.includes('xavier')) return '🐧';
  return '💻';
}

// ════════════════════════════════════════════════════════════════════════════
// ═══ 3-D Topology — Three.js r128 ══════════════════════════════════════════
// ════════════════════════════════════════════════════════════════════════════
let particles      = [];
let spawnTs        = 0;
let inferParticles = [];
let inferSpawnTs   = 0;
let _activeAlgo    = '';
let _prevGradTs      = 0;   // last grad_ts value we saw
let _gradIntervalMs  = 3000; // measured ms between successive grad pings (starts at 3s)
let _prevTokenTs     = 0;   // last token_ts value we saw
let _tokenIntervalMs = 200; // measured ms between successive token pings (starts at 200ms)
let _trainIoReqTs       = 0;   // last observed training request->worker event (epoch ms)
let _trainIoRespTs      = 0;   // last observed training response<-worker event (epoch ms)
let _trainIoRttMs       = 650; // measured request/response round trip for training transport
let _trainIoIntervalMs  = 700; // measured interval between successive training requests
let _lastTrainIoReqPerf = 0;
let _lastTrainIoRespPerf = 0;
let _trainIoPendingReq = 0;
let _trainIoPendingResp = 0;

function _markTrainIoRequestEvent() {
  const nowPerf = performance.now();
  if (_lastTrainIoReqPerf > 0) {
    const measuredReq = nowPerf - _lastTrainIoReqPerf;
    if (measuredReq > 60 && measuredReq < 30000) {
      _trainIoIntervalMs = _trainIoIntervalMs * 0.6 + measuredReq * 0.4;
    }
  }
  _lastTrainIoReqPerf = nowPerf;
  _trainIoReqTs = Date.now();
  _trainIoPendingReq += 1;
}

function _markTrainIoResponseEvent() {
  const nowPerf = performance.now();
  if (_lastTrainIoReqPerf > 0) {
    const measured = nowPerf - _lastTrainIoReqPerf;
    if (measured > 80 && measured < 15000) {
      _trainIoRttMs = _trainIoRttMs * 0.55 + measured * 0.45;
    }
  }
  _lastTrainIoRespPerf = nowPerf;
  _trainIoRespTs = Date.now();
  _trainIoPendingResp += 1;
}

// ── Scene setup ───────────────────────────────────────────────────────────────
const _T3mount = $('topo3d');
const _T3scene = new THREE.Scene();
const _T3camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
_T3camera.position.set(0, 10, 12);
_T3camera.lookAt(0, 0, 0);
const _T3renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
_T3renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
_T3renderer.setClearColor(0x000000, 0);
_T3mount.appendChild(_T3renderer.domElement);

// ── Lighting ───────────────────────────────────────────────────────────────────
_T3scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const _T3sun = new THREE.DirectionalLight(0xffffff, 0.7);
_T3sun.position.set(6, 12, 8); _T3scene.add(_T3sun);
const _T3fill = new THREE.DirectionalLight(0x88ccee, 0.25);
_T3fill.position.set(-5, -2, -6); _T3scene.add(_T3fill);

// ── Grid ──────────────────────────────────────────────────────────────────────
const _T3grid = new THREE.GridHelper(28, 22, 0x1a3a38, 0x1a3a38);
_T3grid.material.opacity = 0.06;
_T3grid.material.transparent = true;
_T3scene.add(_T3grid);

// ── Orbit controls ─────────────────────────────────────────────────────────────
const _T3orbit = new THREE.OrbitControls(_T3camera, _T3renderer.domElement);
_T3orbit.enableDamping = true;
_T3orbit.dampingFactor = 0.085;
_T3orbit.minDistance = 4;
_T3orbit.maxDistance = 40;
_T3orbit.maxPolarAngle = Math.PI * 0.48;

// ── Resize ─────────────────────────────────────────────────────────────────────
function _t3Resize() {
  const w = _T3mount.clientWidth || 600;
  const h = _T3mount.clientHeight || 400;
  _T3renderer.setSize(w, h);
  _T3camera.aspect = w / h;
  _T3camera.updateProjectionMatrix();
}
new ResizeObserver(_t3Resize).observe(_T3mount);
setTimeout(_t3Resize, 0);

// ── Custom drag (XZ-plane) ─────────────────────────────────────────────────────
const _T3ray     = new THREE.Raycaster();
const _T3ptr     = new THREE.Vector2();
const _dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const _dragHit   = new THREE.Vector3();
let   _dragEntry = null;
const _manualNodePos = new Map(); // hostname -> THREE.Vector3 world pos pinned by drag

function _setT3Ptr(e) {
  const el = _T3renderer.domElement;
  _T3ptr.x =  (e.offsetX / el.clientWidth)  * 2 - 1;
  _T3ptr.y = -(e.offsetY / el.clientHeight) * 2 + 1;
}
_T3renderer.domElement.addEventListener('pointerdown', e => {
  _setT3Ptr(e);
  _T3ray.setFromCamera(_T3ptr, _T3camera);
  const hit = _T3ray.intersectObjects([...nodeMeshes.values()].map(n => n.sphere), false)[0];
  if (hit) {
    const entry = [...nodeMeshes.values()].find(n => n.sphere === hit.object);
    if (entry) {
      entry.isDragged = false;
      _T3ray.ray.intersectPlane(_dragPlane, _dragHit);
      _dragEntry = { entry, ox: _dragHit.x - entry.group.position.x, oz: _dragHit.z - entry.group.position.z };
      _T3orbit.enabled = false;
      _T3renderer.domElement.style.cursor = 'grabbing';
    }
  }
});
_T3renderer.domElement.addEventListener('pointermove', e => {
  _setT3Ptr(e);
  if (_dragEntry) {
    _T3ray.setFromCamera(_T3ptr, _T3camera);
    _T3ray.ray.intersectPlane(_dragPlane, _dragHit);
    _dragEntry.entry.isDragged = true;
    _dragEntry.entry.group.position.set(_dragHit.x - _dragEntry.ox, 0, _dragHit.z - _dragEntry.oz);
    _manualNodePos.set(_dragEntry.entry.h, _dragEntry.entry.group.position.clone());
    _T3renderer.domElement.style.cursor = 'grabbing';
  } else {
    _T3ray.setFromCamera(_T3ptr, _T3camera);
    const hit = _T3ray.intersectObjects([...nodeMeshes.values()].map(n => n.sphere), false)[0];
    _T3renderer.domElement.style.cursor = hit ? 'grab' : 'default';
  }
});
_T3renderer.domElement.addEventListener('pointerup', e => {
  const wasDrag = _dragEntry?.entry.isDragged;
  _dragEntry = null;
  _T3orbit.enabled = true;
  _T3renderer.domElement.style.cursor = 'default';
  // Use pointerup for crown selection; after launch this only changes the UI focus,
  // not the already-running server/coordinator choice.
  if (wasDrag) return;
  _setT3Ptr(e);
  _T3ray.setFromCamera(_T3ptr, _T3camera);
  const hit = _T3ray.intersectObjects([...nodeMeshes.values()].map(n => n.sphere), false)[0];
  if (hit) {
    const entry = [...nodeMeshes.values()].find(n => n.sphere === hit.object);
    if (entry) {
      selectedServer = (selectedServer === entry.h) ? null : entry.h;
      generationAutoFillFromSelectedNode(true);
    }
  }
});
_T3renderer.domElement.addEventListener('click', e => {
  // Handled in pointerup; keeping listener to prevent bubbling issues only
});

// ── Layout (virtual 2D coords → Three.js XZ world) ────────────────────────────
const _VC_W = 600, _VC_H = 420;

function getLayout() {
  const W = _VC_W, H = _VC_H;
  const raw = {};
  for (const [h,v] of Object.entries(state.selected)) raw[h] = {...v, running:false};
  for (const [h,v] of Object.entries(state.running))  raw[h] = {...v, running:true};
  const all = dedupByIp(raw);

  const algo = _activeAlgo || $('algo-sel').value;
  const hasServer = !['classicdp','fsdp','ep','mp_pipeline'].includes(algo);
  const isSyncPsStyle = (algo === 'syncps' || algo === 'grpo');

  let server = null; const workers = [];
  for (const [h, info] of Object.entries(all)) {
    if (hasServer && (info.rank === 0 || info.role === 'server')) server = {h, ...info};
    else workers.push({h, ...info});
  }
  workers.sort((a,b) => a.rank - b.rank);

  const sx = W/2, sy = H * 0.22;

  // Non-ClassicDP flat topologies with 3+ nodes: circular polygon mesh
  if (!hasServer && algo !== 'classicdp' && workers.length >= 3) {
    const cx = W/2, cy = H/2;
    const radius = Math.min(W * 0.3, H * 0.34, 160);
    const a0 = -Math.PI / 2;
    return {
      W, H, hasServer, server: null,
      workers: workers.map((w, i) => {
        const angle = a0 + (2 * Math.PI * i) / workers.length;
        return {...w, x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle)};
      }),
    };
  }

  // ClassicDP: horizontal chain (linear line of nodes, all-to-all mesh connections).
  // SyncPS/GRPO: triangle — server at apex, workers fanned below in pyramid formation.
  // Other server topologies: server top, workers spread below.
  const nw = workers.length;
  
  if (isSyncPsStyle && hasServer && nw > 1) {
    // SyncPS/GRPO: triangular pyramid with server at top center, workers fanned below
    const wy1 = H * 0.65;  // first row (lower)
    const wy2 = H * 0.80;  // second row (bottom)
    const spBottom = Math.min(240, (W - 60) / Math.max(nw - 1, 1));
    
    const wpos = [];
    if (nw <= 3) {
      // 1-3 workers: spread across bottom row
      const wx0 = W/2 - (nw - 1) * spBottom / 2;
      for (let i = 0; i < nw; i++) {
        wpos.push({...workers[i], x: wx0 + i * spBottom, y: wy2});
      }
    } else {
      // 4+ workers: split into two rows for pyramid effect
      const bottomCount = Math.ceil(nw / 2);
      const topCount = nw - bottomCount;
      const spBot = Math.min(240, (W - 60) / (bottomCount - 1));
      const spTop = Math.min(160, (W - 60) / Math.max(topCount - 1, 1));
      
      // Bottom row (more workers, wider spread)
      const wx0Bot = W/2 - (bottomCount - 1) * spBot / 2;
      for (let i = 0; i < bottomCount; i++) {
        wpos.push({...workers[i], x: wx0Bot + i * spBot, y: wy2});
      }
      // Top row (fewer workers, narrower spread)
      const wx0Top = W/2 - (topCount - 1) * spTop / 2;
      for (let i = 0; i < topCount; i++) {
        wpos.push({...workers[bottomCount + i], x: wx0Top + i * spTop, y: wy1});
      }
    }
    return {
      W, H, hasServer,
      server: server ? {...server, x: sx, y: sy} : null,
      workers: wpos,
    };
  }
  
  // Default layout (ClassicDP linear chain, or other server topologies)
  const wy = hasServer ? H * 0.74 : H * 0.5;
  const spMax = Math.min(155, (W - 100) / Math.max(nw - 1, 1));
  const sp = nw > 1 ? spMax : 0;
  const wx0 = W/2 - (nw - 1) * sp / 2;

  return {
    W, H, hasServer,
    server: server ? {...server, x: sx, y: sy} : null,
    workers: workers.map((w, i) => ({...w, x: wx0 + i * sp, y: wy})),
  };
}

// Layout pixel → Three.js world XZ
function _toWorld(px, py) {
  return new THREE.Vector3((px - _VC_W/2) / 40, 0, (py - _VC_H/2) / 40);
}

function _activeCrownedHost(server, workers) {
  if (selectedServer && (state.selected[selectedServer] || state.running[selectedServer] || state.discovered[selectedServer])) {
    return selectedServer;
  }
  if (server?.h) return server.h;
  if (workers && workers.length) {
    const rankZero = workers.find(w => Number(w.rank) === 0);
    return rankZero?.h || workers[0].h;
  }
  return null;
}

// ── Label sprite (canvas texture) ─────────────────────────────────────────────
function _makeLabel(label, sub, hexCol) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 90;
  const ctx = c.getContext('2d');
  ctx.fillStyle = hexCol;
  ctx.font = 'bold 26px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillText(label, 128, 36);
  ctx.fillStyle = '#5a7a77';
  ctx.font = '18px "IBM Plex Mono", monospace';
  ctx.fillText(sub.length > 15 ? sub.slice(0, 14) + '\u2026' : sub, 128, 64);
  const tex = new THREE.CanvasTexture(c);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sprite.scale.set(2.6, 0.86, 1);
  return { sprite, tex };
}

// ── Crown sprite helper ───────────────────────────────────────────────────────
// (crown and token are HTML overlay divs — no canvas texture needed)

// Track last token text from logs (set in appendLog)
let _lastTokenText = '';
let _lastTokenRaw  = '';
let _tokenFlashTs  = -999999; // performance.now() when last token arrived
let _lastTokenStamp = 0;      // last seen state.token_ts from SSE
let _pendingTokens = [];      // queued SSE tokens waiting for packet-arrival reveal
const _tokenFallbackDelayMs = 760;
const nodeMeshes = new Map();

function _ensureNode(h, isServer) {
  if (nodeMeshes.has(h)) {
    const entry = nodeMeshes.get(h);
    if (entry.isServer !== isServer) {
      const hexCol = isServer ? '#d16930' : '#167d90';
      const col3   = isServer ? 0xd16930 : 0x167d90;
      const emis3  = isServer ? 0x6b2810 : 0x073c44;
      entry.isServer = isServer;
      entry.hexCol = hexCol;
      entry.mat.color.set(col3);
      entry.mat.emissive.set(emis3);
      if (entry.ringMatA && entry.ringMatB) {
        entry.ringMatA.color.set(col3);
        entry.ringMatA.emissive.set(col3);
        entry.ringMatB.color.set(col3);
        entry.ringMatB.emissive.set(col3);
      }
    }
    return entry;
  }
  const hexCol = isServer ? '#d16930' : '#167d90';
  const col3   = isServer ? 0xd16930 : 0x167d90;
  const emis3  = isServer ? 0x6b2810 : 0x073c44;
  const r      = isServer ? 0.72 : 0.6;
  const group  = new THREE.Group();

  const mat = new THREE.MeshPhongMaterial({
    color: col3, emissive: emis3, emissiveIntensity: 0.28,
    shininess: 90, transparent: true, opacity: 0.92,
  });
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 48), mat);
  group.add(sphere);

  // Back-face glow shell
  group.add(new THREE.Mesh(
    new THREE.SphereGeometry(r + 0.14, 32, 32),
    new THREE.MeshPhongMaterial({ color: col3, emissive: col3, emissiveIntensity: 0.1,
      transparent: true, opacity: 0.07, side: THREE.BackSide, depthWrite: false })
  ));

  // Dual animated rings (processing aura)
  const ringMatA = new THREE.MeshPhongMaterial({
    color: col3, emissive: col3, emissiveIntensity: 0.12, transparent: true, opacity: 0.24
  });
  const ringA = new THREE.Mesh(new THREE.TorusGeometry(r + 0.2, 0.028, 14, 80), ringMatA);
  ringA.rotation.x = Math.PI / 2;
  group.add(ringA);

  const ringMatB = new THREE.MeshPhongMaterial({
    color: col3, emissive: col3, emissiveIntensity: 0.08, transparent: true, opacity: 0.16
  });
  const ringB = new THREE.Mesh(new THREE.TorusGeometry(r + 0.28, 0.02, 14, 80), ringMatB);
  ringB.rotation.x = Math.PI / 2;
  ringB.rotation.y = Math.PI / 5;
  group.add(ringB);

  const { sprite, tex } = _makeLabel('\u2026', h, hexCol);
  sprite.position.y = r + 0.92;
  group.add(sprite);

  _T3scene.add(group);
  const entry = { group, sphere, mat, h, isServer, hexCol, isDragged: false,
                  labelSprite: sprite, labelTex: tex, lastLabel: '',
                  ringA, ringB, ringMatA, ringMatB };
  nodeMeshes.set(h, entry);
  return entry;
}

function _removeStaleNodes(keepSet) {
  for (const [h, entry] of nodeMeshes) {
    if (!keepSet.has(h)) {
      _T3scene.remove(entry.group);
      if (entry.labelTex) entry.labelTex.dispose();
      nodeMeshes.delete(h);
      _manualNodePos.delete(h);
    }
  }
}

// ── Connection lines (rebuilt each frame) ─────────────────────────────────────
const _connObjs = [];

function _clearConns() {
  for (const c of _connObjs) {
    _T3scene.remove(c.l1); _T3scene.remove(c.l2);
    c.geo1.dispose(); c.geo2.dispose();
  }
  _connObjs.length = 0;
}

function _addConn(p0, p1, active) {
  // Lift connectors above nodes so they're visible in side and top views.
  const p0u = p0.clone(); p0u.y += 0.72;
  const p1u = p1.clone(); p1u.y += 0.72;
  const mid  = p0u.clone().lerp(p1u, 0.5);
  const axis = new THREE.Vector3().subVectors(p1u, p0u).normalize();
  const perp = new THREE.Vector3().crossVectors(axis, new THREE.Vector3(0, 1, 0)).normalize();
  const OFF  = 0.42;
  function bezPts(sign) {
    const cp = mid.clone().addScaledVector(perp, sign * OFF);
    cp.y += 1.15;
    const pts = [];
    for (let i = 0; i <= 24; i++) {
      const t = i/24, m = 1-t;
      pts.push(new THREE.Vector3(
        m*m*p0u.x + 2*m*t*cp.x + t*t*p1u.x,
        m*m*p0u.y + 2*m*t*cp.y + t*t*p1u.y,
        m*m*p0u.z + 2*m*t*cp.z + t*t*p1u.z
      ));
    }
    return pts;
  }
  const geo1 = new THREE.BufferGeometry().setFromPoints(bezPts(1));
  const geo2 = new THREE.BufferGeometry().setFromPoints(bezPts(-1));
  const l1 = new THREE.Line(geo1, new THREE.LineBasicMaterial({
    color: active ? 0x1a9e63 : 0x2a5050, transparent: true, opacity: active ? 0.62 : 0.2, depthTest: false
  }));
  const l2 = new THREE.Line(geo2, new THREE.LineBasicMaterial({
    color: active ? 0xd06a2f : 0x2a5050, transparent: true, opacity: active ? 0.54 : 0.15, depthTest: false
  }));
  l1.renderOrder = 9;
  l2.renderOrder = 9;
  _T3scene.add(l1); _T3scene.add(l2);
  _connObjs.push({ l1, l2, geo1, geo2 });
}

// ── Particle pool ─────────────────────────────────────────────────────────────
const _pGeo  = new THREE.SphereGeometry(1, 10, 10);
const _pPool = [];

function _getParticleMesh(col, sz) {
  let m = _pPool.find(x => !x.visible);
  if (!m) {
    m = new THREE.Mesh(_pGeo, new THREE.MeshBasicMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
    }));
    _T3scene.add(m); _pPool.push(m);
  }
  m.material.color.set(new THREE.Color(`rgb(${col})`));
  m.scale.setScalar(sz * 0.028);
  m.visible = true;
  return m;
}

function _qb3(t, p0, cp, p1) {
  const m = 1 - t;
  return new THREE.Vector3(
    m*m*p0.x + 2*m*t*cp.x + t*t*p1.x,
    m*m*p0.y + 2*m*t*cp.y + t*t*p1.y,
    m*m*p0.z + 2*m*t*cp.z + t*t*p1.z
  );
}

// lane: +1 follows the green arc (l1 in _addConn), -1 follows the orange arc (l2)
// — same Y-lift and control-point formula as _addConn so the particle travels
//   exactly along the visible connection line.
function _mkParticle(fp, tp, lane, col, sz, phase, speed = 0.018) {
  const p0u = fp.clone(); p0u.y += 0.72;
  const p1u = tp.clone(); p1u.y += 0.72;
  const mid  = p0u.clone().lerp(p1u, 0.5);
  const axis = new THREE.Vector3().subVectors(p1u, p0u).normalize();
  const perp = new THREE.Vector3().crossVectors(axis, new THREE.Vector3(0, 1, 0)).normalize();
  const cp   = mid.clone().addScaledVector(perp, lane * 0.42);
  cp.y += 1.15;
  return { fp: p0u, tp: p1u, cp, t: 0, speed: speed + Math.random() * 0.003, col, sz, phase,
           mesh: _getParticleMesh(col, sz) };
}

// ── HTML overlays (crown + token flash) injected once ────────────────────────
const _T3overlay = document.createElement('div');
_T3overlay.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden';
_T3mount.appendChild(_T3overlay);

const _crownEl = document.createElement('div');
_crownEl.style.cssText = [
  'position:absolute',
  'display:none',
  'transform:translate(-50%, -100%)',
  'font-size:28px',
  'line-height:1',
  'color:#d16930',
  'text-shadow:0 0 10px rgba(209,105,48,0.9), 0 0 22px rgba(209,105,48,0.5)',
  'will-change:transform'
].join(';');
_crownEl.textContent = '';
_T3overlay.appendChild(_crownEl);

const _tokenEl = document.createElement('div');
_tokenEl.style.cssText = [
  'position:absolute',
  'display:none',
  'transform:translate(-50%, -100%)',
  'font:700 13px \'IBM Plex Mono\',monospace',
  'color:#d16930',
  'white-space:nowrap',
  'text-shadow:0 0 8px rgba(209,105,48,0.9)',
  'pointer-events:none',
  'will-change:transform,opacity',
  'transition:opacity 0.1s'
].join(';');
_T3overlay.appendChild(_tokenEl);

let _tokenDockActive = false;
const _SHOW_TOPO_STATUS_HUD = false;
const _SHOW_CROWN_OVERLAY = false;

function _queueToken(raw) {
  if (typeof raw !== 'string' || raw.length === 0) return;
  _pendingTokens.push({
    raw,
    display: raw.replace(/\n/g, '\u21b5'),
    queuedAt: performance.now(),
  });
  if (_pendingTokens.length > 420) _pendingTokens.splice(0, _pendingTokens.length - 260);
}


function _projectNode(worldPos, yLift = 1.1) {
  const v = worldPos.clone();
  v.y += yLift;
  v.project(_T3camera);
  const el = _T3renderer.domElement;
  return {
    x: (v.x  + 1) / 2 * el.clientWidth,
    y: (-v.y + 1) / 2 * el.clientHeight,
    behind: v.z > 1
  };
}
let _emptyShown = true;
const _T3emptyDiv = document.createElement('div');
_T3emptyDiv.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;pointer-events:none;';
_T3emptyDiv.innerHTML = [
  '<div style="font:14px \'Space Grotesk\',sans-serif;color:rgba(15,41,38,0.22)">Add nodes from the left panel</div>',
  '<div style="font:12px \'Space Grotesk\',sans-serif;color:rgba(15,41,38,0.12);margin-top:6px">Cluster topology will appear here</div>',
  '<div style="font:9.5px \'IBM Plex Mono\',monospace;color:rgba(15,41,38,0.10);margin-top:5px">drag nodes &middot; orbit to rotate &middot; scroll to zoom</div>'
].join('');
_T3mount.appendChild(_T3emptyDiv);

// (2D draw helpers removed — replaced by Three.js 3D scene above)


// ── Main 3D draw loop ─────────────────────────────────────────────────────────
function draw(ts) {
  requestAnimationFrame(draw);
  _T3orbit.update();

  // ── Active state ───────────────────────────────────────────────────────────
  const runningVals  = Object.values(state.running);
  const inferEntry   = runningVals.find(r => r.role === 'inference_launcher' || r.algorithm === 'infer');
  const isInferring  = !!inferEntry;
  const inferAlgo    = inferEntry?.algorithm || $('algo-sel').value;
  const topologyAlgo = _activeAlgo || $('algo-sel').value;
  const isClassicDP  = topologyAlgo === 'classicdp';
  const isTraining   = !isInferring && runningVals.some(
    r => r.role === 'server' || r.role === 'worker' || r.role === 'training_launcher'
  );
  const isActive     = isTraining || isInferring;
  // Lock active algo for BOTH training and inference so layout stays consistent
  // even if user changes the dropdown mid-run.
  const trainEntry   = isTraining ? runningVals.find(r => r.algorithm && (r.role === 'server' || r.role === 'worker' || r.role === 'training_launcher')) : null;
  const trainAlgo    = trainEntry?.algorithm || $('algo-sel').value;
  if      (isInferring) _activeAlgo = inferAlgo;
  else if (isTraining)  _activeAlgo = trainAlgo;
  else                  _activeAlgo = '';

  // Same null-safe merge used in renderMetrics
  const liveTrainingMetrics = { ...trainingFallbackMetrics };
  for (const [k, v] of Object.entries(state.training || {})) {
    if (v !== null && v !== undefined) liveTrainingMetrics[k] = v;
  }
  const trainingHasTelemetry = liveTrainingMetrics.step != null
    || liveTrainingMetrics.loss != null
    || liveTrainingMetrics.throughput != null
    || liveTrainingMetrics.tok_sec_in != null
    || liveTrainingMetrics.tok_sec_out != null
    || liveTrainingMetrics.eta_remaining != null
    || liveTrainingMetrics.eta_tqdm != null
    || liveTrainingMetrics.grad_norm != null;

  // Packets fire only when a real gradient/weight exchange was detected on the wire:
  // the training algorithm touches /tmp/smolcluster_grad_ping after each actual send,
  // and the dashboard exposes its mtime as state.grad_ts — identical to token_ts for inference.
  const gradOk = isTraining && state.grad_ts && (Date.now() / 1000 - state.grad_ts) < 4.0;
  const trainingPacketsActive = gradOk;

  const { W, H, server, workers, hasServer } = getLayout();
  const crownedHost = _activeCrownedHost(server, workers);

  // ── HUD ────────────────────────────────────────────────────────────────────
  const tag = $('topo-tag');
  const allInCluster = clusterDedup();
  if (tag) tag.textContent = _SHOW_TOPO_STATUS_HUD ? tag.textContent : '';
  const legend = $('topo-legend');
  const legendDotA = $('legend-dot-a');
  const legendDotB = $('legend-dot-b');
  const legendLabelA = $('legend-label-a');
  const legendLabelB = $('legend-label-b');
  if (!_SHOW_TOPO_STATUS_HUD && legend) {
    legend.style.display = 'none';
  } else if (legend && legendDotA && legendDotB && legendLabelA && legendLabelB) {
    if (isTraining) {
      legend.style.display = '';
      legendDotA.style.background = 'var(--green)';
      legendDotB.style.background = 'var(--accent)';
      legendLabelA.textContent = 'gradients to coordinator';
      legendLabelB.textContent = 'weights back to workers';
    } else if (isInferring) {
      legend.style.display = '';
      legendDotA.style.background = 'rgb(23,126,137)';
      legendDotB.style.background = 'var(--accent)';
      legendLabelA.textContent = 'requests out to workers';
      legendLabelB.textContent = 'results back to server';
    } else {
      legend.style.display = 'none';
    }
  }

  // ── Empty state overlay ────────────────────────────────────────────────────
  const isEmpty = !server && !workers.length;
  if (isEmpty !== _emptyShown) {
    _T3emptyDiv.style.display = isEmpty ? 'block' : 'none';
    _emptyShown = isEmpty;
  }

  // ── Sync node meshes ────────────────────────────────────────────────────────
  const keepSet  = new Set();
  const allNodes = server ? [server, ...workers] : workers;
  for (const n of allNodes) {
    keepSet.add(n.h);
    const entry    = _ensureNode(n.h, n === server);
    const newLabel = (n === server)
      ? 'SERVER'
      : `RANK ${n.rank ?? 0}`;
    if (entry.lastLabel !== newLabel) {
      entry.lastLabel = newLabel;
      const r2 = entry.isServer ? 0.72 : 0.6;
      const subLabel = state.ssh_aliases[n.h] || state.discovered[n.h]?.alias || n.h;
      const { sprite, tex } = _makeLabel(newLabel, subLabel, entry.hexCol);
      sprite.position.y = r2 + 0.92;
      entry.group.remove(entry.labelSprite);
      entry.labelTex.dispose();
      entry.labelSprite = sprite; entry.labelTex = tex;
      entry.group.add(sprite);
    }
    if (!_dragEntry || _dragEntry.entry !== entry) {
      const tgt = _manualNodePos.get(n.h) || _toWorld(n.x, n.y);
      entry.group.position.lerp(tgt, 0.12);
    }
    const pulse = 0.5 + 0.5 * Math.sin(ts / 380);
    const isCrowned = crownedHost === n.h;
    const spin = ts * 0.001;
    const baseCol = entry.isServer ? 0xd16930 : 0x167d90;
    const baseEm  = entry.isServer ? 0x6b2810 : 0x073c44;
    const crownCol = 0xd16930;
    const crownAccent = 0xf0a060;
    const crownEm = 0x8a3a12;
    if (entry.ringA && entry.ringB) {
      entry.ringA.rotation.z = spin * (entry.isServer ? 1.35 : 1.1);
      entry.ringB.rotation.y = spin * (entry.isServer ? -1.15 : -0.95);
      entry.ringB.rotation.z = spin * 0.55;
    }
    // Crown styling should be distinct from the node's normal server/worker color,
    // otherwise SyncPS server and ClassicDP rank-0 look permanently crowned.
    if (isCrowned) {
      entry.mat.color.set(crownCol);
      entry.mat.emissive.set(crownEm);
      entry.mat.emissiveIntensity = 0.34 + 0.14 * pulse;
      if (entry.ringMatA && entry.ringMatB) {
        entry.ringMatA.color.set(crownCol);
        entry.ringMatA.emissive.set(crownCol);
        entry.ringMatA.opacity = 0.50 + 0.14 * pulse;
        entry.ringMatA.emissiveIntensity = 0.42 + 0.16 * pulse;
        entry.ringMatB.color.set(crownAccent);
        entry.ringMatB.emissive.set(crownAccent);
        entry.ringMatB.opacity = 0.30 + 0.12 * pulse;
        entry.ringMatB.emissiveIntensity = 0.28 + 0.12 * pulse;
      }
    } else {
      entry.mat.color.set(baseCol);
      entry.mat.emissive.set(baseEm);
      entry.mat.emissiveIntensity = n.running ? 0.42 + 0.22 * pulse : 0.18;
      if (entry.ringMatA && entry.ringMatB) {
        entry.ringMatA.color.set(baseCol);
        entry.ringMatA.emissive.set(baseCol);
        entry.ringMatB.color.set(baseCol);
        entry.ringMatB.emissive.set(baseCol);
        if (n.running) {
          entry.ringMatA.opacity = 0.42 + 0.14 * pulse;
          entry.ringMatA.emissiveIntensity = 0.4 + 0.15 * pulse;
          entry.ringMatB.opacity = 0.28 + 0.12 * pulse;
          entry.ringMatB.emissiveIntensity = 0.3 + 0.12 * pulse;
        } else {
          entry.ringMatA.opacity = 0.2;
          entry.ringMatA.emissiveIntensity = 0.1;
          entry.ringMatB.opacity = 0.12;
          entry.ringMatB.emissiveIntensity = 0.08;
        }
      }
    }

    // ── Crown + token HTML overlays (updated in one pass below) ──
  }
  _removeStaleNodes(keepSet);

  // ── Crown + token HTML overlay: update position once per frame ──────────────
  const crownedEntry = crownedHost ? nodeMeshes.get(crownedHost) : null;
  let _newTokenPulse = false;
  if (_SHOW_CROWN_OVERLAY && crownedEntry && isTraining) {
    const wpos = crownedEntry.group.position;
    const sc   = _projectNode(wpos, 2.05);
    if (sc.behind) {
      _crownEl.style.display = 'none';
    } else {
      const bob = 4 * Math.sin(ts / 540);
      _crownEl.style.display = 'block';
      _crownEl.style.left    = sc.x + 'px';
      _crownEl.style.top     = (sc.y + bob) + 'px';
    }
  }
  if (_SHOW_CROWN_OVERLAY && crownedEntry && dashboardMode === 'infer' && isInferring) {
    // Queue tokens from SSE, then reveal them only after return packets arrive.
    const tokenStamp = Number(state.token_ts) || 0;
    if (tokenStamp && tokenStamp !== _lastTokenStamp) {
      _lastTokenStamp = tokenStamp;
      const tok = String(state.token_text || '');
      if (tok) _queueToken(tok);
      // Fallback marker if timestamp advanced but no visible token bytes arrived.
      if (!tok && !_lastTokenText) {
        _lastTokenText = '\u258c';
        _lastTokenRaw = '';
        _tokenFlashTs = ts;
      }
    }

    const hasReturnPacket = inferParticles.some(p => p.phase === 'return');
    const returnReachedServer = inferParticles.some(p => p.phase === 'return' && p.t >= 0.94);
    const queued = _pendingTokens[0];
    const agedOut = queued ? (ts - queued.queuedAt) >= _tokenFallbackDelayMs : false;
    if (queued && (returnReachedServer || (!hasReturnPacket && agedOut) || agedOut)) {
      _pendingTokens.shift();
      _lastTokenRaw = queued.raw;
      _lastTokenText = queued.display.length > 24 ? queued.display.slice(0, 23) + '\u2026' : queued.display;
      _tokenFlashTs = ts;
      _newTokenPulse = true;
    }

    const wpos = crownedEntry.group.position;
    const sc   = _projectNode(wpos, 2.05);
    if (sc.behind) {
      _crownEl.style.display = 'none';
      _tokenEl.style.display = 'none';
    } else {
      const bob = 4 * Math.sin(ts / 540); // px bob
      _crownEl.style.display = 'block';
      _crownEl.style.left    = sc.x + 'px';
      _crownEl.style.top     = (sc.y + bob) + 'px';

      // Token flash
      const sinceFlash = ts - _tokenFlashTs;
      const flashDur   = 2400;
      if (sinceFlash < flashDur && _lastTokenText) {
        const alpha  = Math.max(0, 1 - sinceFlash / flashDur);
        const drift  = sinceFlash * 0.016; // px upward drift over time
        _tokenEl.style.display = 'block';
        _tokenEl.style.left    = sc.x + 'px';
        _tokenEl.style.top     = (sc.y - 40 - drift + bob) + 'px';
        _tokenEl.style.opacity = alpha;
        if (_tokenEl.dataset.txt !== _lastTokenText) {
          _tokenEl.textContent   = _lastTokenText.length > 24 ? _lastTokenText.slice(0,23) + '\u2026' : _lastTokenText;
          _tokenEl.dataset.txt   = _lastTokenText;
        }
        if (_newTokenPulse && _lastTokenRaw) {
          _tokenDockActive = true;
        }
      } else {
        _tokenEl.style.display = 'none';
      }
    }
  } else {
    _crownEl.style.display = 'none';
    _tokenEl.style.display = 'none';
  }

  // ── Connections ────────────────────────────────────────────────────────────
  _clearConns();
  if (isClassicDP && workers.length >= 2) {
    for (let i = 0; i < workers.length; i++)
      for (let j = i+1; j < workers.length; j++) {
        const a = nodeMeshes.get(workers[i].h)?.group.position;
        const b = nodeMeshes.get(workers[j].h)?.group.position;
        if (a && b) _addConn(a, b, isActive && workers[i].running && workers[j].running);
      }
  } else if (server) {
    const sp3 = nodeMeshes.get(server.h)?.group.position;
    if (sp3) workers.forEach(w => {
      const wp = nodeMeshes.get(w.h)?.group.position;
      if (wp) _addConn(sp3, wp, isActive && w.running);
    });
  } else if (workers.length >= 2) {
    const r0p = nodeMeshes.get(workers[0].h)?.group.position;
    if (r0p) workers.slice(1).forEach(w => {
      const wp = nodeMeshes.get(w.h)?.group.position;
      if (wp) _addConn(r0p, wp, isActive && workers[0].running && w.running);
    });
  }

  // Particle speed derived from real step rate: faster training = faster packets.
  // At 1s/step → speed≈0.022; at 10s/step → speed≈0.010; clamped [0.008, 0.045].
  const _pSpeed = () => Math.min(0.045, Math.max(0.008, 1.2 / (_gradIntervalMs / 1000) * 0.018));
  // Spawn gate: fire a new burst after 90% of the measured step interval,
  // so each real gradient exchange gets exactly one animation burst.
  const _spawnGate = Math.max(300, _gradIntervalMs * 0.9);
  // Inference packet speed derived from real token rate.
  // At ≤150ms/tok → speed≈0.060; at 400ms/tok → speed≈0.030; floor at 0.030.
  const _iSpeed = () => Math.min(0.065, Math.max(0.030, 0.006 / (_tokenIntervalMs / 1000)));
  const _iSpawnGate = Math.max(100, _tokenIntervalMs * 0.75);
  const _trainIoLegSpeed = () => {
    const halfLegMs = Math.max(60, Math.min(_trainIoIntervalMs, _trainIoRttMs * 0.5));
    return Math.min(0.12, Math.max(0.018, 0.006 / (halfLegMs / 1000)));
  };

  // ── Training particles ─────────────────────────────────────────────────────
  const coord = server || (workers.length ? workers[0] : null);
  const peers  = server ? workers : workers.slice(1);
  const lastTrainIoTrafficTs = Math.max(_trainIoReqTs || 0, _trainIoRespTs || 0);
  const trainIoTrafficHot = isTraining && ((
    _trainIoPendingReq > 0 || _trainIoPendingResp > 0
  ) || (lastTrainIoTrafficTs > 0 && (Date.now() - lastTrainIoTrafficTs) < 8000));
  if (isTraining && trainIoTrafficHot && coord && peers.length) {
    const hasReq = particles.some(p => p.phase === 'train_req');
    const hasResp = particles.some(p => p.phase === 'train_resp');
    if (_trainIoPendingReq > 0 && !hasReq) {
      peers.forEach(w => {
        const cp3 = nodeMeshes.get(coord.h)?.group.position;
        const tp  = nodeMeshes.get(w.h)?.group.position;
        if (cp3 && tp) particles.push(_mkParticle(cp3.clone(), tp.clone(), -1, '23,126,137', 5.5, 'train_req', _trainIoLegSpeed()));
      });
      _trainIoPendingReq -= 1;
      spawnTs = ts;
    }

    if (_trainIoPendingResp > 0 && !hasResp) {
      peers.forEach(w => {
        const cp3 = nodeMeshes.get(coord.h)?.group.position;
        const tp  = nodeMeshes.get(w.h)?.group.position;
        if (cp3 && tp) particles.push(_mkParticle(tp.clone(), cp3.clone(), 1, '208,106,47', 4.7, 'train_resp', _trainIoLegSpeed()));
      });
      _trainIoPendingResp -= 1;
    }
  } else if (trainingPacketsActive && isClassicDP && workers.length >= 2 && ts - spawnTs > _spawnGate) {
    const hasG = particles.some(p => p.phase === 'gradients');
    const hasW = particles.some(p => p.phase === 'weights');
    if (!hasG && !hasW && !particles.length) {
      for (let i = 0; i < workers.length; i++)
        for (let j = 0; j < workers.length; j++) {
          if (i === j) continue;
          const fp = nodeMeshes.get(workers[i].h)?.group.position;
          const tp = nodeMeshes.get(workers[j].h)?.group.position;
          if (fp && tp) particles.push(_mkParticle(fp.clone(), tp.clone(), 1, '26,158,99', 4.8, 'gradients', _pSpeed()));
        }
      spawnTs = ts;
    }
    const maxG = Math.max(0, ...particles.filter(p=>p.phase==='gradients').map(p=>p.t));
    if (maxG > 0.78 && !particles.some(p=>p.phase==='weights'))
      for (let i = 0; i < workers.length; i++)
        for (let j = 0; j < workers.length; j++) {
          if (i === j) continue;
          const fp = nodeMeshes.get(workers[i].h)?.group.position;
          const tp = nodeMeshes.get(workers[j].h)?.group.position;
          if (fp && tp) particles.push(_mkParticle(fp.clone(), tp.clone(), -1, '209,105,48', 4.5, 'weights', _pSpeed()));
        }
  } else if (trainingPacketsActive && coord && ts - spawnTs > _spawnGate) {
    const hasG = particles.some(p=>p.phase==='gradients');
    const hasW = particles.some(p=>p.phase==='weights');
    if (!hasG && !hasW && !particles.length) {
      peers.forEach(w => {
        const fp  = nodeMeshes.get(w.h)?.group.position;
        const cp3 = nodeMeshes.get(coord.h)?.group.position;
        if (fp && cp3) particles.push(_mkParticle(fp.clone(), cp3.clone(), 1, '26,158,99', 6, 'gradients', _pSpeed()));
      });
      spawnTs = ts;
    }
    const maxG = Math.max(0, ...particles.filter(p=>p.phase==='gradients').map(p=>p.t));
    if (maxG > 0.8 && !particles.some(p=>p.phase==='weights'))
      peers.forEach(w => {
        const cp3 = nodeMeshes.get(coord.h)?.group.position;
        const tp  = nodeMeshes.get(w.h)?.group.position;
        if (cp3 && tp) particles.push(_mkParticle(cp3.clone(), tp.clone(), -1, '209,105,48', 5.5, 'weights', _pSpeed()));
      });
  }
  if (!trainingPacketsActive && !trainIoTrafficHot) {
    particles.forEach(p => { if (p.mesh) p.mesh.visible = false; });
    particles = [];
    if (!isTraining) {
      _trainIoPendingReq = 0;
      _trainIoPendingResp = 0;
    }
  }

  // ── Inference particles ────────────────────────────────────────────────────
  const inferCoord = server || workers[0];
  const inferPeers = server ? workers : workers.slice(1);
  if (isInferring && isClassicDP && workers.length >= 2) {
    const tokenOk = state.token_ts && (Date.now()/1000 - state.token_ts) < 4.0;
    if (tokenOk && ts - inferSpawnTs > _iSpawnGate) {
      const hasOutbound = inferParticles.some(p => p.phase === 'outbound');
      const hasReturn   = inferParticles.some(p => p.phase === 'return');
      if (!hasOutbound && !hasReturn && !inferParticles.length) {
        for (let i = 0; i < workers.length; i++)
          for (let j = 0; j < workers.length; j++) {
            if (i === j) continue;
            const fp = nodeMeshes.get(workers[i].h)?.group.position;
            const tp = nodeMeshes.get(workers[j].h)?.group.position;
            if (fp && tp) inferParticles.push(_mkParticle(fp.clone(), tp.clone(), -1, '23,126,137', 4.8, 'outbound', _iSpeed()));
          }
        inferSpawnTs = ts;
      }
      const maxO = Math.max(0, ...inferParticles.filter(p=>p.phase==='outbound').map(p=>p.t));
      if (maxO > 0.78 && !inferParticles.some(p=>p.phase==='return'))
        for (let i = 0; i < workers.length; i++)
          for (let j = 0; j < workers.length; j++) {
            if (i === j) continue;
            const fp = nodeMeshes.get(workers[i].h)?.group.position;
            const tp = nodeMeshes.get(workers[j].h)?.group.position;
            if (fp && tp) inferParticles.push(_mkParticle(fp.clone(), tp.clone(), 1, '208,106,47', 4.2, 'return', _iSpeed()));
          }
    }
  } else if (isInferring && inferCoord && inferPeers.length) {
    const tokenOk = state.token_ts && (Date.now()/1000 - state.token_ts) < 4.0;
    if (tokenOk && ts - inferSpawnTs > _iSpawnGate) {
      const hasOutbound = inferParticles.some(p => p.phase === 'outbound');
      const hasReturn   = inferParticles.some(p => p.phase === 'return');
      if (!hasOutbound && !hasReturn && !inferParticles.length) {
        inferPeers.forEach(w => {
          const ic3 = nodeMeshes.get(inferCoord.h)?.group.position;
          const tp  = nodeMeshes.get(w.h)?.group.position;
          if (ic3 && tp) inferParticles.push(_mkParticle(ic3.clone(), tp.clone(), -1, '23,126,137', 5.5, 'outbound', _iSpeed()));
        });
        inferSpawnTs = ts;
      }
      const maxO = Math.max(0, ...inferParticles.filter(p=>p.phase==='outbound').map(p=>p.t));
      if (maxO > 0.8 && !inferParticles.some(p=>p.phase==='return'))
        inferPeers.forEach(w => {
          const ic3 = nodeMeshes.get(inferCoord.h)?.group.position;
          const tp  = nodeMeshes.get(w.h)?.group.position;
          if (ic3 && tp) inferParticles.push(_mkParticle(tp.clone(), ic3.clone(), 1, '208,106,47', 4.5, 'return', _iSpeed()));
        });
    }
  }
  if (!isInferring) { inferParticles.forEach(p => { if (p.mesh) p.mesh.visible = false; }); inferParticles = []; }

  // ── Animate particles ──────────────────────────────────────────────────────
  function _movePart(p) {
    p.t += p.speed;
    if (p.t >= 1) { if (p.mesh) p.mesh.visible = false; return false; }
    p.mesh.position.copy(_qb3(p.t, p.fp, p.cp, p.tp));
    p.mesh.material.opacity = Math.min(1, Math.sin(p.t * Math.PI) * 1.7);
    p.mesh.visible = true;
    return true;
  }
  particles      = particles.filter(_movePart);
  inferParticles = inferParticles.filter(_movePart);

  // ── Render ─────────────────────────────────────────────────────────────────
  _T3renderer.render(_T3scene, _T3camera);
}

// ════════════════════════════════════════════════════════════════════════════
// Boot
// ════════════════════════════════════════════════════════════════════════════
requestAnimationFrame(draw);
initGenerationComposer();
startSSE();
startLogs();
loadUI().then(() => initBottomResizer());
