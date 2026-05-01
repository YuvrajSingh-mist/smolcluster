// ════════════════════════════════════════════════════════════════════════════
// Log terminal — state vars and host-colour helpers
// ════════════════════════════════════════════════════════════════════════════
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
