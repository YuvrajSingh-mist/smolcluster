// ════════════════════════════════════════════════════════════════════════════
// Render helpers
// ════════════════════════════════════════════════════════════════════════════

/**
 * Deduplicate a set of hostnames by IP address, using state.discovered for
 * IP lookup. When two hostnames resolve to the same IP, keep whichever is
 * in state.running first, then prefer the shorter key (SSH alias).
 */
function dedupByIp(hostnameMap) {
  const seenIps = new Map(); // ip → kept key
  const result  = {};
  const sorted = Object.keys(hostnameMap).sort((a, b) => {
    const aRun = a in state.running ? 0 : 1;
    const bRun = b in state.running ? 0 : 1;
    if (aRun !== bRun) return aRun - bRun;
    return a.length - b.length;
  });
  for (const h of sorted) {
    const ip = state.discovered[h]?.ip || '';
    if (ip && seenIps.has(ip)) continue;
    if (ip) seenIps.set(ip, h);
    result[h] = hostnameMap[h];
  }
  return result;
}

function discoveredDedup() {
  return dedupByIp(state.discovered);
}

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
