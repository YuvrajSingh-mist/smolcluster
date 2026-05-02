// ════════════════════════════════════════════════════════════════════════════
// Cluster actions
// ════════════════════════════════════════════════════════════════════════════
let selectedServer = null;   // hostname of user-picked server node
let inferLocked    = false;  // true while inference is running — blocks server re-selection

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
  if (['syncps', 'classicdp', 'grpo'].includes(algo)) _manualNodePos.clear();
  const hosts = Object.keys(state.selected);
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

async function startInference() {
  setDashboardMode('infer');
  if (!Object.keys(state.selected).length) { alert('Add nodes first.'); return; }
  clearLogs();
  const algo = $('algo-sel').value;
  _lastTokenStamp = Number(state.token_ts) || 0;
  _lastTokenRaw = '';
  _lastTokenText = '';
  _pendingTokens = [];
  _tokenDockActive = false;
  if (['syncps', 'classicdp'].includes(algo)) _manualNodePos.clear();

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
    syncGenerationAvailability();
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
  if (_lastLayout) {
    const allNodes = [
      ...(_lastLayout.server ? [_lastLayout.server] : []),
      ...(_lastLayout.workers || []),
    ];
    for (const n of allNodes) {
      const entry = nodeMeshes.get(n.h);
      if (entry) entry.group.position.copy(_toWorld(n.x, n.y));
    }
  }
  _T3camera.position.set(0, 10, 12);
  _T3orbit.target.set(0, 0, 0);
  _T3orbit.update();
}

async function stopAll() {
  const btn = $('btn-stop');
  btn.classList.add('stopping');
  const lbl = btn.querySelector('.stop-label');
  if (lbl) lbl.textContent = 'Stopping…';

  if (ssEventSource) { ssEventSource.close(); ssEventSource = null; }
  if (logsEventSource) { logsEventSource.close(); logsEventSource = null; }
  stopGenerationRequest();

  try {
    await fetch('/api/training/stop',  { method:'POST' });
    await fetch('/api/inference/stop', { method:'POST' });
  } finally {
    btn.classList.remove('stopping');
    if (lbl) lbl.textContent = 'Stop';
  }

  inferLocked    = false;
  selectedServer = null;
  trainingFallbackMetrics = {};
  clearLogs();

  setTimeout(() => {
    if (!ssEventSource) startSSE();
    if (!logsEventSource) startLogs();
  }, 500);
}
