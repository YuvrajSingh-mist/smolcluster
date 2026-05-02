// ════════════════════════════════════════════════════════════════════════════
// Left panel — node cards
// ════════════════════════════════════════════════════════════════════════════
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

  const osInfo = state.node_os[h] || n || {};
  const metaEl = c.querySelector('.nmeta');
  if (metaEl) {
    const osStr = [osInfo.os, osInfo.os_version, osInfo.machine]
      .filter(s => s && s !== 'unknown').join(' · ');
    if (metaEl.textContent !== osStr) metaEl.textContent = osStr;
  }

  const inp   = c.querySelector('.sshinput');
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
