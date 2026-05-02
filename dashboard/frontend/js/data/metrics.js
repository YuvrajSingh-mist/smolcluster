// ════════════════════════════════════════════════════════════════════════════
// Status bar (connectivity) + training metrics strip
// ════════════════════════════════════════════════════════════════════════════
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

function renderMetrics() {
  const tRaw = state.training || {};
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
  if (tputOut === null) tputOut = tputGeneric;
  const tputInResolved  = tputIn  !== null ? tputIn  : (tputOut !== null ? tputOut : tputGeneric);
  const tputOutResolved = tputOut !== null ? tputOut : (tputIn  !== null ? tputIn  : tputGeneric);
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

// Pre-populate the filter dropdown from selected+running nodes (don't wait for log lines)
function syncLogFilter() {
  const allNodes = {...state.selected, ...state.running};
  for (const h of Object.keys(allNodes)) {
    knownLogHosts.add(h);
    ensureLogFilterOption(h);
  }
  refreshLogHostLabels();
}
