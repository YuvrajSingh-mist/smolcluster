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
  const elapsedText = pickMetricText(t, ['elapsed_tqdm', 'tqdm_elapsed', 'elapsed', 'elapsed_time', 'time_elapsed', 'runtime', 'duration']);
  const runningVals = Object.values(state.running || {});
  const isTrainingRunning = runningVals.some(r => r.role === 'server' || r.role === 'worker' || r.role === 'training_launcher');
  const isInferringRunning = runningVals.some(r => r.role === 'inference_launcher' || r.algorithm === 'infer');
  const hasTraining = isTrainingRunning && (
    t.step != null || t.loss != null || t.throughput != null || t.grad_norm != null
    || tputInResolved !== null || tputOutResolved !== null || !!etaText || !!elapsedText
  );
  const hasNet = isTrainingRunning || isInferringRunning;
  $('metrics-strip').classList.toggle('show', hasTraining || hasNet);
  if (!hasTraining && !hasNet) return;

  if (hasTraining) {
    setText('m-loss', t.loss       != null ? Number(t.loss).toFixed(4) : '—');
    setText('m-tput-in', fmtTput(tputInResolved));
    setText('m-tput-out', fmtTput(tputOutResolved));
    setText('m-step', t.step       != null ? t.step.toLocaleString()   : '—');
    setText('m-eta', etaText || '—');
    setText('m-elapsed', elapsedText || '—');
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
      let progMeta = '';
      if (elapsedText) progMeta += `Elapsed ${elapsedText}`;
      if (etaText) progMeta += `${progMeta ? ' · ' : ''}ETA ${etaText}`;
      setText('prog-lbl', progMeta
        ? `Step ${t.step.toLocaleString()} / ${totalSteps.toLocaleString()} · ${progMeta}`
        : `Step ${t.step.toLocaleString()} / ${totalSteps.toLocaleString()}`);
      setText('prog-pct', pct < 1 ? pct.toFixed(2) + '%' : Math.round(pct) + '%');
    }
  }

  // ── Network metrics ─────────────────────────────────────────────────────────
  if (hasNet) {
    const fmtMs  = ms  => ms  < 1000 ? `${Math.round(ms)}ms`  : `${(ms / 1000).toFixed(1)}s`;
    const fmtBw  = v => { if (v == null || !Number.isFinite(v) || v <= 0) return '—'; if (v < 1) return `${(v * 1000).toFixed(v < 0.1 ? 1 : 0)}Kb/s`; if (v < 10) return `${v.toFixed(2)}Mb/s`; return `${Math.round(v)}Mb/s`; };
    const fmtMb  = v => { if (v == null || !Number.isFinite(v) || v <= 0) return '—'; if (v < 0.01) return `${Math.round(v * 1000)}KB`; if (v < 10) return `${v.toFixed(2)}MB`; return `${v.toFixed(1)}MB`; };
    const fmtKb  = v   => (v != null && Number.isFinite(v) && v > 0) ? `${v < 1024 ? Math.round(v) + 'KB' : (v / 1024).toFixed(1) + 'MB'}` : '—';

    if (isTrainingRunning) {
      setText('m-intv-lbl', 'Sync Intv');
      setText('m-intv', _gradIntervalMs > 300 ? fmtMs(_gradIntervalMs) : '—');
    } else {
      setText('m-intv-lbl', 'Token Intv');
      setText('m-intv', _tokenIntervalMs > 0 ? fmtMs(_tokenIntervalMs) : '—');
    }

    const fmtMbMem = v => (v != null && Number.isFinite(v) && v > 0) ? (v >= 1024 ? `${(v/1024).toFixed(1)}GB` : `${Math.round(v)}MB`) : '—';
    const _t = trainingFallbackMetrics;
    setText('m-mem-active', fmtMbMem(_t.active_mem_mb));
    setText('m-mem-peak',   fmtMbMem(_t.peak_mem_mb));

    const _n = trainingFallbackMetrics;
    setText('m-net-snd-bw',  fmtBw(_n.send_bandwidth_mbps));
    setText('m-net-rcv-bw',  fmtBw(_n.recv_bandwidth_mbps));
    setText('m-net-snd-lat', _n.avg_send_latency_ms > 0 ? fmtMs(_n.avg_send_latency_ms) : '—');
    setText('m-net-rcv-lat', _n.avg_recv_latency_ms > 0 ? fmtMs(_n.avg_recv_latency_ms) : '—');
    setText('m-net-snd-mb',  fmtMb(_n.total_send_mb));
    setText('m-net-rcv-mb',  fmtMb(_n.total_recv_mb));

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
