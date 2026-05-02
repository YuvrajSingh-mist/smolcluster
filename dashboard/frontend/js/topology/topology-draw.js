// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — main draw loop
// ════════════════════════════════════════════════════════════════════════════
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
  const isActive = isTraining || isInferring;
  const trainEntry = isTraining ? runningVals.find(r => r.algorithm && (r.role === 'server' || r.role === 'worker' || r.role === 'training_launcher')) : null;
  const trainAlgo  = trainEntry?.algorithm || $('algo-sel').value;
  if      (isInferring) _activeAlgo = inferAlgo;
  else if (isTraining)  _activeAlgo = trainAlgo;
  else                  _activeAlgo = '';

  // ── Live training metrics ──────────────────────────────────────────────────
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

  const gradOk = isTraining && state.grad_ts && (Date.now() / 1000 - state.grad_ts) < 4.0;
  const trainingPacketsActive = gradOk;

  const { W, H, server, workers, hasServer } = getLayout();
  _lastLayout = { server, workers };
  const crownedHost = _activeCrownedHost(server, workers);

  // ── HUD tag + legend ────────────────────────────────────────────────────────
  const tag = $('topo-tag');
  if (tag) tag.textContent = _SHOW_TOPO_STATUS_HUD ? tag.textContent : '';
  const legend = $('topo-legend');
  if (!_SHOW_TOPO_STATUS_HUD && legend) {
    legend.style.display = 'none';
  } else if (legend) {
    const legendDotA = $('legend-dot-a');
    const legendDotB = $('legend-dot-b');
    const legendLabelA = $('legend-label-a');
    const legendLabelB = $('legend-label-b');
    if (legendDotA && legendDotB && legendLabelA && legendLabelB) {
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
  }

  // ── Empty state overlay ────────────────────────────────────────────────────
  const isEmpty = !server && !workers.length;
  if (isEmpty !== _emptyShown) {
    _T3emptyDiv.style.display = isEmpty ? 'block' : 'none';
    _emptyShown = isEmpty;
  }

  // ── Sync node meshes + crown overlay ────────────────────────────────────────
  _drawSyncNodes(ts, server, workers, crownedHost, isActive);
  const crownedEntry = crownedHost ? nodeMeshes.get(crownedHost) : null;
  _drawCrownOverlay(ts, crownedEntry, isTraining, isInferring);

  // ── Connections ────────────────────────────────────────────────────────────
  _drawConnections(server, workers, isActive, isClassicDP);

  // ── Particle spawning ──────────────────────────────────────────────────────
  const coord = server || (workers.length ? workers[0] : null);
  const peers  = server ? workers : workers.slice(1);
  const lastTrainIoTrafficTs = Math.max(_trainIoReqTs || 0, _trainIoRespTs || 0);
  const trainIoTrafficHot = isTraining && (
    (_trainIoPendingReq > 0 || _trainIoPendingResp > 0)
    || (lastTrainIoTrafficTs > 0 && (Date.now() - lastTrainIoTrafficTs) < 8000)
  );
  _drawTrainingParticles(ts, coord, peers, workers, isTraining, isClassicDP, trainingPacketsActive, trainIoTrafficHot);

  const inferCoord = server || workers[0];
  const inferPeers = server ? workers : workers.slice(1);
  _drawInferenceParticles(ts, inferCoord, inferPeers, isInferring, isClassicDP, workers);

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
