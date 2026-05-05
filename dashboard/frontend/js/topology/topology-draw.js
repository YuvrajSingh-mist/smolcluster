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
  const isClassicDP  = topologyAlgo === 'classicdp' || topologyAlgo === 'fsdp';
  const isTraining   = !isInferring && runningVals.some(
    r => r.role === 'server' || r.role === 'worker' || r.role === 'training_launcher'
  );
  const isActive = isTraining || isInferring;
  const trainEntry = isTraining ? runningVals.find(r => r.algorithm && (r.role === 'server' || r.role === 'worker' || r.role === 'training_launcher')) : null;
  const trainAlgo  = trainEntry?.algorithm || $('algo-sel').value;
  if      (isInferring) _activeAlgo = inferAlgo;
  else if (isTraining)  _activeAlgo = trainAlgo;
  else                  _activeAlgo = '';

  // ── Legend — per-algorithm colour key ──────────────────────────────────────
  const legend = $('topo-legend');
  if (legend) {
    if (!isActive) {
      legend.style.display = 'none';
      _lastLegendAlgo = '';
    } else {
      const _curAlgo = _activeAlgo || $('algo-sel').value;
      const _algoKey = (isInferring ? 'infer:' : 'train:') + _curAlgo;
      if (_algoKey !== _lastLegendAlgo) {
        _lastLegendAlgo = _algoKey;
        const _items = _legendItems(_curAlgo, isInferring);
        legend.innerHTML = _items.map(([col, lbl]) =>
          `<div class="leg"><div class="legdot" style="background:rgb(${col})"></div><span>${lbl}</span></div>`
        ).join('');
      }
      legend.style.display = '';
    }
  }

  // ── Empty state overlay ────────────────────────────────────────────────────
  const { W, H, server, workers, hasServer } = getLayout();
  _lastLayout = { server, workers };
  const isEmpty = !server && !workers.length;
  if (isEmpty !== _emptyShown) {
    _T3emptyDiv.style.display = isEmpty ? 'block' : 'none';
    _emptyShown = isEmpty;
  }

  // ── Sync node meshes + crown overlay ────────────────────────────────────────
  const crownedHost = _activeCrownedHost(server, workers);
  _drawSyncNodes(ts, server, workers, crownedHost, isActive);
  const crownedEntry = crownedHost ? nodeMeshes.get(crownedHost) : null;
  _drawCrownOverlay(ts, crownedEntry, isTraining, isInferring);

  // ── Connections ────────────────────────────────────────────────────────────
  _drawConnections(server, workers, isActive, isClassicDP);

  // ── Particle spawning (event-driven — drains _smolEventQueue immediately) ──
  _processSmolEvents(server, workers);

  // Clear particles when cluster goes idle
  if (!isActive) _clearAllParticles();

  // ── Animate particles ──────────────────────────────────────────────────────
  function _movePart(p) {
    p.t += p.speed;
    if (p.t >= 1) { if (p.mesh) p.mesh.visible = false; return false; }
    p.mesh.position.copy(_qb3(p.t, p.fp, p.cp, p.tp));
    p.mesh.material.opacity = Math.min(1, Math.sin(p.t * Math.PI) * 1.7);
    p.mesh.visible = true;
    return true;
  }
  particles = particles.filter(_movePart);

  // ── Render ─────────────────────────────────────────────────────────────────
  _T3renderer.render(_T3scene, _T3camera);
}
