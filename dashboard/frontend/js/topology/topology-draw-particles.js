// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — draw helpers: connections + particle spawning
// ════════════════════════════════════════════════════════════════════════════

/** Rebuild connection tube geometry for this frame. */
function _drawConnections(server, workers, isActive, isClassicDP) {
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
}

// ── Color + size per event type ───────────────────────────────────────────────
const _EV_COL  = { gradients: '220,185,40', weights: '60,130,220', rollout: '210,60,60', weight_sync: '208,106,47' };
const _EV_SZ   = { gradients: 5.5, weights: 5.5, rollout: 6.2, weight_sync: 5.0 };
const _EV_SPEED = 0.038;

/**
 * Return legend item pairs [rgbString, label] for the active algorithm.
 * @param {string} algo   – active algorithm key
 * @param {boolean} isInfer – true when in pure inference mode
 */
function _legendItems(algo, isInfer) {
  if (isInfer && algo !== 'grpo') {
    return [['23,126,137', 'Inference Requests']];
  }
  if (algo === 'grpo') return [
    ['210,60,60',  'Rollout Gen'],
    ['208,106,47', 'Weight Sync'],
  ];
  if (algo === 'fsdp') return [
    ['220,185,40', 'Grad Shards'],
    ['60,130,220',  'All-Gather'],
  ];
  if (algo === 'classicdp') return [
    ['220,185,40', 'Gradients'],
    ['60,130,220',  'Weights'],
  ];
  // syncps (default for all gradient-based)
  return [
    ['220,185,40', 'Gradients'],
    ['60,130,220',  'Weights'],
  ];
}

/**
 * Drain _smolEventQueue and spawn particles immediately for each event.
 * Architecture determines topology:
 *   syncps / grpo  → master-worker (server ↔ workers)
 *   classicdp/fsdp → all-to-all   (every worker pair)
 */
function _processSmolEvents(server, workers) {
  if (!_smolEventQueue.length) return;
  const events = _smolEventQueue.splice(0);

  const coord = server || workers[0];
  const peers = server ? workers : workers.slice(1);

  for (const ev of events) {
    const col  = _EV_COL[ev.type]  || '220,185,40';
    const sz   = _EV_SZ[ev.type]   || 5.5;
    const arch = String(ev.arch || '');
    const isAllReduce = arch === 'classicdp' || arch === 'fsdp';
    const isPS        = arch === 'syncps' || arch === 'grpo';

    if (isAllReduce && workers.length >= 2) {
      // Every worker sends to every other worker
      for (let i = 0; i < workers.length; i++)
        for (let j = 0; j < workers.length; j++) {
          if (i === j) continue;
          const fp = nodeMeshes.get(workers[i].h)?.group.position;
          const tp = nodeMeshes.get(workers[j].h)?.group.position;
          if (fp && tp)
            particles.push(_mkParticle(fp.clone(), tp.clone(), i < j ? 1 : -1, col, sz, ev.type, _EV_SPEED));
        }
    } else if (isPS && coord && peers.length) {
      if (ev.dir === 'in') {
        // workers → coordinator
        peers.forEach(w => {
          const wp = nodeMeshes.get(w.h)?.group.position;
          const cp = nodeMeshes.get(coord.h)?.group.position;
          if (wp && cp) particles.push(_mkParticle(wp.clone(), cp.clone(), 1, col, sz, ev.type, _EV_SPEED));
        });
      } else {
        // coordinator → workers
        peers.forEach(w => {
          const cp = nodeMeshes.get(coord.h)?.group.position;
          const wp = nodeMeshes.get(w.h)?.group.position;
          if (cp && wp) particles.push(_mkParticle(cp.clone(), wp.clone(), -1, col, sz, ev.type, _EV_SPEED));
        });
      }
    }
  }
}

/** Clear all live particles (called when cluster goes idle). */
function _clearAllParticles() {
  particles.forEach(p => { if (p.mesh) p.mesh.visible = false; });
  particles = [];
}
