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

// Frames between each wave of packets so each rollout has a distinct, visible journey.
const _SPAWN_FRAMES_GAP = 3;

/**
 * Drain _smolEventQueue and push deferred spawn items into _particleSpawnQueue.
 *
 * For an event with count=N (e.g. 16 rollouts), N waves are queued — one per
 * rollout. Every _SPAWN_FRAMES_GAP frames, _drainSpawnQueue fires one wave:
 * one particle per worker connection, all starting at t=0 and doing a full
 * visible transit.  This gives exactly N distinct animations (not N simultaneous
 * pre-displaced blobs) and works correctly for any number of worker nodes.
 *
 * The coord/peers split always uses server first, then all workers as peers,
 * so every selected node gets packets — not just workers[1..].
 */
function _processSmolEvents(server, workers) {
  if (!_smolEventQueue.length) return;
  const events = _smolEventQueue.splice(0);

  // coord = the hub; peers = all the spokes.  For GRPO the training node IS the
  // server (rank=0); workers are the vLLM nodes.  If somehow server is null,
  // we use all workers as peers (all-to-first-worker star) rather than
  // silently dropping workers[0] with .slice(1).
  const coord = server;
  const peers = server ? workers : workers;

  let batchBaseDelay = 0;  // wave delay offset — keeps consecutive events from overlapping

  for (const ev of events) {
    const col  = _EV_COL[ev.type]  || '220,185,40';
    const sz   = _EV_SZ[ev.type]   || 5.5;
    const arch = String(ev.arch || '');
    const isAllReduce = arch === 'classicdp' || arch === 'fsdp';
    const isPS        = arch === 'syncps' || arch === 'grpo';
    const count = Math.min(Math.max(1, ev.count || 1), 32);

    // Snapshot all (fp, tp, lane) pairs for this event right now so positions
    // are captured at event-fire time (after _drawSyncNodes has run).
    const pairs = [];
    if (isAllReduce && workers.length >= 2) {
      for (let i = 0; i < workers.length; i++)
        for (let j = 0; j < workers.length; j++) {
          if (i === j) continue;
          const fp = nodeMeshes.get(workers[i].h)?.group.position;
          const tp = nodeMeshes.get(workers[j].h)?.group.position;
          if (fp && tp) pairs.push({ fp: fp.clone(), tp: tp.clone(), lane: i < j ? 1 : -1 });
        }
    } else if (isPS) {
      // No coord: treat workers[0] as hub only if there is no server.
      const hub = coord || (workers.length ? workers[0] : null);
      const spokes = coord ? workers : workers.slice(1);
      if (hub && spokes.length) {
        spokes.forEach(w => {
          const hp = nodeMeshes.get(hub.h)?.group.position;
          const wp = nodeMeshes.get(w.h)?.group.position;
          if (!hp || !wp) return;
          if (ev.dir === 'in')
            pairs.push({ fp: wp.clone(), tp: hp.clone(), lane: 1 });
          else
            pairs.push({ fp: hp.clone(), tp: wp.clone(), lane: -1 });
        });
      }
    }

    if (!pairs.length) { batchBaseDelay += _SPAWN_FRAMES_GAP; continue; }

    // Enqueue N waves.  Wave k fires at batchBaseDelay + k*GAP frames from now.
    // All connections in the same wave fire simultaneously so you see a clear
    // "pulse" hitting every worker node at once, N times.
    for (let k = 0; k < count; k++) {
      const delay = batchBaseDelay + k * _SPAWN_FRAMES_GAP;
      for (const pair of pairs)
        _particleSpawnQueue.push({ ...pair, col, sz, phase: ev.type, delay });
    }

    batchBaseDelay += count * _SPAWN_FRAMES_GAP + _SPAWN_FRAMES_GAP;
  }
}

/**
 * Called once per draw frame.  Decrements delays and spawns any items whose
 * delay has reached 0.  Each item spawns a fresh particle from t=0 so it
 * does a full visible arc — no pre-displaced ghost particles.
 */
function _drainSpawnQueue() {
  if (!_particleSpawnQueue.length) return;
  const remaining = [];
  for (const item of _particleSpawnQueue) {
    if (item.delay <= 0) {
      particles.push(_mkParticle(item.fp.clone(), item.tp.clone(), item.lane,
                                 item.col, item.sz, item.phase, _EV_SPEED));
    } else {
      item.delay--;
      remaining.push(item);
    }
  }
  _particleSpawnQueue = remaining;
}

/** Clear all live particles and pending spawn queue (called when cluster goes idle). */
function _clearAllParticles() {
  particles.forEach(p => { if (p.mesh) p.mesh.visible = false; });
  particles = [];
  _particleSpawnQueue = [];
}
