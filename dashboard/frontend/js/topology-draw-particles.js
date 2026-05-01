// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — draw helpers: connections + particle spawning
// (extracted from draw() to stay under 200 lines per file)
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

/** Spawn training gradient/weight particles. */
function _drawTrainingParticles(ts, coord, peers, workers, isTraining, isClassicDP, trainingPacketsActive, trainIoTrafficHot) {
  const _pSpeed = () => Math.min(0.045, Math.max(0.008, 1.2 / (_gradIntervalMs / 1000) * 0.018));
  const _spawnGate = Math.max(300, _gradIntervalMs * 0.9);
  const _trainIoLegSpeed = () => {
    const halfLegMs = Math.max(60, Math.min(_trainIoIntervalMs, _trainIoRttMs * 0.5));
    return Math.min(0.12, Math.max(0.018, 0.006 / (halfLegMs / 1000)));
  };

  const lastTrainIoTrafficTs = Math.max(_trainIoReqTs || 0, _trainIoRespTs || 0);
  if (isTraining && trainIoTrafficHot && coord && peers.length) {
    const hasReq  = particles.some(p => p.phase === 'train_req');
    const hasResp = particles.some(p => p.phase === 'train_resp');
    if (_trainIoPendingReq > 0 && !hasReq) {
      peers.forEach(w => {
        const cp3 = nodeMeshes.get(coord.h)?.group.position;
        const tp  = nodeMeshes.get(w.h)?.group.position;
        if (cp3 && tp) particles.push(_mkParticle(cp3.clone(), tp.clone(), -1, '23,126,137', 5.5, 'train_req', _trainIoLegSpeed()));
      });
      _trainIoPendingReq -= 1;
      spawnTs = ts;
    }
    if (_trainIoPendingResp > 0 && !hasResp) {
      peers.forEach(w => {
        const cp3 = nodeMeshes.get(coord.h)?.group.position;
        const tp  = nodeMeshes.get(w.h)?.group.position;
        if (cp3 && tp) particles.push(_mkParticle(tp.clone(), cp3.clone(), 1, '208,106,47', 4.7, 'train_resp', _trainIoLegSpeed()));
      });
      _trainIoPendingResp -= 1;
    }
  } else if (trainingPacketsActive && isClassicDP && isTraining) {
    if (workers.length >= 2 && ts - spawnTs > _spawnGate) {
      const hasG = particles.some(p => p.phase === 'gradients');
      const hasW = particles.some(p => p.phase === 'weights');
      if (!hasG && !hasW && !particles.length) {
        for (let i = 0; i < workers.length; i++)
          for (let j = 0; j < workers.length; j++) {
            if (i === j) continue;
            const fp = nodeMeshes.get(workers[i].h)?.group.position;
            const tp = nodeMeshes.get(workers[j].h)?.group.position;
            if (fp && tp) particles.push(_mkParticle(fp.clone(), tp.clone(), 1, '26,158,99', 4.8, 'gradients', _pSpeed()));
          }
        spawnTs = ts;
      }
      const maxG = Math.max(0, ...particles.filter(p=>p.phase==='gradients').map(p=>p.t));
      if (maxG > 0.78 && !particles.some(p=>p.phase==='weights'))
        for (let i = 0; i < workers.length; i++)
          for (let j = 0; j < workers.length; j++) {
            if (i === j) continue;
            const fp = nodeMeshes.get(workers[i].h)?.group.position;
            const tp = nodeMeshes.get(workers[j].h)?.group.position;
            if (fp && tp) particles.push(_mkParticle(fp.clone(), tp.clone(), -1, '209,105,48', 4.5, 'weights', _pSpeed()));
          }
    }
  } else if (trainingPacketsActive && coord && peers.length && ts - spawnTs > _spawnGate) {
    const hasG = particles.some(p=>p.phase==='gradients');
    const hasW = particles.some(p=>p.phase==='weights');
    if (!hasG && !hasW && !particles.length) {
      peers.forEach(w => {
        const fp  = nodeMeshes.get(w.h)?.group.position;
        const cp3 = nodeMeshes.get(coord.h)?.group.position;
        if (fp && cp3) particles.push(_mkParticle(fp.clone(), cp3.clone(), 1, '26,158,99', 6, 'gradients', _pSpeed()));
      });
      spawnTs = ts;
    }
    const maxG = Math.max(0, ...particles.filter(p=>p.phase==='gradients').map(p=>p.t));
    if (maxG > 0.8 && !particles.some(p=>p.phase==='weights'))
      peers.forEach(w => {
        const cp3 = nodeMeshes.get(coord.h)?.group.position;
        const tp  = nodeMeshes.get(w.h)?.group.position;
        if (cp3 && tp) particles.push(_mkParticle(cp3.clone(), tp.clone(), -1, '209,105,48', 5.5, 'weights', _pSpeed()));
      });
  }

  if (!trainingPacketsActive && !trainIoTrafficHot) {
    particles.forEach(p => { if (p.mesh) p.mesh.visible = false; });
    particles = [];
    if (!isTraining) { _trainIoPendingReq = 0; _trainIoPendingResp = 0; }
  }
}

/** Spawn inference request/return particles. */
function _drawInferenceParticles(ts, inferCoord, inferPeers, isInferring, isClassicDP, workers) {
  const _iSpeed = () => Math.min(0.065, Math.max(0.030, 0.006 / (_tokenIntervalMs / 1000)));
  const _iSpawnGate = Math.max(100, _tokenIntervalMs * 0.75);

  if (isInferring && isClassicDP && workers.length >= 2) {
    const tokenOk = state.token_ts && (Date.now()/1000 - state.token_ts) < 4.0;
    if (tokenOk && ts - inferSpawnTs > _iSpawnGate) {
      const hasOutbound = inferParticles.some(p => p.phase === 'outbound');
      const hasReturn   = inferParticles.some(p => p.phase === 'return');
      if (!hasOutbound && !hasReturn && !inferParticles.length) {
        for (let i = 0; i < workers.length; i++)
          for (let j = 0; j < workers.length; j++) {
            if (i === j) continue;
            const fp = nodeMeshes.get(workers[i].h)?.group.position;
            const tp = nodeMeshes.get(workers[j].h)?.group.position;
            if (fp && tp) inferParticles.push(_mkParticle(fp.clone(), tp.clone(), -1, '23,126,137', 4.8, 'outbound', _iSpeed()));
          }
        inferSpawnTs = ts;
      }
      const maxO = Math.max(0, ...inferParticles.filter(p=>p.phase==='outbound').map(p=>p.t));
      if (maxO > 0.78 && !inferParticles.some(p=>p.phase==='return'))
        for (let i = 0; i < workers.length; i++)
          for (let j = 0; j < workers.length; j++) {
            if (i === j) continue;
            const fp = nodeMeshes.get(workers[i].h)?.group.position;
            const tp = nodeMeshes.get(workers[j].h)?.group.position;
            if (fp && tp) inferParticles.push(_mkParticle(fp.clone(), tp.clone(), 1, '208,106,47', 4.2, 'return', _iSpeed()));
          }
    }
  } else if (isInferring && inferCoord && inferPeers.length) {
    const tokenOk = state.token_ts && (Date.now()/1000 - state.token_ts) < 4.0;
    if (tokenOk && ts - inferSpawnTs > _iSpawnGate) {
      const hasOutbound = inferParticles.some(p => p.phase === 'outbound');
      const hasReturn   = inferParticles.some(p => p.phase === 'return');
      if (!hasOutbound && !hasReturn && !inferParticles.length) {
        inferPeers.forEach(w => {
          const ic3 = nodeMeshes.get(inferCoord.h)?.group.position;
          const tp  = nodeMeshes.get(w.h)?.group.position;
          if (ic3 && tp) inferParticles.push(_mkParticle(ic3.clone(), tp.clone(), -1, '23,126,137', 5.5, 'outbound', _iSpeed()));
        });
        inferSpawnTs = ts;
      }
      const maxO = Math.max(0, ...inferParticles.filter(p=>p.phase==='outbound').map(p=>p.t));
      if (maxO > 0.8 && !inferParticles.some(p=>p.phase==='return'))
        inferPeers.forEach(w => {
          const ic3 = nodeMeshes.get(inferCoord.h)?.group.position;
          const tp  = nodeMeshes.get(w.h)?.group.position;
          if (ic3 && tp) inferParticles.push(_mkParticle(tp.clone(), ic3.clone(), 1, '208,106,47', 4.5, 'return', _iSpeed()));
        });
    }
  }

  if (!isInferring) {
    inferParticles.forEach(p => { if (p.mesh) p.mesh.visible = false; });
    inferParticles = [];
  }
}
