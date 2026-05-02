// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — layout computation
// ════════════════════════════════════════════════════════════════════════════
const _VC_W = 600, _VC_H = 420;

function getLayout() {
  const W = _VC_W, H = _VC_H;
  const raw = {};
  for (const [h,v] of Object.entries(state.selected)) raw[h] = {...v, running:false};
  for (const [h,v] of Object.entries(state.running))  raw[h] = {...v, running:true};
  const all = dedupByIp(raw);

  const algo = _activeAlgo || $('algo-sel').value;
  const hasServer = !['classicdp','fsdp','ep','mp_pipeline'].includes(algo);
  const isSyncPsStyle = (algo === 'syncps' || algo === 'grpo');

  let server = null; const workers = [];
  for (const [h, info] of Object.entries(all)) {
    if (hasServer && (info.rank === 0 || info.role === 'server')) server = {h, ...info};
    else workers.push({h, ...info});
  }
  workers.sort((a,b) => a.rank - b.rank);

  const sx = W/2, sy = H * 0.22;

  // Non-ClassicDP flat topologies with 3+ nodes: circular polygon mesh
  if (!hasServer && algo !== 'classicdp' && workers.length >= 3) {
    const cx = W/2, cy = H/2;
    const radius = Math.min(W * 0.3, H * 0.34, 160);
    const a0 = -Math.PI / 2;
    return {
      W, H, hasServer, server: null,
      workers: workers.map((w, i) => {
        const angle = a0 + (2 * Math.PI * i) / workers.length;
        return {...w, x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle)};
      }),
    };
  }

  // SyncPS/GRPO: triangular pyramid — server at apex, workers fanned below.
  const nw = workers.length;

  if (isSyncPsStyle && hasServer && nw > 1) {
    const wy1 = H * 0.65;
    const wy2 = H * 0.80;
    const spBottom = Math.min(240, (W - 60) / Math.max(nw - 1, 1));

    const wpos = [];
    if (nw <= 3) {
      const wx0 = W/2 - (nw - 1) * spBottom / 2;
      for (let i = 0; i < nw; i++) {
        wpos.push({...workers[i], x: wx0 + i * spBottom, y: wy2});
      }
    } else {
      const bottomCount = Math.ceil(nw / 2);
      const topCount = nw - bottomCount;
      const spBot = Math.min(240, (W - 60) / (bottomCount - 1));
      const spTop = Math.min(160, (W - 60) / Math.max(topCount - 1, 1));
      const wx0Bot = W/2 - (bottomCount - 1) * spBot / 2;
      for (let i = 0; i < bottomCount; i++) {
        wpos.push({...workers[i], x: wx0Bot + i * spBot, y: wy2});
      }
      const wx0Top = W/2 - (topCount - 1) * spTop / 2;
      for (let i = 0; i < topCount; i++) {
        wpos.push({...workers[bottomCount + i], x: wx0Top + i * spTop, y: wy1});
      }
    }
    return {
      W, H, hasServer,
      server: server ? {...server, x: sx, y: sy} : null,
      workers: wpos,
    };
  }

  // Default layout (ClassicDP linear chain, or other server topologies)
  const wy = hasServer ? H * 0.74 : H * 0.5;
  const spMax = Math.min(155, (W - 100) / Math.max(nw - 1, 1));
  const sp = nw > 1 ? spMax : 0;
  const wx0 = W/2 - (nw - 1) * sp / 2;

  return {
    W, H, hasServer,
    server: server ? {...server, x: sx, y: sy} : null,
    workers: workers.map((w, i) => ({...w, x: wx0 + i * sp, y: wy})),
  };
}

// Layout pixel → Three.js world XZ
function _toWorld(px, py) {
  return new THREE.Vector3((px - _VC_W/2) / 40, 0, (py - _VC_H/2) / 40);
}

function _activeCrownedHost(server, workers) {
  if (selectedServer && (state.selected[selectedServer] || state.running[selectedServer] || state.discovered[selectedServer])) {
    return selectedServer;
  }
  if (server?.h) return server.h;
  if (workers && workers.length) {
    const rankZero = workers.find(w => Number(w.rank) === 0);
    return rankZero?.h || workers[0].h;
  }
  return null;
}
