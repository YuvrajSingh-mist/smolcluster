// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — connection lines + particle pool
// ════════════════════════════════════════════════════════════════════════════
const _connObjs = [];

function _clearConns() {
  for (const c of _connObjs) {
    _T3scene.remove(c.l1); _T3scene.remove(c.l2);
    c.geo1.dispose(); c.geo2.dispose();
    c.mat1.dispose(); c.mat2.dispose();
  }
  _connObjs.length = 0;
}

function _addConn(p0, p1, active) {
  const p0u = p0.clone(); p0u.y += 0.72;
  const p1u = p1.clone(); p1u.y += 0.72;
  const mid  = p0u.clone().lerp(p1u, 0.5);
  const axis = new THREE.Vector3().subVectors(p1u, p0u).normalize();
  const perp = new THREE.Vector3().crossVectors(axis, new THREE.Vector3(0, 1, 0)).normalize();
  const OFF  = 0.42;
  function bezPts(sign) {
    const cp = mid.clone().addScaledVector(perp, sign * OFF);
    cp.y += 1.15;
    const pts = [];
    for (let i = 0; i <= 24; i++) {
      const t = i/24, m = 1-t;
      pts.push(new THREE.Vector3(
        m*m*p0u.x + 2*m*t*cp.x + t*t*p1u.x,
        m*m*p0u.y + 2*m*t*cp.y + t*t*p1u.y,
        m*m*p0u.z + 2*m*t*cp.z + t*t*p1u.z
      ));
    }
    return pts;
  }
  function makeTube(sign, color, opacity) {
    const curve = new THREE.CatmullRomCurve3(bezPts(sign));
    const geo   = new THREE.TubeGeometry(curve, 24, 0.01, 8, false);
    const mat   = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: false, depthWrite: false });
    const mesh  = new THREE.Mesh(geo, mat);
    mesh.renderOrder = 9;
    return { mesh, geo, mat };
  }
  const t1 = makeTube( 1, active ? 0x22c97d : 0x5de0f5, active ? 0.95 : 0.92);
  const t2 = makeTube(-1, active ? 0xe8823a : 0x5de0f5, active ? 0.90 : 0.88);
  _T3scene.add(t1.mesh); _T3scene.add(t2.mesh);
  _connObjs.push({ l1: t1.mesh, l2: t2.mesh, geo1: t1.geo, geo2: t2.geo, mat1: t1.mat, mat2: t2.mat });
}

// ── Particle pool ─────────────────────────────────────────────────────────────
const _pGeo  = new THREE.SphereGeometry(1, 10, 10);
const _pPool = [];

function _getParticleMesh(col, sz) {
  let m = _pPool.find(x => !x.visible);
  if (!m) {
    m = new THREE.Mesh(_pGeo, new THREE.MeshBasicMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
    }));
    _T3scene.add(m); _pPool.push(m);
  }
  m.material.color.set(new THREE.Color(`rgb(${col})`));
  m.scale.setScalar(sz * 0.028);
  m.visible = true;
  return m;
}

function _qb3(t, p0, cp, p1) {
  const m = 1 - t;
  return new THREE.Vector3(
    m*m*p0.x + 2*m*t*cp.x + t*t*p1.x,
    m*m*p0.y + 2*m*t*cp.y + t*t*p1.y,
    m*m*p0.z + 2*m*t*cp.z + t*t*p1.z
  );
}

function _mkParticle(fp, tp, lane, col, sz, phase, speed = 0.018) {
  const p0u = fp.clone(); p0u.y += 0.72;
  const p1u = tp.clone(); p1u.y += 0.72;
  const mid  = p0u.clone().lerp(p1u, 0.5);
  const axis = new THREE.Vector3().subVectors(p1u, p0u).normalize();
  const perp = new THREE.Vector3().crossVectors(axis, new THREE.Vector3(0, 1, 0)).normalize();
  const cp   = mid.clone().addScaledVector(perp, lane * 0.42);
  cp.y += 1.15;
  return { fp: p0u, tp: p1u, cp, t: 0, speed: speed + Math.random() * 0.003, col, sz, phase,
           mesh: _getParticleMesh(col, sz) };
}
