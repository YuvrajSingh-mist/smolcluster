// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — Three.js scene setup
// ════════════════════════════════════════════════════════════════════════════
const _T3mount = $('topo3d');
const _T3scene = new THREE.Scene();
const _T3camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
_T3camera.position.set(0, 10, 12);
_T3camera.lookAt(0, 0, 0);
const _T3renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
_T3renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
_T3renderer.setClearColor(0x000000, 0);
_T3mount.appendChild(_T3renderer.domElement);

// ── Lighting ───────────────────────────────────────────────────────────────────
_T3scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const _T3sun = new THREE.DirectionalLight(0xffffff, 0.7);
_T3sun.position.set(6, 12, 8); _T3scene.add(_T3sun);
const _T3fill = new THREE.DirectionalLight(0x88ccee, 0.25);
_T3fill.position.set(-5, -2, -6); _T3scene.add(_T3fill);

// ── Grid floor ────────────────────────────────────────────────────────────────
const _T3grid = new THREE.GridHelper(36, 28, 0x2aa8bf, 0x167d90);
_T3grid.position.y = -0.85;
const _T3gridMats = Array.isArray(_T3grid.material) ? _T3grid.material : [_T3grid.material];
_T3gridMats.forEach(m => { m.transparent = true; m.depthWrite = false; });
_T3scene.add(_T3grid);

function _t3SyncGrid() {
  const dark = document.documentElement.classList.contains('dark');
  const opacity = dark ? 0.55 : 0.35;
  _T3gridMats.forEach(m => { m.opacity = opacity; });
}
_t3SyncGrid();

// ── Orbit controls ─────────────────────────────────────────────────────────────
const _T3orbit = new THREE.OrbitControls(_T3camera, _T3renderer.domElement);
_T3orbit.enableDamping = true;
_T3orbit.dampingFactor = 0.085;
_T3orbit.minDistance = 4;
_T3orbit.maxDistance = 40;
_T3orbit.maxPolarAngle = Math.PI * 0.48;

// ── Resize ─────────────────────────────────────────────────────────────────────
function _t3Resize() {
  const w = _T3mount.clientWidth || 600;
  const h = _T3mount.clientHeight || 400;
  _T3renderer.setSize(w, h);
  _T3camera.aspect = w / h;
  _T3camera.updateProjectionMatrix();
}
new ResizeObserver(_t3Resize).observe(_T3mount);
setTimeout(_t3Resize, 0);
