// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — drag controls (XZ-plane node dragging)
// ════════════════════════════════════════════════════════════════════════════
const _T3ray     = new THREE.Raycaster();
const _T3ptr     = new THREE.Vector2();
const _dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const _dragHit   = new THREE.Vector3();
let   _dragEntry = null;
const _manualNodePos = new Map(); // hostname -> THREE.Vector3 world pos pinned by drag
let   _lastLayout    = null;       // last getLayout() result, used by resetTopologyLayout()

function _setT3Ptr(e) {
  const el = _T3renderer.domElement;
  _T3ptr.x =  (e.offsetX / el.clientWidth)  * 2 - 1;
  _T3ptr.y = -(e.offsetY / el.clientHeight) * 2 + 1;
}

_T3renderer.domElement.addEventListener('pointerdown', e => {
  _setT3Ptr(e);
  _T3ray.setFromCamera(_T3ptr, _T3camera);
  const hit = _T3ray.intersectObjects([...nodeMeshes.values()].map(n => n.sphere), false)[0];
  if (hit) {
    const entry = [...nodeMeshes.values()].find(n => n.sphere === hit.object);
    if (entry) {
      entry.isDragged = false;
      _T3ray.ray.intersectPlane(_dragPlane, _dragHit);
      _dragEntry = { entry, ox: _dragHit.x - entry.group.position.x, oz: _dragHit.z - entry.group.position.z };
      _T3orbit.enabled = false;
      _T3renderer.domElement.style.cursor = 'grabbing';
    }
  }
});

_T3renderer.domElement.addEventListener('pointermove', e => {
  _setT3Ptr(e);
  if (_dragEntry) {
    _T3ray.setFromCamera(_T3ptr, _T3camera);
    _T3ray.ray.intersectPlane(_dragPlane, _dragHit);
    _dragEntry.entry.isDragged = true;
    _dragEntry.entry.group.position.set(_dragHit.x - _dragEntry.ox, 0, _dragHit.z - _dragEntry.oz);
    _manualNodePos.set(_dragEntry.entry.h, _dragEntry.entry.group.position.clone());
    _T3renderer.domElement.style.cursor = 'grabbing';
  } else {
    _T3ray.setFromCamera(_T3ptr, _T3camera);
    const hit = _T3ray.intersectObjects([...nodeMeshes.values()].map(n => n.sphere), false)[0];
    _T3renderer.domElement.style.cursor = hit ? 'grab' : 'default';
  }
});

_T3renderer.domElement.addEventListener('pointerup', e => {
  const wasDrag = _dragEntry?.entry.isDragged;
  _dragEntry = null;
  _T3orbit.enabled = true;
  _T3renderer.domElement.style.cursor = 'default';
  if (wasDrag) return;
  _setT3Ptr(e);
  _T3ray.setFromCamera(_T3ptr, _T3camera);
  const hit = _T3ray.intersectObjects([...nodeMeshes.values()].map(n => n.sphere), false)[0];
  if (hit) {
    const entry = [...nodeMeshes.values()].find(n => n.sphere === hit.object);
    if (entry) {
      selectedServer = (selectedServer === entry.h) ? null : entry.h;
      generationAutoFillFromSelectedNode(true);
    }
  }
});

_T3renderer.domElement.addEventListener('click', e => {
  // Handled in pointerup; keeping listener to prevent bubbling issues only
});
