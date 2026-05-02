// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — node meshes + label sprites
// ════════════════════════════════════════════════════════════════════════════
const nodeMeshes = new Map();

function _makeLabel(label, sub, hexCol) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 90;
  const ctx = c.getContext('2d');
  ctx.fillStyle = hexCol;
  ctx.font = 'bold 26px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillText(label, 128, 36);
  ctx.fillStyle = '#5a7a77';
  ctx.font = '18px "IBM Plex Mono", monospace';
  ctx.fillText(sub.length > 15 ? sub.slice(0, 14) + '\u2026' : sub, 128, 64);
  const tex = new THREE.CanvasTexture(c);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sprite.scale.set(2.6, 0.86, 1);
  return { sprite, tex };
}

function _ensureNode(h, isServer) {
  if (nodeMeshes.has(h)) {
    const entry = nodeMeshes.get(h);
    if (entry.isServer !== isServer) {
      const hexCol = isServer ? '#d16930' : '#167d90';
      const col3   = isServer ? 0xd16930 : 0x167d90;
      const emis3  = isServer ? 0x6b2810 : 0x073c44;
      entry.isServer = isServer;
      entry.hexCol = hexCol;
      entry.mat.color.set(col3);
      entry.mat.emissive.set(emis3);
      if (entry.ringMatA && entry.ringMatB) {
        entry.ringMatA.color.set(col3);
        entry.ringMatA.emissive.set(col3);
        entry.ringMatB.color.set(col3);
        entry.ringMatB.emissive.set(col3);
      }
    }
    return entry;
  }
  const hexCol = isServer ? '#d16930' : '#167d90';
  const col3   = isServer ? 0xd16930 : 0x167d90;
  const emis3  = isServer ? 0x6b2810 : 0x073c44;
  const r      = isServer ? 0.72 : 0.6;
  const group  = new THREE.Group();

  const mat = new THREE.MeshPhongMaterial({
    color: col3, emissive: emis3, emissiveIntensity: 0.28,
    shininess: 90, transparent: true, opacity: 0.92,
  });
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 48), mat);
  group.add(sphere);

  group.add(new THREE.Mesh(
    new THREE.SphereGeometry(r + 0.14, 32, 32),
    new THREE.MeshPhongMaterial({ color: col3, emissive: col3, emissiveIntensity: 0.1,
      transparent: true, opacity: 0.07, side: THREE.BackSide, depthWrite: false })
  ));

  const ringMatA = new THREE.MeshPhongMaterial({ color: col3, emissive: col3, emissiveIntensity: 0.12, transparent: true, opacity: 0.24 });
  const ringA = new THREE.Mesh(new THREE.TorusGeometry(r + 0.2, 0.028, 14, 80), ringMatA);
  ringA.rotation.x = Math.PI / 2;
  group.add(ringA);

  const ringMatB = new THREE.MeshPhongMaterial({ color: col3, emissive: col3, emissiveIntensity: 0.08, transparent: true, opacity: 0.16 });
  const ringB = new THREE.Mesh(new THREE.TorusGeometry(r + 0.28, 0.02, 14, 80), ringMatB);
  ringB.rotation.x = Math.PI / 2;
  ringB.rotation.y = Math.PI / 5;
  group.add(ringB);

  const { sprite, tex } = _makeLabel('\u2026', h, hexCol);
  sprite.position.y = r + 0.92;
  group.add(sprite);

  _T3scene.add(group);
  const entry = { group, sphere, mat, h, isServer, hexCol, isDragged: false,
                  labelSprite: sprite, labelTex: tex, lastLabel: '',
                  ringA, ringB, ringMatA, ringMatB };
  nodeMeshes.set(h, entry);
  return entry;
}

function _removeStaleNodes(keepSet) {
  for (const [h, entry] of nodeMeshes) {
    if (!keepSet.has(h)) {
      _T3scene.remove(entry.group);
      if (entry.labelTex) entry.labelTex.dispose();
      nodeMeshes.delete(h);
      _manualNodePos.delete(h);
    }
  }
}
