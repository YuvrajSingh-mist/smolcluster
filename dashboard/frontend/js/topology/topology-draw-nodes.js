// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — draw helpers: node sync + crown overlay
// (extracted from draw() to stay under 200 lines per file)
// ════════════════════════════════════════════════════════════════════════════

/** Sync Three.js node mesh positions and materials for one frame. */
function _drawSyncNodes(ts, server, workers, crownedHost, isActive) {
  const keepSet  = new Set();
  const allNodes = server ? [server, ...workers] : workers;
  for (const n of allNodes) {
    keepSet.add(n.h);
    const entry    = _ensureNode(n.h, n === server);
    const newLabel = (n === server) ? 'SERVER' : `RANK ${n.rank ?? 0}`;
    if (entry.lastLabel !== newLabel) {
      entry.lastLabel = newLabel;
      const r2 = entry.isServer ? 0.72 : 0.6;
      const subLabel = state.ssh_aliases[n.h] || state.discovered[n.h]?.alias || n.h;
      const { sprite, tex } = _makeLabel(newLabel, subLabel, entry.hexCol);
      sprite.position.y = r2 + 0.92;
      entry.group.remove(entry.labelSprite);
      entry.labelTex.dispose();
      entry.labelSprite = sprite; entry.labelTex = tex;
      entry.group.add(sprite);
    }
    if (!_dragEntry || _dragEntry.entry !== entry) {
      const tgt = _manualNodePos.get(n.h) || _toWorld(n.x, n.y);
      entry.group.position.lerp(tgt, 0.12);
    }

    const pulse = 0.5 + 0.5 * Math.sin(ts / 380);
    const isCrowned = crownedHost === n.h;
    const spin = ts * 0.001;
    const baseCol = entry.isServer ? 0xd16930 : 0x167d90;
    const baseEm  = entry.isServer ? 0x6b2810 : 0x073c44;
    const crownCol = 0xd16930;
    const crownAccent = 0xf0a060;
    const crownEm = 0x8a3a12;

    if (entry.ringA && entry.ringB) {
      entry.ringA.rotation.z = spin * (entry.isServer ? 1.35 : 1.1);
      entry.ringB.rotation.y = spin * (entry.isServer ? -1.15 : -0.95);
      entry.ringB.rotation.z = spin * 0.55;
    }

    if (isCrowned) {
      entry.mat.color.set(crownCol);
      entry.mat.emissive.set(crownEm);
      entry.mat.emissiveIntensity = 0.34 + 0.14 * pulse;
      if (entry.ringMatA && entry.ringMatB) {
        entry.ringMatA.color.set(crownCol);
        entry.ringMatA.emissive.set(crownCol);
        entry.ringMatA.opacity = 0.50 + 0.14 * pulse;
        entry.ringMatA.emissiveIntensity = 0.42 + 0.16 * pulse;
        entry.ringMatB.color.set(crownAccent);
        entry.ringMatB.emissive.set(crownAccent);
        entry.ringMatB.opacity = 0.30 + 0.12 * pulse;
        entry.ringMatB.emissiveIntensity = 0.28 + 0.12 * pulse;
      }
    } else {
      entry.mat.color.set(baseCol);
      entry.mat.emissive.set(baseEm);
      entry.mat.emissiveIntensity = n.running ? 0.42 + 0.22 * pulse : 0.18;
      if (entry.ringMatA && entry.ringMatB) {
        entry.ringMatA.color.set(baseCol);
        entry.ringMatA.emissive.set(baseCol);
        entry.ringMatB.color.set(baseCol);
        entry.ringMatB.emissive.set(baseCol);
        if (n.running) {
          entry.ringMatA.opacity = 0.42 + 0.14 * pulse;
          entry.ringMatA.emissiveIntensity = 0.4 + 0.15 * pulse;
          entry.ringMatB.opacity = 0.28 + 0.12 * pulse;
          entry.ringMatB.emissiveIntensity = 0.3 + 0.12 * pulse;
        } else {
          entry.ringMatA.opacity = 0.2;
          entry.ringMatA.emissiveIntensity = 0.1;
          entry.ringMatB.opacity = 0.12;
          entry.ringMatB.emissiveIntensity = 0.08;
        }
      }
    }
  }
  _removeStaleNodes(keepSet);
}

/** Update crown + token HTML overlays once per frame. */
function _drawCrownOverlay(ts, crownedEntry, isTraining, isInferring) {
  if (_SHOW_CROWN_OVERLAY && crownedEntry && isTraining) {
    const wpos = crownedEntry.group.position;
    const sc   = _projectNode(wpos, 2.05);
    if (sc.behind) {
      _crownEl.style.display = 'none';
    } else {
      const bob = 4 * Math.sin(ts / 540);
      _crownEl.style.display = 'block';
      _crownEl.style.left    = sc.x + 'px';
      _crownEl.style.top     = (sc.y + bob) + 'px';
    }
  }
  if (_SHOW_CROWN_OVERLAY && crownedEntry && dashboardMode === 'infer' && isInferring) {
    const tokenStamp = Number(state.token_ts) || 0;
    if (tokenStamp && tokenStamp !== _lastTokenStamp) {
      _lastTokenStamp = tokenStamp;
      const tok = String(state.token_text || '');
      if (tok) _queueToken(tok);
      if (!tok && !_lastTokenText) {
        _lastTokenText = '\u258c';
        _lastTokenRaw = '';
        _tokenFlashTs = ts;
      }
    }
    const hasReturnPacket = inferParticles.some(p => p.phase === 'return');
    const returnReachedServer = inferParticles.some(p => p.phase === 'return' && p.t >= 0.94);
    const queued = _pendingTokens[0];
    const agedOut = queued ? (ts - queued.queuedAt) >= _tokenFallbackDelayMs : false;
    let _newTokenPulse = false;
    if (queued && (returnReachedServer || (!hasReturnPacket && agedOut) || agedOut)) {
      _pendingTokens.shift();
      _lastTokenRaw = queued.raw;
      _lastTokenText = queued.display.length > 24 ? queued.display.slice(0, 23) + '\u2026' : queued.display;
      _tokenFlashTs = ts;
      _newTokenPulse = true;
    }
    const wpos = crownedEntry.group.position;
    const sc   = _projectNode(wpos, 2.05);
    if (sc.behind) {
      _crownEl.style.display = 'none';
      _tokenEl.style.display = 'none';
    } else {
      const bob = 4 * Math.sin(ts / 540);
      _crownEl.style.display = 'block';
      _crownEl.style.left    = sc.x + 'px';
      _crownEl.style.top     = (sc.y + bob) + 'px';
      const sinceFlash = ts - _tokenFlashTs;
      const flashDur   = 2400;
      if (sinceFlash < flashDur && _lastTokenText) {
        const alpha  = Math.max(0, 1 - sinceFlash / flashDur);
        const drift  = sinceFlash * 0.016;
        _tokenEl.style.display = 'block';
        _tokenEl.style.left    = sc.x + 'px';
        _tokenEl.style.top     = (sc.y - 40 - drift + bob) + 'px';
        _tokenEl.style.opacity = alpha;
        if (_tokenEl.dataset.txt !== _lastTokenText) {
          _tokenEl.textContent = _lastTokenText.length > 24 ? _lastTokenText.slice(0,23) + '\u2026' : _lastTokenText;
          _tokenEl.dataset.txt = _lastTokenText;
        }
        if (_newTokenPulse && _lastTokenRaw) _tokenDockActive = true;
      } else {
        _tokenEl.style.display = 'none';
      }
    }
  } else {
    _crownEl.style.display = 'none';
    _tokenEl.style.display = 'none';
  }
}
