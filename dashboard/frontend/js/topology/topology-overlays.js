// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — HTML overlays (crown, token flash, empty state)
// ════════════════════════════════════════════════════════════════════════════
const _T3overlay = document.createElement('div');
_T3overlay.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden';
_T3mount.appendChild(_T3overlay);

const _crownEl = document.createElement('div');
_crownEl.style.cssText = [
  'position:absolute',
  'display:none',
  'transform:translate(-50%, -100%)',
  'font-size:28px',
  'line-height:1',
  'color:#d16930',
  'text-shadow:0 0 10px rgba(209,105,48,0.9), 0 0 22px rgba(209,105,48,0.5)',
  'will-change:transform'
].join(';');
_crownEl.textContent = '';
_T3overlay.appendChild(_crownEl);

const _tokenEl = document.createElement('div');
_tokenEl.style.cssText = [
  'position:absolute',
  'display:none',
  'transform:translate(-50%, -100%)',
  'font:700 13px \'IBM Plex Mono\',monospace',
  'color:#d16930',
  'white-space:nowrap',
  'text-shadow:0 0 8px rgba(209,105,48,0.9)',
  'pointer-events:none',
  'will-change:transform,opacity',
  'transition:opacity 0.1s'
].join(';');
_T3overlay.appendChild(_tokenEl);

let _tokenDockActive = false;
const _SHOW_TOPO_STATUS_HUD = false;
const _SHOW_CROWN_OVERLAY = false;

// Track last token text from logs
let _lastTokenText  = '';
let _lastTokenRaw   = '';
let _tokenFlashTs   = -999999; // performance.now() when last token arrived
let _lastTokenStamp = 0;       // last seen state.token_ts from SSE
let _pendingTokens  = [];      // queued SSE tokens waiting for packet-arrival reveal
const _tokenFallbackDelayMs = 760;

function _queueToken(raw) {
  if (typeof raw !== 'string' || raw.length === 0) return;
  _pendingTokens.push({
    raw,
    display: raw.replace(/\n/g, '\u21b5'),
    queuedAt: performance.now(),
  });
  if (_pendingTokens.length > 420) _pendingTokens.splice(0, _pendingTokens.length - 260);
}

function _projectNode(worldPos, yLift = 1.1) {
  const v = worldPos.clone();
  v.y += yLift;
  v.project(_T3camera);
  const el = _T3renderer.domElement;
  return {
    x: (v.x  + 1) / 2 * el.clientWidth,
    y: (-v.y + 1) / 2 * el.clientHeight,
    behind: v.z > 1
  };
}

let _emptyShown = true;
const _T3emptyDiv = document.createElement('div');
_T3emptyDiv.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;pointer-events:none;';
_T3emptyDiv.innerHTML = [
  '<div style="font:14px \'Space Grotesk\',sans-serif;color:rgba(15,41,38,0.22)">Add nodes from the left panel</div>',
  '<div style="font:12px \'Space Grotesk\',sans-serif;color:rgba(15,41,38,0.12);margin-top:6px">Cluster topology will appear here</div>',
  '<div style="font:9.5px \'IBM Plex Mono\',monospace;color:rgba(15,41,38,0.10);margin-top:5px">drag nodes &middot; orbit to rotate &middot; scroll to zoom</div>'
].join('');
_T3mount.appendChild(_T3emptyDiv);
