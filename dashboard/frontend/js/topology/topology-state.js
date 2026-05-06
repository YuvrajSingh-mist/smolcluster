// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — state variables + event queue
// ════════════════════════════════════════════════════════════════════════════
let particles      = [];
let spawnTs        = 0;
let inferParticles = [];
let inferSpawnTs   = 0;
let _activeAlgo    = '';

// Interval tracking for metrics display (updated from SSE, not used for spawning).
let _prevGradTs      = 0;
let _gradIntervalMs  = 3000;
let _prevTokenTs     = 0;
let _tokenIntervalMs = 200;

// ── Typed event queue ────────────────────────────────────────────────────────
// Each entry: { type: 'gradients'|'weights'|'rollout'|'weight_sync',
//               dir:  'in'|'out',
//               arch: 'syncps'|'classicdp'|'fsdp'|'grpo',
//               count: number }
// The draw loop drains this every frame.
let _smolEventQueue = [];

// ── Deferred particle spawn queue ─────────────────────────────────────────────
// Each entry: { fp, tp, lane, col, sz, phase, delay }
// Particles are spawned one wave at a time (every _SPAWN_FRAMES_GAP frames)
// so each rollout/packet gets its own clearly visible journey from t=0.
let _particleSpawnQueue = [];

// RTT tracking per event type (for metrics display only).
const _evOutTs = {};
let _trainIoRttMs = 0;
let _inferIoRttMs = 0;

// Tracks last algo key rendered in the legend so we only rebuild when it changes.
let _lastLegendAlgo = '';

function _handleSmolEvent(ev) {
  const type = ev.type || '';
  if (!type) return;

  if (ev.dir === 'out') {
    _evOutTs[type] = performance.now();
  } else if (ev.dir === 'in' && _evOutTs[type] > 0) {
    const rtt = performance.now() - _evOutTs[type];
    if (type === 'gradients' || type === 'weights') _trainIoRttMs = rtt;
    if (type === 'rollout' || type === 'weight_sync') _inferIoRttMs = rtt;
    delete _evOutTs[type];
  }

  _smolEventQueue.push(ev);
}
