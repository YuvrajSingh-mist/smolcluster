// ════════════════════════════════════════════════════════════════════════════
// 3D Topology — state variables + timing helpers
// ════════════════════════════════════════════════════════════════════════════
let particles      = [];
let spawnTs        = 0;
let inferParticles = [];
let inferSpawnTs   = 0;
let _activeAlgo    = '';
let _prevGradTs      = 0;    // last grad_ts value we saw
let _gradIntervalMs  = 3000; // measured ms between successive grad pings (starts at 3s)
let _prevTokenTs     = 0;    // last token_ts value we saw
let _tokenIntervalMs = 200;  // measured ms between successive token pings (starts at 200ms)
let _trainIoReqTs       = 0;   // last observed training request->worker event (epoch ms)
let _trainIoRespTs      = 0;   // last observed training response<-worker event (epoch ms)
let _trainIoRttMs       = 650; // measured request/response round trip for training transport
let _trainIoIntervalMs  = 700; // measured interval between successive training requests
let _lastTrainIoReqPerf = 0;
let _lastTrainIoRespPerf = 0;
let _trainIoPendingReq = 0;
let _trainIoPendingResp = 0;

function _markTrainIoRequestEvent() {
  const nowPerf = performance.now();
  if (_lastTrainIoReqPerf > 0) {
    const measuredReq = nowPerf - _lastTrainIoReqPerf;
    if (measuredReq > 60 && measuredReq < 30000) {
      _trainIoIntervalMs = _trainIoIntervalMs * 0.6 + measuredReq * 0.4;
    }
  }
  _lastTrainIoReqPerf = nowPerf;
  _trainIoReqTs = Date.now();
  _trainIoPendingReq += 1;
}

function _markTrainIoResponseEvent() {
  const nowPerf = performance.now();
  if (_lastTrainIoReqPerf > 0) {
    const measured = nowPerf - _lastTrainIoReqPerf;
    if (measured > 80 && measured < 15000) {
      _trainIoRttMs = _trainIoRttMs * 0.55 + measured * 0.45;
    }
  }
  _lastTrainIoRespPerf = nowPerf;
  _trainIoRespTs = Date.now();
  _trainIoPendingResp += 1;
}
