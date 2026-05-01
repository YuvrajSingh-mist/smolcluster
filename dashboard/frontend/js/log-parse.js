// ════════════════════════════════════════════════════════════════════════════
// Log parsing helpers — called from appendLog
// ════════════════════════════════════════════════════════════════════════════

/** Handle [TRANSPORT_EVENT] JSON embedded in a log line. */
function _parseTransportEvent(line) {
  const m = line.match(/\[TRANSPORT_EVENT\]\s*(\{.+\})/);
  if (!m) return;
  try {
    const io = JSON.parse(m[1]);
    const phase = String(io.phase || io.event || io.type || '').toLowerCase();
    if (phase === 'request' || phase === 'req' || phase === 'outbound' || phase === 'send') {
      _markTrainIoRequestEvent();
    } else if (phase === 'response' || phase === 'resp' || phase === 'inbound' || phase === 'recv' || phase === 'receive') {
      _markTrainIoResponseEvent();
    }
  } catch (e) {}
}

/** Pattern-based transport event detection (no structured marker needed). */
function _checkTransportPatterns(line) {
  const lowerLine = String(line || '').toLowerCase();
  const looksLikeRequest =
    (/\brequest\b/i.test(line) && /\bn=\d+/i.test(line) && /vllm|worker/i.test(line))
    || /request n=\d+ comple/i.test(lowerLine)
    || /\b(dispatch|sending|send|submit|submitted)\b.*\b(prompt|rollout|batch|request|rpc)\b/.test(lowerLine)
    || /\b(prompt|rollout|batch|request|rpc)\b.*\b(to|->)\b.*\b(worker|server|rank|node)\b/.test(lowerLine);
  const looksLikeResponse =
    /got\s+\d+\/\d+\s+non-empty\s+completion/i.test(lowerLine)
    || /all workers done\.\s*\d+\s+non-empty/i.test(lowerLine)
    || /received\s+\d+\s+usable/i.test(lowerLine)
    || /\b(received|recv|returned|response|reply)\b.*\b(result|completion|rollout|batch|token|output)\b/.test(lowerLine)
    || /\b(result|completion|rollout|batch|output)\b.*\b(from|<-|back)\b.*\b(worker|server|rank|node)\b/.test(lowerLine);
  if (looksLikeRequest)  _markTrainIoRequestEvent();
  if (looksLikeResponse) _markTrainIoResponseEvent();
}

/**
 * Parse training metrics from a log line, updating trainingFallbackMetrics
 * and calling renderMetrics() when new values are found.
 */
function _parseTrainingLogLine(line) {
  const lowerLine = String(line || '').toLowerCase();
  if (!/step|loss|tok\/s|gradient norm|grad_norm|\blr\b|it\/s|eta/.test(lowerLine)) return;
  let dirty = false;

  const stepTotalMatch = line.match(/\[step\s+(\d+)\s*\/\s*(\d+)\]/i) || line.match(/step:\s*(\d+)\s*\/\s*(\d+)/i);
  if (stepTotalMatch) {
    trainingFallbackMetrics.step = Number(stepTotalMatch[1]);
    trainingFallbackMetrics.total_steps = Number(stepTotalMatch[2]);
    dirty = true;
  } else {
    const stepOnlyMatch = line.match(/\[step\s+(\d+)\]/i) || line.match(/step[:=]\s*(\d+)/i);
    if (stepOnlyMatch) { trainingFallbackMetrics.step = Number(stepOnlyMatch[1]); dirty = true; }
  }

  const lossMatch = line.match(/(?:leader\s+loss|training\s+loss|worker\s+\d+\s+loss|last rank mixtral loss|loss)\s*[:=]\s*([0-9]*\.?[0-9]+)/i);
  if (lossMatch) { trainingFallbackMetrics.loss = Number(lossMatch[1]); dirty = true; }

  const gradMatch = line.match(/gradient norm(?: before clipping)?\s*[:=]\s*([0-9]*\.?[0-9]+)/i)
                 || line.match(/\bgrad_norm[:=]\s*([0-9]*\.?[0-9]+)/i);
  if (gradMatch) { trainingFallbackMetrics.grad_norm = Number(gradMatch[1]); dirty = true; }

  const tputInMatch =
    line.match(/tok\/(?:sec|s)\s*\(in\)\s*[:=]?\s*([0-9]*\.?[0-9]+)/i)
    || line.match(/\bin(?:put)?\s+tok\/(?:sec|s)\s*[:=]?\s*([0-9]*\.?[0-9]+)/i);
  if (tputInMatch) { trainingFallbackMetrics.tok_sec_in = Number(tputInMatch[1]); dirty = true; }

  const tputOutMatch =
    line.match(/tok\/(?:sec|s)\s*\(out\)\s*[:=]?\s*([0-9]*\.?[0-9]+)/i)
    || line.match(/\bout(?:put)?\s+tok\/(?:sec|s)\s*[:=]?\s*([0-9]*\.?[0-9]+)/i);
  if (tputOutMatch) { trainingFallbackMetrics.tok_sec_out = Number(tputOutMatch[1]); dirty = true; }

  const tputMatch = line.match(/tok\/(?:sec|s)[^0-9]*([0-9]*\.?[0-9]+)/i) || line.match(/([0-9]*\.?[0-9]+)\s*tok\/(?:sec|s)/i);
  if (tputMatch) {
    const _t = Number(tputMatch[1]);
    trainingFallbackMetrics.throughput = _t;
    if (trainingFallbackMetrics.tok_sec_in == null)  trainingFallbackMetrics.tok_sec_in  = _t;
    if (trainingFallbackMetrics.tok_sec_out == null) trainingFallbackMetrics.tok_sec_out = _t;
    dirty = true;
  }

  const tqdmEtaMatch = line.match(/\b\d+%\|[^|]*\|\s*\d+\/\d+\s*\[[^\]]*?<\s*([^,\]\s]+)\s*,/);
  if (tqdmEtaMatch) {
    const etaText = String(tqdmEtaMatch[1] || '').trim();
    if (etaText) { trainingFallbackMetrics.eta_tqdm = etaText; dirty = true; }
  }

  const etaMatch =
    line.match(/\beta(?:\s*remaining)?\s*[:=]\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)/i)
    || line.match(/\[[^\]]*?<\s*([^,\]]+)/);
  if (etaMatch) {
    const etaText = String(etaMatch[1]).trim().replace(/^<+/, '');
    if (etaText && !trainingFallbackMetrics.eta_tqdm) {
      trainingFallbackMetrics.eta_remaining = etaText;
      dirty = true;
    }
  }

  const lrMatch = line.match(/\bLR[:=]\s*([0-9eE+\-.]+)/i);
  if (lrMatch) {
    const lrVal = Number(lrMatch[1]);
    if (isFinite(lrVal)) { trainingFallbackMetrics.lr = lrVal; dirty = true; }
  }

  if (dirty) {
    trainingFallbackMetrics.algorithm ||= _activeAlgo || $('algo-sel').value;
    renderMetrics();
  }
}
