// ════════════════════════════════════════════════════════════════════════════
// Log parsing helpers — called from appendLog
// ════════════════════════════════════════════════════════════════════════════
let _renderMetricsTimer = null;
function _scheduleRenderMetrics() {
  clearTimeout(_renderMetricsTimer);
  _renderMetricsTimer = setTimeout(renderMetrics, 80);
}

/** Handle [SMOL_EVENT] JSON — the authoritative animation trigger. */
function _parseSmolEvent(line) {
  const m = line.match(/\[SMOL_EVENT\]\s*(\{.+\})/);
  if (!m) return;
  try { _handleSmolEvent(JSON.parse(m[1])); } catch (e) {}
}

/**
 * Parse training metrics from a log line, updating trainingFallbackMetrics
 * and calling renderMetrics() when new values are found.
 */
function _parseTrainingLogLine(line) {
  const lowerLine = String(line || '').toLowerCase();
  if (!/step|loss|tok\/s|gradient norm|grad_norm|\blr\b|it\/s|eta|elapsed/.test(lowerLine)) return;
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

  const tqdmElapsedMatch = line.match(/\b\d+%\|[^|]*\|\s*\d+\/\d+\s*\[\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)\s*</);
  if (tqdmElapsedMatch) {
    const elapsedText = String(tqdmElapsedMatch[1] || '').trim();
    if (elapsedText) { trainingFallbackMetrics.elapsed_tqdm = elapsedText; dirty = true; }
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

  const elapsedMatch =
    line.match(/\belapsed(?:\s*time)?\s*[:=]\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)/i)
    || line.match(/\[[^\]]*?\b(elapsed)\s*[=:]\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)[^\]]*\]/i);
  if (elapsedMatch) {
    const elapsedText = String(elapsedMatch[2] || elapsedMatch[1] || '').trim();
    if (elapsedText && !trainingFallbackMetrics.elapsed_tqdm) {
      trainingFallbackMetrics.elapsed = elapsedText;
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
    _scheduleRenderMetrics();
  }
}
