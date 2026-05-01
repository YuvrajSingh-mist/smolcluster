// ════════════════════════════════════════════════════════════════════════════
// Global state + constants
// ════════════════════════════════════════════════════════════════════════════
let state = {
  discovered:{}, selected:{}, running:{}, usernames:{}, ssh_aliases:{}, training:{}, connectivity:{}, redis:{}
};
const sshOverrides = {};
let _prevNSig='', _prevTSig='', _prevCSig='', _prevRSig='';
let dashboardMode = 'train';
let bottomTab = 'logs';
let ssEventSource = null;  // Global state event source
let logsEventSource = null; // Global logs event source
let generationAbortController = null;
let generationInFlight = false;
let _genStartTime = null, _genTokenCount = 0;
let trainingFallbackMetrics = {};

const TRAIN_ALGOS = [
  { value: 'syncps', label: 'SyncPS — Sync Parameter Server' },
  { value: 'mp', label: 'MP — Model Parallelism' },
  { value: 'classicdp', label: 'ClassicDP — Classic Data Parallelism' },
  { value: 'fsdp', label: 'FSDP — Fully Sharded Data Parallel' },
  { value: 'ep', label: 'EP — Expert Parallelism' },
  { value: 'mp_pipeline', label: 'MP Pipeline — Model Parallelism Pipeline' },
  { value: 'edp', label: 'EDP — Elastic Data Parallelism' },
  { value: 'grpo', label: 'GRPO — Group Relative Policy Optimization' },
];

const INFER_ALGOS = [
  { value: 'syncps', label: 'SyncPS — Sync Parameter Server' },
  { value: 'mp', label: 'MP — Model Parallelism' },
  { value: 'classicdp', label: 'ClassicDP — Classic Data Parallelism' },
];

const GENERATION_PRESETS = [
  { label: 'SyncPS explainer', text: 'Explain parameter server architecture in SyncPS for this cluster in simple terms.' },
  { label: 'Worker role', text: 'What does worker rank 1 do during inference and how does it communicate with the server?' },
  { label: 'Bottlenecks', text: 'List likely bottlenecks in this distributed inference setup and suggest concrete optimizations.' },
  { label: 'Debug checklist', text: 'Give me a practical debugging checklist when token streaming stalls in distributed inference.' },
];

const lines = (...parts) => parts.join('\n');
