// ════════════════════════════════════════════════════════════════════════════
// Entry + Setup views
// ════════════════════════════════════════════════════════════════════════════
function backToEntry() {
  $('entry-view').classList.remove('hidden');
  $('setup-view').classList.add('hidden');
  $('dashboard-shell').classList.add('hidden');
  uiSave({ view: 'entry' });
}

function openSetup() {
  $('entry-view').classList.add('hidden');
  $('setup-view').classList.remove('hidden');
  $('dashboard-shell').classList.add('hidden');
  renderSetup();
  uiSave({ view: 'setup' });
}

function openDashboard() {
  $('entry-view').classList.add('hidden');
  $('setup-view').classList.add('hidden');
  $('dashboard-shell').classList.remove('hidden');
  uiSave({ view: 'dashboard' });
}

function setSetupTrack(track) {
  activeSetupTrack = track === 'jetson' ? 'jetson' : 'mac';
  activeSetupStep = 0;
  $('setup-tab-mac').classList.toggle('active', activeSetupTrack === 'mac');
  $('setup-tab-jetson').classList.toggle('active', activeSetupTrack === 'jetson');
  renderSetup();
  uiSave({ track: activeSetupTrack, step: activeSetupStep });
}

function renderSetup() {
  const steps = setupTracks[activeSetupTrack] || [];
  const list = $('setup-steps');
  list.innerHTML = steps.map((s, i) =>
    `<button class="setup-step ${i === activeSetupStep ? 'active' : ''}" onclick="jumpSetupStep(${i})">${i + 1}. ${s.title}</button>`
  ).join('');
  renderSetupStep();
}

function jumpSetupStep(i) {
  const steps = setupTracks[activeSetupTrack] || [];
  activeSetupStep = Math.max(0, Math.min(steps.length - 1, i));
  renderSetup();
  uiSave({ step: activeSetupStep });
}

function nextSetupStep() {
  jumpSetupStep(activeSetupStep + 1);
}

function prevSetupStep() {
  jumpSetupStep(activeSetupStep - 1);
}

function copyCmd(btn) {
  const txt = btn.previousElementSibling.textContent;
  navigator.clipboard.writeText(txt).then(() => {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1600);
  }).catch(() => {});
}

function renderSetupStep() {
  const steps = setupTracks[activeSetupTrack] || [];
  if (!steps.length) return;
  const s = steps[activeSetupStep];
  setText('setup-kicker', `${activeSetupTrack === 'mac' ? 'Mac Mini (Thunderbolt)' : 'Jetson / Home Router'} • Step ${activeSetupStep + 1}/${steps.length}`);
  setText('setup-title', s.title);
  setText('setup-copy', s.copy);
  setText('setup-command', s.command);
  const pct = Math.round(((activeSetupStep + 1) / steps.length) * 100);
  $('setup-progress-fill').style.width = `${pct}%`;
}
