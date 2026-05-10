// Config side panel — opens on Train, loads the algo's config file
let _cfgRawMode = false;
let _cfgAllRows = [];

async function showConfig(algo) {
  const overlay = document.getElementById('cfg-overlay');
  const kvBody  = document.getElementById('cfg-kv-body');
  const rawBody = document.getElementById('cfg-raw-body');
  const title   = document.getElementById('cfg-title');
  const count   = document.getElementById('cfg-key-count');

  _cfgRawMode = false;
  document.getElementById('cfg-raw-btn').textContent = 'View raw data';
  kvBody.style.display  = '';
  rawBody.style.display = 'none';
  document.getElementById('cfg-search').value = '';
  kvBody.innerHTML = '<div class="cfg-loading">Loading…</div>';
  count.textContent = '';
  title.textContent = `Config · ${algo.toUpperCase()}`;
  overlay.classList.add('visible');

  try {
    const r = await fetch(`/api/config?algorithm=${encodeURIComponent(algo)}`);
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: 'Unknown error' }));
      kvBody.innerHTML = `<div class="cfg-loading cfg-err">Error: ${err.detail}</div>`;
      return;
    }
    const data = await r.json();
    rawBody.textContent = data.yaml;
    _cfgAllRows = parseYaml(data.yaml);
    const topLevel = _cfgAllRows.filter(r => r.depth === 0);
    count.textContent = `{} ${topLevel.length} keys`;
    renderCfgRows(_cfgAllRows);
  } catch (e) {
    kvBody.innerHTML = `<div class="cfg-loading cfg-err">Error: ${e.message}</div>`;
  }
}

function hideConfig() {
  document.getElementById('cfg-overlay').classList.remove('visible');
}

function toggleCfgRaw() {
  _cfgRawMode = !_cfgRawMode;
  document.getElementById('cfg-kv-body').style.display  = _cfgRawMode ? 'none' : '';
  document.getElementById('cfg-raw-body').style.display = _cfgRawMode ? '' : 'none';
  document.getElementById('cfg-raw-btn').textContent    = _cfgRawMode ? 'View structured' : 'View raw data';
}

function filterCfgKeys(query) {
  if (!query) { renderCfgRows(_cfgAllRows); return; }
  let re;
  try { re = new RegExp(query, 'i'); } catch { re = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i'); }
  const visible = _cfgAllRows.filter(row => {
    if (re.test(row.displayKey)) return true;
    if (row.type === 'section') {
      return _cfgAllRows.some(r => r.parent === row.key && re.test(r.displayKey));
    }
    return false;
  });
  renderCfgRows(visible);
}

function renderCfgRows(rows) {
  const body = document.getElementById('cfg-kv-body');
  if (!rows.length) { body.innerHTML = '<div class="cfg-loading">No matching keys.</div>'; return; }
  body.innerHTML = rows.map(row => {
    const indent = row.depth * 16;
    if (row.type === 'section') {
      const childCount = _cfgAllRows.filter(r => r.parent === row.key).length;
      return `<div class="cfg-row cfg-section" data-key="${esc(row.key)}" style="padding-left:${indent+12}px" onclick="toggleSection(this,'${esc(row.key)}')">
        <span class="cfg-chevron">▼</span>
        <span class="cfg-key">${esc(row.displayKey)}</span>
        <span class="cfg-val cfg-val-obj">{} ${childCount} keys</span>
      </div>`;
    }
    const v = colorVal(row.value, row.valType);
    return `<div class="cfg-row" data-parent="${esc(row.parent||'')}" style="padding-left:${indent+12}px">
      <span class="cfg-key">${esc(row.displayKey)}</span>
      <span class="cfg-val ${v.cls}">${v.text}</span>
    </div>`;
  }).join('');
}

function toggleSection(el, key) {
  const collapsed = el.classList.toggle('collapsed');
  el.querySelector('.cfg-chevron').textContent = collapsed ? '▶' : '▼';
  document.getElementById('cfg-kv-body').querySelectorAll('.cfg-row').forEach(row => {
    const parentKey = row.dataset.parent;
    const ownKey    = row.dataset.key;
    if (parentKey !== undefined) {
      // value / list-item row — hide if its parent is the toggled key or deeper
      if (parentKey === key || isUnder(parentKey, key))
        row.style.display = collapsed ? 'none' : '';
    } else if (ownKey && ownKey !== key) {
      // nested section row — hide if it lives under the toggled key
      if (isUnder(ownKey, key))
        row.style.display = collapsed ? 'none' : '';
    }
  });
}

function isUnder(childKey, ancestorKey) {
  let cur = _cfgAllRows.find(r => r.key === childKey);
  while (cur && cur.parent) {
    if (cur.parent === ancestorKey) return true;
    cur = _cfgAllRows.find(r => r.key === cur.parent);
  }
  return false;
}

function colorVal(val, type) {
  if (type === 'bool')   return { cls: 'cfg-val-bool', text: esc(val) };
  if (type === 'number') return { cls: 'cfg-val-num',  text: esc(val) };
  if (type === 'null')   return { cls: 'cfg-val-null', text: 'null' };
  return { cls: 'cfg-val-str', text: esc(val) };
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function parseYaml(yaml) {
  const lines = yaml.split('\n');
  const rows = [];
  const stack = [];
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    if (!raw.trim() || raw.trimStart().startsWith('#')) continue;
    const indent = raw.search(/\S/);
    const content = raw.trim();
    while (stack.length && stack[stack.length-1].indent >= indent) stack.pop();
    const depth = stack.length;
    const parentKey = stack.length ? stack[stack.length-1].key : null;
    if (content.startsWith('- ')) {
      const val = content.slice(2).trim();
      const idx = rows.filter(r => r.parent === parentKey && r.type === 'list-item').length;
      rows.push({ key: `${parentKey}[${idx}]`, displayKey: `[${idx}]`, value: val, valType: inferType(val), type: 'list-item', depth, parent: parentKey });
      continue;
    }
    const col = content.indexOf(':');
    if (col === -1) continue;
    const keyName = content.slice(0, col).trim();
    const rest = content.slice(col+1).trim().replace(/\s*#.*$/, '').trim();
    const fullKey = parentKey ? `${parentKey}.${keyName}` : keyName;
    let nextIndent = -1;
    for (let j = i+1; j < lines.length; j++) {
      const nl = lines[j];
      if (!nl.trim() || nl.trimStart().startsWith('#')) continue;
      nextIndent = nl.search(/\S/);
      break;
    }
    if (!rest && nextIndent > indent) {
      rows.push({ key: fullKey, displayKey: keyName, type: 'section', depth, parent: parentKey });
      stack.push({ key: fullKey, indent });
    } else {
      rows.push({ key: fullKey, displayKey: keyName, value: rest, valType: inferType(rest), type: 'value', depth, parent: parentKey });
    }
  }
  return rows;
}

function inferType(val) {
  if (val === 'true' || val === 'false') return 'bool';
  if (val === 'null' || val === '~' || val === '') return 'null';
  if (val !== '' && !isNaN(Number(val))) return 'number';
  return 'string';
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') hideConfig(); });
