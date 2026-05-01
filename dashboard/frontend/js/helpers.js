// ════════════════════════════════════════════════════════════════════════════
// Utility helpers
// ════════════════════════════════════════════════════════════════════════════
function $(id)      { return document.getElementById(id); }
function setText(id, v) { const e=$(id); if(e && e.textContent!==String(v)) e.textContent=v; }
function guessUser(h) { return ''; }
function escHtml(s)   { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function ansiToHtml(s) {
  const text = String(s || '');
  const parts = text.split(/(\x1b\[[0-9;]*m)/g);
  let bold = false;
  let fg = null;
  let out = '';

  const openSpan = () => {
    const classes = [];
    if (bold) classes.push('ansi-bold');
    if (fg !== null) classes.push(`ansi-fg-${fg}`);
    if (!classes.length) return '';
    return `<span class="${classes.join(' ')}">`;
  };

  for (const part of parts) {
    const m = part.match(/^\x1b\[([0-9;]*)m$/);
    if (m) {
      const codes = (m[1] || '0').split(';').filter(Boolean).map(n => Number(n));
      if (!codes.length) codes.push(0);
      for (const code of codes) {
        if (code === 0) {
          bold = false;
          fg = null;
        } else if (code === 1) {
          bold = true;
        } else if (code === 22) {
          bold = false;
        } else if ((code >= 30 && code <= 37) || (code >= 90 && code <= 97)) {
          fg = code;
        } else if (code === 39) {
          fg = null;
        }
      }
      continue;
    }

    if (!part) continue;
    const safe = escHtml(part);
    const opener = openSpan();
    out += opener ? `${opener}${safe}</span>` : safe;
  }

  return out;
}

function nodeIcon(n) {
  const os=((n.os||'').toLowerCase());
  const h=((n.hostname||'').toLowerCase());
  const a=((n.alias||'').toLowerCase());
  const any = h + ' ' + a;
  if(any.includes('ipad')) return '📱';
  if(any.includes('pi')||any.includes('rasp')) return '🍓';
  if(os==='darwin'||any.includes('mac')||any.includes('mini')) return '💻';
  if(os==='windows'||os==='win32'||any.includes('win')) return '🖥️';
  if(os==='linux'||os==='ubuntu'||os==='debian'||os==='fedora') return '🐧';
  if(any.includes('jetson')||any.includes('nano')||any.includes('xavier')) return '🐧';
  return '💻';
}
