// Guards the one button system (theme.css, "one control system"): only the defined variants are used anywhere in the page,
// and the system block draws no glow (no accent box-shadow, no brightness filter, no blur).
// Run from the repository root:  node rust/engine/src/webui/tests/buttons.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, '..');
const theme = fs.readFileSync(path.join(dir, 'theme.css'), 'utf8');
const sources = fs.readdirSync(dir).filter((f) => /\.(js|html)$/.test(f)).map((f) => [f, fs.readFileSync(path.join(dir, f), 'utf8')]);

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const start = theme.indexOf('one control system');
assert(start > 0, 'the control system block exists in theme.css');
const block = theme.slice(start);

test('every btn-* class in the page is a variant or size of the system', () => {
  const known = new Set(['btn-primary', 'btn-secondary', 'btn-outline', 'btn-ghost', 'btn-destructive', 'btn-danger-text', 'btn-sm', 'btn-xs', 'btn-icon', 'btn-icon-xs']);
  const unknown = new Set();
  for (const [file, text] of sources) {
    if (file.startsWith('tests')) continue;
    for (const m of text.matchAll(/class="([^"]*)"|'(btn[^']*)'/g)) {
      for (const c of (m[1] || m[2] || '').split(/\s+/)) if (/^btn-/.test(c) && !c.includes('$') && !known.has(c)) unknown.add(`${c} in ${file}`);
    }
  }
  assert.deepStrictEqual([...unknown], []);
});

test('the system draws no glow: no brightness filter, no blurred or accent-coloured shadow on a button', () => {
  const rules = block.split('}').map((r) => r.trim()).filter((r) => /\.btn|send-circle|\.cs-btn|ui-switch/.test(r.split('{')[0]));
  for (const rule of rules) {
    const [sel, body = ''] = rule.split('{');
    if (/focus-visible/.test(sel)) continue; // the focus ring is the one allowed outline
    assert(!/filter:\s*(?!none)[a-z]/.test(body), `a filter in: ${sel}`);
    assert(!/box-shadow:\s*[^;]*(var\(--primary|rgba?\([^)]*\)\s+0\s+0\s+\d+px)/.test(body), `a glow in: ${sel}`);
  }
});

test('buttons have one size ladder and one radius', () => {
  assert.match(block, /--btn-h:\s*34px/);
  assert.match(block, /--btn-h-sm:\s*30px/);
  assert.match(block, /--btn-h-xs:\s*26px/);
  assert.match(block, /\.btn-xs \{[^}]*border-radius: var\(--btn-r\)/);
  assert.match(block, /\.btn-sm \{[^}]*border-radius: var\(--btn-r\)/);
});

test('a visible focus ring is defined for buttons, icon buttons and switches', () => {
  assert.match(block, /\.btn:focus-visible[\s\S]*?outline: 2px solid var\(--ring\)/);
  assert.match(block, /ui-switch input:focus-visible \+ \.ui-switch-track/);
});

console.log(`${n} tests passed`);
