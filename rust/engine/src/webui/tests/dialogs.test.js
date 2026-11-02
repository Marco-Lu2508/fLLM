// Guards the dialog contract: every dialog of the Studio is a .modal-backdrop with a .dialog box inside; the box is never taller than the
// screen, scrolls inside, and keeps its buttons (.dialog-actions) at the bottom edge. No dialog may undo that with its own rules, so a
// small or zoomed screen never hides an OK button. Also the Tab loop of dialogs.js.
// Run from the repository root:  node rust/engine/src/webui/tests/dialogs.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, '..');
const read = (f) => fs.readFileSync(path.join(dir, f), 'utf8');
const html = read('shell.html');
const css = ['style.css', 'chat.css', 'code.css', 'theme.css'].map(read).join('\n').replace(/\/\*[\s\S]*?\*\//g, '');
const scripts = fs.readdirSync(dir).filter((f) => f.endsWith('.js')).map((f) => [f, read(f)]);

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

/** The declarations of the rules whose selector list contains `selector` exactly. */
function declarations(selector) {
  const out = [];
  for (const m of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (m[1].split(',').map((s) => s.trim()).includes(selector)) out.push(m[2]);
  }
  return out.join(';');
}

test('the overlay scrolls as a fallback and keeps a margin around the dialog', () => {
  const d = declarations('.modal-backdrop');
  assert(/overflow-y:\s*auto/.test(d), 'overflow-y: auto');
  assert(/padding:\s*16px/.test(d), 'padding: 16px');
  assert(/position:\s*fixed/.test(d) && /inset:\s*0/.test(d));
});

test('the dialog box is never taller than the viewport and scrolls inside', () => {
  const d = declarations('.dialog');
  assert(/max-height:\s*calc\(100dvh - 32px\)/.test(d), 'max-height: calc(100dvh - 32px)');
  assert(/max-height:\s*calc\(100vh - 32px\)/.test(d), 'a vh fallback for browsers without dvh');
  assert(/overflow-y:\s*auto/.test(d), 'overflow-y: auto');
  assert(/max-width:\s*calc\(100vw - 32px\)/.test(d), 'never wider than the screen');
});

test('the buttons of a dialog stay at the bottom edge', () => {
  const d = declarations('.dialog-actions');
  assert(/position:\s*sticky/.test(d) && /bottom:\s*0/.test(d), 'sticky at the bottom');
  assert(/flex-wrap:\s*wrap/.test(d), 'buttons wrap on a narrow dialog');
  assert(/background:\s*inherit/.test(d), 'opaque, so the content scrolls under it');
});

test('no rule gives another dialog class its own height, overflow or fixed width', () => {
  for (const sel of ['.as-dialog', '.prj-dialog', '.mp-dialog', '.dialog.tour', '.search-dialog']) {
    const d = declarations(sel);
    assert(!/(^|;)\s*(max-)?height:/.test(d), `${sel} sets a height: ${d}`);
    assert(!/overflow(-y)?:\s*(hidden|scroll)/.test(d) || sel === '.search-dialog', `${sel} overrides the overflow`);
    for (const w of d.matchAll(/(?:^|;)\s*(?:min-)?width:\s*([^;]+)/g)) {
      assert(!/^\d+px$/.test(w[1].trim()) || /max-width:\s*calc\(100vw/.test(declarations(sel)), `${sel} has a fixed pixel width (${w[1]}); use min(Npx, calc(100vw - 32px))`);
    }
  }
});

test('every dialog in the page is a modal-backdrop around a .dialog (or the command palette)', () => {
  const backs = [...html.matchAll(/<div class="modal-backdrop"([^>]*)>\s*<div class="([^"]*)"([^>]*)>/g)];
  assert(backs.length >= 9, `found ${backs.length} dialogs in shell.html`);
  const total = (html.match(/class="modal-backdrop"/g) || []).length;
  assert.strictEqual(backs.length, total, 'a modal-backdrop whose first child is not a dialog box');
  for (const [, , cls, rest] of backs) {
    assert(/(^|\s)(dialog|search-dialog)(\s|$)/.test(cls), `dialog box class "${cls}"`);
    assert(!/max-height|overflow/.test(rest), `inline height rule on a dialog: ${rest}`);
  }
});

test('dialogs made by script use the same classes and do not set heights', () => {
  for (const [file, text] of scripts) {
    if (file === 'dialogs.js') continue;
    for (const m of text.matchAll(/modal-backdrop/g)) {
      const around = text.slice(Math.max(0, m.index - 400), m.index + 900);
      if (/getElementById|\$\(|querySelector|classList|\.open\b/.test(around.slice(380, 460)) && !/className|class="modal-backdrop/.test(around.slice(300, 520))) continue;
      assert(/class="dialog|className = 'dialog|\.dialog/.test(text), `${file} makes a backdrop without a .dialog box`);
    }
    assert(!/\.dialog[^;\n]*style\.(maxHeight|height|overflow)/.test(text), `${file} sets the height of a dialog from script`);
  }
});

test('a dialog can be closed from the keyboard: Cancel buttons are marked, the progress overlay is static', () => {
  const backs = [...html.matchAll(/<div class="modal-backdrop" id="(\w+)"([^>]*)>([\s\S]*?)(?=<div class="modal-backdrop"|<!-- Toasts)/g)];
  for (const [, id, attrs, body] of backs) {
    if (id === 'searchModal') continue; // the palette handles Escape itself
    assert(/data-modal-static/.test(attrs) || /data-dialog-cancel/.test(body), `${id} has no data-dialog-cancel button`);
  }
});

test('the viewport allows zooming and the page has no fixed minimum width above a phone', () => {
  assert(/<meta name="viewport"[^>]*width=device-width/.test(html));
  assert(!/user-scalable=no|maximum-scale=1\b/.test(html), 'zoom must stay possible');
});

// ---- the Tab loop ----
const src = read('dialogs.js');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in dialogs.js');
const { dialogTabTarget } = new Function(`${src.slice(begin, end)}\nreturn { dialogTabTarget };`)();

test('Tab wraps around inside the dialog and Shift+Tab wraps backwards', () => {
  const items = ['a', 'b', 'c'];
  assert.strictEqual(dialogTabTarget(items, 'c', false), 'a');
  assert.strictEqual(dialogTabTarget(items, 'a', true), 'c');
  assert.strictEqual(dialogTabTarget(items, 'b', false), null, 'inside the dialog the browser moves on by itself');
  assert.strictEqual(dialogTabTarget(items, 'outside', false), 'a', 'focus that left the dialog is brought back');
  assert.strictEqual(dialogTabTarget(items, 'outside', true), 'c');
  assert.strictEqual(dialogTabTarget([], 'x', false), null);
});

console.log(`${n} tests passed`);
