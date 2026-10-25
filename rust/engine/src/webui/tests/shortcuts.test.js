// Tests for the keyboard shortcut registry in boot.js (parsing, formatting, conflicts, rebinding).
// Run from the repository root:  node rust/engine/src/webui/tests/shortcuts.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'boot.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in boot.js');
const S = new Function(`${src.slice(begin, end)}
return { SHORTCUTS, FIXED_KEYS, parseCombo, normCombo, comboFromEvent, formatCombo, effectiveCombos, checkCombo, shortcutFor };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };
const ev = (key, o = {}) => ({ key, code: o.code || '', ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, ...o });

test('the registry has unique ids and unique default keys, none of them fixed', () => {
  const ids = S.SHORTCUTS.map((s) => s.id);
  assert.strictEqual(new Set(ids).size, ids.length);
  const keys = S.SHORTCUTS.map((s) => S.normCombo(s.keys));
  assert.strictEqual(new Set(keys).size, keys.length);
  for (const f of S.FIXED_KEYS) assert(!keys.includes(S.normCombo(f.keys)), f.keys);
});

test('normCombo normalises order and case', () => {
  assert.strictEqual(S.normCombo('Shift+Mod+k'), 'Mod+Shift+K');
  assert.strictEqual(S.normCombo('Alt+Mod+x'), 'Mod+Alt+X');
  assert.strictEqual(S.normCombo('Mod+,'), 'Mod+,');
  assert.strictEqual(S.normCombo('Mod++'), 'Mod++');
  assert.strictEqual(S.normCombo('?'), '?');
  assert.strictEqual(S.normCombo(''), '');
  assert.strictEqual(S.parseCombo('Mod+Shift+Escape').key, 'Escape');
});

test('comboFromEvent: Ctrl on Windows and Linux, Cmd on macOS', () => {
  assert.strictEqual(S.comboFromEvent(ev('k', { ctrlKey: true }), false), 'Mod+K');
  assert.strictEqual(S.comboFromEvent(ev('k', { metaKey: true }), true), 'Mod+K');
  assert.strictEqual(S.comboFromEvent(ev('k', { ctrlKey: true }), true), 'Ctrl+K');
  assert.strictEqual(S.comboFromEvent(ev('K', { ctrlKey: true, shiftKey: true }), false), 'Mod+Shift+K');
  assert.strictEqual(S.comboFromEvent(ev('/', { ctrlKey: true }), false), 'Mod+/');
});

test('comboFromEvent: symbols keep no Shift, Alt uses the physical key, modifiers alone give nothing', () => {
  assert.strictEqual(S.comboFromEvent(ev('?', { shiftKey: true }), false), '?');
  assert.strictEqual(S.comboFromEvent(ev('Control', { ctrlKey: true }), false), '');
  assert.strictEqual(S.comboFromEvent(ev('å', { altKey: true, code: 'KeyA' }), true), 'Alt+A');
  assert.strictEqual(S.comboFromEvent(ev('¡', { altKey: true, code: 'Digit1' }), true), 'Alt+1');
  assert.strictEqual(S.comboFromEvent(ev('Escape', { shiftKey: true }), false), 'Shift+Escape');
  assert.strictEqual(S.comboFromEvent(ev(' ', { ctrlKey: true }), false), 'Mod+Space');
});

test('formatCombo', () => {
  assert.strictEqual(S.formatCombo('Mod+Shift+K', false), 'Ctrl+Shift+K');
  assert.strictEqual(S.formatCombo('Mod+Shift+K', true), 'Cmd Shift K');
  assert.strictEqual(S.formatCombo('Alt+ArrowDown', true), 'Option Down');
  assert.strictEqual(S.formatCombo('Escape', false), 'Esc');
});

test('effectiveCombos applies valid overrides and ignores unknown ids', () => {
  const c = S.effectiveCombos({ search: 'mod+shift+p', nothing: 'Mod+Q', newChat: 5 });
  assert.strictEqual(c.search, 'Mod+Shift+P');
  assert.strictEqual(c.newChat, 'Mod+N');
  assert.strictEqual(c.nothing, undefined);
});

test('checkCombo refuses conflicts, fixed keys and keys that would break typing', () => {
  const c = S.effectiveCombos({});
  const r = S.checkCombo(c, 'newChat', 'Mod+K');
  assert(!r.ok && /Search/.test(r.error), r.error);
  assert(S.checkCombo(c, 'search', 'Mod+K').ok); // its own key is fine
  assert(!S.checkCombo(c, 'search', 'Enter').ok);
  assert(!S.checkCombo(c, 'search', 'Shift+Enter').ok);
  assert(!S.checkCombo(c, 'search', 'A').ok);
  assert(!S.checkCombo(c, 'search', 'Shift+A').ok);
  assert(!S.checkCombo(c, 'search', '?').ok);
  assert(!S.checkCombo(c, 'search', '').ok);
  assert(S.checkCombo(c, 'search', 'F2').ok);
  assert(S.checkCombo(c, 'search', 'Alt+P').ok);
});

test('rebinding frees the old key and takes the new one', () => {
  const c = S.effectiveCombos({ search: 'Mod+Shift+P' });
  assert.strictEqual(S.shortcutFor(c, 'Mod+K'), '');
  assert.strictEqual(S.shortcutFor(c, 'Mod+Shift+P'), 'search');
  assert(S.checkCombo(c, 'newChat', 'Mod+K').ok); // the old key of search is free now
  assert(!S.checkCombo(c, 'newChat', 'Mod+Shift+P').ok);
});

console.log(`${n} tests passed`);
