// The fixtures of the keyboard shortcuts are produced by boot.js itself: the
// @pure-begin..@pure-end block of the Studio is run in a sandbox and every function of it is
// asked the cases the Rust tests ask. Run: node tests/fixtures/shortcuts_fixtures.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const src = fs.readFileSync(
  path.join(__dirname, '../../../../rust/engine/src/webui/boot.js'),
  'utf8',
);
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('the pure block of boot.js was not found');
const zone = src.slice(src.indexOf('\n', begin) + 1, src.lastIndexOf('\n', end));

const scope = {};
const api = vm.runInNewContext(
  `${zone}\n;({ parseCombo, normCombo, comboFromEvent, formatCombo, effectiveCombos, checkCombo, shortcutFor })`,
  scope,
  { timeout: 5000 },
);
const {
  parseCombo,
  normCombo,
  comboFromEvent,
  formatCombo,
  effectiveCombos,
  checkCombo,
  shortcutFor,
} = api;

const out = {};

// parseCombo: the modifiers and the key of a text, or null
out.parse = [
  ['Mod+N', parseCombo('Mod+N')],
  ['mod+n', parseCombo('mod+n')],
  ['Escape', parseCombo('Escape')],
  ['', parseCombo('')],
  ['   ', parseCombo('   ')],
  ['x+y', parseCombo('x+y')],
  ['Ctrl+Shift+P', parseCombo('Ctrl+Shift+P')],
  ['Shift+Mod+P', parseCombo('Shift+Mod+P')],
  ['Mod+Mod+N', parseCombo('Mod+Mod+N')],
  ['Mod+/', parseCombo('Mod+/')],
  ['Mod+,', parseCombo('Mod+,')],
  ['Alt+ArrowDown', parseCombo('Alt+ArrowDown')],
];

// normCombo: the normal form, '' without a combination
out.norm = [
  ['Mod+N', normCombo('Mod+N')],
  ['mod+n', normCombo('mod+n')],
  ['Shift+Mod+P', normCombo('Shift+Mod+P')],
  ['n', normCombo('n')],
  ['Escape', normCombo('Escape')],
  ['', normCombo('')],
  ['x+y', normCombo('x+y')],
];

// formatCombo: how a combination is shown — off macOS and on it
out.format = [
  ['Mod+N', formatCombo('Mod+N', false), formatCombo('Mod+N', true)],
  ['Mod+/', formatCombo('Mod+/', false), formatCombo('Mod+/', true)],
  ['Mod+Shift+P', formatCombo('Mod+Shift+P', false), formatCombo('Mod+Shift+P', true)],
  ['Escape', formatCombo('Escape', false), formatCombo('Escape', true)],
  ['ArrowUp', formatCombo('ArrowUp', false), formatCombo('ArrowUp', true)],
  ['Alt+ArrowDown', formatCombo('Alt+ArrowDown', false), formatCombo('Alt+ArrowDown', true)],
  ['Alt+Option+X', formatCombo('Alt+Option+X', false), formatCombo('Alt+Option+X', true)],
  ['', formatCombo('', false), formatCombo('', true)],
  ['not a combo', formatCombo('not a combo', false), formatCombo('not a combo', true)],
];

// effectiveCombos: the defaults with the saved changes on top
out.effective = [
  [{}, effectiveCombos({})],
  [{ newChat: 'Mod+J' }, effectiveCombos({ newChat: 'Mod+J' })],
  [{ newChat: 'mod+j' }, effectiveCombos({ newChat: 'mod+j' })],
  [{ unknown: 'Mod+Q' }, effectiveCombos({ unknown: 'Mod+Q' })],
  [{ newChat: 'Mod+J', stop: 'no' }, effectiveCombos({ newChat: 'Mod+J', stop: 'no' })],
];

// checkCombo: whether a combination may be given, and what the Studio says when not
const combos = effectiveCombos({});
const withChange = effectiveCombos({ newChat: 'Mod+J' });
out.check = [
  [combos, 'newChat', 'Mod+N', checkCombo(combos, 'newChat', 'Mod+N')],
  [combos, 'newChat', 'Mod+J', checkCombo(combos, 'newChat', 'Mod+J')],
  [combos, 'newChat', 'Q', checkCombo(combos, 'newChat', 'Q')],
  [combos, 'newChat', 'Shift+Q', checkCombo(combos, 'newChat', 'Shift+Q')],
  [combos, 'newChat', 'Enter', checkCombo(combos, 'newChat', 'Enter')],
  [combos, 'newChat', 'Mod+Enter', checkCombo(combos, 'newChat', 'Mod+Enter')],
  [combos, 'newChat', 'Escape', checkCombo(combos, 'newChat', 'Escape')],
  [combos, 'newChat', 'Shift+Escape', checkCombo(combos, 'newChat', 'Shift+Escape')],
  [combos, 'newChat', 'ArrowUp', checkCombo(combos, 'newChat', 'ArrowUp')],
  [combos, 'newChat', 'Mod+ArrowUp', checkCombo(combos, 'newChat', 'Mod+ArrowUp')],
  [combos, 'newChat', 'Mod+N', checkCombo(combos, 'newChat', 'Mod+N')],
  [combos, 'stop', 'Mod+N', checkCombo(combos, 'stop', 'Mod+N')],
  [combos, 'stop', 'Alt+1', checkCombo(combos, 'stop', 'Alt+1')],
  [combos, 'shortcuts', 'Mod+/', checkCombo(combos, 'shortcuts', 'Mod+/')],
  [combos, 'shortcuts', '?', checkCombo(combos, 'shortcuts', '?')],
  [combos, 'search', '', checkCombo(combos, 'search', '')],
  [withChange, 'newChat', 'Mod+N', checkCombo(withChange, 'newChat', 'Mod+N')],
  [combos, 'toggleSidebar', 'mod+b', checkCombo(combos, 'toggleSidebar', 'mod+b')],
];

// shortcutFor: the id a combination triggers, or ''
out.for = [
  [combos, 'Mod+N', shortcutFor(combos, 'Mod+N')],
  [combos, 'mod+n', shortcutFor(combos, 'mod+n')],
  [combos, 'Escape', shortcutFor(combos, 'Escape')],
  [combos, 'Alt+ArrowDown', shortcutFor(combos, 'Alt+ArrowDown')],
  [combos, 'Alt+9', shortcutFor(combos, 'Alt+9')],
  [withChange, 'Mod+J', shortcutFor(withChange, 'Mod+J')],
  [withChange, 'Mod+N', shortcutFor(withChange, 'Mod+N')],
];

// comboFromEvent: the combination of a keyboard event, off macOS and on it
out.fromevent = [
  [{ key: 'n', ctrlKey: true }, false],
  [{ key: 'n', ctrlKey: true }, true],
  [{ key: 'n', metaKey: true }, true],
  [{ key: 'Escape' }, false],
  [{ key: 'Shift' }, false],
  [{ key: 'Control' }, false],
  [{ key: ' ', ctrlKey: true }, false],
  [{ key: 'k', altKey: true, code: 'KeyK' }, false],
  [{ key: '5', altKey: true, code: 'Digit5' }, false],
  [{ key: 'ArrowDown', altKey: true }, false],
  [{ key: '?', shiftKey: true, ctrlKey: true }, false],
  [{ key: 'a', shiftKey: true }, false],
  [{ key: 'A', shiftKey: true }, false],
  [{ key: 'b', altKey: true, shiftKey: true }, false],
  [{ key: '', ctrlKey: true }, false],
  [{ key: 'Escape', shiftKey: true }, false],
].map(([event, isMac]) => [event, isMac, comboFromEvent(event, isMac)]);

const target = path.join(__dirname, 'shortcuts_fixtures.json');
fs.writeFileSync(target, JSON.stringify(out, null, 1) + '\n');
console.log(
  `shortcuts_fixtures.json written: parse=${out.parse.length} norm=${out.norm.length} format=${out.format.length} effective=${out.effective.length} check=${out.check.length} for=${out.for.length} fromevent=${out.fromevent.length}`,
);
