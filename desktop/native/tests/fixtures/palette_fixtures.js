// The fixtures of the command palette are produced by palette.js itself: the
// @pure-begin..@pure-end block of the Studio is run in a sandbox and every function of it is asked
// the cases the Rust tests ask. Run: node tests/fixtures/palette_fixtures.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const src = fs.readFileSync(
  path.join(__dirname, '../../../../rust/engine/src/webui/palette.js'),
  'utf8',
);
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('the pure block of palette.js was not found');
const zone = src.slice(src.indexOf('\n', begin) + 1, src.lastIndexOf('\n', end));

const api = vm.runInNewContext(
  `${zone}\n;({ paletteScore, rankPalette, paletteStep })`,
  {},
  { timeout: 5000 },
);
const { paletteScore, rankPalette, paletteStep } = api;

const out = {};

// paletteScore: 0 = no match, and the grades in between
out.score = [
  // empty query keeps everything
  ['anything', '', paletteScore('anything', '')],
  ['anything', '  ', paletteScore('anything', '  ')],
  // exact, prefix, word start, boundary, inside
  ['keyboard', 'keyboard', paletteScore('keyboard', 'keyboard')],
  ['Keyboard shortcuts', 'keyboard', paletteScore('Keyboard shortcuts', 'keyboard')],
  ['keyboard shortcuts', 'board', paletteScore('keyboard shortcuts', 'board')],
  ['keyboard shortcuts', 'short', paletteScore('keyboard shortcuts', 'short')],
  ['the dark theme', 'dark', paletteScore('the dark theme', 'dark')],
  ['go to chat', 'to', paletteScore('go to chat', 'to')],
  ['go to chat', 'o t', paletteScore('go to chat', 'o t')],
  // case and spaces around the query
  ['Go to Chat', '  GO  ', paletteScore('Go to Chat', '  GO  ')],
  // letters in order at a word start, close together: kbd finds keyboard
  ['keyboard shortcuts', 'kbd', paletteScore('keyboard shortcuts', 'kbd')],
  // scattered letters do not
  ['keyboard shortcuts', 'kbs', paletteScore('keyboard shortcuts', 'kbs')],
  ['keyboard shortcuts', 'zqx', paletteScore('keyboard shortcuts', 'zqx')],
  // a word must be there
  ['go to chat', 'models', paletteScore('go to chat', 'models')],
  // word boundary: a hit inside another word is not a boundary hit
  ['dark mode', 'ark', paletteScore('dark mode', 'ark')],
  ['sand art', 'art', paletteScore('sand art', 'art')],
];

// rankPalette: without a query the first commands then the chats, with one the best first in
// blocks per kind
const commands = [
  { id: 'new', kind: 'Command', title: 'New chat', keywords: 'start conversation' },
  { id: 'chat', kind: 'Command', title: 'Go to Chat' },
  { id: 'code', kind: 'Command', title: 'Go to Code', keywords: 'coding agent folder' },
  { id: 'models', kind: 'Command', title: 'Open Models', keywords: 'download hugging face gguf catalog load' },
  { id: 'settings', kind: 'Command', title: 'Open Settings' },
  { id: 'theme', kind: 'Command', title: 'Switch to the dark theme', keywords: 'appearance colors mode' },
  { id: 'shortcuts', kind: 'Command', title: 'Keyboard shortcuts', keywords: 'keys hotkeys' },
  { id: 'tour', kind: 'Command', title: 'Show the welcome tour' },
  { id: 'export-md', kind: 'Command', title: 'Export this chat as Markdown', keywords: 'save download' },
];
const chats = [
  { id: 'chat-1', kind: 'Chat', title: 'Gold price today' },
  { id: 'chat-2', kind: 'Chat', title: 'Keyboard layout' },
  { id: 'chat-3', kind: 'Chat', title: 'Docker networks' },
];
const settings = [
  { id: 'pane-sampling', kind: 'Setting', title: 'Settings: Sampling' },
  { id: 'pane-general', kind: 'Setting', title: 'Settings: General' },
];
const all = [...commands, ...chats, ...settings];

const ids = (ranked) => ranked.map((i) => i.id);
out.rank = [
  // no query: the first seven commands, the chats after them
  ['', all, 40, ids(rankPalette('', all, 40))],
  // a small limit cuts both blocks
  ['', all, 3, ids(rankPalette('', all, 3))],
  // a title hit beats a keyword hit, a keyword hit beats a sub hit
  ['chat', all, 40, ids(rankPalette('chat', all, 40))],
  ['keyboard', all, 40, ids(rankPalette('keyboard', all, 40))],
  ['folder', all, 40, ids(rankPalette('folder', all, 40))],
  ['dark', all, 40, ids(rankPalette('dark', all, 40))],
  ['gguf', all, 40, ids(rankPalette('gguf', all, 40))],
  ['docker', all, 40, ids(rankPalette('docker', all, 40))],
  // blocks: the kind with the best match comes first
  ['settings', all, 40, ids(rankPalette('settings', all, 40))],
  ['export', all, 40, ids(rankPalette('export', all, 40))],
  // nothing matches
  ['qqq', all, 40, ids(rankPalette('qqq', all, 40))],
  // a weak row stays last in its block
  [
    'docker',
    [...all, { id: 'find-setting', kind: 'Setting', weak: true, title: 'Search the settings for "docker"' }],
    41,
    ids(
      rankPalette(
        'docker',
        [...all, { id: 'find-setting', kind: 'Setting', weak: true, title: 'Search the settings for "docker"' }],
        41,
      ),
    ),
  ],
];

// paletteStep: the arrow keys wrap around, Home and End jump
out.step = [
  [-1, 5, 'ArrowDown', paletteStep(-1, 5, 'ArrowDown')],
  [0, 5, 'ArrowDown', paletteStep(0, 5, 'ArrowDown')],
  [4, 5, 'ArrowDown', paletteStep(4, 5, 'ArrowDown')],
  [0, 5, 'ArrowUp', paletteStep(0, 5, 'ArrowUp')],
  [3, 5, 'ArrowUp', paletteStep(3, 5, 'ArrowUp')],
  [2, 5, 'Home', paletteStep(2, 5, 'Home')],
  [2, 5, 'End', paletteStep(2, 5, 'End')],
  [2, 5, 'ArrowLeft', paletteStep(2, 5, 'ArrowLeft')],
  [2, 0, 'ArrowDown', paletteStep(2, 0, 'ArrowDown')],
];

const target = path.join(__dirname, 'palette_fixtures.json');
fs.writeFileSync(target, JSON.stringify(out, null, 1) + '\n');
console.log(
  `palette_fixtures.json written: score=${out.score.length} rank=${out.rank.length} step=${out.step.length}`,
);
