// Tests for the ranking and keyboard stepping of the command palette in palette.js.
// Run from the repository root:  node rust/engine/src/webui/tests/palette.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'palette.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in palette.js');
const P = new Function(`${src.slice(begin, end)}
return { paletteScore, rankPalette, paletteStep };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const items = [
  { id: 'new', kind: 'Command', title: 'New chat', keywords: 'start conversation' },
  { id: 'models', kind: 'Command', title: 'Open Models', keywords: 'download hugging face' },
  { id: 'theme', kind: 'Command', title: 'Switch to the dark theme', keywords: 'appearance colors' },
  { id: 'shortcuts', kind: 'Command', title: 'Keyboard shortcuts' },
  { id: 'c1', kind: 'Chat', title: 'Rust dot product' },
  { id: 'c2', kind: 'Chat', title: 'Weekly plan', sub: 'the new schedule for the chat room' },
  { id: 's1', kind: 'Setting', title: 'Settings: Hardware' },
];
const ids = (q) => P.rankPalette(q, items).map((i) => i.id);

test('score: exact and prefix beat word start beat inside a word beat letters in order', () => {
  const s = (t, q) => P.paletteScore(t, q);
  assert(s('chat', 'chat') > s('chat room', 'chat'));
  assert(s('chat room', 'chat') > s('new chat', 'chat'));
  assert(s('new chat', 'chat') > s('whatchat', 'chat'));
  assert(s('whatchat', 'chat') > s('keyboard shortcuts', 'ksc'));
  assert.strictEqual(s('new chat', 'zzz'), 0);
});

test('score: every word of the query must match', () => {
  assert(P.paletteScore('Switch to the dark theme', 'dark theme') > 0);
  assert.strictEqual(P.paletteScore('Switch to the dark theme', 'dark banana'), 0);
});

test('score: a very short query does not match scattered letters', () => {
  assert.strictEqual(P.paletteScore('keyboard shortcuts', 'ks'), 0);
  assert(P.paletteScore('keyboard shortcuts', 'kbd') > 0);
});

test('score: regex characters in the query are taken literally', () => {
  assert.strictEqual(P.paletteScore('c++ notes', 'c++') > 0, true);
  assert.doesNotThrow(() => P.paletteScore('anything', '(['));
});

test('no query: the commands first, then the chats in the given order', () => {
  assert.deepStrictEqual(ids(''), ['new', 'models', 'theme', 'shortcuts', 'c1', 'c2']);
});

test('a title match comes before a keyword match and a text match', () => {
  assert.strictEqual(ids('chat')[0], 'new');
  assert(ids('chat').includes('c2'));
  assert(ids('chat').indexOf('new') < ids('chat').indexOf('c2'));
});

test('keywords find a command that the title does not name', () => {
  assert.deepStrictEqual(ids('hugging'), ['models']);
  assert.deepStrictEqual(ids('colors'), ['theme']);
});

test('nothing matches nonsense', () => {
  assert.deepStrictEqual(ids('qqqq'), []);
});

test('the limit is respected', () => {
  const many = Array.from({ length: 100 }, (_, i) => ({ id: `x${i}`, kind: 'Chat', title: `chat ${i}` }));
  assert.strictEqual(P.rankPalette('chat', many, 10).length, 10);
});

test('letters scattered over a long title do not match', () => {
  assert.strictEqual(P.paletteScore('Export this chat as Markdown', 'hard'), 0);
  assert(P.paletteScore('Keyboard shortcuts', 'kbd') > 0);
});

test('results are grouped by kind, the group with the best match first, a weak entry last in its group', () => {
  const set = [
    { id: 'a', kind: 'Command', title: 'Open Models' },
    { id: 'b', kind: 'Setting', title: 'Settings: Models' },
    { id: 'c', kind: 'Setting', title: 'Settings: Model details' },
    { id: 'd', kind: 'Chat', title: 'about models' },
    { id: 'w', kind: 'Setting', title: 'Search the settings for "model"', weak: true },
  ];
  const r = P.rankPalette('model', set).map((i) => i.id);
  const kinds = P.rankPalette('model', set).map((i) => i.kind);
  assert.strictEqual(r[r.length - 1], 'w');
  assert.deepStrictEqual(kinds.filter((k, i) => i === 0 || k !== kinds[i - 1]).length, new Set(kinds).size, 'each kind forms one block');
});

test('arrow keys wrap around and Home and End jump', () => {
  assert.strictEqual(P.paletteStep(-1, 5, 'ArrowDown'), 0);
  assert.strictEqual(P.paletteStep(4, 5, 'ArrowDown'), 0);
  assert.strictEqual(P.paletteStep(0, 5, 'ArrowUp'), 4);
  assert.strictEqual(P.paletteStep(2, 5, 'ArrowUp'), 1);
  assert.strictEqual(P.paletteStep(2, 5, 'End'), 4);
  assert.strictEqual(P.paletteStep(2, 5, 'Home'), 0);
  assert.strictEqual(P.paletteStep(0, 0, 'ArrowDown'), -1);
});

console.log(`${n} tests passed`);
