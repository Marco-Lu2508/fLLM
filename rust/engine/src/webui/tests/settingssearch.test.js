// Tests for the ranking of the settings search in settingssearch.js.
// Run from the repository root:  node rust/engine/src/webui/tests/settingssearch.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'settingssearch.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in settingssearch.js');
const S = new Function(`${src.slice(begin, end)}
return { searchWords, rankSettingsPages };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const pages = [
  { id: 'general', label: 'General', text: 'Version Welcome tour Keyboard shortcuts Delete all chats' },
  { id: 'appearance', label: 'Appearance', text: 'Theme dark or light. Accent color. Font size. Show generation speed' },
  { id: 'hardware', label: 'Hardware', text: 'Compute devices. System memory. Processor feature limit. Speed test' },
  { id: 'voice', label: 'Voice and alerts', text: 'Dictation Reading aloud Speed Pitch Sound' },
];

test('words are lower-cased and split on anything that is not a letter or digit', () => {
  assert.deepStrictEqual(S.searchWords('  Dark-Theme, v2 '), ['dark', 'theme', 'v2']);
  assert.deepStrictEqual(S.searchWords(''), []);
  assert.deepStrictEqual(S.searchWords(null), []);
  assert.deepStrictEqual(S.searchWords('Größe'), ['größe']);
});

test('an empty query keeps every page in its order', () => {
  assert.deepStrictEqual(S.rankSettingsPages('', pages).map((r) => r.id), ['general', 'appearance', 'hardware', 'voice']);
});

test('a page whose name starts with the word comes first', () => {
  assert.strictEqual(S.rankSettingsPages('hard', pages)[0].id, 'hardware');
});

test('a word in the text finds the page, a name hit beats a text hit', () => {
  const r = S.rankSettingsPages('speed', pages).map((x) => x.id);
  assert.deepStrictEqual(r.sort(), ['appearance', 'hardware', 'voice']);
  const named = S.rankSettingsPages('voice', pages.concat([{ id: 'x', label: 'Other', text: 'a voice setting' }]));
  assert.strictEqual(named[0].id, 'voice');
});

test('every word must match', () => {
  assert.deepStrictEqual(S.rankSettingsPages('speed test', pages).map((r) => r.id), ['hardware']);
  assert.deepStrictEqual(S.rankSettingsPages('speed banana', pages), []);
});

test('a word at the start of a word in the text counts more than one inside a word', () => {
  const p = [{ id: 'a', label: 'A', text: 'prefix ixfoo' }, { id: 'b', label: 'B', text: 'foo bar' }];
  assert.strictEqual(S.rankSettingsPages('foo', p)[0].id, 'b');
});

console.log(`${n} tests passed`);
