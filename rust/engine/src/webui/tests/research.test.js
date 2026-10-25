// Tests for the pure logic of Research mode (plan parser, URL selection, note budgeting).
// Run from the repository root:  node rust/engine/src/webui/tests/research.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'research.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in research.js');
const L = new Function('RESEARCH_NOTE_SHARE', `${src.slice(begin, end)}
return { researchClean, parsePlan, parseReflection, urlKey, selectUrls, findDate, fitNotes, noteBudget, researchMaxTokens };`)(0.6);

const est = (t) => Math.ceil(String(t || '').length / 3.6);
let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

test('parsePlan reads strict JSON', () => {
  assert.deepStrictEqual(L.parsePlan('{"queries": ["rust async runtime", "tokio vs async-std"]}'), ['rust async runtime', 'tokio vs async-std']);
});
test('parsePlan reads JSON inside prose, fences and think blocks', () => {
  const t = '<think>hmm {"x":1}</think>Sure!\n```json\n{"queries":["a query","second query"]}\n```';
  assert.deepStrictEqual(L.parsePlan(t), ['a query', 'second query']);
  assert.deepStrictEqual(L.parsePlan('Here: ["one query", "two query"] done'), ['one query', 'two query']);
});
test('parsePlan falls back to list lines', () => {
  assert.deepStrictEqual(L.parsePlan('1. first search\n2) second search\n- "third search",'), ['first search', 'second search', 'third search']);
});
test('parsePlan dedupes, drops junk and caps the count', () => {
  assert.deepStrictEqual(L.parsePlan('{"queries":["Same","same","ok query","x", ""]}'), ['Same', 'ok query']);
  const many = JSON.stringify({ queries: Array.from({ length: 10 }, (_, i) => `query number ${i}`) });
  assert.strictEqual(L.parsePlan(many).length, 6);
  assert.deepStrictEqual(L.parsePlan(''), []);
});
test('parseReflection', () => {
  assert.deepStrictEqual(L.parseReflection('{"enough": true, "queries": ["ignored"]}'), { enough: true, queries: [] });
  assert.deepStrictEqual(L.parseReflection('{"enough": false, "queries": ["more about x", "and y"]}'), { enough: false, queries: ['more about x', 'and y'] });
  assert.deepStrictEqual(L.parseReflection('no idea'), { enough: true, queries: [] });
});
test('urlKey ignores www, query, fragment, trailing slash and index files', () => {
  const k = L.urlKey('https://www.Example.org/Docs/?a=1#x');
  assert.strictEqual(k, 'example.org/docs');
  assert.strictEqual(L.urlKey('http://example.org/docs/index.html'), k);
  assert.strictEqual(L.urlKey('ftp://example.org/'), null);
  assert.strictEqual(L.urlKey('not a url'), null);
});
test('selectUrls dedupes, skips junk, limits per host and prefers pages found twice', () => {
  const r = [
    [{ url: 'https://a.org/x', title: 'A' }, { url: 'https://www.facebook.com/p', title: 'F' }, { url: 'https://b.org/paper.pdf', title: 'P' }, { url: 'https://c.org/login', title: 'L' }],
    [{ url: 'https://d.org/y', title: 'D' }, { url: 'https://www.a.org/x/', title: 'A again' }, { url: 'https://a.org/2', title: 'A2' }, { url: 'https://a.org/3', title: 'A3' }],
  ];
  const out = L.selectUrls(r, { limit: 10 });
  assert.deepStrictEqual(out.map((x) => x.url), ['https://a.org/x', 'https://d.org/y', 'https://a.org/2']);
  assert.strictEqual(out[0].hits, 2);
});
test('selectUrls skips pages already read and honours the limit', () => {
  const seen = new Set(['a.org/x']);
  const out = L.selectUrls([[{ url: 'https://a.org/x' }, { url: 'https://b.org/1' }, { url: 'https://c.org/1' }]], { limit: 1, seen });
  assert.deepStrictEqual(out.map((x) => x.url), ['https://b.org/1']);
});
test('findDate', () => {
  assert.strictEqual(L.findDate('Posted 2025-03-14 by x'), '2025-03-14');
  assert.strictEqual(L.findDate('on 14 March 2025.'), '14 March 2025');
  assert.strictEqual(L.findDate('Mar 3, 2024 - news'), 'Mar 3, 2024');
  assert.strictEqual(L.findDate('no date'), '');
});
test('noteBudget keeps notes within 60 % of the context and asks for condensing otherwise', () => {
  const notes = Array.from({ length: 4 }, (_, i) => ({ n: i + 1, text: 'x'.repeat(1000) }));
  const big = L.noteBudget(notes, 8192, 300, est);
  assert.strictEqual(big.needsCondense, false);
  const small = L.noteBudget(notes, 2048, 300, est);
  assert.strictEqual(small.needsCondense, true);
  assert(small.limit <= 2048 * 0.6);
  assert(small.limit > 0);
  assert.strictEqual(L.noteBudget(notes, 1000, 900, est).limit, 0);
});
test('fitNotes cuts proportionally, then drops the last notes', () => {
  const notes = Array.from({ length: 4 }, (_, i) => ({ n: i + 1, text: `line one\n${'y'.repeat(2000)}` }));
  const total = (l) => l.reduce((s, x) => s + est(x.text) + 8, 0);
  const fit = L.fitNotes(notes, 800, est);
  assert(fit.cut && fit.dropped === 0);
  assert(total(fit.notes) <= 800);
  assert.strictEqual(fit.notes.length, 4);
  const tiny = L.fitNotes(notes, 150, est);
  assert(tiny.dropped > 0 && tiny.notes.length >= 1);
  assert.deepStrictEqual(L.fitNotes(notes, 100000, est), { notes, cut: false, dropped: 0 });
});
test('researchMaxTokens', () => {
  assert.strictEqual(L.researchMaxTokens(4096, 1000, 1800), 1800);
  assert.strictEqual(L.researchMaxTokens(4096, 3900, 1800), 0);
  assert.strictEqual(L.researchMaxTokens(4096, 3000, 1800), 4096 - 3000 - 32 - 82);
});
console.log(`${n} tests passed`);
