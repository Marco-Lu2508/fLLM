// Tests for the document search in rag.js: the BM25 index, and the hybrid ranking (keyword plus vectors).
// Run from the repository root:  node rust/engine/src/webui/tests/rag.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, '..');
const src = fs.readFileSync(path.join(dir, 'rag.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in rag.js');
const R = new Function(`${src.slice(begin, end)}\nreturn { ragTokenize, ragChunk, ragBuildIndex, ragSearch, ragCosine, ragOpening, RAG_VEC_MIN };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const docs = [
  { id: 'a', name: 'motor.md', text: 'Der Wagen hat 200 PS und beschleunigt in sechs Sekunden auf hundert. Der Motor ist ein Reihensechszylinder.' },
  { id: 'b', name: 'reifen.md', text: 'Die Reifen brauchen einen Druck von 2,4 bar. Im Winter sind Winterreifen vorgeschrieben.' },
];

test('a passage is found by a shared word', () => {
  const index = R.ragBuildIndex(docs);
  const hits = R.ragSearch(index, 'Reifendruck Winter', 3);
  assert(hits.length > 0, 'no hit');
  assert.strictEqual(hits[0].docId, 'b');
});

test('without a shared word nothing is found by keyword alone', () => {
  const index = R.ragBuildIndex(docs);
  assert.deepStrictEqual(R.ragSearch(index, 'Auto Leistung', 3), []);
});

test('a vector can find the passage that shares no word', () => {
  const index = R.ragBuildIndex(docs);
  // the fake space: "motor" and "auto leistung" point the same way, "reifen" points elsewhere
  index.vecs = index.chunks.map((c) => (c.docId === 'a' ? [1, 0, 0] : [0, 1, 0]));
  const hits = R.ragSearch(index, 'Auto Leistung', 3, [0.95, 0.1, 0]);
  assert(hits.length === 1, `expected one hit, got ${hits.length}`);
  assert.strictEqual(hits[0].docId, 'a');
});

test('a passage that is close in no way is left out even with vectors', () => {
  const index = R.ragBuildIndex(docs);
  index.vecs = index.chunks.map((c) => (c.docId === 'a' ? [1, 0] : [0, 1]));
  assert.deepStrictEqual(R.ragSearch(index, 'Auto Leistung', 3, [-1, 0]), [], 'a passage pointing the other way has to stay out');
});

test('both signals add up: a keyword hit is not pushed out by a vector hit', () => {
  const index = R.ragBuildIndex(docs);
  index.vecs = index.chunks.map((c) => (c.docId === 'a' ? [1, 0] : [0, 1]));
  const hits = R.ragSearch(index, 'Reifen Druck', 3, [1, 0]);
  assert(hits.length >= 2, `expected both passages, got ${hits.length}`);
  assert.strictEqual(hits[0].docId, 'b', 'the exact keyword match has to stay first');
});

test('vectors of the wrong length are ignored instead of trusted', () => {
  const index = R.ragBuildIndex(docs);
  index.vecs = [[1, 0]];
  const hits = R.ragSearch(index, 'Auto Leistung', 3, [1, 0]);
  assert.deepStrictEqual(hits, [], 'a half-filled index must not be used');
});

test('cosine: equal, orthogonal, opposite and empty', () => {
  assert.strictEqual(R.ragCosine([1, 2], [1, 2]), 1);
  assert.strictEqual(R.ragCosine([1, 0], [0, 1]), 0);
  assert.strictEqual(R.ragCosine([1, 0], [-1, 0]), -1);
  assert.strictEqual(R.ragCosine([0, 0], [1, 1]), 0);
  assert.strictEqual(R.ragCosine([1], [1, 2]), 0);
});

test('the opening passages are the first of every document', () => {
  const index = R.ragBuildIndex(docs);
  const hits = R.ragOpening(index, 4);
  assert.strictEqual(hits.length, 2);
  assert.deepStrictEqual(hits.map((h) => h.n), [1, 1]);
});

console.log(`${n} tests passed`);
