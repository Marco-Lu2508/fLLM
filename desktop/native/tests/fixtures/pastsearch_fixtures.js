// Fixtures of the search over past chats: the @pure block of sessions.js (`ssTokenize`, `ssMessageText`,
// `ssThreadDoc`, `ssCodeDoc`, `ssBuildIndex`, `ssSearch`, `ssSnippet`, `ssMerge`) runs in a vm.
// Run: node tests/fixtures/pastsearch_fixtures.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const src = fs.readFileSync(path.join(WEBUI, 'sessions.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('the pure block of sessions.js was not found');
const zone = src.slice(src.indexOf('\n', begin) + 1, src.lastIndexOf('\n', end));
const api = vm.runInNewContext(`${zone}\n;({ ssTokenize, ssMessageText, ssThreadDoc, ssCodeDoc, ssBuildIndex, ssSearch, ssSnippet, ssMerge, SS_MAX_TEXT })`, { Math, String, Number, Array, Object, RegExp, Map, Set, JSON }, { timeout: 5000 });

const long = (n, c) => c.repeat(n);
const out = {};

out.tokens = ['Hello, World! 42', 'a b cc', 'Ünïcode Straße ΣΊΣΥΦΟΣ', '', '   ', 'x-y_z 12ab', '日本語のテキスト 123', 'don\'t stop', '𝔸𝔹 a'].map((t) => ({ text: t, got: api.ssTokenize(t) }));
out.tokens.push({ text: null, got: api.ssTokenize(null) });

const DOCS = [
  { id: 'a', kind: 'chat', title: 'Rust borrow checker', text: 'The borrow checker rejects two mutable borrows. Use clones or Rc.', updated: 10 },
  { id: 'b', kind: 'chat', title: 'Docker compose', text: 'docker compose up -d starts the services; the borrow word appears once.', updated: 30 },
  { id: 'c', kind: 'code', title: 'Fix the parser', text: 'parser parser parser the borrow of the token stream', updated: 20, workspace: '/w' },
  { id: 'd', kind: 'chat', title: '', text: 'Nothing about it here', updated: 5 },
  { id: 'e', kind: 'chat', title: 'borrow', text: '', updated: 40 },
  { id: 'f', kind: 'chat', title: 'Same score one', text: 'alpha beta', updated: 1 },
  { id: 'g', kind: 'chat', title: 'Same score two', text: 'alpha beta', updated: 2 },
];
const QUERIES = ['borrow', 'borrow checker', 'parser', 'docker compose', 'alpha', 'nothing', 'zzz', '', '   ', 'a', 'BORROW borrow', 'the'];
out.docs = DOCS;
out.searches = QUERIES.map((q) => {
  const index = api.ssBuildIndex(DOCS);
  return { query: q, limit: 20, hits: api.ssSearch(index, q, 20).map((h) => ({ id: h.doc.id, score: h.score })) };
});
out.searches.push({ query: 'borrow', limit: 2, hits: api.ssSearch(api.ssBuildIndex(DOCS), 'borrow', 2).map((h) => ({ id: h.doc.id, score: h.score })) });
out.empty_index = api.ssSearch(api.ssBuildIndex([]), 'borrow', 20);

const TEXT = 'The quick brown fox jumps over the lazy dog. '.repeat(20);
out.snippets = [
  { text: TEXT, query: 'lazy' }, { text: TEXT, query: 'fox' }, { text: TEXT, query: 'nothingthere' }, { text: 'short text', query: 'text' },
  { text: `${long(300, 'x ')}needle${long(300, ' y')}`, query: 'needle' }, { text: '', query: 'a' }, { text: 'line one\n\n  line   two with   spaces', query: 'two' },
  { text: 'abc def ghi', query: 'zz def' }, { text: `${long(10, 'a ')}ünï${long(400, ' b')}`, query: 'ünï' },
].map((c) => ({ ...c, width: 160, got: api.ssSnippet(c.text, c.query) }));
out.snippets.push({ text: TEXT, query: 'lazy', width: 40, got: api.ssSnippet(TEXT, 'lazy', 40) });

const hit = (id) => ({ id });
out.merges = [
  { local: [hit('a'), hit('b')], remote: [hit('b'), hit('c')], limit: 20 },
  { local: [], remote: [hit('x'), hit('x'), hit('y')], limit: 20 },
  { local: [hit('a')], remote: [], limit: 20 },
  { local: [hit('a'), hit('b'), hit('c')], remote: [hit('d')], limit: 2 },
].map((c) => ({ ...c, got: api.ssMerge(c.local, c.remote, c.limit).map((h) => h.id) }));

const CODE = [
  { id: 's1', title: 'T', time: 5, messages: [{ role: 'system', content: 'hidden' }, { role: 'user', content: 'fix it' }, { role: 'assistant', content: '' }, { role: 'assistant', content: 'done' }, { role: 'tool', content: 'tool out' }] },
  { id: 's2', title: '', time: 0, messages: [{ role: 'user', content: long(200001, 'a') }] },
  { id: 's3', title: 'Ünï', time: 9, messages: [{ role: 'user', content: `${long(199998, 'b')}😀😀` }] },
  { id: 's4', title: 'none', time: 1, messages: [] },
];
out.code_docs = CODE.map((s) => ({ session: s, root: '/w', got: api.ssCodeDoc(s, '/w') }));
out.code_docs.push({ session: CODE[0], root: '', got: api.ssCodeDoc(CODE[0], '') });
out.max = api.SS_MAX_TEXT;
fs.writeFileSync(path.join(__dirname, 'pastsearch_fixtures.json'), JSON.stringify(out, null, 1));
console.log('written', out.tokens.length, out.searches.length, out.snippets.length, out.merges.length, out.code_docs.length);
