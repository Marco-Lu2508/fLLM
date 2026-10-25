// Tests for the pure logic of assistants, projects and memory (projects.js) and how project documents are searched (rag.js).
// Run from the repository root:  node rust/engine/src/webui/tests/projects.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const read = (f) => fs.readFileSync(path.join(__dirname, '..', f), 'utf8');
const prjSrc = read('projects.js');
const begin = prjSrc.indexOf('// @pure-begin');
const end = prjSrc.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in projects.js');

// the BM25 part of rag.js: from the constants to the end of ragOpening
const ragSrc = read('rag.js');
const ragBegin = ragSrc.indexOf('const RAG_CHUNK_CHARS');
const ragEnd = ragSrc.indexOf('/** How many passages fit');
assert(ragBegin >= 0 && ragEnd > ragBegin, 'search block not found in rag.js');

const L = new Function(`${ragSrc.slice(ragBegin, ragEnd)}\n${prjSrc.slice(begin, end)}
return { PRJ_LIMITS, asstSanitize, asstMigrate, asstBuiltins, asstStarters, asstExport, asstParseImport, memExtract, memAdd, memSelect, composeSystem,
  prjFitDoc, prjCombinedDocs, prjSidebarView, plibMatch, ragBuildIndex, ragSearch, ragOpening };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };
let counter = 0;
const makeId = () => `id${++counter}`;

// ---- presets to assistants -------------------------------------------------
test('old presets become assistants with the same id and values', () => {
  const presets = [{ id: 'p_abc', name: 'My tone', system: 'Be formal.', temperature: 0.4, top_p: 0.8, top_k: 30 }];
  const r = L.asstMigrate(presets, [], []);
  assert.strictEqual(r.added, 1);
  assert.deepStrictEqual(r.done, ['p_abc']);
  const a = r.assistants[0];
  assert.strictEqual(a.id, 'p_abc');
  assert.strictEqual(a.name, 'My tone');
  assert.strictEqual(a.system, 'Be formal.');
  assert.strictEqual(a.temperature, 0.4);
  assert.strictEqual(a.top_k, 30);
  assert.strictEqual(a.icon, 'bot');
});

test('migration runs once per preset and never replaces an assistant', () => {
  const presets = [{ id: 'p1', name: 'One', system: 'a' }, { id: 'p2', name: 'Two', system: 'b' }];
  const first = L.asstMigrate(presets, [], []);
  const again = L.asstMigrate(presets, first.assistants.filter((a) => a.id !== 'p1'), first.done);
  assert.strictEqual(again.added, 0, 'a deleted assistant must not come back');
  const have = L.asstMigrate(presets, [{ id: 'p2', name: 'Edited', system: 'changed' }], []);
  assert.strictEqual(have.assistants.find((a) => a.id === 'p2').name, 'Edited');
  assert.strictEqual(have.assistants.length, 2);
});

test('migration skips broken presets and keeps defaults for missing values', () => {
  const r = L.asstMigrate([null, { id: 'x' }, { id: 'ok', name: 'Fine' }, { name: 'no id', system: '' }], [], []);
  assert.deepStrictEqual(r.assistants.map((a) => a.id), ['ok']);
  assert.strictEqual(r.assistants[0].temperature, 0.7);
  assert.strictEqual(r.assistants[0].top_p, 0.95);
});

test('built-in presets show up as read-only assistants with starters', () => {
  const presets = [{ id: 'standard', name: 'Standard', system: '', temperature: 0.7, top_p: 0.95, top_k: 20 }];
  const meta = { standard: { icon: 'chat', color: 'teal', starters: ['a', 'b', 'c', 'd', 'e'] } };
  const [a] = L.asstBuiltins(presets, meta);
  assert.strictEqual(a.builtin, true);
  assert.strictEqual(a.icon, 'chat');
  assert.deepStrictEqual(a.starters, ['a', 'b', 'c', 'd']);
});

test('sanitize clamps values and drops unknown icons and colors', () => {
  const a = L.asstSanitize({ name: '  Writer  ', icon: 'nope', color: 'nope', temperature: 9, top_p: -1, top_k: 'x', starters: ['  hi  ', '', 5] }, makeId);
  assert.strictEqual(a.name, 'Writer');
  assert.strictEqual(a.icon, 'bot');
  assert.strictEqual(a.color, 'patina');
  assert.strictEqual(a.temperature, 2);
  assert.strictEqual(a.top_p, 0);
  assert.strictEqual(a.top_k, 20);
  assert.deepStrictEqual(a.starters, ['hi', '5']);
  assert.strictEqual(L.asstSanitize({ name: '   ' }, makeId), null);
});

test('assistant export and import round-trip and give the import a new id', () => {
  const a = L.asstSanitize({ id: 'keep', name: 'Coach', icon: 'star', color: 'red', system: 'Motivate me.', temperature: 0.9, top_p: 0.9, top_k: 40, model: 'm1', starters: ['Plan my day'] }, makeId);
  const r = L.asstParseImport(L.asstExport(a), () => 'fresh');
  assert.strictEqual(r.assistant.id, 'fresh');
  assert.deepStrictEqual({ ...r.assistant, id: a.id }, a);
  assert(L.asstParseImport('nope', makeId).error);
  assert(L.asstParseImport('{"name":"x"}', makeId).error);
  assert(L.asstParseImport(JSON.stringify({ format: 'fllm-assistant', assistant: { name: '' } }), makeId).error);
});

test('suggestions: own first, padded with generic ones to four', () => {
  const s = L.asstStarters({ name: 'Coach', starters: ['Plan my day'] });
  assert.strictEqual(s.length, 4);
  assert.strictEqual(s[0], 'Plan my day');
  assert(s[1].includes('Coach'));
  assert.strictEqual(L.asstStarters({ name: 'X', starters: [] }).length, 4);
});

// ---- memory and the system prompt ---------------------------------------------------
test('"remember that" is recognised only at the start of a message', () => {
  assert.strictEqual(L.memExtract('Remember that I live in Lyon.'), 'I live in Lyon.');
  assert.strictEqual(L.memExtract('please remember: my cat is called Pixel\n\nand then more text'), 'my cat is called Pixel');
  assert.strictEqual(L.memExtract('remember to buy milk'), null);
  assert.strictEqual(L.memExtract('Do you remember that day?'), null);
  assert.strictEqual(L.memExtract('remember that   '), null);
  assert.strictEqual(L.memExtract('remember that ' + 'x'.repeat(1000)).length, L.PRJ_LIMITS.memory);
});

test('memories: duplicates, empty text and the entry limit are refused', () => {
  let list = [];
  let r = L.memAdd(list, ' I like tea ', 1, makeId);
  assert.strictEqual(r.status, 'added');
  list = r.list;
  assert.strictEqual(L.memAdd(list, 'i like TEA', 2, makeId).status, 'duplicate');
  assert.strictEqual(L.memAdd(list, '   ', 2, makeId).status, 'empty');
  const full = Array.from({ length: L.PRJ_LIMITS.memories }, (_, i) => ({ id: `m${i}`, text: `fact ${i}`, createdAt: 0 }));
  assert.strictEqual(L.memAdd(full, 'one more', 3, makeId).status, 'full');
});

test('the memory in the prompt is capped, newest first, in the original order', () => {
  const mem = [1, 2, 3, 4].map((i) => ({ id: `m${i}`, text: `fact ${i}`.padEnd(20, '.'), createdAt: i }));
  const all = L.memSelect(mem, 1000);
  assert.strictEqual(all.dropped, 0);
  assert.deepStrictEqual(all.lines.map((l) => l[5]), ['1', '2', '3', '4']);
  const cut = L.memSelect(mem, 50); // room for two entries of 23 characters
  assert.strictEqual(cut.lines.length, 2);
  assert.strictEqual(cut.dropped, 2);
  assert.deepStrictEqual(cut.lines.map((l) => l[5]), ['3', '4']);
});

test('system prompt: personalization, then project, then the assistant', () => {
  const personal = { useAbout: true, useMemory: true, about: 'I am a nurse.', memories: [{ id: 'a', text: 'I live in Lyon', createdAt: 1 }] };
  const project = { name: 'Clinic', instructions: 'Answer for ward staff.' };
  const out = L.composeSystem({ base: 'Be brief.', personal, project });
  const iAbout = out.indexOf('I am a nurse.');
  const iMem = out.indexOf('- I live in Lyon');
  const iProj = out.indexOf('Answer for ward staff.');
  const iBase = out.indexOf('Be brief.');
  assert(iAbout >= 0 && iAbout < iMem && iMem < iProj && iProj < iBase, out);
  assert(out.includes('"Clinic"'));
});

test('system prompt: switches, empty parts and the per-chat opt-out', () => {
  const personal = { useAbout: false, useMemory: false, about: 'secret', memories: [{ id: 'a', text: 'secret fact', createdAt: 1 }] };
  assert.strictEqual(L.composeSystem({ base: 'Base.', personal, project: null }), 'Base.');
  assert.strictEqual(L.composeSystem({ base: 'Base.', personal: null, project: { name: 'P', instructions: '  ' } }), 'Base.');
  assert.strictEqual(L.composeSystem({ base: '', personal: { ...personal, useAbout: true, about: ' hi ' }, project: null }).trim(),
    'About the user and how they want answers (in their own words):\nhi');
  assert.strictEqual(L.composeSystem({ base: '', personal: null, project: null }), '');
});

test('system prompt: the about text and the memory block are capped', () => {
  const personal = { useAbout: true, useMemory: true, about: 'a'.repeat(5000), memories: Array.from({ length: 100 }, (_, i) => ({ id: `m${i}`, text: 'x'.repeat(100) + i, createdAt: i })) };
  const out = L.composeSystem({ base: '', personal, project: null });
  assert(out.length < L.PRJ_LIMITS.about + L.PRJ_LIMITS.memoryPrompt + 300, `too long: ${out.length}`);
  assert(out.includes('99'), 'the newest memory must be kept');
});

// ---- projects -----------------------------------------------------------------
const doc = (id, name, text) => ({ id, name, text });
const project = (id, docs) => ({ id, name: id, instructions: '', docs });

test('a project document is cut to the room left and refused when a limit is hit', () => {
  const p = project('p', []);
  assert.deepStrictEqual(L.prjFitDoc([p], p, 'a.txt', 'hello'), { text: 'hello' });
  assert(L.prjFitDoc([p], p, 'a.txt', '   ').error.includes('no readable text'));
  const big = L.prjFitDoc([p], p, 'big.txt', 'x'.repeat(L.PRJ_LIMITS.doc + 10));
  assert.strictEqual(big.text.length, L.PRJ_LIMITS.doc);
  assert(big.note.includes('big.txt'));
  const nearlyFull = project('q', [doc('1', 'old', 'y'.repeat(L.PRJ_LIMITS.project - 1000))]);
  assert(L.prjFitDoc([nearlyFull], nearlyFull, 'n.txt', 'z'.repeat(5000)).error.includes('project is full'));
  const others = [1, 2, 3].map((i) => project(`o${i}`, [doc('d', 'd', 'y'.repeat(1200000))])); // 3.6M in all projects
  const fresh = project('f', []);
  assert(L.prjFitDoc([...others, fresh], fresh, 'f.txt', 'z'.repeat(5000)).error.includes('browser storage'));
});

test('the documents searched for a chat: its project first, then its own', () => {
  const projects = [project('p1', [doc('k1', 'handbook.md', 'Refunds are paid within 14 days.')]), project('p2', [doc('k2', 'other.md', 'unrelated')])];
  const thread = { id: 't', project: 'p1', docs: [doc('c1', 'notes.txt', 'My notes')] };
  assert.deepStrictEqual(L.prjCombinedDocs(projects, thread, false).map((d) => d.name), ['handbook.md', 'notes.txt']);
  assert.deepStrictEqual(L.prjCombinedDocs(projects, thread, true).map((d) => d.name), ['notes.txt']);
  assert.deepStrictEqual(L.prjCombinedDocs(projects, { id: 't2', docs: [] }, false), []);
  assert.deepStrictEqual(L.prjCombinedDocs(projects, { id: 't3', project: 'gone', docs: [] }, false), [], 'a deleted project adds nothing');
});

test('a question in a project chat finds the passages of the project documents', () => {
  const projects = [
    project('p1', [
      doc('k1', 'handbook.md', 'Refunds are paid within 14 days after we receive the returned goods.\n\nShipping to Canada takes ten working days.'),
      doc('k2', 'pricing.md', 'The premium plan costs twelve euros per month and includes priority support.'),
    ]),
    project('p2', [doc('k3', 'secret.md', 'The vault code is hidden in the refund policy of the other project.')]),
  ];
  const thread = { id: 't', project: 'p1', docs: [doc('c1', 'notes.txt', 'Remember to water the plants on Friday.')] };
  const index = L.ragBuildIndex(L.prjCombinedDocs(projects, thread, false));
  const hits = L.ragSearch(index, 'How long do refunds take?', 4);
  assert(hits.length >= 1);
  assert.strictEqual(hits[0].doc, 'handbook.md');
  assert(!hits.some((h) => h.doc === 'secret.md'), 'documents of other projects are never searched');
  const plants = L.ragSearch(index, 'when do I water the plants', 4);
  assert.strictEqual(plants[0].doc, 'notes.txt', 'the documents of the chat itself are searched too');
  const chatOnly = L.ragBuildIndex(L.prjCombinedDocs(projects, thread, true));
  assert.deepStrictEqual(L.ragSearch(chatOnly, 'refunds', 4), []);
  assert.strictEqual(L.ragOpening(chatOnly, 3).every((p) => p.doc === 'notes.txt'), true, 'the opening of a fresh document ignores project knowledge');
});

// ---- sidebar view and prompt library ------------------------------------------------
test('sidebar: project chats and archived chats stay out of the main list, a search finds all', () => {
  const t = (id, title, extra = {}) => ({ id, title, updatedAt: Number(id.slice(1)), ...extra });
  const threads = [t('t1', 'Alpha'), t('t2', 'Beta budget', { project: 'p1' }), t('t3', 'Gamma', { archived: true }), t('t4', 'Delta', { project: 'gone' }), t('t5', 'Budget plan', { archived: true })];
  const textOf = (x) => x.title.toLowerCase();
  const base = { projectIds: ['p1'], textOf };
  const plain = L.prjSidebarView(threads, { ...base, q: '', showArchived: false });
  assert.deepStrictEqual(plain.main.map((x) => x.id), ['t4', 't1'], 'a chat of a deleted project is a normal chat');
  assert.deepStrictEqual(plain.archived, []);
  assert.strictEqual(plain.archivedCount, 2);
  const withArchive = L.prjSidebarView(threads, { ...base, q: '', showArchived: true });
  assert.deepStrictEqual(withArchive.archived.map((x) => x.id), ['t5', 't3']);
  const found = L.prjSidebarView(threads, { ...base, q: 'budget', showArchived: false });
  assert.deepStrictEqual(found.main.map((x) => x.id), ['t2']);
  assert.deepStrictEqual(found.archived.map((x) => x.id), ['t5'], 'archived chats are searchable');
});

test('prompt library: "/" filters by title and leaves the built-in commands alone', () => {
  const prompts = [{ id: '1', title: 'Summarize', text: 'x' }, { id: '2', title: 'Email reply', text: 'y' }, { id: '3', title: 'Code summary', text: 'z' }];
  assert.strictEqual(L.plibMatch(prompts, '/').length, 3);
  assert.deepStrictEqual(L.plibMatch(prompts, '/summ').map((p) => p.id), ['1', '3'], 'titles that start with the text come first');
  assert.deepStrictEqual(L.plibMatch(prompts, '/mail').map((p) => p.id), ['2']);
  assert.deepStrictEqual(L.plibMatch(prompts, '/clear'), []);
  assert.deepStrictEqual(L.plibMatch([{ id: '4', title: 'system check', text: 'q' }], '/system you are'), []);
  assert.deepStrictEqual(L.plibMatch(prompts, 'summ'), []);
});

console.log(`${n} tests passed`);
