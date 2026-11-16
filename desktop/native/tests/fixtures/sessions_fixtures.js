// Row 79 fixtures: the sessions of a folder — `heuristicTitle`, `cleanTitle`, `filterSessions`,
// `firstChangeAfter`, `cutBefore`, `exportMarkdown`, `clipMiddle` and the arithmetic of the compaction
// of the page's own code.js (its @pure block), `estimateTokens` and `fileSafe` of chat.js, run in a
// vm. Run: node tests/fixtures/sessions_fixtures.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const src = fs.readFileSync(path.join(WEBUI, 'code.js'), 'utf8');
const chat = fs.readFileSync(path.join(WEBUI, 'chat.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('the pure block of code.js was not found');
const zone = src.slice(src.indexOf('\n', begin) + 1, src.lastIndexOf('\n', end));
const ratioLine = src.split('\n').find((l) => /const COMPACT_RATIO\s*=/.test(l));
const estLine = chat.split('\n').find((l) => /const estimateTokens\s*=/.test(l));
const safeLine = chat.split('\n').find((l) => /const fileSafe\s*=/.test(l));
if (!ratioLine || !estLine || !safeLine) throw new Error('a helper line of the page was not found');

const api = vm.runInNewContext(
  `${ratioLine}\n${estLine}\n${safeLine}\n${zone}\n;({ COMPACT_RATIO, estimateTokens, fileSafe, heuristicTitle, cleanTitle, filterSessions, firstChangeAfter, cutBefore, exportMarkdown, clipMiddle, needsCompaction, splitForCompaction, buildCompactedMessages })`,
  { Date, Math, JSON, String, Number, Array, Object, Map, Set, RegExp },
  { timeout: 5000 },
);

const out = { ratio: api.COMPACT_RATIO };

// titles
const FIRSTS = ['Fix the bug in main.rs', '   spaced   out\n words   here  ', '', 'x'.repeat(100), 'Ünï ' + 'ö'.repeat(70), 'look at this\n\nContents of @file.txt\nabc', '[Summary of the earlier conversation] and more', '😀'.repeat(40)];
out.heuristic = FIRSTS.map((t) => ({ text: t, title: api.heuristicTitle([{ role: 'user', content: t }]) }));
out.heuristic.push({ text: '__none__', title: api.heuristicTitle([]) });
out.heuristic.push({ text: '__summary_then_real__', title: api.heuristicTitle([{ role: 'user', content: '[Summary x' }, { role: 'user', content: 'Real question' }]), second: 'Real question' });
const CLEAN = ['Title: Fix the bug', '"Quoted title"', '**Bold**', '# Heading', '', '\n\n  second line first  \nthird', 'session : the thing.', 'Many words. ', "'single'", 'x'.repeat(80), 'Title:', '___', '`code`', 'ends with dots...', 'TITLE:   Upper'];
out.clean = CLEAN.map((t) => ({ text: t, title: api.cleanTitle(t) }));

// the search of the list
const SESS = [
  { id: 'a', title: 'Fix the parser', view: [{ kind: 'user', text: 'the parser crashes on unicode' }] },
  { id: 'b', title: 'Write docs', view: [{ kind: 'user', text: 'document the API' }, { kind: 'assistant', content: 'unicode' }] },
  { id: 'c', title: '', view: [] },
];
out.filter = ['', '  ', 'parser', 'PARSER unicode', 'docs api', 'unicode', 'nothing', 'fix unicode'].map((q) => ({ query: q, ids: api.filterSessions(SESS, q).map((s) => s.id) }));
out.filter_sessions = SESS;

// the first change after a mark
const CH = [[], [{ id: 3 }, { id: 5 }, { id: 9 }], [{ id: 9 }, { id: 2 }, { id: 4 }]];
out.change = [];
for (const c of CH) for (const m of [undefined, 0, 2, 4, 9, 100]) out.change.push({ ids: c.map((x) => x.id), mark: m === undefined ? null : m, got: api.firstChangeAfter(c, m) });

// the markdown of an export
const MD_CASES = [
  { title: 'Fix the bug', time: 1699990000000, root: '/w', view: [
    { kind: 'user', text: 'please fix' },
    { kind: 'assistant', content: 'On it.' },
    { kind: 'tool', name: 'read_file', args: { path: 'src/main.rs' }, status: 'ok', output: 'x' },
    { kind: 'tool', name: 'bash', args: { command: 'cargo test\nsecond' }, status: 'error', output: 'boom ``` fenced' },
    { kind: 'notice', text: 'Undone: x\nline two' },
    { kind: 'assistant', content: '' },
    { kind: 'assistant', content: 'Done.' },
  ] },
  { title: '', time: 1, root: '', view: [{ kind: 'user', text: 'hi' }] },
  { title: 'Sub', time: 1700000000000, root: '', view: [
    { kind: 'tool', name: 'task', args: { description: 'dig' }, status: 'ok', output: 'report\nmore', children: [
      { kind: 'assistant', content: 'child says' },
      { kind: 'tool', name: 'bash', args: { command: 'ls' }, status: 'ok', output: 'a\nb' },
    ] },
    { kind: 'tool', name: 'grep', args: { pattern: 'foo', glob: '*.rs' }, status: 'ok', output: '' },
    { kind: 'tool', name: 'web_fetch', args: { url: 'https://x.y/z' }, status: 'ok', output: '' },
    { kind: 'tool', name: 'weird', args: {}, status: 'error', output: '' },
    { kind: 'tool', name: 'bash', args: { command: 'echo' }, status: 'ok', output: 'y'.repeat(2000) },
  ] },
];
out.markdown = MD_CASES.map((c) => ({ ...c, md: api.exportMarkdown({ title: c.title, time: c.time, view: c.view }, c.root) }));

// clipMiddle
out.clip = [['short', 10], ['x'.repeat(30), 20], ['a'.repeat(50) + 'b'.repeat(50), 40], ['😀'.repeat(20), 16], ['', 5]].map(([t, m]) => ({ text: t, max: m, got: api.clipMiddle(t, m) }));

// the arithmetic of the compaction
out.estimate = ['', 'a', 'abcd', 'x'.repeat(36), 'x'.repeat(37), 'Ünï', '😀😀😀', 'y'.repeat(1000)].map((t) => ({ text: t, got: api.estimateTokens(t) }));
out.needs = [[0, 0], [100, 0], [71, 100], [70, 100], [70.0001, 100], [5000, 4096], [2000, 4096], [2867, 4096], [2868, 4096]].map(([e, m]) => ({ estimate: e, max: m, got: api.needsCompaction(e, m) }));
const role = (r) => ({ role: r, content: 'c' });
const MSGS = (s) => s.split('').map((c) => role({ u: 'user', a: 'assistant', t: 'tool' }[c]));
out.split = ['uatuatuatuat', 'uatuat', 'uuuuuu', 'uattttttttttt', 'uatuatuatua', 'uauauauauauauau', 'uatttt', 'uatuatttttt', ''].map((s) => {
  const r = api.splitForCompaction(MSGS(s));
  return { roles: s, start: r ? r.head.length : null, headLen: r ? r.head.length : null, tailLen: r ? r.tail.length : null };
});
out.split_custom = [['uatuatuatuat', 2, 1], ['uatuat', 3, 3], ['uatuatuat', 4, 5]].map(([s, k, h]) => {
  const r = api.splitForCompaction(MSGS(s), k, h);
  return { roles: s, keepLast: k, minHead: h, start: r ? r.head.length : null };
});
const TODOS = [[], [{ status: 'pending', content: 'one' }, { status: 'completed', content: 'two' }]];
const FILES = [[], ['a.rs', 'b.rs'], Array.from({ length: 45 }, (_, i) => `f${i}.rs`)];
out.build = [];
for (const todos of TODOS) for (const files of FILES) for (const tail of [[], MSGS('uat'), MSGS('atu')]) {
  const got = api.buildCompactedMessages('  A summary.\nline two  ', tail, { todos, files });
  out.build.push({ todos, files, tailRoles: tail.map((m) => m.role), prefix: got.slice(0, got.length - tail.length).map((m) => [m.role, m.content]) });
}
out.file_safe = ['Fix the bug', 'a/b\\c:d*e', 'Ünï cödé ok', '', '???', 'x'.repeat(90), 'dots.and-dashes ok'].map((t) => ({ text: t, got: api.fileSafe(t) }));

fs.writeFileSync(path.join(__dirname, 'sessions_fixtures.json'), JSON.stringify(out, null, 1));
console.log('written', Object.keys(out).map((k) => `${k}:${Array.isArray(out[k]) ? out[k].length : '-'}`).join(' '));
