// Tests for skills.js (skill files, tool schemas, proposals) and sessions.js (BM25 search over chats), and a check that
// the whole page script still parses.
// Run from the repository root:  node rust/engine/src/webui/tests/skills.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, '..');
const read = (f) => fs.readFileSync(path.join(dir, f), 'utf8');
const pure = (file, exports) => {
  const src = read(file);
  const begin = src.indexOf('// @pure-begin');
  const end = src.indexOf('// @pure-end');
  assert(begin >= 0 && end > begin, `pure block markers not found in ${file}`);
  return new Function(`${src.slice(begin, end)}\nreturn { ${exports} };`)();
};

const S = pure('skills.js', 'skValidName, skSlug, skParse, skCompose, skIndexText, SK_TOOL, skCheckToolArgs, SC_TOOL, scCheckToolArgs, skTranscript, skProposalPrompt, skParseProposal');
const Q = pure('sessions.js', 'ssTokenize, ssMessageText, ssThreadDoc, ssCodeDoc, ssBuildIndex, ssSearch, ssSnippet, ssMerge');

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

test('skill names are slugs', () => {
  for (const ok of ['a', 'deploy-app', 'x_1', '0day']) assert(S.skValidName(ok), ok);
  for (const bad of ['', '../x', 'a/b', 'A', '-a', '.h', 'a b', 'a'.repeat(65), null, 3]) assert(!S.skValidName(bad), String(bad));
  assert.strictEqual(S.skSlug('  Release the App!  '), 'release-the-app');
  assert.strictEqual(S.skSlug('###'), '');
});

test('front matter is parsed and composed', () => {
  const { meta, body } = S.skParse('---\nname: deploy\ndescription: "Ship it"\nwhen: after a release\n---\n# Steps\n1. go\n');
  assert.deepStrictEqual(meta, { name: 'deploy', description: 'Ship it', when: 'after a release' });
  assert(body.startsWith('# Steps'));
  assert.deepStrictEqual(S.skParse('# only text\n'), { meta: {}, body: '# only text\n' });
  assert.deepStrictEqual(S.skParse('---\nname: open\nno closing\n').meta, {});
  const text = S.skCompose({ name: 'a', description: 'one\ntwo', when: 'now', body: '1. x' });
  assert.strictEqual(text, '---\nname: a\ndescription: one two\nwhen: now\n---\n1. x\n');
  assert.deepStrictEqual(S.skParse(text).meta, { name: 'a', description: 'one two', when: 'now' });
  assert.strictEqual(S.skParse('﻿---\nname: b\n---\nx').meta.name, 'b');
});

test('the skill index for the system prompt is capped', () => {
  assert.strictEqual(S.skIndexText([]), '');
  assert.strictEqual(S.skIndexText([{ name: '../bad', description: 'x' }]), '');
  const text = S.skIndexText([{ name: 'deploy', description: 'Ship it', when: 'after a release', scope: 'user' }, { name: 'notes', description: 'Take notes', scope: 'workspace' }]);
  assert(text.includes('- deploy: Ship it (use when: after a release)'));
  assert(text.includes('- notes [workspace]: Take notes'));
  const many = Array.from({ length: 100 }, (_, i) => ({ name: `skill-${i}`, description: 'd'.repeat(100) }));
  const capped = S.skIndexText(many, 1000);
  assert(capped.length < 1200, String(capped.length));
  assert(/more: call the skill tool with action "list"/.test(capped));
  assert(S.skIndexText(many, 100000, 5).split('\n- ').length === 6);
});

test('the skill tool schema and its argument check', () => {
  assert.strictEqual(S.SK_TOOL.name, 'skill');
  assert.strictEqual(S.SK_TOOL.permission, 'edit');
  assert.deepStrictEqual(S.SK_TOOL.properties.action.enum, ['list', 'read', 'create', 'update', 'delete']);
  assert.deepStrictEqual(S.SK_TOOL.required, ['action']);
  assert.strictEqual(S.skCheckToolArgs({ action: 'list' }), '');
  assert.strictEqual(S.skCheckToolArgs({ action: 'read', name: 'deploy' }), '');
  assert(S.skCheckToolArgs({ action: 'fly' }));
  assert(S.skCheckToolArgs({ action: 'read', name: '../x' }));
  assert(S.skCheckToolArgs({ action: 'create', name: 'x' }), 'create needs content');
  assert(S.skCheckToolArgs({ action: 'create', name: 'x', content: 'y'.repeat(70000) }));
  assert(S.skCheckToolArgs({ action: 'read', name: 'x', scope: 'galaxy' }));
  assert.strictEqual(S.skCheckToolArgs({ action: 'update', name: 'x', content: '---\n---\nz', scope: 'workspace' }), '');
});

test('the schedule tool schema and its argument check', () => {
  assert.strictEqual(S.SC_TOOL.name, 'schedule');
  assert.deepStrictEqual(S.SC_TOOL.required, ['name', 'prompt', 'when']);
  assert.strictEqual(S.scCheckToolArgs({ name: 'n', prompt: 'p', when: 'hourly' }), '');
  assert(S.scCheckToolArgs({ name: 'n', prompt: 'p' }));
  assert(S.scCheckToolArgs({ name: 'n', prompt: 'p', when: 'hourly', mode: 'agent' }), 'agent needs a folder');
  assert.strictEqual(S.scCheckToolArgs({ name: 'n', prompt: 'p', when: 'hourly', mode: 'agent', folder: '/work' }), '');
  assert(S.scCheckToolArgs({ name: 'n', prompt: 'p', when: 'hourly', mode: 'root' }));
});

test('a session becomes a transcript for the skill proposal', () => {
  const messages = [
    { role: 'system', content: 'hidden' },
    { role: 'user', content: 'Deploy the app' },
    { role: 'assistant', content: '', tool_calls: [{ function: { name: 'bash', arguments: '{"command":"make"}' } }] },
    { role: 'tool', content: 'ok' },
    { role: 'assistant', content: 'Done.' },
  ];
  const t = S.skTranscript(messages);
  assert(!t.includes('hidden'));
  assert(t.includes('USER: Deploy the app') && t.includes('[calls: bash({"command":"make"})]') && t.includes('TOOL RESULT: ok'));
  const long = S.skTranscript(Array.from({ length: 200 }, (_, i) => ({ role: 'user', content: `step ${i} ${'x'.repeat(300)}` })), 3000);
  assert(long.length < 3200 && long.includes('left out') && long.includes('step 0') && long.includes('step 199'));
  assert(S.skProposalPrompt(t).includes('=== SESSION ===') && S.skProposalPrompt(t).includes('NONE'));
});

test('the reply of the model is turned into a proposal', () => {
  const good = '---\nname: Deploy App\ndescription: Ship the app\nwhen: after a release\n---\n# Deploy\n1. make\n';
  assert.deepStrictEqual(S.skParseProposal(good), { name: 'deploy-app', content: good });
  const fenced = S.skParseProposal(`<think>hmm</think>\n\`\`\`markdown\n${good.trim()}\n\`\`\``);
  assert.strictEqual(fenced.name, 'deploy-app');
  assert(fenced.content.startsWith('---\nname: Deploy App'));
  assert.strictEqual(S.skParseProposal(`Here you go:\n${good}`).name, 'deploy-app');
  assert(S.skParseProposal('NONE').error);
  assert(S.skParseProposal('just some prose').error);
  assert(S.skParseProposal('---\nname: x\n---\n').error, 'no description and no steps');
});

// ---- sessions.js ----------------------------------------------------------------------------------------
const thread = (id, title, texts, updatedAt = 1) => ({
  id, title, updatedAt,
  nodes: Object.fromEntries([['root', { id: 'root', role: 'root', children: [] }]].concat(texts.map((t, i) => [`m${i}`, { id: `m${i}`, role: i % 2 ? 'assistant' : 'user', content: t, reasoning: 'secret thoughts', createdAt: i }]))),
});

test('tokenizing keeps words and digits and drops one-letter words', () => {
  assert.deepStrictEqual(Q.ssTokenize('Hello, World! a 42 x-ray Über'), ['hello', 'world', '42', 'ray', 'über']);
  assert.strictEqual(Q.ssMessageText([{ type: 'text', text: 'a' }, { type: 'image_url' }, { text: 'b' }]), 'a  b');
});

test('thread and code session documents', () => {
  const d = Q.ssThreadDoc(thread('t1', 'Pasta', ['how to cook pasta', 'boil water']));
  assert.strictEqual(d.kind, 'chat');
  assert(d.text.includes('how to cook pasta') && d.text.includes('boil water') && !d.text.includes('secret'));
  const c = Q.ssCodeDoc({ id: 's1', title: 'Fix build', time: 5, messages: [{ role: 'user', content: 'fix it' }, { role: 'tool', content: 'noise' }, { role: 'assistant', content: 'fixed' }] }, '/work');
  assert.deepStrictEqual([c.id, c.kind, c.workspace, c.text], ['code:s1', 'code', '/work', 'fix it\nfixed']);
  assert(Q.ssThreadDoc(thread('big', 't', ['x'.repeat(300000)])).text.length <= 200000);
});

test('BM25 ranks rare words and titles first', () => {
  const docs = [
    Q.ssThreadDoc(thread('a', 'Cooking', ['pasta with tomato sauce and basil'], 1)),
    Q.ssThreadDoc(thread('b', 'Rust borrow checker', ['lifetimes and ownership'], 2)),
    Q.ssThreadDoc(thread('c', 'Misc', ['the borrow of a book from the library'], 3)),
  ];
  const index = Q.ssBuildIndex(docs);
  const hits = Q.ssSearch(index, 'borrow checker');
  assert.deepStrictEqual(hits.map((h) => h.doc.id), ['b', 'c']);
  assert(hits[0].score > hits[1].score && hits[1].score > 0);
  assert.deepStrictEqual(Q.ssSearch(index, 'zzzz'), []);
  assert.deepStrictEqual(Q.ssSearch(index, 'a'), []);
  assert.deepStrictEqual(Q.ssSearch(Q.ssBuildIndex([]), 'x'), []);
  assert.strictEqual(Q.ssSearch(index, 'borrow', 1).length, 1);
  // a word repeated in a short document beats the same word in a long one
  const long = Q.ssThreadDoc(thread('long', 'Long', [`needle ${'filler '.repeat(500)}`], 1));
  const short = Q.ssThreadDoc(thread('short', 'Short', ['needle needle here'], 1));
  assert.strictEqual(Q.ssSearch(Q.ssBuildIndex([long, short]), 'needle')[0].doc.id, 'short');
});

test('snippets surround the match and hits merge without repeats', () => {
  const text = `${'x '.repeat(100)}needle${' y'.repeat(100)}`;
  const s = Q.ssSnippet(text, 'NEEDLE');
  assert(s.includes('needle') && s.startsWith('...') && s.endsWith('...') && s.length < 200, s);
  assert.strictEqual(Q.ssSnippet('short text', 'nothing'), 'short text');
  const merged = Q.ssMerge([{ id: 'a' }, { id: 'b' }], [{ id: 'b' }, { id: 'c' }], 3);
  assert.deepStrictEqual(merged.map((h) => h.id), ['a', 'b', 'c']);
  assert.strictEqual(Q.ssMerge([{ id: 'a' }], [{ id: 'c' }, { id: 'd' }], 2).length, 2);
});

test('the whole page script still parses with the new parts', () => {
  const rs = read('../webui.rs');
  const block = /const SCRIPT: \[&str; (\d+)\] = \[([\s\S]*?)\];/.exec(rs);
  assert(block, 'SCRIPT list not found in webui.rs');
  const files = [...block[2].matchAll(/include_str!\("webui\/([^"]+)"\)/g)].map((m) => m[1]);
  assert.strictEqual(Number(block[1]), files.length, 'array length of SCRIPT');
  assert(files.indexOf('skills.js') > files.indexOf('providers.js') && files.indexOf('sessions.js') > files.indexOf('skills.js'));
  const joined = files.filter((f) => f !== 'runner.js').map(read).join('');
  new Function(`'use strict';\n${joined}`); // throws on a syntax error or a duplicate declaration
});

console.log(`${n} tests passed`);
