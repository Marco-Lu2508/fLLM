// Tests for the pure logic of the Code surface (permission rules, compaction, slash commands).
// Run from the repository root:  node rust/engine/src/webui/tests/code.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'code.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in code.js');
const block = src.slice(begin, end);
const COMPACT_RATIO = 0.7;
const L = new Function('COMPACT_RATIO', `${block}
return { computeMaxTokens, parseContextError, retryMaxTokens, compactTools, globToRegex, parseRules, evaluateRules, commandKey, needsCompaction, splitForCompaction, buildCompactedMessages, parseCommandFile, expandCommand, commandNamesFromPaths, safeToolName, parseJobId, normPath,
  modeFor, decidePermission, patchPaths, repairMessages, callSignature, isRepeating, clipMiddle, filterSessions, cutBefore, firstChangeAfter, exportMarkdown, parseShellEscape, summariseGit, heuristicTitle, cleanTitle };`)(COMPACT_RATIO);

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };
const rules = (allow, deny = '') => ({ allow: L.parseRules(allow), deny: L.parseRules(deny) });

test('parseRules reads category: pattern lines and skips junk', () => {
  const r = L.parseRules('bash: git status*\n# comment\n\nedit: src/**\nnonsense\nfoo: bar\nnetwork: *');
  assert.deepStrictEqual(r.map((x) => x.category), ['bash', 'edit', 'network']);
  assert.strictEqual(r[0].pattern, 'git status*');
});

test('bash allow rules match with wildcards', () => {
  const r = rules('bash: npm test*');
  assert.strictEqual(L.evaluateRules(r, 'bash', 'npm test'), 'allow');
  assert.strictEqual(L.evaluateRules(r, 'bash', 'npm test -- --watch'), 'allow');
  assert.strictEqual(L.evaluateRules(r, 'bash', 'npm install'), null);
});

test('allow rules never cover compound commands', () => {
  const r = rules('bash: npm test*');
  assert.strictEqual(L.evaluateRules(r, 'bash', 'npm test && rm -rf x'), null);
  assert.strictEqual(L.evaluateRules(r, 'bash', 'npm test; curl evil'), null);
  assert.strictEqual(L.evaluateRules(rules('bash: ls | wc -l'), 'bash', 'ls | wc -l'), 'allow');
});

test('risky programs need a rule that names them', () => {
  assert.strictEqual(L.evaluateRules(rules('bash: *'), 'bash', 'git push'), null);
  assert.strictEqual(L.evaluateRules(rules('bash: g*'), 'bash', 'git push'), null);
  assert.strictEqual(L.evaluateRules(rules('bash: git status*'), 'bash', 'git status -s'), 'allow');
  assert.strictEqual(L.evaluateRules(rules('bash: git status*'), 'bash', 'git push'), null);
  assert.strictEqual(L.evaluateRules(rules('bash: *'), 'bash', 'ls -la'), 'allow');
});

test('deny wins and also catches parts of compound commands', () => {
  const r = rules('bash: *', 'bash: rm *');
  assert.strictEqual(L.evaluateRules(r, 'bash', 'rm -rf build'), 'deny');
  assert.strictEqual(L.evaluateRules(r, 'bash', 'ls && rm -rf build'), 'deny');
  assert.strictEqual(L.evaluateRules(r, 'bash', 'ls'), 'allow');
});

test('edit rules use path globs', () => {
  const r = rules('edit: src/**', 'edit: **/.env');
  assert.strictEqual(L.evaluateRules(r, 'edit', 'src/a/b.rs'), 'allow');
  assert.strictEqual(L.evaluateRules(r, 'edit', './src\\x.rs'), 'allow');
  assert.strictEqual(L.evaluateRules(r, 'edit', 'docs/a.md'), null);
  assert.strictEqual(L.evaluateRules(r, 'edit', 'src/.env'), 'deny');
  assert.strictEqual(L.evaluateRules(r, 'edit', '.env'), 'deny');
  assert.strictEqual(L.evaluateRules(rules('edit: src/*.rs'), 'edit', 'src/a/b.rs'), null);
});

test('network, mcp and the command alias', () => {
  assert.strictEqual(L.evaluateRules(rules('network: *'), 'network', 'https://example.com'), 'allow');
  assert.strictEqual(L.evaluateRules(rules('mcp: github/*'), 'mcp', 'github/create_issue'), 'allow');
  assert.strictEqual(L.evaluateRules(rules('mcp: github/*'), 'mcp', 'other/x'), null);
  assert.strictEqual(L.evaluateRules(rules('command: github/*'), 'mcp', 'github/x'), 'allow');
  assert.strictEqual(L.evaluateRules(rules('network: *'), 'edit', 'a'), null);
});

test('commandKey is null for risky and compound commands', () => {
  assert.strictEqual(L.commandKey('npm test'), 'npm');
  assert.strictEqual(L.commandKey('C:\\tools\\Cargo.exe build'), 'cargo');
  assert.strictEqual(L.commandKey('git status'), null);
  assert.strictEqual(L.commandKey('ls > out.txt'), null);
});

test('compaction trigger at 70 percent of the context', () => {
  assert.strictEqual(L.needsCompaction(6999, 10000), false);
  assert.strictEqual(L.needsCompaction(7001, 10000), true);
  assert.strictEqual(L.needsCompaction(5000, 0), false);
});

test('splitForCompaction keeps tool results with their call', () => {
  const msgs = [];
  for (let i = 0; i < 4; i++) msgs.push({ role: 'user', content: `u${i}` }, { role: 'assistant', content: '', tool_calls: [{ id: `c${i}` }] }, { role: 'tool', tool_call_id: `c${i}`, content: 'r' });
  const s = L.splitForCompaction(msgs, 6);
  assert.ok(s);
  assert.notStrictEqual(s.tail[0].role, 'tool');
  assert.strictEqual(s.head.length + s.tail.length, msgs.length);
  assert.strictEqual(L.splitForCompaction(msgs.slice(0, 5), 6), null);
});

test('buildCompactedMessages alternates roles and carries tasks and files', () => {
  const out = L.buildCompactedMessages('did things', [{ role: 'user', content: 'next' }], { todos: [{ status: 'pending', content: 'x' }], files: ['a.rs'] });
  assert.deepStrictEqual(out.map((m) => m.role), ['user', 'assistant', 'user']);
  assert.ok(out[0].content.includes('did things') && out[0].content.includes('[pending] x') && out[0].content.includes('a.rs'));
  const out2 = L.buildCompactedMessages('s', [{ role: 'assistant', content: 'a' }]);
  assert.deepStrictEqual(out2.map((m) => m.role), ['user', 'assistant']);
});

test('slash command files: front matter, arguments', () => {
  const f = L.parseCommandFile('---\ndescription: Review a file\n---\nReview $1 and then $ARGUMENTS.\n');
  assert.strictEqual(f.description, 'Review a file');
  assert.strictEqual(L.expandCommand(f.body, 'src/a.rs "be strict"'), 'Review src/a.rs and then src/a.rs "be strict".');
  assert.strictEqual(L.expandCommand('Do it', 'now'), 'Do it\n\nARGUMENTS: now');
  assert.strictEqual(L.expandCommand('Do it', ''), 'Do it');
  assert.strictEqual(L.parseCommandFile('# Title line\nbody').description, 'Title line');
});

test('commandNamesFromPaths', () => {
  const r = L.commandNamesFromPaths(['.fllm/commands/review.md', '.fllm/commands/git/pr.md', '.fllm/commands/review.md', 'other/x.md']);
  assert.deepStrictEqual(r.map((c) => c.name), ['review', 'git:pr']);
});

test('tool names and job ids', () => {
  assert.strictEqual(L.safeToolName('mcp__my server__do.it'), 'mcp__my_server__do_it');
  assert.strictEqual(L.parseJobId('Started job 7f3a2c in the background'), '7f3a2c');
  assert.strictEqual(L.parseJobId('started bash_12'), 'bash_12');
  assert.strictEqual(L.parseJobId('nothing'), null);
});

test('computeMaxTokens fits the answer into what is left', () => {
  assert.strictEqual(L.computeMaxTokens(32768, 2000), 2048);
  const n = L.computeMaxTokens(4096, 2472);
  assert.ok(n >= 200 && n + 2472 < 4096, `got ${n}`);
  assert.strictEqual(L.computeMaxTokens(4096, 3800), 0);
  assert.strictEqual(L.computeMaxTokens(4096, 4500), 0);
});

test('parseContextError and retry size for the real server message', () => {
  const msg = "This model's maximum context length is 4096 tokens. However, you requested 4520 tokens (2472 in the prompt, 2048 in the completion). Please reduce the length.";
  assert.deepStrictEqual(L.parseContextError(msg), { max: 4096, prompt: 2472 });
  const r = L.retryMaxTokens(L.parseContextError(msg));
  assert.ok(r > 200 && r + 2472 <= 4096 && r < 2048, `got ${r}`);
  assert.strictEqual(L.parseContextError('something else'), null);
  assert.strictEqual(L.retryMaxTokens(null), 0);
  assert.strictEqual(L.retryMaxTokens({ max: 4096, prompt: 3900 }), 0);
});

test('compactTools shortens and drops rare tools', () => {
  const tools = [
    { type: 'function', function: { name: 'grep', description: 'Search files. Returns lines with a long explanation that goes on.', parameters: { type: 'object', properties: { pattern: { type: 'string', description: 'regex' } }, required: ['pattern'] } } },
    { type: 'function', function: { name: 'web_search', description: 'Search the web.', parameters: { type: 'object', properties: {} } } },
  ];
  const c = L.compactTools(tools, false);
  assert.strictEqual(c.length, 1);
  assert.strictEqual(c[0].function.description, 'Search files.');
  assert.strictEqual(c[0].function.parameters.properties.pattern.description, undefined);
  assert.deepStrictEqual(c[0].function.parameters.required, ['pattern']);
  assert.strictEqual(L.compactTools(tools, true).length, 2);
  assert.ok(JSON.stringify(c).length < JSON.stringify(tools).length);
});

test('rules can name the agent they are for', () => {
  const r = L.parseRules('[plan] bash: ls*\nbash: git status*');
  assert.deepStrictEqual(r.map((x) => x.agent), ['plan', '']);
  const rs = { allow: r, deny: [] };
  assert.strictEqual(L.evaluateRules(rs, 'bash', 'ls -la', 'plan'), 'allow');
  assert.strictEqual(L.evaluateRules(rs, 'bash', 'ls -la', 'build'), null);
  assert.strictEqual(L.evaluateRules(rs, 'bash', 'git status', 'build'), 'allow');
});

test('permission modes: defaults per agent, user table, auto-edit, rules', () => {
  const none = { allow: [], deny: [] };
  const ask = (extra) => L.decidePermission({ rules: none, modes: {}, agent: 'build', autoEdit: false, category: 'edit', subject: 'a.rs', ...extra });
  assert.strictEqual(ask({}), 'ask');
  assert.strictEqual(ask({ autoEdit: true }), 'allow');
  assert.strictEqual(ask({ agent: 'plan' }), 'deny', 'the plan agent never edits');
  assert.strictEqual(ask({ category: 'read' }), 'allow');
  assert.strictEqual(ask({ category: 'memory' }), 'allow');
  assert.strictEqual(ask({ modes: { build: { edit: 'deny' } } }), 'deny');
  assert.strictEqual(ask({ modes: { build: { bash: 'allow' } }, category: 'bash', subject: 'make' }), 'allow');
  // an allow rule answers a question but never overrides a deny mode; a deny rule beats an allow mode
  const allow = { allow: L.parseRules('edit: src/**'), deny: [] };
  assert.strictEqual(ask({ rules: allow, subject: 'src/a.rs' }), 'allow');
  assert.strictEqual(ask({ rules: allow, subject: 'src/a.rs', agent: 'plan' }), 'deny');
  const deny = { allow: [], deny: L.parseRules('edit: **/.env') };
  assert.strictEqual(ask({ rules: deny, subject: '.env', modes: { build: { edit: 'allow' } } }), 'deny');
  assert.strictEqual(ask({ remembered: true, category: 'bash', subject: 'npm test' }), 'allow');
  assert.strictEqual(L.modeFor({ build: { edit: 'bogus' } }, 'build', 'edit', false), 'ask');
});

test('patchPaths lists the files of a unified diff', () => {
  const p = '--- a/src/a.rs\n+++ b/src/a.rs\n@@ -1 +1 @@\n-x\n+y\n--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+z\n--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-q\n';
  assert.deepStrictEqual(L.patchPaths(p), ['src/a.rs', 'new.txt', 'gone.txt']);
  assert.deepStrictEqual(L.patchPaths('nothing'), []);
});

test('repairMessages closes interrupted tool calls and keeps roles valid', () => {
  const call = (id) => ({ id, type: 'function', function: { name: 'read_file', arguments: '{}' } });
  const broken = [
    { role: 'user', content: 'a' },
    { role: 'assistant', content: '', tool_calls: [call('1'), call('2')] },
    { role: 'tool', tool_call_id: '2', content: 'two' },
    { role: 'user', content: 'b' },
    { role: 'user', content: 'c' },
    { role: 'tool', tool_call_id: 'zzz', content: 'orphan' },
    { role: 'assistant', content: '' },
  ];
  const { messages, changed } = L.repairMessages(broken);
  assert.ok(changed > 0);
  assert.deepStrictEqual(messages.map((m) => m.role), ['user', 'assistant', 'tool', 'tool', 'user', 'assistant', 'user']);
  assert.strictEqual(messages[2].tool_call_id, '1');
  assert.ok(/Interrupted/.test(messages[2].content));
  assert.strictEqual(messages[3].content, 'two');
  // a valid conversation is left alone
  const ok = [{ role: 'user', content: 'x' }, { role: 'assistant', content: 'y' }];
  assert.strictEqual(L.repairMessages(ok).changed, 0);
  // a trailing assistant call without any result gets one
  const tail = L.repairMessages([{ role: 'user', content: 'x' }, { role: 'assistant', content: '', tool_calls: [call('9')] }]).messages;
  assert.strictEqual(tail[tail.length - 1].tool_call_id, '9');
});

test('a repeated identical call is detected, key order does not matter', () => {
  const a = L.callSignature('grep', { pattern: 'x', path: 'src' });
  const b = L.callSignature('grep', { path: 'src', pattern: 'x' });
  assert.strictEqual(a, b);
  assert.ok(!L.isRepeating([a], b, 3));
  assert.ok(L.isRepeating([a, b], b, 3));
  assert.ok(!L.isRepeating([a, L.callSignature('grep', { pattern: 'y' })], b, 3));
});

test('clipMiddle keeps the start and the end', () => {
  const s = `${'a'.repeat(500)}${'b'.repeat(500)}END`;
  const c = L.clipMiddle(s, 200);
  assert.ok(c.startsWith('aaaa') && c.endsWith('END') && c.includes('cut from the middle'));
  assert.strictEqual(L.clipMiddle('short', 200), 'short');
});

test('sessions: search, cut before a message, files to revert', () => {
  const sessions = [
    { id: '1', title: 'Fix parser', view: [{ kind: 'user', text: 'the lexer crashes' }] },
    { id: '2', title: 'Docs', view: [{ kind: 'user', text: 'write a readme' }] },
  ];
  assert.deepStrictEqual(L.filterSessions(sessions, 'lexer fix').map((s) => s.id), ['1']);
  assert.strictEqual(L.filterSessions(sessions, '').length, 2);
  const view = [
    { id: 'u1', kind: 'user', text: 'one', msgIndex: 0, changeMark: 0, todosBefore: [] },
    { id: 'a1', kind: 'assistant', content: 'ok' },
    { id: 'u2', kind: 'user', text: 'two', msgIndex: 2, changeMark: 5, todosBefore: [{ content: 't', status: 'pending' }] },
    { id: 'a2', kind: 'assistant', content: 'done' },
  ];
  const messages = [{ role: 'user', content: 'one' }, { role: 'assistant', content: 'ok' }, { role: 'user', content: 'two' }, { role: 'assistant', content: 'done' }];
  const cut = L.cutBefore(view, messages, 'u2');
  assert.strictEqual(cut.view.length, 2);
  assert.strictEqual(cut.messages.length, 2);
  assert.strictEqual(cut.text, 'two');
  assert.strictEqual(cut.todos.length, 1);
  assert.strictEqual(L.cutBefore(view, messages, 'a1'), null);
  assert.strictEqual(L.cutBefore([{ id: 'x', kind: 'user', text: 't' }], messages, 'x'), null, 'compacted: no index');
  assert.strictEqual(L.firstChangeAfter([{ id: 3 }, { id: 6 }, { id: 8 }], 5), 6);
  assert.strictEqual(L.firstChangeAfter([{ id: 3 }], 5), null);
});

test('export as Markdown', () => {
  const md = L.exportMarkdown({ title: 'My session', time: 0, view: [
    { kind: 'user', text: 'run the tests' },
    { kind: 'tool', name: 'bash', args: { command: 'cargo test' }, status: 'ok', output: 'all green ``` fence' },
    { kind: 'assistant', content: 'All tests pass.' },
  ] }, '/work/x');
  assert.ok(md.startsWith('# My session') && md.includes('## User') && md.includes('run the tests'));
  assert.ok(md.includes('Tool `bash` `cargo test` (ok)') && md.includes('## Assistant'));
  assert.ok(md.includes('````'), 'a fence inside the output needs a longer fence');
});

test('small helpers: shell escape, git summary, titles', () => {
  assert.strictEqual(L.parseShellEscape('!ls -la'), 'ls -la');
  assert.strictEqual(L.parseShellEscape('! git status'), 'git status');
  assert.strictEqual(L.parseShellEscape('hello !'), null);
  assert.strictEqual(L.summariseGit('## main...origin/main\n'), 'clean');
  assert.strictEqual(L.summariseGit('## main...origin/main [ahead 2]\n M a.rs\n?? b.rs\n'), '2 changed files (1 modified, 1 untracked), ahead 2');
  assert.strictEqual(L.cleanTitle('"Fix the lexer crash."\nbecause...'), 'Fix the lexer crash');
  assert.strictEqual(L.cleanTitle('Title: **Docs**'), 'Docs');
  assert.strictEqual(L.heuristicTitle([{ role: 'user', content: 'hello\n\nContents of @a.rs:\n...' }]), 'hello');
});

console.log(`${n} tests passed`);
