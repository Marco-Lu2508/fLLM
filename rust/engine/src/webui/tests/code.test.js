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
return { computeMaxTokens, parseContextError, retryMaxTokens, compactTools, globToRegex, parseRules, evaluateRules, commandKey, needsCompaction, splitForCompaction, buildCompactedMessages, parseCommandFile, expandCommand, commandNamesFromPaths, safeToolName, parseJobId, normPath };`)(COMPACT_RATIO);

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

console.log(`${n} tests passed`);
