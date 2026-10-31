// Tests for the system prompts of the Code agents and for the assembled page script.
// Run from the repository root:  node rust/engine/src/webui/tests/codeprompts.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, '..');
const src = fs.readFileSync(path.join(dir, 'codeprompts.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in codeprompts.js');
const P = new Function(`${src.slice(begin, end)}
return { codeModelFamily, codeModelBillions, codePromptTier, codeEnvironment, memoryBlock, instructionsBlock, buildSystemPrompt, subAgentPrompt, FAMILY_NOTES, PERSONALITIES, personalityNote, isCleanDiagnostic, reviewFeedback, sessionInsights, searchHitsText };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };
const ALL_TOOLS = new Set(['list_dir', 'read_file', 'write_file', 'edit_file', 'multi_edit', 'patch', 'glob', 'grep', 'bash', 'diagnostics', 'memory', 'todo_write', 'todo_read', 'task', 'ask_user']);

test('model families come from the model name', () => {
  const f = P.codeModelFamily;
  assert.strictEqual(f('Qwen3-Coder-30B-A3B-Instruct-Q4_K_M'), 'qwen');
  assert.strictEqual(f('Meta-Llama-3.1-8B-Instruct'), 'llama');
  assert.strictEqual(f('Mistral-Small-3.2-24B'), 'mistral');
  assert.strictEqual(f('devstral-small'), 'mistral');
  assert.strictEqual(f('gemma-3-12b-it'), 'gemma');
  assert.strictEqual(f('gpt-oss-20b'), 'gptoss');
  assert.strictEqual(f('some-new-model'), 'generic');
  assert.strictEqual(f('custom', 'chatml qwen'), 'qwen', 'the template name counts too');
});

test('size tiers: small contexts and small models get the compact prompt', () => {
  assert.strictEqual(P.codeModelBillions('qwen3-8b-instruct'), 8);
  assert.strictEqual(P.codeModelBillions('Qwen3-30B-A3B'), 3, 'active parameters of a mixture of experts');
  assert.strictEqual(P.codeModelBillions('llama-3.1-70b'), 70);
  assert.strictEqual(P.codeModelBillions('tiny'), 0);
  assert.strictEqual(P.codePromptTier('qwen3-8b', 32768), 'compact');
  assert.strictEqual(P.codePromptTier('big-model', 4096), 'compact');
  assert.strictEqual(P.codePromptTier('qwen3-32b', 32768), 'full');
  assert.strictEqual(P.codePromptTier('unknown', 32768), 'full');
});

test('the environment block names folder, system, shell, git and date', () => {
  const env = P.codeEnvironment({ root: '/work/app', os: 'linux', shell: 'sh', git_branch: 'main', git_status: '2 changed files' }, new Date(2026, 9, 26));
  assert.ok(env.includes('/work/app') && env.includes('linux') && env.includes('sh') && env.includes('branch main') && env.includes('2 changed files'));
  assert.ok(env.includes('Monday 2026-10-26'));
  const win = P.codeEnvironment({ root: 'C:\\w', os: 'windows', shell: 'cmd.exe' });
  assert.ok(win.includes('cmd.exe') && win.includes('not ls'));
  assert.ok(P.codeEnvironment({ root: '/x', os: 'linux' }).includes('not a repository'));
});

test('the build prompt is complete for large models and short for small ones', () => {
  const base = { family: 'qwen', env: 'Environment\n- x', tools: ALL_TOOLS };
  const full = P.buildSystemPrompt({ ...base, agent: 'build', tier: 'full' });
  const compact = P.buildSystemPrompt({ ...base, agent: 'build', tier: 'compact' });
  for (const needle of ['Read a file before you change it', 'edit_file', 'patch', 'diagnostics', 'todo_write', 'ask_user', 'task', 'Safety', 'Memory:']) assert.ok(full.includes(needle), needle);
  assert.ok(compact.length < full.length * 0.55, `${compact.length} vs ${full.length}`);
  assert.ok(compact.includes('ONE tool call'));
  assert.ok(full.includes('Environment') && compact.includes('Environment'));
  assert.ok(full.includes('"name"') === false || full.includes('call grep'), 'a worked example is included');
  assert.ok(!/PLAN MODE/.test(full));
});

test('the plan prompt forbids changes', () => {
  for (const tier of ['full', 'compact']) {
    const p = P.buildSystemPrompt({ agent: 'plan', tier, family: 'generic', tools: ALL_TOOLS });
    assert.ok(/PLAN MODE/.test(p) && /read/i.test(p));
  }
});

test('tools that do not exist are not mentioned', () => {
  const few = new Set(['read_file', 'edit_file', 'grep', 'glob', 'bash']);
  const p = P.buildSystemPrompt({ agent: 'build', tier: 'full', family: 'generic', tools: few });
  assert.ok(!p.includes('todo_write') && !p.includes('ask_user') && !p.includes('diagnostics tool') && !p.includes('task with') && !p.includes('Memory:'));
  assert.ok(p.includes('edit_file') && p.includes('Read a file before you change it'));
});

test('every family has notes with an example call', () => {
  for (const family of ['qwen', 'llama', 'mistral', 'gemma', 'gptoss', 'generic']) {
    const notes = P.FAMILY_NOTES[family];
    assert.ok(notes && notes.length >= 1, family);
    assert.ok(notes.join(' ').includes('call '), `${family} needs an example`);
    const p = P.buildSystemPrompt({ agent: 'build', tier: 'compact', family, tools: ALL_TOOLS });
    assert.ok(p.includes(notes[0]));
  }
});

test('instructions and memory are capped and labelled', () => {
  const files = [{ file: 'AGENTS.md', text: 'Run npm test.' }, { file: '/up/CLAUDE.md', text: 'x'.repeat(20000) }];
  const block = P.instructionsBlock(files, 9000);
  assert.ok(block.includes('(AGENTS.md)') && block.includes('Run npm test.') && block.length < 9400);
  assert.strictEqual(P.instructionsBlock([]), '');
  const mem = P.memoryBlock({ workspace: ['tests: npm test'], global: ['likes short answers'] });
  assert.ok(mem.includes('About this workspace') && mem.includes('likes short answers'));
  assert.strictEqual(P.memoryBlock({ workspace: [], global: [] }), '');
  const p = P.buildSystemPrompt({ agent: 'build', tier: 'full', family: 'generic', tools: ALL_TOOLS, files, memory: { workspace: ['w'], global: [] } });
  assert.ok(p.indexOf('Instructions of this project') > p.indexOf('Safety') && p.indexOf('Memory (facts') > p.indexOf('Instructions of this project'));
});

test('the computer use rules are in the prompt only when the tools exist and the agent can act', () => {
  const withTools = new Set([...ALL_TOOLS, 'computer_screenshot', 'computer_click']);
  const full = P.buildSystemPrompt({ agent: 'build', tier: 'full', family: 'generic', tools: withTools });
  assert.ok(full.includes('Computer use (screen, mouse, keyboard)') && full.includes('ask_user') && full.includes('Take a screenshot first'));
  const compact = P.buildSystemPrompt({ agent: 'build', tier: 'compact', family: 'generic', tools: withTools });
  assert.ok(compact.includes('computer_screenshot shows the screen') && !compact.includes('Computer use (screen, mouse, keyboard)'));
  assert.ok(!P.buildSystemPrompt({ agent: 'build', tier: 'full', family: 'generic', tools: ALL_TOOLS }).includes('computer_'), 'off by default: no word about it');
  assert.ok(!P.buildSystemPrompt({ agent: 'plan', tier: 'full', family: 'generic', tools: withTools }).includes('computer_click'), 'the plan agent cannot act');
});

test('sub-agent prompts: explore reads only, general may change files', () => {
  const args = { description: 'find the parser', prompt: 'Where is the parser?' };
  const e = P.subAgentPrompt(args, 'explore');
  const g = P.subAgentPrompt(args, 'general');
  assert.ok(e.includes('Type: explore') && e.includes('only read and search') && e.includes('Where is the parser?') && e.includes('Job: find the parser'));
  assert.ok(g.includes('Type: general') && g.includes('change files'));
  assert.ok(P.subAgentPrompt(args, 'whatever').includes('Type: explore'), 'unknown types are read only');
});

test('the browser is explained only when the model has the browser tools', () => {
  const without = P.buildSystemPrompt({ agent: 'build', tier: 'full', family: 'generic', tools: ALL_TOOLS });
  assert.ok(!/browser_open/.test(without), 'no browser tools, no browser section');
  const withTools = new Set([...ALL_TOOLS, 'browser_open', 'browser_click']);
  for (const tier of ['full', 'compact']) {
    const p = P.buildSystemPrompt({ agent: 'build', tier, family: 'generic', tools: withTools });
    assert.ok(p.includes('browser_open') && p.includes('browser_snapshot') && p.includes('ref'), tier);
  }
});

test('all page scripts together are valid JavaScript', () => {
  const rs = fs.readFileSync(path.join(dir, '..', 'webui.rs'), 'utf8');
  const files = [...rs.matchAll(/include_str!\("webui\/([\w-]+\.js)"\)/g)].map((m) => m[1]).filter((f) => f !== 'runner.js');
  assert.ok(files.includes('codeprompts.js') && files.indexOf('codeprompts.js') < files.indexOf('code.js'), 'codeprompts.js loads before code.js');
  const all = files.map((f) => fs.readFileSync(path.join(dir, f), 'utf8')).join('');
  // syntax only: the function is built, never called
  new Function(`(() => {\n'use strict';\n${all}\n})();`);
});

test('a personality changes the style line and nothing else', () => {
  const base = { agent: 'build', tier: 'full', family: 'qwen', tools: ALL_TOOLS };
  const plain = P.buildSystemPrompt(base);
  const concise = P.buildSystemPrompt({ ...base, personality: 'concise' });
  assert.ok(concise.includes(P.PERSONALITIES.concise.note) && !plain.includes('Style:'));
  const squeeze = (t) => t.split('\n').filter((l) => l.trim() !== '').join('\n');
  assert.strictEqual(squeeze(concise.split(P.PERSONALITIES.concise.note).join('')), squeeze(plain), 'only the style line differs');
  assert.strictEqual(P.personalityNote('nonsense'), '');
  assert.strictEqual(P.buildSystemPrompt({ ...base, personality: 'nonsense' }), plain, 'an unknown name is the default');
  for (const [name, p] of Object.entries(P.PERSONALITIES)) assert.ok(name === 'default' || p.note.startsWith('Style:'), name);
});

test('the self-review asks once when clean and again only for real problems', () => {
  const clean = { review: 'Review (diff): 1 file(s), +1/-0 lines, 0 finding(s)', problems: [] };
  assert.ok(P.reviewFeedback(clean, 1).includes('Re-read the diff'));
  assert.strictEqual(P.reviewFeedback(clean, 2), null);
  const bad = { review: '- [low] x', problems: [{ path: 'a.py', output: 'Problems found (python): SyntaxError' }] };
  const text = P.reviewFeedback(bad, 2);
  assert.ok(text.includes('a.py') && text.includes('Fix every real problem'));
  assert.ok(P.reviewFeedback({ review: '- [high] secret', problems: [] }, 2));
  assert.ok(P.isCleanDiagnostic('No problems found (node --check).'));
  assert.ok(P.isCleanDiagnostic('The check (cargo check) is not available here: the tool is not installed'));
  assert.ok(!P.isCleanDiagnostic('Problems found (node --check, exit code 1): x'));
});

test('insights count tools and edits of saved sessions', () => {
  const call = (n) => ({ function: { name: n } });
  const ins = P.sessionInsights([
    { messages: [{ role: 'user', content: 'a' }, { role: 'assistant', tool_calls: [call('read_file'), call('edit_file')] }, { role: 'tool' }, { role: 'assistant', content: 'done' }] },
    { messages: [{ role: 'user', content: 'b' }, { role: 'assistant', tool_calls: [call('read_file')] }] },
  ]);
  assert.deepStrictEqual([ins.sessions, ins.userMessages, ins.steps, ins.edits], [2, 2, 3, 1]);
  assert.deepStrictEqual(ins.top[0], ['read_file', 2]);
  assert.strictEqual(P.sessionInsights(null).sessions, 0);
});

test('search hits become readable text', () => {
  assert.ok(P.searchHitsText([], 'lexer').includes('No stored session'));
  const t = P.searchHitsText([{ title: 'Fix the lexer', kind: 'code', updated: 0, snippet: '...tokenizer bug...' }], 'lexer');
  assert.ok(t.includes('1 stored session for') && t.includes('Fix the lexer') && t.includes('tokenizer bug'));
});

console.log(`${n} tests passed`);
