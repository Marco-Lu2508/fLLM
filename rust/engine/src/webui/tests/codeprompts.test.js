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
return { codeModelFamily, codeModelBillions, codePromptTier, codeEnvironment, memoryBlock, instructionsBlock, buildSystemPrompt, subAgentPrompt, FAMILY_NOTES };`)();

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

test('sub-agent prompts: explore reads only, general may change files', () => {
  const args = { description: 'find the parser', prompt: 'Where is the parser?' };
  const e = P.subAgentPrompt(args, 'explore');
  const g = P.subAgentPrompt(args, 'general');
  assert.ok(e.includes('Type: explore') && e.includes('only read and search') && e.includes('Where is the parser?') && e.includes('Job: find the parser'));
  assert.ok(g.includes('Type: general') && g.includes('change files'));
  assert.ok(P.subAgentPrompt(args, 'whatever').includes('Type: explore'), 'unknown types are read only');
});

test('all page scripts together are valid JavaScript', () => {
  const rs = fs.readFileSync(path.join(dir, '..', 'webui.rs'), 'utf8');
  const files = [...rs.matchAll(/include_str!\("webui\/([\w-]+\.js)"\)/g)].map((m) => m[1]).filter((f) => f !== 'runner.js');
  assert.ok(files.includes('codeprompts.js') && files.indexOf('codeprompts.js') < files.indexOf('code.js'), 'codeprompts.js loads before code.js');
  const all = files.map((f) => fs.readFileSync(path.join(dir, f), 'utf8')).join('');
  // syntax only: the function is built, never called
  new Function(`(() => {\n'use strict';\n${all}\n})();`);
});

console.log(`${n} tests passed`);
