// Tests for the system prompts (prompts.js): families, sizes, assembly order, titles; and that every script of the page parses together.
// Run from the repository root:  node rust/engine/src/webui/tests/prompts.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, '..');
const read = (f) => fs.readFileSync(path.join(dir, f), 'utf8');
const pure = (file) => {
  const src = read(file);
  const b = src.indexOf('// @pure-begin');
  const e = src.indexOf('// @pure-end');
  assert(b >= 0 && e > b, `pure block markers not found in ${file}`);
  return src.slice(b, e);
};
const L = new Function(`const SP_KEY = 'x';\n${pure('prompts.js')}\n${pure('projects.js')}
return { SP_LIMITS, spFamily, spSize, spBuild, spResolve, spFill, spNow, spClock, spTitleMessages, spCleanTitle, composeSystem, PRJ_LIMITS };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };
const NOW = new Date(2026, 9, 26, 14, 5).getTime();
const base = { size: 'full', family: 'generic', now: NOW, locale: 'fr', local: true, modelName: 'Some-Model-7B', thinking: false, tools: {} };

test('the family comes from the name, the architecture and the chat template', () => {
  const f = (name, extra = {}) => L.spFamily({ name, ...extra });
  assert.strictEqual(f('Qwen3-4B-Instruct'), 'qwen');
  assert.strictEqual(f('unsloth/QwQ-32B'), 'qwen');
  assert.strictEqual(f('Llama-3.2-3B'), 'llama');
  assert.strictEqual(f('Mistral-7B-Instruct'), 'mistral');
  assert.strictEqual(f('Ministral-8B'), 'mistral');
  assert.strictEqual(f('gemma-3-4b-it'), 'gemma');
  assert.strictEqual(f('gpt-oss-20b'), 'gptoss');
  assert.strictEqual(f('DeepSeek-R1-Distill-Qwen-7B'), 'deepseek', 'the distilled model is a DeepSeek model first');
  assert.strictEqual(f('Phi-4-mini'), 'phi');
  assert.strictEqual(f('SmolLM3-3B'), 'smollm');
  assert.strictEqual(f('local.gguf', { chat_template: '<start_of_turn>user' }), 'gemma');
  assert.strictEqual(f('x', { chat_template: '... [INST] ...' }), 'mistral');
  assert.strictEqual(f('x', { chat_template: '<|start|>assistant<|channel|>final' }), 'gptoss');
  assert.strictEqual(f('x', { chat_template: '<' + String.fromCharCode(0xFF5C) + 'User' + String.fromCharCode(0xFF5C) + '>' }), 'deepseek');
  assert.strictEqual(f('mystery'), 'generic');
  assert.strictEqual(L.spFamily(null), 'generic');
});

test('small models and small contexts get the compact prompt', () => {
  assert.strictEqual(L.spSize({ max_context: 4096, parameters: 8e9 }), 'compact');
  assert.strictEqual(L.spSize({ max_context: 32768, parameters: 1.7e9 }), 'compact');
  assert.strictEqual(L.spSize({ max_context: 32768, parameters: 8e9 }), 'full');
  assert.strictEqual(L.spSize({ max_context: 32768 }), 'full', 'unknown size: judged by the context');
});

test('the default prompt: time, language, honesty, locality; compact is much shorter', () => {
  const full = L.spBuild(base);
  const compact = L.spBuild({ ...base, size: 'compact' });
  assert(full.includes('October 26, 2026') && full.includes('14:05') && full.includes('UTC'), full);
  assert(full.includes('French') && compact.includes('French'));
  assert(/never invent/i.test(full) && /never invent/i.test(compact));
  assert(/running locally on the user's own computer/.test(full));
  assert(/runs? locally|running locally/.test(compact));
  assert(full.includes('Some-Model-7B'));
  assert(compact.length < 600 && compact.length < full.length / 2, `${compact.length} vs ${full.length}`);
});

test('the prompt never claims abilities that are not offered', () => {
  const none = L.spBuild(base);
  assert(!/web_search|web_fetch|memory tool|search_chats/.test(none));
  assert(/no tools in this chat/.test(none));
  const web = L.spBuild({ ...base, tools: { web: true } });
  assert(web.includes('web_search') && !web.includes('memory (') && !web.includes('search_chats'));
  assert(!/no tools in this chat/.test(web));
  assert(/only for current|current or changing/i.test(web));
  assert(web.includes('[n]'));
  const all = L.spBuild({ ...base, tools: { web: true, memory: true, chats: true } });
  assert(all.includes('web_search') && all.includes('Memory tool') && all.includes('search_chats') && /secrets/.test(all));
  assert(all.length < 3500, `full prompt with all tools: ${all.length}`);
  const remote = L.spBuild({ ...base, local: false });
  assert(!/locally/.test(remote));
});

test('family adjustments are added, and thinking gets a hint where the family has none', () => {
  assert(L.spBuild({ ...base, family: 'smollm' }).includes('very small model'));
  assert(L.spBuild({ ...base, family: 'gptoss' }).includes('final answer short'));
  assert(L.spBuild({ ...base, family: 'llama', thinking: true }).includes('Keep your reasoning short.'));
  assert(!L.spBuild({ ...base, family: 'qwen', thinking: true }).includes('Keep your reasoning short.'), 'Qwen has its own line');
  for (const fam of ['qwen', 'llama', 'mistral', 'gemma', 'gptoss', 'deepseek', 'phi', 'smollm']) {
    assert(L.spBuild({ ...base, family: fam }).length > L.spBuild(base).length, fam);
  }
});

test('user choices: append, override, assistant on top, placeholders, empty override ignored', () => {
  const o = { now: NOW, locale: 'fr' };
  const gen = 'DEFAULT';
  assert.strictEqual(L.spResolve(gen, { mode: 'default', text: 'ignored' }, null, o), 'DEFAULT');
  assert.strictEqual(L.spResolve(gen, { mode: 'append', text: 'Be formal.' }, null, o), 'DEFAULT\n\nBe formal.');
  assert.strictEqual(L.spResolve(gen, { mode: 'override', text: 'Only this.' }, null, o), 'Only this.');
  assert.strictEqual(L.spResolve(gen, { mode: 'override', text: '   ' }, null, o), 'DEFAULT');
  assert.strictEqual(L.spResolve(gen, { mode: 'append', text: 'G' }, { mode: 'append', text: 'A' }, o), 'DEFAULT\n\nG\n\nA');
  assert.strictEqual(L.spResolve(gen, { mode: 'append', text: 'G' }, { mode: 'override', text: 'A' }, o), 'A');
  const filled = L.spResolve(gen, { mode: 'override', text: 'Reply in {{language}}. Now: {{time}}. Day: {{date}}.' }, null, o);
  assert(filled.includes('French') && filled.includes('14:05') && /Day: \w+day, October 26, 2026\./.test(filled), filled);
  assert(L.spResolve(gen, { mode: 'append', text: 'x'.repeat(10000) }, null, o).length < 6100);
});

test('order of the system message: default prompt, personalization, project, assistant', () => {
  const personal = { useAbout: true, useMemory: true, about: 'ABOUT-ME', memories: [{ id: 'a', text: 'MEMORY-FACT', createdAt: 1 }] };
  const out = L.composeSystem({ defaults: 'DEFAULT-PROMPT', personal, project: { name: 'P', instructions: 'PROJECT-TEXT' }, base: 'ASSISTANT-TEXT' });
  const at = (s) => out.indexOf(s);
  assert(at('DEFAULT-PROMPT') === 0 && at('DEFAULT-PROMPT') < at('ABOUT-ME') && at('ABOUT-ME') < at('MEMORY-FACT') && at('MEMORY-FACT') < at('PROJECT-TEXT') && at('PROJECT-TEXT') < at('ASSISTANT-TEXT'), out);
  const noMem = L.composeSystem({ defaults: 'D', personal, project: null, base: '', memoryOff: true });
  assert(!noMem.includes('MEMORY-FACT') && noMem.includes('ABOUT-ME'));
  assert.strictEqual(L.composeSystem({ defaults: '', personal: null, project: null, base: '' }), '');
});

test('the memory block is capped smaller for compact prompts', () => {
  const personal = { useAbout: false, useMemory: true, memories: Array.from({ length: 50 }, (_, i) => ({ id: `m${i}`, text: 'x'.repeat(60) + i, createdAt: i })) };
  const big = L.composeSystem({ defaults: '', personal, project: null, base: '' });
  const small = L.composeSystem({ defaults: '', personal, project: null, base: '', memoryCap: L.PRJ_LIMITS.memoryPromptCompact });
  assert(small.length < big.length && small.length < L.PRJ_LIMITS.memoryPromptCompact + 200, `${small.length}`);
  assert(small.includes('49'), 'the newest stays');
});

test('the clock is frozen for 30 minutes so the cached prefix stays valid', () => {
  const h = {};
  const t0 = L.spClock(h, 1000);
  assert.strictEqual(L.spClock(h, 1000 + 29 * 60000), t0);
  assert.strictEqual(L.spClock(h, 1000 + 31 * 60000), 1000 + 31 * 60000);
});

test('titles: short robust prompt, thinking blocks and echoes are cleaned', () => {
  const m = L.spTitleMessages('q'.repeat(1000), 'a'.repeat(1000));
  assert.strictEqual(m.length, 2);
  assert(m[0].content.length < 400 && /never mention these instructions/i.test(m[0].content));
  assert(m[1].content.length < 800);
  assert.strictEqual(L.spCleanTitle('<think>hmm, what title</think>Trip planning for Lisbon.'), 'Trip planning for Lisbon');
  assert.strictEqual(L.spCleanTitle('<think>never closed'), null);
  assert.strictEqual(L.spCleanTitle('Title: "Rust lifetimes"\nbecause it is about them'), 'Rust lifetimes');
  assert.strictEqual(L.spCleanTitle('**Pasta recipe**'), 'Pasta recipe');
  assert.strictEqual(L.spCleanTitle('My system prompt says I am a helpful assistant'), null);
  assert.strictEqual(L.spCleanTitle('x'.repeat(80)), null);
  assert.strictEqual(L.spCleanTitle(''), null);
  assert.strictEqual(L.spCleanTitle(undefined), null);
});

test('every script of the page parses together, with the new files registered', () => {
  const rs = read('../webui.rs');
  const files = [...rs.matchAll(/include_str!\("webui\/([a-z-]+\.js)"\)/g)].map((m) => m[1]).filter((f) => f !== 'runner.js');
  assert(files.includes('prompts.js') && files.includes('tools.js'));
  assert(files.indexOf('prompts.js') === 1 && files.indexOf('tools.js') > files.indexOf('research.js'));
  assert.strictEqual(Number(/SCRIPT: \[&str; (\d+)\]/.exec(rs)[1]), files.length, 'the array length matches the list');
  const source = `(() => {\n'use strict';\n${files.map(read).join('')}\n})();`;
  new Function(source); // compiles only
});

console.log(`${n} tests passed`);
