#!/usr/bin/env node
// Row 77 fixtures: run the page's own `runSubAgent` (code.js) in a vm with the world stubbed, over
// the cases below, and write what it decides — the refusals, the type, the tools, the system
// text, the first message, the limits, the status and the reply — for the Rust port to match.
// Also the pure helpers it leans on (`subAgentPrompt`, the token estimates), the tool definitions
// of `task` and `create_agent` and the way the `task` tool lists the custom agents.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const lines = (name) => fs.readFileSync(path.join(WEBUI, name), 'utf8').split(/\r?\n/);
const slice = (l, from, to) => l.slice(from - 1, to).join('\n');
const code = lines('code.js');
const prompts = lines('codeprompts.js');
const chat = lines('chat.js');

// a slice is found by the first line of its text, never by a number remembered from a notes file
const at = (l, start, endText) => {
  const from = l.findIndex((s) => s.includes(start)) + 1;
  if (from === 0) throw new Error(`no line with ${start}`);
  let to = from;
  if (endText) {
    to = l.findIndex((s, i) => i >= from - 1 && s.includes(endText)) + 1;
    if (to === 0) throw new Error(`no line with ${endText}`);
  }
  return slice(l, from, to);
};
const check = (text, name) => { try { new vm.Script(text); } catch (e) { throw new Error(`${name}: ${e.message}`); } return text; };

const PURE_CODE = slice(code, 19, 560);
const PURE_PROMPTS = slice(prompts, 7, 309);
const CLIP = at(code, 'const clipOutput = (text) => {', '    };');
const AGENT_LOOKUP = [at(code, 'const customAgent = (name)'), at(code, 'const agentNames = () =>')].join('\n');
const ESTIMATES = [
  at(chat, 'const estimateTokens = (text) =>'),
  at(code, 'const contentTokens = (c) =>'),
  at(code, 'const estimateMessages = (messages) =>'),
  at(code, 'const fixedTokens = (ctx) =>'),
].join('\n');
const RUN_SUB_AGENT = check(at(code, 'async function runSubAgent(item, args, parent) {', '      return clipOutput(item.status === \'ok\'') + '\n    }', 'runSubAgent');
const CLIENT_TOOLS = check(`(() => {\n${at(code, 'const CLIENT_TOOLS = {', '    };').replace('const CLIENT_TOOLS =', 'return')}\n})()`, 'CLIENT_TOOLS');
const TOOL_LIST_FOR = at(code, 'const toolListFor = (plan) =>', '      });');

const harness = `
  const CHILD_MAX_STEPS = 16;
  const MAX_PARALLEL_CHILDREN = 4;
  const code = { perm: 'ask', agents: [] };
  let __ctxMax = 8192;
  const ctxMax = () => __ctxMax;
  const paintItem = () => {};
  const paintSoon = () => {};
  let __ran = null;
  let __loop = async () => 'REPORT';
  const runLoop = async (ctx, status) => {
    __ran = {
      system: ctx.system, tools: ctx.tools.map((t) => t.function.name), messages: ctx.messages,
      depth: ctx.depth, readOnly: ctx.readOnly, maxSteps: ctx.maxSteps, model: ctx.model,
      budget: ctx.budget, child: ctx.child,
    };
    if (status) status('step 1');
    return __loop(ctx);
  };
  ${ESTIMATES.replace('const fixedTokens', 'const __fixed')}
  ${PURE_CODE}
  ${PURE_PROMPTS}
  ${CLIP}
  ${AGENT_LOOKUP}
  ${RUN_SUB_AGENT}
`;

const TOOLS = ['read_file', 'grep', 'glob', 'list_dir', 'write_file', 'edit_file', 'bash', 'memory', 'todo_write', 'task'];
const tool = (name) => ({ type: 'function', function: { name } });
const REVIEWER = { name: 'code-reviewer', description: 'Reads a diff and reports problems.', tools: ['read_file', 'grep'], model: '', body: 'You review code.\nBe strict.', max_steps: 0, scope: 'workspace' };
const FIXER = { name: 'fixer', description: '', tools: [], model: 'big-model', body: 'You fix bugs.', max_steps: 5, scope: 'user' };
const WRITER = { name: 'test-writer', description: 'Writes tests.', tools: ['read_file', 'write_file', 'bash'], model: '', body: '', scope: 'workspace' };

const CASES = [
  { name: 'explore by default', args: { description: 'Find the parser', prompt: 'Where is the SSE parser?' } },
  { name: 'explore asked for', args: { description: 'x', prompt: 'p', agent: 'explore' } },
  { name: 'general', args: { description: 'Do it', prompt: 'Change the thing.', agent: 'general' } },
  { name: 'general in plan mode stays explore', perm: 'plan', args: { description: 'Do it', prompt: 'Change the thing.', agent: 'general' } },
  { name: 'a padded general is only explore', args: { description: 'Do it', prompt: 'Change the thing.', agent: ' general ' } },
  { name: 'a padded explore', args: { description: 'x', prompt: 'p', agent: ' explore' } },
  { name: 'a zero agent is no agent', args: { description: 'x', prompt: 'p', agent: 0 } },
  { name: 'mode full is general', args: { description: 'Do it', prompt: 'Change the thing.', mode: 'full' } },
  { name: 'mode full in plan mode', perm: 'plan', args: { description: 'Do it', prompt: 'Change the thing.', mode: 'full' } },
  { name: 'no prompt', args: { description: 'x' } },
  { name: 'blank prompt', args: { description: 'x', prompt: '   \n ' } },
  { name: 'a sub-agent starts none', depth: 1, args: { description: 'x', prompt: 'p' } },
  { name: 'no prompt beats the depth', depth: 1, args: { description: 'x' } },
  { name: 'unknown agent', args: { description: 'x', prompt: 'p', agent: 'ghost' } },
  { name: 'unknown agent lists the custom ones', agents: [REVIEWER, FIXER], args: { description: 'x', prompt: 'p', agent: 'ghost' } },
  { name: 'agent name is trimmed', agents: [REVIEWER], args: { description: 'x', prompt: 'p', agent: '  code-reviewer ' } },
  { name: 'a custom agent with read tools only', agents: [REVIEWER], args: { description: 'Review it', prompt: 'Look at the diff.', agent: 'code-reviewer' } },
  { name: 'a custom agent with no tool list', agents: [FIXER], args: { description: 'Fix it', prompt: 'Bug in parser.', agent: 'fixer' } },
  { name: 'a custom agent with write tools', agents: [WRITER], args: { description: 'Tests', prompt: 'Cover the parser.', agent: 'test-writer' } },
  { name: 'a custom agent whose only tool is bash', agents: [{ name: 'solo', description: '', tools: ['bash'], model: '', body: 'b', scope: 'user' }], args: { description: 'x', prompt: 'p', agent: 'solo' } },
  { name: 'a custom agent whose only tool is edit_file', agents: [{ name: 'solo', description: '', tools: ['edit_file'], model: '', body: 'b', scope: 'user' }], args: { description: 'x', prompt: 'p', agent: 'solo' } },
  { name: 'a custom agent whose only tool is write_file', agents: [{ name: 'solo', description: '', tools: ['write_file'], model: '', body: 'b', scope: 'user' }], args: { description: 'x', prompt: 'p', agent: 'solo' } },
  { name: 'a custom agent whose only tool is multi_edit', agents: [{ name: 'solo', description: '', tools: ['multi_edit'], model: '', body: 'b', scope: 'user' }], args: { description: 'x', prompt: 'p', agent: 'solo' } },
  { name: 'a custom agent whose only tool is patch', agents: [{ name: 'solo', description: '', tools: ['patch'], model: '', body: 'b', scope: 'user' }], args: { description: 'x', prompt: 'p', agent: 'solo' } },
  { name: 'a custom agent whose only tool is bash_kill', agents: [{ name: 'solo', description: '', tools: ['bash_kill'], model: '', body: 'b', scope: 'user' }], args: { description: 'x', prompt: 'p', agent: 'solo' } },
  { name: 'a custom agent whose only tool is memory', agents: [{ name: 'solo', description: '', tools: ['memory'], model: '', body: 'b', scope: 'user' }], args: { description: 'x', prompt: 'p', agent: 'solo' } },
  { name: 'a custom agent whose only tool is glob', agents: [{ name: 'solo', description: '', tools: ['glob'], model: '', body: 'b', scope: 'user' }], args: { description: 'x', prompt: 'p', agent: 'solo' } },
  { name: 'a custom agent with an empty body and no description', agents: [WRITER], args: { description: 'Tests', prompt: 'Cover it.', agent: 'test-writer' } },
  { name: 'a custom agent in plan mode keeps its tools', perm: 'plan', agents: [WRITER], args: { description: 'Tests', prompt: 'Cover it.', agent: 'test-writer' } },
  { name: 'a long description is cut at eighty', args: { description: 'd'.repeat(120), prompt: 'p' } },
  { name: 'no description', args: { prompt: 'p' } },
  { name: 'a number as the prompt', args: { description: 'n', prompt: 42 } },
  { name: 'the prompt keeps its spaces in the first message', args: { description: 'n', prompt: '  indented\n  text  ' } },
  { name: 'the context sets the budget', ctx: 100000, args: { description: 'n', prompt: 'p' } },
  { name: 'an empty report', loop: { report: '' }, args: { description: 'Find it', prompt: 'p' } },
  { name: 'a long report is clipped', ctx: 8192, loop: { report: 'r'.repeat(40000) }, args: { description: 'Find it', prompt: 'p' } },
  { name: 'a report of several lines', loop: { report: 'Found it.\nsrc/api.rs:10' }, args: { description: 'Find it', prompt: 'p' } },
  { name: 'the loop fails', loop: { throws: { name: 'Error', message: 'The server answered with error 500' } }, args: { description: 'Find it', prompt: 'p' } },
  { name: 'the loop is stopped', stopped: true, loop: { throws: { name: 'AbortError', message: 'aborted' } }, args: { description: 'Find it', prompt: 'p' } },
  { name: 'the loop ends while stopped', stopped: true, loop: { report: 'partial' }, args: { description: 'Find it', prompt: 'p' } },
  { name: 'a stopped loop without a report', stopped: true, loop: { report: '' }, args: { description: 'Find it', prompt: 'p' } },
];

// ---- the helpers the port also has to match --------------------------------------------------
const ESTIMATE_TEXTS = ['', 'a', 'abcd', 'a'.repeat(36), 'a'.repeat(37), 'äöü€', '😀😀', JSON.stringify({ a: [1, 2, 3] })];
const ESTIMATE_MESSAGES = [
  [],
  [{ role: 'user', content: 'hello' }],
  [{ role: 'user', content: 'x'.repeat(100) }, { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'grep', arguments: '{"pattern":"a"}' } }] }, { role: 'tool', tool_call_id: 'c1', content: 'out' }],
];

async function main() {
  const context = { console, JSON, Math, Set, Map, Array, Object, String, Number };
  vm.createContext(context);
  vm.runInContext(harness, context, { filename: 'page-slices.js' });
  const out = [];
  for (const c of CASES) {
    vm.runInContext(
      `code.perm = ${JSON.stringify(c.perm || 'ask')}; code.agents = ${JSON.stringify(c.agents || [])}; __ctxMax = ${c.ctx || 8192}; __ran = null;
       __loop = async () => { ${c.loop && c.loop.throws ? `const e = new Error(${JSON.stringify(c.loop.throws.message)}); e.name = ${JSON.stringify(c.loop.throws.name)}; throw e;` : `return ${JSON.stringify(c.loop && 'report' in c.loop ? c.loop.report : 'REPORT')};`} };`,
      context,
    );
    const item = { kind: 'tool', name: 'task' };
    context.__item = item;
    context.__args = JSON.parse(JSON.stringify(c.args));
    context.__parent = {
      depth: c.depth || 0, system: 'SYSTEM', tools: TOOLS.map(tool), signal: null,
      isStopped: () => Boolean(c.stopped),
    };
    let reply = null;
    let error = null;
    try {
      reply = await vm.runInContext('runSubAgent(__item, __args, __parent)', context);
    } catch (e) {
      error = e.message;
    }
    const ran = vm.runInContext('__ran', context);
    out.push({
      ...c,
      parentTools: TOOLS, ctx: c.ctx || 8192, perm: c.perm || 'ask', agents: c.agents || [], depth: c.depth || 0,
      error, reply,
      item: { status: item.status, output: item.output, progress: item.progress ?? null, agentType: item.agentType ?? null, children: item.children ? item.children.length : null },
      ran,
      argsAfter: context.__args,
    });
  }

  const SIGNATURES = [
    ['read_file', { path: 'a.rs', start: 1 }], ['read_file', { start: 1, path: 'a.rs' }], ['grep', {}], ['grep', null],
    ['x', { b: { d: 1, c: [3, { z: 1, y: 2 }] }, a: 'é"\n' }], ['x', { a: 1.5, b: -0, c: 1e21, d: true, e: null }], ['x', []], ['x', [1, 'a']],
    ['bash', { command: 'ls' }], ['x', { 10: 1, 9: 2, a: 3, '01': 4, 4294967295: 5, 4294967294: 6 }], ['x', { n: 1e-7, m: 0.000001, l: 1.5e20, k: 1e21, j: 5e-324, i: 0.1, h: 100 }],
    ['x', { s: '\u2028\t\u0001', '': 1, '\ud83d\ude00': 2, '\uffff': 3 }],
  ];
  const signatures = SIGNATURES.map(([name, args]) => ({ name, args, signature: vm.runInContext(`callSignature(${JSON.stringify(name)}, ${JSON.stringify(args)})`, context) }));
  const REPEATS = [[[], 'a', 3], [['a'], 'a', 3], [['a', 'a'], 'a', 3], [['b', 'a', 'a'], 'a', 3], [['a', 'a', 'b'], 'a', 3], [['a', 'a'], 'b', 3], [['a', 'a', 'a', 'a'], 'a', 3], [['a'], 'a', 2], [[], 'a', 2], [['a', 'b', 'a'], 'a', 4]];
  const repeats = REPEATS.map(([history, signature, times]) => ({ history, signature, times, repeating: vm.runInContext(`isRepeating(${JSON.stringify(history)}, ${JSON.stringify(signature)}, ${times})`, context) }));
  const guardless = vm.runInContext(`(() => { ${at(code, 'const NO_LOOP_GUARD = new Set(').replace('const NO_LOOP_GUARD', 'NO_LOOP_GUARD_X')} return [...NO_LOOP_GUARD_X]; })()`, context);

  const estimates = ESTIMATE_TEXTS.map((text) => ({ text, tokens: vm.runInContext(`estimateTokens(${JSON.stringify(text)})`, context) }));
  const messages = ESTIMATE_MESSAGES.map((m) => ({ messages: m, tokens: vm.runInContext(`estimateMessages(${JSON.stringify(m)})`, context) }));
  const prompts_ = [
    ['Find the parser', 'Where is the SSE parser?', 'explore'],
    ['Do it', 'Change the thing.', 'general'],
  ].map(([description, prompt, type]) => ({ description, prompt, type, text: vm.runInContext(`subAgentPrompt(${JSON.stringify({ description, prompt })}, ${JSON.stringify(type)})`, context) }));

  // the definitions of the two tools, and the task tool as the page lists it for a folder with agents
  const defs = vm.runInNewContext(CLIENT_TOOLS, {});
  const toolDefs = { task: defs.task, create_agent: defs.create_agent };
  const listContext = { code: { registry: new Map(), agents: [REVIEWER, FIXER], perm: 'ask' }, JSON };
  vm.createContext(listContext);
  vm.runInContext(
    `const readOnlyTool = (e) => e.permission === 'read' && e.source !== 'client';
     ${AGENT_LOOKUP}
     code.registry.set('task', { name: 'task', permission: 'read', source: 'client', def: { description: ${JSON.stringify(defs.task.description)}, parameters: { type: 'object', properties: ${JSON.stringify(defs.task.properties)}, required: ${JSON.stringify(defs.task.required)} } } });
     ${TOOL_LIST_FOR}
     globalThis.__listed = toolListFor(false);`,
    listContext,
  );
  const listed = listContext.__listed.find((t) => t.function.name === 'task');
  const taskWithAgents = JSON.parse(JSON.stringify(listed));

  fs.writeFileSync(
    path.join(__dirname, 'subagent_fixtures.json'),
    JSON.stringify({ cases: out, signatures, repeats, guardless, estimates, messages, prompts: prompts_, toolDefs, taskWithAgents, agents: [REVIEWER, FIXER] }, null, 1),
  );
  const bad = out.filter((c) => c.error);
  console.log(`cases: ${out.length}, errors: ${bad.length}`);
  for (const c of bad) console.log('ERROR', c.name, '->', c.error);
  if (bad.length) process.exit(1);
  console.log('FIXTURES WRITTEN');
}

main();
