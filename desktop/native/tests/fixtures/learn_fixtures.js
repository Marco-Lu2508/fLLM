#!/usr/bin/env node
// Row 88 fixtures: the pure block of the page's learn.js (`@pure-begin` .. `@pure-end`) run in a vm
// over the cases below; the Rust port in src/learn.rs must give the same answers.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const src = fs.readFileSync(path.join(WEBUI, 'learn.js'), 'utf8');
const a = src.indexOf('// @pure-begin');
const b = src.indexOf('// @pure-end');
if (a < 0 || b < 0) throw new Error('the pure block of learn.js is gone');
const context = { console, JSON, Math, Set, Map, Array, Object, String, Number };
vm.createContext(context);
vm.runInContext(src.slice(a, b), context, { filename: 'learn.pure.js' });
const run = (expr, vars) => { Object.assign(context, vars); return vm.runInContext(expr, context); };

const call = (name, args) => ({ id: 'c1', type: 'function', function: { name, arguments: typeof args === 'string' ? args : JSON.stringify(args) } });
const asst = (...calls) => ({ role: 'assistant', content: '', tool_calls: calls });

const SKILL_CASES = [
  { name: 'none', messages: [] },
  { name: 'a read', messages: [asst(call('skill', { action: 'read', name: 'deploy' }))] },
  { name: 'twice and in order', messages: [asst(call('skill', { action: 'read', name: 'zeta' }), call('skill', { action: 'read', name: 'deploy' })), asst(call('skill', { action: 'read', name: 'zeta' }))] },
  { name: 'list is no read', messages: [asst(call('skill', { action: 'list' }))] },
  { name: 'another tool', messages: [asst(call('grep', { action: 'read', name: 'x' }))] },
  { name: 'bad name', messages: [asst(call('skill', { action: 'read', name: 'Bad Name' })), asst(call('skill', { action: 'read', name: '' })), asst(call('skill', { action: 'read' }))] },
  { name: 'a name with 64 and 65 characters', messages: [asst(call('skill', { action: 'read', name: 'a'.repeat(64) })), asst(call('skill', { action: 'read', name: 'b'.repeat(65) }))] },
  { name: 'a number is a name', messages: [asst(call('skill', { action: 'read', name: 5 })), asst(call('skill', { action: 'read', name: 0 }))] },
  { name: 'arguments that are no json', messages: [asst(call('skill', '{not json'))] },
  { name: 'arguments that are no object', messages: [asst(call('skill', '[1]')), asst(call('skill', '"read"')), asst(call('skill', 'null'))] },
  { name: 'dash and underscore', messages: [asst(call('skill', { action: 'read', name: 'a_b-c' })), asst(call('skill', { action: 'read', name: '-lead' }))] },
  { name: 'a message with no calls', messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'x', tool_calls: [] }] },
];

const long = (n) => 'é'.repeat(n);
const RUN_CASES = [
  { name: 'empty', messages: [], from: 0 },
  { name: 'roles', messages: [{ role: 'system', content: 's' }, { role: 'user', content: 'u' }, { role: 'assistant', content: 'a' }, { role: 'tool', content: 't', tool_call_id: 'x' }, { role: 'developer', content: 'd' }], from: 0 },
  { name: 'from skips', messages: [{ role: 'user', content: 'old' }, { role: 'user', content: 'new' }, { role: 'assistant', content: 'ans' }], from: 1 },
  { name: 'from past the end', messages: [{ role: 'user', content: 'x' }], from: 5 },
  { name: 'text cut at 3000', messages: [{ role: 'user', content: long(3001) }, { role: 'user', content: 'a'.repeat(3000) }], from: 0 },
  { name: 'calls', messages: [asst(call('grep', { pattern: 'p' }), call('skill', '{"action":"read","name":"x"}'))], from: 0 },
  { name: 'arguments cut at 800', messages: [asst(call('bash', 'x'.repeat(900)))], from: 0 },
  { name: 'no arguments', messages: [{ role: 'assistant', content: '', tool_calls: [{ id: 'c', function: { name: 'n' } }] }], from: 0 },
  { name: 'the last 300', messages: Array.from({ length: 305 }, (_, i) => ({ role: 'user', content: `m${i}` })), from: 0 },
  { name: 'the last 3 of 5', messages: Array.from({ length: 5 }, (_, i) => ({ role: 'user', content: `m${i}` })), from: 0, max: 3 },
  { name: 'from counts before the filter', messages: [{ role: 'system', content: 's' }, { role: 'user', content: 'u1' }, { role: 'user', content: 'u2' }], from: 2 },
  { name: 'empty calls list has no key', messages: [{ role: 'assistant', content: 'x', tool_calls: [] }], from: 0 },
];

const cases = [];
for (const c of SKILL_CASES) cases.push({ kind: 'skills', ...c, result: run(`lnSkillsRead(__m)`, { __m: JSON.parse(JSON.stringify(c.messages)) }) });
for (const c of RUN_CASES) {
  const max = c.max || 300;
  cases.push({ kind: 'run', ...c, max, result: run(`lnRunMessages(__m, __f, __max, 3000)`, { __m: JSON.parse(JSON.stringify(c.messages)), __f: c.from, __max: max }) });
}
const failed = [[false, false], [true, false], [false, true], [true, true]].map(([error, hitLimit]) => ({ error, hitLimit, result: run('lnRunFailed({ error: __e, hitLimit: __h })', { __e: error, __h: hitLimit }) }));
const toasts = [null, {}, { summary: '' }, { summary: 'a skill' }].map((last) => ({ last, result: run('lnToastText(__l)', { __l: last }) }));
fs.writeFileSync(path.join(__dirname, 'learn_fixtures.json'), JSON.stringify({ cases, failed, toasts }, null, 1));
console.log(`cases: ${cases.length}`);
console.log('FIXTURES WRITTEN');
