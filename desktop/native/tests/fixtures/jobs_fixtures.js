#!/usr/bin/env node
// Row 87 fixtures: the background jobs of the Code page — `parseJobId`, `jobArgName`, `trackJob`,
// `jobOutput` and `jobKill` of the page's own code.js, run in a vm with the world stubbed.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const code = fs.readFileSync(path.join(WEBUI, 'code.js'), 'utf8').split(/\r?\n/);
// slices are found by their first line's text, never by remembered line numbers
const at = (start, endText) => {
  const from = code.findIndex((s) => s.includes(start));
  if (from < 0) throw new Error(`no line with ${start}`);
  let to = from;
  if (endText) {
    to = code.findIndex((s, i) => i >= from && s.includes(endText));
    if (to < 0) throw new Error(`no line with ${endText}`);
  }
  return code.slice(from, to + 1).join('\n');
};
const PARSE = at('function parseJobId(text) {', '      return m ? m[1] : null;') + '\n    }';
// from `jobArgName` to the end of `jobKill`
const start = code.findIndex((s) => s.includes('const jobArgName = () => {'));
const end = code.findIndex((s, i) => i > start && s.includes("const isTask = (name) =>"));
const JOBS_ALL = code.slice(start, end).join('\n');
new vm.Script(PARSE); new vm.Script(JOBS_ALL);

const harness = `
  const code = { jobs: [], registry: new Map() };
  let __calls = [];
  let __next = null;
  const renderCodeSide = () => {};
  const toast = (m) => { __toasts.push(m); };
  const __toasts = [];
  const toolCall = async (name, args) => { __calls.push([name, args]); if (__next instanceof Error) throw __next; return __next; };
  ${PARSE}
  ${JOBS_ALL}
`;
const context = { console, JSON, Math, Set, Map, Array, Object, String, Number, Promise, Error, RegExp };
vm.createContext(context);
vm.runInContext(harness, context, { filename: 'jobs.js' });
const run = (expr) => vm.runInContext(expr, context);

const TEXTS = [
  'Started background job job1. Read its output with bash_output {job_id: "job1"} and stop it with bash_kill.',
  'Job 12 started', 'started shell abc-1', 'no id here at all', 'pid 4242', 'process: p_9', 'deadbeef01 is running', 'bash_7f3a started',
  '', 'task=7', 'id=#42', 'ID: Ünï', "job '(x_y-z)'", 'the job', 'job', 'jobs 5', 'taskid 5', 'background job: ', 'exit code 0', '12',
  'Started job_1', 'Process 99 (python)', 'shell "s-1"',
];
const parse = TEXTS.concat([null, undefined, 42]).map((t) => ({ text: t === undefined ? '__undefined__' : t, id: run(`parseJobId(${t === undefined ? 'undefined' : JSON.stringify(t)})`) }));

// `trackJob` and the two buttons, over a registry that names the argument of bash_output
const SCENARIOS = [
  { name: 'a background bash starts a job', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'npm run dev', background: true }, 'Started background job job1. Read its output.'] },
  ] },
  { name: 'a bash that is not in the background is no job', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'ls' }, 'Started background job job1.'] },
  ] },
  { name: 'a background bash with no id in the answer', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'x', background: true }, 'nothing to see'] },
  ] },
  { name: 'output keeps the text and an exit ends the job', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'trackJob', args: ['bash_output', { job_id: 'j1' }, 'building...'] },
    { fn: 'trackJob', args: ['bash_output', { job_id: 'j1' }, 'process exited with exit code 0'] },
  ] },
  { name: 'a word that only contains finished is not finished', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'trackJob', args: ['bash_output', { job_id: 'j1' }, 'unfinished business, exited_ok'] },
  ] },
  { name: 'case does not matter for the exit', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'trackJob', args: ['bash_output', { job_id: 'j1' }, 'Process FINISHED'] },
  ] },
  { name: 'kill marks the job', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'trackJob', args: ['bash_kill', { job_id: 'j1' }, 'killed'] },
  ] },
  { name: 'an id the page does not know', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash_output', { job_id: 'zzz' }, 'out'] },
    { fn: 'trackJob', args: ['bash_kill', { job_id: 'zzz' }, 'out'] },
  ] },
  { name: 'the first value of the arguments is the id, numbers too', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job 7'] },
    { fn: 'trackJob', args: ['bash_output', { n: 7 }, 'ok'] },
  ] },
  { name: 'two jobs', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'a', background: true }, 'job a1'] },
    { fn: 'trackJob', args: ['bash', { command: 'b' }, 'x'] },
    { fn: 'trackJob', args: ['bash', { command: 'c', background: true }, 'job c3'] },
    { fn: 'trackJob', args: ['bash_output', { job_id: 'c3' }, 'finished'] },
  ] },
  { name: 'the output button opens, reads and closes', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobOutput', args: ['j1'], answer: { ok: true, output: 'hello\nexit code 1' } },
    { fn: 'jobOutput', args: ['j1'], answer: { ok: true, output: 'not read' } },
    { fn: 'jobOutput', args: ['j1'], answer: { ok: false, error: 'gone' } },
  ] },
  { name: 'the output button with a refusal', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobOutput', args: ['j1'], answer: { ok: false, error: 'no such job' } },
    { fn: 'jobOutput', args: ['j1'], answer: { ok: false } },
    { fn: 'jobOutput', args: ['j1'], answer: { ok: false } },
  ] },
  { name: 'the output button with a failed call', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobOutput', args: ['j1'], answer: 'throw:no server' },
  ] },
  { name: 'the output button for an unknown job', registry: { props: ['job_id'] }, steps: [
    { fn: 'jobOutput', args: ['nope'], answer: { ok: true, output: 'x' } },
  ] },
  { name: 'stop', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobKill', args: ['j1'], answer: { ok: true, output: 'ok' } },
  ] },
  { name: 'stop refused', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobKill', args: ['j1'], answer: { ok: false, error: 'not running' } },
    { fn: 'jobKill', args: ['j1'], answer: { ok: false } },
  ] },
  { name: 'stop with a failed call', registry: { props: ['job_id'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobKill', args: ['j1'], answer: 'throw:no server' },
  ] },
  { name: 'the argument name of a server that calls it shell', registry: { props: ['shell'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobOutput', args: ['j1'], answer: { ok: true, output: 'x' } },
  ] },
  { name: 'the argument name: first that looks like an id, else the first', registry: { props: ['since', 'bash_id', 'x'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobOutput', args: ['j1'], answer: { ok: true, output: 'x' } },
  ] },
  { name: 'the argument name: none looks like an id', registry: { props: ['since', 'x'] }, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobOutput', args: ['j1'], answer: { ok: true, output: 'x' } },
  ] },
  { name: 'the argument name: no bash_output tool at all', registry: null, steps: [
    { fn: 'trackJob', args: ['bash', { command: 'make', background: true }, 'job j1'] },
    { fn: 'jobKill', args: ['j1'], answer: { ok: true, output: 'ok' } },
  ] },
];

(async () => {
  const scenarios = [];
  for (const sc of SCENARIOS) {
    vm.runInContext('code.jobs = []; __calls = []; __toasts.length = 0;', context);
    context.__reg = sc.registry;
    vm.runInContext(`code.registry = new Map(); if (__reg) code.registry.set('bash_output', { def: { parameters: { properties: Object.fromEntries(__reg.props.map((p) => [p, {}])) } } });`, context);
    const out = [];
    for (const st of sc.steps) {
      context.__args = JSON.parse(JSON.stringify(st.args));
      if (st.answer !== undefined) {
        if (typeof st.answer === 'string' && st.answer.startsWith('throw:')) {
          context.__nextMsg = st.answer.slice(6);
          vm.runInContext('__next = new Error(__nextMsg);', context);
        } else {
          context.__ans = st.answer;
          vm.runInContext('__next = __ans;', context);
        }
      }
      await vm.runInContext(`${st.fn}(...__args)`, context);
      out.push({
        jobs: JSON.parse(vm.runInContext('JSON.stringify(code.jobs)', context)),
        calls: JSON.parse(vm.runInContext('JSON.stringify(__calls)', context)),
        toasts: JSON.parse(vm.runInContext('JSON.stringify(__toasts)', context)),
      });
    }
    scenarios.push({ ...sc, after: out });
  }
  fs.writeFileSync(path.join(__dirname, 'jobs_fixtures.json'), JSON.stringify({ parse, scenarios }, null, 1));
  console.log(`parse: ${parse.length}, scenarios: ${scenarios.length}`);
  console.log('FIXTURES WRITTEN');
})();
