#!/usr/bin/env node
// Rows 83 and 84 fixtures: the tool cards of the Code page. The page's own `TOOL_LABEL` and
// `toolLabel`, `diffRows`, `outputHtml` and `defaultOpen` (code.js) are run in a vm over the cases
// below; the port in src/toolcard.rs must give the same words, rows and open/closed state.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const lines = fs.readFileSync(path.join(WEBUI, 'code.js'), 'utf8').split(/\r?\n/);
const find = (text, from = 0) => { const i = lines.findIndex((l, k) => k >= from && l.includes(text)); if (i < 0) throw new Error(`no line with ${text}`); return i; };
const slice = (a, b) => lines.slice(find(a), find(b)).join('\n');

const LABELS = slice('const TOOL_LABEL = {', 'const EDIT_TOOLS');
const TOOLLABEL = slice('const toolLabel = (item) => {', 'const defaultOpen');
const DIFFROWS = slice('function diffRows(item) {', 'function diffHtml');
const OUTPUT = slice('function outputHtml(item) {', 'const toolLabel = (item)');
const DEFOPEN = slice('const defaultOpen = (item) =>', 'function toolHtml');
const PATCH = (() => { const a = find('function patchPaths(text) {'); let b = a; while (!/^    }$/.test(lines[b])) b++; return lines.slice(a, b + 1).join('\n'); })();
const EDIT = lines[find('const EDIT_TOOLS = ')];
const ISTASK = lines[find('const isTask = ')];

const harness = `
  const OUT_LINES = 40;
  const DIFF_LINES = 60;
  const escapeHtml = (s) => String(s);
  const code = { registry: new Map() };
  ${ISTASK}
  ${EDIT}
  ${PATCH}
  ${LABELS}
  ${TOOLLABEL}
  ${DIFFROWS}
  ${OUTPUT}
  ${DEFOPEN}
`;
const context = { console, JSON, Math, Set, Map, Array, Object, String, Number, Boolean, RegExp };
vm.createContext(context);
vm.runInContext(harness, context, { filename: 'toolcard.js' });
const run = (expr, vars) => { Object.assign(context, vars); return JSON.parse(JSON.stringify(vm.runInContext(`(() => { const r = ${expr}; return r === undefined ? { __undefined: true } : r; })()`, context))); };

const NAMES = ['list_dir', 'read_file', 'write_file', 'edit_file', 'multi_edit', 'glob', 'grep', 'bash', 'bash_output', 'bash_kill', 'web_fetch', 'web_search',
  'computer_click', 'browser_open', 'browser_type', 'browser_scroll', 'browser_close', 'todo_write', 'todo_read', 'ask_user', 'spawn_agent', 'task', 'patch', 'review', 'diagnostics', 'memory',
  'notebook_edit', 'notebook_read', 'create_agent', 'skill', 'something_else'];
const ARGS = [null, {}, { path: 'src/a.rs' }, { path: 'src/a.rs', offset: 12 }, { path: 'src/a.rs', offset: 0 }, { path: '' }, { path: 'a', edits: [1, 2, 3] }, { path: 'a', edits: [] }, { path: 'a', edits: 'x' },
  { pattern: '*.rs' }, { pattern: 'foo', glob: '*.js' }, { pattern: 'foo', glob: '' }, { command: 'ls -la' }, { command: 'npm run dev', background: true }, { command: 'x', background: 0 },
  { job_id: 'j1' }, { job_id: 7 }, { job_id: null }, { a: 1, b: 2 }, { url: 'https://example.com' }, { query: 'rust' }, { description: 'd', agent: 'explore' }, { description: 'd', mode: 'full' }, { description: 'd' },
  { question: 'Which one?' }, { patch: '--- a/x\n+++ b/x\n@@\n-a\n+b' }, { patch: '' }, { scope: 'branch' }, { security: true }, { scope: 'diff', security: true }, { action: 'add', scope: 'global', text: 'hi' }, { action: 'list' },
  { path: 'nb.ipynb', cell: 2 }, { name: 'x', content: 'y' }, { first: 'z'.repeat(100), second: 'w' }, { a: 5, b: 'str' }, { a: [1], b: { c: 1 } }];
const labels = [];
for (const name of NAMES) for (const args of ARGS) labels.push({ name, args, label: run('toolLabel(__i)', { __i: { name, args } }) });
// an item with no args at all, and a tool of an MCP server
labels.push({ name: 'read_file', args: undefined, label: run('toolLabel(__i)', { __i: { name: 'read_file' } }) });
const mcp = [];
for (const args of [{}, { q: 'x' }, { n: 1, s: 'str' }, { s: 'a'.repeat(120) }]) {
  vm.runInContext("code.registry = new Map([['mcp__srv__tool', { source: 'mcp', tool: 'tool', server: 'srv' }]]);", context);
  mcp.push({ name: 'mcp__srv__tool', args, label: run('toolLabel(__i)', { __i: { name: 'mcp__srv__tool', args } }) });
  vm.runInContext('code.registry = new Map();', context);
}

const diffs = [];
const DARGS = [
  ['edit_file', { old_string: 'a\nb', new_string: 'c' }], ['edit_file', {}], ['edit_file', { old_string: '', new_string: '' }], ['edit_file', { old_string: null, new_string: 5 }],
  ['multi_edit', { edits: [{ old_string: 'a', new_string: 'b' }, { old_string: 'c\nd', new_string: 'e' }] }], ['multi_edit', { edits: [] }], ['multi_edit', { edits: [null, { old_string: 'x' }] }], ['multi_edit', { edits: 'no' }], ['multi_edit', {}],
  ['write_file', { content: 'l1\nl2\nl3' }], ['write_file', { content: '' }], ['write_file', {}], ['write_file', { content: 12 }],
  ['patch', { patch: '--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n ctx\n-old\n+new\n' }], ['patch', { patch: '' }], ['patch', {}], ['patch', { patch: 'plain\n+add\n-del\n' }],
  ['bash', { command: 'ls' }], ['read_file', { path: 'x' }],
  ['write_file', { content: Array.from({ length: 80 }, (_, i) => `line ${i}`).join('\n') }],
];
for (const [name, args] of DARGS) diffs.push({ name, args, rows: run('diffRows(__i)', { __i: { name, args } }) });
diffs.push({ name: 'edit_file', args: undefined, rows: run('diffRows(__i)', { __i: { name: 'edit_file' } }) });

const bigLine = 'x'.repeat(7000);
const TEXTS = ['', 'short', 'a\nb\nc', Array.from({ length: 40 }, (_, i) => `l${i}`).join('\n'), Array.from({ length: 41 }, (_, i) => `l${i}`).join('\n'), Array.from({ length: 200 }, (_, i) => `line number ${i}`).join('\n'),
  bigLine, `${bigLine}\nsecond`, 'y'.repeat(61000), `${'z'.repeat(5990)}\n${'q'.repeat(20)}\nend`, 'tab\tseparated\r\nwindows', Array.from({ length: 41 }, () => 'm'.repeat(200)).join('\n')];
const outputs = [];
for (const text of TEXTS) for (const expanded of [false, true]) {
  const html = run('outputHtml(__i)', { __i: { id: 'i1', output: text, expanded } });
  const m = /^<pre class="tool-out">([\s\S]*)<\/pre>([\s\S]*)$/.exec(html);
  outputs.push({ text, expanded, shown: m[1], more: m[2] });
}
outputs.push({ text: 5, expanded: false, ...(() => { const html = run('outputHtml(__i)', { __i: { id: 'i1', output: 5, expanded: false } }); const m = /^<pre class="tool-out">([\s\S]*)<\/pre>([\s\S]*)$/.exec(html); return { shown: m[1], more: m[2] }; })() });

const opens = [];
const STATUSES = ['pending', 'running', 'ok', 'error', 'denied'];
const NAMES2 = ['bash', 'read_file', 'write_file', 'edit_file', 'multi_edit', 'patch', 'ask_user', 'task', 'computer_screenshot', 'computer_click', 'browser_screenshot'];
for (const name of NAMES2) for (const status of STATUSES) for (const extra of [{}, { output: 'x' }, { image: { url: 'u' } }, { shot: { url: 'u' } }, { check: 'problems' }]) {
  const item = { name, status, ...extra };
  opens.push({ item, open: run('defaultOpen(__i)', { __i: item }) });
}
fs.writeFileSync(path.join(__dirname, 'toolcard_fixtures.json'), JSON.stringify({ labels, mcp, diffs, outputs, opens }, null, 1));
console.log(`labels ${labels.length}, mcp ${mcp.length}, diffs ${diffs.length}, outputs ${outputs.length}, opens ${opens.length}`);
console.log('FIXTURES WRITTEN');
