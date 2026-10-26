'use strict';
// Tests of extractBareJsonCalls (code.js): small models often answer with the tool call as plain JSON.
// Run: node rust/engine/src/webui/tests/barecalls.test.js
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'code.js'), 'utf8');
const start = src.indexOf('function extractBareJsonCalls');
const end = src.indexOf('/** ctx: { system, tools, messages, signal, onUsage }', start);
assert.ok(start > 0 && end > start, 'function not found in code.js');
let counter = 0;
const extract = new Function('cid', `${src.slice(start, end)}\nreturn extractBareJsonCalls;`)(() => `n${counter++}`);

const names = ['list_dir', 'read_file'];
let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

test('a bare call object', () => {
  const calls = extract('{"name": "list_dir", "arguments": {"path": ".", "depth": 1}}', names);
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0].name, 'list_dir');
  assert.deepStrictEqual(JSON.parse(calls[0].args), { path: '.', depth: 1 });
});

test('inside a code fence, with "parameters" and the function wrapper', () => {
  assert.strictEqual(extract('```json\n{"name": "read_file", "parameters": {"path": "a.txt"}}\n```', names)[0].name, 'read_file');
  assert.strictEqual(extract('{"function": {"name": "read_file", "arguments": "{\\"path\\": \\"a\\"}"}}', names)[0].args, '{"path": "a"}');
});

test('a list of calls', () => {
  const calls = extract('[{"name":"list_dir","arguments":{}},{"name":"read_file","arguments":{"path":"x"}}]', names);
  assert.deepStrictEqual(calls.map((c) => c.name), ['list_dir', 'read_file']);
});

test('text, unknown tools and other JSON are not calls', () => {
  assert.deepStrictEqual(extract('I will list the folder now.', names), []);
  assert.deepStrictEqual(extract('{"name": "rm_everything", "arguments": {}}', names), []);
  assert.deepStrictEqual(extract('{"title": "Report", "items": [1, 2]}', names), []);
  assert.deepStrictEqual(extract('{"name": "list_dir", "arguments": 5}', names), []);
  assert.deepStrictEqual(extract('Here: {"name": "list_dir", "arguments": {}}', names), []);
  assert.deepStrictEqual(extract('', names), []);
});

console.log(`${n} tests passed`);
