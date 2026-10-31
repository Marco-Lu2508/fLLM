// Tests for the sub-agents page (agents.js): tool groups, the checked tools of the form and the request it sends.
// Run from the repository root:  node rust/engine/src/webui/tests/agents.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'agents.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in agents.js');
const A = new Function(`${src.slice(begin, end)}
return { AGENT_BUILTINS, agentToolGroups, agentCheckedTools, agentPayload, agentReach };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const tools = [
  { name: 'read_file', permission: 'read', description: 'Reads a file.\nMore.' },
  { name: 'grep', permission: 'read', description: 'Searches.' },
  { name: 'bash', permission: 'command', description: 'Runs a command.' },
  { name: 'edit_file', permission: 'edit', description: 'Edits.' },
  { name: 'task', permission: 'read', description: 'Starts a sub-agent.' },
  { name: 'computer', permission: 'computer', description: 'Screen.' },
];

test('tools are grouped into reading and acting; sub-agent and screen tools are not offered', () => {
  const g = A.agentToolGroups(tools);
  assert.deepStrictEqual(g.read.map((t) => t.name), ['grep', 'read_file']);
  assert.deepStrictEqual(g.act.map((t) => t.name), ['bash', 'edit_file']);
  assert.strictEqual(g.read[1].description, 'Reads a file.');
});

test('process and file tools are never in the reading group, even when the server calls them read', () => {
  const g = A.agentToolGroups([{ name: 'bash_kill', permission: 'read' }, { name: 'memory', permission: 'read' }, { name: 'glob', permission: 'read' }]);
  assert.deepStrictEqual(g.read.map((t) => t.name), ['glob']);
  assert.deepStrictEqual(g.act.map((t) => t.name), ['bash_kill', 'memory']);
  assert.match(A.agentReach({ tools: ['glob', 'bash_kill'] }, g), /change files/);
  assert.match(A.agentReach({ tools: ['bash_output', 'glob'] }, g), /reads and searches/);
});

test('a new agent starts with the reading tools checked, an existing one with its own list', () => {
  const g = A.agentToolGroups(tools);
  assert.deepStrictEqual([...A.agentCheckedTools({}, g)].sort(), ['grep', 'read_file']);
  assert.deepStrictEqual([...A.agentCheckedTools({ tools: ['bash'] }, g)], ['bash']);
});

const form = (o) => ({ name: 'code-reviewer', description: ' Reviews\n a change ', body: ' You review. ', tools: ['grep', 'grep', ''], model: ' ', max_steps: '12', scope: 'user', ...o });

test('a valid form becomes the request body of the route', () => {
  const r = A.agentPayload(form({}), null, true, []);
  assert.deepStrictEqual(r.payload, { name: 'code-reviewer', description: 'Reviews a change', body: 'You review.', tools: ['grep'], model: '', max_steps: 12, scope: 'user', workspace: null });
});

test('names, instructions and scope are checked with a plain message', () => {
  assert.match(A.agentPayload(form({ name: 'Code Reviewer' }), null, true, []).error, /lowercase/);
  assert.match(A.agentPayload(form({ name: '' }), null, true, []).error, /name/);
  assert.match(A.agentPayload(form({ name: 'explore' }), null, true, []).error, /built-in/);
  assert.match(A.agentPayload(form({}), null, true, ['code-reviewer']).error, /already have/);
  assert.ok(A.agentPayload(form({}), null, false, ['code-reviewer']).payload, 'saving an existing one is fine');
  assert.match(A.agentPayload(form({ body: '  ' }), null, true, []).error, /instructions/);
  assert.match(A.agentPayload(form({ scope: 'workspace' }), null, true, []).error, /folder/);
  assert.strictEqual(A.agentPayload(form({ scope: 'workspace' }), 'C:\\work', true, []).payload.workspace, 'C:\\work');
});

test('the step limit stays between 0 and 60', () => {
  assert.strictEqual(A.agentPayload(form({ max_steps: '500' }), null, true, []).payload.max_steps, 60);
  assert.strictEqual(A.agentPayload(form({ max_steps: '-3' }), null, true, []).payload.max_steps, 0);
  assert.strictEqual(A.agentPayload(form({ max_steps: 'x' }), null, true, []).payload.max_steps, 0);
});

test('the list says what an agent can reach', () => {
  const g = A.agentToolGroups(tools);
  assert.match(A.agentReach({ tools: [] }, g), /reads and searches only/);
  assert.match(A.agentReach({ tools: ['grep'] }, g), /reads and searches/);
  assert.match(A.agentReach({ tools: ['grep', 'bash'] }, g), /change files/);
});

console.log(`${n} tests passed`);
