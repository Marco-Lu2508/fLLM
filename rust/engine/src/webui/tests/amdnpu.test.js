// Tests for the pure logic of the AMD NPU card (FastFlowLM): the card model for every state.
// Run from the repository root:  node rust/engine/src/webui/tests/amdnpu.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'amdnpu.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in amdnpu.js');
const L = new Function(`${src.slice(begin, end)}
return { amdNpuModel };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const models = [
  { tag: 'gemma3:4b', installed: false },
  { tag: 'llama3.2:3b', installed: true },
  { tag: 'llama3.1:8b', installed: true },
];
const base = (over = {}) => ({
  state: 'ready',
  hardware: [{ name: 'AMD Ryzen AI NPU (accel0)', problem: null }],
  flm: { installed: true, version: 'v1.0.11', install: { text: 'Install it.', url: 'https://example.invalid/releases' } },
  validation: { ok: true, problems: [], lines: [] },
  models,
  serving: { state: 'idle' },
  pull: { busy: false, model: '', line: '', error: null },
  error: null,
  ...over,
});

test('the card is hidden without an AMD NPU and without FastFlowLM', () => {
  assert.strictEqual(L.amdNpuModel(null).visible, false);
  assert.strictEqual(L.amdNpuModel({ state: 'no_npu' }).visible, false);
});

test('an NPU without FastFlowLM shows where to get it and offers nothing to run', () => {
  const m = L.amdNpuModel(base({ state: 'flm_missing', flm: { installed: false, install: { text: 'Install it.', url: 'https://example.invalid/releases' } }, models: [] }));
  assert.strictEqual(m.visible, true);
  assert.match(m.summary, /install yourself/);
  assert.strictEqual(m.install.url, 'https://example.invalid/releases');
  assert.strictEqual(m.start, false);
  assert.strictEqual(m.pull, false);
});

test('a problem found by flm validate is shown and nothing can be started', () => {
  const m = L.amdNpuModel(base({ state: 'needs_attention', validation: { ok: false, problems: ['Memlock limit is too low (8MB).'], lines: [] } }));
  assert.match(m.summary, /Memlock limit is too low/);
  assert.deepStrictEqual(m.problems, ['Memlock limit is too low (8MB).']);
  assert.strictEqual(m.start, false);
  assert.strictEqual(m.tone, 'warn');
});

test('the installed models come first and the first of them is chosen', () => {
  const m = L.amdNpuModel(base(), '');
  assert.deepStrictEqual(m.models.map((x) => x.tag), ['llama3.1:8b', 'llama3.2:3b', 'gemma3:4b']);
  assert.strictEqual(m.selected, 'llama3.1:8b');
  assert.strictEqual(m.start, true);
  assert.strictEqual(m.pull, false);
  assert.strictEqual(m.models[2].label, 'gemma3:4b (not downloaded)');
});

test('a model that is not downloaded offers the download, not the start', () => {
  const m = L.amdNpuModel(base(), 'gemma3:4b');
  assert.strictEqual(m.selected, 'gemma3:4b');
  assert.strictEqual(m.pull, true);
  assert.strictEqual(m.start, false);
});

test('a download shows its last line, a cancel button and keeps the card polling', () => {
  const m = L.amdNpuModel(base({ pull: { busy: true, model: 'gemma3:4b', line: '41%', error: null } }), 'gemma3:4b');
  assert.strictEqual(m.cancelPull, true);
  assert.strictEqual(m.pull, false);
  assert.strictEqual(m.busy, true);
  assert.strictEqual(m.pullText, 'Downloading gemma3:4b: 41%');
});

test('a model that is starting can be stopped and the card keeps polling', () => {
  const m = L.amdNpuModel(base({ serving: { state: 'starting', model: 'llama3.2:3b' } }), 'llama3.2:3b');
  assert.match(m.summary, /Starting llama3.2:3b/);
  assert.strictEqual(m.stop, true);
  assert.strictEqual(m.start, false);
  assert.strictEqual(m.busy, true);
});

test('a running model says where to pick it and can be stopped', () => {
  const m = L.amdNpuModel(base({ serving: { state: 'running', model: 'llama3.2:3b', port: 5000, base_url: 'http://127.0.0.1:5000/v1' } }));
  assert.match(m.summary, /AMD NPU \(FastFlowLM\) · llama3.2:3b/);
  assert.strictEqual(m.selected, 'llama3.2:3b', 'the list follows the running model');
  assert.strictEqual(m.stop, true);
  assert.strictEqual(m.start, false);
  assert.strictEqual(m.tone, 'good');
  assert.strictEqual(m.busy, false);
});

test('a failed start shows the reason and allows another try', () => {
  const m = L.amdNpuModel(base({ serving: { state: 'failed', model: 'llama3.2:3b', error: 'flm stopped (exit status: 1). NPU busy' } }), 'llama3.2:3b');
  assert.match(m.error, /NPU busy/);
  assert.strictEqual(m.start, true);
  assert.strictEqual(m.stop, false);
});

test('a cancelled download is not shown as an error', () => {
  const m = L.amdNpuModel(base({ pull: { busy: false, model: 'gemma3:4b', line: '10%', error: 'cancelled' } }));
  assert.strictEqual(m.error, null);
});

test('the card says whether fLLM\'s native kernels can drive the NPU', () => {
  assert.strictEqual(L.amdNpuModel(base()).nativeText, '', 'a build without the native backend says nothing');
  const ready = L.amdNpuModel(base({ native: { compiled: true, available: true, devices: 1 } }));
  assert.match(ready.nativeText, /Native fLLM kernels: ready/);
  assert.strictEqual(ready.nativeReady, true);
  const missing = L.amdNpuModel(base({ native: { compiled: true, available: false, reason: 'the XDNA shim was not found' } }));
  assert.match(missing.nativeText, /not available \(the XDNA shim was not found\)/);
  assert.match(missing.nativeText, /FastFlowLM, below, does not need them/);
  assert.strictEqual(missing.nativeReady, false);
});

console.log(`${n} tests passed`);
