// Tests for the pure logic of the CPU-fallback notice and the GPU check lines (gpucheck.js).
// Run from the repository root:  node rust/engine/src/webui/tests/gpucheck.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'gpucheck.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in gpucheck.js');
const L = new Function(`${src.slice(begin, end)}\nreturn { fallbackNotice, gpuVerdictLine };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const mostly = {
  mode: 'mostly_cpu', fallback: true, device: 'NVIDIA GeForce RTX 2070 SUPER', backend: 'cuda', operations_total: 64, operations_on_cpu: 44,
  layers_on_device: 10, layers_feed_forward_on_cpu: 22, layers_on_cpu: 0, whole_model_on_cpu: false,
  reasons: [{ reason: 'not enough device memory', operations: 44 }], operation_fallbacks: [], notices: [],
  headline: 'GPU NVIDIA GeForce RTX 2070 SUPER (cuda): 44 of 64 operations run on the CPU: not enough device memory (44)', summary: 'Weights in device memory, compute mostly on the CPU',
};

test('nothing to show when the model runs on the device or no GPU is involved', () => {
  assert.strictEqual(L.fallbackNotice(null).show, false);
  assert.strictEqual(L.fallbackNotice({ mode: 'device', fallback: false }).show, false);
  assert.strictEqual(L.fallbackNotice({ mode: 'none', fallback: false }).show, false);
});

test('a model that mostly computes on the CPU is named exactly that', () => {
  const m = L.fallbackNotice(mostly);
  assert.strictEqual(m.show, true);
  assert.strictEqual(m.level, 'mostly');
  assert.ok(/Weights in GPU memory, compute mostly on the CPU/.test(m.title));
  assert.ok(m.lines.some((l) => l.includes('44 of 64 operations') && l.includes('10 layers')));
  assert.ok(m.lines.some((l) => l.startsWith('Why: not enough device memory (44)')));
  assert.ok(m.footerTitle.includes('fllm-rs doctor --gpu'));
});

test('a partial fallback and a whole model on the CPU', () => {
  const p = L.fallbackNotice({ ...mostly, mode: 'partial', operations_on_cpu: 4 });
  assert.strictEqual(p.level, 'partial');
  assert.strictEqual(p.title, 'Part of this model runs on the CPU');
  const c = L.fallbackNotice({ ...mostly, mode: 'cpu', whole_model_on_cpu: true, operations_on_cpu: 0 });
  assert.strictEqual(c.level, 'cpu');
  assert.ok(c.lines[0].includes('is not used for this model'));
});

test('operations that fall back one by one are listed, those by design are not', () => {
  const o = L.fallbackNotice({ ...mostly, mode: 'partial', operations_on_cpu: 0, reasons: [], operation_fallbacks: [
    { operation: 'moe_route', reason: 'no kernel', calls: 12, by_design: false },
    { operation: 'sampling_logits', reason: 'host sampling', calls: 900, by_design: true },
  ] });
  assert.ok(o.lines.some((l) => l.includes('moe_route: no kernel (12 calls)')));
  assert.ok(!o.lines.some((l) => l.includes('sampling_logits')));
});

test('the verdict lines get a colour', () => {
  assert.strictEqual(L.gpuVerdictLine({ device: 'cuda:0', status: 'ok', verdict: 'OK (9 times faster than the CPU)' }).cls, 'ok');
  assert.strictEqual(L.gpuVerdictLine({ device: 'cuda:0', status: 'slow', verdict: 'SLOWER THAN THE CPU' }).cls, 'warn');
  const bad = L.gpuVerdictLine({ device: 'vulkan:1', status: 'falls_back', verdict: 'FALLS BACK: x' });
  assert.strictEqual(bad.cls, 'bad');
  assert.strictEqual(bad.text, 'vulkan:1: FALLS BACK: x');
});

console.log(`${n} tests passed`);
