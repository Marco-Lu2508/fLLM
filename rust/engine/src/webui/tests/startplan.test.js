// Tests for the empty start screen in panels.js: which model is offered when nothing is loaded.
// Run from the repository root:  node rust/engine/src/webui/tests/startplan.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'panels.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in panels.js');
const P = new Function(`${src.slice(begin, end)}
return { modelFileTitle, startPlan };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const GB = 1e9;
const rec = { id: 'qwen3.5:9b', display_name: 'qwen3.5:9b', default_quant: 'Q4_K_M', variants: [{ quant: 'Q4_K_M', estimated_bytes: 5.5 * GB, fits_memory: true }], installed: null };
const hub = (over) => ({ available_memory_bytes: 40 * GB, recommended: ['qwen3.5:9b'], catalog: [rec], cached: [], local: [], found: [], ...over });

test('nothing on the computer: the recommended model is offered as a download', () => {
  const p = P.startPlan(hub(), '');
  assert.strictEqual(p.mode, 'download');
  assert.strictEqual(p.download.id, 'qwen3.5:9b');
  assert.strictEqual(p.primary, null);
});

test('any model on the computer turns the offer into a load, with the model that was used last first', () => {
  const h = hub({
    cached: [{ name: 'hf://unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf', path: 'C:\\m\\Qwen3-4B-Q4_K_M.gguf', size_bytes: 2.5 * GB }],
    found: [{ path: 'C:\\Users\\a\\Downloads\\tiny.gguf', name: 'tiny.gguf', size_bytes: 0.4 * GB, source: 'Downloads' }],
  });
  let p = P.startPlan(h, '');
  assert.strictEqual(p.mode, 'load');
  assert.strictEqual(p.primary.title, 'Qwen3-4B-Q4_K_M', 'without a last model the largest that fits');
  assert.strictEqual(p.others.length, 1);
  p = P.startPlan(h, 'C:\\Users\\a\\Downloads\\tiny.gguf');
  assert.strictEqual(p.primary.title, 'tiny');
  assert.strictEqual(p.primary.last, true);
  p = P.startPlan(h, 'hf://unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf');
  assert.strictEqual(p.primary.title, 'Qwen3-4B-Q4_K_M');
});

test('the recommended model on disk is loaded, never offered for download', () => {
  const installed = { ref: 'C:\\m\\Qwen3.5-9B-Q4_K_M.gguf', path: 'C:\\m\\Qwen3.5-9B-Q4_K_M.gguf', size_bytes: 5.5 * GB };
  const h = hub({
    catalog: [{ ...rec, installed }],
    found: [
      { path: 'C:\\m\\Qwen3.5-9B-Q4_K_M.gguf', name: 'Qwen3.5-9B-Q4_K_M.gguf', size_bytes: 5.5 * GB, source: 'models folder' },
      { path: 'C:\\m\\big.gguf', name: 'big.gguf', size_bytes: 30 * GB, source: 'Downloads' },
    ],
  });
  const p = P.startPlan(h, '');
  assert.strictEqual(p.mode, 'load');
  assert.strictEqual(p.primary.title, 'Qwen3.5-9B-Q4_K_M', 'the recommended one comes before a larger one');
  assert.strictEqual(p.download, null);
});

test('the last model given as a catalog name finds its file', () => {
  const installed = { ref: 'C:\\m\\a.gguf', path: 'C:\\m\\a.gguf', size_bytes: 1 * GB };
  const h = hub({ catalog: [{ ...rec, id: 'qwen3:8b', installed }], recommended: [], found: [
    { path: 'C:\\m\\a.gguf', name: 'a.gguf', size_bytes: 1 * GB, source: 'x' }, { path: 'C:\\m\\b.gguf', name: 'b.gguf', size_bytes: 3 * GB, source: 'x' }] });
  assert.strictEqual(P.startPlan(h, 'qwen3:8b').primary.title, 'a');
});

test('a model too large for the free memory goes last and is flagged', () => {
  const h = hub({ found: [
    { path: 'C:\\m\\huge.gguf', name: 'huge.gguf', size_bytes: 60 * GB, source: 'Downloads' },
    { path: 'C:\\m\\small.gguf', name: 'small.gguf', size_bytes: 2 * GB, source: 'Downloads' }] });
  assert.strictEqual(P.startPlan(h, '').primary.title, 'small');
  const only = P.startPlan(hub({ found: [{ path: 'C:\\m\\huge.gguf', name: 'huge.gguf', size_bytes: 60 * GB, source: 'Downloads' }] }), '');
  assert.strictEqual(only.primary.fits, false);
});

test('the same file listed twice is offered once; file titles lose folders and the extension', () => {
  const h = hub({ local: [{ path: 'C:\\m\\x.gguf', name: 'x.gguf', size_bytes: GB }], found: [{ path: 'c:\\m\\X.GGUF', name: 'X.GGUF', size_bytes: GB, source: 'a' }] });
  const p = P.startPlan(h, '');
  assert.strictEqual(1 + p.others.length, 1);
  assert.strictEqual(P.modelFileTitle('hf://o/r/Model-Q4.gguf'), 'Model-Q4');
  assert.strictEqual(P.modelFileTitle('qwen3:8b'), 'qwen3:8b');
  assert.strictEqual(P.modelFileTitle(undefined), '');
});

test('no hub data at all is a download offer without a model', () => {
  assert.strictEqual(P.startPlan(null, '').mode, 'download');
  assert.strictEqual(P.startPlan(null, '').download, null);
});

console.log(`${n} tests passed`);
