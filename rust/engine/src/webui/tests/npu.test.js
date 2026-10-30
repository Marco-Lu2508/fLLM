// Tests for the pure logic of the NPU card (card model, progress, device choice of the load options).
// Run from the repository root:  node rust/engine/src/webui/tests/npu.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'npu.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in npu.js');
const L = new Function(`${src.slice(begin, end)}
return { npuCardModel, npuProgressModel, deviceChoices, runningDevice, deviceMemory, deviceRole, NPU_WARNING };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const idle = { phase: 'idle', busy: false, done_bytes: 0, total_bytes: 0, message: '', error: null };
const base = (over = {}) => ({
  state: 'no_npu',
  warning: L.NPU_WARNING,
  hardware: [],
  runtime: { installed: false, version: null, path: null, managed: false, error: null },
  setup: { available: true, pinned_version: '2026.4.1', download_bytes: 59144434, progress: idle },
  last_error: null,
  quantizations: { eight_bit: ['Q4_0', 'Q4_K'], exact: ['Q4_0', 'Q8_0'], converted_int4: ['Q4_K'], float16: ['F16'] },
  ...over,
});
const intel = { vendor: 'intel', name: 'Intel NPU (accel0)', supported: true, problem: null };
const amd = { vendor: 'amd', name: 'AMD XDNA NPU (accel0)', supported: false, problem: null };

test('without data the card says it is checking and offers nothing', () => {
  const m = L.npuCardModel(null);
  assert.match(m.summary, /Checking/);
  assert.strictEqual(m.button, null);
  assert.strictEqual(m.remove, false);
});

test('an Intel NPU without the runtime offers the install with its size', () => {
  const m = L.npuCardModel(base({ state: 'runtime_missing', hardware: [intel] }));
  assert.match(m.summary, /not installed yet/);
  assert.match(m.summary, /59 MB/);
  assert.deepStrictEqual(m.button, { action: 'install', label: 'Install the runtime (about 59 MB)' });
  assert.strictEqual(m.hardware[0].supported, true);
  assert.strictEqual(m.remove, false);
  assert.strictEqual(m.cancel, false);
});

test('a Qualcomm NPU is shown as detected, never usable, and gets no button', () => {
  const m = L.npuCardModel(base({ state: 'unsupported_hardware', hardware: [{ vendor: 'qualcomm', name: 'Hexagon', supported: false }] }));
  assert.match(m.summary, /not supported yet/);
  assert.strictEqual(m.button, null);
  assert.deepStrictEqual(m.hardware.map((h) => h.badge), ['detected, not supported yet']);
  assert.ok(m.hardware.every((h) => !h.supported));
});

test('an AMD NPU is pointed to the FastFlowLM card instead of "not supported"', () => {
  const m = L.npuCardModel(base({ state: 'unsupported_hardware', hardware: [amd] }));
  assert.match(m.summary, /FastFlowLM/);
  assert.strictEqual(m.button, null);
  assert.match(m.hardware[0].badge, /FastFlowLM/);
});

test('a running setup shows progress and a cancel button instead of the install button', () => {
  const progress = { phase: 'downloading', busy: true, done_bytes: 29572217, total_bytes: 59144434 };
  const m = L.npuCardModel(base({ state: 'installing', hardware: [intel], setup: { available: true, pinned_version: '2026.4.1', download_bytes: 59144434, progress } }));
  assert.strictEqual(m.button, null);
  assert.strictEqual(m.cancel, true);
  assert.strictEqual(m.remove, false);
  assert.deepStrictEqual(m.progress, { visible: true, percent: 50, text: 'Downloading (29.6 of 59.1 MB)' });
});

test('progress handles unknown totals and the other phases', () => {
  assert.strictEqual(L.npuProgressModel(idle).visible, false);
  assert.strictEqual(L.npuProgressModel(undefined).visible, false);
  assert.deepStrictEqual(L.npuProgressModel({ phase: 'unpacking', busy: true, done_bytes: 0, total_bytes: 0 }), { visible: true, percent: 0, text: 'Unpacking' });
  assert.strictEqual(L.npuProgressModel({ phase: 'verifying', busy: true, done_bytes: 5, total_bytes: 4 }).percent, 100);
});

test('a failed setup shows the error and "Try again"', () => {
  const progress = { phase: 'failed', busy: false, done_bytes: 0, total_bytes: 0, error: 'the checksum of the download is wrong' };
  const m = L.npuCardModel(base({ state: 'runtime_missing', hardware: [intel], setup: { available: true, pinned_version: '2026.4.1', download_bytes: 1e8, progress } }));
  assert.strictEqual(m.error, 'the checksum of the download is wrong');
  assert.strictEqual(m.button.label, 'Try again');
});

test('a ready NPU shows the version and the remove button only for our own installation', () => {
  const ready = base({ state: 'ready', hardware: [intel], runtime: { installed: true, version: '2026.4.1', path: '/x', managed: true, error: null } });
  const m = L.npuCardModel(ready);
  assert.match(m.summary, /Ready \(experimental\), OpenVINO 2026\.4\.1/);
  assert.strictEqual(m.tone, 'good');
  assert.strictEqual(m.remove, true);
  assert.strictEqual(m.button, null);
  const system = L.npuCardModel({ ...ready, runtime: { ...ready.runtime, managed: false } });
  assert.strictEqual(system.remove, false);
});

test('the status line of the prompt offload is passed on, empty when it is not in use', () => {
  const idleCard = L.npuCardModel(base({ state: 'ready', hardware: [intel] }));
  assert.strictEqual(idleCard.offload, '');
  const working = L.npuCardModel(base({ state: 'ready', hardware: [intel], offload_line: 'Prompt offload on Intel(R) AI Boost: 120 calls, 30000 rows, 40 models compiled, 700 MB held', devices: [{ full_name: 'Intel(R) AI Boost', architecture: '3720' }] }));
  assert.match(working.offload, /^Prompt offload on/);
  assert.ok(working.summary.includes("Intel(R) AI Boost (platform 3720)"));
});

test('a runtime without a device explains and keeps the loader error visible', () => {
  const m = L.npuCardModel(base({ state: 'no_device', hardware: [intel], runtime: { installed: true, version: null, path: '/x', managed: false, error: 'symbol missing' } }));
  assert.match(m.summary, /does not list an NPU/);
  assert.strictEqual(m.error, 'symbol missing');
  assert.strictEqual(m.tone, 'warn');
});

test('the reason of the last fallback to the processor is shown', () => {
  const m = L.npuCardModel(base({ state: 'ready', last_error: 'The NPU could not take this model. The model runs on the processor instead.' }));
  assert.match(m.error, /runs on the processor instead/);
});

test('the card is honest about what the NPU does and lists the formats', () => {
  const m = L.npuCardModel(base());
  const text = m.explanation.join(' ');
  assert.match(text, /Writing the answer is not sped up/);
  assert.match(text, /needs about its own size again in memory|about its own size again/);
  assert.match(text, /never chosen automatically/);
  assert.deepStrictEqual(m.formats.map((f) => f.title.split(" (")[0].split(" with ")[0]), ["Computed as 8 bit", "Half precision", "Exact", "Kept at 4 bit"]);
  assert.deepStrictEqual(m.formats[0].items, ["Q4_0", "Q4_K"]);
  assert.ok(!/https?:/.test(text), 'no links');
});

const hw = {
  devices: [
    { id: 'cpu', backend: 'cpu', name: 'Some CPU', total_memory_bytes: 16e9 },
    { id: 'vulkan:0', backend: 'vulkan', name: 'Some GPU', total_memory_bytes: 8e9 },
    { id: 'vulkan:1', backend: 'vulkan', name: 'No memory', total_memory_bytes: 0 },
    { id: 'npu-unavailable', backend: 'npu', name: 'NPU', total_memory_bytes: 0, available: false },
    { id: 'npu:0', backend: 'npu', name: 'Intel NPU', total_memory_bytes: 8e9 },
  ],
};

test('the device choice lists automatic, processor, graphics devices and the NPU only when ready', () => {
  const without = L.deviceChoices(hw, base({ state: 'runtime_missing' }), 'auto');
  assert.deepStrictEqual(without.map((c) => c.value), ['auto', 'cpu', 'vulkan:0']);
  const withNpu = L.deviceChoices(hw, base({ state: 'ready' }), 'auto');
  assert.deepStrictEqual(withNpu.map((c) => c.value), ['auto', 'cpu', 'vulkan:0', 'npu']);
  const npu = withNpu.find((c) => c.value === 'npu');
  assert.match(npu.label, /experimental/i);
  assert.match(npu.warning, /speeds up reading long prompts/);
  assert.match(npu.warning, /converted copy of the weights/);
  assert.match(npu.warning, /integrated graphics are usually faster/);
  assert.ok(withNpu.filter((c) => c.value !== 'npu').every((c) => c.warning === null));
});

test('the device choice survives missing data and keeps the current value', () => {
  assert.deepStrictEqual(L.deviceChoices(null, null, '').map((c) => c.value), ['auto', 'cpu']);
  const kept = L.deviceChoices(hw, null, 'CUDA:1');
  assert.strictEqual(kept[kept.length - 1].value, 'cuda:1');
  const npuSaved = L.deviceChoices(hw, base({ state: 'runtime_missing' }), 'npu');
  assert.strictEqual(npuSaved[npuSaved.length - 1].value, 'npu');
  assert.match(npuSaved[npuSaved.length - 1].warning, /Experimental/);
  assert.strictEqual(L.deviceChoices(hw, base({ state: 'ready' }), 'npu').filter((c) => c.value === 'npu').length, 1);
});

const gpuHw = { auto_choice: 'vulkan:0', devices: [
  { id: 'cpu', backend: 'cpu', name: 'Intel(R) Core(TM) Ultra 5 235U' },
  { id: 'vulkan:0', backend: 'vulkan', name: 'Intel(R) Graphics' },
  { id: 'cuda:0', backend: 'cuda', name: 'NVIDIA RTX 2000 Ada' },
] };

test('the footer names the device the automatic choice picks', () => {
  const r = L.runningDevice('auto', gpuHw, 'avx-vnni');
  assert.strictEqual(r.label, 'Intel Graphics · Vulkan');
  assert.strictEqual(r.auto, true);
  assert.match(r.title, /automatic choice/);
});

test('an explicit device and the processor are named as chosen', () => {
  assert.strictEqual(L.runningDevice('cuda:0', gpuHw, 'x').label, 'NVIDIA RTX 2000 Ada · CUDA');
  assert.match(L.runningDevice('cuda:0', gpuHw, 'x').title, /load options/);
  assert.strictEqual(L.runningDevice('cpu', gpuHw, 'avx2').label, 'CPU · avx2');
});

test('without a hardware report or with a processor choice the footer shows the processor', () => {
  assert.strictEqual(L.runningDevice('auto', null, 'avx2').label, 'CPU · avx2');
  assert.strictEqual(L.runningDevice(undefined, { auto_choice: 'cpu', devices: [] }, '').label, 'CPU · simd');
  assert.strictEqual(L.runningDevice('vulkan:3', gpuHw, 'x').label, 'vulkan:3 · Vulkan');
});

test('the memory bar of a device: size known or not, and the warning levels', () => {
  assert.deepStrictEqual(L.deviceMemory({ total_memory_bytes: 0, free_memory_bytes: 0 }), { known: false, pct: 0, level: 'ok' });
  assert.deepStrictEqual(L.deviceMemory(null), { known: false, pct: 0, level: 'ok' });
  assert.deepStrictEqual(L.deviceMemory({ total_memory_bytes: 100, free_memory_bytes: 60 }), { known: true, pct: 40, level: 'ok' });
  assert.strictEqual(L.deviceMemory({ total_memory_bytes: 100, free_memory_bytes: 20 }).level, 'warn');
  assert.strictEqual(L.deviceMemory({ total_memory_bytes: 100, free_memory_bytes: 5 }).level, 'high');
  assert.strictEqual(L.deviceMemory({ total_memory_bytes: 100, free_memory_bytes: 500 }).pct, 0);
});

test('which device runs the model: the chosen one, or the automatic choice', () => {
  const gpu = { id: 'vulkan:0' };
  const cpu = { id: 'cpu' };
  assert.strictEqual(L.deviceRole(gpu, 'auto', 'vulkan:0', true), 'running');
  assert.strictEqual(L.deviceRole(cpu, 'auto', 'vulkan:0', true), '');
  assert.strictEqual(L.deviceRole(cpu, 'cpu', 'vulkan:0', true), 'running');
  assert.strictEqual(L.deviceRole(gpu, 'cpu', 'vulkan:0', true), '');
  assert.strictEqual(L.deviceRole(gpu, 'auto', 'vulkan:0', false), 'auto');
  assert.strictEqual(L.deviceRole(cpu, undefined, undefined, true), 'running');
});

console.log(`${n} tests passed`);
