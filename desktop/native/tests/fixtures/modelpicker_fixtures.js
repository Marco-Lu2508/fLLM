// Fixtures of the model picker: `mtpSummary`, `factRows` and `cleanPathText` (the @pure block of
// modelpicker.js) and `formatBytes` of core.js, run in a vm. Run: node tests/fixtures/modelpicker_fixtures.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const src = fs.readFileSync(path.join(WEBUI, 'modelpicker.js'), 'utf8');
const core = fs.readFileSync(path.join(WEBUI, 'core.js'), 'utf8').split('\n');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('the pure block was not found');
const zone = src.slice(src.indexOf('\n', begin) + 1, src.lastIndexOf('\n', end));
const fbStart = core.findIndex((l) => l.includes('function formatBytes(bytes) {'));
const fb = core.slice(fbStart, fbStart + 7).join('\n');
const api = vm.runInNewContext(`${fb}\n${zone}\n;({ mtpSummary, factRows, cleanPathText, formatBytes })`, { Number, Math, String }, { timeout: 5000 });

const FACTS = [
  null,
  { speculation_blocked: 'Not with this build.' },
  { mtp: true },
  { mtp: false },
  {},
  { ok: false },
  { ok: true, format: 'GGUF', size_bytes: 4831838208, files: 1, architecture: 'qwen3', quantization: 'Q4_K_M', context_length: 40960, fits: true },
  { ok: true, format: 'safetensors', size_bytes: 15032385536, files: 4, architecture: 'llama', quantization: 'BF16', context_length: 131072, fits: false },
  { ok: true, format: 'GGUF', size_bytes: 300000, files: 1 },
  { ok: true, format: 'GGUF', size_bytes: 0, files: 0, architecture: '', quantization: 'F16', context_length: 0 },
  { ok: true, format: 'GGUF', size_bytes: 1073741824, files: 2, context_length: 512, fits: null },
  { ok: true, format: 'GGUF', size_bytes: 5242880, files: 1 },
];
const out = { facts: FACTS.map((f) => ({ facts: f, mtp: api.mtpSummary(f), rows: api.factRows(f) })) };
const PATHS = ['  /models/a.gguf  ', '"C:\\Models\\a.gguf"', "'/x y/z'", '', '   ', '"""quoted"""', 'owner/name', '" spaced "', null, undefined, '\'"mixed\'"'];
out.paths = PATHS.map((p) => ({ text: p === undefined ? '__undefined__' : p, got: api.cleanPathText(p) }));
const BYTES = [0, -5, 1, 1023, 1024, 1536, 1048576, 5242880, 4831838208, 1099511627776, 5e15, null, undefined, '2048'];
out.bytes = BYTES.map((b) => ({ bytes: b === undefined ? '__undefined__' : b, got: api.formatBytes(b) }));
fs.writeFileSync(path.join(__dirname, 'modelpicker_fixtures.json'), JSON.stringify(out, null, 1));
console.log('written', out.facts.length, out.paths.length, out.bytes.length);
