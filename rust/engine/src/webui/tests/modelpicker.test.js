// Tests of the pure parts of the model dialog (modelpicker.js): what is said about a model before it is loaded.
// Run from the repository root:  node rust/engine/src/webui/tests/modelpicker.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'modelpicker.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in modelpicker.js');
const { mtpSummary, factRows, cleanPathText } = new Function(`${src.slice(begin, end)}\nreturn { mtpSummary, factRows, cleanPathText };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

test('heads are described honestly: present, absent, blocked, unknown', () => {
  assert(/Built-in multi-token prediction heads were found/.test(mtpSummary({ mtp: true })));
  assert(/has no multi-token prediction heads/.test(mtpSummary({ mtp: false })));
  assert(/cannot be rolled back/.test(mtpSummary({ mtp: true, speculation_blocked: 'Speculative decoding does not work: state that cannot be rolled back.' })), 'a blocked architecture is explained, not offered');
  assert(/known once the files are on this computer/.test(mtpSummary({ mtp: null })));
  assert.strictEqual(mtpSummary(null), '');
});

test('the facts list leaves out what is not known and says whether it fits', () => {
  const rows = factRows({ ok: true, format: 'GGUF file', size_bytes: 5.5 * 1073741824, files: 1, architecture: 'llama', quantization: 'Q4_K_M', context_length: 32768, fits: true });
  assert.deepStrictEqual(rows.map((r) => r[0]), ['Format', 'Size', 'Architecture', 'Quantization', 'Context length', 'Memory']);
  assert.strictEqual(rows[1][1], '5.5 GB');
  assert.strictEqual(rows[4][1], '32,768 tokens');
  const split = factRows({ ok: true, format: 'GGUF file, split into 3 parts', size_bytes: 3 * 1073741824, files: 3 });
  assert.strictEqual(split[1][1], '3.0 GB in 3 files');
  assert.strictEqual(factRows({ ok: true, format: 'x', fits: false }).pop()[1], 'May not fit into the free memory');
  assert.deepStrictEqual(factRows({ ok: false, problem: 'x' }), []);
});

test('quotes and spaces around a pasted path are removed', () => {
  assert.strictEqual(cleanPathText('  "C:\\Models\\a b\\m.gguf" '), 'C:\\Models\\a b\\m.gguf');
  assert.strictEqual(cleanPathText("'owner/name:Q4_K_M'"), 'owner/name:Q4_K_M');
});

console.log(`${n} tests passed`);
