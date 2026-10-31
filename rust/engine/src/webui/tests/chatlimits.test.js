// Tests for the output limit of a chat request and the context counter in chat.js.
// Run from the repository root:  node rust/engine/src/webui/tests/chatlimits.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'chat.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in chat.js');
const T = new Function(`${src.slice(begin, end)}
return { outputLimitFields, contextMeterModel };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

test('automatic: no max_tokens is sent, the server decides, and the clamp flag is on', () => {
  for (const tools of [false, true]) {
    const f = T.outputLimitFields(0, false, tools, 32768);
    assert.strictEqual('max_tokens' in f, false);
    assert.strictEqual(f.x_fllm_clamp_max_tokens, true);
  }
});

test('a chosen number is a ceiling that the server may cut', () => {
  const f = T.outputLimitFields(1000, false, false, 32768);
  assert.deepStrictEqual(f, { x_fllm_clamp_max_tokens: true, max_tokens: 1000 });
});

test('a chosen number with tools leaves room for thinking, search and answer', () => {
  assert.strictEqual(T.outputLimitFields(300, false, true, 32768).max_tokens, 2048);
  assert.strictEqual(T.outputLimitFields(300, false, true, 1024).max_tokens, 512);
  assert.strictEqual(T.outputLimitFields(5000, false, true, 32768).max_tokens, 5000);
});

test('a provider model gets a number and no local flag', () => {
  assert.deepStrictEqual(T.outputLimitFields(0, true, false), { max_tokens: 4096 });
  assert.deepStrictEqual(T.outputLimitFields(900, true, true), { max_tokens: 900 });
});

test('junk values count as automatic', () => {
  for (const v of [NaN, -5, undefined, null, 'x']) assert.strictEqual('max_tokens' in T.outputLimitFields(v, false, false), false);
});

test('the counter uses the server count once there is one, and says estimate before', () => {
  const first = T.contextMeterModel(632, null, 0);
  assert.strictEqual(first.used, 632);
  assert.strictEqual(first.exact, false);
  const real = T.contextMeterModel(0, { total_tokens: 1090 }, 0);
  assert.strictEqual(real.used, 1090);
  assert.strictEqual(real.exact, true);
  const more = T.contextMeterModel(0, { total_tokens: 1090 }, 40);
  assert.strictEqual(more.used, 1130);
  assert.strictEqual(more.exact, false);
});

console.log(`${n} tests passed`);
