// Tests for the thinking chip model in sampler.js (capabilities, levels, request fields, chip view) and the speculative note.
// Run from the repository root:  node rust/engine/src/webui/tests/thinking.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'sampler.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in sampler.js');
const T = new Function(`${src.slice(begin, end)}
return { thinkingCapsOf, thinkingLevelFor, nextThinkingLevel, thinkingRequestFields, thinkingChipView, speculativeNote };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

const server = (o) => T.thinkingCapsOf({ supported: true, variable: '', default: 'off', ...o }, false);
const qwen = () => server({ control: 'toggle', variable: 'enable_thinking', levels: ['off', 'low', 'medium', 'high'], default: 'high' });
const gptOss = () => server({ control: 'effort', variable: 'reasoning_effort', levels: ['low', 'medium', 'high'], default: 'medium' });
const always = () => server({ control: 'always', levels: ['high'], default: 'high' });

test('a model without thinking hides the chip and sends nothing', () => {
  for (const raw of [{ supported: false, control: 'none', levels: [] }, undefined, null, false, 'x']) {
    const caps = T.thinkingCapsOf(raw, false);
    assert.strictEqual(caps.supported, false);
    assert.strictEqual(T.thinkingChipView(caps, null).visible, false);
    assert.strictEqual(T.thinkingLevelFor(caps, 'high'), null);
    assert.deepStrictEqual(T.thinkingRequestFields(caps, 'high'), {});
  }
});

test('capabilities without usable levels count as unsupported', () => {
  assert.strictEqual(T.thinkingCapsOf({ supported: true, control: 'toggle', levels: ['wild'] }, false).supported, false);
  assert.strictEqual(T.thinkingCapsOf({ supported: true, control: 'weird', levels: ['low', 'bogus'] }, false).control, 'toggle');
});

test('an older server with a plain boolean offers off and high', () => {
  const caps = T.thinkingCapsOf(true, false);
  assert.deepStrictEqual(caps.levels, ['off', 'high']);
  assert.deepStrictEqual(T.thinkingRequestFields(caps, 'high'), { reasoning_effort: 'high', chat_template_kwargs: { enable_thinking: true } });
});

test('a model with a switch offers Off, Low, Medium, High and starts at its own default', () => {
  const caps = qwen();
  assert.deepStrictEqual(caps.levels, ['off', 'low', 'medium', 'high']);
  assert.strictEqual(T.thinkingLevelFor(caps, undefined), 'high');
  assert.strictEqual(T.thinkingLevelFor(caps, 'low'), 'low');
  assert.strictEqual(T.thinkingLevelFor(caps, 'turbo'), 'high');
});

test('clicking steps through the levels and wraps', () => {
  const caps = qwen();
  const seen = ['off'];
  for (let i = 0; i < 4; i++) seen.push(T.nextThinkingLevel(caps, seen[i]));
  assert.deepStrictEqual(seen, ['off', 'low', 'medium', 'high', 'off']);
  const effort = gptOss();
  assert.strictEqual(T.nextThinkingLevel(effort, 'high'), 'low');
  assert.strictEqual(T.nextThinkingLevel(always(), 'high'), 'high');
});

test('request fields: switch models send the switch and the level, effort models the effort', () => {
  const caps = qwen();
  assert.deepStrictEqual(T.thinkingRequestFields(caps, 'off'), { chat_template_kwargs: { enable_thinking: false } });
  assert.deepStrictEqual(T.thinkingRequestFields(caps, 'low'), { reasoning_effort: 'low', chat_template_kwargs: { enable_thinking: true } });
  assert.deepStrictEqual(T.thinkingRequestFields(gptOss(), 'medium'), { reasoning_effort: 'medium' });
  assert.deepStrictEqual(T.thinkingRequestFields(always(), 'high'), {});
  assert.deepStrictEqual(T.thinkingRequestFields(caps, null), {});
});

test('a model of another provider: off sends nothing, a level goes as the reasoning effort', () => {
  const caps = T.thinkingCapsOf(undefined, true);
  assert.strictEqual(caps.supported, true);
  assert.strictEqual(T.thinkingLevelFor(caps, undefined), 'off');
  assert.deepStrictEqual(T.thinkingRequestFields(caps, 'off'), {});
  assert.deepStrictEqual(T.thinkingRequestFields(caps, 'high'), { reasoning_effort: 'high' });
});

test('the chip shows the chosen level, or why it cannot be changed', () => {
  const caps = qwen();
  assert.strictEqual(T.thinkingChipView(caps, 'medium').label, 'Thinking: Medium');
  assert.strictEqual(T.thinkingChipView(caps, 'medium').active, true);
  const off = T.thinkingChipView(caps, 'off');
  assert.deepStrictEqual([off.label, off.active, off.disabled, off.visible], ['Thinking', false, false, true]);
  assert.match(T.thinkingChipView(caps, 'low').title, /512/);
  assert.doesNotMatch(T.thinkingChipView(gptOss(), 'low').title, /512/);
  const fixed = T.thinkingChipView(always(), 'high');
  assert.deepStrictEqual([fixed.visible, fixed.disabled], [true, true]);
  assert.match(fixed.title, /always thinks/);
});

test('speculative note: honest about what changes and who cannot use it', () => {
  assert.strictEqual(T.speculativeNote(''), '');
  for (const mode of ['lookup', 'draft', 'mtp']) {
    const note = T.speculativeNote(mode);
    assert.match(note, /only the speed changes/);
    assert.match(note, /recurrent/);
  }
  assert.match(T.speculativeNote('mtp'), /CUDA/);
});

console.log(`${n} tests passed`);
