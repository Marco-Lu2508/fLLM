// Tests for the text preparation of reading aloud (Markdown removed, code skipped, long text chunked).
// Run from the repository root:  node rust/engine/src/webui/tests/voice.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'voice.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in voice.js');
const V = new Function(`${src.slice(begin, end)}
return { stripForSpeech, chunkForSpeech };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

test('code blocks are dropped, inline code keeps its words', () => {
  const t = V.stripForSpeech('Run this:\n```bash\nrm -rf /\necho hi\n```\nThen call `foo()` again.');
  assert.strictEqual(t, 'Run this:\nThen call foo() again.');
});

test('an unfinished code block is dropped to the end', () => {
  assert.strictEqual(V.stripForSpeech('Before\n```js\nlet a = 1;'), 'Before');
});

test('headings, lists, quotes and emphasis lose their marks', () => {
  const t = V.stripForSpeech('# Title\n\n- **bold** item\n- *it* and _that_\n1. first\n> quoted ~~old~~');
  assert.strictEqual(t, 'Title\nbold item\nit and that\nfirst\nquoted old');
});

test('snake_case words stay as they are', () => {
  assert.strictEqual(V.stripForSpeech('use my_var_name here'), 'use my_var_name here');
});

test('links and images give their text, bare addresses are dropped', () => {
  assert.strictEqual(V.stripForSpeech('See [the docs](https://x.org/a) and ![a cat](c.png) or https://x.org/b now'), 'See the docs and a cat or now');
});

test('tables are read as words without pipes and separator rows', () => {
  assert.strictEqual(V.stripForSpeech('| a | b |\n|---|:--:|\n| 1 | 2 |'), 'a b\n1 2');
});

test('html tags and think blocks are removed', () => {
  assert.strictEqual(V.stripForSpeech('<think>secret plan</think>Hello <b>you</b>'), 'Hello you');
});

test('nothing but code gives an empty text', () => {
  assert.strictEqual(V.stripForSpeech('```\ncode\n```'), '');
  assert.deepStrictEqual(V.chunkForSpeech(''), []);
});

test('chunkForSpeech cuts at sentence ends and respects the limit', () => {
  const text = 'One two three. Four five six. Seven eight nine.';
  assert.deepStrictEqual(V.chunkForSpeech(text, 32), ['One two three. Four five six.', 'Seven eight nine.']);
  assert.deepStrictEqual(V.chunkForSpeech(text, 200), [text]);
});

test('a very long sentence is cut at commas or spaces, nothing is lost', () => {
  const sentence = `${'word, '.repeat(100)}end.`;
  const parts = V.chunkForSpeech(sentence, 50);
  assert(parts.every((p) => p.length <= 50), parts.map((p) => p.length).join());
  assert.strictEqual(parts.join(' ').replace(/\s+/g, ' '), sentence.replace(/\s+/g, ' '));
  assert.strictEqual(V.chunkForSpeech('x'.repeat(130), 50).join(''), 'x'.repeat(130));
});

test('lines stay apart', () => {
  assert.deepStrictEqual(V.chunkForSpeech('First line\nSecond line', 200), ['First line', 'Second line']);
});

console.log(`${n} tests passed`);
