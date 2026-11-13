// Generates tests/fixtures/speech_fixtures.json: the expected results of the pure speech helpers of
// the Studio (rust/engine/src/webui/voice.js, the part between @pure-begin and @pure-end),
// produced BY that code, so the Rust port in src/speech.rs can be checked against the original.
// Run from this directory: node speech_fixtures.js > fixtures/speech_fixtures.json
'use strict';
const fs = require('fs');
const path = require('path');

const voicePath = path.resolve(__dirname, '../../../../rust/engine/src/webui/voice.js');
const src = fs.readFileSync(voicePath, 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('markers not found in voice.js');
const pure = src.slice(src.lastIndexOf('\n', begin) + 1, src.lastIndexOf('\n', end) + 1);

const sandbox = new Function(pure + '\n  return { stripForSpeech, chunkForSpeech };')();
const { stripForSpeech, chunkForSpeech } = sandbox;

// ---- fixtures -----------------------------------------------------------------------------
const strips = [
  '',
  'Just a sentence.',
  '# Heading\n\nSome **bold** and *italic* text with `code`.',
  '```rust\nfn main() { println!("hi"); }\n```\nAfter the code.',
  'A [link](https://example.com/page) and an ![image alt](pic.png) here.',
  'Visit https://example.com for more. <div>markup</div> here.',
  '> quoted line\n- item one\n1. item two',
  '| a | b |\n|---|---|\n| 1 | 2 |',
  'think block: <think>secret reasoning</think> visible.',
  '***\n--- rule above',
  'A very long line with ~~struck~~ and __under__ text and ***stars***.',
];
const chunks = [
  { text: '', max: undefined },
  { text: 'One short sentence.', max: undefined },
  { text: 'First sentence. Second one! Third? Fourth.', max: undefined },
  { text: 'A'.repeat(500), max: undefined },
  { text: ('word '.repeat(60)).trim() + '.', max: 100 },
  { text: 'Line one is here.\nLine two follows with more words.\nLine three.', max: 80 },
  { text: 'A'.repeat(250) + ',' + 'B'.repeat(100), max: 220 },
];

process.stdout.write(
  JSON.stringify(
    {
      strips: strips.map((input) => ({ input, out: stripForSpeech(input) })),
      chunks: chunks.map(({ text, max }) => ({ text, max, out: chunkForSpeech(text, max) })),
    },
    null,
    2,
  ) + '\n',
);
