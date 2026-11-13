// Generates tests/fixtures/style_fixtures.json: the expected results of the pure response style of
// the Studio (rust/engine/src/webui/artifacts.js, the part between @pure-begin and @pure-end),
// produced BY that code, so the Rust port in src/promptlib.rs/prompts.rs can be checked against it.
// Run from this directory: node style_fixtures.js > fixtures/style_fixtures.json
'use strict';
const fs = require('fs');
const path = require('path');

const artifactsPath = path.resolve(__dirname, '../../../../rust/engine/src/webui/artifacts.js');
const src = fs.readFileSync(artifactsPath, 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('markers not found in artifacts.js');
const pure = src.slice(src.lastIndexOf('\n', begin) + 1, src.lastIndexOf('\n', end) + 1);

const sandbox = new Function(pure + '\n  return { STYLES, withStylePrompt };')();
const { STYLES, withStylePrompt } = sandbox;

// ---- fixtures -----------------------------------------------------------------------------
const bases = ['', 'You are a helpful assistant.', 'Line one\nLine two'];
const cases = [];
for (const base of bases) {
  for (const s of STYLES) cases.push({ base, style: s.id, out: withStylePrompt(base, s.id) });
}
cases.push({ base: bases[1], style: 'unknown-style', out: withStylePrompt(bases[1], 'unknown-style') });
cases.push({ base: bases[1], style: '', out: withStylePrompt(bases[1], '') });
cases.push({ base: bases[1], style: undefined, out: withStylePrompt(bases[1], undefined) });

process.stdout.write(JSON.stringify({ styles: STYLES, cases }, null, 2) + '\n');
