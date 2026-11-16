// Fixtures of the General card: the two snippets of "Start it from a terminal" (`genCliPre` and
// `genTomlPre` of `updateCliAndSdkSnippets` in panels.js), run in a vm with the world stubbed.
// Run: node tests/fixtures/general_fixtures.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const lines = fs.readFileSync(path.join(WEBUI, 'panels.js'), 'utf8').split(/\r?\n/);
const from = lines.findIndex((l) => l.includes("$('genCliPre').textContent = ["));
const to = lines.findIndex((l, i) => i > from && l.includes("$('genTomlPre').textContent ="));
if (from < 0 || to < 0) throw new Error('the snippets were not found');
const body = lines.slice(from, to + 1).join('\n');

function run(m) {
  const out = {};
  const ctx = { state: { status: { model: m } }, $: (id) => ({ set textContent(v) { out[id] = v; } }), window: { location: { origin: 'http://x' } } };
  vm.runInNewContext(`const mm = state.status?.model || {}; (function () { const m = mm; const model = m.name || 'fllm-studio-demo'; const aliases = [model, ...(m.aliases || [])].join(','); ${body} })();`, ctx);
  return { cli: out.genCliPre, toml: out.genTomlPre };
}

const MODELS = [
  { name: 'qwen', max_context: 8192, max_sequences: 4, aliases: ['q', 'qq'] },
  { name: 'llama-3', max_context: 0, aliases: [] },
  {},
  { name: 'm', max_context: 512, max_sequences: 0 },
  { name: 'Ünï', max_context: 131072, max_sequences: 16, aliases: ['a'] },
];
const cases = MODELS.map((m) => ({ model: m, ...run(m) }));
fs.writeFileSync(path.join(__dirname, 'general_fixtures.json'), JSON.stringify({ cases }, null, 1));
console.log('written', cases.length);
