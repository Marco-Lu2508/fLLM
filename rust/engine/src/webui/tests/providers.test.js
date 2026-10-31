// Tests for the pure logic of the model providers (model picker merge, selection ids, request cleaning).
// Run from the repository root:  node rust/engine/src/webui/tests/providers.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'providers.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in providers.js');
const L = new Function(`${src.slice(begin, end)}
return { providerPresetGroups, providerKeyLink, providerTestSummary, remoteModelId, parseRemoteModelId, mergeModelEntries, resolveSelection, modelLabel, cleanRemoteRequest, chatEndpoint };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };
const remote = [
  { provider: 'openai', provider_name: 'OpenAI', id: 'gpt-x' },
  { provider: 'or', provider_name: 'OpenRouter', id: 'meta/llama-3' },
];

test('selection ids round trip, including model names with slashes', () => {
  assert.strictEqual(L.remoteModelId('or', 'meta/llama-3'), 'remote:or/meta/llama-3');
  assert.deepStrictEqual(L.parseRemoteModelId('remote:or/meta/llama-3'), { provider: 'or', model: 'meta/llama-3' });
  for (const bad of ['qwen3', '', null, undefined, 'remote:', 'remote:/x', 'remote:p/', 'remote:p']) assert.strictEqual(L.parseRemoteModelId(bad), null, String(bad));
});

test('the picker lists local models first, then Provider · model', () => {
  const e = L.mergeModelEntries(['qwen3', 'alias', 'qwen3'], remote, 'remote:openai/gpt-x');
  assert.deepStrictEqual(e.map((x) => x.label), ['qwen3', 'alias', 'OpenAI · gpt-x', 'OpenRouter · meta/llama-3']);
  assert.deepStrictEqual(e.map((x) => x.remote), [false, false, true, true]);
  assert.deepStrictEqual(e.map((x) => x.active), [false, false, true, false]);
  assert.strictEqual(e[3].id, 'remote:or/meta/llama-3');
});

test('the picker survives empty and odd input', () => {
  assert.deepStrictEqual(L.mergeModelEntries(undefined, undefined, ''), []);
  assert.deepStrictEqual(L.mergeModelEntries([], [null, { provider: 'p' }, { id: 'x' }], ''), []);
  const dup = L.mergeModelEntries([], [remote[0], remote[0]], '');
  assert.strictEqual(dup.length, 1);
});

test('a model of a removed provider falls back to the local one', () => {
  const e = L.mergeModelEntries(['qwen3'], remote, '');
  assert.strictEqual(L.resolveSelection('remote:openai/gpt-x', e, 'qwen3'), 'remote:openai/gpt-x');
  assert.strictEqual(L.resolveSelection('remote:gone/m', e, 'qwen3'), 'qwen3');
  assert.strictEqual(L.resolveSelection('qwen3', e, 'other'), 'qwen3');
});

test('labels name the provider', () => {
  assert.strictEqual(L.modelLabel('qwen3', remote), 'qwen3');
  assert.strictEqual(L.modelLabel('', remote), 'Choose a model');
  assert.strictEqual(L.modelLabel(undefined, remote), 'Choose a model');
  assert.strictEqual(L.modelLabel('remote:openai/gpt-x', remote), 'OpenAI · gpt-x');
  assert.strictEqual(L.modelLabel('remote:unknown/m', remote), 'unknown · m');
});

test('requests for a provider lose the settings of the local engine', () => {
  const body = { model: 'remote:or/meta/llama-3', messages: [{ role: 'user', content: 'hi' }], stream: true, stream_options: { include_usage: true },
    temperature: 0.5, top_k: 40, min_p: 0.05, mirostat: 2, dry_multiplier: 0.8, repetition_penalty: 1.1, tool_call_grammar: true, grammar: 'root ::= x', speculative: true, tools: [{ type: 'function' }] };
  const out = L.cleanRemoteRequest(body, 'meta/llama-3');
  assert.strictEqual(out.model, 'meta/llama-3');
  for (const k of ['top_k', 'min_p', 'mirostat', 'dry_multiplier', 'repetition_penalty', 'tool_call_grammar', 'grammar', 'speculative']) assert(!(k in out), k);
  assert.strictEqual(out.temperature, 0.5);
  assert.deepStrictEqual(out.tools, [{ type: 'function' }]);
  assert.strictEqual(body.top_k, 40, 'the original is not changed');
});

test('local models keep the local route, provider models use the proxy', () => {
  const local = L.chatEndpoint('qwen3');
  assert.strictEqual(local.url, '/v1/chat/completions');
  assert.strictEqual(local.remote, false);
  const b = { model: 'qwen3', messages: [] };
  assert.strictEqual(local.transform(b), b);
  const ep = L.chatEndpoint('remote:openai/gpt-x');
  assert.strictEqual(ep.url, '/api/studio/providers/chat');
  assert.strictEqual(ep.remote, true);
  assert.deepStrictEqual(ep.transform({ model: 'x', messages: [], top_k: 3 }), { provider: 'openai', request: { model: 'gpt-x', messages: [] } });
});

const presets = [
  { id: 'openai', name: 'OpenAI', group: 'cloud', key_url: 'https://platform.openai.com/api-keys' },
  { id: 'custom', name: 'Other', group: 'cloud', key_url: '' },
  { id: 'ollama', name: 'Ollama server', group: 'network', local: true },
  { id: 'old-local', name: 'Old', local: true }, // a server from before the groups existed
  { id: 'old-cloud', name: 'Older' },
];

test('templates are grouped: cloud services first, own servers in a separate group', () => {
  const [cloud, network] = L.providerPresetGroups(presets);
  assert.strictEqual(cloud.title, 'Cloud APIs');
  assert.deepStrictEqual(cloud.items.map((p) => p.id), ['openai', 'custom', 'old-cloud']);
  assert.strictEqual(network.title, 'Other servers on your network');
  assert.deepStrictEqual(network.items.map((p) => p.id), ['ollama', 'old-local']);
  assert.deepStrictEqual(L.providerPresetGroups(null).map((g) => g.items.length), [0, 0]);
});

test('the key link is an https page of the service', () => {
  assert.deepStrictEqual(L.providerKeyLink(presets[0]), { url: 'https://platform.openai.com/api-keys', host: 'platform.openai.com' });
  assert.strictEqual(L.providerKeyLink(presets[1]), null);
  assert.strictEqual(L.providerKeyLink({ key_url: 'javascript:alert(1)' }), null);
  assert.strictEqual(L.providerKeyLink({ key_url: 'http://example.org/k' }), null);
  assert.strictEqual(L.providerKeyLink(null), null);
});

test('the test result names the models or the problem', () => {
  assert.strictEqual(L.providerTestSummary({ ok: true, count: 2, models: ['a', 'b'] }), 'Connected. 2 models found: a, b.');
  assert.match(L.providerTestSummary({ ok: true, count: 12, models: Array.from({ length: 12 }, (_, i) => `m${i}`) }), /12 models found: m0.*m7 and 4 more\.$/);
  assert.strictEqual(L.providerTestSummary({ ok: true, count: 1, models: ['x'] }), 'Connected. 1 model found: x.');
  assert.match(L.providerTestSummary({ ok: true, count: 0, models: [] }), /no models/);
  assert.strictEqual(L.providerTestSummary({ ok: false, error: 'HTTP 401' }), 'HTTP 401');
  assert.strictEqual(L.providerTestSummary(null), 'The test failed.');
});

console.log(`${n} tests passed`);
