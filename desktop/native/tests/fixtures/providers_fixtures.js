// The fixtures of the model providers are produced by providers.js itself: the
// @pure-begin..@pure-end block of the Studio is run in a sandbox and every function of it is asked
// the cases the Rust tests ask. Run: node tests/fixtures/providers_fixtures.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const src = fs.readFileSync(
  path.join(__dirname, '../../../../rust/engine/src/webui/providers.js'),
  'utf8',
);
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('the pure block of providers.js was not found');
const zone = src.slice(src.indexOf('\n', begin) + 1, src.lastIndexOf('\n', end));

// `URL` is a Node global and not part of a fresh vm context, so it is handed in: without it
// `providerKeyLink` would swallow a ReferenceError in its catch and answer null for every url.
const api = vm.runInNewContext(
  `${zone}\n;({ remoteModelId, parseRemoteModelId, mergeModelEntries, resolveSelection, providerPresetGroups, providerKeyLink, providerTestSummary, modelLabel, LOCAL_ONLY_FIELDS, cleanRemoteRequest, chatEndpoint })`,
  { URL },
  { timeout: 5000 },
);
const {
  remoteModelId, parseRemoteModelId, mergeModelEntries, resolveSelection,
  providerPresetGroups, providerKeyLink, providerTestSummary, modelLabel,
  LOCAL_ONLY_FIELDS, cleanRemoteRequest, chatEndpoint,
} = api;

const out = {};

// remoteModelId: the id under which a provider's model is selected
out.remote_id = [
  ['openai', 'gpt-4o', remoteModelId('openai', 'gpt-4o')],
  ['ollama', 'llama3:8b', remoteModelId('ollama', 'llama3:8b')],
  ['my-server', 'a/model/with/slashes', remoteModelId('my-server', 'a/model/with/slashes')],
];

// parseRemoteModelId: only a string with the prefix and one slash that is not at either end
out.parse_remote_id = [
  ['openai/gpt-4o', parseRemoteModelId('openai/gpt-4o')],
  ['remote:', parseRemoteModelId('remote:')],
  ['remote:openai', parseRemoteModelId('remote:openai')],
  ['remote:openai/', parseRemoteModelId('remote:openai/')],
  ['remote:/gpt-4o', parseRemoteModelId('remote:/gpt-4o')],
  ['remote:my-server/a/b', parseRemoteModelId('remote:my-server/a/b')],
  ['local-model', parseRemoteModelId('local-model')],
  ['', parseRemoteModelId('')],
  [null, parseRemoteModelId(null)],
  [42, parseRemoteModelId(42)],
];

// mergeModelEntries: the local names first, then one entry per model of a provider
const remote_models = [
  { provider: 'openai', provider_name: 'OpenAI', id: 'gpt-4o' },
  { provider: 'ollama', id: 'llama3:8b' },
  { provider: 'openai', id: 'gpt-4o' },
  { provider: '', id: 'orphan' },
  { provider: 'lonely', id: '' },
  null,
];
out.merge_entries = [
  [['local-a', 'local-b'], remote_models, 'local-a', mergeModelEntries(['local-a', 'local-b'], remote_models, 'local-a')],
  [['local-a'], remote_models, 'remote:openai/gpt-4o', mergeModelEntries(['local-a'], remote_models, 'remote:openai/gpt-4o')],
  [[], [], '', mergeModelEntries([], [], '')],
  // a local name that appears twice is one entry, and a remote id that is already there is skipped
  [['same', 'same'], remote_models, 'same', mergeModelEntries(['same', 'same'], remote_models, 'same')],
  // null instead of lists (the Studio passes what the server sent)
  [null, null, 'x', mergeModelEntries(null, null, 'x')],
];

// resolveSelection: a selection that still exists, else the fallback
const entries = mergeModelEntries(['local-a'], remote_models, 'remote:openai/gpt-4o');
// [the selection, the fallback to fall back to, what the Studio answers]
const resolve_cases = [
  ['local-a', 'fallback'],
  ['remote:openai/gpt-4o', 'fallback'],
  ['remote:openai/gone', 'fallback'],
  ['', 'fallback'],
  ['remote:openai/gone', 'local-a'],
];
out.resolve_selection = resolve_cases.map(([selected, fallback]) => [selected, fallback, resolveSelection(selected, entries, fallback)]);

// providerPresetGroups: cloud first, the network group when it says so or when it is local
const presets = [
  { id: 'openai', name: 'OpenAI', group: 'cloud' },
  { id: 'anthropic', name: 'Anthropic', group: 'cloud' },
  { id: 'ollama', name: 'Ollama', local: true },
  { id: 'lmstudio', name: 'LM Studio', group: 'network' },
  { id: 'custom', name: 'A server I run' },
];
out.preset_input = presets;
out.preset_groups = JSON.parse(JSON.stringify(providerPresetGroups(presets)));
out.preset_groups_empty = JSON.parse(JSON.stringify(providerPresetGroups([])));
out.preset_groups_not_a_list = JSON.parse(JSON.stringify(providerPresetGroups('nope')));

// providerKeyLink: https pages only, with the host that hands out the key
out.key_link = [
  ['https://platform.openai.com/keys', providerKeyLink({ key_url: 'https://platform.openai.com/keys' })],
  ['https://console.anthropic.com/', providerKeyLink({ key_url: 'https://console.anthropic.com/' })],
  ['http://insecure.example.com/key', providerKeyLink({ key_url: 'http://insecure.example.com/key' })],
  ['not a url', providerKeyLink({ key_url: 'not a url' })],
  ['no key_url', providerKeyLink({})],
  ['null preset', providerKeyLink(null)],
];

// providerTestSummary: the line under "Test connection" — the answer as the server sends it and
// the line the Studio writes for it
const answers = [
  ['failed', { ok: false, error: 'the key was refused' }],
  ['failed without a message', { ok: false }],
  ['nothing', { ok: true, models: [], count: 0 }],
  ['one', { ok: true, models: ['gpt-4o'], count: 1 }],
  ['three', { ok: true, models: ['a', 'b', 'c'], count: 3 }],
  ['more than eight', { ok: true, models: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], count: 10 }],
  ['count without models', { ok: true, count: 4 }],
  ['models without count', { ok: true, models: ['a', 'b'] }],
  ['no answer', null],
];
out.test_summary = answers.map(([name, answer]) => [name, answer, providerTestSummary(answer)]);

// modelLabel: what the picker shows for a selection. The list of the server holds no null (the
// picker skips one anyway), so this block is asked with the entries it really sends.
const label_models = remote_models.filter(Boolean);
const label_cases = [
  ['nothing', '', label_models],
  ['local', 'local-a', label_models],
  ['remote with a name', 'remote:openai/gpt-4o', label_models],
  ['remote without a name', 'remote:ollama/llama3:8b', label_models],
  ['remote the list does not know', 'remote:gone/model', label_models],
  ['no list', 'remote:openai/gpt-4o', null],
  ['an empty list', 'remote:openai/gpt-4o', []],
];
out.model_label = label_cases.map(([name, selected, list]) => [name, selected, list, modelLabel(selected, list)]);

// LOCAL_ONLY_FIELDS and cleanRemoteRequest: the settings of the local engine a provider does not take
out.local_only_fields = LOCAL_ONLY_FIELDS.slice();
const body = {
  model: 'local-model',
  messages: [{ role: 'user', content: 'hi' }],
  temperature: 0.7,
  top_k: 40,
  min_p: 0.05,
  repetition_penalty: 1.1,
  speculative: 'draft',
  thinking_budget: 1024,
  chat_template_kwargs: { a: 1 },
  dry_multiplier: 0.8,
  dry_base: 1.75,
  dry_allowed_length: 2,
  dry_penalty_last_n: -1,
  mirostat: 2,
  mirostat_tau: 0.8,
  mirostat_eta: 0.001,
  grammar: 'x',
  grammar_type: 'y',
  regex: 'z',
  tool_call_grammar: 'w',
  tfs_z: 1.0,
  typical_p: 0.95,
  ignore_eos: true,
  cache_prompt: true,
  logit_bias_tokens: [1, 2],
  stream: true,
};
out.clean_request = [
  ['gpt-4o', cleanRemoteRequest(body, 'gpt-4o')],
  ['keeps what is not local', cleanRemoteRequest({ temperature: 0.2, top_k: 10, messages: [] }, 'm')],
];

// chatEndpoint: where a chat for a model id goes, and what the body becomes there
const sample = { model: 'x', messages: [], temperature: 0.5, top_k: 40, stream: true };
out.chat_endpoint = [
  ['local-model', describe(chatEndpoint('local-model'))],
  ['remote:openai/gpt-4o', describe(chatEndpoint('remote:openai/gpt-4o'))],
  ['remote:ollama/llama3:8b', describe(chatEndpoint('remote:ollama/llama3:8b'))],
];

function describe(ep) {
  return {
    url: ep.url,
    remote: ep.remote,
    provider: ep.provider === undefined ? null : ep.provider,
    model: ep.model === undefined ? null : ep.model,
    transform: ep.transform(sample),
  };
}

fs.writeFileSync(path.join(__dirname, 'providers_fixtures.json'), JSON.stringify(out, null, 1) + '\n');
console.log('providers_fixtures.json written');
