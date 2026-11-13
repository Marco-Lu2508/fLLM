// Generates tests/fixtures/tools_fixtures.json: the expected results of the pure tool machinery of
// the Studio (rust/engine/src/webui/tools.js, the part between @pure-begin and @pure-end), produced
// BY that code — ctRun runs against a mock env (canned fetches), so the texts, cards, budget and
// sources of the web tools come from the original — so the Rust port in src/tools.rs can be checked
// against it.
// Run from this directory: node tools_fixtures.js > fixtures/tools_fixtures.json
'use strict';
const fs = require('fs');
const path = require('path');

const toolsPath = path.resolve(__dirname, '../../../../rust/engine/src/webui/tools.js');
const src = fs.readFileSync(toolsPath, 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('markers not found in tools.js');
// the two limits stand just above the marker: they belong to the block
const limits = src.indexOf('const CT_MAX_ROUNDS');
const start = Math.min(src.lastIndexOf('\n', begin) + 1, limits < 0 ? Infinity : src.lastIndexOf('\n', limits) + 1);
const pure = src.slice(start, src.lastIndexOf('\n', end) + 1);

const sandbox = new Function(
  pure + '\n  return { CT_MAX_ROUNDS, CT_MAX_CALLS, CT_EXAMPLE, ctBudget, ctClip, ctHost, ctShort, ctHttpError, ctSourceNumber, ctDefs, ctParseArgs, ctLabel, ctRunTool };',
)();
const {
  CT_MAX_ROUNDS, CT_MAX_CALLS, CT_EXAMPLE, ctBudget, ctClip, ctHost, ctShort, ctHttpError,
  ctSourceNumber, ctDefs, ctParseArgs, ctLabel, ctRunTool,
} = sandbox;

// ---- helpers ------------------------------------------------------------------------------
function mockEnv(spec) {
  // spec: { left, per, pages: { url: {ok, text, url, error, status} }, search: {ok, results, error, status}, memory }
  const budget = { per: spec.per ?? 2000, left: spec.left ?? 5000 };
  const sources = spec.sources ? JSON.parse(JSON.stringify(spec.sources)) : [];
  const apiFetch = async (route, opts) => {
    const body = opts && opts.body ? opts.body : {};
    if (route === '/api/studio/web/search') {
      const s = spec.search || { ok: true, results: [] };
      return { ok: s.status ? s.status < 400 : true, status: s.status || 200, json: async () => JSON.parse(JSON.stringify(s)) };
    }
    if (route === '/api/studio/web/fetch') {
      const p = (spec.pages || {})[body.url] || { ok: false, status: 404, error: 'not found' };
      return { ok: p.status ? p.status < 400 : true, status: p.status || 200, json: async () => JSON.parse(JSON.stringify(p)) };
    }
    throw new Error('unexpected route ' + route);
  };
  const canned = spec.memory || {};
  const memory = { apply: () => ({ text: canned.text || 'Already remembered.', change: canned.change || null }) };
  return { apiFetch, signal: null, budget, sources, memory, chats: { allow: () => false, search: () => [] } };
}

async function run(name, args, spec) {
  const env = mockEnv(spec || {});
  const call = { name, args: typeof args === 'string' ? args : JSON.stringify(args) };
  const out = await ctRunTool(call, env);
  return { name, args, spec: spec || {}, out, after: { left: env.budget.left, sources: env.sources } };
}

// ---- fixtures ------------------------------------------------------------------------------
(async () => {
  const cases = [];
  // the pure helpers
  cases.push({ kind: 'budget', outputs: [4096, 16384, 200, 0, 999999].map((c) => ctBudget(c)) });
  cases.push({ kind: 'clip', outputs: [['short', 10], ['a much longer text that has to be cut in the middle', 20], ['', 5]].map(([t, m]) => ctClip(t, m)) });
  cases.push({ kind: 'short', outputs: [['  spaced   out  ', 100], ['x'.repeat(50), 10], ['', 5]].map(([s, n]) => ctShort(s, n)) });
  cases.push({ kind: 'host', outputs: ['https://example.com:8443/a/b', 'not a url', 'ftp://x.test/p'].map((u) => ctHost(u)) });
  cases.push({ kind: 'http', outputs: [[404, ''], [403, 'because'], [500, ''], [401, '']].map(([s, d]) => ctHttpError(s, d)) });
  cases.push({ kind: 'limits', rounds: CT_MAX_ROUNDS, calls: CT_MAX_CALLS, example: CT_EXAMPLE });
  cases.push({ kind: 'defs', outputs: [
    [{ web: true, memory: false, chats: false }, false],
    [{ web: false, memory: true, chats: false }, false],
    [{ web: true, memory: true, chats: true }, true],
    [{ web: false, memory: false, chats: false }, false],
  ].map(([f, c]) => ctDefs(f, c)) });
  cases.push({ kind: 'parse', outputs: ['{"a": 1}', '```json\n{"a": 1}\n```', 'not json', '', '  {"query":"x"}  '].map((s) => ctParseArgs(s)) });
  cases.push({ kind: 'label', outputs: [
    [{ query: 'rust version' }, false], [{ query: 'rust version' }, true],
    [{ url: 'https://example.com/page' }, false], [{ url: 'https://example.com/page' }, true],
  ].map(([a, d], i) => (i < 2 ? ctLabel.web_search : ctLabel.web_fetch)(a, d)) });
  // source numbering: same url twice, then a new one, then a fetched page
  const srcEnv = { sources: [] };
  const n1 = ctSourceNumber(srcEnv.sources, 'https://a.test/one', 'One', false);
  const n2 = ctSourceNumber(srcEnv.sources, 'https://a.test/two', 'Two', false);
  const n3 = ctSourceNumber(srcEnv.sources, 'https://a.test/one', 'One again', false);
  const n4 = ctSourceNumber(srcEnv.sources, 'https://a.test/page', 'a.test', true);
  cases.push({ kind: 'sources', numbers: [n1, n2, n3, n4], list: srcEnv.sources });

  // ctRun against a mock server: the texts, cards, budget and sources of the Studio
  cases.push(await run('web_search', { query: 'fLLM release' }, {
    search: { ok: true, results: [
      { url: 'https://example.com/blog', title: 'The fLLM blog', snippet: 'Version 2.0 was released today.' },
      { url: 'not-a-url', title: 'Ignored', snippet: 'no protocol' },
      { url: 'https://other.test/post', title: 'Another take', snippet: 'It is fast.' },
    ] },
  }));
  cases.push(await run('web_search', { query: 'nothing here' }, { search: { ok: true, results: [] } }));
  cases.push(await run('web_search', { query: 'blocked' }, { search: { ok: false, status: 404, error: 'gone' } }));
  cases.push(await run('web_search', {}, { search: { ok: true, results: [] } }));
  cases.push(await run('web_search', { query: 'no room' }, { left: 300, search: { ok: true, results: [] } }));
  cases.push(await run('web_fetch', { url: 'https://example.com/doc' }, {
    pages: { 'https://example.com/doc': { ok: true, url: 'https://example.com/doc', text: 'The page says hello.' } },
  }));
  cases.push(await run('web_fetch', { url: 'ftp://nope' }, {}));
  cases.push(await run('web_fetch', { url: 'https://empty.test/' }, {
    pages: { 'https://empty.test/': { ok: true, url: 'https://empty.test/', text: '   ' } },
  }));
  cases.push(await run('web_fetch', { url: 'https://missing.test/' }, {
    pages: { 'https://missing.test/': { ok: false, status: 500, error: 'boom' } },
  }));
  cases.push(await run('nope', { x: 1 }, {}));
  cases.push(await run('memory', 'not json at all', {}));
  cases.push(await run('memory', { action: 'add', text: 'Flies the DA40' }, {
    memory: { text: 'Remembered.', change: { kind: 'add', id: 'm1', before: null, after: { id: 'm1', text: 'Flies the DA40', created_at: 1, updated_at: 1, uses: 0, category: null } } },
  }));

  process.stdout.write(JSON.stringify(cases, null, 2) + '\n');
})();
