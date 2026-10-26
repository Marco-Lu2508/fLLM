// Tests for the chat tool loop and its tools (tools.js), the memory tool and secret refusal (projects.js).
// Run from the repository root:  node rust/engine/src/webui/tests/tools.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const pure = (file) => {
  const src = fs.readFileSync(path.join(__dirname, '..', file), 'utf8');
  const b = src.indexOf('// @pure-begin');
  const e = src.indexOf('// @pure-end');
  assert(b >= 0 && e > b, `pure block markers not found in ${file}`);
  return src.slice(b, e);
};
const L = new Function(`const CT_MAX_ROUNDS = 5; const CT_MAX_CALLS = 4;\n${pure('projects.js')}\n${pure('tools.js')}
return { PRJ_LIMITS, memSecret, memAdd, memToolApply, memUndo, memSelect, memSanitize, ctDefs, ctParseArgs, ctTextCalls, ctBudget, ctClip,
  ctChatSearch, ctRunTool, ctLoop };`)();

let n = 0;
const queue = [];
const test = (name, fn) => queue.push([name, fn]);
let counter = 0;
const makeId = () => `id${++counter}`;

// ---- memory tool ------------------------------------------------------------------
test('memory tool: add, duplicate, update, remove, list', () => {
  let list = [];
  let r = L.memToolApply(list, { action: 'add', text: 'Lives in Lisbon', category: 'Profile!' }, 10, makeId);
  assert.strictEqual(r.text, 'Saved.');
  assert.strictEqual(r.change.kind, 'add');
  assert.strictEqual(r.list[0].category, 'profile');
  assert.strictEqual(r.list[0].createdAt, 10);
  list = r.list;
  r = L.memToolApply(list, { action: 'add', text: 'lives in lisbon' }, 11, makeId);
  assert.strictEqual(r.text, 'Already remembered.');
  assert.strictEqual(r.list.length, 1);
  assert.strictEqual(r.list[0].uses, 1, 'a repeated fact counts as used');
  assert.strictEqual(r.change, null);
  list = r.list;
  r = L.memToolApply(list, { action: 'update', match: 'lisbon', text: 'Lives in Porto' }, 20, makeId);
  assert.strictEqual(r.text, 'Updated.');
  assert.strictEqual(r.list[0].text, 'Lives in Porto');
  assert.strictEqual(r.list[0].createdAt, 10);
  assert.strictEqual(r.list[0].updatedAt, 20);
  list = r.list;
  assert(L.memToolApply(list, { action: 'list' }, 21, makeId).text.includes('Lives in Porto'));
  r = L.memToolApply(list, { action: 'remove', id: '1' }, 22, makeId); // by position
  assert.strictEqual(r.text, 'Removed.');
  assert.strictEqual(r.list.length, 0);
  assert.strictEqual(L.memToolApply([], { action: 'list' }, 1, makeId).text, 'The memory is empty.');
});

test('memory tool: errors tell the model what to do', () => {
  const list = [{ id: 'a', text: 'likes tea', createdAt: 1 }, { id: 'b', text: 'likes tea with milk', createdAt: 2 }];
  assert(L.memToolApply(list, { action: 'add' }, 1, makeId).text.startsWith('Error'));
  assert(L.memToolApply(list, { action: 'dance' }, 1, makeId).text.startsWith('Error'));
  assert(L.memToolApply(list, null, 1, makeId).text.startsWith('Error'));
  assert(L.memToolApply(list, { action: 'remove', match: 'coffee' }, 1, makeId).text.includes('no matching entry'));
  const many = L.memToolApply(list, { action: 'remove', match: 'tea' }, 1, makeId);
  assert(many.text.includes('2 entries match') && many.text.includes('[a]'));
  assert.strictEqual(many.list, list, 'nothing changed');
  assert(L.memToolApply(list, { action: 'update', id: 'a' }, 1, makeId).text.includes('"text" is required'));
  const full = Array.from({ length: L.PRJ_LIMITS.memories }, (_, i) => ({ id: `m${i}`, text: `fact ${i}`, createdAt: 0 }));
  assert(L.memToolApply(full, { action: 'add', text: 'one more' }, 1, makeId).text.includes('full'));
});

test('memory tool: secrets are refused on add and update', () => {
  for (const s of ['my password is hunter2xyz', 'API key: sk-abcdefghijklmnop1234', 'card 4111 1111 1111 1111', 'pin = 4821',
    'token is ghp_abcdefghijklmnopqrstuv', 'IBAN DE89 3704 0044 0532 0130 00', 'key aGVsbG8gd29ybGQgdGhpcyBpcyBsb25nIGVub3VnaA==']) {
    assert(L.memSecret(s), `should refuse: ${s}`);
    const r = L.memToolApply([], { action: 'add', text: s }, 1, makeId);
    assert(r.text.startsWith('Not saved'), s);
    assert.strictEqual(r.list.length, 0);
    assert.strictEqual(r.change, null);
  }
  const one = [{ id: 'a', text: 'likes tea', createdAt: 1 }];
  assert(L.memToolApply(one, { action: 'update', id: 'a', text: 'password: correcthorse' }, 2, makeId).text.startsWith('Not saved'));
  for (const s of ['Prefers metric units', 'Works as a nurse in Lyon', 'Born in 1990, two kids', 'The secret is out, they say', 'Phone model: Pixel 8 Pro', 'order number 12345678']) {
    assert(!L.memSecret(s), `should allow: ${s}`);
  }
});

test('memory: undo restores the state before each kind of change', () => {
  let list = [];
  const add = L.memToolApply(list, { action: 'add', text: 'Likes tea' }, 1, makeId);
  list = add.list;
  const upd = L.memToolApply(list, { action: 'update', match: 'tea', text: 'Likes coffee' }, 2, makeId);
  assert.deepStrictEqual(L.memUndo(upd.list, upd.change).map((m) => m.text), ['Likes tea']);
  const rem = L.memToolApply(list, { action: 'remove', match: 'tea' }, 3, makeId);
  assert.strictEqual(rem.list.length, 0);
  assert.deepStrictEqual(L.memUndo(rem.list, rem.change).map((m) => m.text), ['Likes tea']);
  assert.deepStrictEqual(L.memUndo(add.list, add.change), []);
  assert.strictEqual(L.memUndo([], add.change), null, 'already gone');
});

test('memory: stored entries from older versions get the new fields; the prompt prefers entries that were used', () => {
  const old = L.memSanitize([{ id: 'x', text: 'old fact', createdAt: 5 }, { text: '' }, null], makeId);
  assert.strictEqual(old.length, 1);
  assert.strictEqual(old[0].updatedAt, 5);
  assert.strictEqual(old[0].uses, 0);
  const mem = [
    { id: 'a', text: 'A'.repeat(40), createdAt: 1, uses: 3 },
    { id: 'b', text: 'B'.repeat(40), createdAt: 2 },
    { id: 'c', text: 'C'.repeat(40), createdAt: 3 },
  ];
  const sel = L.memSelect(mem, 90); // room for two
  assert.deepStrictEqual(sel.lines.map((l) => l[0]), ['A', 'C'], 'used first, then newest, original order');
  assert.strictEqual(sel.dropped, 1);
});

// ---- helpers ----------------------------------------------------------------------
test('tool definitions follow the flags and the size', () => {
  assert.deepStrictEqual(L.ctDefs({}, false), []);
  assert.deepStrictEqual(L.ctDefs({ web: true }, false).map((d) => d.function.name), ['web_search', 'web_fetch']);
  const all = L.ctDefs({ web: true, memory: true, chats: true }, true);
  assert.deepStrictEqual(all.map((d) => d.function.name), ['web_search', 'web_fetch', 'memory', 'search_chats']);
  assert(all.every((d) => d.type === 'function' && d.function.parameters.required.length));
  assert(JSON.stringify(L.ctDefs({ web: true }, true)).length < JSON.stringify(L.ctDefs({ web: true }, false)).length);
});

test('arguments: strict JSON, fenced JSON, empty, and rubbish', () => {
  assert.deepStrictEqual(L.ctParseArgs('{"query":"a"}'), { query: 'a' });
  assert.deepStrictEqual(L.ctParseArgs('```json\n{"query":"a"}\n```'), { query: 'a' });
  assert.deepStrictEqual(L.ctParseArgs(''), {});
  assert.strictEqual(L.ctParseArgs('{"query": '), null);
  assert.strictEqual(L.ctParseArgs('[1,2]'), null);
  assert.strictEqual(L.ctParseArgs('"text"'), null);
});

test('calls written as text are found, others are ignored', () => {
  const text = 'ok <tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call> <tool_call>{"name":"rm","arguments":{}}</tool_call> <tool_call>nope</tool_call>';
  const calls = L.ctTextCalls(text, ['web_search']);
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0].name, 'web_search');
  assert.deepStrictEqual(JSON.parse(calls[0].args), { query: 'x' });
});

test('an answer that is one bare JSON call is a call; prose and unknown tools are not', () => {
  const names = ['web_search', 'memory'];
  assert.deepStrictEqual(L.ctTextCalls('{"name": "web_search", "arguments": {"query": "rust"}}', names).map((c) => c.name), ['web_search']);
  assert.deepStrictEqual(L.ctTextCalls('```json\n[{"name":"memory","arguments":{"action":"list"}}]\n```', names).map((c) => c.name), ['memory']);
  assert.deepStrictEqual(L.ctTextCalls('{"name": "format_disk", "arguments": {}}', names), []);
  assert.deepStrictEqual(L.ctTextCalls('Sure: {"name": "web_search", "arguments": {}}', names), []);
});

test('the budget grows with the context and is bounded', () => {
  const small = L.ctBudget(4096);
  const big = L.ctBudget(131072);
  assert(small.per >= 1200 && small.per < big.per);
  assert.strictEqual(big.per, 12000);
  assert(small.total < 4096 * 3.4 * 0.5);
  assert.strictEqual(L.ctClip('abcdef', 3).startsWith('abc\n...'), true);
  assert.strictEqual(L.ctClip('abc', 5), 'abc');
});

test('chat search finds titles and text, skips the current and temporary chats', () => {
  const chat = (id, title, texts, extra = {}) => ({ id, title, updatedAt: 1000 + id.length, nodes: { root: { role: 'root' }, ...Object.fromEntries(texts.map((t, i) => [`n${i}`, { role: i % 2 ? 'assistant' : 'user', content: t }])) }, ...extra });
  const threads = [
    chat('a', 'Trip to Lisbon', ['Where should I stay in Alfama?', 'Try the guesthouse near the tram line.']),
    chat('b', 'Pasta', ['How long to boil penne?']),
    chat('c', 'Lisbon again', ['hidden'], { temporary: true }),
    chat('d', 'Current', ['Lisbon weather?']),
  ];
  const hits = L.ctChatSearch(threads, 'lisbon alfama', 'd', 4);
  assert.deepStrictEqual(hits.map((h) => h.id), ['a']);
  assert(hits[0].snippets.some((s) => s.includes('Alfama')));
  assert.deepStrictEqual(L.ctChatSearch(threads, 'zzz', 'd'), []);
  assert.deepStrictEqual(L.ctChatSearch(threads, '', 'd'), []);
});

// ---- running tools ----------------------------------------------------------------------
const json = (body, ok = true, status = 200) => ({ ok, status, json: async () => body });
function env(routes, extra = {}) {
  const calls = [];
  return {
    calls,
    apiFetch: async (url, opts) => { calls.push({ url, body: opts.body }); const r = routes[url]; if (!r) return json({}, false, 404); return typeof r === 'function' ? r(opts.body) : r; },
    signal: {}, sources: [], budget: { per: 3000, left: 6000 }, memory: null, chats: null, ...extra,
  };
}

test('web_search: numbered results, sources, a compact card, budget', async () => {
  const e = env({ '/api/studio/web/search': json({ ok: true, results: [{ title: 'Rust 1.90', url: 'https://blog.rust-lang.org/a', snippet: 'Released.' }, { title: 'Bad', url: 'javascript:alert(1)', snippet: '' }] }) });
  const r = await L.ctRunTool({ name: 'web_search', args: '{"query":"rust version"}' }, e);
  assert(r.text.includes('[1] Rust 1.90 (https://blog.rust-lang.org/a)'));
  assert(!r.text.includes('javascript:'));
  assert.strictEqual(r.card.label, 'Searched: rust version');
  assert.strictEqual(r.card.status, 'done');
  assert.deepStrictEqual(e.sources.map((s) => s.n), [1]);
  assert.strictEqual(e.calls[0].body.query, 'rust version');
  assert(e.budget.left < 6000);
});

test('web_fetch: reuses the number of a known source, clips to the budget, reports errors', async () => {
  const page = 'x'.repeat(5000);
  const e = env({ '/api/studio/web/fetch': json({ ok: true, url: 'https://a.example/p', text: page }) });
  e.sources.push({ n: 1, title: 'A', url: 'https://a.example/p', fetched: false });
  const r = await L.ctRunTool({ name: 'web_fetch', args: '{"url":"https://a.example/p"}' }, e);
  assert(r.text.startsWith('[1] https://a.example/p'));
  assert(r.text.length <= 3000 + 80 + 40, `clipped: ${r.text.length}`);
  assert.strictEqual(e.sources.length, 1);
  assert.strictEqual(e.sources[0].fetched, true);
  const bad = await L.ctRunTool({ name: 'web_fetch', args: '{"url":"file:///etc/passwd"}' }, e);
  assert(bad.text.startsWith('Error') && bad.card.status === 'error');
  const refused = await L.ctRunTool({ name: 'web_fetch', args: '{"url":"http://10.0.0.1/"}' }, env({ '/api/studio/web/fetch': json({ ok: false, error: 'the address is not public' }) }));
  assert(refused.text.includes('not public'));
  const none = await L.ctRunTool({ name: 'web_search', args: '{"query":"q"}' }, env({}));
  assert(none.text.includes('no web tools'));
});

test('invalid JSON arguments and unknown tools give the model an error to fix', async () => {
  const e = env({});
  const r = await L.ctRunTool({ name: 'web_search', args: '{"query": ' }, e);
  assert(r.text.includes('not valid JSON') && r.text.includes('{"query"'));
  assert.strictEqual(e.calls.length, 0);
  assert((await L.ctRunTool({ name: 'format_disk', args: '{}' }, e)).text.includes('unknown tool'));
  assert((await L.ctRunTool({ name: 'web_search', args: '{}' }, e)).text.includes('"query" is required'));
  e.budget.left = 100;
  assert((await L.ctRunTool({ name: 'web_search', args: '{"query":"a"}' }, e)).text.startsWith('Skipped'));
});

test('memory and search_chats through ctRunTool', async () => {
  let list = [];
  const memory = { apply: (args) => { const r = L.memToolApply(list, args, 5, makeId); list = r.list; return r; } };
  const e = env({}, { memory });
  const saved = await L.ctRunTool({ name: 'memory', args: '{"action":"add","text":"Name is Sam"}' }, e);
  assert.strictEqual(saved.card.label, 'Saved to memory: Name is Sam');
  assert.strictEqual(saved.card.change.kind, 'add');
  const secret = await L.ctRunTool({ name: 'memory', args: '{"action":"add","text":"password: swordfish99"}' }, e);
  assert.strictEqual(secret.card.status, 'error');
  assert.strictEqual(list.length, 1);
  assert((await L.ctRunTool({ name: 'memory', args: '{"action":"add","text":"x"}' }, env({}))).text.includes('switched off'));
  const denied = await L.ctRunTool({ name: 'search_chats', args: '{"query":"lisbon"}' }, env({}, { chats: { allow: async () => false, search: () => { throw new Error('must not search'); } } }));
  assert(denied.text.includes('did not allow'));
  const found = await L.ctRunTool({ name: 'search_chats', args: '{"query":"lisbon"}' }, env({}, { chats: { allow: async () => true, search: () => [{ id: 'a', title: 'Trip', updatedAt: 0, snippets: ['User: Lisbon?'] }] } }));
  assert(found.text.includes('"Trip"') && found.text.includes('Lisbon?'));
});

// ---- the loop ---------------------------------------------------------------------------
function scripted(steps, record = {}) {
  let i = 0;
  record.requests = [];
  return async (messages, opts) => {
    record.requests.push({ messages: messages.map((m) => ({ ...m })), tools: opts.tools, final: opts.final });
    const step = steps[Math.min(i++, steps.length - 1)];
    return typeof step === 'function' ? step(opts) : step;
  };
}
const call = (name, args, id) => ({ id, name, args: typeof args === 'string' ? args : JSON.stringify(args) });

test('loop: call, result, final answer', async () => {
  const rec = {};
  const ran = [];
  const out = await L.ctLoop({
    messages: [{ role: 'user', content: 'rust version?' }], tools: ['T'],
    stream: scripted([{ content: 'Let me check.', calls: [call('web_search', { query: 'rust' }, 'c1')] }, { content: 'It is 1.90 [1].', calls: [] }], rec),
    run: async (c) => { ran.push(c.name); return { text: 'RESULT' }; }, isStopped: () => false,
  });
  assert.deepStrictEqual(out, { endedBy: 'answer', rounds: 1 });
  assert.deepStrictEqual(ran, ['web_search']);
  const second = rec.requests[1].messages;
  assert.strictEqual(second[1].role, 'assistant');
  assert.strictEqual(second[1].tool_calls[0].function.name, 'web_search');
  assert.deepStrictEqual(second[2], { role: 'tool', tool_call_id: 'c1', content: 'RESULT' });
  assert.deepStrictEqual(rec.requests[0].tools, ['T']);
});

test('loop: a model that never calls tools just answers', async () => {
  const rec = {};
  const out = await L.ctLoop({ messages: [], tools: ['T'], stream: scripted([{ content: 'Hello', calls: [] }], rec), run: async () => { throw new Error('no'); }, isStopped: () => false });
  assert.deepStrictEqual(out, { endedBy: 'answer', rounds: 0 });
  assert.strictEqual(rec.requests.length, 1);
});

test('loop: at most five rounds of calls, then one answer without tools', async () => {
  const rec = {};
  let k = 0;
  const out = await L.ctLoop({
    messages: [{ role: 'user', content: 'q' }], tools: ['T'],
    stream: scripted([(opts) => (opts.final ? { content: 'Best effort.', calls: [call('web_search', { query: 'again' }, 'zz')] } : { content: '', calls: [call('web_search', { query: `q${k++}` }, `c${k}`)] })], rec),
    run: async () => ({ text: 'r' }), isStopped: () => false,
  });
  assert.strictEqual(out.endedBy, 'limit');
  assert.strictEqual(rec.requests.length, 6);
  const last = rec.requests[5];
  assert.strictEqual(last.final, true);
  assert.deepStrictEqual(last.tools, []);
  assert.strictEqual(last.messages[last.messages.length - 1].role, 'user');
});

test('loop: stop ends it, also between calls', async () => {
  let stopped = false;
  const ran = [];
  const out = await L.ctLoop({
    messages: [], tools: ['T'],
    stream: scripted([{ content: '', calls: [call('web_search', { query: 'a' }, '1'), call('web_search', { query: 'b' }, '2')] }]),
    run: async (c) => { ran.push(c.args); stopped = true; return { text: 'r' }; }, isStopped: () => stopped,
  });
  assert.strictEqual(out.endedBy, 'stopped');
  assert.strictEqual(ran.length, 1);
  const early = await L.ctLoop({ messages: [], tools: [], stream: async () => ({ content: 'x', calls: [call('a', {}, '1')] }), run: async () => ({ text: '' }), isStopped: () => true });
  assert.strictEqual(early.endedBy, 'stopped');
});

test('loop: an identical repeated call is not run again; many calls in one round are capped', async () => {
  const ran = [];
  const same = call('web_search', { query: 'same' });
  await L.ctLoop({
    messages: [], tools: ['T'],
    stream: scripted([{ content: '', calls: [same] }, { content: '', calls: [same] }, { content: 'done', calls: [] }]),
    run: async (c) => { ran.push(c.args); return { text: 'r' }; }, isStopped: () => false,
  });
  assert.strictEqual(ran.length, 1);
  const many = [];
  await L.ctLoop({
    messages: [], tools: ['T'],
    stream: scripted([{ content: '', calls: Array.from({ length: 9 }, (_, i) => call('web_search', { query: `q${i}` })) }, { content: 'ok', calls: [] }]),
    run: async (c) => { many.push(c.id); return { text: 'r' }; }, isStopped: () => false,
  });
  assert.strictEqual(many.length, 4);
  assert.strictEqual(new Set(many).size, 4, 'missing ids are made up and unique');
});

test('loop with the real tools and a faked fetch: search, read, answer', async () => {
  const e = env({
    '/api/studio/web/search': json({ ok: true, results: [{ title: 'Doc', url: 'https://d.example/x', snippet: 's' }] }),
    '/api/studio/web/fetch': json({ ok: true, url: 'https://d.example/x', text: 'The answer is 42.' }),
  });
  const rec = {};
  const out = await L.ctLoop({
    messages: [{ role: 'user', content: 'meaning?' }], tools: ['T'],
    stream: scripted([
      { content: '', calls: [call('web_search', { query: 'meaning' }, 'a')] },
      { content: '', calls: [call('web_fetch', { url: 'https://d.example/x' }, 'b')] },
      { content: 'It is 42 [1].', calls: [] },
    ], rec),
    run: (c) => L.ctRunTool(c, e), isStopped: () => false,
  });
  assert.strictEqual(out.rounds, 2);
  assert.deepStrictEqual(e.calls.map((c) => c.url), ['/api/studio/web/search', '/api/studio/web/fetch']);
  assert(rec.requests[2].messages.some((m) => m.role === 'tool' && m.content.includes('The answer is 42.')));
  assert.strictEqual(e.sources.length, 1);
});

(async () => {
  for (const [name, fn] of queue) { await fn(); n++; console.log(`ok - ${name}`); }
  console.log(`${n} tests passed`);
})().catch((e) => { console.error(e); process.exit(1); });
