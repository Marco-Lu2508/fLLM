// Fixtures of Settings > Engine: engine.js of the Studio runs in a vm with the page stubbed. It records
//  - the rows of the pane (id of the control, title, description) as the page builds them, and
//  - for each scenario of server answers, the lines of live numbers and the state of the live controls.
// Run: node tests/fixtures/engine_fixtures.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const src = fs.readFileSync(path.join(WEBUI, 'engine.js'), 'utf8');
const core = fs.readFileSync(path.join(WEBUI, 'core.js'), 'utf8').split('\n');
const fbStart = core.findIndex((l) => l.includes('function formatBytes(bytes) {'));
const formatBytes = core.slice(fbStart, fbStart + 7).join('\n');

async function run(routes) {
  const els = {};
  const mk = (id) => (els[id] ||= { id, textContent: '', checked: false, disabled: false, value: '', style: {}, addEventListener() {} });
  let html = '';
  const pane = { className: '', id: '', querySelector: (sel) => mk(sel.slice(1)), set innerHTML(v) { html = v; }, get innerHTML() { return html; } };
  const button = { dataset: {}, className: '', set innerHTML(v) {}, addEventListener(type, fn) { if (type === 'click') button.cb = fn; } };
  const nav = { appendChild() {} };
  const document = {
    getElementById: (id) => (id === 'settingsMenuNav' ? nav : null),
    querySelector: () => ({ appendChild() {} }),
    createElement: (tag) => (tag === 'button' ? button : pane),
  };
  const apiFetch = async (url) => {
    const r = routes[url.split('?')[0]];
    if (r === 'throw') throw new Error('network');
    if (!r) return { ok: false, status: 404, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => r };
  };
  const ctx = { document, apiFetch, toast() {}, confirmDialog: async () => true, waitForRestart: async () => {}, $: () => ({ classList: { add() {}, remove() {} }, textContent: '' }), console };
  vm.createContext(ctx);
  vm.runInContext(`${formatBytes}\n${src}`, ctx, { timeout: 5000 });
  button.cb();
  await new Promise((r) => setTimeout(r, 30));
  const pick = (id) => (els[id] ? { text: els[id].textContent, checked: els[id].checked, disabled: els[id].disabled } : null);
  return { html, cacheStats: pick('engCacheStats'), otelStats: pick('engOtelStats'), kvStats: pick('engKvStats'), stateStats: pick('engStateStats'), drain: pick('engDrain'), cacheClear: pick('engCacheClear'), kvClear: pick('engKvClear'), prefixClear: pick('engPrefixClear') };
}

const FULL = {
  '/api/studio/engine': { defaults: {}, values: {} },
  '/admin/config': { output_cache: { entries: 12, bytes: 1048576, hits: 5, misses: 7, saved_prompt_tokens: 900 }, coalescing: { followers: 3 }, telemetry: { enabled: true, sent: 20, dropped: 1 }, draining: true },
  '/admin/kv-tier': { enabled: true, dir: '/fast', used_bytes: 5242880, max_bytes: 1073741824, ram_max_bytes: 268435456, ram_used_bytes: 1048576, ram_blocks: 4, ram_hits: 2, hits: 9, spills: 8, drops: 1 },
  '/admin/prefix-cache': { state_checkpoints: { enabled: true, entries: 3, bytes: 314572800, budget_bytes: 1073741824, hits: 6, tokens_skipped: 12000 } },
};
const scenarios = {
  full: FULL,
  no_coalescing_no_ram: { ...FULL, '/admin/config': { output_cache: {}, telemetry: {}, draining: false }, '/admin/kv-tier': { enabled: true, dir: '', hits: 0, spills: 0, drops: 0 } },
  kv_off: { ...FULL, '/admin/kv-tier': { enabled: false } },
  state_off: { ...FULL, '/admin/prefix-cache': { state_checkpoints: { enabled: false } } },
  state_missing: { ...FULL, '/admin/prefix-cache': {} },
  state_route_404: { ...FULL, '/admin/prefix-cache': undefined },
  no_model: { '/api/studio/engine': { defaults: {}, values: {} } },
  network_down: { '/api/studio/engine': 'throw', '/admin/config': 'throw', '/admin/kv-tier': 'throw', '/admin/prefix-cache': 'throw' },
};

(async () => {
  const out = { scenarios: {} };
  for (const [name, routes] of Object.entries(scenarios)) {
    const r = await run(routes);
    if (name === 'full') {
      const rows = [];
      const re = /<div class="app-card-item"><div class="app-item-info"><div class="app-item-title">(.*?)<\/div><div class="app-item-desc">(.*?)<\/div><\/div><div class="app-item-actions">(.*?)<\/div><\/div>/g;
      let m;
      while ((m = re.exec(r.html))) rows.push({ title: m[1], desc: m[2], id: (/id="(eng\w+)"/.exec(m[3]) || [])[1] || null });
      out.rows = rows;
      out.cards = [...r.html.matchAll(/<div class="app-card-title">(.*?)<\/div>/g)].map((x) => x[1]);
      out.buttons = [...r.html.matchAll(/<button class="btn [^"]*" id="(eng\w+)"[^>]*>(.*?)<\/button>/g)].map((x) => [x[1], x[2]]);
    }
    delete r.html;
    out.scenarios[name] = { routes: Object.fromEntries(Object.entries(routes).filter(([, v]) => v !== undefined)), ...r };
  }
  fs.writeFileSync(path.join(__dirname, 'engine_fixtures.json'), JSON.stringify(out, null, 1));
  console.log('written', Object.keys(out.scenarios).length, out.rows.length, JSON.stringify(out.cards), JSON.stringify(out.buttons));
})();
