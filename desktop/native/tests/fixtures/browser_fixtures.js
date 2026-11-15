#!/usr/bin/env node
// Row 85 fixtures: the page's own browser.js (the live view of the browser page and Settings > Browser)
// is run in a vm with a stub world — elements that record their listeners, a fetch that answers from
// a table, timers that run when told to — and the scenarios below drive it the way a user would.
// What each step leaves behind (the requests sent, the words and flags of the panel, the
// comments text) is the fixture the Rust port in src/browser.rs must reproduce.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const SRC = fs.readFileSync(path.join(WEBUI, 'browser.js'), 'utf8');

function boot({ platform = 'Linux x86_64', respond = () => ({}), stageRect = { left: 100, top: 50, width: 640, height: 400 }, imgWidth = 640 } = {}) {
  const els = new Map();
  const calls = [];
  const toasts = [];
  const timers = [];
  const confirms = [];
  let confirmAnswer = true;
  const mk = (name) => {
    const handlers = {};
    const attrs = {};
    const classes = new Set();
    const e = {
      __name: name, __handlers: handlers, hidden: false, innerHTML: '', textContent: '', value: '', disabled: false, checked: false,
      style: {}, dataset: {}, className: '', clientWidth: imgWidth, id: name,
      classList: { toggle(c, on) { if (on === undefined ? !classes.has(c) : on) classes.add(c); else classes.delete(c); }, add(c) { classes.add(c); }, remove(c) { classes.delete(c); }, contains: (c) => classes.has(c) },
      addEventListener(t, f) { (handlers[t] = handlers[t] || []).push(f); },
      setAttribute(k, v) { attrs[k] = String(v); }, getAttribute(k) { return k in attrs ? attrs[k] : null; }, removeAttribute(k) { delete attrs[k]; },
      querySelector(sel) { return get(`${name}>${sel}`); }, closest() { return null; },
      getBoundingClientRect() { return stageRect; }, focus() { e.__focused = true; }, click() { e.__clicked = true; }, appendChild() {}, __classes: classes, __attrs: attrs,
    };
    Object.defineProperty(e, 'src', { get() { return attrs.src ?? ''; }, set(v) { attrs.src = String(v); } });
    return e;
  };
  function get(id) { if (!els.has(id)) els.set(id, mk(id)); return els.get(id); }
  const surface = get('surface-code');
  surface.classList.add('active');
  const document = {
    hidden: false, activeElement: null,
    getElementById: (id) => get(id),
    querySelector: (sel) => get(`doc:${sel}`),
    createElement: (tag) => get(`new:${tag}:${els.size}`),
    body: get('body'),
  };
  const fakeSetTimeout = (fn, ms) => { const t = { fn, ms, id: timers.length + 1, live: true }; timers.push(t); return t.id; };
  const fakeClearTimeout = (id) => { const t = timers.find((x) => x.id === id); if (t) t.live = false; };
  const apiFetch = async (p, opts = {}) => {
    const body = opts.body === undefined ? undefined : JSON.parse(JSON.stringify(opts.body));
    const rec = { path: p, method: opts.method || 'GET', body };
    calls.push(rec);
    const out = respond(p, body, calls);
    rec.ok = !(out && out.__status && out.__status >= 400);
    rec.response = JSON.parse(JSON.stringify(out && out.__body !== undefined ? out.__body : (out ?? null)));
    return { ok: out && out.__status ? out.__status < 400 : true, json: async () => (out && out.__body !== undefined ? out.__body : out) };
  };
  const context = {
    console, JSON, Math, Set, Map, Array, Object, String, Number, Promise, Error, RegExp, Date, Boolean,
    document, navigator: { platform }, apiFetch, toast: (m) => toasts.push(m), confirmDialog: async (m) => { confirms.push(m); return confirmAnswer; },
    agentRoutesUp: () => true, escapeHtml: (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
    setTimeout: fakeSetTimeout, clearTimeout: fakeClearTimeout, window: {},
  };
  context.window = context; // the page's window is its global
  vm.createContext(context);
  vm.runInContext(SRC, context, { filename: 'browser.js' });
  return { els, get, calls, toasts, timers, confirms, context, setConfirm: (v) => { confirmAnswer = v; }, document };
}

const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); await new Promise((r) => setImmediate(r)); };
const fire = async (el, type, ev = {}) => { for (const f of (el.__handlers[type] || [])) await f({ preventDefault() {}, target: { closest: () => null }, ...ev }); await flush(); };
const runTimers = async (w) => { const live = w.timers.filter((t) => t.live); for (const t of live) { t.live = false; await t.fn(); } await flush(); };

const panelState = (w) => {
  const g = w.get;
  const s = {
    state: g('brState').textContent,
    empty: { hidden: g('brEmpty').hidden, html: g('brEmpty').innerHTML },
    img: { hidden: g('brImg').hidden, src: g('brImg').getAttribute('src') },
    liveDot: g('btnCodeBrowser>.br-live').hidden,
    disabled: Object.fromEntries(['brBack', 'brForward', 'brReload', 'brStop', 'brComment'].map((id) => [id, g(id).disabled])),
    urlDisabled: g('brUrl').disabled, url: g('brUrl').value,
    foot: g('brFoot').textContent, commenting: g('brStage').__classes.has('br-commenting'), pressed: g('brComment').__attrs['aria-pressed'],
    comments: g('brComments').innerHTML, note: { hidden: g('brNote').hidden, html: g('brNote').innerHTML },
    pin: { hidden: g('brPin').hidden, style: { ...g('brPin').style } },
    open: { hidden: g('codeBrowser').hidden, pressed: g('btnCodeBrowser').__attrs['aria-pressed'] },
  };
  return JSON.parse(JSON.stringify(s));
};

async function scenario(name, opts, steps) {
  const w = boot(opts);
  const out = [];
  const ctx = { w, g: w.get };
  await flush();
  out.push({ step: 'boot (after load)', state: panelState(w), calls: JSON.parse(JSON.stringify(w.calls)), toasts: [...w.toasts] });
  w.calls.length = 0; w.toasts.length = 0;
  for (const st of steps) {
    await st.run(ctx);
    const extra = st.extra ? st.extra(ctx) : undefined;
    out.push({ step: st.name, action: st.action, state: panelState(w), calls: JSON.parse(JSON.stringify(w.calls)), toasts: [...w.toasts], confirms: [...w.confirms], extra });
    w.calls.length = 0; w.toasts.length = 0; w.confirms.length = 0;
  }
  return { name, opts: { platform: opts.platform || 'Linux x86_64', stageRect: opts.stageRect || { left: 100, top: 50, width: 640, height: 400 } }, steps: out };
}


const P = '/api/studio/agent/browser';
const respondWith = (table) => (p, body, calls) => { const f = table[p]; return typeof f === 'function' ? f(body, calls) : (f ?? {}); };
const ev = (x, y, more = {}) => ({ clientX: x, clientY: y, ...more });
const closest = (sel) => ({ target: { closest: (s) => (s === sel ? { dataset: {} } : null) } });
const A = {
  open: () => ({ name: 'open the panel', action: { kind: 'open' }, run: async ({ w }) => { await fire(w.get('btnCodeBrowser'), 'click'); } }),
  hide: () => ({ name: 'hide the panel', action: { kind: 'hide' }, run: async ({ w }) => { await fire(w.get('brHide'), 'click'); } }),
  tick: () => ({ name: 'a tick', action: { kind: 'tick' }, run: async ({ w }) => { await runTimers(w); } }),
  click: (name, x, y, over) => ({ name, action: { kind: 'click', x, y, onButton: Boolean(over) }, run: async ({ w }) => { await fire(w.get('brStage'), 'click', over ? { ...ev(x, y), target: { closest: (s) => (s === 'button' ? {} : null) } } : ev(x, y)); } }),
  dblclick: (name, x, y) => ({ name, action: { kind: 'dblclick', x, y }, run: async ({ w }) => { await fire(w.get('brStage'), 'dblclick', ev(x, y)); } }),
  wheel: (name, evs) => ({ name, action: { kind: 'wheel', events: evs }, run: async ({ w }) => { for (const e of evs) await fire(w.get('brStage'), 'wheel', ev(e.x, e.y, { deltaX: e.dx, deltaY: e.dy, deltaMode: e.mode })); await runTimers(w); } }),
  key: (name, e) => ({ name, action: { kind: 'key', ...e }, run: async ({ w }) => { await fire(w.get('brStage'), 'keydown', { key: e.key, ctrlKey: !!e.ctrl, altKey: !!e.alt, shiftKey: !!e.shift, metaKey: !!e.meta, isComposing: !!e.composing }); } }),
  paste: (name, text) => ({ name, action: { kind: 'paste', text }, run: async ({ w }) => { await fire(w.get('brStage'), 'paste', { clipboardData: { getData: () => text } }); } }),
  button: (name, id) => ({ name, action: { kind: id }, run: async ({ w }) => { await fire(w.get(id), 'click'); } }),
  submit: (name, url) => ({ name, action: { kind: 'submit', url }, run: async ({ w }) => { w.get('brUrl').value = url; await fire(w.get('brBar'), 'submit'); } }),
  close: (name, confirm) => ({ name, action: { kind: 'close', confirm }, run: async ({ w }) => { w.setConfirm(confirm); await fire(w.get('brStop'), 'click'); } }),
  comment: () => ({ name: 'comment mode', action: { kind: 'comment' }, run: async ({ w }) => { await fire(w.get('brComment'), 'click'); } }),
  noteAdd: (name, note) => ({ name, action: { kind: 'note_add', note }, run: async ({ w }) => { w.get('brNoteText').value = note; await fire(w.get('codeBrowser'), 'click', closest('#brNoteAdd')); } }),
  noteCancel: () => ({ name: 'cancel the note', action: { kind: 'note_cancel' }, run: async ({ w }) => { await fire(w.get('codeBrowser'), 'click', closest('#brNoteCancel')); } }),
  del: (i) => ({ name: `delete comment ${i}`, action: { kind: 'del', i }, run: async ({ w }) => { await fire(w.get('codeBrowser'), 'click', { target: { closest: (s) => (s === '[data-br-del]' ? { dataset: { brDel: String(i) } } : null) } }); } }),
  take: (name = 'take the comments') => ({ name, action: { kind: 'take' }, run: async () => {}, extra: ({ w }) => w.context.window.fllmBrowserTakeComments() }),
  send: () => ({ name: 'send to the agent', action: { kind: 'send' }, run: async ({ w }) => { await fire(w.get('codeBrowser'), 'click', closest('#brSend')); }, extra: ({ w }) => ({ box: w.get('codeTextarea').value, clicked: Boolean(w.get('btnCodeSend').__clicked) }) }),
  url: (name = 'the address for the rules') => ({ name, action: { kind: 'url' }, run: async () => {}, extra: ({ w }) => w.context.window.fllmBrowserUrl() }),
};
const STATUS = (over = {}, sets = {}) => ({ browser: '/usr/bin/chromium', env_browser: false, running: false, url: '', profile: '/data/browser-profile', settings: { enabled: true, show_window: false, persistent_profile: true, local_addresses: 'auto', executable: '', ...sets }, ...over });
const FRAME = (over = {}) => ({ running: true, image: 'data:image/jpeg;base64,AAAA', key: 'k1', width: 1280, height: 800, url: 'http://localhost:3000/', title: 'App', ...over });
const live = { [P]: STATUS({ running: true, url: 'http://localhost:3000/' }), [`${P}/frame`]: FRAME(), [`${P}/input`]: { ok: true } };
const frameCount = (calls) => calls.filter((c) => c.path.endsWith('/frame')).length;
const located = { ok: true, ref: 7, role: 'button', name: 'Save changes to the account now please this is long', tag: 'button', text: '  Save   changes \n now ', url: 'http://localhost:3000/settings', rect: { x: 128, y: 80, w: 256, h: 40 } };
const letters = ['a', 'A', ' ', 'Enter'];

(async () => {
  const scenarios = [];
  const run = async (name, opts, steps) => { scenarios.push(await scenario(name, opts, steps)); };
  const R = (t) => ({ respond: respondWith(t) });

  await run('the browser is switched off', R({ [P]: STATUS({}, { enabled: false }) }), [A.open(), A.tick()]);
  await run('no browser on this computer (linux)', { platform: 'Linux x86_64', ...R({ [P]: STATUS({ browser: null, browser_error: 'none' }) }) }, [A.open()]);
  await run('no browser on this computer (windows)', { platform: 'Win32', ...R({ [P]: STATUS({ browser: null }) }) }, [A.open()]);
  await run('no browser on this computer (mac)', { platform: 'MacIntel', ...R({ [P]: STATUS({ browser: '' }) }) }, [A.open()]);
  await run('ready, nothing open', R({ [P]: STATUS(), [`${P}/frame`]: { running: false } }), [A.open(), A.tick()]);
  await run('the status is not there', R({ [P]: { __status: 500, __body: {} } }), [A.open(), A.tick()]);
  await run('a page is shown', R(live), [A.open(), A.tick(), A.hide()]);
  await run('window shown and about:blank', R({ ...live, [P]: STATUS({ running: true }, { show_window: true }), [`${P}/frame`]: FRAME({ url: 'about:blank' }) }), [A.open(), A.tick()]);
  await run('a page that cannot be drawn', R({ ...live, [`${P}/frame`]: { running: true, undrawable: true, url: 'chrome-error://chromewebdata/' } }), [A.open(), A.tick()]);
  await run('an undrawable page without an address', R({ ...live, [`${P}/frame`]: { running: true, undrawable: true } }), [A.open(), A.tick()]);
  await run('a frame with an error keeps what it had', R({ ...live, [`${P}/frame`]: (b, calls) => (frameCount(calls) > 1 ? { running: true, error: 'busy' } : FRAME()) }), [A.open(), A.tick(), A.tick()]);
  await run('the frame asks since the last key', R({ ...live, [`${P}/frame`]: (b, calls) => FRAME({ key: `k${frameCount(calls)}`, width: 1000, height: 500 }) }), [A.open(), A.tick(), A.tick(), A.tick()]);
  await run('the browser ends', R({ ...live, [`${P}/frame`]: (b, calls) => (frameCount(calls) > 1 ? { running: false } : FRAME()) }), [A.open(), A.tick(), A.tick()]);
  await run('a frame without an image or a size keeps the last', R({ ...live, [`${P}/frame`]: (b, calls) => (frameCount(calls) > 1 ? { running: true, key: 'same', url: 'http://localhost:3000/next' } : FRAME()) }), [A.open(), A.tick(), A.tick()]);

  await run('clicks and wheel', R(live), [
    A.open(), A.tick(),
    A.click('click in the middle', 420, 250), A.click('click at a corner', 100, 50), A.click('click on a button of the stage', 420, 250, true), A.click('click outside the picture', 5000, 5000),
    A.dblclick('double click', 300, 150),
    A.wheel('wheel lines', [{ x: 420, y: 250, dx: 0, dy: 3, mode: 1 }, { x: 420, y: 250, dx: 1, dy: 2, mode: 1 }]),
    A.wheel('wheel pixels', [{ x: 740, y: 450, dx: -4, dy: 120, mode: 0 }]),
    A.wheel('wheel pages', [{ x: 420, y: 250, dx: 0, dy: 1, mode: 2 }]),
    A.wheel('wheel moves on', [{ x: 200, y: 100, dx: 0, dy: 10, mode: 0 }, { x: 300, y: 150, dx: 0, dy: 10, mode: 0 }]),
  ]);
  const K = (name, e) => A.key(name, e);
  await run('keys', R(live), [
    A.open(), A.tick(),
    K('a letter', { key: 'a' }), K('a capital', { key: 'A', shift: true }), K('space', { key: ' ' }), K('enter', { key: 'Enter' }),
    K('ctrl a', { key: 'a', ctrl: true }), K('ctrl shift Tab', { key: 'Tab', ctrl: true, shift: true }), K('shift Tab', { key: 'Tab', shift: true }),
    K('alt f', { key: 'f', alt: true }), K('meta c', { key: 'c', meta: true }), K('ctrl alt shift x', { key: 'x', ctrl: true, alt: true, shift: true }),
    K('F5', { key: 'F5' }), K('F12', { key: 'F12' }), K('F13 is no key', { key: 'F13' }), K('F0 is no key', { key: 'F0' }), K('Home', { key: 'Home' }), K('ArrowLeft', { key: 'ArrowLeft' }),
    K('Insert', { key: 'Insert' }), K('Escape', { key: 'Escape' }), K('Backspace', { key: 'Backspace' }), K('Delete', { key: 'Delete' }), K('PageDown', { key: 'PageDown' }), K('PageUp', { key: 'PageUp' }), K('End', { key: 'End' }),
    K('ArrowUp', { key: 'ArrowUp' }), K('ArrowDown', { key: 'ArrowDown' }), K('ArrowRight', { key: 'ArrowRight' }),
    K('shift alone', { key: 'Shift' }), K('control alone', { key: 'Control' }), K('alt alone', { key: 'Alt' }), K('meta alone', { key: 'Meta' }), K('caps lock', { key: 'CapsLock' }), K('dead key', { key: 'Dead' }), K('process key', { key: 'Process' }), K('composing', { key: 'a', composing: true }),
    K('ctrl v is the paste event', { key: 'v', ctrl: true }), K('meta V is the paste event', { key: 'V', meta: true }), K('ctrl alt v is a key', { key: 'v', ctrl: true, alt: true }),
    K('an emoji', { key: '\u{1F600}' }), K('a name that is no key', { key: 'ContextMenu' }), K('umlaut', { key: '\u00e4' }), K('shift 1', { key: '!', shift: true }), K('ctrl shift 1', { key: '!', ctrl: true, shift: true }),
    A.paste('paste', 'hello\nworld'), A.paste('paste nothing', ''),
  ]);
  await run('keys and clicks while nothing runs', R({ [P]: STATUS(), [`${P}/frame`]: { running: false } }), [
    A.open(), A.tick(), K('a letter', { key: 'a' }), A.click('click', 300, 200), A.paste('paste', 'x'), A.wheel('wheel', [{ x: 300, y: 200, dx: 0, dy: 5, mode: 0 }]),
  ]);

  await run('the bar', R(live), [
    A.open(), A.tick(), A.button('back', 'brBack'), A.button('forward', 'brForward'), A.button('reload', 'brReload'),
    A.submit('go with an address', '  localhost:3000/app  '), A.submit('go with nothing', '   '),
    A.close('close is asked first', false), A.close('close confirmed', true),
  ]);
  await run('a refusal of the input is a toast', R({ ...live, [`${P}/input`]: { ok: false, error: 'That address is not allowed.' } }), [A.open(), A.tick(), A.submit('go', 'http://10.0.0.1'), A.click('a click that is refused', 420, 250)]);
  await run('an input that fails to arrive', R({ ...live, [`${P}/input`]: () => { throw new Error('connection refused'); } }), [A.open(), A.tick(), A.click('click', 420, 250)]);

  await run('comments', R({ ...live, [`${P}/input`]: (b) => (b.type === 'locate' ? located : { ok: true }) }), [
    A.open(), A.tick(), A.comment(), A.click('click an element', 420, 250),
    A.noteAdd('add without a note', '   '), A.noteAdd('add with a note', ' make it blue '),
    A.click('another one', 300, 200), A.noteCancel(),
    A.click('a third one', 300, 200), A.noteAdd('add the second', 'second <b>one</b>'),
    A.del(0), A.take(), A.take('take them again'), A.dblclick('double click in comment mode does nothing', 300, 150), A.comment(),
  ]);
  await run('a comment with no name and no text', R({ ...live, [`${P}/input`]: (b) => (b.type === 'locate' ? { ok: true, ref: 2, role: 'generic', rect: { x: 0, y: 0, w: 0, h: 0 } } : { ok: true }) }), [
    A.open(), A.tick(), A.comment(), A.click('click', 420, 250), A.noteAdd('add', 'x'), A.take(),
  ]);
  await run('a comment whose text is its name', R({ ...live, [`${P}/input`]: (b) => (b.type === 'locate' ? { ok: true, ref: 3, role: 'link', name: 'Home', tag: 'a', text: 'Home', rect: { x: 10, y: 10, w: 50, h: 20 } } : { ok: true }) }), [
    A.open(), A.tick(), A.comment(), A.click('click', 420, 250), A.noteAdd('add', 'x'), A.take(),
  ]);
  await run('a failed locate', R({ ...live, [`${P}/input`]: (b) => (b.type === 'locate' ? { ok: false, error: 'No element there.' } : { ok: true }) }), [A.open(), A.tick(), A.comment(), A.click('click', 420, 250)]);
  await run('send to the agent', R(live), [A.open(), A.send()]);
  await run('the address for the rules', R(live), [A.url('before anything'), A.open(), A.tick(), A.url('with a page')]);


  // ---- Settings > Browser ----
  const settingsScenario = async (name, opts, steps) => {
    const w = boot(opts);
    await flush();
    const pane = [...w.els.values()].find((e) => String(e.innerHTML).includes('Browser for the coding agent'));
    const q = (sel) => w.get(`${pane.__name}>${sel}`);
    const snap = () => JSON.parse(JSON.stringify({
      enabled: q('#brsEnabled').checked, show: q('#brsShow').checked, keep: q('#brsKeep').checked, local: q('#brsLocal').value, exe: q('#brsExe').value,
      found: q('#brsFound').textContent, foundColor: q('#brsFound').style.color ?? null, profile: q('#brsProfile').textContent, msg: q('#brsMsg').textContent,
    }));
    const out = [{ step: 'load', action: { kind: 'load' }, state: snap(), calls: JSON.parse(JSON.stringify(w.calls)), toasts: [...w.toasts], confirms: [] }];
    w.calls.length = 0;
    for (const st of steps) {
      await st.run({ w, q });
      out.push({ step: st.name, action: st.action, state: snap(), calls: JSON.parse(JSON.stringify(w.calls)), toasts: [...w.toasts], confirms: [...w.confirms] });
      w.calls.length = 0; w.confirms.length = 0;
    }
    return { name, paneHtml: pane.innerHTML, steps: out };
  };
  const S = {
    enable: (name, checked, confirm) => ({ name, action: { kind: 'enable', checked, confirm }, run: async ({ w, q }) => { w.setConfirm(confirm); q('#brsEnabled').checked = checked; await fire(q('#brsEnabled'), 'change', { target: q('#brsEnabled') }); } }),
    save: (name, v) => ({ name, action: { kind: 'save', ...v }, run: async ({ q }) => { q('#brsShow').checked = v.show; q('#brsKeep').checked = v.keep; q('#brsLocal').value = v.local; q('#brsExe').value = v.exe; await fire(q('#brsSave'), 'click'); } }),
  };
  const settingsFor = (over = {}, sets = {}) => ({ ...STATUS(over, sets), settings: { enabled: false, show_window: true, persistent_profile: false, local_addresses: 'all', executable: '/opt/br', ...sets } });
  const settingsSaved = (b) => ({ ...settingsFor({}, { enabled: true, ...b }) });
  const settingsList = [];
  settingsList.push(await settingsScenario('a found browser', R({ [P]: settingsFor({ browser: '/usr/bin/chromium', profile: '/data/profile' }), [`${P}/settings`]: (b) => settingsSaved(b) }), [
    S.enable('enable and say yes', true, true), S.enable('enable and say no', true, false), S.enable('disable', false, true),
    S.save('save the settings', { show: true, keep: false, local: 'off', exe: '  /opt/chrome  ' }),
  ].map((x) => x)));
  settingsList.push(await settingsScenario('a browser from the environment', R({ [P]: settingsFor({ browser: '/x/chrome', env_browser: true }) }), []));
  settingsList.push(await settingsScenario('no browser, with the reason', R({ [P]: settingsFor({ browser: null, browser_error: 'Nothing named chrome was found.' }) }), []));
  settingsList.push(await settingsScenario('no browser, no reason', R({ [P]: settingsFor({ browser: null }) }), []));
  settingsList.push(await settingsScenario('the server has no model', R({ [P]: { __status: 409, __body: { error: { message: 'Load a model first.' } } } }), []));
  settingsList.push(await settingsScenario('the server answers with no message', R({ [P]: { __status: 500, __body: {} } }), []));
  settingsList.push(await settingsScenario('the server is not there', R({ [P]: () => { throw new Error('connection refused'); } }), []));
  settingsList.push(await settingsScenario('saving is refused', R({ [P]: settingsFor({ browser: '/b' }), [`${P}/settings`]: { __status: 400, __body: { error: { message: 'That program does not exist.' } } } }), [
    S.save('save', { show: false, keep: true, local: 'auto', exe: 'nope' }), S.enable('enable', true, true),
  ]));
  settingsList.push(await settingsScenario('saving is refused without a message', R({ [P]: settingsFor({ browser: '/b' }), [`${P}/settings`]: { __status: 500, __body: {} } }), [S.save('save', { show: false, keep: true, local: 'auto', exe: '' })]));
  settingsList.push(await settingsScenario('saving does not reach the server', R({ [P]: settingsFor({ browser: '/b' }), [`${P}/settings`]: () => { throw new Error('offline'); } }), [S.save('save', { show: false, keep: true, local: 'auto', exe: '' })]));
  const settingsAnswer = (b) => settingsSaved(b);
  settingsList.push(await settingsScenario('saving answers with the new settings', R({ [P]: settingsFor({ browser: '/b' }), [`${P}/settings`]: settingsAnswer }), [
    S.save('save', { show: true, keep: true, local: 'all', exe: 'chromium' }), S.enable('enable', true, true),
  ]));
  fs.writeFileSync(path.join(__dirname, 'browser_fixtures.json'), JSON.stringify({ scenarios, settings: settingsList }, null, 1));
  console.log(`scenarios: ${scenarios.length}`);
  console.log('FIXTURES WRITTEN');
})().catch((e) => { console.error(e); process.exit(1); });
