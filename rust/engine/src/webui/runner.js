// Web Worker that runs a code block of the Studio (served at /studio/runner.js with its own strict CSP).
// It has no DOM; fetch, XMLHttpRequest, WebSocket, importScripts and friends throw. The final value of the code is
// reported and console output is captured.
(function runnerMain(self) {
  'use strict';
  const MAX_LINES = 500;
  const MAX_CHARS = 20000;
  const BLOCKED = ['fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'importScripts', 'Worker', 'SharedWorker', 'WebTransport'];
  let lines = 0;
  let chars = 0;
  let cut = false;
  const fmt = (v, quote) => {
    if (typeof v === 'string') return quote ? JSON.stringify(v) : v;
    if (v === undefined) return 'undefined';
    if (typeof v === 'number') return String(v);
    if (typeof v === 'function') return `[Function${v.name ? `: ${v.name}` : ''}]`;
    if (typeof v === 'bigint') return `${v}n`;
    if (typeof v === 'symbol') return String(v);
    if (v instanceof Error) return `${v.name || 'Error'}: ${v.message}`;
    try {
      const seen = new WeakSet();
      const s = JSON.stringify(v, (k, x) => {
        if (typeof x === 'bigint') return `${x}n`;
        if (typeof x === 'function') return '[Function]';
        if (x instanceof Map) return { Map: Array.from(x.entries()) };
        if (x instanceof Set) return { Set: Array.from(x.values()) };
        if (x && typeof x === 'object') { if (seen.has(x)) return '[Repeated]'; seen.add(x); }
        return x;
      });
      return s === undefined ? String(v) : s;
    } catch (err) { return String(v); }
  };
  const emit = (level, args) => {
    if (cut) return;
    const text = args.map((a) => fmt(a, false)).join(' ');
    lines++;
    chars += text.length;
    if (lines > MAX_LINES || chars > MAX_CHARS) {
      cut = true;
      self.postMessage({ t: 'log', level: 'warn', text: '... output truncated' });
      return;
    }
    self.postMessage({ t: 'log', level, text });
  };
  const con = {
    log: (...a) => emit('log', a), info: (...a) => emit('info', a), debug: (...a) => emit('log', a),
    warn: (...a) => emit('warn', a), error: (...a) => emit('error', a), table: (...a) => emit('log', a), dir: (...a) => emit('log', a),
    assert: (ok, ...a) => { if (!ok) emit('error', ['Assertion failed:', ...a]); }, clear: () => {},
  };
  const stub = (name) => function blocked() { throw new Error(`${name} is disabled: code blocks run offline`); };
  self.onmessage = async (e) => {
    const code = String((e.data && e.data.code) ?? '');
    const stubs = BLOCKED.map(stub);
    BLOCKED.forEach((name, i) => {
      try { Object.defineProperty(self, name, { value: stubs[i], configurable: false, writable: false }); } catch (err) { /* already locked */ }
    });
    let started = false;
    const mark = () => { started = true; };
    const exec = (source) => new Function('console', '__mark', ...BLOCKED, '__code', '"use strict"; return eval(__code);')(con, mark, ...stubs, source);
    let value;
    let error;
    try {
      let res;
      try {
        res = exec(`__mark();\n${code}`);
      } catch (first) {
        if (started || !(first instanceof SyntaxError)) throw first;
        try { res = exec(`(async () => { __mark();\n${code}\n})()`); } catch (second) { throw !started && second instanceof SyntaxError ? first : second; }
      }
      if (res && typeof res.then === 'function') res = await res;
      value = res;
    } catch (err) {
      error = err && err.name ? `${err.name}: ${err.message}` : String(err);
    }
    self.postMessage({ t: 'done', value: value === undefined ? undefined : fmt(value, true), error });
  };
})(self);
