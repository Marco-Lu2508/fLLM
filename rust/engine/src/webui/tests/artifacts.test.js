// Tests for artifacts, the JavaScript runner, message styles, temporary chats and math.
// Run from the repository root:  node rust/engine/src/webui/tests/artifacts.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, '..');
const read = (f) => fs.readFileSync(path.join(dir, f), 'utf8');

// ---- pure block of artifacts.js ------------------------------------------------
const src = read('artifacts.js');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in artifacts.js');
const A = new Function(`${src.slice(begin, end)}
return { artifactKind, parseInfo, extractArtifacts, buildFamilies, withStylePrompt, STYLES, parseCsv, fileNameFor, previewDocument };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

test('artifactKind maps languages to preview kinds', () => {
  assert.strictEqual(A.artifactKind('html'), 'html');
  assert.strictEqual(A.artifactKind('SVG'), 'svg');
  assert.strictEqual(A.artifactKind('md'), 'markdown');
  assert.strictEqual(A.artifactKind('csv'), 'csv');
  assert.strictEqual(A.artifactKind('python'), 'code');
  assert.strictEqual(A.artifactKind(''), 'code');
});

test('parseInfo reads quoted and bare titles', () => {
  assert.strictEqual(A.parseInfo('title="Todo app"').title, 'Todo app');
  assert.strictEqual(A.parseInfo(' title=Clock').title, 'Clock');
  assert.strictEqual(A.parseInfo(" artifact='My page' extra").title, 'My page');
  assert.strictEqual(A.parseInfo('').title, '');
  assert.strictEqual(A.parseInfo('{.numberLines}').title, '');
});

test('extractArtifacts finds previewable and titled blocks only', () => {
  const text = 'Hi\n```html\n<p>a</p>\n```\ntext\n```python\nprint(1)\n```\n```python title="Tool"\nprint(2)\n```\n```svg\n<svg/>\n```';
  const found = A.extractArtifacts(text);
  assert.deepStrictEqual(found.map((x) => [x.kind, x.title]), [['html', ''], ['code', 'Tool'], ['svg', '']]);
  assert.strictEqual(found[0].code, '<p>a</p>');
});

test('extractArtifacts ignores unclosed blocks and fences inside longer fences', () => {
  assert.strictEqual(A.extractArtifacts('```html\n<p>cut').length, 0);
  const f = A.extractArtifacts('````markdown\n```js\nx\n```\n````');
  assert.strictEqual(f.length, 1);
  assert.strictEqual(f[0].code, '```js\nx\n```');
});

test('versions: an untitled block of the same kind in a later message is a new version', () => {
  const fam = A.buildFamilies([
    { id: 'm1', text: '```html\n<button>Go</button>\n```' },
    { id: 'm2', text: 'Red now:\n```html\n<button style="color:red">Go</button>\n```' },
  ]);
  assert.strictEqual(fam.length, 1);
  assert.strictEqual(fam[0].versions.length, 2);
  assert.deepStrictEqual(fam[0].versions.map((v) => v.msgId), ['m1', 'm2']);
  assert(fam[0].versions[1].code.includes('red'));
});

test('versions: same title joins, other title or kind starts a new artifact', () => {
  const fam = A.buildFamilies([
    { id: 'm1', text: '```html title="Clock"\n<p>1</p>\n```' },
    { id: 'm2', text: '```html title="Timer"\n<p>t</p>\n```\n```svg\n<svg/>\n```' },
    { id: 'm3', text: '```html title="clock"\n<p>2</p>\n```' },
  ]);
  assert.deepStrictEqual(fam.map((f) => [f.title, f.versions.length]), [['Clock', 2], ['Timer', 1], ['', 1]]);
});

test('versions: two blocks of one kind in one message are two artifacts; identical code adds no version', () => {
  const fam = A.buildFamilies([
    { id: 'm1', text: '```html\n<p>a</p>\n```\n```html\n<p>b</p>\n```' },
    { id: 'm2', text: '```html\n<p>a</p>\n```' },
  ]);
  assert.strictEqual(fam.length, 2);
  assert.strictEqual(fam[0].versions.length, 1);
  assert.strictEqual(fam[1].versions.length, 1);
});

test('previewDocument carries a CSP that forbids the network', () => {
  const d = A.previewDocument('html', '<h1>x</h1>');
  assert(d.includes("default-src 'none'"));
  assert(d.includes("img-src data: blob:"));
  assert(d.indexOf('Content-Security-Policy') < d.indexOf('<h1>'));
  assert(A.previewDocument('svg', '<svg></svg>').includes('<svg></svg>'));
});

test('parseCsv handles quotes, commas and newlines', () => {
  assert.deepStrictEqual(A.parseCsv('a,b\n"x, y","he said ""hi"""\n1,2'), [['a', 'b'], ['x, y', 'he said "hi"'], ['1', '2']]);
  assert.deepStrictEqual(A.parseCsv('a;b;c\n1;2;3'), [['a', 'b', 'c'], ['1', '2', '3']]);
});

test('fileNameFor builds safe names with the right extension', () => {
  assert.strictEqual(A.fileNameFor({ kind: 'html', title: 'My App!' }), 'My-App.html');
  assert.strictEqual(A.fileNameFor({ kind: 'svg', title: '' }), 'artifact.svg');
  assert.strictEqual(A.fileNameFor({ kind: 'code', title: 'tool', lang: 'python' }), 'tool.py');
});

test('style prompt: default adds nothing, others append a short instruction', () => {
  assert.strictEqual(A.withStylePrompt('Be kind.', 'default'), 'Be kind.');
  assert.strictEqual(A.withStylePrompt('', undefined), '');
  const c = A.withStylePrompt('Be kind.', 'concise');
  assert(c.startsWith('Be kind.\n\n') && c.length < 200);
  assert.strictEqual(A.withStylePrompt('', 'formal'), A.STYLES.find((s) => s.id === 'formal').prompt);
  assert.strictEqual(A.withStylePrompt('x', 'nonsense'), 'x');
  assert.deepStrictEqual(A.STYLES.map((s) => s.id), ['default', 'concise', 'explanatory', 'formal', 'casual']);
});

// ---- the JavaScript runner (worker source run against a fake worker scope) ------
async function runInFake(code) {
  const out = { msgs: [] };
  const fake = { postMessage: (m) => out.msgs.push(m), fetch: () => 'real', XMLHttpRequest: function X() {}, importScripts: () => {} };
  new Function('self', read('runner.js'))(fake);
  await fake.onmessage({ data: { code } });
  out.fake = fake;
  return out;
}
const done = (o) => o.msgs.find((m) => m.t === 'done');
const logs = (o) => o.msgs.filter((m) => m.t === 'log').map((m) => `${m.level}:${m.text}`);

(async () => {
  const run = async (name, fn) => { await fn(); n++; console.log(`ok - ${name}`); };

  await run('runner returns the final value and captures console output', async () => {
    const o = await runInFake('console.log("hi", 1, {a: 2}); console.error("bad"); 6 * 7');
    assert.deepStrictEqual(logs(o), ['log:hi 1 {"a":2}', 'error:bad']);
    assert.strictEqual(done(o).value, '42');
    assert.strictEqual(done(o).error, undefined);
  });
  await run('runner awaits promises and supports top-level await', async () => {
    const o = await runInFake('const x = await Promise.resolve(5); console.log(x); return x + 1;');
    assert.deepStrictEqual(logs(o), ['log:5']);
    assert.strictEqual(done(o).value, '6');
    const p = await runInFake('Promise.resolve("later")');
    assert.strictEqual(done(p).value, '"later"');
  });
  await run('runner reports errors', async () => {
    const o = await runInFake('throw new TypeError("nope")');
    assert(/TypeError: nope/.test(done(o).error));
    const s = await runInFake('let = ;');
    assert(/SyntaxError/.test(done(s).error));
  });
  await run('runner blocks network APIs', async () => {
    const o = await runInFake('try { fetch("/x"); } catch (e) { console.log("blocked", e.message); }');
    assert(/blocked/.test(logs(o)[0]));
    const x = await runInFake('new XMLHttpRequest()');
    assert(done(x).error);
    const w = await runInFake('importScripts("a.js")');
    assert(done(w).error);
  });
  await run('runner caps the amount of output', async () => {
    const o = await runInFake('for (let i = 0; i < 5000; i++) console.log(i)');
    assert(logs(o).length <= 501);
    assert(o.msgs.some((m) => m.t === 'log' && /truncated/.test(m.text)));
  });

  // ---- chat.js: temporary chats are never written to storage -----------------------
  const chat = read('chat.js');
  assert(/filter\(\(t\) => !t\.temporary\)/.test(chat), 'saveThreads must skip temporary chats');
  n++; console.log('ok - chat.js skips temporary chats when saving');

  // ---- math ------------------------------------------------------------------------
  const md = read('markdown.js');
  const MD = new Function(`${md}\nreturn MD;`)();
  await run('math: fractions, scripts, greek, roots and sums are typeset', async () => {
    const h = MD.texToHtml('\\frac{a}{b} + x^2_i + \\alpha \\sqrt{2} \\sum_{k=1}^{n} k', false);
    assert(h.includes('tx-frac') && h.includes('<sup>') && h.includes('α') && h.includes('tx-sqrt') && h.includes('∑'));
    assert(MD.texToHtml('\\sum_{k=1}^{n} k', true).includes('tx-lim'));
  });
  await run('math: unsupported input falls back to a source chip, and HTML is escaped', async () => {
    assert.strictEqual(MD.texToHtml('\\begin{matrix} a \\end{matrix}', false), null);
    assert.strictEqual(MD.texToHtml('\\frac{a}', false), null);
    const html = MD.render('Value $\\foo{x}$ and $<img src=x onerror=alert(1)>$ here');
    assert(html.includes('class="math-inline"'));
    assert(!html.includes('<img'));
    const typeset = MD.render('So $\\frac{1}{2}$ and $<b>$.');
    assert(!typeset.includes('<b>'));
  });
  await run('math: prices with dollar signs stay text', async () => {
    const html = MD.render('It costs $5 and then $6 more.');
    assert(!html.includes('tex'));
    assert(!html.includes('math-inline'));
  });
  await run('math: display blocks and \\[ \\] work', async () => {
    assert(MD.render('$$\nx = \\frac{-b}{2a}\n$$').includes('tex-display'));
    assert(MD.render('\\[\n\\int_0^1 x\\,dx\n\\]').includes('tx-op'));
  });
  await run('code block hook adds actions only to closed blocks', async () => {
    MD.hooks.actions = (lang) => `<button>${lang}</button>`;
    assert(MD.render('```html\n<p>x</p>\n```').includes('<button>html</button>'));
    assert(!MD.render('```html\n<p>x').includes('<button>'));
  });

  console.log(`${n} tests passed`);
})().catch((e) => { console.error(e); process.exit(1); });
