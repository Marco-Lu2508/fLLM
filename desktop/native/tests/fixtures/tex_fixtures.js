// Produces the math fixtures from the Studio's own markdown.js: texToHtml and the inline math
// patterns, so the Rust port is checked against what the JavaScript really returns.
const fs = require('fs');
const src = fs.readFileSync('/opt/data/cache/scratch/fLLM/rust/engine/src/webui/markdown.js', 'utf8');

const escDef = src.slice(src.indexOf('const esc ='), src.indexOf('\n', src.indexOf('const esc =')) + 1);
const greekStart = src.indexOf('const GREEK =');
const mathEnd = src.indexOf('// ---- inline');
if (greekStart < 0 || mathEnd < 0) throw new Error('math block not found');
const block = src.slice(greekStart, mathEnd);

const factory = new Function(escDef + block + '\nreturn { texToHtml, mathSpan, GREEK, GREEK_UP, TEX_SYM, TEX_BIG, TEX_FN, TEX_ACCENT, BB };');
const M = factory();

// the inline pattern of the Studio (markdown.js), with its four groups
const MATH_RE = /\$\$([^$\n]+)\$\$|\\\[([^\n]+?)\\\]|\\\(([\n]+?)\\\)|(?<![\w$\\])\$(?!\s)([^$\n]*?[^\\s$\\])\$(?![\w$])/g;
// the real one, copied exactly (the escaped form above keeps the file readable):
// the real one, taken from the file itself so it cannot drift (the probe reads it this way):
const _pl = src.split('\n').find((x) => x.includes('$$(') || x.includes('\\$\\$('));
const REAL = new RegExp(_pl.slice(_pl.indexOf('/') + 1, _pl.lastIndexOf('/g')), 'g');

const CASES = {};
CASES.tex = [
  'x + y',
  '\\frac{1}{2}',
  '\\sqrt{4}',
  '\\sqrt[3]{x}',
  'x^2 + y_{1}',
  '\\sum_{i=1}^{n} i',
  '\\alpha + \\beta',
  '\\Gamma \\Delta',
  '\\infty \\partial \\nabla',
  '\\leq \\geq \\neq \\approx',
  '\\int_0^1 x\\,dx',
  '\\text{hello}',
  '\\mathrm{d}',
  '\\textbf{bold}',
  '\\mathbf{x}',
  '\\mathbb{R}',
  '\\hat{x} + \\vec{v}',
  '\\lim_{x \\to 0}',
  'a \\to b',
  '\\left( x \\right)',
  '\\overline{AB}',
  '100\\%',
  'x \\cdot y',
  '\\emptyset',
  'f(x) = \\frac{x+1}{x-1}',
  '\\alpha\\beta\\gamma',
  '\\pi r^2',
  'n! = \\prod_{k=1}^{n} k',
  '\\varphi(x)',
  '\\Omega',
  'e^{i\\pi} + 1 = 0',
  '\\text{if } x > 0',
  '\\sqrt{a^2 + b^2}',
  '\\xrightarrow{f}',
  'a_b^c',
  '\\cos(\\theta)',
  'x \\in A \\cup B',
  '\\forall x \\exists y',
  '\\neg P \\land Q',
  '\\begin{nothing}',
  '\\frac{1}{',
  '^{}',
  '\\unknowncommand',
  '2 +',
  'a}b',
].map((src) => [src, M.texToHtml(src, false), M.texToHtml(src, true)]);

CASES.span = [
  ['x', false],
  ['\\frac{1}{2}', false],
  ['\\frac{1}{2}', true],
  ['not latex', false],
  ['', false],
  ['   ', false],
  ['x'.repeat(2001), false],
].map(([src, display]) => [src, display, M.mathSpan(src, display)]);

CASES.patterns = [
  '$x$',
  '$ x$',
  '$x $',
  'a $x$ b',
  '$$x$$',
  '\\[x\\]',
  '\\(x\\)',
  'price $5 and $10$',
  '100$ note',
  '$x$5',
  'x$y$z',
  'not $math$ here',
  'two $a$ and $b$ cases',
  'display $$\\frac{1}{2}$$ inline $x$',
  'escaped \\$5 stays',
].map((text) => {
  const hits = [];
  const re = new RegExp(REAL.source, 'g');
  let m;
  while ((m = re.exec(text)) !== null) {
    const body = m[1] || m[2] || m[3] || m[4] || '';
    hits.push({ start: m.index, end: m.index + m[0].length, display: Boolean(m[1] || m[2]), body });
    if (m.index === re.lastIndex) re.lastIndex++;
  }
  return [text, hits];
});

fs.writeFileSync('/opt/data/cache/scratch/fLLM/desktop/native/tests/fixtures/tex_fixtures.json', JSON.stringify(CASES, null, 1));
console.log('tex_fixtures.json written:',
  Object.keys(CASES).map((k) => k + '=' + CASES[k].length).join(' '));
