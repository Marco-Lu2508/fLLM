// Produces the artifacts fixtures from the Studio's own @pure block (artifacts.js), so the Rust
// port is checked against what the JavaScript really returns.
const fs = require('fs');
const src = fs.readFileSync('/opt/data/cache/scratch/fLLM/rust/engine/src/webui/artifacts.js', 'utf8');

const start = src.indexOf('@pure-begin');
const end = src.indexOf('@pure-end');
if (start < 0 || end < 0) throw new Error('pure block not found');
// the block runs from the marker to the end marker; it defines the functions on its own level
const block = src.slice(src.indexOf('\n', start) + 1, end);

const PRE = `
const STYLES = null; // not needed here
`;
// evaluate the pure block in its own scope and pick the functions out
const factory = new Function(block + `
  return { artifactKind, parseInfo, extractArtifacts, buildFamilies, fileNameFor, parseCsv,
           KIND_BY_LANG, EXT_BY_LANG, KIND_EXT, KIND_MIME };
`);
const A = factory();

const extract = (text) => A.extractArtifacts(text).map((a) => ({ lang: a.lang, kind: a.kind, title: a.title, code: a.code }));

const CASES = {};
CASES.limits = {
  kind_by_lang: A.KIND_BY_LANG,
  ext_by_lang: A.EXT_BY_LANG,
  kind_ext: A.KIND_EXT,
  kind_mime: A.KIND_MIME,
};
CASES.kind = ['html', 'HTM', 'xhtml', 'svg', 'md', 'markdown', 'csv', 'TSV', 'js', 'py', '', undefined].map((l) => [String(l), A.artifactKind(l)]);
CASES.parse_info = [
  'title="My page"',
  "title='Single' rest",
  'artifact=bare.html extra',
  'name="Named"',
  'tab=4 title="Tabs too"',
  'no title here',
  '',
  'title = "Spaced"',
].map((i) => [i, A.parseInfo(i).title]);
CASES.extract = [
  ['a plain fence', '```js\nconsole.log(1);\n```'],
  ['an html page', '```html\n<h1>Hi</h1>\n```'],
  ['with a title on a code block', '```python title="calc.py"\nprint(1)\n```'],
  ['tilde fence with info', '~~~ svg title="logo"\n<circle/>\n~~~'],
  ['unclosed fence is skipped', '```html\n<h1>never closed'],
  ['closing fence needs only backticks', '```md\n# Hi\n``` trailing junk not allowed\n```'],
  ['markdown variant', '```markdown\n# Doc\n```'],
  ['tsv is a table', '```tsv\ta\tb\n1\t2\n```'],
  ['the info string may be long', '```js {wrap} title="Run me" other\nx\n```'],
  ['cr lf line endings', '```html\r\n<p>ok</p>\r\n```'],
  ['body keeps blank lines', '```csv\na,b\n\n1,2\n```'],
  ['four backticks', '````html\n<p>deep</p>\n````'],
  ['kind code without a title is not an artifact', '```rust\nfn main() {}\n```'],
  ['indented fence', '  ```html\n  <p>x</p>\n  ```'],
].map(([name, text]) => [name, extract(text)]);

CASES.families = [
  ['two messages with the same title', [
    { id: 'm1', text: '```html title="a"\nv1\n```' },
    { id: 'm2', text: '```html title="a"\nv2\n```' },
  ]],
  ['an untitled block continues the newest of its kind', [
    { id: 'm1', text: '```csv\na,b\n```' },
    { id: 'm2', text: '```html\n<p>x</p>\n```' },
    { id: 'm3', text: '```csv\na,b,c\n```' },
  ]],
  ['identical code stays with its own family', [
    { id: 'm1', text: '```md\nsame\n```' },
    { id: 'm2', text: '```html\nx\n```' },
    { id: 'm3', text: '```md\nsame\n```' },
  ]],
  ['one message changes a family at most once', [
    { id: 'm1', text: '```html title="a"\none\n```' },
    { id: 'm2', text: '```html title="a"\ntwo\n``` \n```html title="a"\nthree\n```' },
  ]],
  ['titles are compared without case', [
    { id: 'm1', text: '```js title="Tool"\nx\n```' },
    { id: 'm2', text: '```js title="tool"\ny\n```' },
  ]],
  ['several kinds start their own families', [
    { id: 'm1', text: '```html\n<p/>\n```\n```svg\n<circle/>\n```' },
    { id: 'm2', text: '```html\n<p>2</p>\n```' },
  ]],
].map(([name, msgs]) => {
  const fams = A.buildFamilies(msgs).map((f) => ({
    id: f.id, kind: f.kind, title: f.title, lang: f.lang, touched: f.touched,
    versions: f.versions.map((v) => ({ n: v.n, msgId: v.msgId, code: v.code })),
  }));
  return [name, fams];
});

CASES.file_name = [
  { title: 'My page', kind: 'html', lang: 'html' },
  { title: '', kind: 'html', lang: 'html' },
  { title: 'weird:/chars*', kind: 'markdown', lang: 'md' },
  { title: 'table here', kind: 'csv', lang: 'csv' },
  { title: 'script', kind: 'code', lang: 'python' },
  { title: 'script', kind: 'code', lang: 'unknownlang' },
  { title: 'a'.repeat(90), kind: 'code', lang: 'rs' },
  { title: 'double  spaced name', kind: 'code', lang: 'js' },
].map((a) => [a, A.fileNameFor(a)]);

CASES.parse_csv = [
  'a,b,c\n1,2,3',
  'name;age\n"Smith; Jr";42',
  'x\ty\n1\t2',
  '"quoted, field",b\n"line\nbreak",2',
  '',
  'single',
  'a,b\n\n1,2',
  'a,,b\n1,,3',
  'empty,rows,too\n,,\nreal,data,here',
].map((t) => A.parseCsv(t));

fs.writeFileSync('/opt/data/cache/scratch/fLLM/desktop/native/tests/fixtures/artifacts_fixtures.json', JSON.stringify(CASES, null, 1));
console.log('artifacts_fixtures.json written:',
  Object.keys(CASES).map((k) => k + '=' + CASES[k].length).join(' '));
