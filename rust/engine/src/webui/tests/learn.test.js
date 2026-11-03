// Tests for learn.js (the self-improvement loop: skill notes, revisions, memory and profile pages) and its wiring in the page.
// Run from the repository root:  node rust/engine/src/webui/tests/learn.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, '..');
const read = (f) => fs.readFileSync(path.join(dir, f), 'utf8');
const src = read('learn.js');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in learn.js');
const L = new Function(`${src.slice(begin, end)}\nreturn { lnSkillsRead, lnRunMessages, lnRunFailed, lnDiffHtml, lnStatText, lnSkillRelevant, lnProfileShape, lnPendingText, lnLogTitle, lnFeedHtml, lnToastText };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

test('skills the model read are found in tool calls, once, with valid names only', () => {
  const messages = [
    { role: 'assistant', content: '', tool_calls: [
      { id: '1', function: { name: 'skill', arguments: '{"action":"read","name":"deploy-app"}' } },
      { id: '2', function: { name: 'skill', arguments: '{"action":"list"}' } },
      { id: '3', function: { name: 'read_file', arguments: '{"path":"a"}' } },
      { id: '4', function: { name: 'skill', arguments: '{"action":"read","name":"../evil"}' } },
      { id: '5', function: { name: 'skill', arguments: 'not json' } },
    ] },
    { role: 'assistant', content: '', tool_calls: [{ id: '6', function: { name: 'skill', arguments: { action: 'read', name: 'deploy-app' } } }, { id: '7', function: { name: 'skill', arguments: { action: 'read', name: 'sql_query' } } }] },
  ];
  assert.deepStrictEqual(L.lnSkillsRead(messages), ['deploy-app', 'sql_query']);
  assert.deepStrictEqual(L.lnSkillsRead(null), []);
});

test('the messages of a run are reduced to text and tool calls, cut short, without pictures', () => {
  const messages = [
    { role: 'user', content: 'old' },
    { role: 'user', content: [{ type: 'text', text: 'build it' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } }] },
    { role: 'assistant', content: '', tool_calls: [{ id: 'a', function: { name: 'bash', arguments: { command: 'make' } } }] },
    { role: 'tool', tool_call_id: 'a', content: 'x'.repeat(9000) },
    { role: 'system', content: 'ignored' },
  ];
  const out = L.lnRunMessages(messages, 1);
  assert.strictEqual(out.length, 3);
  assert.strictEqual(out[0].content, 'build it');
  assert.strictEqual(out[1].tool_calls[0].function.arguments, '{"command":"make"}');
  assert.strictEqual(out[2].content.length, 3000);
  assert(!JSON.stringify(out).includes('base64'));
  const many = Array.from({ length: 500 }, (_, i) => ({ role: 'user', content: `m${i}` }));
  const kept = L.lnRunMessages(many, 0, 300);
  assert.strictEqual(kept.length, 300);
  assert.strictEqual(kept[299].content, 'm499');
  assert.deepStrictEqual(L.lnRunMessages(null, 0), []);
});

test('a run counts as failed on an error or the step limit', () => {
  assert(L.lnRunFailed({ error: 'boom' }));
  assert(L.lnRunFailed({ hitLimit: true }));
  assert(!L.lnRunFailed({}));
});

test('a diff is coloured per line and escaped', () => {
  const html = L.lnDiffHtml('  keep\n- old <b>\n+ new\n  ...\n', esc);
  assert(html.includes('var(--destructive,#c33)') && html.includes('var(--success,#2a8a4a)'));
  assert(html.includes('&lt;b>') && !html.includes('<b>'));
  assert.strictEqual(html.split('\n').length, 4);
});

test('the statistics read well and only skills with something to show are listed', () => {
  assert.strictEqual(L.lnStatText({ uses: 1 }), '1 use');
  assert.strictEqual(L.lnStatText({ uses: 5, failed: 1, up: 2, down: 0, retries: 3 }), '5 uses, 1 failed, thumbs 2 up / 0 down, 3 retries');
  assert(!L.lnSkillRelevant({ uses: 0, pending: false, versions: [] }));
  assert(L.lnSkillRelevant({ uses: 2 }) && L.lnSkillRelevant({ pending: true }) && L.lnSkillRelevant({ versions: [{ version: 1 }] }) && L.lnSkillRelevant({ due: 'x' }));
  assert(!L.lnSkillRelevant(null));
});

test('the profile shape and the pending counters are described', () => {
  assert.deepStrictEqual(L.lnProfileShape('# User profile\n\n## Languages\n- Rust\n- Go\n## Notes\n- x\n'), { sections: ['Languages', 'Notes'], lines: 3 });
  assert.strictEqual(L.lnPendingText({ skills: [{}], memory: [{}, {}], profile: [] }), 'Waiting for a decision: 1 skill revision, 2 memory entries.');
  assert.strictEqual(L.lnPendingText({ skills: [], memory: [], profile: [] }), 'Nothing waits for a decision.');
  assert.strictEqual(L.lnPendingText(null), '');
});

test('every element learn.js looks up exists in the page, and the script is registered after skills.js', () => {
  const html = read('shell.html');
  const ids = new Set([...src.matchAll(/\$\('([A-Za-z0-9_]+)'\)/g)].map((m) => m[1]));
  for (const id of ids) assert(html.includes(`id="${id}"`), `shell.html has no element with id ${id}`);
  const rs = read('../webui.rs');
  assert(rs.includes('include_str!("webui/learn.js")'));
  const files = rs.match(/"[a-z-]+\.js"/g).map((s) => s.slice(1, -1));
  assert(files.indexOf('learn.js') > files.indexOf('skills.js') && files.indexOf('learn.js') < files.indexOf('boot.js'));
  const order = [...rs.matchAll(/include_str!\("webui\/([a-z-]+\.js)"\)/g)].map((m) => m[1]);
  assert(order.indexOf('learn.js') > order.indexOf('skills.js') && order.indexOf('learn.js') > order.indexOf('code.js') && order.indexOf('learn.js') < order.indexOf('boot.js'));
});

test('the Code page offers the toggle, the commands and the hook', () => {
  const code = read('code.js');
  assert(code.includes('data-learn') && code.includes("case 'learn'") && code.includes("case 'good'") && code.includes('window.fllmLearnAfterRun'));
  assert(code.includes('learn: code.learn'), 'the setting is saved with the other Code settings');
  assert(/if \(code\.learn && typeof window\.fllmLearnAfterRun === 'function'\)/.test(code), 'learning only runs when the toggle is on');
  const html = read('shell.html');
  assert(html.includes('<option value="learn">'), 'the schedule offers the nightly consolidation');
});

test('the feed lists every automatic change with its reason and an undo button', () => {
  const entries = [
    { id: 'chg-1', t: 1793643870, kind: 'skill_revise', name: 'build', session: 's1', reason: 'make is missing', diff: '- 1. run make\n+ 1. run cargo build\n' },
    { id: 'chg-2', t: 1793643871, kind: 'memory_add', text: 'Tests run with <b>cargo test</b>', session: 's1', reason: 'reflection', undone: true },
  ];
  const html = L.lnFeedHtml(entries, esc, () => 'when');
  assert(html.includes('Revised the skill &quot;build&quot;') || html.includes('Revised the skill "build"'));
  assert(html.includes('data-ln-chg="chg-1"') && html.includes('data-ln-undo') && html.includes('make is missing') && html.includes('run cargo build'));
  assert(html.includes('undone') && html.includes('&lt;b>cargo test') && !html.includes('<b>cargo'), 'escaped, and an undone entry has no undo button');
  assert.strictEqual((html.match(/data-ln-undo/g) || []).length, 1);
  assert(L.lnFeedHtml([], esc, () => '').includes('Nothing was learned'));
  assert.strictEqual(L.lnLogTitle({ kind: 'skill_create', name: 'x' }), 'Wrote the skill "x"');
  assert.strictEqual(L.lnLogTitle({ kind: 'profile_add', text: 'Prefers tabs (2026-11-03)' }), 'Added to your profile: Prefers tabs (2026-11-03)');
  assert.strictEqual(L.lnToastText({ summary: 'created skill "a", saved 2 notes.' }), 'Learned: created skill "a", saved 2 notes.');
  assert.strictEqual(L.lnToastText({ summary: '' }), '');
  assert.strictEqual(L.lnToastText(null), '');
});

test('learning is on by default in the Code page and kept in step with the server', () => {
  const code = read('code.js');
  assert(code.includes('learn: true,') && code.includes('saved.learn !== false'), 'on unless somebody switched it off');
  assert(code.includes('Learn from my sessions') && code.includes('fllmLearnSetEnabled') && code.includes('fllmLearnNotice'));
  assert(code.includes("case 'learn'") && code.includes('/learn project'), 'a per-project override');
  assert(code.includes('data-learn-fb'), 'thumbs');
});

console.log(`${n} tests passed`);
