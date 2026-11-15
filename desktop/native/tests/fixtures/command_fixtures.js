#!/usr/bin/env node
// Row 79 fixtures: slice the page's own code (the @pure blocks of code.js and codeprompts.js,
// the BUILTIN_COMMANDS table and the whole runCodeCommand dispatcher), run it in a vm with
// recorders where the page talks to the world, and print what the page answers for every case.
// The Rust port compares against the printed `expected` values.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const lines = (name) => fs.readFileSync(path.join(WEBUI, name), 'utf8').split(/\r?\n/);
const slice = (l, from, to) => l.slice(from - 1, to).join('\n');

const code = lines('code.js');
const prompts = lines('codeprompts.js');
const PURE_CODE = slice(code, 19, 560);            // rules, permission core (marked @pure)
const PURE_PROMPTS = slice(prompts, 7, 309);       // PERSONALITIES, insights, search hits
const BUILTIN = slice(code, 2477, 2491);           // the command table (sandbox listed twice)
const TOGGLE = slice(code, 2463, 2467);            // toggleAgent, so /agent's own text is pinned
const MAX_SESSIONS = slice(code, 1095, 1095);      // the number /insights prints
const AGENT_NAME = slice(code, 651, 651);          // authorize() reads the agent from this
const READ_ONLY = slice(code, 735, 735);           // the plan check reads a tool from this
const PARSED_RULES = slice(code, 1069, 1069);      // authorize() parses the rules with this
const DISPATCH = slice(code, 2510, 2769);          // runCodeCommand, whole switch
const INIT = slice(code, 2476, 2476);              // the prompt /init sends
// outside the markers, but free of the DOM: the decision reads the objects it is given
const OUTSIDE = [slice(code, 1460, 1473), slice(code, 1475, 1487), slice(code, 1718, 1733)].join('\n');
// row 81: the switch of `/autoreview` runs these — the pair of codeprompts.js is inside
// PURE_PROMPTS already; the check after an edit and the clip of its reply sit outside it
const AUTOCHECK = slice(code, 1937, 1954);         // CHECKED_EXTENSIONS + autoCheckEdit
const CLIP = slice(code, 1884, 1887);              // clipOutput, over a ctxMax of the harness
const SPECS_A = slice(code, 921, 933);             // loadSpecs + specAction (row 80)
const SPECS_B = slice(code, 945, 978);             // runSpecTask + specCommand

const log = [];
const rec = (name) =>
  (...args) => {
    log.push([name, ...args]);
  };

// identifiers the slices themselves declare: a recorder of the harness must not collide with them
const pageNames = new Set(
  [PURE_CODE, PURE_PROMPTS, BUILTIN, INIT, TOGGLE, MAX_SESSIONS, AGENT_NAME, READ_ONLY, PARSED_RULES, OUTSIDE, DISPATCH, AUTOCHECK, CLIP, SPECS_A, SPECS_B].join('\n').match(
    /(?:const|let|function|class)\s+([A-Za-z_$][\w$]*)/g,
  ).map((s) => s.split(/\s+/)[1]),
);

// the state every case starts from (the page loads it from its server; here it is the seed)
const CODE_SEED =
  "{ messages: [], view: [], workspace: { root: '/home/marco/p' }, sessionId: 's1', title: 't'," +
  " titleLocked: false, running: false, perm: 'ask', modes: {}, autoApprove: true, turbo: false," +
  " sandbox: 'workspace', sandboxAvailable: true, rules: { allow: '', deny: '' }, custom: []," +
  " personality: 'default', learn: false, autoCheck: false, autoReview: false, lastBuildPerm: 'ask', memory: ''," +
  " compactedAt: 0, reverted: null, alwaysAllow: new Set(), registry: new Map(), abort: null," +
  " sideOpen: {}, memDraft: null, todos: [], changed: new Map(), disabled: null, specs: [] }";

// what the page talks to, replaced by recorders; the pure code above supplies the real logic
const stubs = [
  ['window', "{ fllmUsageSummary: () => ({ text: '42 in / 7 out in 3 answer(s)' }), fllmSkillPropose: null }"],
  ['localStorage', "{ getItem: () => null, setItem: () => {}, removeItem: () => {} }"],
  ['state', "{ selectedModel: 'qwen', status: { model: { name: 'qwen' } } }"],
  ['code', CODE_SEED],
  ['codeNotice', "__rec('notice')"],
  ['pushChoices', "__rec('choices')"],
  ['toast', "__rec('toast')"],
  ['confirmDialog', "async (...a) => { __log.push(['confirm', ...a]); return true; }"],
  ['apiFetch', "async (...a) => { __log.push(['api', ...a]); return { ok: true, json: async () => ({}) }; }"],
  ['toolCall', "async (...a) => { __log.push(['tool', ...a]); return { ok: true, output: 'out', error: '' }; }"],
  ['userRun', "__rec('userRun')"],
  ['userTool', "__rec('userTool')"],
  ['agentRun', "__rec('agentRun')"],
  ['setPerm', "(p) => { code.perm = p; if (p !== 'plan') code.lastBuildPerm = p; __log.push(['setPerm', p]); }"],
  ['toggleAgent', "__rec('toggleAgent')"],
  ['newCodeSession', "__rec('newSession')"],
  ['undoChange', "__rec('undo')"],
  ['showRewind', "__rec('rewind')"],
  ['compactSession', "__rec('compact')"],
  ['workspaceSessions', "() => __savedSessions"],
  ['turnChoices', "() => [{ label: 'Fix the bug', data: 'm1' }]"],
  ['forkSession', "__rec('fork')"],
  ['revertToMessage', "__rec('revert')"],
  ['unrevert', "__rec('unrevert')"],
  ['cleanTitle', "(t) => 'Clean ' + t"],
  ['renameSession', "__rec('rename')"],
  ['exportSession', "__rec('export')"],
  ['modelNames', "() => ['qwen', 'llama']"],
  ['ctxMax', '() => 8192'],
  ['serverHas', "() => true"],
  ['pickModel', "__rec('pickModel')"],
  ['saveMemoryEntry', "async (...a) => { __log.push(['memory', ...a]); return true; }"],
  ['renderCodeSide', "__rec('renderSide')"],
  ['renderCode', "__rec('render')"],
  ['saveRules', "__rec('saveRules')"],
  ['saveCodeSettings', "__rec('saveSettings')"],
  ['saveCodeSession', "__rec('saveSession')"],
  ['stopAgent', "__rec('stopAgent')"],
  ['setWorkspace', "__rec('setWorkspace')"],
  ['pickWorkspace', "__rec('pickWorkspace')"],
  ['resumeSession', "__rec('resume')"],
  ['readCustomCommand', "async (c) => { __log.push(['readCustom', c.name]); return 'Do it\\n\\nARGUMENTS: x'; }"],
  ['fllmLearnStatus', "async () => null"],
  ['fllmLearnSetEnabled', "async () => {}"],
  ['fllmLearnFeedback', "async () => {}"],
  ['fllmSkillPropose', "null"],
  ['fireHook', "__rec('hook')"],
  ['$', "{ scrollIntoView: () => {}, querySelector: () => null, querySelectorAll: () => [], textContent: '', classList: { remove() {} } }"],
  ['AbortController', "class { constructor() { this.signal = {}; } }"],
  ['setCodeBusy', "__rec('busy')"],
  ['codeSystemPrompt', "() => 'sys'"],
  ['toolList', "() => []"],
];

function harness() {
  const head = `
  const __log = [];
  const __rec = (name) => (...args) => { __log.push([name, ...args]); };
  const __out = () => __log.slice();
  // The chain of one call as runToolCall walks it (code.js 1955-2010): the plan check, then
  // authorize, then the table of autoapprove when the answer was a question. The fns are the
  // page's own; only the walking lives here, because it is a method of the runner.
  const __decision = (o) => {
    code.turbo = !!o.turbo;
    code.modes = o.modes || {};
    code.perm = o.perm || 'ask';
    code.rules = { allow: (o.rules && o.rules.allow) || '', deny: (o.rules && o.rules.deny) || '' };
    code.alwaysAllow = new Set(o.always || []);
    code.autoApprove = o.autoApprove !== false;
    const entry = o.entry;
    const args = o.args || {};
    if (code.perm === 'plan' && !readOnlyTool(entry) && entry.source !== 'client') {
      return { decision: 'plan', autoNote: null };
    }
    let decision = authorize(entry, args);
    let autoNote = null;
    if (decision === 'ask' && code.autoApprove) {
      const info = permissionInfo(entry, args);
      const verdict = autoVerdict(info.cat, entry.name, info.subject);
      if (verdict.verdict === 'safe') {
        decision = 'allow';
        autoNote = verdict.reason;
      }
    }
    return { decision, autoNote };
  };
  const __snapshot = () => ({
    running: code.running, perm: code.perm, lastBuildPerm: code.lastBuildPerm,
    autoApprove: code.autoApprove, turbo: code.turbo, sandbox: code.sandbox,
    sandboxAvailable: code.sandboxAvailable, autoCheck: code.autoCheck, autoReview: code.autoReview,
    personality: code.personality, rules: { allow: code.rules.allow, deny: code.rules.deny },
    custom: code.custom.map((c) => ({ name: c.name, description: c.description })),
    specs: code.specs,
  });
  const __defaultSaved = () => [{ id: 'other', title: 'Older', time: 1699990000 }];
  let __savedSessions = __defaultSaved();
  function __reset() {
    __log.length = 0;
    __savedSessions = __defaultSaved();
    Object.assign(code, (${CODE_SEED}));
    confirmDialog = async (...a) => { __log.push(['confirm', ...a]); return true; };
    apiFetch = async (...a) => { __log.push(['api', ...a]); return { ok: true, json: async () => ({}) }; };
  }`;
  const reassign = new Set(['confirmDialog', 'apiFetch', 'toolCall', 'ctxMax']);
  const decls = stubs
    .filter(([name]) => !pageNames.has(name))
    .map(([name, body]) => `${reassign.has(name) ? 'let' : 'const'} ${name} = ${body};`)
    .join('\n');
  return head + '\n' + decls;
}

const legacyHarness = `
  const __log = [];
  const __rec = (name) => (...args) => { __log.push([name, ...args]); };
  const __out = () => __log.slice();
  const window = { fllmUsageSummary: () => ({ text: '42 in / 7 out in 3 answer(s)' }), fllmSkillPropose: null };
  const localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
  const state = { selectedModel: 'qwen', status: { model: { name: 'qwen' } } };
  const code = {
    messages: [], view: [], workspace: { root: '/home/marco/p' }, sessionId: 's1', title: 't',
    titleLocked: false, running: false, perm: 'ask', modes: {}, autoApprove: true, turbo: false,
    sandbox: 'workspace', sandboxAvailable: true, rules: { allow: '', deny: '' }, custom: [],
    personality: 'default', learn: false, autoCheck: false, autoReview: false, lastBuildPerm: 'ask', memory: '',
    compactedAt: 0, reverted: null, alwaysAllow: new Set(), registry: new Map(), abort: null,
    sideOpen: {}, memDraft: null, todos: [], changed: new Map(), disabled: null,
  };
  const codeNotice = __rec('notice');
  const pushChoices = (...a) => { __log.push(['choices', ...a]); };
  const toast = __rec('toast');
  const confirmDialog = async (...a) => { __log.push(['confirm', ...a]); return true; };
  const apiFetch = async (...a) => { __log.push(['api', ...a]); return { ok: true, json: async () => ({}) }; };
  let toolCall = async (...a) => { __log.push(['tool', ...a]); return { ok: true, output: 'out', error: '' }; };
  const userRun = __rec('userRun');
  const userTool = __rec('userTool');
  const agentRun = __rec('agentRun');
  const setPerm = __rec('setPerm');
  const toggleAgent = __rec('toggleAgent');
  const newCodeSession = __rec('newSession');
  const undoChange = __rec('undo');
  const showRewind = __rec('rewind');
  const compactSession = __rec('compact');
  const specCommand = __rec('spec');
  const workspaceSessions = () => [{ id: 'other', title: 'Older', time: 1699990000 }];
  const turnChoices = () => [{ label: 'Fix the bug', data: 'm1' }];
  const forkSession = __rec('fork');
  const revertToMessage = __rec('revert');
  const unrevert = __rec('unrevert');
  const cleanTitle = (t) => 'Clean ' + t;
  const renameSession = __rec('rename');
  const exportSession = __rec('export');
  const modelNames = () => ['qwen', 'llama'];
  const pickModel = __rec('pickModel');
  const saveMemoryEntry = async (...a) => { __log.push(['memory', ...a]); return true; };
  const userToolSide = __rec('side');
  const renderCodeSide = __rec('renderSide');
  const renderCode = __rec('render');
  const renderCodeSideTrue = __rec('renderSide');
  const saveRules = __rec('saveRules');
  const saveCodeSettings = __rec('saveSettings');
  const saveCodeSession = __rec('saveSession');
  const stopAgent = __rec('stopAgent');
  const setWorkspace = __rec('setWorkspace');
  const pickWorkspace = __rec('pickWorkspace');
  const resumeSession = __rec('resume');
  const pickMention = __rec('pickMention');
  const readCustomCommand = async (c) => { __log.push(['readCustom', c.name]); return 'Do it\\n\\nARGUMENTS: x'; };
  const fllmLearnStatus = async () => null;
  const fllmLearnSetEnabled = async () => {};
  const fllmLearnFeedback = async () => {};
  const fllmSkillPropose = null;
  const fireHook = __rec('hook');
  const $ = (id) => ({ scrollIntoView: () => {}, querySelector: () => null, querySelectorAll: () => [], textContent: '', classList: { remove() {} } });
`;

const CASES = [
  // [text, pre-setup script (optional), canned answers for confirmDialog/apiFetch/json]
  ['/clear'],
  ['/new'],
  ['/autoapprove off'],
  ['/autoapprove'],
  ['/AutoApprove on'],
  ['/plan'],
  ['/ask'],
  ['/edit'],
  ['/build'],
  ['/agent'],
  ['/stop'],
  ['/status'],
  ['/diff'],
  ['/cd /tmp/other'],
  ['/cd'],
  ['/help'],
  ['/help', '', { custom: [{ name: 'deploy', description: 'Ship it' }] }],
  ['/rules'],
  ['/allow bash: npm test*'],
  ['/allow nope: x'],
  ['/allow'],
  ['/deny read: /tmp/*'],
  ['/personality'],
  ['/personality concise'],
  ['/personality nope'],
  ['/remember a short fact'],
  ['/remember global everywhere'],
  ['/remember'],
  ['/autoreview on'],
  ['/autoreview'],
  ['/turbo on', '', { turboConfirm: true, runtime: { turbo: true, sandbox: 'off' } }],
  ['/turbo on', '', { turboConfirm: false }],
  ['/turbo'],
  ['/sandbox off', '', { runtime: { sandbox: 'off', sandbox_available: true } }],
  ['/sandbox weird'],
  ['/sandbox', '', { runtime: { active: false, mode: 'workspace', sandbox: { reason: 'no' } } }],
  ['/search'],
  ['/search words', '', { search: { hits: [{ id: 'h1', title: 'T', kind: 'chat', updated: 1699990000, snippet: 'snip' }] } }],
  ['/cost'],
  ['/insights'],
  ['/insights', '', { savedSessions: [
    { messages: [
      { role: 'user' },
      { role: 'assistant', tool_calls: [{ function: { name: 'bash' } }, { function: { name: 'write_file' } }] },
      { role: 'user' },
      { role: 'assistant', tool_calls: [{ function: { name: 'bash' } }] },
    ] },
    { messages: [ { role: 'user' }, { role: 'assistant', content: 'x', tool_calls: [{ function: { name: 'bash' } }] } ] },
  ] }],
  ['/model'],
  ['/sessions'],
  ['/memory'],
  ['/init'],
  ['/undo 3'],
  ['/rename New title'],
  ['/rename'],
  ['/export'],
  ['/skill'],
  ['/compact'],
  ['/resume'],
  ['/fork'],
  ['/revert'],
  ['/retry'],
  ['/rewind'],
  ['/spec a b'],
  ['/review staged'],
  ['/security-review'],
  ['/worktree'],
  ['/batch'],
  ['/check src'],
  ['/good', '', { learn: {} }],
  ['/bad', '', { learn: {} }],
  ['/learn', '', { learn: { status: null } }],
  ['/learn', '', { learn: { status: { enabled: true } } }],
  ['/learn', '', { learn: { status: { enabled: true, ask_first: true } } }],
  ['/learn', '', { learn: { status: { enabled: true, pending: 1 } } }],
  ['/learn', '', { learn: { status: { enabled: false, pending: 3, ask_first: true } } }],
  ['/learn', '', { learn: { status: { pending: 0 } } }],
  ['/learn on', '', { learn: { status: { enabled: true } } }],
  ['/learn ON', '', { learn: { status: { enabled: true, pending: 2 } } }],
  ['/learn off', '', { learn: { status: { enabled: false } } }],
  ['/learn off', 'code.learn = true;', { learn: { status: null } }],
  ['/learn banana', '', { learn: { status: { enabled: true } } }],
  ['/learn project on', '', { learn: { status: { enabled: true } } }],
  ['/learn project OFF', '', { learn: { status: { enabled: false } } }],
  ['/learn project default', '', { learn: { status: { enabled: true } } }],
  ['/learn project', '', { learn: { status: { enabled: true } } }],
  ['/learn project maybe', '', { learn: { status: { enabled: true } } }],
  ['/learn project on', '', { learn: { status: null } }],
  ['/deploy something', '', { custom: [{ name: 'deploy', description: 'Ship it' }] }],
  ['/agent', 'code.running = true;'],
  ['/stop', 'code.running = true;'],
  ['/unknownthing'],
  ['not a command at all'],
  ['/autoapprover off'],
];

// What one call decides, given the state the page holds for it: `{ decision, autoNote }` —
// 'allow', 'ask', 'deny' or the plan refusal, with the table's reason when it lifted a question.
const DECISIONS = [
  { name: 'plan refuses an edit', entry: { name: 'edit_file', permission: 'edit', source: 'server' }, args: { path: 'a.rs' }, perm: 'plan' },
  { name: 'plan lets a read run', entry: { name: 'read_file', permission: 'read', source: 'server' }, args: { path: 'a.rs' }, perm: 'plan' },
  { name: 'turbo runs a risky command', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'rm -rf /tmp/x' }, turbo: true },
  { name: 'deny rule wins over the mode', entry: { name: 'read_file', permission: 'read', source: 'server' }, args: { path: '/etc/hosts' }, rules: { deny: 'read: /etc/*' } },
  { name: 'an allow rule lifts a question', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'npm test -- --watch' }, rules: { allow: 'bash: npm test*' } },
  { name: 'a risky program only when the rule names it', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'git log -1' }, rules: { allow: 'bash: git log*' } },
  { name: 'a risky program without its name asks', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'git log -1' }, rules: { allow: 'bash: *' } },
  { name: 'an allow rule never covers a compound command', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'git status && rm -rf /' }, rules: { allow: 'bash: git status' } },
  { name: 'a compound command matches itself exactly', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'ls && ls' }, rules: { allow: 'bash: ls && ls' } },
  { name: 'a path rule allows an edit', entry: { name: 'edit_file', permission: 'edit', source: 'server' }, args: { path: 'src/lib.rs' }, rules: { allow: 'edit: src/*.rs' } },
  { name: 'a deny rule blocks an edit', entry: { name: 'write_file', permission: 'edit', source: 'server' }, args: { path: '/etc/passwd' }, rules: { deny: 'edit: /etc/*' } },
  { name: 'one path of a patch is enough to deny', entry: { name: 'multi_edit', permission: 'edit', source: 'server' }, args: { patch: '--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-1\n+2\n--- a/y.txt\n+++ b/y.txt\n@@ -1 +1 @@\n-1\n+2\n' }, rules: { deny: 'edit: y.txt' } },
  { name: 'a remembered command runs without a question', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'npm test --watch' }, always: ['bash:npm'] },
  { name: 'a remembered external tool runs', entry: { name: 'tool', permission: 'read', source: 'mcp', server: 'srv', tool: 'fetch' }, args: {}, always: ['mcp:srv/fetch'] },
  { name: 'a remembered network request runs', entry: { name: 'web_fetch', permission: 'network', source: 'server' }, args: { url: 'https://api.github.com/x' }, always: ['network'] },
  { name: 'a network rule allows the url', entry: { name: 'web_fetch', permission: 'network', source: 'server' }, args: { url: 'https://api.github.com/x' }, rules: { allow: 'network: https://api.github.com/*' } },
  { name: 'a memory rule allows the scope', entry: { name: 'memory', permission: 'memory', source: 'server' }, args: { scope: 'workspace' }, rules: { allow: 'memory: workspace' } },
  { name: 'the star category reaches every call', entry: { name: 'read_file', permission: 'read', source: 'server' }, args: { path: 'anything' }, rules: { deny: '*: any*' } },
  { name: 'a rule of another agent does not apply', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'ls' }, rules: { allow: '[explore] bash: ls' } },
  { name: 'a rule of the plan agent does not apply to build', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'ls -la' }, rules: { allow: '[plan] bash: ls*' } },
  { name: 'the table lifts a harmless command', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'ls -la' } },
  { name: 'the table leaves a risky one asking', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'git push' } },
  { name: 'without the switch the harmless command asks', entry: { name: 'bash', permission: 'command', source: 'server' }, args: { command: 'ls -la' }, autoApprove: false },
  { name: 'the auto-edit switch allows edits for build', entry: { name: 'edit_file', permission: 'edit', source: 'server' }, args: { path: 'a.rs' }, perm: 'edit' },
  { name: 'the command category is the one the rules judge', entry: { name: 'shell', permission: 'command', source: 'server' }, args: { words: 'ls -la' }, rules: { deny: 'command: *ls*' } },
];

// the scopes of `/review` the page reads out of its argument (row 81)
const REVIEW_TEXTS = ['/review', '/review branch', '/review branch v2', '/security', '/check'];

// the worktree commands with the answers of the server canned (row 82): every row brings the
// data of its own fetch, the rest of the apiFetch stays the recorder of the harness
const WORKTREE_CASES = [
  {
    text: '/worktree',
    canned: {
      worktree: {
        data: {
          worktrees: [
            { current: true, branch: 'main', path: '/home/marco/p' },
            { current: false, name: 'feat', branch: 'feat', path: '/home/marco/p/.fllm/worktrees/feat' },
          ],
        },
      },
    },
  },
  { text: '/worktree new feat', canned: { worktree: { data: { path: '/home/marco/p/.fllm/worktrees/feat', branch: 'feat' } } } },
  { text: '/worktree leave', canned: { worktree: { data: { path: '/home/marco/p', branch: 'main' } } } },
  { text: '/worktree remove feat', canned: { worktree: { data: { removed: 'feat' } } } },
  { text: '/worktree remove', canned: null },
  { text: '/wt new x', canned: { worktree: { data: { path: '/home/marco/p/.fllm/worktrees/x', branch: 'x' } } } },
  { text: '/batch', canned: null },
  { text: '/batch feat', canned: null },
  { text: '/batch feat do it', canned: { worktree: { data: { path: '/home/marco/p/.fllm/worktrees/feat', branch: 'feat' } } } },
  {
    // the server refuses: the page shows the message of its body
    text: '/worktree leave',
    canned: { worktree: { ok: false, data: { error: { message: 'not a git repository' } } } },
  },
];

// the spec commands with the answers of the server canned (row 80): `specs` is what the page
// has loaded when the command comes (`code.specs`), `spec.list` what a reload then reads,
// `spec.prompt` the prompt `task_prompt` / `new_prompt` hand back, `spec.error` a refusal
const SPEC_EXPORT = {
  name: 'export', requirements: true, design: true, tasks_file: true, total: 3, done: 1,
  tasks: [
    { index: 1, text: 'Write the exporter', done: true, depth: 0 },
    { index: 2, text: 'Wire the button', done: false, depth: 0 },
    { index: 3, text: 'Add a test', done: false, depth: 1 },
  ],
};
const SPEC_DONE = {
  name: 'search', requirements: true, design: true, tasks_file: true, total: 1, done: 1,
  tasks: [{ index: 1, text: 'Index the pages', done: true, depth: 0 }],
};
const SPEC_CASES = [
  { text: '/spec', specs: [], canned: { spec: { list: [] } } },
  { text: '/spec', specs: [], canned: { spec: { list: [SPEC_EXPORT, SPEC_DONE] } } },
  { text: '/spec   ', specs: [SPEC_EXPORT], canned: { spec: { list: [SPEC_EXPORT] } } },
  { text: '/spec run export', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 2 of export.' } } },
  { text: '/spec run export 3', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 3 of export.' } } },
  { text: '/spec run export 9', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 9.' } } },
  { text: '/spec run export 0', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 2 of export.' } } },
  { text: '/spec run export x', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 2 of export.' } } },
  { text: '/spec run export -1', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task -1.' } } },
  { text: '/spec run export 1.5', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 1.5.' } } },
  { text: '/spec run export 0x2', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 2 of export.' } } },
  { text: '/spec run export 1e1', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 10.' } } },
  { text: '/spec run export inf', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 2 of export.' } } },
  { text: '/spec run export Infinity', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task Infinity.' } } },
  { text: '/spec run export 3 extra words', specs: [SPEC_EXPORT], canned: { spec: { prompt: 'Do task 3 of export.' } } },
  { text: '/spec run search', specs: [SPEC_DONE], canned: { spec: { prompt: 'unused' } } },
  { text: '/spec run search 1', specs: [SPEC_DONE], canned: { spec: { prompt: 'Do task 1 again.' } } },
  { text: '/spec run nope', specs: [SPEC_EXPORT], canned: { spec: { list: [SPEC_EXPORT] } } },
  { text: '/spec run', specs: [], canned: { spec: { list: [] } } },
  // the list the page holds is stale: the reload finds the spec and the run goes on
  { text: '/spec run export', specs: [], canned: { spec: { list: [SPEC_EXPORT], prompt: 'Do task 2 of export.' } } },
  { text: '/spec run export', specs: [], canned: { spec: { list: [SPEC_DONE] } } },
  { text: '/spec run search', specs: [], canned: { spec: { list: [SPEC_DONE] } } },
  { text: '/spec csv-export Export the report as CSV.', specs: [], canned: { spec: { prompt: 'Write requirements.md for csv-export.' } } },
  { text: '/spec   csv-export   Export   the report', specs: [], canned: { spec: { prompt: 'Write requirements.md for csv-export.' } } },
  { text: '/spec csv-export', specs: [], canned: null },
  { text: '/spec run export 2', specs: [SPEC_EXPORT], canned: { spec: { ok: false, error: 'the task has no text' } } },
  { text: '/spec run export 2', specs: [SPEC_EXPORT], canned: { spec: { ok: false } } },
  { text: '/spec csv-export Export it', specs: [], canned: { spec: { ok: false, error: 'a spec of that name exists' } } },
];

// ---- row 81: the self-review pair of codeprompts.js, called as the page calls it ----------
const REVIEW_CASES = [
  // isCleanDiagnostic: the four shapes the page counts as clean, and two that are not
  { fn: 'isCleanDiagnostic', args: ['No problems found.'] },
  { fn: 'isCleanDiagnostic', args: ['src/lib.rs: is not available here'] },
  { fn: 'isCleanDiagnostic', args: ['cargo check: no automatic check is known for .foo'] },
  { fn: 'isCleanDiagnostic', args: ['No such file or directory'] },
  { fn: 'isCleanDiagnostic', args: ['src/a.rs:10:5: expected `i32`, found `u8`'] },
  { fn: 'isCleanDiagnostic', args: ['  No problems found'] },
  { fn: 'isCleanDiagnostic', args: [''] },
  { fn: 'isCleanDiagnostic', args: [null] },
  // reviewFeedback: round 1 always sends, round 2 only for something a machine can state
  { fn: 'reviewFeedback', args: [{ review: '', problems: [] }, 1] },
  { fn: 'reviewFeedback', args: [{ review: 'all good', problems: [] }, 1] },
  { fn: 'reviewFeedback', args: [{ review: '[high] unsafe unwrap at src/main.rs:20', problems: [] }, 1] },
  { fn: 'reviewFeedback', args: [{ review: '', problems: [{ path: 'src/lib.rs', output: 'line 3: unused variable\n' }] }, 1] },
  { fn: 'reviewFeedback', args: [{ review: 'note: something', problems: [] }, 2] },
  { fn: 'reviewFeedback', args: [{ review: '', problems: [{ path: 'a', output: 'x' }] }, 2] },
  { fn: 'reviewFeedback', args: [{ review: '[high] in review', problems: [{ path: 'b', output: 'y' }] }, 2] },
  { fn: 'reviewFeedback', args: [{ review: '', problems: [{ path: 'big.rs', output: 'z'.repeat(4000) }] }, 1] },
  { fn: 'reviewFeedback', args: [{ review: 'r'.repeat(20000), problems: [] }, 1] },
];

// ---- the check after an edit (autoCheckEdit), with canned answers per diagnostics call -----
const AUTOCHECK_CASES = [
  { name: 'edit_file', args: { path: 'src/lib.rs' }, tools: [{ ok: true, output: 'src/lib.rs:3: unused variable\n' }] },
  { name: 'edit_file', args: { path: 'src/lib.rs' }, tools: [{ ok: true, output: 'No problems found' }] },
  { name: 'edit_file', args: { path: 'notes.txt' }, tools: [] },
  { name: 'edit_file', args: { path: 'a.json' }, tools: [{ ok: true, output: 'a.json:1: trailing comma' }] },
  {
    name: 'patch',
    args: {
      patch: '--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-1\n+2\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-1\n+2\n--- a/readme.md\n+++ b/readme.md\n@@ -1 +1 @@\n-a\n+b\n',
    },
    tools: [{ ok: true, output: 'src/main.rs:9: unused\n' }, { ok: true, output: 'src/lib.rs:2: dead code' }],
  },
  { name: 'edit_file', args: { path: 'src/denied.rs' }, tools: [], deny: 'bash: diagnostics src/denied.rs' },
  {
    // every extension of CHECKED_EXTENSIONS in one patch: one report per extension, and a
    // `.md` file that is never checked
    name: 'patch',
    args: {
      patch: ['a.py', 'b.js', 'c.mjs', 'd.cjs', 'e.ts', 'f.tsx', 'g.go', 'h.json', 'i.md']
        .map((f) => `--- a/${f}\n+++ b/${f}\n@@ -1 +1 @@\n-1\n+2`)
        .join('\n'),
    },
    tools: ['a.py', 'b.js', 'c.mjs', 'd.cjs', 'e.ts', 'f.tsx', 'g.go', 'h.json'].map((f) => ({ ok: true, output: f })),
  },
  {
    // the four paths after them: ts, tsx, go and json are checked as well (the walk looks at
    // `paths.slice(0, 4)` only, so the first four above never reach these extensions)
    name: 'patch',
    args: {
      patch: ['e.ts', 'f.tsx', 'g.go', 'h.json', 'i.md']
        .map((f) => `--- a/${f}\n+++ b/${f}\n@@ -1 +1 @@\n-1\n+2`)
        .join('\n'),
    },
    tools: ['e.ts', 'f.tsx', 'g.go', 'h.json'].map((f) => ({ ok: true, output: f })),
  },
  {
    name: 'edit_file',
    args: { path: 'src/quiet.rs' },
    tools: [{ ok: true, output: 'src/quiet.rs:1: is not available here' }, { ok: true, output: 'more' }],
  },
  {
    // a deleted file: the diff names `b/` as /dev/null, so the `a/` side is the path
    name: 'patch',
    args: { patch: '--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-1\n' },
    tools: [{ ok: true, output: 'gone.py: went away' }],
  },
  {
    // the first file is denied — the `seen` set is filled *after* the rule check, so the
    // second file of the same extension is still asked about
    name: 'patch',
    args: {
      patch: ['src/denied.rs', 'src/clean.rs'].map((f) => `--- a/${f}\n+++ b/${f}\n@@ -1 +1 @@\n-1\n+2`).join('\n'),
    },
    deny: 'bash: diagnostics src/denied.rs',
    tools: [{ ok: true, output: 'src/clean.rs:2: unused' }],
  },
  {
    name: 'edit_file',
    args: { path: 'src/slow.rs' },
    tools: [{ ok: true, output: 'cargo check did not finish within 60s' }],
  },
];

// ---- the reply the model gets back when the check found something ---------------------------
const REPLY_CASES = [
  { reply: 'Edited one line.', problems: 'src/lib.rs:3: unused variable' },
  { reply: 'x'.repeat(9000), problems: 'y'.repeat(8000) },
  { reply: '', problems: '' },
  // a small context: the cap of clipOutput never sinks below 3000
  { reply: 'z'.repeat(4000), problems: '', ctx: 1024 },
];

async function main() {
  const context = { console, __log: [], __out: () => [] };
  const setup = [harness(), PURE_CODE, PURE_PROMPTS, BUILTIN, INIT, TOGGLE, MAX_SESSIONS, AGENT_NAME, READ_ONLY, PARSED_RULES, OUTSIDE, DISPATCH, AUTOCHECK, CLIP, SPECS_A, SPECS_B].join('\n');
  vm.createContext(context);
  try {
    vm.runInContext(setup, context, { filename: 'page-slices.js' });
  } catch (e) {
    console.error('HARNESS SETUP FAILED:', e.message);
    process.exit(2);
  }

  const out = [];
  let gaps = 0;
  const allCases = [
    ...CASES,
    ...REVIEW_TEXTS.map((text) => [text, null, null]),
    ...WORKTREE_CASES.map((row) => [row.text, null, row.canned]),
    ...SPEC_CASES.map((row) => [row.text, `code.specs = ${JSON.stringify(row.specs)};`, row.canned]),
  ];
  for (const [text, setupScript, canned] of allCases) {
    vm.runInContext('__reset();', context);
    if (setupScript) vm.runInContext(setupScript, context, { filename: 'setup.js' });
    if (canned && canned.custom) vm.runInContext(`code.custom = ${JSON.stringify(canned.custom)};`, context);
    if (canned && canned.savedSessions) vm.runInContext(`__savedSessions = ${JSON.stringify(canned.savedSessions)};`, context);
    const pre = vm.runInContext('__snapshot()', context);
    // reset the recorders of this run
    vm.runInContext('__log.length = 0;', context);
    if (canned && 'turboConfirm' in canned) {
      vm.runInContext(`confirmDialog = async (...a) => { __log.push(['confirm', ...a]); return ${canned.turboConfirm}; };`, context);
    }
    if (canned && canned.runtime) {
      vm.runInContext(
        `apiFetch = async (...a) => { __log.push(['api', ...a]); return { ok: true, json: async () => (${JSON.stringify(canned.runtime)}) }; };`,
        context,
      );
    }
    if (canned && canned.worktree) {
      const wt = canned.worktree;
      vm.runInContext(
        `apiFetch = async (url, opts) => {
           __log.push(['api', url, opts]);
           if (String(url).includes('/agent/worktree')) {
             return { ok: ${wt.ok !== false}, json: async () => (${JSON.stringify(wt.data || {})}) };
           }
           return { ok: true, json: async () => ({}) };
         };`,
        context,
      );
    }
    if (canned && canned.spec) {
      const sp = canned.spec;
      vm.runInContext(
        `apiFetch = async (url, opts) => {
           __log.push(opts === undefined ? ['api', url] : ['api', url, opts]);
           if (String(url).includes('/agent/specs')) {
             if (opts && opts.method === 'POST') {
               return { ok: ${sp.ok !== false}, json: async () => (${sp.ok === false ? JSON.stringify({ error: sp.error ? { message: sp.error } : undefined }) : JSON.stringify({ prompt: sp.prompt })}) };
             }
             return { ok: true, json: async () => ({ specs: ${JSON.stringify(sp.list || [])} }) };
           }
           return { ok: true, json: async () => ({}) };
         };`,
        context,
      );
    }
    if (canned && canned.learn) {
      // row 88: the three functions learn.js hangs on `window`, with the status of the server canned
      vm.runInContext(
        `window.fllmLearnStatus = async () => { __log.push(['learnStatus']); return ${JSON.stringify(canned.learn.status === undefined ? null : canned.learn.status)}; };
         window.fllmLearnSetEnabled = async (on) => { __log.push(['learnSet', on]); return true; };
         window.fllmLearnFeedback = async (value) => { __log.push(['learnFeedback', value]); return true; };`,
        context,
      );
    }
    if (canned && canned.search) {
      vm.runInContext(
        `apiFetch = async (...a) => { __log.push(['api', ...a]); return { ok: true, json: async () => (${JSON.stringify(canned.search)}) }; };`,
        context,
      );
    }
    let ret = null;
    let error = null;
    try {
      ret = await vm.runInContext(`runCodeCommand(${JSON.stringify(text)})`, context);
    } catch (e) {
      error = e.message;
      if (/is not defined/.test(e.message)) gaps++;
    }
    const record = vm.runInContext('__out()', context);
    const state = vm.runInContext(
      "(() => ({ title: code.title, titleLocked: code.titleLocked, rules: code.rules," +
        " personality: code.personality, learn: code.learn, autoCheck: code.autoCheck, autoReview: code.autoReview," +
        " autoApprove: code.autoApprove, perm: code.perm, lastBuildPerm: code.lastBuildPerm," +
        " sandbox: code.sandbox, turbo: code.turbo, view: code.view.length }))()",
      context,
    );
    out.push({ text, ret, error, record, pre, state, canned: canned || null });
  }

  // the decisions of one call, inputs and all, so the window replays them state by state
  const decisions = [];
  let decisionGaps = 0;
  for (const d of DECISIONS) {
    let result = null;
    let error = null;
    try {
      result = vm.runInContext(`__decision(${JSON.stringify(d)})`, context);
    } catch (e) {
      error = e.message;
      if (/is not defined/.test(e.message)) decisionGaps++;
    }
    decisions.push({ ...d, result, error });
  }

  // ---- row 81: the self-review pair, called the way the page calls it --------------------
  const review = [];
  let reviewGaps = 0;
  for (const c of REVIEW_CASES) {
    let result = null;
    let error = null;
    try {
      result = vm.runInContext(`${c.fn}(${c.args.map((a) => JSON.stringify(a)).join(',')})`, context);
    } catch (e) {
      error = e.message;
      if (/is not defined/.test(e.message)) reviewGaps++;
    }
    review.push({ fn: c.fn, args: c.args, result, error });
  }

  // ---- the check after an edit: autoCheckEdit with canned diagnostics answers -------------
  const autocheck = [];
  for (const c of AUTOCHECK_CASES) {
    vm.runInContext('__reset();', context);
    if (c.setup) vm.runInContext(c.setup, context, { filename: 'setup.js' });
    if (c.deny) vm.runInContext(`code.rules = { allow: '', deny: ${JSON.stringify(c.deny)} };`, context);
    vm.runInContext('__log.length = 0;', context);
    vm.runInContext(
      `code.autoCheck = true; __queue = ${JSON.stringify(c.tools)}; toolCall = async (...a) => { __log.push(['tool', ...a]); return __queue.shift() || { ok: false, error: 'none' }; };`,
      context,
    );
    let result = null;
    let error = null;
    try {
      // eslint-disable-next-line no-await-in-loop
      result = await vm.runInContext(`autoCheckEdit(${JSON.stringify(c.name)}, ${JSON.stringify(c.args)}, { readOnly: false })`, context);
    } catch (e) {
      error = e.message;
      if (/is not defined/.test(e.message)) reviewGaps++;
    }
    const calls = vm.runInContext('__log.slice()', context);
    autocheck.push({ name: c.name, args: c.args, deny: c.deny || null, tools: c.tools, result, calls, error });
  }

  // ---- the reply the model gets back when the check found something ------------------------
  const replyLine = code.find((l) => l.includes('[automatic check found problems]')) || '';
  const replyRhs = replyLine.trim().replace(/^reply\s*=\s*/, '').replace(/;\s*$/, '');
  const reply = [];
  for (const c of REPLY_CASES) {
    let result = null;
    let error = null;
    try {
      // the cap of clipOutput reads the model's context — every row brings its own
      vm.runInContext(`ctxMax = () => ${c.ctx ?? 8192};`, context);
      result = vm.runInContext(
        `(() => { ${CLIP} const build = (reply, problems) => ${replyRhs}; return build(${JSON.stringify(c.reply)}, ${JSON.stringify(c.problems)}); })()`,
        context,
      );
    } catch (e) {
      error = e.message;
      if (/is not defined/.test(e.message)) reviewGaps++;
    }
    reply.push({ ...c, ctx: c.ctx ?? 8192, result, error });
  }

  fs.writeFileSync(
    path.join(__dirname, 'command_fixtures.json'),
    JSON.stringify({ cases: out, decisions, review, autocheck, reply }, null, 1),
  );
  console.log(`cases: ${out.length}, review: ${review.length}, autocheck: ${autocheck.length}, reply: ${reply.length}, harness gaps: ${gaps}, decision gaps: ${decisionGaps}, row81 gaps: ${reviewGaps}`);
  if (reviewGaps) {
    for (const r of [...review, ...autocheck, ...reply]) if (r.error) console.log('ROW81 GAP', r.fn || r.name || r.reply?.slice(0, 20), '->', r.error);
  }
  if (decisionGaps) {
    for (const d of decisions) if (d.error) console.log('DECISION GAP', d.name, '->', d.error);
  }
  if (gaps) {
    for (const c of out) if (c.error && /is not defined/.test(c.error)) console.log('GAP', c.text, '->', c.error);
    process.exit(1);
  }
  if (decisionGaps || reviewGaps) process.exit(1);
  console.log('FIXTURES WRITTEN');
}

main();
