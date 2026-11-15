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

const log = [];
const rec = (name) =>
  (...args) => {
    log.push([name, ...args]);
  };

// identifiers the slices themselves declare: a recorder of the harness must not collide with them
const pageNames = new Set(
  [PURE_CODE, PURE_PROMPTS, BUILTIN, INIT, TOGGLE, MAX_SESSIONS, AGENT_NAME, READ_ONLY, PARSED_RULES, OUTSIDE, DISPATCH].join('\n').match(
    /(?:const|let|function|class)\s+([A-Za-z_$][\w$]*)/g,
  ).map((s) => s.split(/\s+/)[1]),
);

// the state every case starts from (the page loads it from its server; here it is the seed)
const CODE_SEED =
  "{ messages: [], view: [], workspace: { root: '/home/marco/p' }, sessionId: 's1', title: 't'," +
  " titleLocked: false, running: false, perm: 'ask', modes: {}, autoApprove: true, turbo: false," +
  " sandbox: 'workspace', sandboxAvailable: true, rules: { allow: '', deny: '' }, custom: []," +
  " personality: 'default', learn: false, autoReview: false, lastBuildPerm: 'ask', memory: ''," +
  " compactedAt: 0, reverted: null, alwaysAllow: new Set(), registry: new Map(), abort: null," +
  " sideOpen: {}, memDraft: null, todos: [], changed: new Map(), disabled: null }";

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
  ['specCommand', "__rec('spec')"],
  ['workspaceSessions', "() => __savedSessions"],
  ['turnChoices', "() => [{ label: 'Fix the bug', data: 'm1' }]"],
  ['forkSession', "__rec('fork')"],
  ['revertToMessage', "__rec('revert')"],
  ['unrevert', "__rec('unrevert')"],
  ['cleanTitle', "(t) => 'Clean ' + t"],
  ['renameSession', "__rec('rename')"],
  ['exportSession', "__rec('export')"],
  ['modelNames', "() => ['qwen', 'llama']"],
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
    sandboxAvailable: code.sandboxAvailable, autoReview: code.autoReview,
    personality: code.personality, rules: { allow: code.rules.allow, deny: code.rules.deny },
    custom: code.custom.map((c) => ({ name: c.name, description: c.description })),
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
  const reassign = new Set(['confirmDialog', 'apiFetch']);
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
    personality: 'default', learn: false, autoReview: false, lastBuildPerm: 'ask', memory: '',
    compactedAt: 0, reverted: null, alwaysAllow: new Set(), registry: new Map(), abort: null,
    sideOpen: {}, memDraft: null, todos: [], changed: new Map(), disabled: null,
  };
  const codeNotice = __rec('notice');
  const pushChoices = (...a) => { __log.push(['choices', ...a]); };
  const toast = __rec('toast');
  const confirmDialog = async (...a) => { __log.push(['confirm', ...a]); return true; };
  const apiFetch = async (...a) => { __log.push(['api', ...a]); return { ok: true, json: async () => ({}) }; };
  const toolCall = async (...a) => { __log.push(['tool', ...a]); return { ok: true, output: 'out', error: '' }; };
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
  ['/learn'],
  ['/good'],
  ['/bad'],
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

async function main() {
  const context = { console, __log: [], __out: () => [] };
  const setup = [harness(), PURE_CODE, PURE_PROMPTS, BUILTIN, INIT, TOGGLE, MAX_SESSIONS, AGENT_NAME, READ_ONLY, PARSED_RULES, OUTSIDE, DISPATCH].join('\n');
  vm.createContext(context);
  try {
    vm.runInContext(setup, context, { filename: 'page-slices.js' });
  } catch (e) {
    console.error('HARNESS SETUP FAILED:', e.message);
    process.exit(2);
  }

  const out = [];
  let gaps = 0;
  for (const [text, setupScript, canned] of CASES) {
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
        " personality: code.personality, learn: code.learn, autoReview: code.autoReview," +
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

  fs.writeFileSync(
    path.join(__dirname, 'command_fixtures.json'),
    JSON.stringify({ cases: out, decisions }, null, 1),
  );
  console.log(`cases: ${out.length}, harness gaps: ${gaps}, decision gaps: ${decisionGaps}`);
  if (decisionGaps) {
    for (const d of decisions) if (d.error) console.log('DECISION GAP', d.name, '->', d.error);
  }
  if (gaps) {
    for (const c of out) if (c.error && /is not defined/.test(c.error)) console.log('GAP', c.text, '->', c.error);
    process.exit(1);
  }
  if (decisionGaps) process.exit(1);
  console.log('FIXTURES WRITTEN');
}

main();
