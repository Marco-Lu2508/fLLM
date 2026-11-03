// fLLM Agent for VS Code: a skeleton that starts `fllm-rs` in an integrated terminal. It has no UI of its own: the terminal shows the
// agent's progress and answer, and every permission is the flag you set in the settings (nothing runs without it). For a richer
// integration (streamed messages, approval dialogs) use the JSON-RPC mode `fllm-rs agent --rpc` (scripts/sdk/fllm-agent.mjs is a client).
'use strict';
const vscode = require('vscode');

function settings() {
  const c = vscode.workspace.getConfiguration('fllm');
  return { exe: c.get('executable', 'fllm-rs'), url: c.get('serverUrl', 'http://127.0.0.1:8000'), edits: c.get('allowEdits', false), commands: c.get('allowCommands', false) };
}

/**
 * A single argument for the terminal, in single quotes so that nothing inside is expanded: PowerShell (the default terminal on Windows)
 * doubles a quote, POSIX shells close and reopen it. The Windows command prompt (cmd.exe) does not understand single quotes: use PowerShell.
 */
function quote(text) {
  if (process.platform === 'win32') return `'${String(text).replace(/'/g, "''")}'`;
  return `'${String(text).replace(/'/g, `'\\''`)}'`;
}

function folder() {
  const f = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  return f ? f.uri.fsPath : null;
}

function inTerminal(args) {
  const dir = folder();
  if (!dir) { vscode.window.showWarningMessage('Open a folder first: the agent works in it.'); return; }
  const terminal = vscode.window.createTerminal({ name: 'fLLM agent', cwd: dir });
  terminal.show();
  terminal.sendText(args.join(' '));
}

function agentArgs(task, extra = []) {
  const s = settings();
  const args = [quote(s.exe), 'agent', '-p', quote(task), '--url', quote(s.url), '--folder', quote(folder())];
  if (s.edits) args.push('--allow-edit');
  if (s.commands) args.push('--allow-command');
  return args.concat(extra);
}

function activate(context) {
  const register = (id, fn) => context.subscriptions.push(vscode.commands.registerCommand(id, fn));
  register('fllm.runTask', async () => {
    const task = await vscode.window.showInputBox({ prompt: 'What should the agent do?', ignoreFocusOut: true });
    if (task) inTerminal(agentArgs(task));
  });
  register('fllm.askAboutSelection', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const question = await vscode.window.showInputBox({ prompt: 'Your question about the selected code', ignoreFocusOut: true });
    if (!question) return;
    const file = vscode.workspace.asRelativePath(editor.document.uri);
    const sel = editor.selection;
    const text = editor.document.getText(sel);
    const where = `${file}:${sel.start.line + 1}-${sel.end.line + 1}`;
    // read only: no --allow-* flags whatever the settings say
    const s = settings();
    inTerminal([quote(s.exe), 'agent', '-p', quote(`About ${where}:\n${text}\n\n${question}`), '--url', quote(s.url), '--folder', quote(folder())]);
  });
  register('fllm.reviewChanges', () => {
    const s = settings();
    inTerminal([quote(s.exe), 'review', '--url', quote(s.url), '--folder', quote(folder())]);
  });
  register('fllm.specNew', async () => {
    const name = await vscode.window.showInputBox({ prompt: 'Name of the spec (letters, digits, dashes)', ignoreFocusOut: true });
    if (!name) return;
    const what = await vscode.window.showInputBox({ prompt: 'What do you want built?', ignoreFocusOut: true });
    if (!what) return;
    const s = settings();
    inTerminal([quote(s.exe), 'spec', 'new', quote(name), quote(what), '--url', quote(s.url), '--folder', quote(folder())]);
  });
  register('fllm.specRun', async () => {
    const name = await vscode.window.showInputBox({ prompt: 'Name of the spec', ignoreFocusOut: true });
    if (!name) return;
    const s = settings();
    const args = [quote(s.exe), 'spec', 'run', quote(name), '--url', quote(s.url), '--folder', quote(folder())];
    if (s.commands) args.push('--allow-command');
    inTerminal(args);
  });
}

function deactivate() {}

module.exports = { activate, deactivate };
