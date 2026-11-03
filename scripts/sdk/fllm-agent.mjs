// TypeScript/JavaScript client for `fllm-rs agent --rpc`: drive the fLLM coding agent from your own program.
//
// Plain ES module for Node 18 or newer, no dependencies; fllm-agent.d.ts has the types. The agent runs as a child process and
// talks JSON-RPC 2.0, one message per line, over its standard input and output. A server with a model that can call tools
// must be running (`fllm-rs serve ...`).
//
//   import { Agent } from './fllm-agent.mjs';
//   const agent = await Agent.start({ url: 'http://127.0.0.1:8000' });
//   const session = await agent.newSession('/path/to/project', { allow: ['edit'] });
//   const result = await session.prompt('Fix the failing test', {
//     onEvent: (e) => console.log(e.type, e.name ?? e.text?.slice(0, 60) ?? ''),
//     approve: (call) => call.tool === 'bash' && call.args.command.startsWith('npm test'),
//   });
//   console.log(result.status, result.answer);
//   await agent.close();
//
// approve(call) is asked for every action the session's allow list does not cover; return true to allow once,
// { allow: true, always: true } to allow that category for the session. Without it such actions are refused.
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';

export class AgentError extends Error {
  constructor(code, message) {
    super(`${message} (code ${code})`);
    this.code = code;
  }
}

export class Agent {
  #proc;
  #nextId = 1;
  #pending = new Map(); // id -> { resolve, reject, onMessage }
  #exited = false;

  constructor(proc) {
    this.#proc = proc;
    const lines = createInterface({ input: proc.stdout });
    lines.on('line', (line) => this.#onLine(line));
    proc.on('exit', () => {
      this.#exited = true;
      for (const p of this.#pending.values()) p.reject(new AgentError(-32000, 'the agent process ended'));
      this.#pending.clear();
    });
  }

  /** Starts `fllm-rs agent --rpc` and waits for initialize. */
  static async start({ exe = 'fllm-rs', url, apiKey, model, extraArgs = [] } = {}) {
    const args = ['agent', '--rpc'];
    if (url) args.push('--url', url);
    if (apiKey) args.push('--api-key', apiKey);
    if (model) args.push('--model', model);
    args.push(...extraArgs);
    const proc = spawn(exe, args, { stdio: ['pipe', 'pipe', 'inherit'] });
    const agent = new Agent(proc);
    await new Promise((resolve, reject) => { proc.once('spawn', resolve); proc.once('error', reject); });
    agent.info = await agent.request('initialize');
    return agent;
  }

  #write(message) {
    this.#proc.stdin.write(`${JSON.stringify(message)}\n`);
  }

  #onLine(line) {
    if (!line.trim()) return;
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.id !== undefined && msg.method === undefined) {
      const p = this.#pending.get(msg.id);
      if (!p) return;
      this.#pending.delete(msg.id);
      if (msg.error) p.reject(new AgentError(msg.error.code ?? -32603, msg.error.message ?? 'error'));
      else p.resolve(msg.result);
      return;
    }
    // a notification or a request from the agent belongs to the prompt that is running (one at a time per process)
    for (const p of this.#pending.values()) if (p.onMessage) { p.onMessage(msg); return; }
  }

  request(method, params = {}, onMessage = null) {
    if (this.#exited) return Promise.reject(new AgentError(-32000, 'the agent process ended'));
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject, onMessage });
      this.#write({ jsonrpc: '2.0', id, method, params });
    });
  }

  notify(method, params = {}) {
    this.#write({ jsonrpc: '2.0', method, params });
  }

  respond(id, result) {
    this.#write({ jsonrpc: '2.0', id, result });
  }

  /** folder must be absolute. allow: edit, command, network, memory or 'all' (reading is always allowed). */
  async newSession(folder, { allow = [], plan = false, model, system, maxSteps, allowRules = [], denyRules = [], askRules = [] } = {}) {
    const params = { folder, allow, plan, allow_rules: allowRules, deny_rules: denyRules, ask_rules: askRules };
    if (model) params.model = model;
    if (system) params.system = system;
    if (maxSteps) params.max_steps = maxSteps;
    const result = await this.request('session/new', params);
    return new Session(this, result.session);
  }

  async tools() { return (await this.request('tools/list')).tools; }

  async close() {
    if (this.#exited) return;
    try { await this.request('shutdown'); } catch { /* the process may already be going */ }
    this.#proc.stdin.end();
    await new Promise((resolve) => { if (this.#exited) resolve(); else this.#proc.once('exit', resolve); setTimeout(resolve, 3000); });
    if (!this.#exited) this.#proc.kill();
  }
}

export class Session {
  constructor(agent, id) {
    this.agent = agent;
    this.id = id;
  }

  /** Runs one task. Resolves to { answer, status, steps, edited, tool_calls }. */
  prompt(text, { onEvent, approve } = {}) {
    const handle = (msg) => {
      if (msg.method === 'session/event' && onEvent) onEvent(msg.params);
      else if (msg.method === 'session/approve') {
        Promise.resolve(approve ? approve(msg.params) : false).then((v) => {
          this.agent.respond(msg.id, typeof v === 'object' && v !== null ? { allow: Boolean(v.allow), always: Boolean(v.always) } : { allow: Boolean(v) });
        });
      }
    };
    return this.agent.request('session/prompt', { session: this.id, text }, handle);
  }

  cancel() { this.agent.notify('session/cancel', { session: this.id }); }
  async history() { return (await this.agent.request('session/history', { session: this.id })).messages; }
  close() { return this.agent.request('session/close', { session: this.id }); }
}
