    // ==========================================================================
    // Code: a coding agent in the browser. The model works through tools (read, search, change files,
    // run commands, ask the user, start sub-agents, call MCP tools) that the server executes inside one
    // workspace folder; this part is the loop, the permission questions and the view. Nothing is sent
    // anywhere except to this server.
    // ==========================================================================
    const CODE_MAX_TURNS = 40;
    const CHILD_MAX_STEPS = 16;
    const MAX_PARALLEL_CHILDREN = 4;
    const CODE_RECENT_KEY = 'fllm.studio.workspaces.v1';
    const CODE_SESSION_KEY = 'fllm.studio.codesession.v1'; // old single-session storage, still read once
    const CODE_SESSIONS_KEY = 'fllm.studio.codesessions.v2';
    const CODE_RULES_KEY = 'fllm.studio.coderules.v1';
    const INSTRUCTION_FILES = ['FLLM.md', 'AGENTS.md', 'CLAUDE.md'];
    const COMPACT_RATIO = 0.7;
    const TOO_SMALL = 'This model\'s context is too small for the agent; load it with a larger context under Models > Advanced.';

    // @pure-begin  (no DOM access in this block: it is loaded by tests/code.test.js)
    const RISKY_COMMANDS = new Set(['rm', 'rmdir', 'del', 'erase', 'rd', 'format', 'shutdown', 'reboot', 'mkfs', 'dd', 'sudo', 'su', 'curl', 'wget', 'iwr', 'invoke-webrequest', 'powershell', 'pwsh', 'cmd', 'sh', 'bash', 'chmod', 'chown', 'kill', 'taskkill', 'git']);
    const SHELL_OPS = /[;&|><`$(){}\n]/;

    /** Turns a rule pattern into a regular expression. Mode "path": * stops at "/", ** does not. Mode "text": * matches anything. */
    function globToRegex(pattern, mode) {
      const p = String(pattern);
      let re = '';
      for (let i = 0; i < p.length; i++) {
        const c = p[i];
        if (c === '*') {
          if (mode === 'path') {
            if (p[i + 1] === '*') {
              i++;
              if (p[i + 1] === '/') { i++; re += '(?:.*/)?'; } else re += '.*';
            } else re += '[^/]*';
          } else { while (p[i + 1] === '*') i++; re += '.*'; }
        } else if (c === '?') re += mode === 'path' ? '[^/]' : '.';
        else re += c.replace(/[.+^${}()|[\]\\]/g, '\\$&');
      }
      return new RegExp(`^${re}$`, 's');
    }

    const normPath = (p) => String(p || '').replace(/\\/g, '/').replace(/^\.\//, '').replace(/^\/+/, '');
    const RULE_CATEGORIES = new Set(['bash', 'edit', 'network', 'mcp', 'command', 'read', '*']);

    /** "bash: git status*" per line; blank lines and # comments are ignored. */
    function parseRules(text) {
      const rules = [];
      for (const line of String(text || '').split('\n')) {
        const m = line.match(/^\s*([A-Za-z*]+)\s*:\s*(.+?)\s*$/);
        if (!m || line.trim().startsWith('#')) continue;
        const category = m[1].toLowerCase();
        if (RULE_CATEGORIES.has(category)) rules.push({ category, pattern: m[2] });
      }
      return rules;
    }

    const commandBase = (command) => {
      const first = String(command || '').trim().split(/\s+/)[0] || '';
      return first.replace(/^.*[\\/]/, '').replace(/\.(exe|cmd|bat)$/i, '').toLowerCase();
    };
    const hasShellOps = (command) => SHELL_OPS.test(String(command));
    const commandSegments = (command) => String(command).split(/&&|\|\||[;&|\n]/).map((s) => s.trim()).filter(Boolean);

    /** The key of a command that may be allowed "always": the program name, or null when risky or compound. */
    function commandKey(command) {
      const base = commandBase(command);
      if (!base || RISKY_COMMANDS.has(base) || hasShellOps(command)) return null;
      return base;
    }

    const ruleAppliesTo = (rule, category) => rule.category === '*' || rule.category === category
      || (rule.category === 'command' && (category === 'bash' || category === 'mcp'));
    const patternMatches = (rule, category, subject) => {
      const path = category === 'edit' || category === 'read';
      return globToRegex(path ? normPath(rule.pattern) : rule.pattern, path ? 'path' : 'text').test(path ? normPath(subject) : String(subject).trim());
    };

    /** 'deny', 'allow' or null (no rule: ask). Deny wins. Allow rules never cover compound shell commands
     *  (unless the rule is that exact command) and only cover risky programs when the rule names the program. */
    function evaluateRules(rules, category, subject) {
      const deny = (rules.deny || []).filter((r) => ruleAppliesTo(r, category));
      const allow = (rules.allow || []).filter((r) => ruleAppliesTo(r, category));
      if (category === 'bash') {
        const parts = [String(subject).trim(), ...commandSegments(subject)];
        if (deny.some((r) => parts.some((s) => patternMatches(r, category, s)))) return 'deny';
        const compound = hasShellOps(subject);
        const base = commandBase(subject);
        const risky = RISKY_COMMANDS.has(base);
        const hit = allow.some((r) => {
          if (!patternMatches(r, category, subject)) return false;
          if (compound && r.pattern.trim() !== String(subject).trim()) return false;
          if (risky) {
            const head = r.pattern.trim().split(/\s+/)[0] || '';
            if (/[*?]/.test(head) || commandBase(head) !== base) return false;
          }
          return true;
        });
        return hit ? 'allow' : null;
      }
      if (deny.some((r) => patternMatches(r, category, subject))) return 'deny';
      return allow.some((r) => patternMatches(r, category, subject)) ? 'allow' : null;
    }

    const needsCompaction = (estimate, max, ratio = COMPACT_RATIO) => max > 0 && estimate > max * ratio;

    /** Splits the messages into the part to summarise and the recent part to keep. Tool results never lose their call. */
    function splitForCompaction(messages, keepLast = 6, minHead = 3) {
      let start = messages.length - keepLast;
      if (start < minHead) return null;
      while (start > 0 && messages[start].role === 'tool') start--;
      if (start < minHead) return null;
      return { head: messages.slice(0, start), tail: messages.slice(start) };
    }

    function buildCompactedMessages(summary, tail, extra = {}) {
      const parts = ['[Summary of the earlier conversation, written by you so that work can continue from it]', String(summary).trim()];
      if ((extra.todos || []).length) parts.push(`Task list:\n${extra.todos.map((t) => `- [${t.status}] ${t.content}`).join('\n')}`);
      if ((extra.files || []).length) parts.push(`Files changed in this session: ${extra.files.slice(0, 40).join(', ')}`);
      const out = [{ role: 'user', content: parts.join('\n\n') }];
      if (!tail.length || tail[0].role === 'user') out.push({ role: 'assistant', content: 'Understood. I will continue from this summary.' });
      return out.concat(tail);
    }

    /** A custom command file: optional front matter with description, then the prompt body. */
    function parseCommandFile(text) {
      let body = String(text || '').replace(/\r\n/g, '\n');
      let description = '';
      const fm = body.match(/^---\n([\s\S]*?)\n---\n?/);
      if (fm) {
        const d = fm[1].match(/^description:\s*(.+)$/m);
        if (d) description = d[1].trim().replace(/^["']|["']$/g, '');
        body = body.slice(fm[0].length);
      }
      body = body.trim();
      if (!description) description = (body.split('\n').find((l) => l.trim()) || '').replace(/^#+\s*/, '').slice(0, 80);
      return { description, body };
    }

    function splitArgs(text) {
      return [...String(text || '').matchAll(/"([^"]*)"|'([^']*)'|(\S+)/g)].map((m) => m[1] ?? m[2] ?? m[3]);
    }

    /** $ARGUMENTS is the whole text, $1..$9 single words. Without $ARGUMENTS the text is appended. */
    function expandCommand(body, argsText) {
      const args = splitArgs(argsText);
      const hasAll = /\$ARGUMENTS/.test(body);
      let out = body.replace(/\$ARGUMENTS/g, argsText.trim()).replace(/\$([1-9])/g, (_, n) => args[Number(n) - 1] ?? '');
      if (!hasAll && argsText.trim()) out += `\n\nARGUMENTS: ${argsText.trim()}`;
      return out;
    }

    /** Names of custom commands from the paths found under .fllm/commands. */
    function commandNamesFromPaths(paths) {
      const seen = new Set();
      const out = [];
      for (const raw of paths) {
        const m = normPath(raw).match(/^\.fllm\/commands\/(.+)\.md$/);
        if (!m) continue;
        const name = m[1].replace(/\//g, ':');
        if (/^[\w:.-]+$/.test(name) && !seen.has(name)) { seen.add(name); out.push({ name, path: normPath(raw) }); }
      }
      return out;
    }

    /** Tool names for the model: letters, digits, _ and -, at most 64 characters. */
    const safeToolName = (s) => String(s).replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 64);

    /** The id of a background job from the answer of the bash tool. */
    function parseJobId(text) {
      const s = String(text || '');
      const m = s.match(/\b(?:job|shell|process|task|id)\b[\s:=#"'(]*([A-Za-z0-9_-]+)/i) || s.match(/\b(bash_[A-Za-z0-9]+|[a-f0-9]{6,}|\d+)\b/);
      return m ? m[1] : null;
    }
    const MIN_COMPLETION = 200;
    /** Tokens the answer may use: the wanted size, cut to what is left of the context. 0 when less than MIN_COMPLETION remain. */
    function computeMaxTokens(maxContext, promptTokens, wanted = 2048) {
      const margin = 32 + Math.ceil(maxContext * 0.02);
      const room = Math.floor(maxContext - promptTokens - margin);
      const n = Math.min(wanted, room);
      return n >= MIN_COMPLETION ? n : 0;
    }

    /** Reads the numbers out of "maximum context length is 4096 tokens. However, you requested 4520 tokens (2472 in the prompt, 2048 in the completion)". */
    function parseContextError(message) {
      const s = String(message || '');
      const max = s.match(/maximum context length is (\d+)/i);
      const prompt = s.match(/(\d+) in the prompt/i);
      if (!max || !prompt) return null;
      return { max: Number(max[1]), prompt: Number(prompt[1]) };
    }

    /** max_tokens for the retry after a context error, or 0 when the agent cannot work in this context. */
    function retryMaxTokens(info, wanted = 2048) {
      return info ? computeMaxTokens(info.max, info.prompt, wanted) : 0;
    }

    const RARE_TOOLS = /^(web_search|notebook)/;
    /** Small contexts get one short sentence per tool, no per-parameter text, and no rarely needed tools. */
    function compactTool(tool) {
      const f = tool.function;
      const first = String(f.description || '').split(/(?<=[.!?])\s/)[0].slice(0, 90);
      const slim = (schema) => {
        if (Array.isArray(schema)) return schema.map(slim);
        if (!schema || typeof schema !== 'object') return schema;
        const o = {};
        for (const [k, v] of Object.entries(schema)) {
          if (k === 'description' && typeof v === 'string') continue;
          o[k] = k === 'properties' && v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([n, s]) => [n, slim(s)])) : slim(v);
        }
        return o;
      };
      return { type: 'function', function: { name: f.name, description: first, parameters: slim(f.parameters) } };
    }
    function compactTools(tools, keepRare) {
      return tools.filter((t) => keepRare || !RARE_TOOLS.test(t.function.name)).map(compactTool);
    }
    // @pure-end

    // ---- tools ----------------------------------------------------------------------------------------
    // Fallback list when the server does not offer /api/studio/agent/tools.
    const TOOL_DEFS = {
      list_dir: { permission: 'read', description: 'List the files and folders of a directory in the workspace.', properties: { path: { type: 'string', description: 'Folder relative to the workspace. Default ".".' } }, required: [] },
      read_file: { permission: 'read', description: 'Read a text file. The result has line numbers. Long files are read in parts with offset and limit.', properties: { path: { type: 'string' }, offset: { type: 'integer', description: 'First line to read, starting at 1.' }, limit: { type: 'integer', description: 'Number of lines to read.' } }, required: ['path'] },
      write_file: { permission: 'edit', description: 'Create a file or replace it completely. Prefer edit_file for changes to an existing file.', properties: { path: { type: 'string' }, content: { type: 'string', description: 'The complete new content of the file.' } }, required: ['path', 'content'] },
      edit_file: { permission: 'edit', description: 'Change a file by replacing text. old_string must match the file exactly (including indentation) and must occur once, unless replace_all is true.', properties: { path: { type: 'string' }, old_string: { type: 'string' }, new_string: { type: 'string' }, replace_all: { type: 'boolean' } }, required: ['path', 'old_string', 'new_string'] },
      glob: { permission: 'read', description: 'Find files by name pattern, for example "src/**/*.rs" or "*.md". Newest files first.', properties: { pattern: { type: 'string' } }, required: ['pattern'] },
      grep: { permission: 'read', description: 'Search the lines of files with a regular expression. Returns path:line: text.', properties: { pattern: { type: 'string' }, path: { type: 'string', description: 'File or folder to search. Default: the whole workspace.' }, glob: { type: 'string', description: 'Only search files whose name matches, for example "*.rs".' } }, required: ['pattern'] },
      bash: { permission: 'command', description: 'Run a shell command in the workspace folder and return its output. Use it to build, test, run programs and use git.', properties: { command: { type: 'string' }, timeout_ms: { type: 'integer', description: 'Stop the command after this many milliseconds. Default 120000.' } }, required: ['command'] },
    };
    // Tools that run in the page.
    const CLIENT_TOOLS = {
      todo_write: { description: 'Keep a task list for work with several steps. Send the complete list every time; exactly one task should be in_progress.', properties: { todos: { type: 'array', items: { type: 'object', properties: { content: { type: 'string' }, status: { type: 'string', enum: ['pending', 'in_progress', 'completed'] } }, required: ['content', 'status'] } } }, required: ['todos'] },
      spawn_agent: { description: 'Start a sub-agent with a fresh context for a self-contained job and get back only its final report. Several spawn_agent calls in one turn run in parallel. The sub-agent does not see this conversation, so the prompt must contain everything it needs.', properties: { description: { type: 'string', description: 'Three to six words naming the job.' }, prompt: { type: 'string', description: 'The complete task and what the report should contain.' }, mode: { type: 'string', enum: ['read-only', 'full'], description: 'read-only (default) can only read and search; full may also edit and run commands, with the usual permission questions.' } }, required: ['description', 'prompt'] },
      ask_user: { description: 'Ask the user a question with a few answer choices when a decision is theirs and you cannot find the answer yourself. The answer comes back as the result.', properties: { question: { type: 'string' }, options: { type: 'array', items: { type: 'string' }, description: 'Two to five short choices.' } }, required: ['question'] },
    };

    const code = {
      workspace: null, // { root, name, git_branch, os, shell }
      perm: localStorage.getItem('fllm.studio.perm') || 'ask',
      alwaysAllow: new Set(),
      messages: [], // what the model sees
      view: [], // what the user sees
      todos: [],
      changed: new Map(),
      changes: null, // from the server when it offers the changes route
      registry: new Map(), // name -> { name, def, permission, source: server|client|mcp, server, tool }
      custom: [], // custom slash commands { name, path, description }
      rules: { allow: '', deny: '' },
      rulesDraft: null,
      rulesOpen: false,
      jobs: [],
      sessionId: null,
      instructions: '',
      running: false,
      abort: null,
      used: 0,
      lastUsed: 0,
      compactedAt: 0,
      seq: 0,
      waiters: new Map(), // item id -> { resolve, cancel }
      noGrammar: false,
      calib: 1, // server-counted prompt tokens / our estimate of the last request
    };
    code.perm = ['ask', 'edit', 'plan'].includes(code.perm) ? code.perm : 'ask';

    const cid = () => `c${++code.seq}${Date.now().toString(36).slice(-3)}`;
    const ownerOf = new WeakMap(); // sub-agent items -> the spawn_agent item that is painted
    const ctxMax = () => state.status?.model?.max_context || 4096;

    function registryFromDefs() {
      const map = new Map();
      for (const [name, d] of Object.entries(TOOL_DEFS)) map.set(name, { name, permission: d.permission, source: 'server', def: { description: d.description, parameters: { type: 'object', properties: d.properties, required: d.required } } });
      return map;
    }

    function addClientTools(map) {
      for (const [name, d] of Object.entries(CLIENT_TOOLS)) map.set(name, { name, permission: 'read', source: 'client', def: { description: d.description, parameters: { type: 'object', properties: d.properties, required: d.required } } });
    }

    /** Server tools (fallback: the built-in list), then page tools, then MCP tools. Stable order keeps the prompt cache valid. */
    async function loadRegistry() {
      let map = null;
      try {
        const res = await apiFetch('/api/studio/agent/tools');
        if (res.ok) {
          const data = await res.json();
          const list = Array.isArray(data) ? data : (data.tools || []);
          const parsed = new Map();
          for (const t of list) {
            const f = t.function || t;
            if (!f?.name) continue;
            parsed.set(f.name, { name: f.name, permission: t.permission || f.permission || 'command', source: 'server', def: { description: f.description || '', parameters: f.parameters || { type: 'object', properties: {} } } });
          }
          if (parsed.size) map = parsed;
        }
      } catch { /* the route is missing: use the built-in list */ }
      map = map || registryFromDefs();
      addClientTools(map);
      try {
        const res = await apiFetch('/api/studio/mcp/tools');
        if (res.ok) {
          const data = await res.json();
          for (const t of data.tools || []) {
            const name = safeToolName(`mcp__${t.server}__${t.name}`);
            map.set(name, { name, permission: 'command', source: 'mcp', server: t.server, tool: t.name, def: { description: `[${t.server}] ${t.description || ''}`.trim(), parameters: t.input_schema || { type: 'object', properties: {} } } });
          }
        }
      } catch { /* no MCP client: no MCP tools */ }
      code.registry = map;
    }

    const readOnlyTool = (entry) => entry.permission === 'read' && entry.source !== 'client';
    const toolListFor = (plan) => [...code.registry.values()]
      .filter((e) => !plan || readOnlyTool(e) || e.source === 'client')
      .map((e) => ({ type: 'function', function: { name: e.name, description: e.def.description, parameters: e.def.parameters } }));
    /** Contexts of 8192 tokens or less get the compact prompt and tool forms. */
    const compactMode = () => ctxMax() <= 8192;
    const toolList = () => {
      const full = toolListFor(code.perm === 'plan');
      return compactMode() ? compactTools(full, ctxMax() >= 16384 || localStorage.getItem('fllm.studio.codefulltools') === '1') : full;
    };
    const fixedTokens = (ctx) => estimateTokens(ctx.system) + estimateTokens(JSON.stringify(ctx.tools || []));
    const serverHas = (name) => code.registry.get(name)?.source === 'server';

    function codeSystemPrompt() {
      const w = code.workspace || {};
      if (compactMode()) {
        const c = [
          'You are fLLM Code, a coding agent on the user\'s computer. Use the tools to inspect and change files in the workspace and to run commands.',
          '- Read before you change; never guess file contents. Prefer edit_file (exact, unique old_string); write_file only for new files.',
          '- Build or test after changes and fix failures. Keep a todo_write list for 3+ steps.',
          '- Delegate broad searches or independent investigations with spawn_agent (self-contained prompt; several in one turn run in parallel). Do small jobs yourself.',
          '- Ask with ask_user when a decision is the user\'s. Keep answers short, in the user\'s language. Do not delete files or send data elsewhere unless asked.',
        ];
        if (code.perm === 'plan') c.push('PLAN MODE: read only. Investigate, then write a short numbered plan and stop.');
        c.push(`Workspace: ${w.root || '(none)'} (${w.os || 'unknown'}, ${w.shell || 'shell'})${w.git_branch ? `, branch ${w.git_branch}` : ''}`, `Date: ${new Date().toISOString().slice(0, 10)}`);
        if (code.instructions) c.push('Project instructions:', code.instructions.slice(0, 1200));
        return c.join('\n');
      }
      const lines = [
        'You are fLLM Code, a coding agent that runs on the user\'s own computer. You work inside the workspace folder shown below and use tools to look at files, change them and run commands.',
        '',
        'How to work:',
        '- Look before you change anything: use list_dir, glob and grep to find files, read_file to read them. Never guess what a file contains.',
        '- Change files with edit_file (old_string must match exactly and be unique), or multi_edit for several changes in one file when it is available. Use write_file only for new files or complete rewrites. Make the smallest change that solves the task and keep the style of the code around it.',
        '- After changing code, build or test it with bash when the project has a build or tests, and fix what fails. Long-running programs (servers, watchers) belong in the background when bash supports it; read their output with bash_output.',
        '- For work with three or more steps keep a task list with todo_write.',
        '- When a decision is the user\'s (which of two designs, whether to delete something) and you cannot find out yourself, ask with ask_user instead of guessing.',
        '- Keep answers short. Say what you did and what the user should know. Answer in the language the user writes in.',
        '- Do not run commands that delete files, change the system or send data to other computers unless the user asked for it.',
        '',
        'Sub-agents (spawn_agent):',
        '- Delegate broad searches ("where is X handled across the code base"), reading many files to answer one question, and independent investigations. This keeps your own context small. When several jobs are independent, start them in the same turn: they run in parallel.',
        '- Do NOT delegate small jobs (one to three tool calls), reading a file you already know, steps that depend on the result of the previous step, or edits that need your judgement. Do those yourself.',
        '- A sub-agent starts with an empty context. Write a self-contained prompt: the goal, where to look, what the report must contain (file paths, line numbers, short findings). Use mode "read-only" unless the job really must change files.',
        '- Only its final report comes back. Check important claims in it before you rely on them.',
      ];
      if (code.perm === 'plan') lines.push('', 'PLAN MODE: You may only read. Do not change anything. Investigate, then write a short numbered plan and stop. The user will tell you when to carry it out.');
      lines.push('', `Workspace: ${w.root || '(none)'}`, `System: ${w.os || 'unknown'}, commands run with ${w.shell || 'the system shell'}`);
      if (w.git_branch) lines.push(`Git branch: ${w.git_branch}`);
      lines.push(`Date: ${new Date().toISOString().slice(0, 10)}`);
      if (code.instructions) lines.push('', 'Instructions of this project:', code.instructions);
      return lines.join('\n');
    }

    // ---- workspace ------------------------------------------------------------------
    const recentWorkspaces = () => { try { return JSON.parse(localStorage.getItem(CODE_RECENT_KEY) || '[]'); } catch { return []; } };

    async function setWorkspace(path) {
      const res = await apiFetch('/api/studio/agent/workspace', { method: 'POST', body: { path } });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { toast(data.error?.message || 'The folder could not be opened.'); return false; }
      adoptWorkspace(data);
      const recent = [data.root, ...recentWorkspaces().filter((p) => p !== data.root)].slice(0, 6);
      localStorage.setItem(CODE_RECENT_KEY, JSON.stringify(recent));
      await onWorkspaceOpened();
      renderCode();
      return true;
    }

    /** Everything that belongs to the opened folder: instructions, tools, commands, rules, last session. */
    async function onWorkspaceOpened() {
      await Promise.all([loadInstructions(), loadRegistry(), loadCustomCommands()]);
      loadRules();
      restoreCodeSession();
      refreshChanges();
    }

    function adoptWorkspace(data) {
      code.workspace = data.root ? data : null;
      if (data.root) localStorage.setItem(CODE_RECENT_KEY, JSON.stringify([data.root, ...recentWorkspaces().filter((p) => p !== data.root)].slice(0, 6)));
      $('wsName').textContent = data.root ? data.name : 'Choose folder';
      const branch = $('wsBranch');
      branch.style.display = data.git_branch ? 'inline' : 'none';
      branch.textContent = data.git_branch || '';
    }

    async function loadWorkspace() {
      try {
        const res = await apiFetch('/api/studio/agent/workspace', { method: 'POST', body: {} });
        if (res.ok) {
          const data = await res.json();
          adoptWorkspace(data);
          code.disabled = null;
          if (data.root) await onWorkspaceOpened();
        } else {
          code.disabled = (await res.json().catch(() => ({}))).error?.message || 'The coding agent is not available.';
        }
        renderCode();
      } catch { /* the page is opened before the server is ready: the next visit asks again */ }
    }

    async function toolCall(name, args, signal) {
      const res = await apiFetch('/api/studio/agent/tool', { method: 'POST', body: { name, args }, signal });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        return { ok: false, error: err.error?.message || `Error ${res.status}` };
      }
      return res.json();
    }

    async function mcpCall(entry, args, signal) {
      const res = await apiFetch('/api/studio/mcp/call', { method: 'POST', body: { server: entry.server, name: entry.tool, arguments: args }, signal });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        return { ok: false, error: err.error?.message || `Error ${res.status}` };
      }
      const data = await res.json();
      return data.is_error ? { ok: false, error: String(data.content || 'The tool failed.') } : { ok: true, output: String(data.content ?? '') };
    }

    async function loadInstructions() {
      code.instructions = '';
      for (const file of INSTRUCTION_FILES) {
        const r = await toolCall('read_file', { path: file, limit: 200 }).catch(() => null);
        if (r?.ok) { code.instructions = `(${file})\n${r.output.replace(/^\s*\d+\t/gm, '').slice(0, 6000)}`; break; }
      }
    }

    async function pickWorkspace() {
      const recent = recentWorkspaces();
      const where = await askDialog({
        title: 'Choose the workspace folder',
        text: `The agent may read, change and run commands only inside this folder. Enter the full path, for example ${code.workspace?.os === 'windows' || /Windows/.test(navigator.userAgent) ? 'C:\\Users\\you\\project' : '/home/you/project'}.${recent.length ? ` Last used: ${recent[0]}` : ''}`,
        input: code.workspace?.root || recent[0] || '',
        ok: 'Open',
      });
      if (where) await setWorkspace(where);
    }

    /** The rail shows the folders the agent has worked in while the Code tab is open. */
    function renderWorkspaceList() {
      const recent = recentWorkspaces();
      const here = code.workspace?.root;
      const rows = recent.map((p) => `
        <div class="thread-row ${p === here && state.surface === 'code' ? 'active' : ''}" data-ws-row="${escapeHtml(p)}" title="${escapeHtml(p)}">
          <span class="thread-row-title">${escapeHtml(p.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || p)}</span>
        </div>`).join('');
      $('threadList').innerHTML = `${rows || '<div class="thread-empty">No folder opened yet</div>'}
        <div class="thread-row" data-ws-open><span class="thread-row-title">Open folder ...</span></div>`;
    }
    $('threadList').addEventListener('click', async (e) => {
      if (state.tab !== 'code') return;
      const row = e.target.closest('[data-ws-row]');
      if (row) { switchSurface('code'); await setWorkspace(row.dataset.wsRow); return; }
      if (e.target.closest('[data-ws-open]')) { switchSurface('code'); await pickWorkspace(); }
    });

    // ---- custom slash commands (.fllm/commands/*.md) ---------------------------------------------
    async function loadCustomCommands() {
      code.custom = [];
      const paths = [];
      for (const pattern of ['.fllm/commands/*.md', '.fllm/commands/**/*.md']) {
        const r = await toolCall('glob', { pattern }).catch(() => null);
        if (r?.ok) paths.push(...r.output.split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('...') && l !== 'no files match'));
      }
      code.custom = commandNamesFromPaths(paths).map((c) => ({ ...c, description: '' }));
      for (const c of code.custom) {
        const r = await toolCall('read_file', { path: c.path, limit: 4 }).catch(() => null);
        if (r?.ok) c.description = parseCommandFile(r.output.replace(/^\s*\d+\t/gm, '')).description;
      }
    }

    async function readCustomCommand(c) {
      const r = await toolCall('read_file', { path: c.path, limit: 2000 });
      if (!r?.ok) throw new Error(r?.error || 'The command file could not be read.');
      return parseCommandFile(r.output.replace(/^\s*\d+\t/gm, '')).body;
    }

    // ---- permission rules ------------------------------------------------------------------------
    function loadRules() {
      try { code.rules = { allow: '', deny: '', ...(JSON.parse(localStorage.getItem(CODE_RULES_KEY) || '{}')[code.workspace.root] || {}) }; } catch { code.rules = { allow: '', deny: '' }; }
      code.rulesDraft = null;
    }
    function saveRules(allow, deny) {
      code.rules = { allow, deny };
      code.rulesDraft = null;
      try {
        const all = JSON.parse(localStorage.getItem(CODE_RULES_KEY) || '{}');
        all[code.workspace.root] = code.rules;
        localStorage.setItem(CODE_RULES_KEY, JSON.stringify(all));
      } catch { /* browser storage is full: the rules only last while the page is open */ }
    }
    const parsedRules = () => ({ allow: parseRules(code.rules.allow), deny: parseRules(code.rules.deny) });

    // ---- saving sessions --------------------------------------------------------------------------
    const clipForSave = (items) => items.map((i) => {
      const c = { ...i };
      if (typeof c.output === 'string' && c.output.length > 20000) c.output = `${c.output.slice(0, 20000)}\n... [cut]`;
      if (c.children) c.children = clipForSave(c.children);
      return c;
    });
    const sessionTitle = (messages) => {
      const first = messages.find((m) => m.role === 'user' && !String(m.content).startsWith('[Summary'));
      return String(first?.content || 'Session').replace(/\s+/g, ' ').slice(0, 60);
    };
    function readSessions() {
      try { return JSON.parse(localStorage.getItem(CODE_SESSIONS_KEY) || '{}'); } catch { return {}; }
    }
    const workspaceSessions = () => (code.workspace ? readSessions()[code.workspace.root] || [] : []);

    function saveCodeSession() {
      if (!code.workspace || (!code.messages.length && !code.view.length)) return;
      try {
        const all = readSessions();
        if (!code.sessionId) code.sessionId = cid();
        const mine = (all[code.workspace.root] || []).filter((s) => s.id !== code.sessionId);
        const view = clipForSave(code.view.filter((i) => !(i.kind === 'notice' && i.choices)));
        mine.unshift({ id: code.sessionId, title: sessionTitle(code.messages), time: Date.now(), messages: code.messages, view, todos: code.todos, used: code.used });
        all[code.workspace.root] = mine.slice(0, 15);
        let text = JSON.stringify(all);
        while (text.length > 4_000_000 && all[code.workspace.root].length > 1) {
          all[code.workspace.root].pop();
          text = JSON.stringify(all);
        }
        if (text.length <= 4_500_000) localStorage.setItem(CODE_SESSIONS_KEY, text);
      } catch { /* the browser storage is full: the session is only kept while the page is open */ }
    }

    const heal = (items) => items.map((i) => {
      const c = (i.status === 'running' || i.status === 'pending') ? { ...i, status: 'error', output: 'Interrupted.' } : { ...i, streaming: false };
      if (c.children) c.children = heal(c.children);
      return c;
    });

    function applySession(s) {
      code.sessionId = s.id;
      code.messages = s.messages || [];
      code.view = heal(s.view || []);
      code.todos = s.todos || [];
      code.used = s.used || 0;
      code.lastUsed = 0;
      code.compactedAt = 0;
      code.changed = new Map();
      code.jobs = [];
      reindexOwners(code.view);
    }

    function reindexOwners(items) {
      for (const i of items) for (const c of i.children || []) { ownerOf.set(c, i); reindexOwners([c]); }
    }

    function restoreCodeSession() {
      code.messages = []; code.view = []; code.todos = []; code.changed = new Map(); code.used = 0; code.lastUsed = 0; code.sessionId = null; code.jobs = []; code.alwaysAllow = new Set();
      try {
        let s = workspaceSessions()[0];
        if (!s) {
          const old = JSON.parse(localStorage.getItem(CODE_SESSION_KEY) || '{}')[code.workspace.root];
          if (old) s = { id: cid(), ...old };
        }
        if (s) applySession(s);
      } catch { /* an unreadable saved session is ignored */ }
    }

    function resumeSession(id) {
      if (code.running) stopAgent();
      const s = workspaceSessions().find((x) => x.id === id);
      if (!s) { toast('That session is gone.'); return; }
      saveCodeSession();
      applySession(s);
      renderCode();
    }

    // ---- changes (rewind) -----------------------------------------------------------------------------
    async function refreshChanges() {
      try {
        const res = await apiFetch('/api/studio/agent/changes');
        if (!res.ok) { code.changes = null; return; }
        const data = await res.json();
        code.changes = Array.isArray(data.changes) ? data.changes : [];
        renderCodeSide();
      } catch { code.changes = null; }
    }

    async function undoChange(steps) {
      const body = steps && steps > 1 ? { steps } : {};
      const r = await apiFetch('/api/studio/agent/undo', { method: 'POST', body }).then((x) => x.json()).catch((e) => ({ ok: false, error: e.message }));
      codeNotice(r.ok ? `Undone: ${r.output}` : (r.error || r.error?.message || 'Nothing to undo.'));
      refreshChanges();
    }

    async function showRewind() {
      await refreshChanges();
      if (code.changes === null) { codeNotice('Rewind is not available on this server. /undo reverts the last file change.'); return; }
      if (!code.changes.length) { codeNotice('No file changes to rewind.'); return; }
      const rows = [...code.changes].reverse().slice(0, 15).map((c) => ({ label: `${c.kind || 'change'} ${c.path}${c.time ? `  (${new Date(c.time > 1e12 ? c.time : c.time * 1000).toLocaleTimeString()})` : ''}`, data: c.id }));
      pushChoices('Rewind: pick the oldest change to take back. It and every later change are undone.', 'rewind', rows);
    }

    async function rewindTo(id) {
      const r = await apiFetch('/api/studio/agent/undo', { method: 'POST', body: { to: id } }).then((x) => x.json()).catch((e) => ({ ok: false, error: e.message }));
      codeNotice(r.ok === false ? (r.error?.message || r.error || 'The rewind failed.') : `Rewound: ${r.output || 'done'}`);
      refreshChanges();
    }

    // ---- rendering ---------------------------------------------------------------------------
    const TOOL_LABEL = {
      list_dir: (a) => ['List folder', a.path || '.'],
      read_file: (a) => ['Read file', `${a.path || ''}${a.offset ? ` from line ${a.offset}` : ''}`],
      write_file: (a) => ['Write file', a.path || ''],
      edit_file: (a) => ['Edit file', a.path || ''],
      multi_edit: (a) => ['Edit file', `${a.path || ''}${Array.isArray(a.edits) ? `  (${a.edits.length} changes)` : ''}`],
      glob: (a) => ['Find files', a.pattern || ''],
      grep: (a) => ['Search text', `${a.pattern || ''}${a.glob ? `  in ${a.glob}` : ''}`],
      bash: (a) => [a.background ? 'Background command' : 'Command', a.command || ''],
      bash_output: (a) => ['Job output', Object.values(a)[0] ?? ''],
      bash_kill: (a) => ['Stop job', Object.values(a)[0] ?? ''],
      web_fetch: (a) => ['Fetch page', a.url || ''],
      web_search: (a) => ['Web search', a.query || ''],
      todo_write: () => ['Task list', ''],
      ask_user: (a) => ['Question', a.question || ''],
      spawn_agent: (a) => ['Sub-agent', a.description || ''],
    };
    const EDIT_TOOLS = new Set(['write_file', 'edit_file', 'multi_edit']);
    const STATUS_ICON = {
      pending: '<svg viewBox="0 0 24 24" class="icon icon-sm"><circle cx="12" cy="12" r="9"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>',
      running: '<span class="spin"></span>',
      ok: '<svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M20 6 9 17l-5-5"/></svg>',
      error: '<svg viewBox="0 0 24 24" class="icon icon-sm"><circle cx="12" cy="12" r="9"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>',
      denied: '<svg viewBox="0 0 24 24" class="icon icon-sm"><circle cx="12" cy="12" r="9"/><path d="m5 5 14 14"/></svg>',
    };
    const STATUS_TEXT = { pending: 'waiting for you', running: 'running', ok: 'done', error: 'error', denied: 'denied' };
    const OUT_LINES = 40;
    const DIFF_LINES = 60;

    function diffRows(item) {
      const a = item.args || {};
      const rows = [];
      const pair = (o, n) => {
        String(o || '').split('\n').forEach((l) => rows.push(['del', l]));
        String(n || '').split('\n').forEach((l) => rows.push(['add', l]));
      };
      if (item.name === 'edit_file') pair(a.old_string, a.new_string);
      else if (item.name === 'multi_edit' && Array.isArray(a.edits)) a.edits.forEach((e, i) => { if (i) rows.push(['more', '···']); pair(e?.old_string, e?.new_string); });
      else if (item.name === 'write_file') String(a.content || '').split('\n').forEach((l) => rows.push(['add', l]));
      return rows;
    }

    function diffHtml(item) {
      const rows = diffRows(item);
      if (!rows.length) return '';
      const shown = item.expanded ? rows : rows.slice(0, DIFF_LINES);
      const more = rows.length - shown.length;
      const html = shown.map(([k, l]) => `<div class="diff-${k}"><span class="diff-mark">${k === 'add' ? '+' : k === 'del' ? '−' : ''}</span><span>${escapeHtml(l) || ' '}</span></div>`).join('');
      return `<div class="diff">${html}${more > 0 ? `<div class="diff-more"><button class="btn btn-xs btn-ghost" data-expand="${item.id}">Show ${more} more lines</button></div>` : ''}</div>`;
    }

    function outputHtml(item) {
      const text = String(item.output);
      const lines = text.split('\n');
      let shown = text;
      let hidden = 0;
      if (!item.expanded && (lines.length > OUT_LINES || text.length > 6000)) {
        shown = lines.slice(0, OUT_LINES).join('\n').slice(0, 6000);
        hidden = lines.length - shown.split('\n').length;
      } else if (text.length > 60000) shown = `${text.slice(0, 60000)}\n... [cut]`;
      const more = shown.length < text.length && !item.expanded ? `<button class="btn btn-xs btn-ghost tool-more" data-expand="${item.id}">Show all${hidden > 0 ? ` (${hidden} more lines)` : ''}</button>` : '';
      return `<pre class="tool-out">${escapeHtml(shown)}</pre>${more}`;
    }

    const toolLabel = (item) => {
      const fn = TOOL_LABEL[item.name];
      if (fn) return fn(item.args || {});
      const entry = code.registry.get(item.name);
      if (entry?.source === 'mcp') return [entry.tool, entry.server];
      const first = Object.values(item.args || {}).find((v) => typeof v === 'string') || '';
      return [item.name, String(first).slice(0, 80)];
    };

    const defaultOpen = (item) => item.status === 'pending' || EDIT_TOOLS.has(item.name) || item.name === 'ask_user'
      || (item.name === 'bash' && item.output && item.status === 'error') || (item.status === 'running' && item.name === 'bash') || item.name === 'spawn_agent' && item.status === 'running';

    function toolHtml(item) {
      const [verb, target] = toolLabel(item);
      let body = '';
      if (item.name === 'bash') body += `<pre class="tool-cmd">$ ${escapeHtml(item.args?.command || '')}</pre>`;
      body += diffHtml(item);
      if (item.name === 'todo_write' && Array.isArray(item.args?.todos)) body += todoListHtml(item.args.todos);
      else if (item.name === 'spawn_agent') {
        body += `<pre class="tool-cmd sub-prompt">${escapeHtml(String(item.args?.prompt || '').slice(0, 1500))}</pre>`;
        body += `<div class="tool-children">${(item.children || []).map(itemHtml).join('') || '<div class="side-empty">Starting ...</div>'}</div>`;
        if (item.output) body += `<div class="sub-report"><h5>Report</h5>${MD.render(String(item.output))}</div>`;
      } else if (item.name === 'ask_user') {
        if (item.output) body += `<pre class="tool-out">${escapeHtml(item.output)}</pre>`;
      } else if (item.output) body += outputHtml(item);
      const hasDetail = body || item.status === 'pending';
      const open = (item.open !== undefined ? item.open : defaultOpen(item)) ? ' open' : '';
      const ask = item.status === 'pending' ? (item.name === 'ask_user' ? askUserHtml(item) : permissionHtml(item)) : '';
      const progress = item.name === 'spawn_agent' && item.status === 'running' && item.progress ? `<span class="tool-progress">${escapeHtml(item.progress)}</span>` : '';
      return `
        <div class="tool-card is-${item.status}${item.name === 'spawn_agent' ? ' is-agent' : ''}" data-item="${item.id}">
          <details${open}>
            <summary>
              <span class="tool-status" title="${STATUS_TEXT[item.status] || ''}">${STATUS_ICON[item.status] || ''}</span>
              <span class="tool-verb">${escapeHtml(verb)}</span>
              <span class="tool-target mono">${escapeHtml(target)}</span>${progress}
            </summary>
            ${hasDetail ? `<div class="tool-body">${body}</div>` : ''}
          </details>
          ${ask}
        </div>`;
    }

    function permissionInfo(entry, args) {
      const cat = entry.source === 'mcp' ? 'mcp' : entry.permission === 'edit' ? 'edit' : entry.permission === 'network' ? 'network' : entry.permission === 'command' ? 'bash' : 'read';
      let subject = '';
      if (cat === 'bash') subject = typeof args.command === 'string' ? args.command : `${entry.name} ${Object.values(args).join(' ')}`;
      else if (cat === 'edit' || cat === 'read') subject = String(args.path ?? args.pattern ?? '');
      else if (cat === 'network') subject = String(args.url ?? args.query ?? '');
      else subject = `${entry.server}/${entry.tool}`;
      return { cat, subject };
    }

    /** What "Always" would remember for this call, or null when it can only be allowed once. */
    function alwaysKey(item) {
      const entry = code.registry.get(item.name);
      if (!entry) return null;
      const { cat, subject } = permissionInfo(entry, item.args || {});
      if (cat === 'edit') return 'edit';
      if (cat === 'bash') { const k = entry.name === 'bash' ? commandKey(subject) : null; return k ? `bash:${k}` : null; }
      if (cat === 'network') return 'network';
      if (cat === 'mcp') return `mcp:${subject}`;
      return null;
    }

    function permissionHtml(item) {
      const key = alwaysKey(item);
      const entry = code.registry.get(item.name);
      const cat = entry ? permissionInfo(entry, item.args || {}).cat : 'bash';
      const label = { bash: 'Run this command?', network: 'Allow this network request?', mcp: 'Call this external tool?' }[cat] || (item.name === 'write_file' ? 'Write this file?' : 'Apply this change?');
      const tip = cat === 'edit' ? 'Allow file changes in this session without asking' : cat === 'bash' ? `Allow more "${(key || '').slice(5)}" commands in this session without asking` : 'Allow this in this session without asking';
      return `<div class="perm-ask"><span>${label}</span>
        <button class="btn btn-sm btn-primary" data-perm-answer="allow">Allow</button>
        ${key ? `<button class="btn btn-sm btn-secondary" data-perm-answer="always" title="${escapeHtml(tip)}">Always${cat === 'bash' ? ` (${escapeHtml(key.slice(5))})` : ''}</button>` : ''}
        <button class="btn btn-sm btn-ghost" data-perm-answer="deny">Deny</button></div>`;
    }

    function askUserHtml(item) {
      const opts = Array.isArray(item.args?.options) ? item.args.options.map(String).slice(0, 6) : [];
      return `<div class="perm-ask ask-user"><span>${escapeHtml(item.args?.question || '')}</span>
        <div class="ask-options">${opts.map((o, i) => `<button class="btn btn-sm btn-secondary" data-ask-opt="${i}">${escapeHtml(o)}</button>`).join('')}</div>
        <div class="ask-free"><input type="text" class="ui-input" data-ask-text placeholder="Or write an answer"><button class="btn btn-sm btn-primary" data-ask-send>Send</button></div></div>`;
    }

    function todoListHtml(todos) {
      const mark = { completed: '✓', in_progress: '●', pending: '○' };
      return `<ul class="todo-list">${todos.map((t) => `<li class="todo-${t.status}"><span>${mark[t.status] || '○'}</span> ${escapeHtml(t.content)}</li>`).join('')}</ul>`;
    }

    function itemHtml(item) {
      if (item.kind === 'user') return `<div class="code-user" data-item="${item.id}"><div class="msg-user-bubble">${escapeHtml(item.text)}</div></div>`;
      if (item.kind === 'assistant') {
        const parsed = splitReasoning(item.content || '');
        const reasoning = item.reasoning || parsed.reasoning;
        const answer = item.reasoning ? item.content : parsed.answer;
        const thinking = reasoning ? `<details class="cot-box" data-k="cot"${item.cotOpen ? ' open' : ''}><summary>${svgIcon('brain')}<span>${item.streaming && !answer ? 'Thinking' : 'Thought process'}</span></summary><div class="cot-body">${escapeHtml(reasoning)}</div></details>` : '';
        const body = answer ? MD.render(answer) : (item.streaming && !reasoning ? '<span class="typing"><i></i><i></i><i></i></span>' : '');
        const plan = item.planDone ? '<div class="plan-actions"><button class="btn btn-sm btn-primary" data-run-plan>Run the plan</button><span>The agent keeps asking before every change.</span></div>' : '';
        return `<div class="code-assistant" data-item="${item.id}">${thinking}<div class="msg-assistant-prose">${body}</div>${item.error ? `<div class="msg-error">${escapeHtml(item.error)}</div>` : ''}${plan}</div>`;
      }
      if (item.kind === 'tool') return toolHtml(item);
      const choices = item.choices ? `<div class="notice-choices">${item.choices.map((c, i) => `<button class="btn btn-sm btn-secondary" data-choice="${item.id}:${i}">${escapeHtml(c.label)}</button>`).join('')}</div>` : '';
      return `<div class="code-notice${item.choices ? ' has-choices' : ''}" data-item="${item.id}">${escapeHtml(item.text)}${choices}</div>`;
    }

    function emptyCodeHtml() {
      if (code.disabled) return `<div class="code-empty"><h1>Code</h1><p>${escapeHtml(code.disabled)}</p></div>`;
      if (!code.workspace) {
        const recent = recentWorkspaces();
        return `<div class="code-empty">
          <h1>Code</h1>
          <p>An agent that reads, searches and changes files and runs commands inside one folder of this computer. You see every step and confirm changes.</p>
          <div class="code-open"><input type="text" class="ui-input mono" id="wsPathInput" placeholder="Folder path, e.g. ${/Windows/.test(navigator.userAgent) ? 'C:\\Users\\you\\project' : '/home/you/project'}" value="${escapeHtml(recent[0] || '')}"><button class="btn btn-primary" id="wsOpenBtn">Open folder</button></div>
          ${recent.length > 1 ? `<div class="code-recent">${recent.map((r) => `<button class="starter-pill" data-ws="${escapeHtml(r)}">${escapeHtml(r)}</button>`).join('')}</div>` : ''}
          <p class="code-fine">The agent can only work inside this folder. It can only be reached from this computer, even when the server has an API key.</p>
        </div>`;
      }
      const modes = { ask: 'asks before every change and every command', edit: 'changes files on its own and asks before commands', plan: 'only reads and makes a plan' };
      const past = workspaceSessions().filter((s) => s.id !== code.sessionId).slice(0, 3);
      return `<div class="code-empty">
        <h1>${escapeHtml(code.workspace.name)}</h1>
        <p>${escapeHtml(code.workspace.root)}${code.workspace.git_branch ? `, branch ${escapeHtml(code.workspace.git_branch)}` : ''}</p>
        <p>The agent ${modes[code.perm]}. Describe the task below. Type / for commands.</p>
        <div class="code-recent">
          <button class="starter-pill" data-code-prompt="Explain the structure of this project: the most important folders and files and how they fit together.">Explain the project</button>
          <button class="starter-pill" data-code-prompt="Find all TODO and FIXME comments and summarize them by file.">Collect the TODOs</button>
          <button class="starter-pill" data-code-prompt="Find out how this project is built and tested, and run the tests.">Run the tests</button>
          ${code.instructions ? '' : '<button class="starter-pill" data-code-command="/init">Create FLLM.md</button>'}
        </div>
        ${past.length ? `<p class="code-fine">Earlier sessions:</p><div class="code-recent">${past.map((s) => `<button class="starter-pill" data-resume="${s.id}" title="${escapeHtml(new Date(s.time).toLocaleString())}">${escapeHtml(s.title)}</button>`).join('')}</div>` : ''}
        ${code.instructions ? `<p class="code-fine">Project instructions read: ${escapeHtml(code.instructions.split('\n')[0].replace(/[()]/g, ''))}</p>` : ''}
      </div>`;
    }

    function renderCode() {
      $('codeColumn').innerHTML = code.view.length ? code.view.map(itemHtml).join('') : emptyCodeHtml();
      if (!code.view.length && !code.workspace) setTimeout(() => $('wsPathInput')?.focus(), 30);
      renderCodeSide();
      if (state.tab === 'code') renderWorkspaceList();
      updateCodeMeter();
      $('permSeg').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.perm === code.perm));
      $('codeScroll').scrollTop = $('codeScroll').scrollHeight;
    }

    let codeStick = true;
    $('codeScroll').addEventListener('scroll', () => {
      const sc = $('codeScroll');
      codeStick = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 100;
    });

    /** Rewrite one item (or add it) without touching the rest. Items of sub-agents repaint their spawn_agent card. */
    function paintItem(item) {
      const top = ownerOf.get(item) || item;
      const column = $('codeColumn');
      if (!code.view.length || column.querySelector('.code-empty')) { renderCode(); return; }
      const html = itemHtml(top);
      const existing = column.querySelector(`[data-item="${top.id}"]`);
      if (existing) existing.outerHTML = html;
      else column.insertAdjacentHTML('beforeend', html);
      if (codeStick) $('codeScroll').scrollTop = $('codeScroll').scrollHeight;
    }

    const paintQueue = new Set();
    /** Streaming text and parallel sub-agents repaint at most once per frame. */
    function paintSoon(item) {
      const top = ownerOf.get(item) || item;
      if (paintQueue.has(top)) return;
      paintQueue.add(top);
      requestAnimationFrame(() => { paintQueue.delete(top); paintItem(top); });
    }

    function findItem(id, items = code.view) {
      for (const i of items) {
        if (i.id === id) return i;
        const f = i.children && findItem(id, i.children);
        if (f) return f;
      }
      return null;
    }

    function jobsHtml() {
      if (!code.jobs.length) return '<p class="side-empty">Commands started in the background show up here.</p>';
      return code.jobs.map((j) => `<div class="job" data-job="${escapeHtml(j.id)}">
        <div class="job-head"><span class="job-dot job-${j.status}"></span><span class="mono job-cmd" title="${escapeHtml(j.command)}">${escapeHtml(j.command)}</span></div>
        <div class="job-actions"><button class="btn btn-xs btn-ghost" data-job-out="${escapeHtml(j.id)}">Output</button>${j.status === 'running' ? `<button class="btn btn-xs btn-ghost" data-job-kill="${escapeHtml(j.id)}">Stop</button>` : ''}<span class="side-count">${escapeHtml(j.status)}</span></div>
        ${j.open ? `<pre class="job-out">${escapeHtml(j.out || '(no output yet)')}</pre>` : ''}</div>`).join('');
    }

    function renderCodeSide() {
      const side = $('codeSide');
      const done = code.todos.filter((t) => t.status === 'completed').length;
      const files = code.changes ? code.changes.map((c) => [c.path, c.kind]) : [...code.changed.entries()].map(([p, n]) => [p, n > 1 ? `${n}×` : '']);
      const uniqueFiles = [...new Map(files.map(([p, k]) => [p, k])).entries()];
      const sessions = workspaceSessions().slice(0, 6);
      const draft = code.rulesDraft || code.rules;
      const nRules = parseRules(code.rules.allow).length + parseRules(code.rules.deny).length;
      side.innerHTML = `
        <section class="side-block"><h4>Tasks${code.todos.length ? `<span class="side-count">${done}/${code.todos.length}</span>` : ''}</h4>
          ${code.todos.length ? todoListHtml(code.todos) : '<p class="side-empty">For bigger jobs the agent keeps a task list here.</p>'}</section>
        <section class="side-block"><h4>Changed files${uniqueFiles.length ? `<span class="side-count">${uniqueFiles.length}</span>` : ''}</h4>
          ${uniqueFiles.length ? `<ul class="changed-list">${uniqueFiles.slice(-30).map(([p, k]) => `<li><span class="mono">${escapeHtml(p)}</span>${k ? `<span class="side-count">${escapeHtml(k)}</span>` : ''}</li>`).join('')}</ul>${code.changes ? '<button class="btn btn-xs btn-ghost" data-code-command="/rewind">Rewind ...</button>' : ''}` : '<p class="side-empty">Nothing changed yet.</p>'}</section>
        <section class="side-block"><h4>Background jobs${code.jobs.length ? `<span class="side-count">${code.jobs.length}</span>` : ''}</h4>${jobsHtml()}</section>
        <section class="side-block"><h4>Sessions</h4>
          ${sessions.length ? `<ul class="session-list">${sessions.map((s) => `<li class="${s.id === code.sessionId ? 'active' : ''}"><button data-resume="${s.id}" title="${escapeHtml(new Date(s.time).toLocaleString())}">${escapeHtml(s.title)}</button></li>`).join('')}</ul>` : '<p class="side-empty">Sessions of this folder are saved automatically.</p>'}</section>
        <section class="side-block"><details class="rules-box" data-k="rules"${code.rulesOpen ? ' open' : ''}><summary><h4>Permission rules${nRules ? `<span class="side-count">${nRules}</span>` : ''}</h4></summary>
          <p class="side-empty">One rule per line: <span class="mono">category: pattern</span>. Categories: bash, edit, network, mcp, read. Deny beats allow. Risky programs need a rule that names them, for example <span class="mono">bash: git status*</span>.</p>
          <label class="rules-label">Always allow</label><textarea class="ui-textarea mono rules-text" data-rules="allow" rows="4" placeholder="bash: npm test*&#10;edit: src/**&#10;network: *">${escapeHtml(draft.allow)}</textarea>
          <label class="rules-label">Always deny</label><textarea class="ui-textarea mono rules-text" data-rules="deny" rows="3" placeholder="bash: rm *&#10;edit: .env">${escapeHtml(draft.deny)}</textarea>
          <button class="btn btn-sm btn-secondary" data-rules-save>Save rules</button></details></section>`;
    }

    function updateCodeMeter() {
      setMeter('codeCtxFill', 'codeCtxText', code.used || codeEstimate(), ctxMax());
    }
    const estimateMessages = (messages) => messages.reduce((s, m) => s + estimateTokens(m.content) + estimateTokens(JSON.stringify(m.tool_calls || '')) + 6, 0);
    // system prompt + tool list + messages, scaled by how far the last estimate was off from the server's own count
    const codeEstimate = () => Math.round((estimateTokens(codeSystemPrompt()) + estimateTokens(JSON.stringify(toolList())) + estimateMessages(code.messages)) * code.calib);

    function setCodeBusy(busy) {
      code.running = busy;
      const btn = $('btnCodeSend');
      btn.classList.toggle('stop', busy);
      btn.title = busy ? 'Stop (Esc)' : 'Send (Enter)';
      btn.innerHTML = busy
        ? '<svg viewBox="0 0 24 24" class="icon icon-sm"><rect width="10" height="10" x="7" y="7" rx="1" fill="currentColor"/></svg>'
        : '<svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>';
      $('codeHint').textContent = busy ? 'The agent is working. Esc stops it.' : 'Enter sends, Shift+Enter adds a new line';
    }

    // ---- permissions ----------------------------------------------------------------------------------
    /** 'deny' (a rule says no), 'allow', or 'ask'. */
    function authorize(entry, args) {
      const { cat, subject } = permissionInfo(entry, args);
      const verdict = evaluateRules(parsedRules(), cat, subject);
      if (verdict === 'deny') return 'deny';
      if (entry.source === 'client' || cat === 'read') return 'allow';
      if (verdict === 'allow') return 'allow';
      if (cat === 'edit') return code.perm === 'edit' ? 'allow' : 'ask';
      if (cat === 'bash') {
        const key = entry.name === 'bash' ? commandKey(subject) : null;
        return key && code.alwaysAllow.has(`bash:${key}`) ? 'allow' : 'ask';
      }
      if (cat === 'network') return code.alwaysAllow.has('network') ? 'allow' : 'ask';
      return code.alwaysAllow.has(`mcp:${subject}`) ? 'allow' : 'ask';
    }

    /** Waits for a click on a permission or question card. `cancel` is the answer when the run is stopped. */
    function waitFor(item, cancel) {
      return new Promise((resolve) => {
        code.waiters.set(item.id, { resolve, cancel });
        item.status = 'pending';
        paintItem(item);
        $('codeScroll').scrollTop = $('codeScroll').scrollHeight;
      });
    }
    const askPermission = (item) => waitFor(item, 'deny');

    function answerPermission(answer, id) {
      const w = code.waiters.get(id);
      if (!w) return;
      code.waiters.delete(id);
      if (answer === 'always') {
        const item = findItem(id);
        const key = item && alwaysKey(item);
        if (key === 'edit') { code.perm = 'edit'; localStorage.setItem('fllm.studio.perm', code.perm); renderCode(); }
        else if (key) code.alwaysAllow.add(key);
      }
      w.resolve(answer === 'deny' ? 'deny' : 'allow');
    }

    function cancelWaiters() {
      for (const [, w] of code.waiters) w.resolve(w.cancel);
      code.waiters.clear();
    }

    // ---- one model turn -----------------------------------------------------------------------------------
    /** Tool calls the server could not parse sometimes arrive as text: <tool_call>{...}</tool_call>. */
    function extractTextToolCalls(content) {
      const calls = [];
      for (const m of String(content).matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g)) {
        let text = m[1].trim();
        for (let i = 0; i < 3; i++) {
          try {
            const obj = JSON.parse(text);
            if (obj && obj.name) { calls.push({ id: `call_${cid()}`, name: obj.name, args: JSON.stringify(obj.arguments || obj.parameters || {}) }); }
            break;
          } catch {
            text = text.replace(/^\{\{/, '{').replace(/\}\}$/, '}');
          }
        }
      }
      return calls;
    }

    /** ctx: { system, tools, messages, signal, onUsage }. Every request starts with the same system prompt and tool list. */
    async function streamAssistant(item, ctx) {
      const p = getSampler();
      const body = {
        model: state.selectedModel || state.status?.model?.name || 'tiny',
        messages: [{ role: 'system', content: ctx.system }, ...ctx.messages],
        stream: true, stream_options: { include_usage: true },
        temperature: Math.min(p.temperature, 0.4), top_p: 0.9,
      };
      const estimate = fixedTokens(ctx) + estimateMessages(ctx.messages);
      const wanted = Math.max(p.max_tokens, 2048);
      const fit = computeMaxTokens(ctxMax(), Math.round(estimate * code.calib), wanted);
      if (!fit) throw new Error(TOO_SMALL);
      body.max_tokens = fit;
      if (ctx.tools.length) { body.tools = ctx.tools; if (!code.noGrammar) body.tool_call_grammar = true; }
      if (state.thinkingEnabled) body.reasoning_effort = 'high';
      let res = await apiFetch('/v1/chat/completions', { method: 'POST', body: { ...body }, signal: ctx.signal });
      for (let attempt = 0; attempt < 2 && !res.ok && res.status === 400; attempt++) {
        const err = await res.clone().json().catch(() => ({}));
        const info = parseContextError(err.error?.message);
        if (info) {
          // the server counted more prompt tokens than we did: learn the ratio and retry once with a smaller answer
          code.calib = Math.min(2, Math.max(1, info.prompt / Math.max(1, estimate)));
          const smaller = retryMaxTokens(info, wanted);
          if (!smaller) throw new Error(TOO_SMALL);
          body.max_tokens = smaller;
        } else if (body.tool_call_grammar && !code.noGrammar) {
          code.noGrammar = true; // this server does not know the field
          delete body.tool_call_grammar;
        } else break;
        res = await apiFetch('/v1/chat/completions', { method: 'POST', body: { ...body }, signal: ctx.signal });
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error?.message || `The server answered with error ${res.status}`);
      }
      const calls = [];
      let finish = '';
      await readEventStream(res, (chunk) => {
        const choice = chunk.choices?.[0];
        const delta = choice?.delta;
        if (delta?.reasoning_content) { item.reasoning = (item.reasoning || '') + delta.reasoning_content; paintSoon(item); }
        if (delta?.content) { item.content += delta.content; paintSoon(item); }
        for (const tc of delta?.tool_calls || []) {
          const i = tc.index ?? calls.length - 1;
          if (!calls[i]) calls[i] = { id: tc.id || `call_${cid()}`, name: '', args: '' };
          if (tc.id) calls[i].id = tc.id;
          if (tc.function?.name) calls[i].name += tc.function.name;
          if (tc.function?.arguments) calls[i].args += tc.function.arguments;
        }
        if (choice?.finish_reason) finish = choice.finish_reason;
        if (chunk.usage?.total_tokens) ctx.onUsage?.(chunk.usage.total_tokens);
        if (chunk.usage?.prompt_tokens) code.calib = Math.min(2, Math.max(0.7, chunk.usage.prompt_tokens / Math.max(1, estimate)));
        if (chunk.timings?.predicted_per_second) item.tps = chunk.timings.predicted_per_second;
      });
      let found = calls.filter(Boolean);
      if (!found.length && item.content.includes('<tool_call>')) {
        found = extractTextToolCalls(item.content);
        if (found.length) item.content = item.content.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();
      }
      return { calls: found.filter((c) => c.name), finish };
    }

    function parseArgs(text) {
      const raw = String(text || '').trim();
      if (!raw) return {};
      try { return JSON.parse(raw); } catch { /* try again below */ }
      try { return JSON.parse(raw.replace(/^```(?:json)?\s*|\s*```$/g, '')); } catch { return null; }
    }

    const clipOutput = (text) => {
      const max = Math.max(3000, Math.floor(ctxMax() * 3.4 * 0.22));
      const s = String(text ?? '');
      return s.length > max ? `${s.slice(0, max)}\n... [${s.length - max} characters cut]` : s;
    };

    /** A new view item in the list of this context; items of sub-agents know the card they are painted in. */
    function newItem(ctx, props) {
      const item = { id: cid(), ...props };
      ctx.view.push(item);
      if (ctx.root) ownerOf.set(item, ctx.root);
      return item;
    }

    // Background jobs: remember the id of a job started with bash {background:true}, keep its output.
    const jobArgName = () => {
      const props = Object.keys(code.registry.get('bash_output')?.def.parameters?.properties || {});
      return props.find((k) => /id|job|shell/i.test(k)) || props[0] || 'id';
    };
    function trackJob(name, args, output) {
      if (name === 'bash' && args.background) {
        const id = parseJobId(output);
        if (id) { code.jobs.push({ id, command: String(args.command || ''), status: 'running', out: '', open: false }); renderCodeSide(); }
      } else if (name === 'bash_output' || name === 'bash_kill') {
        const id = String(Object.values(args)[0] ?? '');
        const job = code.jobs.find((j) => j.id === id);
        if (job) {
          if (name === 'bash_kill') job.status = 'killed';
          else { job.out = output; if (/\b(exited|finished|completed|exit code)\b/i.test(output)) job.status = 'done'; }
          renderCodeSide();
        }
      }
    }
    async function jobOutput(id) {
      const job = code.jobs.find((j) => j.id === id);
      if (!job) return;
      job.open = !job.open;
      if (job.open) {
        const r = await toolCall('bash_output', { [jobArgName()]: id }).catch((e) => ({ ok: false, error: e.message }));
        job.out = r.ok ? r.output : (r.error || '');
        if (r.ok && /\b(exited|finished|completed|exit code)\b/i.test(r.output)) job.status = 'done';
      }
      renderCodeSide();
    }
    async function jobKill(id) {
      const r = await toolCall('bash_kill', { [jobArgName()]: id }).catch((e) => ({ ok: false, error: e.message }));
      const job = code.jobs.find((j) => j.id === id);
      if (job && r.ok) job.status = 'killed';
      if (!r.ok) toast(r.error || 'The job could not be stopped.');
      renderCodeSide();
    }

    /** Runs one tool call and returns the text the model gets back. ctx: { view, root, depth, readOnly, signal, isStopped }. */
    async function runToolCall(call, ctx) {
      const args = parseArgs(call.args);
      const item = newItem(ctx, { kind: 'tool', callId: call.id, name: call.name, args: args || {}, status: 'running', output: '' });
      const entry = code.registry.get(call.name);
      const finish = (status, out, reply) => { item.status = status; item.output = out; paintItem(item); return clipOutput(reply ?? `Error: ${out}`); };
      if (args === null) return finish('error', 'The arguments are not valid JSON.', 'Error: the arguments were not valid JSON. Call the tool again with a valid JSON object.');
      if (!entry) return finish('error', `Unknown tool ${call.name}`, `Error: unknown tool '${call.name}'. Available tools: ${[...code.registry.keys()].join(', ')}.`);
      if (ctx.depth > 0 && entry.source === 'client') return finish('denied', 'Not available to sub-agents.', `Denied: ${call.name} is not available to a sub-agent. Do the work yourself and put the answer in your report.`);
      if (ctx.readOnly && !readOnlyTool(entry)) return finish('denied', 'This sub-agent is read-only.', 'Denied: this sub-agent is read-only. Describe the change in your report instead.');
      if (code.perm === 'plan' && !readOnlyTool(entry) && entry.source !== 'client') return finish('denied', 'Plan mode only allows reading.', 'Denied: plan mode only allows reading. Describe the change in your plan instead.');

      if (call.name === 'todo_write') {
        code.todos = (Array.isArray(args.todos) ? args.todos : []).filter((t) => t && t.content).map((t) => ({ content: String(t.content), status: ['pending', 'in_progress', 'completed'].includes(t.status) ? t.status : 'pending' }));
        item.args = { todos: code.todos };
        renderCodeSide();
        return finish('ok', '', 'Todos updated.');
      }
      if (call.name === 'ask_user') {
        if (!args.question) return finish('error', 'No question.', 'Error: ask_user needs a question.');
        paintItem(item);
        const answer = await waitFor(item, null);
        if (answer === null || ctx.isStopped()) return finish('denied', 'Stopped.', 'The user did not answer.');
        return finish('ok', `Answer: ${answer}`, `The user answered: ${answer}`);
      }
      if (call.name === 'spawn_agent') return runSubAgent(item, args, ctx);

      const decision = authorize(entry, args);
      if (decision === 'deny') return finish('denied', 'Blocked by a permission rule.', 'Denied: a permission rule of the user blocks this action. Do not try it again; choose another way or ask the user.');
      if (decision === 'ask') {
        paintItem(item);
        const verdict = await askPermission(item);
        if (verdict === 'deny' || ctx.isStopped()) return finish('denied', ctx.isStopped() ? 'Stopped.' : 'You denied this.', 'The user denied this action. Do not try it again unchanged; ask what they want instead.');
      }
      item.status = 'running';
      paintItem(item);
      try {
        const r = entry.source === 'mcp' ? await mcpCall(entry, args, ctx.signal) : await toolCall(call.name, args, ctx.signal);
        item.status = r.ok ? 'ok' : 'error';
        item.output = r.ok ? r.output : r.error;
        if (r.ok && EDIT_TOOLS.has(call.name) && args.path) {
          code.changed.set(args.path, (code.changed.get(args.path) || 0) + 1);
          renderCodeSide();
          refreshChanges();
        }
        if (r.ok) trackJob(call.name, args, r.output);
        paintItem(item);
        return clipOutput(r.ok ? r.output : `Error: ${r.error}`);
      } catch (e) {
        return finish('error', e.name === 'AbortError' ? 'Stopped.' : e.message);
      }
    }

    // ---- sub-agents ------------------------------------------------------------------------------------------
    function subAgentTask(args, readOnly) {
      return [
        'You are a sub-agent. The main agent handed you the job below and cannot see this conversation, only your final reply.',
        readOnly ? 'You may only read and search; changes and commands are refused.' : 'You may also change files and run commands; the user is asked for permission as usual.',
        'You cannot ask the user questions or start other sub-agents. Work on your own with as few tool calls as it takes. Your last message is the report: findings first, with file paths and line numbers, no filler, no repeated task description.',
        '',
        `Job: ${args.description}`,
        '',
        args.prompt,
      ].join('\n');
    }

    async function runSubAgent(item, args, parent) {
      const prompt = String(args.prompt || '').trim();
      if (!prompt) { item.status = 'error'; item.output = 'No prompt.'; paintItem(item); return 'Error: spawn_agent needs a prompt.'; }
      if (parent.depth > 0) { item.status = 'denied'; item.output = 'Sub-agents cannot start sub-agents.'; paintItem(item); return 'Denied: sub-agents cannot start sub-agents.'; }
      args.description = String(args.description || 'Sub-task').slice(0, 80);
      const readOnly = args.mode !== 'full' || code.perm === 'plan';
      item.children = [];
      item.status = 'running';
      item.progress = 'starting';
      paintItem(item);
      const max = ctxMax();
      const ctx = {
        system: parent.system, tools: parent.tools, messages: [{ role: 'user', content: subAgentTask(args, readOnly) }],
        view: item.children, root: item, depth: 1, readOnly, signal: parent.signal, isStopped: parent.isStopped,
        maxSteps: CHILD_MAX_STEPS, budget: Math.floor(max * 0.75), onUsage: null, child: true,
      };
      let report = '';
      try {
        report = await runLoop(ctx, (text) => { item.progress = text; paintSoon(item); });
        item.status = parent.isStopped() ? 'denied' : 'ok';
        item.output = report || '(the sub-agent wrote no report)';
        item.progress = '';
      } catch (e) {
        item.status = e.name === 'AbortError' ? 'denied' : 'error';
        item.output = e.name === 'AbortError' ? 'Stopped.' : e.message;
      }
      paintItem(item);
      return clipOutput(item.status === 'ok' ? `Sub-agent report (${args.description}):\n${item.output}` : `Sub-agent failed: ${item.output}`);
    }

    // ---- the loop (parent and sub-agents) ------------------------------------------------------------------
    /** Shortens old tool output, then drops old steps. Last resort when summarising is not possible. */
    function fitContext(messages, sysTokens) {
      const limit = Math.floor(ctxMax() * COMPACT_RATIO);
      const est = () => sysTokens + estimateMessages(messages);
      if (est() <= limit) return 0;
      messages.forEach((m, i) => {
        if (m.role === 'tool' && i < messages.length - 6 && m.content.length > 400) m.content = `${m.content.slice(0, 300)}\n... [shortened to save space]`;
      });
      let dropped = 0;
      while (est() > limit) {
        const next = messages.findIndex((m, i) => i > 0 && m.role === 'user');
        if (next === -1) break;
        messages.splice(0, next);
        dropped++;
      }
      while (est() > limit && messages.length > 3) {
        let end = 2;
        while (messages[end] && messages[end].role === 'tool') end++;
        messages.splice(1, end - 1);
        dropped++;
      }
      return dropped;
    }

    async function summarise(messages, system, tools, signal, focus) {
      const ask = 'Summarise the conversation above so that work can continue from your summary alone. Include: what the user wants, decisions made, files read or changed (paths) and what changed, commands run and their results, open errors, and the next steps. Keep it under 500 words. Do not call tools.' + (focus ? ` Pay special attention to: ${focus}` : '');
      const body = { model: state.selectedModel || state.status?.model?.name || 'tiny', messages: [{ role: 'system', content: system }, ...messages, { role: 'user', content: ask }], stream: false, temperature: 0.2 };
      const room = computeMaxTokens(ctxMax(), Math.round((estimateTokens(system) + estimateTokens(JSON.stringify(tools)) + estimateMessages(body.messages.slice(1))) * code.calib), 1200);
      if (!room) throw new Error(TOO_SMALL);
      body.max_tokens = room;
      if (tools.length) body.tools = tools; // same prefix as the conversation: the server cache is reused
      const res = await apiFetch('/v1/chat/completions', { method: 'POST', body, signal });
      if (!res.ok) throw new Error(`The server answered with error ${res.status}`);
      const data = await res.json();
      return splitReasoning(String(data.choices?.[0]?.message?.content || '')).answer.trim();
    }

    /** Summarises older turns and replaces them. Returns true when it did. Never retries by itself. */
    async function compactSession(ctx, { force = false, focus = '' } = {}) {
      const plan = splitForCompaction(code.messages);
      if (!plan) { if (force) codeNotice('There is not enough conversation to compact yet.'); return false; }
      if (!force && code.messages.length < code.compactedAt + 4) return false;
      const before = codeEstimate();
      const note = { kind: 'notice', id: cid(), text: 'Compacting the conversation ...' };
      code.view.push(note);
      paintItem(note);
      let summary = '';
      try { summary = await summarise(plan.head, ctx.system, ctx.tools, ctx.signal, focus); } catch (e) { if (e.name === 'AbortError') throw e; }
      if (!summary) { note.text = 'Compacting failed; older steps are trimmed instead.'; paintItem(note); return false; }
      code.messages = buildCompactedMessages(summary, plan.tail, { todos: code.todos, files: [...code.changed.keys()] });
      code.compactedAt = code.messages.length;
      code.lastUsed = 0;
      code.used = 0;
      note.text = `Context compacted: ${plan.head.length} earlier messages became a summary (about ${before} to ${codeEstimate()} tokens). You still see everything here.`;
      paintItem(note);
      ctx.messages = code.messages;
      return true;
    }

    /** The agent loop. Returns the text of the last answer. `status` gets short progress texts. */
    async function runLoop(ctx, status) {
      let last = null;
      let compactions = 0;
      let wrapUp = false;
      for (let step = 0; step < ctx.maxSteps && !ctx.isStopped(); step++) {
        if (!ctx.child) {
          const limit = ctxMax();
          if (compactions < 2 && needsCompaction(Math.max(codeEstimate(), code.lastUsed), limit)) {
            compactions++;
            await compactSession(ctx);
          }
          const dropped = fitContext(code.messages, fixedTokens(ctx));
          if (dropped) codeNotice('Older steps were removed from the context of the model so that it does not overflow. You still see them here.');
          ctx.messages = code.messages;
        } else if (!wrapUp && (step === ctx.maxSteps - 1 || fixedTokens(ctx) + estimateMessages(ctx.messages) > ctx.budget)) {
          wrapUp = true; // out of steps or budget: ask for the report and ignore further tool calls
          ctx.messages.push({ role: 'user', content: 'Stop investigating now. Write your final report with what you found so far, without calling tools.' });
        }
        status?.(`step ${step + 1}${wrapUp ? ', writing report' : ''}`);
        const item = newItem(ctx, { kind: 'assistant', content: '', reasoning: '', streaming: true });
        paintItem(item);
        let result;
        try { result = await streamAssistant(item, ctx); } finally { item.streaming = false; }
        last = item;
        const calls = wrapUp ? [] : result.calls;
        const message = { role: 'assistant', content: item.content || '' };
        if (calls.length) message.tool_calls = calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args || '{}' } }));
        if (item.content || calls.length) ctx.messages.push(message);
        if (!calls.length) {
          if (!item.content) item.error = result.finish === 'length' ? 'The answer was cut off by the length limit.' : 'The model did not answer.';
          paintItem(item);
          break;
        }
        if (!item.content && !item.reasoning) {
          ctx.view.splice(ctx.view.indexOf(item), 1);
          if (!ctx.root) $('codeColumn').querySelector(`[data-item="${item.id}"]`)?.remove(); else paintItem(ctx.root);
        } else paintItem(item);

        // sub-agents of one turn run in parallel, everything else one after the other, results in call order
        const replies = new Array(calls.length);
        let i = 0;
        while (i < calls.length && !ctx.isStopped()) {
          if (calls[i].name === 'spawn_agent') {
            let j = i;
            while (j < calls.length && calls[j].name === 'spawn_agent' && j - i < MAX_PARALLEL_CHILDREN) j++;
            const group = calls.slice(i, j);
            status?.(`${group.length} sub-agent${group.length > 1 ? 's' : ''} running`);
            const out = await Promise.all(group.map((c) => runToolCall(c, ctx)));
            out.forEach((r, k) => { replies[i + k] = r; });
            i = j;
          } else {
            replies[i] = await runToolCall(calls[i], ctx);
            i++;
          }
        }
        calls.forEach((c, k) => ctx.messages.push({ role: 'tool', tool_call_id: c.id, content: replies[k] ?? 'Stopped before this call ran.' }));
        if (!ctx.child) { saveCodeSession(); if (step === ctx.maxSteps - 1) codeNotice(`Stopped after ${ctx.maxSteps} steps. Write "continue" if the agent should go on.`); }
      }
      return last ? splitReasoning(last.content || '').answer.trim() : '';
    }

    async function agentRun(userText, display) {
      if (!code.workspace) { await pickWorkspace(); if (!code.workspace) return; }
      const expanded = await expandMentions(userText);
      code.view.push({ kind: 'user', id: cid(), text: display || userText });
      code.messages.push({ role: 'user', content: expanded });
      renderCode();
      code.abort = new AbortController();
      setCodeBusy(true);
      const ctx = {
        system: codeSystemPrompt(), tools: toolList(), messages: code.messages, view: code.view, root: null, depth: 0, readOnly: false,
        signal: code.abort.signal, isStopped: () => !code.running, maxSteps: CODE_MAX_TURNS, child: false,
        onUsage: (n) => { code.used = n; code.lastUsed = n; },
      };
      try {
        await runLoop(ctx);
        const lastItem = [...code.view].reverse().find((i) => i.kind === 'assistant');
        if (lastItem && code.perm === 'plan' && lastItem.content && code.view[code.view.length - 1] === lastItem) { lastItem.planDone = true; paintItem(lastItem); }
      } catch (e) {
        if (e.name !== 'AbortError') code.view.push({ kind: 'assistant', id: cid(), content: '', error: e.message });
      } finally {
        cancelWaiters();
        code.abort = null;
        setCodeBusy(false);
        saveCodeSession();
        renderCode();
        refreshChanges();
        refreshStudioStatus();
      }
    }

    function stopAgent() {
      if (!code.running) return;
      code.running = false;
      code.abort?.abort();
      cancelWaiters();
    }

    // ---- @files, commands --------------------------------------------------------------------------------
    async function expandMentions(text) {
      const names = [...new Set([...text.matchAll(/(?:^|\s)@([\w./\\-]+\.\w+|[\w./\\-]+\/)/g)].map((m) => m[1]))].slice(0, 5);
      if (!names.length) return text;
      let out = text;
      for (const name of names) {
        const r = await toolCall(name.endsWith('/') ? 'list_dir' : 'read_file', name.endsWith('/') ? { path: name } : { path: name, limit: 300 }).catch(() => null);
        if (r?.ok) out += `\n\nContents of @${name}:\n\`\`\`\n${r.output}\n\`\`\``;
      }
      return out;
    }

    async function userRun(command) {
      const item = { kind: 'tool', id: cid(), name: 'bash', args: { command }, status: 'running', output: '' };
      code.view.push(item);
      paintItem(item);
      const r = await toolCall('bash', { command }).catch((e) => ({ ok: false, error: e.message }));
      item.status = r.ok ? 'ok' : 'error';
      item.output = r.ok ? r.output : r.error;
      paintItem(item);
      saveCodeSession();
    }

    function codeNotice(text) {
      const item = { kind: 'notice', id: cid(), text };
      code.view.push(item);
      paintItem(item);
    }

    /** A notice with buttons (sessions, rewind points). Not saved with the session. */
    function pushChoices(text, action, rows) {
      const item = { kind: 'notice', id: cid(), text, action, choices: rows };
      code.view.push(item);
      paintItem(item);
    }

    function setPerm(perm) {
      code.perm = perm;
      localStorage.setItem('fllm.studio.perm', perm);
      $('permSeg').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.perm === perm));
      if (!code.view.length) renderCode();
    }

    function newCodeSession() {
      if (code.running) stopAgent();
      saveCodeSession();
      code.messages = []; code.view = []; code.todos = []; code.changed = new Map(); code.used = 0; code.lastUsed = 0; code.compactedAt = 0; code.alwaysAllow = new Set(); code.jobs = []; code.sessionId = null;
      renderCode();
    }

    const INIT_PROMPT = 'Explore this workspace and write a file named FLLM.md in its root with instructions for future coding sessions. Cover: what the project is, how to build, test and run it (exact commands, checked against the build files), the layout of the important folders, code style and conventions you can see, and pitfalls. Keep it under 80 lines, concrete, with no filler. If FLLM.md already exists, read it and improve it instead of starting over.';
    const BUILTIN_COMMANDS = [
      ['clear', 'Start a new session'], ['compact', 'Summarise older turns to free context'], ['init', 'Let the agent write FLLM.md'], ['resume', 'Pick an earlier session'],
      ['rewind', 'Undo file changes back to a chosen point'], ['undo', 'Undo the last file change (/undo 3 for three)'], ['plan', 'Read only mode'], ['ask', 'Ask before every change'], ['edit', 'Auto-edit mode'],
      ['diff', 'git diff'], ['status', 'git status'], ['cd', 'Open another folder'], ['allow', 'Add an allow rule, e.g. /allow bash: npm test*'], ['deny', 'Add a deny rule'], ['rules', 'Show the permission rules'], ['help', 'List the commands'],
    ];

    async function runCodeCommand(text) {
      const [cmd, ...rest] = text.slice(1).split(/\s+/);
      const argText = text.slice(1).slice(cmd.length).trim();
      const name = cmd.toLowerCase();
      switch (name) {
        case 'clear': case 'new': newCodeSession(); return true;
        case 'undo': await undoChange(Number(rest[0]) || 1); return true;
        case 'rewind': await showRewind(); return true;
        case 'compact':
          if (code.running) { toast('Wait until the agent has finished.'); return true; }
          if (!code.workspace) return true;
          code.abort = new AbortController();
          setCodeBusy(true);
          try { await compactSession({ system: codeSystemPrompt(), tools: toolList(), signal: code.abort.signal }, { force: true, focus: argText }); } catch (e) { if (e.name !== 'AbortError') codeNotice(`Compacting failed: ${e.message}`); } finally { code.abort = null; setCodeBusy(false); saveCodeSession(); renderCode(); }
          return true;
        case 'init': await agentRun(INIT_PROMPT, '/init'); return true;
        case 'resume': {
          const rows = workspaceSessions().filter((s) => s.id !== code.sessionId).map((s) => ({ label: `${s.title}  (${new Date(s.time).toLocaleString()})`, data: s.id }));
          if (rows.length) pushChoices('Pick a session to continue:', 'resume', rows); else codeNotice('There is no earlier session for this folder.');
          return true;
        }
        case 'plan': case 'ask': case 'edit': setPerm(name); codeNotice({ plan: 'Plan mode: the agent only reads.', ask: 'The agent asks before every change and every command.', edit: 'Auto-edit: the agent changes files itself and asks before commands.' }[name]); return true;
        case 'diff': await userRun('git diff --stat && git diff'); return true;
        case 'status': await userRun('git status -sb'); return true;
        case 'cd': case 'folder': if (rest.length) await setWorkspace(rest.join(' ')); else await pickWorkspace(); return true;
        case 'allow': case 'deny': {
          if (!parseRules(argText).length) { codeNotice(`Usage: /${name} category: pattern, for example /${name} bash: npm test*`); return true; }
          saveRules(name === 'allow' ? `${code.rules.allow}\n${argText}`.trim() : code.rules.allow, name === 'deny' ? `${code.rules.deny}\n${argText}`.trim() : code.rules.deny);
          renderCodeSide(); codeNotice(`Rule saved: ${name} ${argText}`);
          return true;
        }
        case 'rules': codeNotice(`Allow:\n${code.rules.allow || '(none)'}\n\nDeny:\n${code.rules.deny || '(none)'}`); return true;
        case 'help': {
          const custom = code.custom.map((c) => `/${c.name}${c.description ? ` (${c.description})` : ''}`).join(', ');
          codeNotice(`Commands: ${BUILTIN_COMMANDS.map(([n]) => `/${n}`).join(', ')}.${custom ? ` Custom commands from .fllm/commands: ${custom}.` : ''} Use @file to attach a file.`);
          return true;
        }
        default: {
          const c = code.custom.find((x) => x.name === name || x.name === cmd);
          if (!c) return false;
          try {
            const body = await readCustomCommand(c);
            await agentRun(expandCommand(body, argText), text);
          } catch (e) { codeNotice(e.message); }
          return true;
        }
      }
    }

    // ---- composer: @ and / completion ---------------------------------------------------------------------------
    let mentionTimer = null;
    let mentionItems = []; // { value, hint }
    let mentionIndex = 0;
    let mentionSlash = false;
    function closeMention() { $('mentionPop').classList.remove('open'); mentionItems = []; }
    function renderMention() {
      const pop = $('mentionPop');
      if (!mentionItems.length) { closeMention(); return; }
      pop.innerHTML = mentionItems.map((p, i) => `<button class="${i === mentionIndex ? 'active' : ''}" data-mention="${i}"><span class="mono">${escapeHtml(p.value)}</span>${p.hint ? `<small class="mention-hint">${escapeHtml(p.hint)}</small>` : ''}</button>`).join('');
      pop.classList.add('open');
    }
    function pickMention(i) {
      const ta = $('codeTextarea');
      const cursor = ta.selectionStart;
      let before;
      if (mentionSlash) before = `/${mentionItems[i].value.slice(1)} `;
      else before = ta.value.slice(0, cursor).replace(/@[\w./\\*-]*$/, `@${mentionItems[i].value} `);
      ta.value = mentionSlash ? before + ta.value.slice(cursor) : before + ta.value.slice(cursor);
      ta.setSelectionRange(before.length, before.length);
      closeMention();
      ta.focus();
    }
    function onCodeInput() {
      const ta = $('codeTextarea');
      ta.style.height = 'auto';
      ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
      clearTimeout(mentionTimer);
      const slash = ta.value.match(/^\/([\w:.-]*)$/);
      if (slash) {
        const all = [...BUILTIN_COMMANDS.map(([n, h]) => ({ value: `/${n}`, hint: h })), ...code.custom.map((c) => ({ value: `/${c.name}`, hint: c.description || 'custom command' }))];
        mentionItems = all.filter((c) => c.value.slice(1).startsWith(slash[1].toLowerCase())).slice(0, 8);
        mentionSlash = true;
        mentionIndex = 0;
        renderMention();
        return;
      }
      mentionSlash = false;
      const m = ta.value.slice(0, ta.selectionStart).match(/(?:^|\s)@([\w./\\*-]*)$/);
      if (!m || !code.workspace) { closeMention(); return; }
      mentionTimer = setTimeout(async () => {
        const q = m[1].replace(/\\/g, '/');
        const r = await toolCall('glob', { pattern: q.includes('*') ? q : `**/*${q}*` }).catch(() => null);
        if (!r?.ok) { closeMention(); return; }
        mentionItems = r.output.split('\n').filter((l) => l && !l.startsWith('...') && l !== 'no files match').slice(0, 8).map((value) => ({ value }));
        mentionIndex = 0;
        renderMention();
      }, 140);
    }

    async function submitCode() {
      if (code.running) { stopAgent(); return; }
      const ta = $('codeTextarea');
      const text = ta.value.trim();
      if (!text) return;
      if (state.setup) { toast('Load a model first.'); switchSurface('hub'); return; }
      ta.value = '';
      ta.style.height = 'auto';
      closeMention();
      if (text.startsWith('/') && (await runCodeCommand(text))) return;
      await agentRun(text);
    }

    // ---- events --------------------------------------------------------------------------------------------------------
    $('codeTextarea').addEventListener('input', onCodeInput);
    $('codeTextarea').addEventListener('keydown', (e) => {
      if ($('mentionPop').classList.contains('open')) {
        if (e.key === 'ArrowDown') { e.preventDefault(); mentionIndex = (mentionIndex + 1) % mentionItems.length; renderMention(); return; }
        if (e.key === 'ArrowUp') { e.preventDefault(); mentionIndex = (mentionIndex + mentionItems.length - 1) % mentionItems.length; renderMention(); return; }
        if (e.key === 'Tab' || (e.key === 'Enter' && !mentionSlash)) { e.preventDefault(); pickMention(mentionIndex); return; }
        if (e.key === 'Enter' && mentionSlash && $('codeTextarea').value.trim() !== mentionItems[mentionIndex].value) { e.preventDefault(); pickMention(mentionIndex); return; }
        if (e.key === 'Escape') { e.preventDefault(); closeMention(); return; }
      }
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); submitCode(); }
    });
    $('mentionPop').addEventListener('mousedown', (e) => {
      const b = e.target.closest('[data-mention]');
      if (b) { e.preventDefault(); pickMention(Number(b.dataset.mention)); }
    });
    $('btnCodeSend').addEventListener('click', submitCode);
    $('btnWorkspace').addEventListener('click', pickWorkspace);
    $('btnCodeUndo').addEventListener('click', () => undoChange());
    $('btnCodeNew').addEventListener('click', newCodeSession);
    $('permSeg').addEventListener('click', (e) => {
      const b = e.target.closest('[data-perm]');
      if (b) setPerm(b.dataset.perm);
    });

    // remember which cards the user opened or closed, so repainting keeps them
    document.addEventListener('toggle', (e) => {
      const el = e.target;
      if (!(el instanceof HTMLDetailsElement)) return;
      if (el.dataset.k === 'rules') { code.rulesOpen = el.open; return; }
      const id = el.closest('[data-item]')?.dataset.item;
      const item = id && findItem(id);
      if (!item) return;
      if (el.dataset.k === 'cot') item.cotOpen = el.open; else if (el.parentElement?.matches('[data-item]')) item.open = el.open;
    }, true);

    $('codeColumn').addEventListener('click', async (e) => {
      const answer = e.target.closest('[data-perm-answer]');
      if (answer) { answerPermission(answer.dataset.permAnswer, answer.closest('[data-item]')?.dataset.item); return; }
      const opt = e.target.closest('[data-ask-opt]');
      const sendFree = e.target.closest('[data-ask-send]');
      if (opt || sendFree) {
        const card = (opt || sendFree).closest('[data-item]');
        const item = findItem(card.dataset.item);
        const text = opt ? String(item.args.options[Number(opt.dataset.askOpt)]) : card.querySelector('[data-ask-text]').value.trim();
        const w = code.waiters.get(card.dataset.item);
        if (w && text) { code.waiters.delete(card.dataset.item); w.resolve(text); }
        return;
      }
      const expand = e.target.closest('[data-expand]');
      if (expand) { const item = findItem(expand.dataset.expand); if (item) { item.expanded = true; paintItem(item); } return; }
      const choice = e.target.closest('[data-choice]');
      if (choice) {
        const [id, idx] = choice.dataset.choice.split(':');
        const item = findItem(id);
        const row = item?.choices?.[Number(idx)];
        if (!row) return;
        code.view.splice(code.view.indexOf(item), 1);
        if (item.action === 'resume') resumeSession(row.data); else if (item.action === 'rewind') { renderCode(); await rewindTo(row.data); }
        return;
      }
      const resume = e.target.closest('[data-resume]');
      if (resume) { resumeSession(resume.dataset.resume); return; }
      const cmd = e.target.closest('[data-code-command]');
      if (cmd) { await runCodeCommand(cmd.dataset.codeCommand); return; }
      if (e.target.closest('[data-run-plan]')) { setPerm('ask'); await agentRun('Carry out the plan now.'); return; }
      const prompt = e.target.closest('[data-code-prompt]');
      if (prompt) { await agentRun(prompt.dataset.codePrompt); return; }
      const ws = e.target.closest('[data-ws]');
      if (ws) { await setWorkspace(ws.dataset.ws); return; }
      if (e.target.closest('#wsOpenBtn')) { const v = $('wsPathInput').value.trim(); if (v) await setWorkspace(v); return; }
      const copy = e.target.closest('[data-copy-code]');
      if (copy) { const pre = $(copy.dataset.copyCode); if (pre) { await navigator.clipboard.writeText(pre.textContent); toast('Code copied'); } return; }
      const wrap = e.target.closest('[data-wrap-code]');
      if (wrap) $(wrap.dataset.wrapCode)?.classList.toggle('wrap');
    });
    $('codeColumn').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target.id === 'wsPathInput') $('wsOpenBtn').click();
      if (e.key === 'Enter' && e.target.matches('[data-ask-text]')) e.target.closest('.ask-free').querySelector('[data-ask-send]').click();
    });
    $('codeSide').addEventListener('input', (e) => {
      if (e.target.dataset.rules) code.rulesDraft = { ...(code.rulesDraft || code.rules), [e.target.dataset.rules]: e.target.value };
    });
    $('codeSide').addEventListener('click', async (e) => {
      const save = e.target.closest('[data-rules-save]');
      if (save) {
        const d = code.rulesDraft || code.rules;
        saveRules(d.allow, d.deny);
        renderCodeSide();
        toast(`${parseRules(d.allow).length + parseRules(d.deny).length} rules saved`);
        return;
      }
      const resume = e.target.closest('[data-resume]');
      if (resume) { resumeSession(resume.dataset.resume); return; }
      const cmd = e.target.closest('[data-code-command]');
      if (cmd) { await runCodeCommand(cmd.dataset.codeCommand); return; }
      const out = e.target.closest('[data-job-out]');
      if (out) { await jobOutput(out.dataset.jobOut); return; }
      const kill = e.target.closest('[data-job-kill]');
      if (kill) await jobKill(kill.dataset.jobKill);
    });
