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
    const CODE_MODES_KEY = 'fllm.studio.codemodes.v1';
    const CODE_SETTINGS_KEY = 'fllm.studio.codesettings.v1';
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
    const RULE_CATEGORIES = new Set(['bash', 'edit', 'network', 'mcp', 'command', 'read', 'memory', '*']);

    /** "bash: git status*" per line, optionally for one agent: "[plan] bash: ls*". Blank lines and # comments are ignored. */
    function parseRules(text) {
      const rules = [];
      for (const line of String(text || '').split('\n')) {
        const m = line.match(/^\s*(?:\[([A-Za-z]+)\]\s*)?([A-Za-z*]+)\s*:\s*(.+?)\s*$/);
        if (!m || line.trim().startsWith('#')) continue;
        const category = m[2].toLowerCase();
        if (RULE_CATEGORIES.has(category)) rules.push({ category, pattern: m[3], agent: m[1] ? m[1].toLowerCase() : '' });
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
    function evaluateRules(rules, category, subject, agent = '') {
      const mine = (r) => !r.agent || !agent || r.agent === agent;
      const deny = (rules.deny || []).filter((r) => mine(r) && ruleAppliesTo(r, category));
      const allow = (rules.allow || []).filter((r) => mine(r) && ruleAppliesTo(r, category));
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

    // ---- permissions per tool category and agent ----
    const MODE_CATEGORIES = ['read', 'edit', 'bash', 'network', 'mcp', 'memory'];
    const AGENTS = ['build', 'plan'];
    /** What each agent does without being told otherwise. 'edit' follows the quick switch (ask / auto-edit). */
    const DEFAULT_MODES = {
      build: { read: 'allow', edit: 'ask', bash: 'ask', network: 'ask', mcp: 'ask', memory: 'allow' },
      plan: { read: 'allow', edit: 'deny', bash: 'deny', network: 'ask', mcp: 'ask', memory: 'deny' },
    };
    const MODE_VALUES = new Set(['allow', 'ask', 'deny']);

    /** The mode of a category for an agent: the user's table, else the default (auto-edit makes edit allow for build). */
    function modeFor(modes, agent, category, autoEdit) {
      const own = modes?.[agent]?.[category];
      if (MODE_VALUES.has(own)) return own;
      if (agent === 'build' && category === 'edit' && autoEdit) return 'allow';
      return DEFAULT_MODES[agent]?.[category] || 'ask';
    }

    /**
     * The final answer for one tool call: 'deny', 'allow' or 'ask'.
     * rules: parsed allow / deny rules; category: read / edit / bash / network / mcp / memory; subject: command, path, url ...
     * Deny rules win over everything; an allow rule beats a mode of 'ask' but never a mode of 'deny'.
     */
    function decidePermission({ rules, modes, agent, autoEdit, category, subject, remembered }) {
      if (evaluateRules(rules, category, subject, agent) === 'deny') return 'deny';
      const mode = modeFor(modes, agent, category, autoEdit);
      if (mode === 'deny') return 'deny';
      if (mode === 'allow') return 'allow';
      if (evaluateRules(rules, category, subject, agent) === 'allow') return 'allow';
      return remembered ? 'allow' : 'ask';
    }

    /** The files a unified diff names (the new side of each header). */
    function patchPaths(text) {
      const out = [];
      const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
      for (let i = 0; i < lines.length - 1; i++) {
        if (!lines[i].startsWith('--- ') || !lines[i + 1].startsWith('+++ ')) continue;
        const clean = (s) => s.slice(4).split('\t')[0].trim().replace(/^"|"$/g, '').replace(/^[ab]\//, '');
        const a = clean(lines[i]);
        const b = clean(lines[i + 1]);
        const p = b === '/dev/null' ? a : b;
        if (p && p !== '/dev/null' && !out.includes(p)) out.push(p);
      }
      return out;
    }

    // ---- consistent conversations ----
    const INTERRUPTED = 'Interrupted: this call did not finish. The user stopped the agent or the page was closed.';
    const NO_ANSWER = '(Interrupted before I could answer.)';

    /**
     * Makes a message list valid for every chat template: each tool call is answered by a tool message right after
     * its assistant message (missing ones get an "interrupted" result), tool messages without a call are dropped,
     * empty assistant messages are dropped and two user messages in a row get a short assistant line between them.
     * Returns { messages, changed }.
     */
    function repairMessages(messages) {
      const out = [];
      let changed = 0;
      const list = messages.filter((m) => m && m.role);
      for (let i = 0; i < list.length; i++) {
        const m = list[i];
        if (m.role === 'tool') { changed++; continue; } // answers that did not follow their call directly
        if (m.role === 'assistant') {
          const calls = m.tool_calls || [];
          if (!calls.length && !String(m.content || '').trim()) { changed++; continue; }
          out.push(m);
          if (!calls.length) continue;
          const answers = new Map();
          let j = i + 1;
          while (j < list.length && list[j].role === 'tool') { answers.set(list[j].tool_call_id, list[j]); j++; }
          for (const c of calls) {
            if (answers.has(c.id)) out.push(answers.get(c.id));
            else { out.push({ role: 'tool', tool_call_id: c.id, content: INTERRUPTED }); changed++; }
          }
          if (answers.size > calls.length || [...answers.keys()].some((id) => !calls.some((c) => c.id === id))) changed++;
          i = j - 1;
          continue;
        }
        const last = out[out.length - 1];
        if (m.role === 'user' && last && last.role === 'user') {
          // two user messages in a row (the first request was stopped before it was answered): templates that
          // require alternating roles refuse that, so a short assistant line stands between them
          out.push({ role: 'assistant', content: NO_ANSWER });
          changed++;
        }
        out.push(m);
      }
      return { messages: out, changed };
    }

    /** Same tool with the same arguments, regardless of key order. */
    function callSignature(name, args) {
      const norm = (v) => (v && typeof v === 'object' ? (Array.isArray(v) ? v.map(norm) : Object.fromEntries(Object.keys(v).sort().map((k) => [k, norm(v[k])]))) : v);
      return `${name}:${JSON.stringify(norm(args))}`;
    }
    /** True when this signature repeats the last `times - 1` ones of the history (a model stuck in a loop). */
    function isRepeating(history, signature, times = 3) {
      if (history.length < times - 1) return false;
      return history.slice(-(times - 1)).every((s) => s === signature);
    }

    /** The start and the end of a long text, with a marker in between (errors are usually at the end). */
    function clipMiddle(text, max) {
      const s = String(text ?? '');
      if (s.length <= max) return s;
      const head = Math.floor(max * 0.4);
      const tail = max - head;
      return `${s.slice(0, head)}\n... [${s.length - max} characters cut from the middle] ...\n${s.slice(s.length - tail)}`;
    }

    // ---- sessions ----
    /** Sessions of this folder whose title or user messages contain every word of the query. */
    function filterSessions(sessions, query) {
      const words = String(query || '').toLowerCase().split(/\s+/).filter(Boolean);
      if (!words.length) return sessions;
      return sessions.filter((s) => {
        const hay = `${s.title || ''} ${(s.view || []).filter((i) => i.kind === 'user').map((i) => i.text).join(' ')}`.toLowerCase();
        return words.every((w) => hay.includes(w));
      });
    }

    /**
     * Cuts a session just before the user message `itemId`: everything from that message on is gone.
     * Needs the message index recorded when the turn started (compaction clears it). Returns null when impossible.
     */
    function cutBefore(view, messages, itemId) {
      const at = view.findIndex((i) => i.id === itemId && i.kind === 'user');
      if (at < 0) return null;
      const item = view[at];
      if (!Number.isInteger(item.msgIndex) || item.msgIndex > messages.length) return null;
      return {
        view: view.slice(0, at),
        messages: messages.slice(0, item.msgIndex),
        text: item.text,
        todos: item.todosBefore || [],
        changeMark: item.changeMark,
      };
    }

    /** The first change id after `mark` (the oldest change of the cut-off turns), or null. */
    function firstChangeAfter(changes, mark) {
      const ids = (changes || []).map((c) => c.id).filter((id) => id > (mark || 0));
      return ids.length ? Math.min(...ids) : null;
    }

    /** A session as Markdown: user messages, answers, and the tools used (names and arguments, shortened results). */
    function exportMarkdown(session, root = '') {
      const out = [`# ${session.title || 'Session'}`, ''];
      if (root) out.push(`Workspace: \`${root}\``, '');
      out.push(`Exported ${new Date(session.time || Date.now()).toISOString().slice(0, 16).replace('T', ' ')} UTC`, '');
      const fence = (text, lang = '') => {
        const t = String(text ?? '');
        const ticks = '`'.repeat(Math.max(3, (t.match(/`+/g) || ['']).reduce((a, b) => Math.max(a, b.length), 0) + 1));
        return `${ticks}${lang}\n${t}\n${ticks}`;
      };
      const walk = (items, depth) => {
        for (const i of items) {
          if (i.kind === 'user') out.push('## User', '', i.text, '');
          else if (i.kind === 'assistant') { if (i.content) out.push(depth ? '**Sub-agent:**' : '## Assistant', '', i.content, ''); }
          else if (i.kind === 'tool') {
            const a = i.args || {};
            const what = a.command || a.path || a.pattern || a.url || a.query || a.description || '';
            out.push(`- Tool \`${i.name}\`${what ? ` ${'`'}${String(what).split('\n')[0].slice(0, 120)}${'`'}` : ''} (${i.status})`);
            if (i.name === 'bash' && i.output) out.push('', fence(clipMiddle(i.output, 1500)), '');
            if (i.children && i.children.length) walk(i.children, depth + 1);
            if (i.name === 'task' || i.name === 'spawn_agent') { if (i.output) out.push('', '> ' + String(i.output).split('\n').join('\n> '), ''); }
          } else if (i.kind === 'notice' && !i.choices) out.push(`> ${i.text.split('\n').join('\n> ')}`, '');
        }
      };
      walk(session.view || [], 0);
      return `${out.join('\n').replace(/\n{3,}/g, '\n\n').trim()}\n`;
    }

    /** `git status --porcelain -b` as one short phrase: "clean", "3 changed files (1 modified, 2 untracked), ahead 1". */
    function summariseGit(output) {
      const lines = String(output || '').split('\n').map((l) => l.replace(/\r$/, '')).filter(Boolean);
      const head = lines.find((l) => l.startsWith('## ')) || '';
      const files = lines.filter((l) => !l.startsWith('## '));
      const ahead = (head.match(/ahead (\d+)/) || [])[1];
      const behind = (head.match(/behind (\d+)/) || [])[1];
      const sync = `${ahead ? `, ahead ${ahead}` : ''}${behind ? `, behind ${behind}` : ''}`;
      if (!files.length) return `clean${sync}`;
      const untracked = files.filter((l) => l.startsWith('??')).length;
      const modified = files.length - untracked;
      const parts = [modified ? `${modified} modified` : '', untracked ? `${untracked} untracked` : ''].filter(Boolean).join(', ');
      return `${files.length} changed file${files.length === 1 ? '' : 's'} (${parts})${sync}`;
    }

    /** "!ls -la" is a shell command typed by the user. */
    function parseShellEscape(text) {
      const m = String(text || '').match(/^!\s*(\S[\s\S]*)$/);
      return m ? m[1].trim() : null;
    }

    /** A short title for a session from the first user message, when the model gives none. */
    function heuristicTitle(messages) {
      const first = messages.find((m) => m.role === 'user' && !String(m.content).startsWith('[Summary'));
      const text = String(first?.content || 'Session').split('\n\nContents of @')[0];
      return text.replace(/\s+/g, ' ').trim().slice(0, 60) || 'Session';
    }

    /** Cleans a title the model wrote: one line, no quotes or markdown, at most 60 characters. */
    function cleanTitle(text) {
      const line = String(text || '').split('\n').map((l) => l.trim()).find(Boolean) || '';
      return line.replace(/^(title|session)\s*:\s*/i, '').replace(/^[#*_`"'\s]+|[#*_`"'.\s]+$/g, '').slice(0, 60);
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
      todo_read: { description: 'Read the current task list (use it when you have lost track of the plan).', properties: {}, required: [] },
      task: { description: 'Start a sub-agent with a fresh context and get back only its final report. agent "explore" is read-only and fast: use it for broad research ("where is X handled", "how does Y work"). agent "general" may also edit files and run commands: use it for a self-contained job. Several task calls in one turn run in parallel. The sub-agent does not see this conversation, so the prompt must contain everything it needs: the goal, where to look, and what the report must contain.', properties: { description: { type: 'string', description: 'Three to six words naming the job.' }, prompt: { type: 'string', description: 'The complete task and what the report should contain.' }, agent: { type: 'string', enum: ['explore', 'general'], description: 'explore (default): read and search only. general: may also edit and run commands, with the usual permission questions.' } }, required: ['description', 'prompt'] },
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
      sideOpen: {}, // which blocks of the side panel are open
      title: '',
      titleLocked: false,
      lastBuildPerm: 'ask',
      history: [],
      jobs: [],
      sessionId: null,
      instructions: [], // [{ file, text }] AGENTS.md / FLLM.md / CLAUDE.md, nearest first
      memory: { workspace: [], global: [] },
      memDraft: null, // { workspace, global } text being edited in the memory panel
      git: '', // short working tree state for the system prompt
      modes: {}, // { build: { edit: 'ask' ... }, plan: {...} } overrides of DEFAULT_MODES
      autoCheck: false, // run diagnostics after every edit and append the first problems
      queue: [], // messages typed while the agent was busy
      reverted: null, // what /revert took away, so it can be put back
      sessionQuery: '',
      titling: false,
      runChain: Promise.resolve(), // runs of the agent one after the other
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

    try {
      const saved = JSON.parse(localStorage.getItem(CODE_SETTINGS_KEY) || '{}');
      code.autoCheck = Boolean(saved.autoCheck);
      const modes = JSON.parse(localStorage.getItem(CODE_MODES_KEY) || '{}');
      if (modes && typeof modes === 'object') code.modes = modes;
    } catch { /* unreadable settings: defaults */ }
    const saveCodeSettings = () => {
      try {
        localStorage.setItem(CODE_SETTINGS_KEY, JSON.stringify({ autoCheck: code.autoCheck }));
        localStorage.setItem(CODE_MODES_KEY, JSON.stringify(code.modes));
      } catch { /* storage full */ }
    };
    const agentName = () => (code.perm === 'plan' ? 'plan' : 'build');

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

    const codeModelName = () => state.selectedModel || state.status?.model?.name || '';
    function codeSystemPrompt() {
      const name = codeModelName();
      return buildSystemPrompt({
        agent: agentName(),
        tier: codePromptTier(name, ctxMax()),
        family: codeModelFamily(name, state.status?.model?.chat_template || state.status?.model?.template),
        env: codeEnvironment({ ...(code.workspace || {}), git_status: code.git }),
        files: code.instructions,
        memory: code.memory,
        tools: new Set(code.registry.keys()),
      });
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
      await Promise.all([loadInstructions(), loadRegistry(), loadCustomCommands(), loadGitState()]);
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

    /** Project instruction files (workspace and parent folders) and the two memories, from the server. */
    async function loadInstructions() {
      code.instructions = [];
      code.memory = { workspace: [], global: [] };
      try {
        const res = await apiFetch('/api/studio/agent/instructions', { method: 'POST', body: {} });
        if (res.ok) {
          const d = await res.json();
          code.instructions = Array.isArray(d.files) ? d.files : [];
          code.memory = { workspace: d.memory?.workspace || [], global: d.memory?.global || [] };
        }
      } catch { /* an older server: no instructions, no memory */ }
      if (!code.memDraft) renderCodeSide();
    }

    /** One line about the working tree for the system prompt. */
    async function loadGitState() {
      code.git = '';
      if (!code.workspace?.git_branch) return;
      const r = await toolCall('bash', { command: 'git status --porcelain=v1 -b', timeout_ms: 8000 }).catch(() => null);
      if (r?.ok) code.git = summariseGit(r.output);
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
    const sessionTitleNow = () => code.title || heuristicTitle(code.messages);
    function readSessions() {
      try { return JSON.parse(localStorage.getItem(CODE_SESSIONS_KEY) || '{}'); } catch { return {}; }
    }
    function writeSessions(all) {
      try {
        let text = JSON.stringify(all);
        const mine = all[code.workspace?.root] || [];
        while (text.length > 4_000_000 && mine.length > 1) { mine.pop(); text = JSON.stringify(all); }
        if (text.length <= 4_500_000) { localStorage.setItem(CODE_SESSIONS_KEY, text); return true; }
      } catch { /* the browser storage is full: the session is only kept while the page is open */ }
      return false;
    }
    const workspaceSessions = () => (code.workspace ? readSessions()[code.workspace.root] || [] : []);
    const MAX_SESSIONS = 30;

    function saveCodeSession() {
      if (!code.workspace || (!code.messages.length && !code.view.length)) return;
      if (!code.sessionId) code.sessionId = cid();
      const all = readSessions();
      const mine = (all[code.workspace.root] || []).filter((s) => s.id !== code.sessionId);
      const view = clipForSave(code.view.filter((i) => !(i.kind === 'notice' && i.choices)));
      mine.unshift({ id: code.sessionId, title: sessionTitleNow(), titleLocked: code.titleLocked, time: Date.now(), messages: code.messages, view, todos: code.todos, used: code.used });
      all[code.workspace.root] = mine.slice(0, MAX_SESSIONS);
      writeSessions(all);
    }

    const heal = (items) => items.map((i) => {
      const c = (i.status === 'running' || i.status === 'pending') ? { ...i, status: 'error', output: 'Interrupted.' } : { ...i, streaming: false };
      if (c.children) c.children = heal(c.children);
      return c;
    });

    function resetSessionState() {
      code.messages = []; code.view = []; code.todos = []; code.changed = new Map(); code.used = 0; code.lastUsed = 0; code.compactedAt = 0;
      code.sessionId = null; code.jobs = []; code.alwaysAllow = new Set(); code.title = ''; code.titleLocked = false; code.queue = []; code.reverted = null;
      code.history = [];
    }

    function applySession(s) {
      resetSessionState();
      code.sessionId = s.id;
      // a session saved while the agent was working still has calls without results: make it valid for every template
      code.messages = repairMessages(s.messages || []).messages;
      code.view = heal(s.view || []);
      code.todos = s.todos || [];
      code.used = s.used || 0;
      code.title = s.title || '';
      code.titleLocked = Boolean(s.titleLocked);
      reindexOwners(code.view);
    }

    function reindexOwners(items) {
      for (const i of items) for (const c of i.children || []) { ownerOf.set(c, i); reindexOwners([c]); }
    }

    function restoreCodeSession() {
      resetSessionState();
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

    async function renameSession(id) {
      const all = readSessions();
      const list = all[code.workspace.root] || [];
      const s = id === code.sessionId ? null : list.find((x) => x.id === id);
      const current = s ? s.title : sessionTitleNow();
      const name = await askDialog({ title: 'Rename session', input: current, ok: 'Rename' });
      const title = cleanTitle(name);
      if (!title) return;
      if (s) { s.title = title; s.titleLocked = true; writeSessions(all); } else { code.title = title; code.titleLocked = true; saveCodeSession(); }
      renderCodeSide();
    }

    async function deleteSession(id) {
      const ok = await askDialog({ title: 'Delete this session?', text: 'The conversation is removed from this browser. File changes stay as they are.', ok: 'Delete', danger: true });
      if (!ok) return;
      const all = readSessions();
      all[code.workspace.root] = (all[code.workspace.root] || []).filter((s) => s.id !== id);
      writeSessions(all);
      if (id === code.sessionId) { if (code.running) stopAgent(); resetSessionState(); renderCode(); } else renderCodeSide();
    }

    const deepCopy = (v) => JSON.parse(JSON.stringify(v));

    /** A new session that continues from before the user message `itemId` (its text goes back to the input), or from the end. */
    function forkSession(itemId) {
      if (code.running) { toast('Stop the agent first.'); return; }
      let part = { view: code.view, messages: code.messages, text: '', todos: code.todos };
      if (itemId) {
        part = cutBefore(code.view, code.messages, itemId);
        if (!part) { toast('That part of the conversation was compacted. Fork from a later message.'); return; }
      }
      saveCodeSession();
      const parent = sessionTitleNow();
      const view = deepCopy(clipForSave(part.view.filter((i) => !(i.kind === 'notice' && i.choices))));
      const messages = deepCopy(part.messages);
      const todos = deepCopy(part.todos || []);
      const title = `${parent.replace(/ \(fork\)$/, '')} (fork)`.slice(0, 64);
      resetSessionState();
      code.sessionId = cid();
      code.title = title; code.titleLocked = true;
      code.view = view; code.messages = messages; code.todos = todos;
      reindexOwners(code.view);
      saveCodeSession();
      renderCode();
      const ta = $('codeTextarea');
      if (part.text) { ta.value = part.text; ta.focus(); }
      codeNotice(`Forked from "${parent}". The original session is unchanged.`);
    }

    function exportSession(id) {
      const s = id && id !== code.sessionId ? workspaceSessions().find((x) => x.id === id) : { title: sessionTitleNow(), time: Date.now(), view: code.view };
      if (!s || !(s.view || []).length) { toast('Nothing to export.'); return; }
      download(`${fileSafe(s.title)}.md`, 'text/markdown', exportMarkdown(s, code.workspace?.root));
    }

    /** The model gives the session a short title after its first answer (a plain title from the first message until then). */
    async function generateTitle() {
      if (code.titleLocked || code.titling || code.title || !code.sessionId) return;
      const first = code.messages.find((m) => m.role === 'user' && !String(m.content).startsWith('[Summary'));
      if (!first) return;
      const id = code.sessionId;
      code.titling = true;
      let title = '';
      try {
        const body = {
          model: codeModelName() || 'tiny',
          messages: [
            { role: 'system', content: 'You write very short titles for coding sessions.' },
            { role: 'user', content: `Write a title of at most six words for a coding session that starts with this request. Reply with the title only, without quotes.\n\n${heuristicTitle([first])}` },
          ],
          stream: false, temperature: 0.2, max_tokens: 400,
        };
        const res = await apiFetch('/v1/chat/completions', { method: 'POST', body });
        if (res.ok) title = cleanTitle(splitReasoning(String((await res.json()).choices?.[0]?.message?.content || '')).answer);
      } catch { /* the heuristic title stays */ } finally { code.titling = false; }
      if (title && id === code.sessionId && !code.titleLocked) {
        code.title = title;
        saveCodeSession();
        renderCodeSide();
      }
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

    /** Takes the conversation back to before a user message and puts the files back as they were then. */
    async function revertToMessage(itemId) {
      if (code.running) { stopAgent(); await code.runChain; }
      const cut = cutBefore(code.view, code.messages, itemId);
      if (!cut) { toast('That part of the conversation was compacted. Revert to a later message.'); return; }
      await refreshChanges();
      const first = code.changes ? firstChangeAfter(code.changes, cut.changeMark) : null;
      let restored = 0;
      let failed = '';
      if (first !== null) {
        const r = await apiFetch('/api/studio/agent/undo', { method: 'POST', body: { to: first } }).then((x) => x.json()).catch((e) => ({ ok: false, error: e.message }));
        if (r.ok === false) failed = r.error?.message || r.error || 'The files could not be put back.';
        else restored = (r.restored || []).length;
      }
      if (failed) { codeNotice(`Revert stopped: ${failed}`); refreshChanges(); return; }
      code.reverted = { view: code.view, messages: code.messages, todos: code.todos, restored, title: code.title };
      code.view = cut.view; code.messages = cut.messages; code.todos = cut.todos; code.compactedAt = 0; code.used = 0; code.lastUsed = 0;
      saveCodeSession();
      renderCode();
      const ta = $('codeTextarea');
      ta.value = cut.text; ta.focus();
      pushChoices(`Reverted to before that message${restored ? ` and put ${restored} file change${restored === 1 ? '' : 's'} back` : ''}. Its text is in the input.`, 'unrevert', [{ label: 'Undo the revert', data: 'unrevert' }]);
      refreshChanges();
    }

    async function unrevert() {
      const r = code.reverted;
      if (!r) { toast('Nothing to restore.'); return; }
      if (code.running) { toast('Stop the agent first.'); return; }
      if (r.restored) {
        const res = await apiFetch('/api/studio/agent/undo', { method: 'POST', body: { redo: r.restored } }).then((x) => x.json()).catch((e) => ({ ok: false, error: e.message }));
        if (res.ok === false) { codeNotice(`The files could not be restored: ${res.error?.message || res.error}`); return; }
      }
      code.view = r.view; code.messages = r.messages; code.todos = r.todos; code.reverted = null;
      $('codeTextarea').value = '';
      saveCodeSession();
      renderCode();
      refreshChanges();
      codeNotice('The revert was undone.');
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
      todo_read: () => ['Read task list', ''],
      ask_user: (a) => ['Question', a.question || ''],
      spawn_agent: (a) => ['Sub-agent', a.description || ''],
      task: (a) => [`Sub-agent (${a.agent === 'general' || a.mode === 'full' ? 'general' : 'explore'})`, a.description || ''],
      patch: (a) => ['Apply patch', patchPaths(a.patch).join(', ')],
      diagnostics: (a) => ['Check code', a.path || 'project'],
      memory: (a) => [`Memory ${a.action || 'list'}`, `${a.scope || 'workspace'}${a.text ? `: ${a.text}` : ''}`],
    };
    const EDIT_TOOLS = new Set(['write_file', 'edit_file', 'multi_edit', 'patch']);
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
      else if (item.name === 'patch') {
        String(a.patch || '').split('\n').forEach((l) => {
          if (l.startsWith('+++ ') || l.startsWith('--- ') || l.startsWith('@@')) rows.push(['more', l]);
          else if (l.startsWith('+')) rows.push(['add', l.slice(1)]);
          else if (l.startsWith('-')) rows.push(['del', l.slice(1)]);
          else rows.push(['ctx', l.slice(1)]);
        });
      }
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
      || (item.name === 'bash' && item.output && item.status === 'error') || (item.status === 'running' && item.name === 'bash') || (isTask(item.name) && item.status === 'running') || Boolean(item.check);

    function toolHtml(item) {
      const [verb, target] = toolLabel(item);
      let body = '';
      if (item.name === 'bash') body += `<pre class="tool-cmd">$ ${escapeHtml(item.args?.command || '')}</pre>`;
      body += diffHtml(item);
      if (item.name === 'todo_write' && Array.isArray(item.args?.todos)) body += todoListHtml(item.args.todos);
      else if (isTask(item.name)) {
        body += `<pre class="tool-cmd sub-prompt">${escapeHtml(String(item.args?.prompt || '').slice(0, 1500))}</pre>`;
        body += `<div class="tool-children">${(item.children || []).map(itemHtml).join('') || '<div class="side-empty">Starting ...</div>'}</div>`;
        if (item.output) body += `<div class="sub-report"><h5>Report</h5>${MD.render(String(item.output))}</div>`;
      } else if (item.name === 'ask_user') {
        if (item.output) body += `<pre class="tool-out">${escapeHtml(item.output)}</pre>`;
      } else if (item.output) body += outputHtml(item);
      const hasDetail = body || item.status === 'pending';
      const open = (item.open !== undefined ? item.open : defaultOpen(item)) ? ' open' : '';
      const ask = item.status === 'pending' ? (item.name === 'ask_user' ? askUserHtml(item) : permissionHtml(item)) : '';
      const progress = isTask(item.name) && item.status === 'running' && item.progress ? `<span class="tool-progress">${escapeHtml(item.progress)}</span>` : '';
      if (item.check) body += `<div class="tool-check"><h5>Automatic check</h5><pre class="tool-out">${escapeHtml(item.check)}</pre></div>`;
      return `
        <div class="tool-card is-${item.status}${isTask(item.name) ? ' is-agent' : ''}" data-item="${item.id}">
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
      const cat = entry.source === 'mcp' ? 'mcp' : entry.permission === 'edit' ? 'edit' : entry.permission === 'network' ? 'network' : entry.permission === 'command' ? 'bash' : entry.permission === 'memory' ? 'memory' : 'read';
      let subject = '';
      let subjects = null;
      if (cat === 'bash') subject = typeof args.command === 'string' ? args.command : `${entry.name} ${Object.values(args).join(' ')}`.trim();
      else if (cat === 'edit' && typeof args.patch === 'string') { subjects = patchPaths(args.patch); subject = subjects[0] || ''; }
      else if (cat === 'edit' || cat === 'read') subject = String(args.path ?? args.pattern ?? '');
      else if (cat === 'network') subject = String(args.url ?? args.query ?? '');
      else if (cat === 'memory') subject = String(args.scope || 'workspace');
      else subject = `${entry.server}/${entry.tool}`;
      return { cat, subject, subjects: subjects && subjects.length ? subjects : [subject] };
    }

    /** What "Always" would remember for this call, or null when it can only be allowed once. */
    function alwaysKey(item) {
      const entry = code.registry.get(item.name);
      if (!entry) return null;
      const { cat, subject } = permissionInfo(entry, item.args || {});
      if (cat === 'edit') return 'edit';
      if (cat === 'bash') { const k = entry.name === 'bash' ? commandKey(subject) : entry.name; return k ? `bash:${k}` : null; }
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
      if (item.kind === 'user') {
        const turn = Number.isInteger(item.msgIndex) && !code.running
          ? `<div class="turn-actions"><button class="btn btn-xs btn-ghost" data-turn-fork="${item.id}" title="Start a new session from before this message">Fork</button><button class="btn btn-xs btn-ghost" data-turn-revert="${item.id}" title="Go back to before this message: the conversation and the file changes since then are taken back">Revert</button></div>`
          : '';
        return `<div class="code-user" data-item="${item.id}"><div class="msg-user-bubble">${escapeHtml(item.text)}</div>${turn}</div>`;
      }
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
          ${code.instructions.length ? '' : '<button class="starter-pill" data-code-command="/init">Create AGENTS.md</button>'}
        </div>
        ${past.length ? `<p class="code-fine">Earlier sessions:</p><div class="code-recent">${past.map((s) => `<button class="starter-pill" data-resume="${s.id}" title="${escapeHtml(new Date(s.time).toLocaleString())}">${escapeHtml(s.title)}</button>`).join('')}</div>` : ''}
        ${code.instructions.length ? `<p class="code-fine">Project instructions read: ${escapeHtml(code.instructions.map((f) => f.file).join(', '))}</p>` : ''}
        ${code.memory.workspace.length + code.memory.global.length ? `<p class="code-fine">${code.memory.workspace.length + code.memory.global.length} saved memory entries are part of the prompt (see Memory on the right).</p>` : ''}
      </div>`;
    }

    function renderCode() {
      $('codeColumn').innerHTML = code.view.length ? code.view.map(itemHtml).join('') : emptyCodeHtml();
      if (!code.view.length && !code.workspace) setTimeout(() => $('wsPathInput')?.focus(), 30);
      renderCodeSide();
      if (state.tab === 'code') renderWorkspaceList();
      updateCodeMeter();
      syncAgentControls();
      renderQueue();
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

    const MODE_LABEL = { read: 'Read and search', edit: 'Change files', bash: 'Commands', network: 'Web', mcp: 'External (MCP) tools', memory: 'Memory' };

    function permissionsHtml() {
      const modeOf = (agent, cat) => code.modes?.[agent]?.[cat] || '';
      const defaultOf = (agent, cat) => modeFor({}, agent, cat, agent === 'build' && code.perm === 'edit');
      const cell = (agent, cat) => {
        const chosen = modeOf(agent, cat);
        return `<select class="ui-select mode-select" data-mode="${agent}:${cat}" aria-label="${MODE_LABEL[cat]} for the ${agent} agent">
          <option value=""${chosen ? '' : ' selected'}>default (${defaultOf(agent, cat)})</option>
          ${['allow', 'ask', 'deny'].map((m) => `<option value="${m}"${chosen === m ? ' selected' : ''}>${m}</option>`).join('')}</select>`;
      };
      return `<table class="mode-table"><thead><tr><th></th>${AGENTS.map((a) => `<th>${a}</th>`).join('')}</tr></thead>
        <tbody>${MODE_CATEGORIES.map((cat) => `<tr><td>${MODE_LABEL[cat]}</td>${AGENTS.map((a) => `<td>${cell(a, cat)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    }

    function sessionsHtml() {
      const all = workspaceSessions();
      const shown = filterSessions(all, code.sessionQuery).slice(0, 12);
      const search = all.length > 3 ? `<input type="search" class="ui-input session-search" data-session-search placeholder="Search sessions" value="${escapeHtml(code.sessionQuery)}">` : '';
      if (!shown.length) return `${search}<p class="side-empty">${all.length ? 'No session matches.' : 'Sessions of this folder are saved automatically.'}</p>`;
      const act = (a, id, label, title) => `<button class="btn btn-xs btn-ghost" data-session-act="${a}:${id}" title="${title}">${label}</button>`;
      return `${search}<ul class="session-list">${shown.map((s) => `<li class="${s.id === code.sessionId ? 'active' : ''}">
        <button class="session-title" data-resume="${s.id}" title="${escapeHtml(new Date(s.time).toLocaleString())}">${escapeHtml(s.id === code.sessionId ? sessionTitleNow() : s.title)}</button>
        <span class="session-acts">${act('rename', s.id, 'Rename', 'Rename this session')}${act('fork', s.id, 'Fork', 'New session with a copy of this one')}${act('export', s.id, 'Export', 'Save as a Markdown file')}${act('delete', s.id, 'Delete', 'Delete this session')}</span></li>`).join('')}</ul>`;
    }

    function memoryHtml() {
      const draft = code.memDraft || {};
      const text = (scope) => (draft[scope] !== undefined ? draft[scope] : (code.memory[scope] || []).join('\n'));
      return `<p class="side-empty">Short facts the agent keeps between sessions, one per line. They are part of its prompt. The agent adds entries itself when it learns something lasting; edit or delete them here.</p>
        <label class="rules-label">This workspace <span class="mono">(.fllm/memory.md)</span></label>
        <textarea class="ui-textarea mono rules-text" data-memory="workspace" rows="5" placeholder="Tests run with: npm test">${escapeHtml(text('workspace'))}</textarea>
        <label class="rules-label">Everywhere (this user)</label>
        <textarea class="ui-textarea mono rules-text" data-memory="global" rows="4" placeholder="Prefers short answers">${escapeHtml(text('global'))}</textarea>
        <button class="btn btn-sm btn-secondary" data-memory-save>Save memory</button>${code.memDraft ? ' <button class="btn btn-sm btn-ghost" data-memory-reset>Discard changes</button>' : ''}`;
    }

    function renderCodeSide(force) {
      const side = $('codeSide');
      // do not take the cursor away while someone types in a field of this panel
      const active = document.activeElement;
      if (!force && active && side.contains(active) && /^(TEXTAREA|INPUT|SELECT)$/.test(active.tagName)) return;
      const done = code.todos.filter((t) => t.status === 'completed').length;
      const files = code.changes ? code.changes.map((c) => [c.path, c.kind]) : [...code.changed.entries()].map(([p, n]) => [p, n > 1 ? `${n}×` : '']);
      const uniqueFiles = [...new Map(files.map(([p, k]) => [p, k])).entries()];
      const draft = code.rulesDraft || code.rules;
      const nRules = parseRules(code.rules.allow).length + parseRules(code.rules.deny).length;
      const nMemory = code.memory.workspace.length + code.memory.global.length;
      const open = (k) => (code.sideOpen[k] ? ' open' : '');
      side.innerHTML = `
        <section class="side-block"><h4>Tasks${code.todos.length ? `<span class="side-count">${done}/${code.todos.length}</span>` : ''}</h4>
          ${code.todos.length ? todoListHtml(code.todos) : '<p class="side-empty">For bigger jobs the agent keeps a task list here.</p>'}</section>
        <section class="side-block"><h4>Changed files${uniqueFiles.length ? `<span class="side-count">${uniqueFiles.length}</span>` : ''}</h4>
          ${uniqueFiles.length ? `<ul class="changed-list">${uniqueFiles.slice(-30).map(([p, k]) => `<li><span class="mono">${escapeHtml(p)}</span>${k ? `<span class="side-count">${escapeHtml(k)}</span>` : ''}</li>`).join('')}</ul>${code.changes ? '<button class="btn btn-xs btn-ghost" data-code-command="/rewind">Rewind ...</button>' : ''}` : '<p class="side-empty">Nothing changed yet.</p>'}</section>
        <section class="side-block"><h4>Background jobs${code.jobs.length ? `<span class="side-count">${code.jobs.length}</span>` : ''}</h4>${jobsHtml()}</section>
        <section class="side-block"><h4>Sessions</h4>${sessionsHtml()}</section>
        <section class="side-block"><details class="rules-box" data-k="memory"${open('memory')}><summary><h4>Memory${nMemory ? `<span class="side-count">${nMemory}</span>` : ''}</h4></summary>${memoryHtml()}</details></section>
        <section class="side-block"><details class="rules-box" data-k="perms"${open('perms')}><summary><h4>Permissions${nRules ? `<span class="side-count">${nRules}</span>` : ''}</h4></summary>
          <p class="side-empty">What each agent may do without asking. Pattern rules below refine this: deny beats everything, an allow rule skips the question.</p>
          ${permissionsHtml()}
          <p class="side-empty">One rule per line: <span class="mono">category: pattern</span>, optionally for one agent: <span class="mono">[plan] bash: ls*</span>. Categories: bash, edit, network, mcp, read, memory. Risky programs need a rule that names them, for example <span class="mono">bash: git status*</span>.</p>
          <label class="rules-label">Always allow</label><textarea class="ui-textarea mono rules-text" data-rules="allow" rows="4" placeholder="bash: npm test*&#10;edit: src/**&#10;network: *">${escapeHtml(draft.allow)}</textarea>
          <label class="rules-label">Always deny</label><textarea class="ui-textarea mono rules-text" data-rules="deny" rows="3" placeholder="bash: rm *&#10;edit: .env">${escapeHtml(draft.deny)}</textarea>
          <button class="btn btn-sm btn-secondary" data-rules-save>Save rules</button></details></section>
        <section class="side-block"><details class="rules-box" data-k="settings"${open('settings')}><summary><h4>Settings</h4></summary>
          <label class="check-row"><input type="checkbox" data-auto-check${code.autoCheck ? ' checked' : ''}> Check my edits automatically</label>
          <p class="side-empty">After every edit the agent runs a quick syntax or build check (python, node, cargo check, tsc, go vet, JSON) and gets the first errors with the result. Slower, but it catches mistakes at once.</p></details></section>`;
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
      $('codeHint').textContent = busy ? 'The agent is working. Enter queues your message, Esc stops it.' : 'Enter sends, Shift+Enter adds a new line';
    }

    // ---- permissions ----------------------------------------------------------------------------------
    /** 'deny' (a rule says no), 'allow', or 'ask'. */
    function authorize(entry, args) {
      if (entry.source === 'client') return 'allow';
      const { cat, subjects } = permissionInfo(entry, args);
      const rules = parsedRules();
      const results = subjects.map((subject) => {
        let remembered = false;
        if (cat === 'bash') { const key = entry.name === 'bash' ? commandKey(subject) : entry.name; remembered = Boolean(key) && code.alwaysAllow.has(`bash:${key}`); }
        else if (cat === 'network') remembered = code.alwaysAllow.has('network');
        else if (cat === 'mcp') remembered = code.alwaysAllow.has(`mcp:${subject}`);
        return decidePermission({ rules, modes: code.modes, agent: agentName(), autoEdit: code.perm === 'edit', category: cat, subject, remembered });
      });
      if (results.includes('deny')) return 'deny';
      return results.includes('ask') ? 'ask' : 'allow';
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

    /**
     * Small models often answer with the call itself as plain JSON ({"name": "list_dir", "arguments": {...}}), alone or in a code fence,
     * without the markers of their template. When the whole answer is such an object (or a list of them) and the name is a tool we offered,
     * it is a call.
     */
    function extractBareJsonCalls(content, toolNames) {
      const text = String(content || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
      if (!text || (text[0] !== '{' && text[0] !== '[')) return [];
      let value;
      try { value = JSON.parse(text); } catch { return []; }
      const calls = [];
      for (const item of Array.isArray(value) ? value : [value]) {
        const fn = item && typeof item === 'object' && item.function && typeof item.function === 'object' ? item.function : item;
        const name = fn && typeof fn.name === 'string' ? fn.name : '';
        const args = fn ? (fn.arguments !== undefined ? fn.arguments : fn.parameters) : undefined;
        if (!name || !toolNames.includes(name) || (args !== undefined && args !== null && typeof args !== 'object' && typeof args !== 'string')) return [];
        calls.push({ id: `call_${cid()}`, name, args: typeof args === 'string' ? args : JSON.stringify(args || {}) });
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
      if (!found.length && item.content && !item.content.includes('<tool_call>')) {
        const names = (ctx.tools || []).map((d) => (d.function ? d.function.name : d.name)).filter(Boolean);
        const bare = extractBareJsonCalls(item.content, names);
        if (bare.length) { found = bare; item.content = ''; }
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
      // the start and the end are kept: build and test output has its errors and its summary at the end
      return clipMiddle(text, Math.max(3000, Math.floor(ctxMax() * 3.4 * 0.22)));
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

    const isTask = (name) => name === 'task' || name === 'spawn_agent'; // spawn_agent: the name in older sessions
    const NO_LOOP_GUARD = new Set(['bash_output', 'todo_read', 'todo_write', 'ask_user', 'task', 'spawn_agent', 'memory']);
    const CHECKED_EXTENSIONS = new Set(['py', 'js', 'mjs', 'cjs', 'ts', 'tsx', 'rs', 'go', 'json']);

    /** The first problems of a quick check of the files an edit changed (setting "check my edits automatically"). */
    async function autoCheckEdit(name, args, ctx) {
      if (!code.autoCheck || ctx.readOnly || !serverHas('diagnostics') || modeFor(code.modes, agentName(), 'bash', false) === 'deny') return '';
      const paths = name === 'patch' ? patchPaths(args.patch) : (args.path ? [args.path] : []);
      const seen = new Set();
      const reports = [];
      for (const p of paths.slice(0, 4)) {
        const ext = (String(p).split('.').pop() || '').toLowerCase();
        if (!CHECKED_EXTENSIONS.has(ext) || seen.has(ext)) continue;
        if (evaluateRules(parsedRules(), 'bash', `diagnostics ${p}`, agentName()) === 'deny') continue;
        seen.add(ext);
        const r = await toolCall('diagnostics', { path: p, timeout_s: 60 }, ctx.signal).catch(() => null);
        if (r?.ok && !/^No problems found/.test(r.output) && !/is not available here|did not finish within/.test(r.output)) reports.push(r.output);
      }
      return reports.join('\n');
    }

    /** Runs one tool call and returns the text the model gets back. ctx: { view, root, depth, readOnly, signal, isStopped, history }. */
    async function runToolCall(call, ctx) {
      const args = parseArgs(call.args);
      const item = newItem(ctx, { kind: 'tool', callId: call.id, name: call.name, args: args || {}, status: 'running', output: '' });
      const entry = code.registry.get(call.name) || (call.name === 'spawn_agent' ? code.registry.get('task') : null);
      const finish = (status, out, reply) => { item.status = status; item.output = out; paintItem(item); return clipOutput(reply ?? `Error: ${out}`); };
      if (args === null) return finish('error', 'The arguments are not valid JSON.', 'Error: the arguments were not valid JSON. Call the tool again with a valid JSON object.');
      if (!entry) return finish('error', `Unknown tool ${call.name}`, `Error: unknown tool '${call.name}'. Available tools: ${[...code.registry.keys()].join(', ')}.`);
      if (ctx.depth > 0 && entry.source === 'client') return finish('denied', 'Not available to sub-agents.', `Denied: ${call.name} is not available to a sub-agent. Do the work yourself and put the answer in your report.`);
      if (ctx.readOnly && !readOnlyTool(entry)) return finish('denied', 'This sub-agent is read-only.', 'Denied: this sub-agent is read-only. Describe the change in your report instead.');
      if (code.perm === 'plan' && !readOnlyTool(entry) && entry.source !== 'client') return finish('denied', 'Plan mode only allows reading.', 'Denied: plan mode only allows reading. Describe the change in your plan instead.');

      // the same call again and again: the model is stuck, say so instead of running it a fourth time
      const signature = callSignature(call.name, args);
      const history = ctx.history || (ctx.history = []);
      const looping = !NO_LOOP_GUARD.has(call.name) && isRepeating(history, signature, 3);
      history.push(signature);
      if (history.length > 8) history.shift();
      if (looping) return finish('error', 'The same call was repeated three times.', 'Error: you have made this exact call three times in a row. Repeating it will not change the result. Try a different approach (another tool, other arguments, a smaller step), or ask the user with ask_user.');

      if (call.name === 'todo_write') {
        code.todos = (Array.isArray(args.todos) ? args.todos : []).filter((t) => t && t.content).map((t) => ({ content: String(t.content), status: ['pending', 'in_progress', 'completed'].includes(t.status) ? t.status : 'pending' }));
        item.args = { todos: code.todos };
        renderCodeSide();
        return finish('ok', '', 'Todos updated.');
      }
      if (call.name === 'todo_read') {
        const text = code.todos.length ? code.todos.map((t) => `- [${t.status}] ${t.content}`).join('\n') : 'The task list is empty.';
        return finish('ok', text, text);
      }
      if (call.name === 'ask_user') {
        if (!args.question) return finish('error', 'No question.', 'Error: ask_user needs a question.');
        paintItem(item);
        const answer = await waitFor(item, null);
        if (answer === null || ctx.isStopped()) return finish('denied', 'Stopped.', 'The user did not answer.');
        return finish('ok', `Answer: ${answer}`, `The user answered: ${answer}`);
      }
      if (isTask(call.name)) return runSubAgent(item, args, ctx);

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
        let reply = r.ok ? r.output : `Error: ${r.error}`;
        if (r.ok && EDIT_TOOLS.has(call.name)) {
          for (const p of (call.name === 'patch' ? patchPaths(args.patch) : [args.path]).filter(Boolean)) code.changed.set(p, (code.changed.get(p) || 0) + 1);
          renderCodeSide();
          refreshChanges();
          const problems = await autoCheckEdit(call.name, args, ctx);
          if (problems) {
            item.check = problems;
            reply = `${clipOutput(reply)}\n\n[automatic check found problems]\n${clipOutput(problems)}\nFix them if your change caused them.`;
          }
        }
        if (r.ok && call.name === 'memory') loadInstructions();
        if (r.ok) trackJob(call.name, args, r.output);
        paintItem(item);
        return clipOutput(reply);
      } catch (e) {
        return finish('error', e.name === 'AbortError' ? 'Stopped.' : e.message);
      }
    }

    // ---- sub-agents ------------------------------------------------------------------------------------------
    async function runSubAgent(item, args, parent) {
      const prompt = String(args.prompt || '').trim();
      if (!prompt) { item.status = 'error'; item.output = 'No prompt.'; paintItem(item); return 'Error: task needs a prompt.'; }
      if (parent.depth > 0) { item.status = 'denied'; item.output = 'Sub-agents cannot start sub-agents.'; paintItem(item); return 'Denied: sub-agents cannot start sub-agents.'; }
      args.description = String(args.description || 'Sub-task').slice(0, 80);
      // "explore" reads only; "general" may change things unless the plan agent is working. (older calls said mode: "full")
      const type = (args.agent === 'general' || args.mode === 'full') && code.perm !== 'plan' ? 'general' : 'explore';
      item.agentType = type;
      const readOnly = type === 'explore';
      item.children = [];
      item.status = 'running';
      item.progress = 'starting';
      paintItem(item);
      const max = ctxMax();
      const ctx = {
        system: parent.system, tools: parent.tools, messages: [{ role: 'user', content: subAgentPrompt(args, type) }],
        view: item.children, root: item, depth: 1, readOnly, signal: parent.signal, isStopped: parent.isStopped,
        maxSteps: CHILD_MAX_STEPS, budget: Math.floor(max * 0.75), onUsage: null, child: true, history: [],
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

    /** After compaction or trimming the message list no longer lines up with the turns: fork and revert are not offered for them. */
    function forgetTurnIndexes() {
      for (const i of code.view) if (i.kind === 'user') delete i.msgIndex;
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
      forgetTurnIndexes();
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
          if (dropped) { forgetTurnIndexes(); codeNotice('Older steps were removed from the context of the model so that it does not overflow. You still see them here.'); }
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
          if (isTask(calls[i].name)) {
            let j = i;
            while (j < calls.length && isTask(calls[j].name) && j - i < MAX_PARALLEL_CHILDREN) j++;
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

    /** One run of the agent for a user message. Runs go one after the other, so a message sent while the previous run is still winding down waits. */
    function agentRun(userText, display) {
      const next = code.runChain.then(() => agentRunNow(userText, display));
      code.runChain = next.catch(() => {});
      return next;
    }

    async function agentRunNow(userText, display) {
      if (!code.workspace) { await pickWorkspace(); if (!code.workspace) return; }
      code.reverted = null;
      const expanded = await expandMentions(takePending() + userText);
      await refreshChanges();
      const mark = code.changes ? Math.max(0, ...code.changes.map((c) => c.id)) : 0;
      // the new message is appended to a conversation that is valid for every template; its index is kept for fork and revert
      const messages = repairMessages([...code.messages, { role: 'user', content: expanded }]).messages;
      code.messages = messages;
      const turn = { kind: 'user', id: cid(), text: display || userText, msgIndex: messages.length - 1, changeMark: mark, todosBefore: code.todos.map((t) => ({ ...t })) };
      code.view.push(turn);
      renderCode();
      code.abort = new AbortController();
      setCodeBusy(true);
      const ctx = {
        system: codeSystemPrompt(), tools: toolList(), messages: code.messages, view: code.view, root: null, depth: 0, readOnly: false,
        signal: code.abort.signal, isStopped: () => !code.running, maxSteps: CODE_MAX_TURNS, child: false, history: [],
        onUsage: (n) => { code.used = n; code.lastUsed = n; },
      };
      let stopped = false;
      try {
        await runLoop(ctx);
        const lastItem = [...code.view].reverse().find((i) => i.kind === 'assistant');
        if (lastItem && code.perm === 'plan' && lastItem.content && code.view[code.view.length - 1] === lastItem) { lastItem.planDone = true; paintItem(lastItem); }
      } catch (e) {
        if (e.name !== 'AbortError') code.view.push({ kind: 'assistant', id: cid(), content: '', error: e.message });
      } finally {
        stopped = !code.running;
        cancelWaiters();
        code.abort = null;
        // an interrupted run leaves calls without results: close them so the next request is valid
        const fixed = repairMessages(code.messages);
        if (fixed.changed) code.messages = fixed.messages;
        setCodeBusy(false);
        saveCodeSession();
        renderCode();
        refreshChanges();
        refreshStudioStatus();
        loadInstructions();
        loadGitState();
      }
      if (stopped) {
        if (code.queue.length) { $('codeTextarea').value = code.queue.join('\n\n'); code.queue = []; renderQueue(); codeNotice('Stopped. The messages you had queued are back in the input.'); }
        return;
      }
      generateTitle();
      if (code.queue.length) {
        const next = code.queue.shift();
        renderQueue();
        // after this run has returned: the runs are chained, waiting for the chain here would wait for itself
        setTimeout(() => { submitText(next); }, 0);
      }
    }

    function stopAgent() {
      if (!code.running) return;
      code.running = false;
      code.abort?.abort();
      cancelWaiters();
      // commands the server started keep running after the browser gave up the request: end them
      apiFetch('/api/studio/agent/cancel', { method: 'POST', body: {} }).catch(() => {});
    }

    // ---- message queue ---------------------------------------------------------------------------------------------
    function renderQueue() {
      const box = $('codeQueue');
      box.style.display = code.queue.length ? 'flex' : 'none';
      box.innerHTML = code.queue.map((q, i) => `<span class="queue-chip" title="${escapeHtml(q)}"><span>${escapeHtml(q.replace(/\s+/g, ' ').slice(0, 60))}</span><button data-queue-drop="${i}" aria-label="Remove">×</button></span>`).join('');
    }

    // ---- @files, commands --------------------------------------------------------------------------------
    async function expandMentions(text) {
      const names = [...new Set([...text.matchAll(/(?:^|\s)@([^\s@]+)/g)].map((m) => m[1].replace(/[.,;:!?)]+$/, '').replace(/\\/g, '/')))].filter((n) => n.length > 1).slice(0, 5);
      if (!names.length) return text;
      let out = text;
      for (const name of names) {
        const r = await toolCall(name.endsWith('/') ? 'list_dir' : 'read_file', name.endsWith('/') ? { path: name } : { path: name, limit: 300 }).catch(() => null);
        if (r?.ok) out += `\n\nContents of @${name}:\n\`\`\`\n${r.output}\n\`\`\``;
      }
      return out;
    }

    // output of commands the user ran with "!" is given to the model with the next message
    const pending = [];
    function takePending() {
      if (!pending.length) return '';
      const text = `${pending.splice(0).join('\n\n')}\n\n`;
      return text;
    }

    async function userRun(command, share = false) {
      const item = { kind: 'tool', id: cid(), name: 'bash', args: { command }, status: 'running', output: '', byUser: true };
      code.view.push(item);
      paintItem(item);
      const r = await toolCall('bash', { command }).catch((e) => ({ ok: false, error: e.message }));
      item.status = r.ok ? 'ok' : 'error';
      item.output = r.ok ? r.output : r.error;
      paintItem(item);
      if (share) pending.push(`I ran this command myself:\n$ ${command}\n\`\`\`\n${clipMiddle(item.output, 4000)}\n\`\`\``);
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
      if (perm !== 'plan') code.lastBuildPerm = perm;
      localStorage.setItem('fllm.studio.perm', perm);
      syncAgentControls();
      if (!code.view.length) renderCode();
      updateCodeMeter();
    }

    function syncAgentControls() {
      const plan = code.perm === 'plan';
      $('agentSeg').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.agent === (plan ? 'plan' : 'build')));
      $('permSeg').querySelectorAll('button').forEach((b) => b.classList.toggle('active', !plan && b.dataset.perm === code.perm));
      $('permSeg').style.opacity = plan ? '0.45' : '';
      $('codeModelLabel').textContent = codeModelName() || 'Model';
    }

    /** Build and plan agent: Alt+B. */
    function toggleAgent() {
      if (code.running) { toast('Wait until the agent has finished or stop it.'); return; }
      setPerm(code.perm === 'plan' ? (code.lastBuildPerm || 'ask') : 'plan');
      codeNotice(code.perm === 'plan' ? 'Plan agent: it only reads and writes a plan.' : 'Build agent: it can change files and run commands (with the permissions you set).');
    }

    function newCodeSession() {
      if (code.running) stopAgent();
      saveCodeSession();
      resetSessionState();
      renderCode();
    }

    const INIT_PROMPT = 'Explore this workspace and write a file named AGENTS.md in its root with instructions for future coding sessions. Cover: what the project is, how to build, test and run it (exact commands, checked against the build files), the layout of the important folders, code style and conventions you can see, and pitfalls. Keep it under 80 lines, concrete, with no filler. If AGENTS.md, FLLM.md or CLAUDE.md already exists, read it and improve that file instead of starting over.';
    const BUILTIN_COMMANDS = [
      ['clear', 'Start a new session'], ['compact', 'Summarise older turns to free context'], ['init', 'Let the agent write AGENTS.md'], ['resume', 'Pick an earlier session'],
      ['fork', 'New session from an earlier message'], ['revert', 'Go back to before a message, files included'], ['rename', 'Rename this session'], ['export', 'Save this session as Markdown'],
      ['rewind', 'Undo file changes back to a chosen point'], ['undo', 'Undo the last file change (/undo 3 for three)'],
      ['plan', 'Plan agent: read only'], ['build', 'Build agent'], ['ask', 'Build agent, ask before every change'], ['edit', 'Build agent, auto-edit'], ['agent', 'Switch between build and plan'], ['model', 'Pick the model'],
      ['remember', 'Save a fact to memory (/remember global ...)'], ['memory', 'Show and edit the memory'], ['check', 'Check the code for errors (diagnostics)'],
      ['diff', 'git diff'], ['status', 'git status'], ['cd', 'Open another folder'], ['allow', 'Add an allow rule, e.g. /allow bash: npm test*'], ['deny', 'Add a deny rule'], ['rules', 'Show the permission rules'], ['help', 'List the commands'],
    ];

    /** Rows for a pick list of the user messages of this session, newest first. */
    function turnChoices() {
      return code.view.filter((i) => i.kind === 'user' && Number.isInteger(i.msgIndex)).reverse().slice(0, 15)
        .map((i) => ({ label: String(i.text).replace(/\s+/g, ' ').slice(0, 80), data: i.id }));
    }

    async function userTool(name, args) {
      const item = { kind: 'tool', id: cid(), name, args, status: 'running', output: '', byUser: true };
      code.view.push(item);
      paintItem(item);
      const r = await toolCall(name, args).catch((e) => ({ ok: false, error: e.message }));
      item.status = r.ok ? 'ok' : 'error';
      item.output = r.ok ? r.output : r.error;
      paintItem(item);
      saveCodeSession();
    }

    async function saveMemoryEntry(text, scope) {
      const res = await apiFetch('/api/studio/agent/memory', { method: 'POST', body: { scope, op: 'append', text } });
      if (!res.ok) { toast((await res.json().catch(() => ({}))).error?.message || 'The memory could not be saved.'); return false; }
      await loadInstructions();
      return true;
    }

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
        case 'resume': case 'sessions': {
          const rows = workspaceSessions().filter((s) => s.id !== code.sessionId).map((s) => ({ label: `${s.title}  (${new Date(s.time).toLocaleString()})`, data: s.id }));
          if (rows.length) pushChoices('Pick a session to continue:', 'resume', rows); else codeNotice('There is no earlier session for this folder.');
          return true;
        }
        case 'fork': {
          const rows = [{ label: 'The end of this conversation', data: '' }, ...turnChoices()];
          if (rows.length > 1) pushChoices('Fork: the new session continues from before the message you pick (its text goes back to the input).', 'fork', rows); else forkSession('');
          return true;
        }
        case 'revert': {
          const rows = turnChoices();
          if (rows.length) pushChoices('Revert: pick the message to go back to. The conversation and the file changes since then are taken back; you can undo the revert.', 'revert', rows); else codeNotice('There is no message to revert to.');
          return true;
        }
        case 'rename':
          if (argText) { code.title = cleanTitle(argText); code.titleLocked = true; saveCodeSession(); renderCodeSide(); } else await renameSession(code.sessionId);
          return true;
        case 'export': exportSession(code.sessionId); return true;
        case 'plan': case 'ask': case 'edit': setPerm(name); codeNotice({ plan: 'Plan agent: it only reads and writes a plan.', ask: 'Build agent: it asks before every change and every command.', edit: 'Build agent, auto-edit: it changes files itself and asks before commands.' }[name]); return true;
        case 'build': setPerm(code.lastBuildPerm || 'ask'); codeNotice('Build agent.'); return true;
        case 'agent': toggleAgent(); return true;
        case 'model': {
          const rows = modelNames().map((n) => ({ label: n, data: n }));
          if (rows.length) pushChoices('Pick the model for the agent:', 'model', rows); else codeNotice('No model is loaded.');
          return true;
        }
        case 'remember': {
          const global = /^global\b/i.test(argText);
          const entry = global ? argText.replace(/^global\b\s*/i, '') : argText;
          if (!entry) { codeNotice('Usage: /remember a short fact, or /remember global a fact for every workspace.'); return true; }
          if (await saveMemoryEntry(entry, global ? 'global' : 'workspace')) codeNotice(`Remembered (${global ? 'for every workspace' : 'for this workspace'}): ${entry}`);
          return true;
        }
        case 'memory': code.sideOpen.memory = true; code.memDraft = null; renderCodeSide(true); $('codeSide').querySelector('[data-k="memory"]')?.scrollIntoView({ block: 'nearest' }); return true;
        case 'check': await userTool('diagnostics', argText ? { path: argText } : {}); return true;
        case 'diff': await userRun('git diff --stat && git diff', true); return true;
        case 'status': await userRun('git status -sb', true); return true;
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
          codeNotice(`Commands: ${BUILTIN_COMMANDS.map(([n]) => `/${n}`).join(', ')}.${custom ? ` Custom commands from .fllm/commands: ${custom}.` : ''} @file attaches a file (fuzzy search), !command runs a shell command and shows the model its output, Enter while the agent works queues a message, Alt+B switches between the build and plan agent.`);
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

    const modelNames = () => Array.from($('modelDropdownItems').querySelectorAll('[data-pick-model]')).map((b) => b.dataset.pickModel);
    function pickModel(name) {
      state.selectedModel = name;
      $('activeModelLabel').textContent = name;
      $('codeModelMenu').classList.remove('open');
      syncAgentControls();
      refreshStudioStatus();
    }

    // ---- composer: @ and / completion ---------------------------------------------------------------------------
    let mentionTimer = null;
    let mentionItems = []; // { value, hint }
    let mentionIndex = 0;
    let mentionSlash = false;
    let mentionSeq = 0;
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
      else before = ta.value.slice(0, cursor).replace(/@[^\s@]*$/, `@${mentionItems[i].value} `);
      ta.value = before + ta.value.slice(cursor);
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
        const q = slash[1].toLowerCase();
        mentionItems = [...all.filter((c) => c.value.slice(1).startsWith(q)), ...all.filter((c) => !c.value.slice(1).startsWith(q) && (c.value.includes(q) || c.hint.toLowerCase().includes(q)))].slice(0, 8);
        mentionSlash = true;
        mentionIndex = 0;
        renderMention();
        return;
      }
      mentionSlash = false;
      const m = ta.value.slice(0, ta.selectionStart).match(/(?:^|\s)@([^\s@]*)$/);
      if (!m || !code.workspace) { closeMention(); return; }
      const seq = ++mentionSeq;
      mentionTimer = setTimeout(async () => {
        const query = m[1].replace(/\\/g, '/');
        let list = null;
        try {
          const res = await apiFetch('/api/studio/agent/files', { method: 'POST', body: { query, limit: 8 } });
          if (res.ok) list = (await res.json()).files;
        } catch { /* no file route */ }
        if (seq !== mentionSeq) return; // a newer keystroke is already asked
        if (!list) { closeMention(); return; }
        mentionItems = list.map((value) => ({ value, hint: value.endsWith('/') ? 'folder' : '' }));
        mentionIndex = 0;
        renderMention();
      }, 110);
    }

    /** One entered line: a shell command (!), a command (/) or a message for the agent. */
    async function submitText(text) {
      const shell = parseShellEscape(text);
      if (shell !== null) { if (shell) await userRun(shell, true); return; }
      if (text.startsWith('/') && (await runCodeCommand(text))) return;
      await agentRun(text);
    }

    async function submitCode(viaButton) {
      const ta = $('codeTextarea');
      const text = ta.value.trim();
      if (code.running) {
        if (viaButton || !text) { stopAgent(); return; }
        // the agent is busy: the message waits and is sent when it has finished
        code.queue.push(text);
        ta.value = '';
        ta.style.height = 'auto';
        closeMention();
        renderQueue();
        return;
      }
      if (!text) return;
      if (state.setup) { toast('Load a model first.'); switchSurface('hub'); return; }
      ta.value = '';
      ta.style.height = 'auto';
      closeMention();
      await submitText(text);
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
    $('btnCodeSend').addEventListener('click', () => submitCode(true));
    $('btnWorkspace').addEventListener('click', pickWorkspace);
    $('btnCodeUndo').addEventListener('click', () => undoChange());
    $('btnCodeNew').addEventListener('click', newCodeSession);
    $('permSeg').addEventListener('click', (e) => {
      const b = e.target.closest('[data-perm]');
      if (b) setPerm(b.dataset.perm);
    });
    $('agentSeg').addEventListener('click', (e) => {
      const b = e.target.closest('[data-agent]');
      if (!b || code.running) return;
      setPerm(b.dataset.agent === 'plan' ? 'plan' : (code.lastBuildPerm || 'ask'));
    });
    $('btnCodeModel').addEventListener('click', (e) => {
      e.stopPropagation();
      const menu = $('codeModelMenu');
      const names = modelNames();
      menu.innerHTML = names.length
        ? names.map((n) => `<button class="popover-item${n === codeModelName() ? ' active' : ''}" data-code-model="${escapeHtml(n)}">${escapeHtml(n)}</button>`).join('')
        : '<div class="side-empty" style="padding:10px;">No model is loaded.</div>';
      menu.classList.toggle('open');
    });
    $('codeModelMenu').addEventListener('click', (e) => {
      const b = e.target.closest('[data-code-model]');
      if (b) pickModel(b.dataset.codeModel);
    });
    document.addEventListener('click', (e) => { if (!e.target.closest('.code-model-wrap')) $('codeModelMenu').classList.remove('open'); });
    $('codeQueue').addEventListener('click', (e) => {
      const b = e.target.closest('[data-queue-drop]');
      if (b) { code.queue.splice(Number(b.dataset.queueDrop), 1); renderQueue(); }
    });

    // remember which cards the user opened or closed, so repainting keeps them
    document.addEventListener('toggle', (e) => {
      const el = e.target;
      if (!(el instanceof HTMLDetailsElement)) return;
      if (el.dataset.k === 'rules') { code.rulesOpen = el.open; return; }
      if (el.dataset.k && el.closest('#codeSide')) { code.sideOpen[el.dataset.k] = el.open; return; }
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
        renderCode();
        if (item.action === 'resume') resumeSession(row.data);
        else if (item.action === 'rewind') await rewindTo(row.data);
        else if (item.action === 'fork') forkSession(row.data);
        else if (item.action === 'revert') await revertToMessage(row.data);
        else if (item.action === 'unrevert') await unrevert();
        else if (item.action === 'model') pickModel(row.data);
        return;
      }
      const turnAction = e.target.closest('[data-turn-fork], [data-turn-revert]');
      if (turnAction) {
        if (turnAction.dataset.turnFork) forkSession(turnAction.dataset.turnFork); else await revertToMessage(turnAction.dataset.turnRevert);
        return;
      }
      const resume = e.target.closest('[data-resume]');
      if (resume) { resumeSession(resume.dataset.resume); return; }
      const cmd = e.target.closest('[data-code-command]');
      if (cmd) { await runCodeCommand(cmd.dataset.codeCommand); return; }
      if (e.target.closest('[data-run-plan]')) { setPerm(code.lastBuildPerm || 'ask'); await agentRun('Carry out the plan above now. Follow it step by step, keep the task list, and check the result at the end.'); return; }
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
    const memoryFileText = (lines) => `# Memory\n\n${lines.map((l) => `- ${l}`).join('\n')}${lines.length ? '\n' : ''}`;
    async function saveMemoryDraft() {
      const d = code.memDraft;
      if (!d) return;
      for (const scope of ['workspace', 'global']) {
        if (d[scope] === undefined) continue;
        const lines = d[scope].split('\n').map((l) => l.replace(/^\s*-\s*/, '').trim()).filter(Boolean);
        const res = await apiFetch('/api/studio/agent/memory', { method: 'POST', body: { scope, op: 'set', text: memoryFileText(lines) } });
        if (!res.ok) { toast((await res.json().catch(() => ({}))).error?.message || 'The memory could not be saved.'); return; }
      }
      code.memDraft = null;
      await loadInstructions();
      renderCodeSide();
      toast('Memory saved');
    }

    $('codeSide').addEventListener('input', (e) => {
      if (e.target.dataset.rules) code.rulesDraft = { ...(code.rulesDraft || code.rules), [e.target.dataset.rules]: e.target.value };
      if (e.target.dataset.memory) code.memDraft = { ...(code.memDraft || {}), [e.target.dataset.memory]: e.target.value };
      if (e.target.dataset.sessionSearch !== undefined) {
        code.sessionQuery = e.target.value;
        const pos = e.target.selectionStart;
        renderCodeSide(true);
        const box = $('codeSide').querySelector('[data-session-search]');
        if (box) { box.focus(); box.setSelectionRange(pos, pos); }
      }
    });
    $('codeSide').addEventListener('change', (e) => {
      const t = e.target;
      if (t.dataset.mode) {
        const [agent, cat] = t.dataset.mode.split(':');
        code.modes = { ...code.modes, [agent]: { ...(code.modes[agent] || {}) } };
        if (t.value) code.modes[agent][cat] = t.value; else delete code.modes[agent][cat];
        saveCodeSettings();
        renderCodeSide(true);
      }
      if (t.dataset.autoCheck !== undefined) { code.autoCheck = t.checked; saveCodeSettings(); }
    });
    $('codeSide').addEventListener('click', async (e) => {
      if (e.target.closest('[data-memory-save]')) { await saveMemoryDraft(); return; }
      if (e.target.closest('[data-memory-reset]')) { code.memDraft = null; renderCodeSide(); return; }
      const sAct = e.target.closest('[data-session-act]');
      if (sAct) {
        const [act, id] = sAct.dataset.sessionAct.split(':');
        if (act === 'rename') await renameSession(id);
        else if (act === 'delete') await deleteSession(id);
        else if (act === 'export') exportSession(id);
        else if (act === 'fork') { if (id === code.sessionId) forkSession(''); else { resumeSession(id); forkSession(''); } }
        return;
      }
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
