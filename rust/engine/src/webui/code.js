    // ==========================================================================
    // Code: a coding agent in the browser. The model works through tools (read, search, change files,
    // run commands) that the server executes inside one workspace folder; this part is the loop,
    // the permission questions and the view. Nothing is sent anywhere except to this server.
    // ==========================================================================
    const CODE_MAX_TURNS = 40;
    const CODE_RECENT_KEY = 'fllm.studio.workspaces.v1';
    const CODE_SESSION_KEY = 'fllm.studio.codesession.v1';
    const READ_ONLY_TOOLS = new Set(['list_dir', 'read_file', 'glob', 'grep', 'todo_write']);
    const INSTRUCTION_FILES = ['FLLM.md', 'AGENTS.md', 'CLAUDE.md'];
    const RISKY_COMMANDS = new Set(['rm', 'rmdir', 'del', 'erase', 'rd', 'format', 'shutdown', 'reboot', 'mkfs', 'dd', 'sudo', 'su', 'curl', 'wget', 'iwr', 'invoke-webrequest', 'powershell', 'pwsh', 'cmd', 'sh', 'bash', 'chmod', 'chown', 'kill', 'taskkill', 'git']);

    const TOOL_DEFS = {
      list_dir: { description: 'List the files and folders of a directory in the workspace.', properties: { path: { type: 'string', description: 'Folder relative to the workspace. Default ".".' } }, required: [] },
      read_file: { description: 'Read a text file. The result has line numbers. Long files are read in parts with offset and limit.', properties: { path: { type: 'string' }, offset: { type: 'integer', description: 'First line to read, starting at 1.' }, limit: { type: 'integer', description: 'Number of lines to read.' } }, required: ['path'] },
      write_file: { description: 'Create a file or replace it completely. Prefer edit_file for changes to an existing file.', properties: { path: { type: 'string' }, content: { type: 'string', description: 'The complete new content of the file.' } }, required: ['path', 'content'] },
      edit_file: { description: 'Change a file by replacing text. old_string must match the file exactly (including indentation) and must occur once, unless replace_all is true.', properties: { path: { type: 'string' }, old_string: { type: 'string' }, new_string: { type: 'string' }, replace_all: { type: 'boolean' } }, required: ['path', 'old_string', 'new_string'] },
      glob: { description: 'Find files by name pattern, for example "src/**/*.rs" or "*.md". Newest files first.', properties: { pattern: { type: 'string' } }, required: ['pattern'] },
      grep: { description: 'Search the lines of files with a regular expression. Returns path:line: text.', properties: { pattern: { type: 'string' }, path: { type: 'string', description: 'File or folder to search. Default: the whole workspace.' }, glob: { type: 'string', description: 'Only search files whose name matches, for example "*.rs".' } }, required: ['pattern'] },
      bash: { description: 'Run a shell command in the workspace folder and return its output. Use it to build, test, run programs and use git.', properties: { command: { type: 'string' }, timeout_ms: { type: 'integer', description: 'Stop the command after this many milliseconds. Default 120000.' } }, required: ['command'] },
      todo_write: { description: 'Keep a task list for work with several steps. Send the complete list every time; exactly one task should be in_progress.', properties: { todos: { type: 'array', items: { type: 'object', properties: { content: { type: 'string' }, status: { type: 'string', enum: ['pending', 'in_progress', 'completed'] } }, required: ['content', 'status'] } } }, required: ['todos'] },
    };

    const code = {
      workspace: null, // { root, name, git_branch, os, shell }
      perm: localStorage.getItem('fllm.studio.perm') || 'ask',
      alwaysAllow: new Set(),
      messages: [], // what the model sees
      view: [], // what the user sees
      todos: [],
      changed: new Map(),
      instructions: '',
      running: false,
      abort: null,
      used: 0,
      seq: 0,
      pending: null, // the permission question that waits for an answer
    };
    code.perm = ['ask', 'edit', 'plan'].includes(code.perm) ? code.perm : 'ask';

    const cid = () => `c${++code.seq}`;
    const toolList = () => Object.entries(TOOL_DEFS)
      .filter(([name]) => code.perm !== 'plan' || READ_ONLY_TOOLS.has(name))
      .map(([name, d]) => ({ type: 'function', function: { name, description: d.description, parameters: { type: 'object', properties: d.properties, required: d.required } } }));

    function codeSystemPrompt() {
      const w = code.workspace || {};
      const lines = [
        'You are fLLM Code, a coding agent that runs on the user\'s own computer. You work inside the workspace folder shown below and use tools to look at files, change them and run commands.',
        '',
        'How to work:',
        '- Look before you change anything: use list_dir, glob and grep to find files, read_file to read them. Never guess what a file contains.',
        '- Change files with edit_file (old_string must match exactly and be unique). Use write_file only for new files or complete rewrites. Make the smallest change that solves the task and keep the style of the code around it.',
        '- After changing code, build or test it with bash when the project has a build or tests, and fix what fails.',
        '- For work with three or more steps keep a task list with todo_write.',
        '- Keep answers short. Say what you did and what the user should know. Answer in the language the user writes in.',
        '- Do not run commands that delete files, change the system or send data to other computers unless the user asked for it.',
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
      if (!res.ok) { toast(data.error?.message || 'Der Ordner konnte nicht geöffnet werden.'); return false; }
      adoptWorkspace(data);
      const recent = [data.root, ...recentWorkspaces().filter((p) => p !== data.root)].slice(0, 6);
      localStorage.setItem(CODE_RECENT_KEY, JSON.stringify(recent));
      await loadInstructions();
      restoreCodeSession();
      renderCode();
      return true;
    }

    function adoptWorkspace(data) {
      code.workspace = data.root ? data : null;
      if (data.root) localStorage.setItem(CODE_RECENT_KEY, JSON.stringify([data.root, ...recentWorkspaces().filter((p) => p !== data.root)].slice(0, 6)));
      $('wsName').textContent = data.root ? data.name : 'Ordner wählen';
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
          if (data.root) { await loadInstructions(); restoreCodeSession(); }
        } else {
          code.disabled = (await res.json().catch(() => ({}))).error?.message || 'Der Coding-Agent ist nicht verfügbar.';
        }
        renderCode();
      } catch { /* the page is opened before the server is ready: the next visit asks again */ }
    }

    async function toolCall(name, args, signal) {
      const res = await apiFetch('/api/studio/agent/tool', { method: 'POST', body: { name, args }, signal });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        return { ok: false, error: err.error?.message || `Fehler ${res.status}` };
      }
      return res.json();
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
        title: 'Arbeitsordner wählen',
        text: `Der Agent darf nur in diesem Ordner lesen, ändern und Befehle ausführen. Gib den vollständigen Pfad ein, zum Beispiel ${code.workspace?.os === 'windows' || /Windows/.test(navigator.userAgent) ? 'C:\\Users\\du\\projekt' : '/home/du/projekt'}.${recent.length ? ` Zuletzt: ${recent[0]}` : ''}`,
        input: code.workspace?.root || recent[0] || '',
        ok: 'Öffnen',
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
      $('threadList').innerHTML = `${rows || '<div class="thread-empty">Noch kein Ordner geöffnet</div>'}
        <div class="thread-row" data-ws-open><span class="thread-row-title">Ordner öffnen …</span></div>`;
    }
    $('threadList').addEventListener('click', async (e) => {
      if (state.tab !== 'code') return;
      const row = e.target.closest('[data-ws-row]');
      if (row) { switchSurface('code'); await setWorkspace(row.dataset.wsRow); return; }
      if (e.target.closest('[data-ws-open]')) { switchSurface('code'); await pickWorkspace(); }
    });
    // ---- saving the session --------------------------------------------------------------
    function saveCodeSession() {
      if (!code.workspace) return;
      try {
        const all = JSON.parse(localStorage.getItem(CODE_SESSION_KEY) || '{}');
        all[code.workspace.root] = { messages: code.messages, view: code.view.filter((i) => i.kind !== 'ask'), todos: code.todos, used: code.used };
        const text = JSON.stringify(all);
        if (text.length < 3_000_000) localStorage.setItem(CODE_SESSION_KEY, text);
      } catch { /* the browser storage is full: the session is only kept while the page is open */ }
    }

    function restoreCodeSession() {
      code.messages = []; code.view = []; code.todos = []; code.changed = new Map(); code.used = 0;
      try {
        const s = JSON.parse(localStorage.getItem(CODE_SESSION_KEY) || '{}')[code.workspace.root];
        if (s) {
          code.messages = s.messages || [];
          code.view = (s.view || []).map((i) => (i.status === 'running' || i.status === 'pending' ? { ...i, status: 'error', output: 'Unterbrochen.' } : { ...i, streaming: false }));
          code.todos = s.todos || [];
          code.used = s.used || 0;
        }
      } catch { /* an unreadable saved session is ignored */ }
    }

    // ---- rendering ---------------------------------------------------------------------------
    const TOOL_LABEL = {
      list_dir: (a) => ['Ordner ansehen', a.path || '.'],
      read_file: (a) => ['Datei lesen', `${a.path || ''}${a.offset ? ` ab Zeile ${a.offset}` : ''}`],
      write_file: (a) => ['Datei schreiben', a.path || ''],
      edit_file: (a) => ['Datei ändern', a.path || ''],
      glob: (a) => ['Dateien suchen', a.pattern || ''],
      grep: (a) => ['Text suchen', `${a.pattern || ''}${a.glob ? `  in ${a.glob}` : ''}`],
      bash: (a) => ['Befehl', a.command || ''],
      todo_write: () => ['Aufgabenliste', ''],
    };
    const STATUS_ICON = {
      pending: '<svg viewBox="0 0 24 24" class="icon icon-sm"><circle cx="12" cy="12" r="9"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>',
      running: '<span class="spin"></span>',
      ok: '<svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M20 6 9 17l-5-5"/></svg>',
      error: '<svg viewBox="0 0 24 24" class="icon icon-sm"><circle cx="12" cy="12" r="9"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>',
      denied: '<svg viewBox="0 0 24 24" class="icon icon-sm"><circle cx="12" cy="12" r="9"/><path d="m5 5 14 14"/></svg>',
    };
    const STATUS_TEXT = { pending: 'wartet auf dich', running: 'läuft', ok: 'fertig', error: 'Fehler', denied: 'abgelehnt' };

    function diffHtml(item) {
      const a = item.args || {};
      const lines = [];
      if (item.name === 'edit_file') {
        String(a.old_string || '').split('\n').forEach((l) => lines.push(['del', l]));
        String(a.new_string || '').split('\n').forEach((l) => lines.push(['add', l]));
      } else if (item.name === 'write_file') {
        const all = String(a.content || '').split('\n');
        all.slice(0, 24).forEach((l) => lines.push(['add', l]));
        if (all.length > 24) lines.push(['more', `… ${all.length - 24} weitere Zeilen`]);
      } else return '';
      return `<div class="diff">${lines.map(([k, l]) => `<div class="diff-${k}"><span class="diff-mark">${k === 'add' ? '+' : k === 'del' ? '−' : ''}</span><span>${escapeHtml(l) || ' '}</span></div>`).join('')}</div>`;
    }

    function toolHtml(item) {
      const [verb, target] = (TOOL_LABEL[item.name] || (() => [item.name, '']))(item.args || {});
      const hasDetail = item.output || diffHtml(item) || (item.name === 'bash' && item.args?.command);
      const open = item.status === 'pending' || item.name === 'edit_file' || item.name === 'write_file' || (item.name === 'bash' && item.output && item.status === 'error') || item.status === 'running' && item.name === 'bash' ? ' open' : '';
      let body = '';
      if (item.name === 'bash') body += `<pre class="tool-cmd">$ ${escapeHtml(item.args?.command || '')}</pre>`;
      body += diffHtml(item);
      if (item.name === 'todo_write' && Array.isArray(item.args?.todos)) body += todoListHtml(item.args.todos);
      else if (item.output) body += `<pre class="tool-out">${escapeHtml(String(item.output).slice(0, 6000))}</pre>`;
      const ask = item.status === 'pending' ? permissionHtml(item) : '';
      return `
        <div class="tool-card is-${item.status}" data-item="${item.id}">
          <details${open}>
            <summary>
              <span class="tool-status" title="${STATUS_TEXT[item.status] || ''}">${STATUS_ICON[item.status] || ''}</span>
              <span class="tool-verb">${verb}</span>
              <span class="tool-target mono">${escapeHtml(target)}</span>
            </summary>
            ${hasDetail ? `<div class="tool-body">${body}</div>` : ''}
          </details>
          ${ask}
        </div>`;
    }

    function permissionHtml(item) {
      const key = alwaysKey(item);
      const label = item.name === 'bash' ? 'Befehl ausführen?' : item.name === 'edit_file' ? 'Diese Änderung übernehmen?' : 'Datei schreiben?';
      return `<div class="perm-ask"><span>${label}</span>
        <button class="btn btn-sm btn-primary" data-perm-answer="allow">Erlauben</button>
        ${key ? `<button class="btn btn-sm btn-secondary" data-perm-answer="always" title="${escapeHtml(item.name === 'bash' ? `Weitere „${key}“-Befehle in dieser Sitzung ohne Frage erlauben` : 'Änderungen an Dateien in dieser Sitzung ohne Frage erlauben')}">Immer${item.name === 'bash' ? ` (${escapeHtml(key)})` : ''}</button>` : ''}
        <button class="btn btn-sm btn-ghost" data-perm-answer="deny">Ablehnen</button></div>`;
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
        const thinking = reasoning ? `<details class="cot-box"><summary>${svgIcon('brain')}<span>${item.streaming && !answer ? 'Denkt nach' : 'Gedankengang'}</span></summary><div class="cot-body">${escapeHtml(reasoning)}</div></details>` : '';
        const body = answer ? MD.render(answer) : (item.streaming && !reasoning ? '<span class="typing"><i></i><i></i><i></i></span>' : '');
        const plan = item.planDone ? '<div class="plan-actions"><button class="btn btn-sm btn-primary" data-run-plan>Plan ausführen</button><span>Der Agent fragt weiter vor jeder Änderung.</span></div>' : '';
        return `<div class="code-assistant" data-item="${item.id}">${thinking}<div class="msg-assistant-prose">${body}</div>${item.error ? `<div class="msg-error">${escapeHtml(item.error)}</div>` : ''}${plan}</div>`;
      }
      if (item.kind === 'tool') return toolHtml(item);
      return `<div class="code-notice" data-item="${item.id}">${escapeHtml(item.text)}</div>`;
    }

    function emptyCodeHtml() {
      if (code.disabled) return `<div class="code-empty"><h1>Code</h1><p>${escapeHtml(code.disabled)}</p></div>`;
      if (!code.workspace) {
        const recent = recentWorkspaces();
        return `<div class="code-empty">
          <h1>Code</h1>
          <p>Ein Agent, der in einem Ordner deines Rechners liest, sucht, Dateien ändert und Befehle ausführt. Du siehst jeden Schritt und bestätigst Änderungen.</p>
          <div class="code-open"><input type="text" class="ui-input mono" id="wsPathInput" placeholder="Ordnerpfad, z. B. ${/Windows/.test(navigator.userAgent) ? 'C:\\Users\\du\\projekt' : '/home/du/projekt'}" value="${escapeHtml(recent[0] || '')}"><button class="btn btn-primary" id="wsOpenBtn">Ordner öffnen</button></div>
          ${recent.length > 1 ? `<div class="code-recent">${recent.map((r) => `<button class="starter-pill" data-ws="${escapeHtml(r)}">${escapeHtml(r)}</button>`).join('')}</div>` : ''}
          <p class="code-fine">Der Agent kann nur in diesem Ordner arbeiten. Er ist nur von diesem Rechner aus erreichbar, auch wenn der Server einen API-Schlüssel hat.</p>
        </div>`;
      }
      const modes = { ask: 'fragt vor jeder Änderung und jedem Befehl', edit: 'ändert Dateien selbst, fragt vor Befehlen', plan: 'liest nur und macht einen Plan' };
      return `<div class="code-empty">
        <h1>${escapeHtml(code.workspace.name)}</h1>
        <p>${escapeHtml(code.workspace.root)}${code.workspace.git_branch ? `, Branch ${escapeHtml(code.workspace.git_branch)}` : ''}</p>
        <p>Der Agent ${modes[code.perm]}. Beschreibe die Aufgabe unten.</p>
        <div class="code-recent">
          <button class="starter-pill" data-code-prompt="Erkläre mir den Aufbau dieses Projekts: die wichtigsten Ordner, Dateien und wie sie zusammenhängen.">Aufbau des Projekts erklären</button>
          <button class="starter-pill" data-code-prompt="Suche alle TODO- und FIXME-Kommentare und fasse sie nach Dateien zusammen.">TODOs einsammeln</button>
          <button class="starter-pill" data-code-prompt="Finde heraus, wie man dieses Projekt baut und seine Tests ausführt, und führe die Tests aus.">Tests ausführen</button>
        </div>
        ${code.instructions ? `<p class="code-fine">Projektanweisungen gelesen: ${escapeHtml(code.instructions.split('\n')[0].replace(/[()]/g, ''))}</p>` : ''}
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

    /** Rewrite one item (or add it) without touching the rest. */
    function paintItem(item) {
      const column = $('codeColumn');
      if (!code.view.length || column.querySelector('.code-empty')) { renderCode(); return; }
      const html = itemHtml(item);
      const existing = column.querySelector(`[data-item="${item.id}"]`);
      if (existing) {
        const open = existing.querySelector('details')?.open;
        existing.outerHTML = html;
        if (open !== undefined && item.kind === 'tool') {
          const d = column.querySelector(`[data-item="${item.id}"] details`);
          if (d && item.status !== 'pending') d.open = open || (item.name === 'bash' && item.status === 'error');
        }
      } else {
        column.insertAdjacentHTML('beforeend', html);
      }
      if (codeStick) $('codeScroll').scrollTop = $('codeScroll').scrollHeight;
    }

    function renderCodeSide() {
      const side = $('codeSide');
      const files = [...code.changed.entries()];
      const done = code.todos.filter((t) => t.status === 'completed').length;
      side.innerHTML = `
        <section class="side-block"><h4>Aufgaben${code.todos.length ? `<span class="side-count">${done}/${code.todos.length}</span>` : ''}</h4>
          ${code.todos.length ? todoListHtml(code.todos) : '<p class="side-empty">Bei größeren Aufträgen legt der Agent hier eine Liste an.</p>'}</section>
        <section class="side-block"><h4>Geänderte Dateien${files.length ? `<span class="side-count">${files.length}</span>` : ''}</h4>
          ${files.length ? `<ul class="changed-list">${files.map(([p, n]) => `<li><span class="mono">${escapeHtml(p)}</span>${n > 1 ? `<span class="side-count">${n}×</span>` : ''}</li>`).join('')}</ul>` : '<p class="side-empty">Noch nichts geändert.</p>'}</section>`;
    }

    function updateCodeMeter() {
      setMeter('codeCtxFill', 'codeCtxText', code.used || codeEstimate(), state.status?.model?.max_context || 4096);
    }
    const codeEstimate = () => estimateTokens(codeSystemPrompt()) + code.messages.reduce((s, m) => s + estimateTokens(m.content) + estimateTokens(JSON.stringify(m.tool_calls || '')) + 6, 0);

    function setCodeBusy(busy) {
      code.running = busy;
      const btn = $('btnCodeSend');
      btn.classList.toggle('stop', busy);
      btn.title = busy ? 'Anhalten (Esc)' : 'Senden (Eingabe)';
      btn.innerHTML = busy
        ? '<svg viewBox="0 0 24 24" class="icon icon-sm"><rect width="10" height="10" x="7" y="7" rx="1" fill="currentColor"/></svg>'
        : '<svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>';
      $('codeHint').textContent = busy ? 'Der Agent arbeitet. Esc hält ihn an.' : 'Eingabe sendet, Umschalt+Eingabe macht eine neue Zeile';
    }

    // ---- permissions ----------------------------------------------------------------------------------
    function commandKey(command) {
      const first = String(command || '').trim().split(/\s+/)[0] || '';
      const base = first.replace(/^.*[\\/]/, '').replace(/\.(exe|cmd|bat)$/i, '').toLowerCase();
      if (!base || RISKY_COMMANDS.has(base) || /[;&|><`$(){}]/.test(String(command))) return null;
      return base;
    }

    function alwaysKey(item) {
      if (item.name === 'bash') return commandKey(item.args?.command);
      return item.name === 'edit_file' || item.name === 'write_file' ? 'edit' : null;
    }

    function needsPermission(item) {
      if (READ_ONLY_TOOLS.has(item.name)) return false;
      if (item.name === 'bash') {
        const key = commandKey(item.args?.command);
        return !(key && code.alwaysAllow.has(key));
      }
      return code.perm !== 'edit';
    }

    function askPermission(item) {
      return new Promise((resolve) => {
        code.pending = { item, resolve };
        item.status = 'pending';
        paintItem(item);
        $('codeScroll').scrollTop = $('codeScroll').scrollHeight;
      });
    }

    function answerPermission(answer) {
      const p = code.pending;
      if (!p) return;
      code.pending = null;
      if (answer === 'always') {
        const key = alwaysKey(p.item);
        if (key === 'edit') { code.perm = 'edit'; localStorage.setItem('fllm.studio.perm', code.perm); renderCode(); }
        else if (key) code.alwaysAllow.add(key);
      }
      p.resolve(answer === 'deny' ? 'deny' : 'allow');
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

    async function streamAssistant(item) {
      const p = getSampler();
      const body = {
        model: state.selectedModel || state.status?.model?.name || 'tiny',
        messages: [{ role: 'system', content: codeSystemPrompt() }, ...code.messages],
        tools: toolList(), stream: true, stream_options: { include_usage: true },
        temperature: Math.min(p.temperature, 0.4), top_p: 0.9, max_tokens: Math.max(p.max_tokens, 2048),
      };
      if (state.thinkingEnabled) body.reasoning_effort = 'high';
      const res = await apiFetch('/v1/chat/completions', { method: 'POST', body, signal: code.abort.signal });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error?.message || `Der Server antwortet mit Fehler ${res.status}`);
      }
      const calls = [];
      let finish = '';
      let paint = false;
      const queue = () => {
        if (paint) return;
        paint = true;
        requestAnimationFrame(() => { paint = false; paintItem(item); });
      };
      await readEventStream(res, (chunk) => {
        const choice = chunk.choices?.[0];
        const delta = choice?.delta;
        if (delta?.reasoning_content) { item.reasoning = (item.reasoning || '') + delta.reasoning_content; queue(); }
        if (delta?.content) { item.content += delta.content; queue(); }
        for (const tc of delta?.tool_calls || []) {
          const i = tc.index ?? calls.length - 1;
          if (!calls[i]) calls[i] = { id: tc.id || `call_${cid()}`, name: '', args: '' };
          if (tc.id) calls[i].id = tc.id;
          if (tc.function?.name) calls[i].name += tc.function.name;
          if (tc.function?.arguments) calls[i].args += tc.function.arguments;
        }
        if (choice?.finish_reason) finish = choice.finish_reason;
        if (chunk.usage?.total_tokens) code.used = chunk.usage.total_tokens;
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
      const max = Math.max(3000, Math.floor((state.status?.model?.max_context || 4096) * 3.4 * 0.22));
      const s = String(text ?? '');
      return s.length > max ? `${s.slice(0, max)}\n... [${s.length - max} Zeichen gekürzt]` : s;
    };

    async function runToolCall(call) {
      const args = parseArgs(call.args);
      const item = { kind: 'tool', id: cid(), callId: call.id, name: call.name, args: args || {}, status: 'running', output: '' };
      code.view.push(item);
      const reply = (content) => code.messages.push({ role: 'tool', tool_call_id: call.id, content: clipOutput(content) });
      if (args === null) {
        item.status = 'error'; item.output = 'Die Argumente sind kein gültiges JSON.';
        paintItem(item);
        reply('Error: the arguments were not valid JSON. Call the tool again with a valid JSON object.');
        return;
      }
      if (!TOOL_DEFS[call.name]) {
        item.status = 'error'; item.output = `Unbekanntes Werkzeug ${call.name}`;
        paintItem(item);
        reply(`Error: unknown tool '${call.name}'. Available tools: ${Object.keys(TOOL_DEFS).join(', ')}.`);
        return;
      }
      if (code.perm === 'plan' && !READ_ONLY_TOOLS.has(call.name)) {
        item.status = 'denied'; item.output = 'Im Plan-Modus ist nur Lesen erlaubt.';
        paintItem(item);
        reply('Denied: plan mode only allows reading. Describe the change in your plan instead.');
        return;
      }
      if (call.name === 'todo_write') {
        code.todos = (Array.isArray(args.todos) ? args.todos : []).filter((t) => t && t.content).map((t) => ({ content: String(t.content), status: ['pending', 'in_progress', 'completed'].includes(t.status) ? t.status : 'pending' }));
        item.status = 'ok'; item.args = { todos: code.todos };
        paintItem(item); renderCodeSide();
        reply('Todos updated.');
        return;
      }
      if (needsPermission(item)) {
        paintItem(item);
        const verdict = await askPermission(item);
        if (verdict === 'deny' || !code.running) {
          item.status = 'denied'; item.output = code.running ? 'Du hast abgelehnt.' : 'Angehalten.';
          paintItem(item);
          reply('The user denied this action. Do not try it again unchanged; ask what they want instead.');
          return;
        }
      }
      item.status = 'running';
      paintItem(item);
      try {
        const r = await toolCall(call.name, args, code.abort.signal);
        item.status = r.ok ? 'ok' : 'error';
        item.output = r.ok ? r.output : r.error;
        if (r.ok && (call.name === 'write_file' || call.name === 'edit_file') && args.path) {
          code.changed.set(args.path, (code.changed.get(args.path) || 0) + 1);
          renderCodeSide();
        }
        paintItem(item);
        reply(r.ok ? r.output : `Error: ${r.error}`);
      } catch (e) {
        item.status = 'error'; item.output = e.name === 'AbortError' ? 'Angehalten.' : e.message;
        paintItem(item);
        reply(`Error: ${item.output}`);
      }
    }

    /** Keeps the conversation inside the context window: shorten old tool output, then drop old steps. */
    function fitContext() {
      const max = state.status?.model?.max_context || 4096;
      const limit = Math.floor(max * 0.72);
      if (codeEstimate() <= limit) return;
      code.messages.forEach((m, i) => {
        if (m.role === 'tool' && i < code.messages.length - 6 && m.content.length > 400) m.content = `${m.content.slice(0, 300)}\n... [gekürzt, um Platz zu sparen]`;
      });
      let dropped = 0;
      // first whole earlier tasks (everything before the next user message), then single steps of the current task
      while (codeEstimate() > limit) {
        const next = code.messages.findIndex((m, i) => i > 0 && m.role === 'user');
        if (next === -1) break;
        code.messages.splice(0, next);
        dropped++;
      }
      while (codeEstimate() > limit && code.messages.length > 3) {
        let end = 2;
        while (code.messages[end] && code.messages[end].role === 'tool') end++;
        code.messages.splice(1, end - 1);
        dropped++;
      }
      if (dropped) {
        const notice = { kind: 'notice', id: cid(), text: 'Ältere Schritte wurden aus dem Kontext des Modells entfernt, damit das Modell nicht überläuft. Du siehst sie hier weiter.' };
        code.view.push(notice);
        paintItem(notice);
      }
    }
    async function agentRun(userText) {
      if (!code.workspace) { await pickWorkspace(); if (!code.workspace) return; }
      const expanded = await expandMentions(userText);
      code.view.push({ kind: 'user', id: cid(), text: userText });
      code.messages.push({ role: 'user', content: expanded });
      renderCode();
      code.abort = new AbortController();
      setCodeBusy(true);
      try {
        for (let turn = 0; turn < CODE_MAX_TURNS && code.running; turn++) {
          fitContext();
          const item = { kind: 'assistant', id: cid(), content: '', reasoning: '', streaming: true };
          code.view.push(item);
          paintItem(item);
          let result;
          try {
            result = await streamAssistant(item);
          } finally {
            item.streaming = false;
          }
          const message = { role: 'assistant', content: item.content || '' };
          if (result.calls.length) message.tool_calls = result.calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args || '{}' } }));
          if (item.content || result.calls.length) code.messages.push(message);
          if (!result.calls.length) {
            if (code.perm === 'plan' && item.content) item.planDone = true;
            if (!item.content) item.error = result.finish === 'length' ? 'Die Antwort wurde vom Längenlimit abgeschnitten.' : 'Das Modell hat nichts geantwortet.';
            paintItem(item);
            break;
          }
          if (!item.content && !item.reasoning) {
            code.view.splice(code.view.indexOf(item), 1);
            $('codeColumn').querySelector(`[data-item="${item.id}"]`)?.remove();
          } else {
            paintItem(item);
          }
          for (const call of result.calls) {
            if (!code.running) break;
            await runToolCall(call);
          }
          if (turn === CODE_MAX_TURNS - 1) code.view.push({ kind: 'notice', id: cid(), text: `Nach ${CODE_MAX_TURNS} Schritten angehalten. Schreibe „weiter“, wenn der Agent fortfahren soll.` });
        }
      } catch (e) {
        if (e.name !== 'AbortError') {
          const err = { kind: 'assistant', id: cid(), content: '', error: e.message };
          code.view.push(err);
        }
      } finally {
        if (code.pending) { const p = code.pending; code.pending = null; p.resolve('deny'); }
        code.abort = null;
        setCodeBusy(false);
        saveCodeSession();
        renderCode();
        refreshStudioStatus();
      }
    }

    function stopAgent() {
      if (!code.running) return;
      code.running = false;
      code.abort?.abort();
      if (code.pending) { const p = code.pending; code.pending = null; p.resolve('deny'); }
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

    async function userRun(command, label) {
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

    function setPerm(perm) {
      code.perm = perm;
      localStorage.setItem('fllm.studio.perm', perm);
      $('permSeg').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.perm === perm));
      if (!code.view.length) renderCode();
    }

    async function undoChange() {
      const r = await apiFetch('/api/studio/agent/undo', { method: 'POST', body: {} }).then((x) => x.json()).catch((e) => ({ ok: false, error: e.message }));
      codeNotice(r.ok ? `Rückgängig gemacht: ${r.output}` : (r.error || 'Nichts zum Rückgängigmachen.'));
    }

    function newCodeSession() {
      if (code.running) stopAgent();
      code.messages = []; code.view = []; code.todos = []; code.changed = new Map(); code.used = 0; code.alwaysAllow = new Set();
      saveCodeSession();
      renderCode();
    }

    async function runCodeCommand(text) {
      const [cmd, ...rest] = text.slice(1).split(/\s+/);
      switch (cmd.toLowerCase()) {
        case 'clear': case 'neu': newCodeSession(); return true;
        case 'undo': await undoChange(); return true;
        case 'plan': case 'ask': case 'edit': setPerm(cmd.toLowerCase()); codeNotice({ plan: 'Plan-Modus: der Agent liest nur.', ask: 'Der Agent fragt vor jeder Änderung und jedem Befehl.', edit: 'Auto-Edit: Dateien ändert der Agent selbst, Befehle erfragt er.' }[cmd.toLowerCase()]); return true;
        case 'diff': await userRun('git diff --stat && git diff'); return true;
        case 'status': await userRun('git status -sb'); return true;
        case 'cd': case 'ordner': if (rest.length) await setWorkspace(rest.join(' ')); else await pickWorkspace(); return true;
        case 'hilfe': case 'help':
          codeNotice('Befehle: /plan, /ask, /edit (was der Agent allein darf), /undo (letzte Dateiänderung), /diff, /status (git), /cd Pfad (anderer Ordner), /clear (neue Sitzung). Mit @datei hängst du eine Datei an.');
          return true;
        default: return false;
      }
    }

    // ---- composer: @ completion --------------------------------------------------------------------------------
    let mentionTimer = null;
    let mentionItems = [];
    let mentionIndex = 0;
    function closeMention() { $('mentionPop').classList.remove('open'); mentionItems = []; }
    function renderMention() {
      const pop = $('mentionPop');
      if (!mentionItems.length) { closeMention(); return; }
      pop.innerHTML = mentionItems.map((p, i) => `<button class="${i === mentionIndex ? 'active' : ''}" data-mention="${i}"><span class="mono">${escapeHtml(p)}</span></button>`).join('');
      pop.classList.add('open');
    }
    function pickMention(i) {
      const ta = $('codeTextarea');
      const cursor = ta.selectionStart;
      const before = ta.value.slice(0, cursor).replace(/@[\w./\\*-]*$/, `@${mentionItems[i]} `);
      ta.value = before + ta.value.slice(cursor);
      ta.setSelectionRange(before.length, before.length);
      closeMention();
      ta.focus();
    }
    function onCodeInput() {
      const ta = $('codeTextarea');
      ta.style.height = 'auto';
      ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
      const m = ta.value.slice(0, ta.selectionStart).match(/(?:^|\s)@([\w./\\*-]*)$/);
      clearTimeout(mentionTimer);
      if (!m || !code.workspace) { closeMention(); return; }
      mentionTimer = setTimeout(async () => {
        const q = m[1].replace(/\\/g, '/');
        const r = await toolCall('glob', { pattern: q.includes('*') ? q : `**/*${q}*` }).catch(() => null);
        if (!r?.ok) { closeMention(); return; }
        mentionItems = r.output.split('\n').filter((l) => l && !l.startsWith('...') && l !== 'no files match').slice(0, 8);
        mentionIndex = 0;
        renderMention();
      }, 140);
    }

    async function submitCode() {
      if (code.running) { stopAgent(); return; }
      const ta = $('codeTextarea');
      const text = ta.value.trim();
      if (!text) return;
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
        if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); pickMention(mentionIndex); return; }
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
    $('btnCodeUndo').addEventListener('click', undoChange);
    $('btnCodeNew').addEventListener('click', newCodeSession);
    $('permSeg').addEventListener('click', (e) => {
      const b = e.target.closest('[data-perm]');
      if (b) setPerm(b.dataset.perm);
    });
    $('codeColumn').addEventListener('click', async (e) => {
      const answer = e.target.closest('[data-perm-answer]');
      if (answer) { answerPermission(answer.dataset.permAnswer); return; }
      if (e.target.closest('[data-run-plan]')) { setPerm('ask'); await agentRun('Führe den Plan jetzt aus.'); return; }
      const prompt = e.target.closest('[data-code-prompt]');
      if (prompt) { await agentRun(prompt.dataset.codePrompt); return; }
      const ws = e.target.closest('[data-ws]');
      if (ws) { await setWorkspace(ws.dataset.ws); return; }
      if (e.target.closest('#wsOpenBtn')) { const v = $('wsPathInput').value.trim(); if (v) await setWorkspace(v); return; }
      const copy = e.target.closest('[data-copy-code]');
      if (copy) { const pre = $(copy.dataset.copyCode); if (pre) { await navigator.clipboard.writeText(pre.textContent); toast('Code kopiert'); } return; }
      const wrap = e.target.closest('[data-wrap-code]');
      if (wrap) $(wrap.dataset.wrapCode)?.classList.toggle('wrap');
    });
    $('codeColumn').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target.id === 'wsPathInput') $('wsOpenBtn').click();
    });
