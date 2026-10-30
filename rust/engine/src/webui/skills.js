    // ==========================================================================
    // Skills and scheduled tasks.
    //  - A skill is a Markdown file (front matter: name, description, when; then the steps). The user's own skills live
    //    in the Studio folder, the skills of a workspace in <folder>/.fllm/skills/<name>/SKILL.md.
    //  - A scheduled task is a prompt the server runs on a clock (chat answer, or an agent run with read-only tools).
    // Exposed on window for the chat and the code page: fllmSkillIndex, fllmSkillIndexRefresh, fllmSkillRead,
    // fllmSkillTool (+ fllmSkillToolRun), fllmSkillPropose, fllmScheduleTool (+ fllmScheduleToolRun).
    // ==========================================================================

    // @pure-begin
    const SK_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
    const SK_MAX_BYTES = 64 * 1024;

    function skValidName(name) { return typeof name === 'string' && SK_NAME_RE.test(name); }

    /** A readable title turned into a skill name. */
    function skSlug(text) {
      return String(text || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64).replace(/-+$/, '');
    }

    /** `{meta, body}` of a SKILL.md text; meta holds the lowercase keys of the front matter. */
    function skParse(text) {
      let t = String(text || '');
      if (t.charCodeAt(0) === 0xFEFF) t = t.slice(1);
      const m = /^---\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)/.exec(t);
      if (!m) return { meta: {}, body: t };
      const meta = {};
      for (const line of m[1].split(/\r?\n/)) {
        const i = line.indexOf(':');
        if (i < 1) continue;
        meta[line.slice(0, i).trim().toLowerCase()] = line.slice(i + 1).trim().replace(/^["']|["']$/g, '');
      }
      return { meta, body: t.slice(m[0].length) };
    }

    /** The text of a SKILL.md from its parts. */
    function skCompose({ name, description, when, body }) {
      const one = (s) => String(s || '').replace(/\s*\n\s*/g, ' ').trim();
      return `---\nname: ${one(name)}\ndescription: ${one(description)}\nwhen: ${one(when)}\n---\n${String(body || '').replace(/^\n+/, '')}${String(body || '').endsWith('\n') ? '' : '\n'}`;
    }

    /**
     * The part of the system prompt that names the skills: one line each, capped. `skills` are the entries of
     * GET /api/studio/skills ({name, description, when, scope}). Empty when there are none.
     */
    function skIndexText(skills, maxChars = 1800, maxItems = 40) {
      const list = (skills || []).filter((s) => s && skValidName(s.name));
      if (!list.length) return '';
      const head = 'Skills are saved step-by-step procedures. When the task matches one, call the skill tool with action "read" and its name first, then follow it. Available skills:';
      let out = head;
      let shown = 0;
      for (const s of list) {
        const desc = String(s.description || '').replace(/\s+/g, ' ').slice(0, 140);
        const when = s.when ? ` (use when: ${String(s.when).replace(/\s+/g, ' ').slice(0, 100)})` : '';
        const line = `\n- ${s.name}${s.scope === 'workspace' ? ' [workspace]' : ''}: ${desc}${when}`;
        if (shown >= maxItems || out.length + line.length > maxChars) break;
        out += line;
        shown++;
      }
      if (shown < list.length) out += `\n(${list.length - shown} more: call the skill tool with action "list")`;
      return out;
    }

    /** The tool of the model for skills. `permission` is the category the page asks about; list and read need none in practice. */
    const SK_TOOL = {
      name: 'skill',
      permission: 'edit',
      description: 'Procedural memory. Skills are saved Markdown procedures. action "list" shows the names and descriptions, "read" returns one skill, "create" saves a new skill, "update" replaces an existing one, "delete" removes one. Create a skill only for a procedure that is likely to be needed again; content is the complete SKILL.md: a front matter with name, description and when, then numbered steps with exact commands and the pitfalls found.',
      properties: {
        action: { type: 'string', enum: ['list', 'read', 'create', 'update', 'delete'] },
        name: { type: 'string', description: 'Skill name: lowercase letters, digits, - and _ (up to 64 characters).' },
        content: { type: 'string', description: 'The complete SKILL.md text for create and update.' },
        scope: { type: 'string', enum: ['user', 'workspace'], description: 'user (default) keeps the skill for every project; workspace keeps it in the current folder under .fllm/skills.' },
      },
      required: ['action'],
    };

    /** An error text for arguments of the skill tool that cannot work, or ''. */
    function skCheckToolArgs(args) {
      const a = args || {};
      if (!['list', 'read', 'create', 'update', 'delete'].includes(a.action)) return 'action must be list, read, create, update or delete';
      if (a.scope !== undefined && !['user', 'workspace'].includes(a.scope)) return 'scope must be user or workspace';
      if (a.action === 'list') return '';
      if (!skValidName(a.name)) return 'name must be 1 to 64 characters: lowercase letters, digits, - and _';
      if (a.action === 'create' || a.action === 'update') {
        if (typeof a.content !== 'string' || !a.content.trim()) return 'content (the complete SKILL.md text) is missing';
        if (a.content.length > SK_MAX_BYTES) return 'content is too long (64 KB at most)';
      }
      return '';
    }

    /** The tool of the model that schedules a task (the user confirms in the page before anything is saved). */
    const SC_TOOL = {
      name: 'schedule',
      permission: 'edit',
      description: 'Schedule a task that runs later on a clock inside the app, for example "remind me ..." or "every morning summarise ...". The user confirms before it is saved. The prompt must be complete and self-contained, because it runs without this conversation. when is a schedule such as "every 30 minutes", "hourly", "daily at 08:30", "weekdays at 09:00", "weekly on mon,thu at 18:00" or a cron line with five fields. Tasks only run while the app is open. mode "agent" additionally needs an absolute folder and lets the task read and search it and look things up on the web; it cannot change anything.',
      properties: {
        name: { type: 'string', description: 'A short name for the task.' },
        prompt: { type: 'string', description: 'The complete instruction the task runs with.' },
        when: { type: 'string', description: 'The schedule.' },
        mode: { type: 'string', enum: ['chat', 'agent'], description: 'chat (default) answers from the model; agent may read the folder and the web.' },
        folder: { type: 'string', description: 'Absolute folder path, only for mode agent.' },
        notify: { type: 'boolean', description: 'Show a desktop notification when the result is ready.' },
        deliver: { type: 'string', description: 'Optional: send the result also to webhook:URL, ntfy:URL or telegram:CHAT_ID.' },
      },
      required: ['name', 'prompt', 'when'],
    };

    function scCheckToolArgs(args) {
      const a = args || {};
      if (!String(a.name || '').trim()) return 'name is missing';
      if (!String(a.prompt || '').trim()) return 'prompt is missing';
      if (!String(a.when || '').trim()) return 'when is missing';
      if (a.mode !== undefined && !['chat', 'agent'].includes(a.mode)) return 'mode must be chat or agent';
      if (a.mode === 'agent' && !String(a.folder || '').trim()) return 'mode agent needs an absolute folder';
      return '';
    }

    /** The conversation as plain text for the model that writes a skill: the start and the end are kept when it is long. */
    function skTranscript(messages, maxChars = 14000) {
      const lines = [];
      for (const m of messages || []) {
        if (!m || (m.role !== 'user' && m.role !== 'assistant' && m.role !== 'tool')) continue;
        const text = typeof m.content === 'string' ? m.content : '';
        const calls = (m.tool_calls || []).map((c) => `${c.function?.name}(${String(c.function?.arguments || '').slice(0, 200)})`).join(', ');
        if (!text.trim() && !calls) continue;
        const clipped = text.length > 1200 ? `${text.slice(0, 800)} ... ${text.slice(-300)}` : text;
        lines.push(`${m.role === 'tool' ? 'TOOL RESULT' : m.role.toUpperCase()}: ${clipped}${calls ? `\n[calls: ${calls}]` : ''}`);
      }
      const all = lines.join('\n\n');
      if (all.length <= maxChars) return all;
      const keep = Math.floor(maxChars / 2);
      return `${all.slice(0, keep)}\n\n[... the middle of the session is left out ...]\n\n${all.slice(-keep)}`;
    }

    /** The instruction that asks the model to turn a finished session into a skill. */
    function skProposalPrompt(transcript) {
      return [
        'Below is a session in which a task was completed. Turn what was learned into a reusable skill: a short procedure that would let someone do the same kind of task again without the discovery work.',
        '',
        'Answer with the SKILL.md file only, in exactly this form and nothing else:',
        '---',
        'name: short-lowercase-name-with-dashes',
        'description: one sentence saying what the skill does',
        'when: one sentence saying when to use it',
        '---',
        '# Title',
        '',
        '1. Numbered steps with exact commands, file names and settings that worked.',
        '',
        '## Pitfalls',
        '- Things that failed and what to do instead.',
        '',
        'Leave out anything specific to this one run (dates, temporary paths, the names of people). If the session holds nothing reusable, answer with the single word NONE.',
        '',
        '=== SESSION ===',
        transcript,
        '=== END OF SESSION ===',
      ].join('\n');
    }

    /** `{name, content}` from the reply of the model, or `{error}`. Reasoning blocks and a surrounding code fence are removed. */
    function skParseProposal(reply) {
      let t = String(reply || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim();
      if (!t || /^none\.?$/i.test(t)) return { error: 'The model found nothing reusable in this session.' };
      const fenced = /^```[a-z]*\n([\s\S]*?)\n```\s*$/i.exec(t);
      if (fenced) t = fenced[1].trim();
      const at = t.indexOf('---');
      if (at > 0) t = t.slice(at);
      const { meta, body } = skParse(t);
      if (!meta.description || !body.trim()) return { error: 'The model did not write a skill in the expected form.' };
      const name = skValidName(meta.name) ? meta.name : skSlug(meta.name || meta.description.split(' ').slice(0, 5).join(' '));
      if (!skValidName(name)) return { error: 'The skill has no usable name.' };
      return { name, content: `${t.replace(/\s+$/, '')}\n` };
    }
    // @pure-end

    const skUserFolder = { text: '' };
    const skCache = new Map(); // workspace folder ('' for none) -> { at, list }
    let skCurrent = null; // { name, scope } of the skill open in the editor
    const skWorkspace = () => {
      try { return (typeof code !== 'undefined' && code.workspace && code.workspace.root) || ''; } catch { return ''; }
    };

    async function skRequest(method, path, body) {
      const res = await apiFetch(path, body === undefined ? { method } : { method, body });
      let data = {};
      try { data = await res.json(); } catch { /* no body */ }
      if (!res.ok) throw new Error((data.error && data.error.message) || `Request failed (${res.status})`);
      return data;
    }

    const skQuery = (o) => Object.entries(o).filter(([, v]) => v).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');

    /** Loads the skills of the user (and of `workspace`) and returns them. */
    window.fllmSkillIndexRefresh = async (workspace) => {
      const ws = workspace === undefined ? skWorkspace() : workspace;
      try {
        const data = await skRequest('GET', `/api/studio/skills?${skQuery({ workspace: ws })}`);
        skUserFolder.text = data.user_folder || '';
        skCache.set(ws, { at: Date.now(), list: data.skills || [] });
        return data.skills || [];
      } catch {
        return (skCache.get(ws) || { list: [] }).list;
      }
    };

    /**
     * Text for the system prompt that names the available skills ('' when there are none). Synchronous: it answers from
     * the last load and refreshes in the background when that is older than a minute; await
     * `fllmSkillIndexRefresh(workspace)` first when the list must be current.
     */
    window.fllmSkillIndex = (opts) => {
      const ws = opts && opts.workspace !== undefined ? opts.workspace : skWorkspace();
      const hit = skCache.get(ws);
      if (!hit || Date.now() - hit.at > 60000) window.fllmSkillIndexRefresh(ws);
      return skIndexText(hit ? hit.list : [], opts && opts.maxChars);
    };

    /** The text of a skill (a workspace skill wins over a user skill of the same name), or null. */
    window.fllmSkillRead = async (name, opts) => {
      if (!skValidName(name)) return null;
      const ws = opts && opts.workspace !== undefined ? opts.workspace : skWorkspace();
      const scopes = opts && opts.scope ? [opts.scope] : (ws ? ['workspace', 'user'] : ['user']);
      for (const scope of scopes) {
        try {
          const data = await skRequest('GET', `/api/studio/skills/read?${skQuery({ name, scope, workspace: ws })}`);
          return data.content;
        } catch { /* try the next scope */ }
      }
      return null;
    };

    window.fllmSkillTool = SK_TOOL;
    window.fllmSkillToolDef = { type: 'function', function: { name: SK_TOOL.name, description: SK_TOOL.description, parameters: { type: 'object', properties: SK_TOOL.properties, required: SK_TOOL.required } } };
    window.fllmSkillCheckArgs = skCheckToolArgs;

    /** Runs the skill tool; always resolves to the text for the model. `ctx.workspace` overrides the folder of the code page. */
    window.fllmSkillToolRun = async (args, ctx) => {
      const problem = skCheckToolArgs(args);
      if (problem) return `Error: ${problem}`;
      const ws = ctx && ctx.workspace !== undefined ? ctx.workspace : skWorkspace();
      const scope = args.scope === 'workspace' ? 'workspace' : 'user';
      if (scope === 'workspace' && !ws) return 'Error: there is no workspace folder; use scope "user".';
      try {
        if (args.action === 'list') {
          const list = await window.fllmSkillIndexRefresh(ws);
          return list.length ? list.map((s) => `${s.name} [${s.scope}]: ${s.description}${s.when ? ` (use when: ${s.when})` : ''}`).join('\n') : 'No skills yet.';
        }
        if (args.action === 'read') {
          const text = await window.fllmSkillRead(args.name, { workspace: ws, scope: args.scope });
          return text === null ? `Error: no skill named '${args.name}'.` : text;
        }
        const exists = (await window.fllmSkillIndexRefresh(ws)).some((s) => s.name === args.name && s.scope === scope);
        if (args.action === 'create' && exists) return `Error: a skill named '${args.name}' exists already; use update to replace it.`;
        if (args.action === 'update' && !exists) return `Error: no ${scope} skill named '${args.name}'; use create.`;
        if (args.action === 'delete') {
          await skRequest('DELETE', `/api/studio/skills?${skQuery({ name: args.name, scope, workspace: ws })}`);
          skCache.delete(ws);
          return `Deleted the skill '${args.name}'.`;
        }
        await skRequest('POST', '/api/studio/skills', { name: args.name, scope, workspace: ws, content: args.content });
        skCache.delete(ws);
        return `Saved the skill '${args.name}' (${scope}).`;
      } catch (e) {
        return `Error: ${e.message}`;
      }
    };

    // ---- proposing a skill after a long successful task ---------------------------------------------------
    function skEnsureProposalModal() {
      let el = document.getElementById('skProposalModal');
      if (el) return el;
      el = document.createElement('div');
      el.className = 'modal-backdrop';
      el.id = 'skProposalModal';
      el.innerHTML = `<div class="dialog" role="dialog" aria-modal="true" aria-labelledby="skProposalTitle" style="max-width:640px;width:92vw;">
        <h3 id="skProposalTitle">Save this as a skill?</h3>
        <p class="app-item-desc" id="skProposalNote" style="margin-bottom:8px;">The model wrote this skill from the session. Edit it if needed; it is only saved when you press Save.</p>
        <div style="display:flex;gap:8px;margin-bottom:8px;">
          <input type="text" class="ui-input mono" id="skProposalName" placeholder="name" autocomplete="off" style="flex:1;">
          <select class="ui-input" id="skProposalScope" style="width:auto;"><option value="user">For every project</option><option value="workspace">This folder only</option></select>
        </div>
        <textarea class="ui-input mono" id="skProposalText" rows="16" spellcheck="false"></textarea>
        <div class="dialog-actions"><button class="btn btn-ghost" id="skProposalCancel">Discard</button><button class="btn btn-primary" id="skProposalSave">Save skill</button></div>
      </div>`;
      document.body.appendChild(el);
      return el;
    }

    /** Shows `{name, content}` for approval and saves it if the user agrees. Resolves to `{saved, name?, scope?}`. */
    window.fllmSkillShowProposal = (proposal, opts) => new Promise((resolve) => {
      const el = skEnsureProposalModal();
      const ws = opts && opts.workspace !== undefined ? opts.workspace : skWorkspace();
      const nameEl = el.querySelector('#skProposalName');
      const scopeEl = el.querySelector('#skProposalScope');
      const textEl = el.querySelector('#skProposalText');
      nameEl.value = proposal.name;
      textEl.value = proposal.content;
      scopeEl.value = 'user';
      scopeEl.querySelector('option[value="workspace"]').disabled = !ws;
      const done = (value) => { el.classList.remove('open'); el.querySelector('#skProposalSave').onclick = el.querySelector('#skProposalCancel').onclick = null; resolve(value); };
      el.querySelector('#skProposalCancel').onclick = () => done({ saved: false });
      el.querySelector('#skProposalSave').onclick = async () => {
        const name = nameEl.value.trim();
        if (!skValidName(name)) { toast('The name uses lowercase letters, digits, - and _.'); return; }
        try {
          await skRequest('POST', '/api/studio/skills', { name, scope: scopeEl.value, workspace: ws, content: textEl.value });
          skCache.delete(ws);
          toast(`Skill "${name}" saved`);
          skRefreshPane();
          done({ saved: true, name, scope: scopeEl.value });
        } catch (e) { toast(e.message); }
      };
      el.classList.add('open');
      textEl.focus();
    });

    /**
     * Asks the model to turn a finished session into a skill and shows it for approval. `messages` are chat-completions
     * style messages of the session. Call it after a long successful task (for example from a "Save as skill" button).
     * Resolves to `{saved}`; `{saved:false, error}` when the model found nothing reusable or the request failed.
     */
    window.fllmSkillPropose = async ({ messages, transcript, workspace } = {}) => {
      const text = transcript || skTranscript(messages);
      if (!text.trim()) return { saved: false, error: 'There is nothing to learn from yet.' };
      try {
        const ep = window.fllmChatEndpoint(state.selectedModel);
        const res = await apiFetch(ep.url, {
          method: 'POST',
          body: ep.transform({
            model: state.selectedModel || (state.status && state.status.model && state.status.model.name) || 'tiny', stream: false, temperature: 0.3, max_tokens: 1600,
            messages: [{ role: 'user', content: skProposalPrompt(text) }],
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error((data.error && data.error.message) || `Request failed (${res.status})`);
        const parsed = skParseProposal(data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content);
        if (parsed.error) { toast(parsed.error); return { saved: false, error: parsed.error }; }
        return await window.fllmSkillShowProposal(parsed, { workspace });
      } catch (e) {
        toast(`Could not write a skill: ${e.message}`);
        return { saved: false, error: e.message };
      }
    };

    // ---- the pane: skills --------------------------------------------------------------------------------------
    const skTemplate = () => skCompose({ name: 'my-skill', description: 'What this skill does, in one sentence', when: 'When to use it', body: '# Title\n\n1. First step\n2. Second step\n\n## Pitfalls\n- \n' });

    function skFormWorkspace() { return $('skWorkspace') ? $('skWorkspace').value.trim() : ''; }

    // other panes (the library) can ask for a reload after installing a skill
    window.fllmSkillsReload = () => { try { if (typeof skRefreshPane === 'function') skRefreshPane(); } catch { /* the pane is closed */ } };

    async function skRefreshPane() {
      const box = $('skList');
      if (!box) return;
      const ws = skFormWorkspace();
      try {
        const data = await skRequest('GET', `/api/studio/skills?${skQuery({ workspace: ws })}`);
        skCache.set(ws, { at: Date.now(), list: data.skills || [] });
        $('skFolderInfo').textContent = `Your skills are saved in ${data.user_folder}. Workspace skills are saved in <folder>/.fllm/skills/<name>/SKILL.md.`;
        box.innerHTML = (data.skills || []).length ? data.skills.map((s) => `
          <div class="app-card-item column" style="border-bottom:1px solid var(--border);padding-bottom:8px;">
            <div style="display:flex;gap:8px;align-items:center;">
              <span class="app-item-title">${escapeHtml(s.name)}</span><span class="quant-chip">${escapeHtml(s.scope)}</span><span style="flex:1"></span>
              <button class="btn btn-xs btn-ghost" data-sk-edit="${escapeHtml(s.name)}" data-sk-scope="${escapeHtml(s.scope)}">Edit</button>
              <button class="btn btn-xs btn-ghost" data-sk-del="${escapeHtml(s.name)}" data-sk-scope="${escapeHtml(s.scope)}">Delete</button>
            </div>
            <div class="app-item-desc">${escapeHtml(s.description)}${s.when ? ` <em>Use when: ${escapeHtml(s.when)}</em>` : ''}</div>
          </div>`).join('') : '<div class="app-item-desc">No skills yet. Write one below, or let the model propose one after a long task.</div>';
      } catch (e) { box.textContent = e.message; }
    }

    async function skEdit(name, scope) {
      try {
        const data = await skRequest('GET', `/api/studio/skills/read?${skQuery({ name, scope, workspace: skFormWorkspace() })}`);
        skCurrent = { name, scope };
        $('skName').value = name;
        $('skScope').value = scope;
        $('skText').value = data.content;
      } catch (e) { toast(e.message); }
    }

    async function skSave() {
      const name = $('skName').value.trim();
      if (!skValidName(name)) { toast('The name uses lowercase letters, digits, - and _ (up to 64 characters).'); return; }
      try {
        await skRequest('POST', '/api/studio/skills', { name, scope: $('skScope').value, workspace: skFormWorkspace(), content: $('skText').value });
        skCache.clear();
        toast('Skill saved');
        skCurrent = { name, scope: $('skScope').value };
        skRefreshPane();
      } catch (e) { toast(e.message); }
    }

    async function skDelete(name, scope) {
      if (!(await askDialog({ title: `Delete the skill "${name}"?`, text: 'The file is removed from disk.', ok: 'Delete', danger: true }))) return;
      try {
        await skRequest('DELETE', `/api/studio/skills?${skQuery({ name, scope, workspace: skFormWorkspace() })}`);
        skCache.clear();
        if (skCurrent && skCurrent.name === name) { skCurrent = null; $('skText').value = ''; $('skName').value = ''; }
        skRefreshPane();
      } catch (e) { toast(e.message); }
    }

    // ---- scheduled tasks ----------------------------------------------------------------------------------------
    let scTimer = null;
    let scEditing = null; // id of the task in the form
    const SC_SEEN = 'fllm.studio.scheduleSeen.v1';

    const scWhen = (t) => new Date(t * 1000).toLocaleString();

    async function scCheckWhen(when) {
      try { return await skRequest('GET', `/api/studio/schedule/check?${skQuery({ when })}`); } catch (e) { return { ok: false, error: e.message }; }
    }

    async function scRefreshPane() {
      const box = $('scList');
      if (!box) return;
      try {
        const data = await skRequest('GET', '/api/studio/schedule');
        $('scNote').textContent = data.clock_running ? data.note : 'The scheduler starts when a model is loaded. Tasks can be defined now.';
        box.innerHTML = data.tasks.length ? data.tasks.map((t) => `
          <div class="app-card-item column" data-sc="${escapeHtml(t.id)}" style="border-bottom:1px solid var(--border);padding-bottom:8px;">
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
              <span class="app-item-title">${escapeHtml(t.name)}</span>
              <span class="quant-chip">${escapeHtml(t.when)}</span><span class="quant-chip">${t.mode === 'agent' ? 'agent' : 'chat'}</span>
              ${t.deliver ? `<span class="quant-chip">${escapeHtml(String(t.deliver).split(':')[0])}</span>` : ''}${t.enabled ? '' : '<span class="quant-chip">off</span>'}${t.running ? '<span style="color:var(--primary);font-size:12px;">running</span>' : ''}
              <span style="flex:1"></span>
              <button class="btn btn-xs btn-ghost" data-sc-act="run">Run now</button>
              <button class="btn btn-xs btn-ghost" data-sc-act="results">Results</button>
              <button class="btn btn-xs btn-ghost" data-sc-act="edit">Edit</button>
              <button class="btn btn-xs btn-ghost" data-sc-act="remove">Delete</button>
            </div>
            <div class="app-item-desc">${t.next_run ? `Next run: ${escapeHtml(scWhen(t.next_run))}. ` : ''}${t.last ? `Last run: ${escapeHtml(scWhen(t.last.time))} (${t.last.ok ? 'ok' : 'failed'}${t.last.delivery ? `; delivery ${escapeHtml(t.last.delivery)}` : ''}).` : 'Has not run yet.'}${t.mode === 'agent' ? ` Folder: <code>${escapeHtml(t.folder)}</code>` : ''}</div>
            <div class="sc-results" style="display:none;"></div>
          </div>`).join('') : '<div class="app-item-desc">No scheduled tasks yet.</div>';
        window.__scTasks = data.tasks;
        clearTimeout(scTimer);
        const pane = $('spane-skills');
        if (pane && pane.classList.contains('active') && data.tasks.some((t) => t.running)) scTimer = setTimeout(scRefreshPane, 2000);
      } catch (e) { box.textContent = e.message; }
    }

    async function scShowResults(card, id) {
      const out = card.querySelector('.sc-results');
      if (out.style.display === 'block') { out.style.display = 'none'; return; }
      try {
        const data = await skRequest('GET', `/api/studio/schedule/results?${skQuery({ id })}`);
        out.innerHTML = data.results.length ? data.results.map((r) => `
          <details style="margin-top:6px;"><summary class="app-item-desc" style="cursor:pointer;">${escapeHtml(scWhen(r.time))} - ${r.ok ? 'ok' : 'failed'} (${r.seconds}s)</summary>
          <pre style="white-space:pre-wrap;max-height:300px;overflow:auto;">${escapeHtml(r.text)}</pre></details>`).join('') : '<div class="app-item-desc">No results yet.</div>';
        out.style.display = 'block';
      } catch (e) { toast(e.message); }
    }

    function scFill(t) {
      scEditing = t ? t.id : null;
      $('scName').value = t ? t.name : '';
      $('scPrompt').value = t ? t.prompt : '';
      $('scWhenInput').value = t ? t.when : '';
      $('scMode').value = t ? t.mode : 'chat';
      $('scFolder').value = t ? t.folder : '';
      $('scNotify').checked = t ? t.notify : false;
      $('scDeliver').value = t ? (t.deliver || '') : '';
      $('scEnabled').checked = t ? t.enabled : true;
      $('scFormTitle').textContent = t ? 'Edit the task' : 'New task';
      scFolderVisibility();
      scPreview();
    }

    function scFolderVisibility() { $('scFolderRow').style.display = $('scMode').value === 'agent' ? 'flex' : 'none'; }

    async function scPreview() {
      const when = $('scWhenInput').value.trim();
      const out = $('scPreview');
      if (!when) { out.textContent = ''; return; }
      const r = await scCheckWhen(when);
      out.textContent = r.ok ? `Next runs: ${r.next.map(scWhen).join('; ')}` : r.error;
      out.style.color = r.ok ? '' : 'var(--destructive,#d33)';
    }

    async function scSave() {
      const body = {
        id: scEditing || undefined, name: $('scName').value, prompt: $('scPrompt').value, when: $('scWhenInput').value, mode: $('scMode').value,
        folder: $('scFolder').value, notify: $('scNotify').checked, enabled: $('scEnabled').checked, deliver: $('scDeliver').value.trim(),
      };
      try {
        await skRequest('POST', '/api/studio/schedule', body);
        toast('Task saved');
        scFill(null);
        scRefreshPane();
      } catch (e) { toast(e.message); }
    }

    /** Looks for new results; shows a toast and, when the task asked for it and the browser allows it, a desktop notification. */
    async function scPoll() {
      if (document.hidden && (typeof Notification === 'undefined' || Notification.permission !== 'granted')) return;
      let seen = Number(localStorage.getItem(SC_SEEN) || 0);
      if (!seen) { seen = Math.floor(Date.now() / 1000); localStorage.setItem(SC_SEEN, String(seen)); return; }
      try {
        const data = await skRequest('GET', `/api/studio/schedule/results?since=${seen}`);
        if (!data.results.length) return;
        localStorage.setItem(SC_SEEN, String(Math.max(...data.results.map((r) => r.time))));
        for (const r of data.results.slice(0, 3)) {
          toast(`Scheduled task "${r.task_name}" ${r.ok ? 'finished' : 'failed'}`);
          if (r.notify && typeof Notification !== 'undefined' && Notification.permission === 'granted') {
            try { new Notification(r.task_name, { body: String(r.text).slice(0, 160) }); } catch { /* not allowed here */ }
          }
        }
        scRefreshPane();
      } catch { /* the server is not reachable right now */ }
    }

    window.fllmScheduleTool = SC_TOOL;
    window.fllmScheduleToolDef = { type: 'function', function: { name: SC_TOOL.name, description: SC_TOOL.description, parameters: { type: 'object', properties: SC_TOOL.properties, required: SC_TOOL.required } } };
    window.fllmScheduleCheckArgs = scCheckToolArgs;

    /**
     * Runs the schedule tool: checks the schedule, asks the user to confirm and only then saves the task. Always resolves
     * to the text for the model.
     */
    window.fllmScheduleToolRun = async (args) => {
      const problem = scCheckToolArgs(args);
      if (problem) return `Error: ${problem}`;
      const check = await scCheckWhen(args.when);
      if (!check.ok) return `Error: ${check.error}`;
      const summary = [`Name: ${args.name}`, `When: ${args.when} (first run ${scWhen(check.next[0])})`, `Mode: ${args.mode === 'agent' ? `agent in ${args.folder} (read-only)` : 'chat answer'}`, `Prompt: ${String(args.prompt).slice(0, 400)}`, 'It runs only while the app is open.'].join('\n');
      const ok = await askDialog({ title: 'Schedule this task?', text: summary, ok: 'Schedule' });
      if (!ok) return 'The user declined to schedule the task.';
      try {
        await skRequest('POST', '/api/studio/schedule', { name: args.name, prompt: args.prompt, when: args.when, mode: args.mode || 'chat', folder: args.folder || '', notify: args.notify === true, deliver: String(args.deliver || '') });
        scRefreshPane();
        return `Scheduled "${args.name}" (${args.when}). Next run: ${scWhen(check.next[0])}.`;
      } catch (e) {
        return `Error: ${e.message}`;
      }
    };

    // ---- wiring ---------------------------------------------------------------------------------------------
    if ($('spane-skills')) {
      $('settingsMenuNav').addEventListener('click', (e) => {
        if (!e.target.closest('[data-spane="skills"]')) return;
        const root = skWorkspace();
        if (root && !$('skWorkspace').value) $('skWorkspace').value = root;
        skRefreshPane();
        scRefreshPane();
        if (window.fllmSessionIndexInfo) window.fllmSessionIndexInfo();
      });
      $('skList').addEventListener('click', (e) => {
        const edit = e.target.closest('[data-sk-edit]');
        if (edit) skEdit(edit.dataset.skEdit, edit.dataset.skScope);
        const del = e.target.closest('[data-sk-del]');
        if (del) skDelete(del.dataset.skDel, del.dataset.skScope);
      });
      $('skWorkspace').addEventListener('change', skRefreshPane);
      $('btnSkNew').addEventListener('click', () => { skCurrent = null; $('skName').value = ''; $('skScope').value = 'user'; $('skText').value = skTemplate(); });
      $('btnSkSave').addEventListener('click', skSave);
      $('skText').addEventListener('input', () => {
        const name = skParse($('skText').value).meta.name;
        if (name && !$('skName').value && skValidName(name)) $('skName').value = name;
      });
      $('scList').addEventListener('click', async (e) => {
        const btn = e.target.closest('[data-sc-act]');
        if (!btn) return;
        const card = btn.closest('[data-sc]');
        const id = card.dataset.sc;
        const task = (window.__scTasks || []).find((t) => t.id === id);
        if (btn.dataset.scAct === 'results') scShowResults(card, id);
        if (btn.dataset.scAct === 'edit' && task) scFill(task);
        if (btn.dataset.scAct === 'run') {
          try { await skRequest('POST', '/api/studio/schedule/run', { id }); toast('Task started'); setTimeout(scRefreshPane, 400); } catch (err) { toast(err.message); }
        }
        if (btn.dataset.scAct === 'remove') {
          if (!(await askDialog({ title: `Delete the task "${task ? task.name : ''}"?`, text: 'Its saved results are deleted too.', ok: 'Delete', danger: true }))) return;
          try { await skRequest('DELETE', `/api/studio/schedule?${skQuery({ id })}`); scRefreshPane(); } catch (err) { toast(err.message); }
        }
      });
      $('scMode').addEventListener('change', scFolderVisibility);
      $('scWhenInput').addEventListener('input', () => { clearTimeout(scPreview.t); scPreview.t = setTimeout(scPreview, 350); });
      $('scWhenPresets').addEventListener('change', (e) => { if (e.target.value) { $('scWhenInput').value = e.target.value; e.target.value = ''; scPreview(); } });
      $('scNotify').addEventListener('change', (e) => {
        if (e.target.checked && typeof Notification !== 'undefined' && Notification.permission === 'default') Notification.requestPermission();
      });
      $('btnScSave').addEventListener('click', scSave);
      $('btnScClear').addEventListener('click', () => scFill(null));
      window.setInterval(scPoll, 60000);
      window.fllmSkillIndexRefresh();
    }
