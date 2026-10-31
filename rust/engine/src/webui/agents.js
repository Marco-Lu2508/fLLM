// ==========================================================================
    // Sub-agents: Markdown files in `agents/` (Studio folder) and `.fllm/agents/` of a workspace folder.
    //  - The coding agent hands a job to a sub-agent by name with the `task` tool; each one brings its own instructions,
    //    tool list, model and step budget (see docs/user/studio.md).
    //  - Files: <name>.md with front matter (name, description, tools, model, max_steps) and the instructions.
    //  - Routes: GET/POST/DELETE /api/studio/agents
    //  - The page: a list (built-in and own sub-agents), a form for a new or an existing one, and a way to try the instructions.
    // ==========================================================================

    // @pure-begin
    const AGENT_BUILTINS = [
      { name: 'explore', description: 'Reads and searches the project and reports what it found. It cannot change anything.' },
      { name: 'general', description: 'Does a whole job on its own: it may also edit files and run commands.' },
    ];
    const AGENT_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
    const AGENT_HIDDEN_TOOLS = new Set(['task', 'spawn_agent', 'create_agent']);

    /** The tools a sub-agent can be given, in two groups: what only reads (the default) and what changes things. */
    function agentToolGroups(tools) {
      const read = [];
      const act = [];
      for (const t of tools || []) {
        if (!t || !t.name || AGENT_HIDDEN_TOOLS.has(t.name) || t.permission === 'computer') continue;
        (t.permission === 'read' ? read : act).push({ name: t.name, description: String(t.description || '').split('\n')[0].slice(0, 140) });
      }
      const byName = (a, b) => a.name.localeCompare(b.name);
      return { read: read.sort(byName), act: act.sort(byName) };
    }

    /** Which boxes start checked: the agent's own tool list, or (new agent, or an old one without a list) everything that only reads. */
    function agentCheckedTools(agent, groups) {
      return agent && agent.tools && agent.tools.length ? new Set(agent.tools) : new Set(groups.read.map((t) => t.name));
    }

    /** What the form sends, or `{error}` in plain words. `form`: name, description, body, tools (array), model, max_steps, scope. */
    function agentPayload(form, workspace, isNew, existingNames) {
      const name = String(form.name || '').trim();
      if (!AGENT_NAME_RE.test(name)) return { error: 'Give the sub-agent a short name: lowercase letters, digits, - and _ (for example code-reviewer).' };
      if (AGENT_BUILTINS.some((b) => b.name === name)) return { error: `"${name}" is a built-in sub-agent. Pick another name.` };
      if (isNew && (existingNames || []).includes(name)) return { error: `You already have a sub-agent called "${name}". Pick another name or edit that one.` };
      const body = String(form.body || '').trim();
      if (!body) return { error: 'Write the instructions: what this sub-agent does and what its report should contain.' };
      const scope = form.scope === 'workspace' ? 'workspace' : 'user';
      if (scope === 'workspace' && !workspace) return { error: 'Open a folder on the Code page first, or store the sub-agent for all folders.' };
      const steps = Math.floor(Number(form.max_steps) || 0);
      return {
        payload: {
          name, description: String(form.description || '').replace(/\s+/g, ' ').trim(), body,
          tools: [...new Set((form.tools || []).map((t) => String(t).trim()).filter(Boolean))],
          model: String(form.model || '').trim(), max_steps: Math.max(0, Math.min(60, steps)), scope, workspace: workspace || null,
        },
      };
    }

    /** A line about an agent's reach for the list. */
    function agentReach(agent, groups) {
      const tools = agent.tools || [];
      if (!tools.length) return 'reads and searches only';
      const actNames = new Set((groups ? groups.act : []).map((t) => t.name));
      const writes = tools.filter((t) => actNames.has(t) || /^(write|edit|multi_edit|patch|bash)/.test(t));
      return writes.length ? `can change files or run commands (${tools.length} tools)` : `reads and searches (${tools.length} tools)`;
    }
    // @pure-end

    (function agentsSettings() {
      const nav = document.getElementById('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      if (!nav || !content) return;

      const button = document.createElement('button');
      button.className = 'settings-nav-btn';
      button.dataset.spane = 'agents';
      button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><circle cx="9" cy="8" r="3"/><circle cx="17" cy="10" r="2"/><path d="M3 20c0-3 3-5 6-5s6 2 6 5"/><path d="M15 20c0-2 2-3 4-3s3 1 3 2.5"/></svg><span>Sub-agents</span>';
      nav.appendChild(button);

      const pane = document.createElement('div');
      pane.className = 'settings-pane';
      pane.id = 'spane-agents';
      pane.innerHTML =
        '<div class="app-card">' +
        '  <div class="agents-head"><div><div class="app-card-title">Sub-agents</div>' +
        '  <div class="app-item-desc">Helpers that the coding agent can hand a job to. Each works on its own with its own instructions and reports back.</div></div>' +
        '  <button class="btn btn-sm btn-primary" id="btnAgentNew">New sub-agent</button></div>' +
        '  <div id="agentList"></div>' +
        '  <div id="agentFormWrap"></div>' +
        '  <div class="app-item-desc agents-stored" id="agentStored"></div>' +
        '</div>';
      content.appendChild(pane);

      const listEl = pane.querySelector('#agentList');
      const formWrap = pane.querySelector('#agentFormWrap');
      const storedEl = pane.querySelector('#agentStored');
      const esc = (text) => String(text == null ? '' : text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

      let agents = [];
      let folders = {};
      let editing = null; // { agent, isNew }
      let toolGroups = { read: [], act: [] };
      let notice = '';

      const workspace = () => (window.fllmCodeWorkspace && window.fllmCodeWorkspace()) || null;

      async function loadTools() {
        try {
          const res = await apiFetch('/api/studio/agent/tools');
          if (!res.ok) return;
          const data = await res.json();
          const list = (Array.isArray(data) ? data : (data.tools || [])).map((t) => {
            const f = t.function || t;
            return { name: f.name, permission: t.permission || f.permission || 'command', description: f.description || '' };
          });
          toolGroups = agentToolGroups(list);
        } catch { /* the form then lists no tools; the agent keeps its saved list */ }
      }

      function renderList() {
        const own = agents.filter((a) => !a.read_only);
        const plugin = agents.filter((a) => a.read_only);
        const builtin = AGENT_BUILTINS.map((b) => `
          <div class="agent-row"><div class="agent-row-main"><div class="agent-row-name">${esc(b.name)} <span class="quant-chip">built in</span></div>
          <div class="app-item-desc">${esc(b.description)}</div></div></div>`).join('');
        const ownRows = own.map((a) => `
          <div class="agent-row"><div class="agent-row-main"><div class="agent-row-name">${esc(a.name)}${a.scope === 'workspace' ? ' <span class="quant-chip">this folder only</span>' : ''}</div>
          <div class="app-item-desc">${esc(a.description || 'No description yet: say when the coding agent should use it.')}</div>
          <div class="app-item-desc agent-reach">${esc(agentReach(a, toolGroups))}${a.model ? ` · model ${esc(a.model)}` : ''}</div></div>
          <div class="agent-row-actions"><button class="btn btn-xs btn-ghost" data-agent-edit="${esc(a.name)}" data-agent-scope="${esc(a.scope)}">Edit</button>
          <button class="btn btn-xs btn-ghost btn-danger-text" data-agent-del="${esc(a.name)}" data-agent-scope="${esc(a.scope)}">Delete</button></div></div>`).join('');
        const pluginRows = plugin.map((a) => `
          <div class="agent-row"><div class="agent-row-main"><div class="agent-row-name">${esc(a.name)} <span class="quant-chip">from a plugin</span></div>
          <div class="app-item-desc">${esc(a.description || '')}</div></div></div>`).join('');
        listEl.innerHTML =
          `<div class="agents-section">Built in</div>${builtin}` +
          `<div class="agents-section">Your sub-agents</div>` +
          (ownRows || '<div class="app-item-desc agents-empty">You have not created one yet. A sub-agent is useful for a job you repeat, for example reviewing a change or writing tests. Choose New sub-agent to make one.</div>') +
          (pluginRows ? `<div class="agents-section">From plugins</div>${pluginRows}` : '') +
          (notice ? `<div class="agents-notice" role="status">${esc(notice)}</div>` : '');
        storedEl.innerHTML = folders.user ? `Stored as one file per sub-agent in <button class="agents-path" id="btnAgentPath" title="${esc(folders.user)}">the agents folder</button> (click to copy the path).` : '';
      }

      function renderForm() {
        if (!editing) { formWrap.innerHTML = ''; return; }
        const a = editing.agent;
        const checked = agentCheckedTools(a, toolGroups);
        // tools the file names but this build does not list (an MCP tool, an older name): they stay checkable
        const known = new Set([...toolGroups.read, ...toolGroups.act].map((t) => t.name));
        const extra = (a.tools || []).filter((t) => !known.has(t)).map((name) => ({ name, description: '' }));
        const box = (t) => `<label class="agent-tool" title="${esc(t.description)}"><input type="checkbox" data-agent-tool="${esc(t.name)}"${checked.has(t.name) ? ' checked' : ''}><span class="mono">${esc(t.name)}</span></label>`;
        const ws = workspace();
        const models = new Set();
        if (state.status && state.status.model) { if (state.status.model.name) models.add(state.status.model.name); (state.status.model.aliases || []).forEach((x) => models.add(x)); }
        formWrap.innerHTML = `
          <form class="agent-form" id="agentForm" autocomplete="off">
            <div class="agent-form-title">${editing.isNew ? 'New sub-agent' : `Edit ${esc(a.name)}`}</div>
            <label class="agent-field"><span>Name</span><input class="ui-input mono" data-f="name" value="${esc(a.name)}" placeholder="code-reviewer" maxlength="64"${editing.isNew ? '' : ' disabled'}>
              <small>The coding agent picks it by this name. Lowercase letters, digits, - and _.</small></label>
            <label class="agent-field"><span>When to use it</span><input class="ui-input" data-f="description" value="${esc(a.description)}" placeholder="Reviews a change for bugs and missing tests">
              <small>One sentence. The coding agent reads it to decide whether this sub-agent fits a job.</small></label>
            <label class="agent-field"><span>Instructions</span><textarea class="ui-input" data-f="body" rows="8" placeholder="You review code changes. Read the diff, name the file and line of each problem and explain why it is one. End with a short list of what to fix first.">${esc(a.body)}</textarea>
              <small>The system prompt of this sub-agent. It does not see your conversation, only the job it is given.</small></label>
            <fieldset class="agent-field"><legend>Allowed tools</legend>
              <div class="agent-tools-label">Read and search</div><div class="agent-tools">${toolGroups.read.map(box).join('') || '<span class="app-item-desc">No tool list available yet.</span>'}</div>
              ${toolGroups.act.length || extra.length ? `<div class="agent-tools-label">Change files and run commands</div><div class="agent-tools">${[...toolGroups.act, ...extra].map(box).join('')}</div>` : ''}
              <small>A sub-agent with only reading tools cannot change your files.</small></fieldset>
            <div class="agent-field-row">
              <label class="agent-field"><span>Model</span><input class="ui-input mono" data-f="model" list="agentModels" value="${esc(a.model)}" placeholder="Same as the coding agent">
                <datalist id="agentModels">${[...models].map((m) => `<option value="${esc(m)}">`).join('')}</datalist><small>Leave empty to use the model that runs the session.</small></label>
              <label class="agent-field"><span>Step limit</span><input class="ui-input mono" type="number" min="0" max="60" data-f="max_steps" value="${Number(a.max_steps) || 0}">
                <small>Most tool calls before it must report. 0 means the default (16).</small></label>
            </div>
            <div class="agent-field"><span>Available in</span>
              <label class="agent-radio"><input type="radio" name="agentScope" value="user"${a.scope === 'workspace' ? '' : ' checked'}> All folders</label>
              <label class="agent-radio"><input type="radio" name="agentScope" value="workspace"${a.scope === 'workspace' ? ' checked' : ''}${ws ? '' : ' disabled'}> Only the open folder${ws ? '' : ' (open a folder on the Code page first)'}</label></div>
            <div class="agent-actions">
              <button class="btn btn-sm btn-primary" type="submit">${editing.isNew ? 'Create sub-agent' : 'Save changes'}</button>
              <button class="btn btn-sm btn-ghost" type="button" id="btnAgentCancel">Cancel</button>
              <span class="app-item-desc" id="agentStatus" role="status"></span>
            </div>
            <details class="agent-try"><summary>Try the instructions</summary>
              <div class="app-item-desc">Sends a sample job to the loaded model with these instructions. It has no tools here, so it can only answer in words.</div>
              <textarea class="ui-input" id="agentTryJob" rows="2" placeholder="Review this function: function add(a, b) { return a - b; }"></textarea>
              <button class="btn btn-sm btn-ghost" type="button" id="btnAgentTry">Run the sample</button>
              <pre class="agent-try-out" id="agentTryOut" hidden></pre></details>
          </form>`;
      }

      function collect() {
        const form = formWrap.querySelector('#agentForm');
        const get = (f) => { const el = form.querySelector(`[data-f="${f}"]`); return el ? el.value : ''; };
        const scope = form.querySelector('input[name="agentScope"]:checked');
        return {
          name: editing.isNew ? get('name') : editing.agent.name, description: get('description'), body: get('body'), model: get('model'),
          max_steps: get('max_steps'), scope: scope ? scope.value : 'user',
          tools: [...form.querySelectorAll('[data-agent-tool]')].filter((c) => c.checked).map((c) => c.dataset.agentTool),
        };
      }

      const say = (text) => { const el = formWrap.querySelector('#agentStatus'); if (el) el.textContent = text || ''; };

      async function load() {
        const root = workspace();
        const query = root ? `?workspace=${encodeURIComponent(root)}` : '';
        try {
          const res = await apiFetch(`/api/studio/agents${query}`);
          if (!res.ok) throw new Error(`the server answered with error ${res.status}`);
          const data = await res.json();
          agents = Array.isArray(data.agents) ? data.agents : [];
          folders = data.folders || {};
        } catch (e) {
          notice = `The sub-agents could not be loaded: ${(e && e.message) || e}`;
        }
        renderList();
      }

      async function save(ev) {
        ev.preventDefault();
        const names = agents.map((a) => a.name);
        const result = agentPayload(collect(), workspace(), editing.isNew, names);
        if (result.error) { say(result.error); return; }
        try {
          const res = await apiFetch('/api/studio/agents', { method: 'POST', body: result.payload });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { say((data.error && data.error.message) || 'The sub-agent could not be saved.'); return; }
          notice = editing.isNew ? `Created "${result.payload.name}". The coding agent can use it from its next run.` : `Saved "${result.payload.name}".`;
          editing = null;
          renderForm();
          await load();
          if (typeof window.fllmReloadAgents === 'function') window.fllmReloadAgents();
        } catch (e) {
          say(`The sub-agent could not be saved: ${(e && e.message) || e}`);
        }
      }

      async function remove(name, scope) {
        const ok = typeof confirmDialog === 'function' ? await confirmDialog(`Delete the sub-agent "${name}"? Its file is removed.`) : true;
        if (!ok) return;
        const root = workspace();
        const query = `?name=${encodeURIComponent(name)}&scope=${encodeURIComponent(scope || 'user')}` + (root ? `&workspace=${encodeURIComponent(root)}` : '');
        try {
          const res = await apiFetch(`/api/studio/agents${query}`, { method: 'DELETE' });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { notice = (data.error && data.error.message) || 'The sub-agent could not be deleted.'; renderList(); return; }
          notice = `Deleted "${name}".`;
          if (editing && editing.agent.name === name) { editing = null; renderForm(); }
          await load();
          if (typeof window.fllmReloadAgents === 'function') window.fllmReloadAgents();
        } catch (e) {
          notice = `The sub-agent could not be deleted: ${(e && e.message) || e}`;
          renderList();
        }
      }

      async function tryIt() {
        const out = formWrap.querySelector('#agentTryOut');
        const job = formWrap.querySelector('#agentTryJob').value.trim();
        const form = collect();
        if (!form.body.trim()) { say('Write the instructions first.'); return; }
        if (!job) { say('Describe a sample job to try.'); return; }
        if (state.setup) { say('Load a model first (Models page).'); return; }
        out.hidden = false;
        out.textContent = 'Working ...';
        try {
          const model = form.model.trim() || state.selectedModel || (state.status && state.status.model && state.status.model.name) || '';
          const ep = window.fllmChatEndpoint ? window.fllmChatEndpoint(model) : { url: '/v1/chat/completions', transform: (b) => b };
          const res = await apiFetch(ep.url, {
            method: 'POST',
            body: ep.transform({ model, stream: false, temperature: 0.3, max_tokens: 400, x_fllm_clamp_max_tokens: true, messages: [{ role: 'system', content: form.body }, { role: 'user', content: job }] }),
          });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { out.textContent = (data.error && data.error.message) || `The server answered with error ${res.status}.`; return; }
          out.textContent = (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content || '').trim() || '(no answer)';
        } catch (e) {
          out.textContent = `The sample could not run: ${(e && e.message) || e}`;
        }
      }

      async function startEdit(agent, isNew) {
        notice = '';
        await loadTools();
        editing = { agent: { name: '', description: '', body: '', model: '', tools: [], max_steps: 0, scope: 'user', ...agent }, isNew };
        renderList();
        renderForm();
        const first = formWrap.querySelector(isNew ? '[data-f="name"]' : '[data-f="description"]');
        if (first) { first.focus(); first.scrollIntoView({ block: 'nearest' }); }
      }

      pane.querySelector('#btnAgentNew').addEventListener('click', () => startEdit({}, true));
      pane.addEventListener('click', (e) => {
        const edit = e.target.closest('[data-agent-edit]');
        if (edit) { const a = agents.find((x) => x.name === edit.dataset.agentEdit && x.scope === edit.dataset.agentScope); if (a) startEdit(a, false); return; }
        const del = e.target.closest('[data-agent-del]');
        if (del) { remove(del.dataset.agentDel, del.dataset.agentScope); return; }
        if (e.target.closest('#btnAgentCancel')) { editing = null; renderForm(); return; }
        if (e.target.closest('#btnAgentTry')) { tryIt(); return; }
        if (e.target.closest('#btnAgentPath')) {
          if (navigator.clipboard) navigator.clipboard.writeText(folders.user || '').then(() => { notice = 'Path copied.'; renderList(); }).catch(() => {});
        }
      });
      pane.addEventListener('submit', (e) => { if (e.target.id === 'agentForm') save(e); });

      loadTools().then(load);
      // the list follows the open folder: it is read again when this page is shown
      new MutationObserver(() => { if (pane.classList.contains('active')) load(); }).observe(pane, { attributes: true, attributeFilter: ['class'] });
    })();
