// ==========================================================================
    // Sub-agents: Markdown files in `agents/` (Studio folder) and `.fllm/agents/` of a workspace folder.
    //  - The coding agent delegates to them by name with the `task` tool; each one brings its own instructions,
    //    tool list, model and step budget (see docs/user/studio.md).
    //  - Files: <name>.md with front matter (name, description, tools, model, max_steps) and the instructions.
    //  - Routes: GET/POST/DELETE /api/studio/agents
    // ==========================================================================

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
        '<div class="app-card"><div class="app-card-title">Sub-agents</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;">A sub-agent is a worker with a fresh context: the coding agent hands it a self-contained job with <span class="mono">task</span> and gets back only its report. <span class="mono">explore</span> (read only) and <span class="mono">general</span> are built in; the ones you define here appear in that tool by name and can use their own tool list, model and step budget.</div>' +
        '<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;">' +
        '  <select class="ui-input" id="selAgent" style="min-width:200px;"></select>' +
        '  <button class="btn btn-sm btn-ghost" id="btnAgentNew">New</button>' +
        '  <button class="btn btn-sm btn-ghost" id="btnAgentDelete">Delete</button>' +
        '  <span class="app-item-desc mono" id="agentFolder" style="margin-left:auto;font-size:11px;"></span>' +
        '</div>' +
        '<div id="agentForm"></div>' +
        '<div class="app-card-item" style="display:flex;gap:8px;align-items:center;margin-top:10px;">' +
        '  <button class="btn btn-sm" id="btnAgentSave">Save</button>' +
        '  <span class="app-item-desc mono" id="agentStatus" style="font-size:11px;"></span>' +
        '</div>' +
        '</div>';
      content.appendChild(pane);

      const sel = pane.querySelector('#selAgent');
      const form = pane.querySelector('#agentForm');
      const status = pane.querySelector('#agentStatus');
      const say = (text) => { status.textContent = text || ''; };

      let agents = [];
      let folders = {};
      let current = null; // {name, description, tools, model, max_steps, scope, body, isNew}

      const workspace = () => (window.fllmCodeWorkspace && window.fllmCodeWorkspace()) || null;

      function renderForm() {
        if (!current) { form.innerHTML = '<div class="app-item-desc">No sub-agent selected.</div>'; return; }
        form.innerHTML =
          row('Name', '<input class="ui-input mono" data-f="name" value="' + agentsAttr(current.name) + '" placeholder="reviewer" style="width:220px;"' + (current.isNew ? '' : ' disabled') + '>') +
          row('Description', '<input class="ui-input" data-f="description" value="' + agentsAttr(current.description || '') + '" placeholder="When should the agent use it" style="width:420px;">') +
          row('Tools', '<input class="ui-input mono" data-f="tools" value="' + agentsAttr((current.tools || []).join(', ')) + '" placeholder="read_file, grep, glob (empty = all tools of the session)" style="width:420px;">') +
          row('Model', '<input class="ui-input mono" data-f="model" value="' + agentsAttr(current.model || '') + '" placeholder="empty = the model of the session, or e.g. qwen3.5:9b" style="width:320px;">') +
          row('Max steps', '<input class="ui-input mono" type="number" min="0" max="60" data-f="max_steps" value="' + (current.max_steps || 0) + '" style="width:90px;" title="0 = the built-in limit for sub-agents">') +
          row('Scope', '<select class="ui-input" data-f="scope"><option value="user"' + (current.scope === 'workspace' ? '' : ' selected') + '>This computer (all folders)</option><option value="workspace"' + (current.scope === 'workspace' ? ' selected' : '') + '>This workspace (.fllm/agents)</option></select>') +
          '<div class="app-card-item" style="display:block;"><div class="app-item-title" style="margin-bottom:4px;">Instructions</div>' +
          '<textarea class="ui-input mono" data-f="body" rows="10" style="width:100%;resize:vertical;" placeholder="You review changes. Read the diff, name file and line, explain the bug.">' + agentsEsc(current.body || '') + '</textarea></div>';
      }

      function row(label, control) {
        return '<div class="app-card-item" style="display:flex;gap:8px;align-items:center;">' +
          '<div class="app-item-info" style="min-width:110px;"><div class="app-item-title">' + label + '</div></div>' + control + '</div>';
      }
      function agentsEsc(text) { return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
      function agentsAttr(text) { return agentsEsc(text).replace(/"/g, '&quot;'); }

      function collectForm() {
        if (!current) return;
        form.querySelectorAll('[data-f]').forEach((el) => {
          const key = el.dataset.f;
          if (key === 'max_steps') current[key] = Number(el.value) || 0;
          else if (key === 'tools') current[key] = el.value.split(',').map((s) => s.trim()).filter(Boolean);
          else current[key] = el.value;
        });
      }

      async function load(selectName) {
        const root = workspace();
        const query = root ? `?workspace=${encodeURIComponent(root)}` : '';
        try {
          const res = await apiFetch(`/api/studio/agents${query}`);
          const data = await res.json();
          agents = Array.isArray(data.agents) ? data.agents : [];
          folders = data.folders || {};
          pane.querySelector('#agentFolder').textContent = folders.user || '';
          sel.innerHTML = agents.length
            ? agents.map((a) => `<option value="${agentsAttr(a.name)}"${a.scope === 'workspace' ? ' data-ws="1"' : ''}>${agentsEsc(a.name)}${a.scope === 'workspace' ? ' (workspace)' : ''}</option>`).join('')
            : '<option value="">(none yet)</option>';
          const pick = selectName || (agents[0] && agents[0].name) || '';
          if (pick) selectAgent(pick);
          else { current = null; renderForm(); }
          say('');
        } catch (e) {
          say('Could not load the sub-agents: ' + ((e && e.message) || e));
        }
      }

      function selectAgent(name) {
        const found = agents.find((a) => a.name === name);
        sel.value = name;
        current = found ? { ...found, isNew: false } : null;
        renderForm();
      }

      async function save() {
        if (!current) return;
        collectForm();
        const payload = {
          name: current.name, description: current.description, tools: current.tools, model: current.model,
          max_steps: current.max_steps, body: current.body, scope: current.scope || 'user', workspace: workspace(),
        };
        try {
          const res = await apiFetch('/api/studio/agents', { method: 'POST', body: payload });
          const data = await res.json();
          if (!res.ok) { say(data.error && data.error.message ? data.error.message : 'Could not save.'); return; }
          say('Saved. The coding agent offers it from the next run.');
          await load(current.name);
        } catch (e) {
          say('Could not save: ' + ((e && e.message) || e));
        }
      }

      async function remove() {
        if (!current || current.isNew) return;
        const root = workspace();
        const query = `?name=${encodeURIComponent(current.name)}&scope=${encodeURIComponent(current.scope || 'user')}` + (root ? `&workspace=${encodeURIComponent(root)}` : '');
        try {
          const res = await apiFetch(`/api/studio/agents${query}`, { method: 'DELETE' });
          const data = await res.json();
          if (!res.ok) { say(data.error && data.error.message ? data.error.message : 'Could not delete.'); return; }
          say('Deleted.');
          current = null;
          await load();
        } catch (e) {
          say('Could not delete: ' + ((e && e.message) || e));
        }
      }

      sel.addEventListener('change', () => selectAgent(sel.value));
      pane.querySelector('#btnAgentNew').addEventListener('click', () => {
        current = { name: '', description: '', tools: [], model: '', max_steps: 0, scope: 'user', body: '', isNew: true };
        sel.value = '';
        renderForm();
        say('New sub-agent. Give it a name (lowercase, digits, - and _) and instructions.');
      });
      pane.querySelector('#btnAgentSave').addEventListener('click', save);
      pane.querySelector('#btnAgentDelete').addEventListener('click', remove);

      load();
    })();
