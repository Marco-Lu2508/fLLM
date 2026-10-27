// ==========================================================================
    // Plugins: one folder that adds skills, sub-agents, commands and hooks at once.
    //  - plugins/<name>/plugin.json + optional skills/, agents/, commands/, hooks.json
    //  - Contributions are read from the plugin folder every time; removing the plugin removes everything it added.
    //  - Routes: GET/POST/DELETE /api/studio/plugins, GET /api/studio/plugins/commands
    // ==========================================================================

    (function pluginsSettings() {
      const nav = document.getElementById('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      if (!nav || !content) return;

      const button = document.createElement('button');
      button.className = 'settings-nav-btn';
      button.dataset.spane = 'plugins';
      button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><path d="M12 2v6"/><path d="M5 8h14l-1 12H6z"/><path d="M9 12v4"/><path d="M15 12v4"/></svg><span>Plugins</span>';
      nav.appendChild(button);

      const pane = document.createElement('div');
      pane.className = 'settings-pane';
      pane.id = 'spane-plugins';
      pane.innerHTML =
        '<div class="app-card"><div class="app-card-title">Plugins</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;">A plugin is a folder that brings several things at once: skills (<span class="mono">skills/&lt;name&gt;/SKILL.md</span>), sub-agents (<span class="mono">agents/&lt;name&gt;.md</span>), slash commands (<span class="mono">commands/&lt;name&gt;.md</span>) and hooks (<span class="mono">hooks.json</span>), described by <span class="mono">plugin.json</span>. Nothing is copied into your own files: the contributions stay in the plugin folder, so removing the plugin removes everything it added.</div>' +
        '<div id="pluginList"></div>' +
        '<div style="display:flex;gap:6px;align-items:center;margin-top:12px;flex-wrap:wrap;">' +
        '  <input class="ui-input mono" id="pluginPath" placeholder="/home/you/plugins/my-plugin" style="flex:1;min-width:260px;" autocomplete="off">' +
        '  <button class="btn btn-sm" id="btnPluginInstall">Install from folder</button>' +
        '</div>' +
        '<div style="display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap;">' +
        '  <input class="ui-input mono" id="pluginUrl" placeholder="https://github.com/you/your-plugin" style="flex:1;min-width:260px;" autocomplete="off">' +
        '  <input class="ui-input mono" id="pluginRef" placeholder="branch (optional)" style="width:150px;" autocomplete="off">' +
        '  <button class="btn btn-sm" id="btnPluginClone">Install from repository</button>' +
        '</div>' +
        '<div class="app-item-desc mono" id="pluginFolder" style="margin-top:8px;font-size:11px;"></div>' +
        '<div class="app-item-desc mono" id="pluginStatus" style="margin-top:6px;font-size:11px;white-space:pre-wrap;"></div>' +
        '</div>';
      content.appendChild(pane);

      const list = pane.querySelector('#pluginList');
      const status = pane.querySelector('#pluginStatus');
      const say = (text) => { status.textContent = text || ''; };

      async function refresh() {
        try {
          const res = await apiFetch('/api/studio/plugins');
          const data = await res.json();
          pane.querySelector('#pluginFolder').textContent = data.folder ? `Plugins live in ${data.folder}` : '';
          const plugins = Array.isArray(data.plugins) ? data.plugins : [];
          list.innerHTML = plugins.length ? plugins.map((p) => {
            const parts = [];
            if (p.skills) parts.push(`${p.skills} skill(s)`);
            if (p.agents) parts.push(`${p.agents} sub-agent(s)`);
            if (p.commands) parts.push(`${p.commands} command(s)`);
            if (p.hooks) parts.push(`${p.hooks} hook(s)`);
            return '<div class="app-card-item column" data-plugin="' + pluginsAttr(p.name) + '" style="border-bottom:1px solid var(--border);padding-bottom:8px;">' +
              '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">' +
              '  <span class="app-item-title">' + pluginsEsc(p.name) + '</span>' +
              (p.version ? '  <span class="app-item-desc">v' + pluginsEsc(p.version) + '</span>' : '') +
              '  <span style="flex:1"></span>' +
              '  <button class="btn btn-xs btn-ghost" data-plugin-act="remove">Remove</button>' +
              '</div>' +
              (p.description ? '<div class="app-item-desc">' + pluginsEsc(p.description) + '</div>' : '') +
              '<div class="app-item-desc">' + (parts.length ? parts.join(' · ') : 'nothing contributed yet') +
              (p.author ? ' · by ' + pluginsEsc(p.author) : '') + '</div>' +
              '<div class="app-item-desc mono" style="font-size:11px;word-break:break-all;">' + pluginsEsc(p.path) + '</div>' +
              '</div>';
          }).join('') : '<div class="app-item-desc">No plugins installed.</div>';
        } catch (e) {
          say('Could not read the plugins: ' + ((e && e.message) || e));
        }
      }

      function pluginsEsc(text) { return String(text == null ? '' : text).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
      function pluginsAttr(text) { return pluginsEsc(text).replace(/"/g, '&quot;'); }

      list.addEventListener('click', async (e) => {
        const act = e.target.dataset && e.target.dataset.pluginAct;
        if (act !== 'remove') return;
        const row = e.target.closest('[data-plugin]');
        if (!row) return;
        const name = row.dataset.plugin;
        if (!window.confirm(`Remove the plugin "${name}" and everything it contributes?`)) return;
        try {
          const res = await apiFetch('/api/studio/plugins', { method: 'DELETE', body: { name } });
          const data = await res.json();
          if (!res.ok) { say(data.error && data.error.message ? data.error.message : 'Could not remove it.'); return; }
          say(`Removed ${name}.`);
          await refresh();
        } catch (err) {
          say('Could not remove it: ' + ((err && err.message) || err));
        }
      });

      pane.querySelector('#btnPluginInstall').addEventListener('click', async () => {
        const path = pane.querySelector('#pluginPath').value.trim();
        if (!path) { say('Give the absolute path of the plugin folder.'); return; }
        say('Installing ...');
        try {
          const res = await apiFetch('/api/studio/plugins', { method: 'POST', body: { path } });
          const data = await res.json();
          if (!res.ok) { say(data.error && data.error.message ? data.error.message : 'Could not install it.'); return; }
          pane.querySelector('#pluginPath').value = '';
          say('Installed. Its skills, agents, commands and hooks are available now.');
          await refresh();
        } catch (err) {
          say('Could not install it: ' + ((err && err.message) || err));
        }
      });

      pane.querySelector('#btnPluginClone').addEventListener('click', async () => {
        const url = pane.querySelector('#pluginUrl').value.trim();
        const ref = pane.querySelector('#pluginRef').value.trim();
        if (!url) { say('Give the address of a repository with a plugin.json.'); return; }
        say('Cloning ...');
        try {
          const res = await apiFetch('/api/studio/plugins', { method: 'POST', body: { url, ref } });
          const data = await res.json();
          if (!res.ok) { say(data.error && data.error.message ? data.error.message : 'Could not install it.'); return; }
          pane.querySelector('#pluginUrl').value = '';
          pane.querySelector('#pluginRef').value = '';
          say('Installed from the repository. Its skills, agents, commands and hooks are available now.');
          await refresh();
        } catch (err) {
          say('Could not install it: ' + ((err && err.message) || err));
        }
      });

      refresh();
    })();
