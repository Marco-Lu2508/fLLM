    // ==========================================================================
    // Settings: Tools (MCP). Servers are programs started on this computer; the list lives in mcp.json.
    // ==========================================================================
    let mcpServers = [];
    let mcpEditing = null; // the server whose saved environment variables are kept when a value is left empty
    let mcpTimer = null;

    async function mcpRequest(method, path, body) {
      const res = await apiFetch(path, body === undefined ? { method } : { method, body });
      let data = {};
      try { data = await res.json(); } catch (e) { /* no body */ }
      if (!res.ok) throw new Error((data.error && data.error.message) || `Request failed (${res.status})`);
      return data;
    }

    function mcpRender() {
      const box = $('mcpServerList');
      if (!mcpServers.length) {
        box.innerHTML = '<div class="app-item-desc">No MCP servers yet. Add one below.</div>';
        return;
      }
      box.innerHTML = mcpServers.map((s) => {
        const color = s.state === 'ready' ? 'var(--primary)' : (s.state === 'failed' ? 'var(--destructive,#d33)' : 'var(--muted-foreground)');
        const tools = (s.tools || []).map((t) =>
          `<li><code>${escapeHtml(t.name)}</code> <span class="app-item-desc">${escapeHtml(t.description || '')}</span></li>`).join('');
        const line = [s.command].concat(s.args || []).join(' ');
        return `<div class="app-card-item column" data-mcp="${escapeHtml(s.name)}" style="border-bottom:1px solid var(--border);padding-bottom:10px;">
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <span class="app-item-title">${escapeHtml(s.name)}</span>
            <span style="color:${color};font-size:12px;">${escapeHtml(s.state)}${s.state === 'ready' ? ` - ${s.tool_count} tool${s.tool_count === 1 ? '' : 's'}` : ''}</span>
            <span style="flex:1"></span>
            <button class="btn btn-xs btn-ghost" data-mcp-act="restart">Restart</button>
            <button class="btn btn-xs btn-ghost" data-mcp-act="edit">Edit</button>
            <button class="btn btn-xs btn-ghost" data-mcp-act="remove">Remove</button>
          </div>
          <div class="app-item-desc mono" style="word-break:break-all;">${escapeHtml(line)}</div>
          ${s.message ? `<div class="app-item-desc" style="color:var(--destructive,#d33);">${escapeHtml(s.message)}</div>` : ''}
          ${tools ? `<details><summary class="app-item-desc" style="cursor:pointer;">Tools</summary><ul style="margin:6px 0 0 18px;">${tools}</ul></details>` : ''}
          ${s.stderr ? `<details><summary class="app-item-desc" style="cursor:pointer;">Error output of the server</summary><pre style="max-height:160px;overflow:auto;">${escapeHtml(s.stderr)}</pre></details>` : ''}
        </div>`;
      }).join('');
    }

    async function mcpRefresh() {
      try {
        const data = await mcpRequest('GET', '/api/studio/mcp/servers');
        mcpServers = data.servers || [];
        $('mcpFileInfo').textContent = `Saved in ${data.config_file}` + (data.config_error ? ` - problem: ${data.config_error}` : '');
        mcpRender();
        clearTimeout(mcpTimer);
        const pane = $('spane-mcp');
        if (pane && pane.classList.contains('active') && mcpServers.some((s) => s.state === 'starting')) {
          mcpTimer = setTimeout(mcpRefresh, 1500);
        }
      } catch (e) {
        $('mcpServerList').textContent = e.message;
      }
    }

    function mcpClearForm() {
      mcpEditing = null;
      ['mcpName', 'mcpCommand', 'mcpArgs', 'mcpEnv'].forEach((id) => { $(id).value = ''; });
      $('mcpName').disabled = false;
      $('mcpFormTitle').textContent = 'Add a server';
      $('mcpEnvHint').textContent = '';
    }

    function mcpEdit(name) {
      const s = mcpServers.find((x) => x.name === name);
      if (!s) return;
      mcpEditing = s;
      $('mcpName').value = s.name;
      $('mcpName').disabled = true;
      $('mcpCommand').value = s.command;
      $('mcpArgs').value = (s.args || []).join('\n');
      // the values stay on disk (they often are secrets): a name without a value keeps the saved value
      $('mcpEnv').value = (s.env_keys || []).map((k) => `${k}=`).join('\n');
      $('mcpEnvHint').textContent = s.env_keys && s.env_keys.length ? 'Saved values are not shown. A variable left without a value keeps its saved value.' : '';
      $('mcpFormTitle').textContent = `Edit ${s.name}`;
      $('mcpName').scrollIntoView({ block: 'center' });
    }

    async function mcpSave() {
      const env = {};
      for (const line of $('mcpEnv').value.split('\n')) {
        const t = line.trim();
        if (!t) continue;
        const i = t.indexOf('=');
        if (i < 1) { toast(`Environment line "${t}" needs the form NAME=value`); return; }
        const key = t.slice(0, i).trim();
        const value = t.slice(i + 1);
        env[key] = value === '' && mcpEditing && (mcpEditing.env_keys || []).includes(key) ? null : value;
      }
      const body = {
        name: $('mcpName').value.trim(),
        command: $('mcpCommand').value.trim(),
        args: $('mcpArgs').value.split('\n').map((a) => a.trim()).filter((a) => a !== ''),
        env,
      };
      try {
        await mcpRequest('POST', '/api/studio/mcp/servers', body);
        toast(`Saved ${body.name}`);
        mcpClearForm();
        mcpRefresh();
      } catch (e) {
        toast(e.message);
      }
    }

    $('btnMcpSave').addEventListener('click', mcpSave);
    $('btnMcpCancel').addEventListener('click', mcpClearForm);
    $('settingsMenuNav').addEventListener('click', (e) => {
      if (e.target.closest('[data-spane="mcp"]')) mcpRefresh();
    });
    $('mcpServerList').addEventListener('click', async (e) => {
      const btn = e.target.closest('[data-mcp-act]');
      const row = e.target.closest('[data-mcp]');
      if (!btn || !row) return;
      const name = row.dataset.mcp;
      try {
        if (btn.dataset.mcpAct === 'edit') mcpEdit(name);
        if (btn.dataset.mcpAct === 'restart') {
          btn.disabled = true;
          await mcpRequest('POST', '/api/studio/mcp/restart', { name });
          mcpRefresh();
        }
        if (btn.dataset.mcpAct === 'remove') {
          if (!confirm(`Remove the MCP server "${name}"?`)) return;
          await mcpRequest('DELETE', '/api/studio/mcp/servers', { name });
          mcpRefresh();
        }
      } catch (err) {
        toast(err.message);
        mcpRefresh();
      }
    });
