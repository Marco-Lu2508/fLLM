// ==========================================================================
    // Hooks: shell commands that run at fixed points of the agent loop.
    //  - pre_tool_call / post_tool_call / on_file_edit / on_session_start / on_session_end
    //  - user_prompt_submit / stop / subagent_stop / notification / pre_compact / file_changed (a hook may carry a prompt for the agent instead of a command)
    //  - a hook gets one JSON object on stdin (event, tool, arguments, workspace) and decides with its exit code:
    //    0 = pass (stdout is added to the tool result as notes), 2 = block the call (stderr is the reason),
    //    anything else = a failure that is reported but does not stop the agent.
    // The list lives in hooks.json next to studio.json (GET/POST /api/studio/hooks, POST /api/studio/hooks/test).
    // ==========================================================================

    (function hooksSettings() {
      const nav = document.getElementById('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      if (!nav || !content) return;

      const EVENTS = [
        ['pre_tool_call', 'Before a tool call', 'Can block the call. Good for: a guard that refuses writes to protected paths, a check that logs every command.'],
        ['post_tool_call', 'After a tool call', 'Sees the tool result in its arguments only; its output is added to the result. Good for: recording what ran.'],
        ['on_file_edit', 'After a file was changed', 'Runs only when an edit succeeded. Exit 2 reports a problem without undoing the edit. Good for: a formatter, a linter, a build check.'],
        ['on_session_start', 'When a workspace is opened', 'Runs when the agent switches folder. Output is shown once in the Code page. Good for: preparing an environment, printing a reminder.'],
        ['on_session_end', 'When the session ends', 'Runs when the agent session is closed. Good for: cleanup, a summary of the work.'],
        ['user_prompt_submit', 'When you send a message', 'Can block the message (exit 2) or add notes the agent sees with it. Good for: adding the ticket number, refusing prompts with secrets.'],
        ['stop', 'When the agent is about to finish', 'Exit 2 (or a block decision) sends the reason back to the agent as the next instruction, once. Good for: "run the tests before you finish".'],
        ['subagent_stop', 'When a sub-agent finishes', 'Runs when a sub-agent has reported back. Good for: logging, checking its result.'],
        ['notification', 'When the agent waits for you', 'Runs when a permission question or a choice is waiting. Good for: a sound, a desktop notice, a message to your phone.'],
        ['pre_compact', 'Before the conversation is compacted', 'Runs before older turns are summarised. Good for: saving the transcript.'],
        ['file_changed', 'When a watched file changes', 'Fired by fllm-rs agent --watch. The matcher is a list of globs (src/**/*.rs, *.md); "on" limits it to created, modified or deleted. With a prompt, the agent starts a run with that text; {{file}} is the path.'],
      ];

      const button = document.createElement('button');
      button.className = 'settings-nav-btn';
      button.dataset.spane = 'hooks';
      button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><path d="M9 18c-4 0-4-6 0-6s4 6 0 6z"/><path d="M15 18c4 0 4-6 0-6s-4 6 0 6z"/><path d="M12 2v4"/></svg><span>Hooks</span>';
      nav.appendChild(button);

      const pane = document.createElement('div');
      pane.className = 'settings-pane';
      pane.id = 'spane-hooks';
      pane.innerHTML =
        '<div class="app-card"><div class="app-card-title">Hooks</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;">Shell commands that run when the agent reaches a point in its loop. They are the deterministic half of the loop: a prompt asks the model to run the formatter, a hook makes sure it happened. A hook gets one JSON object on stdin (<span class="mono">{event, tool, arguments, workspace}</span>) and runs in the workspace folder with your rights.</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;">Exit code <span class="mono">0</span> passes and adds the output to the tool result as notes, <span class="mono">2</span> blocks the call (stderr becomes the reason the model sees), any other code is reported as a failed hook. A hook may also print <span class="mono">{"decision":"block","reason":"..."}</span> or <span class="mono">{"context":"..."}</span>.</div>' +
        '<div id="hooksBody"></div>' +
        '<div id="hooksPlugins"></div>' +
        '<div class="app-card-item" style="display:flex;gap:8px;align-items:center;margin-top:12px;">' +
        '  <button class="btn btn-sm" id="btnHooksSave">Save</button>' +
        '  <button class="btn btn-sm btn-ghost" id="btnHooksReload">Reload</button>' +
        '  <span class="app-item-desc mono" id="hooksFile" style="margin-left:auto;font-size:11px;"></span>' +
        '</div>' +
        '<div class="app-item-desc mono" id="hooksStatus" style="margin-top:8px;white-space:pre-wrap;"></div>' +
        '</div>';
      content.appendChild(pane);

      const body = pane.querySelector('#hooksBody');
      const status = pane.querySelector('#hooksStatus');
      const say = (text) => { status.textContent = text || ''; };

      // the list being edited: {event: [{matcher, command, timeout_seconds, enabled}]}
      let draft = {};

      const rowHtml = (event, hook, index) =>
        '<div class="app-card-item" data-event="' + event + '" data-index="' + index + '" style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">' +
        '  <input class="ui-input mono" data-field="matcher" value="' + escAttr(hook.matcher || '') + '" placeholder="' + (event === 'file_changed' ? 'globs, e.g. src/**/*.rs' : 'tools (empty = all)') + '" style="width:170px;" title="' + (event === 'file_changed' ? 'Globs for the path of the file, comma separated. Empty means every file.' : 'Tool names, comma separated; a trailing * matches a prefix. Empty or * means every tool.') + '">' +
        '  <input class="ui-input mono" data-field="command" value="' + escAttr(hook.command || '') + '" placeholder="command, e.g. cargo fmt --quiet" style="flex:1;min-width:220px;">' +
        '  <input class="ui-input" data-field="prompt" value="' + escAttr(hook.prompt || '') + '" placeholder="or a prompt for the agent" style="flex:1;min-width:200px;" title="Instead of a command: this text goes to the agent as context. {{file}} is replaced by the path.">' +
        (event === 'file_changed' ? '  <input class="ui-input mono" data-field="on" value="' + escAttr(hook.on || '') + '" placeholder="created, modified, deleted" style="width:170px;" title="Only for these kinds of change. Empty means all.">' : '') +
        '  <input class="ui-input mono" data-field="timeout_seconds" type="number" min="1" max="300" value="' + (hook.timeout_seconds || 30) + '" style="width:70px;" title="Timeout in seconds">' +
        '  <label class="ui-switch" title="Enabled"><input type="checkbox" data-field="enabled"' + (hook.enabled === false ? '' : ' checked') + '><span class="ui-switch-track"></span></label>' +
        '  <button class="btn btn-sm btn-ghost" data-action="test">Test</button>' +
        '  <button class="btn btn-sm btn-ghost" data-action="remove">Remove</button>' +
        '</div>';

      function escAttr(text) {
        return String(text).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
      }

      function render() {
        body.innerHTML = EVENTS.map(([event, title, hint]) => {
          const hooks = draft[event] || [];
          return '<div style="margin-top:14px;">' +
            '<div class="app-item-title">' + title + ' <span class="mono" style="color:var(--muted-foreground);font-size:11px;">' + event + '</span></div>' +
            '<div class="app-item-desc" style="margin-bottom:6px;">' + hint + '</div>' +
            hooks.map((hook, i) => rowHtml(event, hook, i)).join('') +
            '<button class="btn btn-sm btn-ghost" data-action="add" data-event="' + event + '">+ Add hook</button>' +
            '</div>';
        }).join('');
      }

      function collect() {
        const next = {};
        body.querySelectorAll('[data-event][data-index]').forEach((row) => {
          const event = row.dataset.event;
          const index = Number(row.dataset.index);
          const value = (field) => row.querySelector('[data-field="' + field + '"]');
          const entry = {
            matcher: value('matcher').value.trim(),
            command: value('command').value.trim(),
            prompt: value('prompt').value.trim(),
            on: value('on') ? value('on').value.trim() : '',
            timeout_seconds: Number(value('timeout_seconds').value) || 30,
            enabled: value('enabled').checked,
          };
          if (!entry.command && !entry.prompt) return;
          if (!next[event]) next[event] = [];
          next[event][index] = entry;
        });
        Object.keys(next).forEach((event) => { next[event] = next[event].filter(Boolean); });
        draft = next;
      }

      async function load() {
        try {
          const res = await apiFetch('/api/studio/hooks');
          const data = await res.json();
          draft = data.hooks || {};
          pane.querySelector('#hooksFile').textContent = data.file || '';
          render();
          renderPluginHooks(data.from_plugins || []);
          say('');
        } catch (e) {
          say('Could not load the hooks: ' + ((e && e.message) || e));
        }
      }

      /** Hooks that an installed plugin brings: shown, not editable here — they belong to the plugin. */
      function renderPluginHooks(items) {
        const box = pane.querySelector('#hooksPlugins');
        if (!box) return;
        if (!items.length) { box.innerHTML = ''; return; }
        box.innerHTML = '<div style="margin-top:16px;"><div class="app-item-title">From plugins</div>' +
          '<div class="app-item-desc" style="margin-bottom:6px;">These run like your own hooks. Remove or change them in the plugin (Settings &gt; Plugins), not here.</div>' +
          items.map((h) =>
            '<div class="app-card-item" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;opacity:.85;">' +
            '  <span class="mono" style="font-size:11px;">' + hooksEsc(h.event) + '</span>' +
            '  <span class="mono" style="font-size:11px;color:var(--muted-foreground);">' + hooksEsc(h.matcher || '*') + '</span>' +
            '  <span class="mono" style="flex:1;min-width:200px;">' + hooksEsc(h.command) + '</span>' +
            '  <span class="app-item-desc" style="font-size:11px;">' + hooksEsc(h.plugin) + '</span>' +
            '  <span class="app-item-desc" style="font-size:11px;">' + (h.enabled === false ? 'off' : h.timeout_seconds + ' s') + '</span>' +
            '</div>').join('') + '</div>';
      }

      function hooksEsc(text) { return String(text == null ? '' : text).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }

      async function save() {
        collect();
        try {
          const res = await apiFetch('/api/studio/hooks', { method: 'POST', body: { hooks: draft } });
          const data = await res.json();
          if (!res.ok) { say(data.error && data.error.message ? data.error.message : 'Could not save.'); return; }
          draft = data.hooks || {};
          render();
          say('Saved. The next tool call runs them.');
        } catch (e) {
          say('Could not save: ' + ((e && e.message) || e));
        }
      }

      function sampleFor(event) {
        if (event === 'on_file_edit') return { tool: 'edit_file', arguments: { path: 'README.md' } };
        if (event === 'file_changed') return { tool: 'README.md', arguments: { path: 'README.md', kind: 'modified' } };
        if (event === 'user_prompt_submit') return { tool: '', arguments: { prompt: 'hello' } };
        if (event === 'pre_tool_call' || event === 'post_tool_call') return { tool: 'bash', arguments: { command: 'echo hello' } };
        return { tool: '', arguments: {} };
      }

      async function test(event, row) {
        collect();
        const index = Number(row.dataset.index);
        const hook = (draft[event] || [])[index];
        if (!hook) return;
        const sample = sampleFor(event);
        say('Running ' + (hook.command || hook.prompt) + ' ...');
        try {
          // save first so the server runs exactly what is on screen
          await apiFetch('/api/studio/hooks', { method: 'POST', body: { hooks: draft } });
          const res = await apiFetch('/api/studio/hooks/test', {
            method: 'POST',
            body: { event, tool: sample.tool, arguments: sample.arguments },
          });
          const data = await res.json();
          const ran = (data.ran || []).map((r) =>
            '  ' + (r.command || ('prompt: ' + (r.prompt || ''))) + ' -> exit ' + r.exit_code + ' (' + r.milliseconds + ' ms)' +
            (r.stdout ? '\n    stdout: ' + r.stdout : '') +
            (r.stderr ? '\n    stderr: ' + r.stderr : '')).join('\n');
          say((data.blocked ? 'BLOCKED: ' + (data.reason || '') + '\n' : 'passed\n') + (ran || '  (no hook matched)'));
        } catch (e) {
          say('The test failed: ' + ((e && e.message) || e));
        }
      }

      body.addEventListener('click', (e) => {
        const action = e.target.dataset && e.target.dataset.action;
        if (!action) return;
        if (action === 'add') {
          collect();
          const event = e.target.dataset.event;
          if (!draft[event]) draft[event] = [];
          draft[event].push({ matcher: '', command: '', prompt: '', on: '', timeout_seconds: 30, enabled: true });
          render();
          return;
        }
        const row = e.target.closest('[data-index]');
        if (!row) return;
        const event = row.dataset.event;
        if (action === 'remove') {
          collect();
          draft[event].splice(Number(row.dataset.index), 1);
          if (!draft[event].length) delete draft[event];
          render();
          say('Removed. Press Save to write it.');
        } else if (action === 'test') {
          test(event, row);
        }
      });

      pane.querySelector('#btnHooksSave').addEventListener('click', save);
      pane.querySelector('#btnHooksReload').addEventListener('click', load);
      load();
    })();
