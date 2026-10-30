// ==========================================================================
    // Settings > Connect other apps, second card: "Connect a coding tool".
    // The text comes from the server (GET /api/studio/connect), which builds it with the same code as
    // `fllm-rs connect TOOL`, so the page and the command always agree. Nothing is written from here: the card
    // shows the environment lines and the config files with a copy button each.
    // ==========================================================================

    (function connectCodingTool() {
      const pane = document.getElementById('spane-claude-code');
      if (!pane) return;

      const card = document.createElement('div');
      card.className = 'app-card';
      card.style.marginTop = '12px';
      card.innerHTML =
        '<div class="app-card-title">Connect a coding tool</div>' +
        '<p class="app-item-desc" style="margin-bottom:10px;">Pick your tool: you get the exact settings for this server and the model that is loaded now. ' +
        'The same text comes from <span class="mono">fllm-rs connect TOOL</span> in a terminal, which can also write the files for you (<span class="mono">--write</span>).</p>' +
        '<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:10px;">' +
        '  <select class="ui-select" id="ctTool" style="width:auto;"></select>' +
        '  <select class="ui-select" id="ctShell" style="width:auto;"><option value="powershell">PowerShell</option><option value="sh">bash / zsh</option><option value="cmd">cmd</option></select>' +
        '</div>' +
        '<div id="ctOut"></div>' +
        '<div class="app-item-desc mono" id="ctStatus" style="margin-top:8px;font-size:11px;"></div>';
      pane.appendChild(card);

      const out = card.querySelector('#ctOut');
      const status = card.querySelector('#ctStatus');
      const toolSelect = card.querySelector('#ctTool');
      const shellSelect = card.querySelector('#ctShell');
      shellSelect.value = /Win/i.test(navigator.platform || '') ? 'powershell' : 'sh';
      let loaded = null;

      function block(title, text) {
        const wrap = document.createElement('div');
        wrap.className = 'code-card';
        wrap.style.margin = '8px 0 0';
        const head = document.createElement('div');
        head.className = 'code-card-header';
        const label = document.createElement('span');
        label.textContent = title;
        const copy = document.createElement('button');
        copy.className = 'btn btn-xs btn-ghost';
        copy.textContent = 'Copy';
        copy.addEventListener('click', async () => {
          try { await navigator.clipboard.writeText(text); copy.textContent = 'Copied'; } catch (e) { copy.textContent = 'Copy failed'; }
          setTimeout(() => { copy.textContent = 'Copy'; }, 1500);
        });
        head.appendChild(label);
        head.appendChild(copy);
        const pre = document.createElement('pre');
        pre.textContent = text;
        wrap.appendChild(head);
        wrap.appendChild(pre);
        return wrap;
      }

      function render() {
        out.textContent = '';
        if (!loaded) return;
        (loaded.notes || []).forEach((note) => {
          const p = document.createElement('div');
          p.className = 'app-item-desc';
          p.style.margin = '4px 0';
          p.textContent = note;
          out.appendChild(p);
        });
        const env = (loaded.env && loaded.env[shellSelect.value]) || [];
        if (env.length) out.appendChild(block('Environment', env.join('\n')));
        (loaded.files || []).forEach((file) => {
          out.appendChild(block(file.path + (file.problem ? ' (cannot be created here)' : ''), file.content || file.problem));
        });
        if (loaded.snippet) out.appendChild(block(loaded.snippet.title, loaded.snippet.text));
        if ((loaded.start || []).length) out.appendChild(block('Start', loaded.start.join('\n')));
        const hasKey = typeof state !== 'undefined' && state && state.apiKeyRequired;
        status.textContent = 'The key "fllm" is a placeholder' + (hasKey ? '; this server wants a real key, put yours in its place.' : ': this server accepts any value.');
      }

      async function load() {
        try {
          const res = await apiFetch('/api/studio/connect?tool=' + encodeURIComponent(toolSelect.value || 'claude-code'));
          loaded = await res.json();
          if (!toolSelect.options.length && Array.isArray(loaded.tools)) {
            loaded.tools.forEach((t) => {
              const o = document.createElement('option');
              o.value = t.id;
              o.textContent = t.id + ' - ' + t.what;
              toolSelect.appendChild(o);
            });
            toolSelect.value = 'claude-code';
          }
          render();
        } catch (e) {
          status.textContent = 'Could not read the settings: ' + ((e && e.message) || e);
        }
      }

      toolSelect.addEventListener('change', load);
      shellSelect.addEventListener('change', render);
      // the card is built when the pane is first shown, so the page start does not wait for it
      const nav = document.querySelector('#settingsMenuNav [data-spane="claude-code"]');
      let started = false;
      const start = () => { if (!started) { started = true; load(); } };
      if (nav) nav.addEventListener('click', start);
      if (pane.classList.contains('active')) start();
    })();
