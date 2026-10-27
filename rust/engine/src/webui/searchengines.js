// ==========================================================================
    // Settings > Search, second card: the engines of the web search.
    //  - DuckDuckGo and Wikipedia need no key and no server: they work out of the box.
    //  - Extra engines: a SearXNG instance (yours or one that allows format=json) or Brave with an API key.
    //  - "Test" runs one real query and shows per engine what happened, so a blocked engine is visible as blocked.
    // ==========================================================================

    (function searchEngines() {
      const pane = document.getElementById('spane-search');
      if (!pane) return;

      const card = document.createElement('div');
      card.className = 'app-card';
      card.style.marginTop = '12px';
      card.innerHTML =
        '<div class="app-card-title">Web search engines</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;">DuckDuckGo and Wikipedia need no key and no server, so search works the moment the app starts. Add your own engine if you run SearXNG or have a Brave key: every engine is asked, the results are merged, and the tool reports which engine failed.</div>' +
        '<div id="seEngineList"></div>' +
        '<div style="display:flex;gap:6px;align-items:center;margin-top:8px;flex-wrap:wrap;">' +
        '  <select class="ui-input" id="seKind" style="width:130px;">' +
        '    <option value="ddg">DuckDuckGo</option><option value="wikipedia">Wikipedia</option>' +
        '    <option value="searxng">SearXNG</option><option value="brave">Brave</option>' +
        '  </select>' +
        '  <input class="ui-input mono" id="seName" placeholder="name (optional)" style="width:150px;" autocomplete="off">' +
        '  <input class="ui-input mono" id="seUrl" placeholder="http://127.0.0.1:8888" style="flex:1;min-width:180px;" autocomplete="off">' +
        '  <input class="ui-input mono" id="seKey" placeholder="API key (Brave)" style="width:170px;" autocomplete="off">' +
        '  <button class="btn btn-sm btn-ghost" id="seAdd">Add</button>' +
        '</div>' +
        '<div style="display:flex;gap:6px;align-items:center;margin-top:10px;flex-wrap:wrap;">' +
        '  <button class="btn btn-sm btn-primary" id="seSave">Save engines</button>' +
        '  <button class="btn btn-sm btn-secondary" id="seTest">Test a search</button>' +
        '</div>' +
        '<div class="app-item-desc mono" id="seStatus" style="margin-top:8px;font-size:11px;white-space:pre-wrap;"></div>';
      pane.appendChild(card);

      let engines = [];
      const say = (text) => { card.querySelector('#seStatus').textContent = text || ''; };

      function render() {
        const list = card.querySelector('#seEngineList');
        if (!engines.length) { list.innerHTML = '<div class="app-item-desc">No engine yet.</div>'; return; }
        list.innerHTML = engines.map((e, i) =>
          '<div style="display:flex;gap:8px;align-items:center;border-bottom:1px solid var(--border);padding:4px 0;" data-engine="' + i + '">' +
          '  <span class="app-item-title" style="font-size:12px;">' + escapeHtml(e.name || e.kind) + '</span>' +
          '  <span class="app-item-desc mono" style="font-size:11px;">' + escapeHtml(e.kind) + (e.url ? ' · ' + escapeHtml(e.url) : '') + (e.key ? ' · key set' : '') + '</span>' +
          '  <span style="flex:1"></span>' +
          '  <button class="btn btn-xs btn-ghost" data-engine-remove="' + i + '">Remove</button>' +
          '</div>').join('');
      }

      async function load() {
        try {
          const res = await apiFetch('/api/studio/search/engines');
          const data = await res.json();
          engines = (data.engines || []).map((e) => ({ kind: e.kind, name: e.name, url: e.url, key: e.key }));
          render();
          say(`${engines.length} engine(s) · saved in ${data.file}`);
        } catch (e) {
          say('Could not read the engines: ' + ((e && e.message) || e));
        }
      }

      card.querySelector('#seAdd').addEventListener('click', () => {
        const kind = card.querySelector('#seKind').value;
        const name = card.querySelector('#seName').value.trim();
        const url = card.querySelector('#seUrl').value.trim();
        const key = card.querySelector('#seKey').value.trim();
        if (kind === 'searxng' && !url) { say('A SearXNG engine needs its address.'); return; }
        if (kind === 'brave' && !key) { say('A Brave engine needs an API key.'); return; }
        engines.push({ kind, name, url, key });
        card.querySelector('#seName').value = '';
        card.querySelector('#seUrl').value = '';
        card.querySelector('#seKey').value = '';
        render();
      });

      card.querySelector('#seEngineList').addEventListener('click', (e) => {
        const btn = e.target.closest('[data-engine-remove]');
        if (!btn) return;
        engines.splice(Number(btn.dataset.engineRemove), 1);
        render();
      });

      card.querySelector('#seSave').addEventListener('click', async () => {
        try {
          const res = await apiFetch('/api/studio/search/engines', { method: 'POST', body: { engines } });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { say((data.error && data.error.message) || 'Could not save the engines.'); return; }
          engines = (data.engines || []).map((e) => ({ kind: e.kind, name: e.name, url: e.url, key: e.key }));
          render();
          say('Saved. The next search uses these engines.');
        } catch (e) {
          say('Could not save the engines: ' + ((e && e.message) || e));
        }
      });

      card.querySelector('#seTest').addEventListener('click', async () => {
        say('Searching "rust axum middleware" ...');
        try {
          const res = await apiFetch('/api/studio/search', { method: 'POST', body: { query: 'rust axum middleware', limit: 3 } });
          const data = await res.json().catch(() => ({}));
          const lines = (data.engines || []).map((r) => (r.ok ? `${r.engine}: ${r.found} Treffer` : `${r.engine}: ${r.error}`));
          const hits = (data.results || []).slice(0, 3).map((h, i) => `${i + 1}. ${h.title}`);
          say([...lines, '', ...hits].join('\n') || (data.error && data.error.message) || 'No answer.');
        } catch (e) {
          say('The search failed: ' + ((e && e.message) || e));
        }
      });

      load();
    })();
