// ==========================================================================
    // Settings > Engine: the options of the engine that are not tied to one model — the caches, the order in which work is dropped,
    // telemetry and the admin key — and the live controls (stop taking work, empty the cache).
    //  - the options are saved on this computer and take effect when the server starts again (the page offers to restart it)
    //  - the numbers of the caches and the drain switch are live and come from the running server
    // ==========================================================================

    (function engineSettings() {
      const nav = document.getElementById('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      if (!nav || !content) return;

      const button = document.createElement('button');
      button.className = 'settings-nav-btn';
      button.dataset.spane = 'engine';
      button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/></svg><span>Engine</span>';
      nav.appendChild(button);

      const pane = document.createElement('div');
      pane.className = 'settings-pane';
      pane.id = 'spane-engine';
      const row = (id, title, desc, control) =>
        `<div class="app-card-item"><div class="app-item-info"><div class="app-item-title">${title}</div><div class="app-item-desc">${desc}</div></div>` +
        `<div class="app-item-actions">${control}</div></div>`;
      const select = (id, options) => `<select id="${id}" class="ui-select mono">${options.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select>`;
      const number = (id, min, width) => `<input type="number" id="${id}" class="ui-input mono" min="${min}" style="width:${width || 110}px;">`;
      const toggle = (id) => `<label class="ui-switch"><input type="checkbox" id="${id}"><span class="ui-switch-track"></span></label>`;
      pane.innerHTML =
        '<div class="app-card"><div class="app-card-title">Answers that were given before</div>' +
        row('', 'Remember identical answers', 'When the same request with temperature 0 comes again, the saved answer is returned at once instead of being computed again. Requests with a temperature above 0, with images, tools or streaming never use it.', select('engOutputCache', [['exact', 'On'], ['off', 'Off']])) +
        row('', 'Memory for saved answers', 'The cache never grows beyond this size; the oldest answers go first.', number('engCacheMb', 1) + ' <span class="app-item-desc">MB</span>') +
        row('', 'Keep answers for', 'After this time a saved answer is dropped.', number('engCacheTtl', 1) + ' <span class="app-item-desc">seconds</span>') +
        row('', 'Who shares saved answers', '“Everyone” shares one cache. “Per key and user” keeps the answers of each API key (and the user field of the request) apart; a request that has neither is not saved. Use it when several people use the server.', select('engCacheScope', [['global', 'Everyone'], ['tenant', 'Per key and user'], ['off', 'Nobody (no saving)']])) +
        '<div class="app-item-desc mono" id="engCacheStats" style="padding:8px 4px 0;font-size:11px;"></div>' +
        '<div style="display:flex;gap:8px;padding:8px 4px 0;"><button class="btn btn-ghost" id="engCacheClear">Empty the cache now</button></div>' +
        '</div>' +
        '<div class="app-card" style="margin-top:12px;"><div class="app-card-title">Speed and memory</div>' +
        row('', 'Share the start of prompts', 'Requests that begin with the same text (a system prompt, a long document) reuse the work already done for it. Turn it off if people on this server must not share anything.', toggle('engPrefixCache')) +
        row('', 'Separate this installation', 'A text that keeps the shared prompt blocks of this server apart from those of another server. Leave it empty unless you run several.', '<input type="text" id="engPrefixSalt" class="ui-input mono" style="width:180px;" autocomplete="off">') +
        row('', 'When memory runs out', '“Recompute” drops the work of a waiting request and does it again later (saves memory). “Move to RAM” keeps it in the main memory (faster to continue, needs RAM).', select('engPreemption', [['recompute', 'Recompute'], ['swap', 'Move to RAM']])) +
        '</div>' +
        '<div class="app-card" style="margin-top:12px;"><div class="app-card-title">Telemetry</div>' +
        row('', 'Send traces to', 'The address of an OpenTelemetry collector (for example http://127.0.0.1:4318). Nothing is sent while it is empty. Only timings and sizes are sent, never the text of a conversation.', '<input type="text" id="engOtelEndpoint" class="ui-input mono" style="width:260px;" placeholder="off" autocomplete="off">') +
        row('', 'Name of this server', 'How the server is called in the traces.', '<input type="text" id="engOtelService" class="ui-input mono" style="width:180px;" autocomplete="off">') +
        '<div class="app-item-desc mono" id="engOtelStats" style="padding:8px 4px 0;font-size:11px;"></div>' +
        '</div>' +
        '<div class="app-card" style="margin-top:12px;"><div class="app-card-title">Administration</div>' +
        row('', 'Admin key', 'Without it only this computer may stop the server taking work, empty the cache or read the settings. With it, another computer can do that too by sending it as <span class="mono">x-admin-key</span>. An API key for chatting is never enough. The key is never shown again after you save it.', '<input type="password" id="engAdminKey" class="ui-input mono" style="width:200px;" placeholder="not set" autocomplete="new-password">') +
        row('', 'Stop taking new requests', 'The server finishes what it is doing and answers new requests with “busy”. Used before a restart or an update, so nobody loses an answer. Takes effect at once.', toggle('engDrain')) +
        '</div>' +
        '<div style="display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap;">' +
        '<button class="btn btn-primary" id="engSave">Save</button>' +
        '<button class="btn btn-ghost" id="engRestart" style="display:none;">Restart the server now</button>' +
        '<button class="btn btn-ghost" id="engReset">Back to the defaults</button>' +
        '<span class="app-item-desc" id="engStatus"></span></div>';
      content.appendChild(pane);

      const el = (id) => pane.querySelector('#' + id);
      const say = (text) => { el('engStatus').textContent = text || ''; };
      const FIELDS = {
        output_cache: ['engOutputCache', 'value'], output_cache_mb: ['engCacheMb', 'value'], output_cache_ttl: ['engCacheTtl', 'value'],
        cache_scope: ['engCacheScope', 'value'], prefix_cache: ['engPrefixCache', 'checked'], prefix_salt: ['engPrefixSalt', 'value'],
        preemption: ['engPreemption', 'value'], otel_endpoint: ['engOtelEndpoint', 'value'], otel_service: ['engOtelService', 'value'],
      };
      let defaults = {};
      let saved = {};

      function fill(values) {
        for (const [key, [id, prop]] of Object.entries(FIELDS)) el(id)[prop] = values[key] ?? defaults[key] ?? (prop === 'checked' ? false : '');
        // the custom drop-down lists follow the value on their own
      }

      function collect() {
        const body = {};
        for (const [key, [id, prop]] of Object.entries(FIELDS)) body[key] = el(id)[prop];
        const key = el('engAdminKey').value;
        if (key) body.admin_key = key;
        return body;
      }

      async function live() {
        try {
          const res = await apiFetch('/admin/config');
          if (!res.ok) throw new Error(String(res.status));
          const data = await res.json();
          const cache = data.output_cache || {};
          el('engCacheStats').textContent = `now: ${cache.entries ?? 0} saved answers, ${formatBytes(cache.bytes || 0)} · answered from the cache ${cache.hits ?? 0} times, computed ${cache.misses ?? 0} times, ${cache.saved_prompt_tokens ?? 0} prompt tokens saved` +
            (data.coalescing ? ` · identical requests shared: ${data.coalescing.followers ?? 0}` : '');
          const t = data.telemetry || {};
          el('engOtelStats').textContent = t.enabled ? `sending traces · ${t.sent ?? 0} batches sent, ${t.dropped ?? 0} dropped` : 'telemetry is off';
          el('engDrain').checked = Boolean(data.draining);
          el('engCacheClear').disabled = false;
          el('engDrain').disabled = false;
        } catch {
          el('engCacheStats').textContent = 'Load a model to see the live numbers of the cache.';
          el('engOtelStats').textContent = '';
          el('engCacheClear').disabled = true;
          el('engDrain').disabled = true;
        }
      }

      async function load() {
        try {
          const res = await apiFetch('/api/studio/engine');
          const data = await res.json();
          defaults = data.defaults || {};
          saved = data.values || {};
          fill(saved);
          el('engAdminKey').value = '';
          el('engAdminKey').placeholder = data.admin_key_set ? 'set (type a new one to change it)' : 'not set';
          const env = data.set_by_environment || [];
          say(env.length ? `Set by the environment of the server and not changeable here: ${env.join(', ')}.` : '');
        } catch (e) {
          say('Could not read the settings: ' + ((e && e.message) || e));
        }
        live();
      }

      async function save(body, message) {
        try {
          const res = await apiFetch('/api/studio/engine', { method: 'POST', body });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { say((data.error && data.error.message) || 'Could not save it.'); return false; }
          saved = data.values || {};
          el('engAdminKey').value = '';
          el('engAdminKey').placeholder = data.admin_key_set ? 'set (type a new one to change it)' : 'not set';
          el('engRestart').style.display = '';
          say(message || 'Saved. The changes take effect when the server restarts.');
          return true;
        } catch (e) {
          say('Could not save it: ' + ((e && e.message) || e));
          return false;
        }
      }

      el('engSave').addEventListener('click', () => save(collect()));
      el('engReset').addEventListener('click', async () => {
        if (!await confirmDialog('Set every option on this page back to its default? The admin key is kept.')) return;
        const body = {};
        for (const key of Object.keys(FIELDS)) body[key] = typeof defaults[key] === 'boolean' ? defaults[key] : (typeof defaults[key] === 'number' ? defaults[key] : String(defaults[key] ?? ''));
        if (await save(body, 'Back to the defaults. They take effect when the server restarts.')) fill(defaults);
      });
      el('engRestart').addEventListener('click', async () => {
        $('switchTitle').textContent = 'Restarting';
        $('switchText').textContent = 'The server starts again with the new options ...';
        $('switchOverlay').classList.add('open');
        const res = await apiFetch('/api/studio/restart', { method: 'POST' });
        if (!res.ok) { $('switchOverlay').classList.remove('open'); say('The server could not be restarted.'); return; }
        await waitForRestart(() => true, 'Starting the server ...');
      });
      el('engCacheClear').addEventListener('click', async () => {
        const res = await apiFetch('/admin/cache', { method: 'DELETE' });
        const data = await res.json().catch(() => ({}));
        toast(res.ok ? `${data.cleared ?? 0} saved answers removed.` : 'The cache could not be emptied.');
        live();
      });
      el('engDrain').addEventListener('change', async (e) => {
        const res = await apiFetch('/admin/drain', { method: e.target.checked ? 'POST' : 'DELETE' });
        if (!res.ok) { e.target.checked = !e.target.checked; toast('The server did not accept that.'); }
        else toast(e.target.checked ? 'The server takes no new requests.' : 'The server takes requests again.');
      });

      button.addEventListener('click', load);
    })();
