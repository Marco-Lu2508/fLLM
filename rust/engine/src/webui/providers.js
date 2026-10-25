    // ==========================================================================
    // Model providers: models of other servers and cloud services next to the local one. The list lives in
    // providers.json on this computer; keys never reach the page (the server adds them), chats go through
    // /api/studio/providers/chat.
    // ==========================================================================

    // @pure-begin
    const REMOTE_PREFIX = 'remote:';

    /** The id under which a model of a provider is selected: `remote:<provider id>/<model>`. */
    function remoteModelId(provider, model) { return `${REMOTE_PREFIX}${provider}/${model}`; }

    /** `{provider, model}` for a selection of a provider's model, or null for a local model. Provider ids hold no slash; model names may. */
    function parseRemoteModelId(id) {
      if (typeof id !== 'string' || !id.startsWith(REMOTE_PREFIX)) return null;
      const rest = id.slice(REMOTE_PREFIX.length);
      const slash = rest.indexOf('/');
      if (slash < 1 || slash === rest.length - 1) return null;
      return { provider: rest.slice(0, slash), model: rest.slice(slash + 1) };
    }

    /** The entries of the model picker: the local models first, then "Provider · model" for every model of a provider. */
    function mergeModelEntries(localNames, remoteModels, selected) {
      const out = [];
      const seen = new Set();
      for (const name of localNames || []) {
        if (!name || seen.has(name)) continue;
        seen.add(name);
        out.push({ id: name, label: name, remote: false, active: name === selected });
      }
      for (const m of remoteModels || []) {
        if (!m || !m.provider || !m.id) continue;
        const id = remoteModelId(m.provider, m.id);
        if (seen.has(id)) continue;
        seen.add(id);
        out.push({ id, label: `${m.provider_name || m.provider} · ${m.id}`, remote: true, provider: m.provider, model: m.id, active: id === selected });
      }
      return out;
    }

    /** Keeps a selection that still exists; a model of a provider that was removed or switched off falls back to `fallback`. */
    function resolveSelection(selected, entries, fallback) {
      if (!parseRemoteModelId(selected)) return selected;
      return entries.some((e) => e.id === selected) ? selected : fallback;
    }

    /** What the label of the picker shows for a selection. */
    function modelLabel(selected, remoteModels) {
      const r = parseRemoteModelId(selected);
      if (!r) return selected;
      const known = (remoteModels || []).find((m) => m.provider === r.provider && m.id === r.model);
      return `${(known && (known.provider_name || known.provider)) || r.provider} · ${r.model}`;
    }

    // settings of the local engine that other servers do not understand
    const LOCAL_ONLY_FIELDS = ['top_k', 'min_p', 'repetition_penalty', 'speculative', 'dry_multiplier', 'dry_base', 'dry_allowed_length',
      'dry_penalty_last_n', 'mirostat', 'mirostat_tau', 'mirostat_eta', 'grammar', 'grammar_type', 'regex', 'tool_call_grammar',
      'tfs_z', 'typical_p', 'ignore_eos', 'cache_prompt', 'logit_bias_tokens'];

    /** The request for a provider's model: the same body without the settings of the local engine and with the model's own name. */
    function cleanRemoteRequest(body, model) {
      const out = Object.assign({}, body, { model });
      for (const k of LOCAL_ONLY_FIELDS) delete out[k];
      return out;
    }

    /**
     * Where a chat for `modelId` goes: `{url, remote, provider, model, transform}`. `transform(body)` turns the
     * chat-completions body the page built into the body of that url.
     */
    function chatEndpoint(modelId) {
      const r = parseRemoteModelId(modelId);
      if (!r) return { url: '/v1/chat/completions', remote: false, transform: (body) => body };
      return {
        url: '/api/studio/providers/chat',
        remote: true,
        provider: r.provider,
        model: r.model,
        transform: (body) => ({ provider: r.provider, request: cleanRemoteRequest(body, r.model) }),
      };
    }
    // @pure-end

    const provState = { providers: [], presets: [], models: [], errors: [], editing: null, loaded: false };

    function isRemoteSelected() { return Boolean(parseRemoteModelId(state.selectedModel)); }

    /**
     * For any code that sends chats: `{url, headers, remote, provider, model, transform}` for a model id. Use
     * `fetch(ep.url, {method: 'POST', headers: ep.headers, body: JSON.stringify(ep.transform(body))})`; the answer is an
     * OpenAI chat-completions body or stream either way.
     */
    window.fllmChatEndpoint = (modelId) => {
      const ep = chatEndpoint(modelId === undefined ? state.selectedModel : modelId);
      const headers = { 'Content-Type': 'application/json' };
      if (state.apiKey) headers.Authorization = `Bearer ${state.apiKey}`;
      return Object.assign({ headers }, ep);
    };
    window.fllmRemoteSelected = isRemoteSelected;

    /** The meter next to the chat when a provider's model answers: tokens used, no limit known. Returns true when it handled the meter. */
    window.fllmRemoteMeter = (used) => {
      if (!isRemoteSelected()) return false;
      $('ctxFill').style.width = '0%';
      $('ctxFill').parentElement.parentElement.classList.remove('is-high');
      $('ctxText').textContent = `${Number(used || 0).toLocaleString('en-US')} tokens`;
      return true;
    };

    /** Called after the picker of the header was drawn: adds the models of the providers and fixes the labels. */
    window.fllmPickerHook = () => {
      const local = [state.status?.model?.name, ...(state.status?.model?.aliases || [])].filter(Boolean);
      const entries = mergeModelEntries(local, provState.models, state.selectedModel);
      if (provState.loaded) {
        const kept = resolveSelection(state.selectedModel, entries, local[0] || 'tiny');
        if (kept !== state.selectedModel) { state.selectedModel = kept; entries.forEach((e) => { e.active = e.id === kept; }); }
      }
      const remote = isRemoteSelected();
      $('activeModelLabel').textContent = modelLabel(state.selectedModel, provState.models);
      if (remote) $('activeQuantChip').textContent = '';
      const box = $('modelDropdownItems');
      const old = $('provPickerItems');
      if (old) old.remove();
      const rows = entries.filter((e) => e.remote);
      if (rows.length) {
        box.insertAdjacentHTML('beforeend', `<div id="provPickerItems" style="max-height:260px;overflow:auto;">
          <div class="app-item-desc" style="padding:6px 10px 2px;">Other providers</div>${rows.map((e) => `
          <button class="nav-item ${e.active ? 'active' : ''}" data-pick-model="${escapeHtml(e.id)}">
            <svg viewBox="0 0 24 24" class="icon icon-sm" style="color:var(--muted-foreground);"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>
            <span class="mono">${escapeHtml(e.label)}</span>
          </button>`).join('')}</div>`);
      }
      if (remote) {
        $('welcomeHeading').textContent = modelLabel(state.selectedModel, provState.models);
        $('welcomeSub').textContent = 'The answers come from another server; what you type is sent there.';
      }
    };

    async function provRequest(method, path, body) {
      const res = await apiFetch(path, body === undefined ? { method } : { method, body });
      let data = {};
      try { data = await res.json(); } catch (e) { /* no body */ }
      if (!res.ok) throw new Error((data.error && data.error.message) || `Request failed (${res.status})`);
      return data;
    }

    /** Loads the models of the enabled providers (from the cache, or freshly listed) for the picker. */
    async function provLoadModels(refresh) {
      try {
        const data = await provRequest('GET', `/api/studio/providers/models${refresh ? '?refresh=1' : ''}`);
        provState.models = data.models || [];
        provState.errors = data.errors || [];
        provState.loaded = true;
        if (typeof window.fllmPickerHook === 'function' && state.status) window.fllmPickerHook();
      } catch (e) { /* the picker just shows the local models */ }
    }

    function provShow(text, ok) {
      const el = $('provResult');
      el.textContent = text;
      el.style.color = ok ? 'var(--primary)' : 'var(--destructive,#d33)';
    }

    function provRender() {
      const box = $('provList');
      if (!provState.providers.length) {
        box.innerHTML = '<div class="app-item-desc">No providers yet. Pick a template below.</div>';
        return;
      }
      box.innerHTML = provState.providers.map((p) => `
        <div class="app-card-item column" data-prov="${escapeHtml(p.id)}" style="border-bottom:1px solid var(--border);padding-bottom:10px;">
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <span class="app-item-title">${escapeHtml(p.name)}</span>
            <span class="app-item-desc">${escapeHtml(p.kind)}${p.local ? ', this computer or network' : ''}${p.has_key ? ', key saved' : ', no key'}${p.models.length ? `, ${p.models.length} model${p.models.length === 1 ? '' : 's'}` : ''}</span>
            <span style="flex:1"></span>
            <label class="app-item-desc" style="display:flex;gap:4px;align-items:center;"><input type="checkbox" data-prov-act="toggle" ${p.enabled ? 'checked' : ''}> On</label>
            <button class="btn btn-xs btn-ghost" data-prov-act="test">Test and refresh models</button>
            <button class="btn btn-xs btn-ghost" data-prov-act="edit">Edit</button>
            <button class="btn btn-xs btn-ghost" data-prov-act="remove">Remove</button>
          </div>
          <div class="app-item-desc mono" style="word-break:break-all;">${escapeHtml(p.base_url)}</div>
        </div>`).join('');
    }

    function provRenderPresets() {
      $('provPresets').innerHTML = provState.presets.map((t) =>
        `<button class="btn btn-xs btn-ghost" data-prov-preset="${escapeHtml(t.id)}">${escapeHtml(t.name)}</button>`).join('');
    }

    async function provRefresh() {
      try {
        const data = await provRequest('GET', '/api/studio/providers');
        provState.providers = data.providers || [];
        provState.presets = data.presets || [];
        $('provFileInfo').textContent = `Saved in ${data.config_file}`;
        provRender();
        provRenderPresets();
      } catch (e) {
        $('provList').textContent = e.message;
      }
    }

    function provClearForm() {
      provState.editing = null;
      ['provName', 'provUrl', 'provKey'].forEach((id) => { $(id).value = ''; });
      $('provKind').value = 'openai';
      $('provLocal').checked = false;
      $('provClearKey').checked = false;
      $('provClearKeyRow').style.display = 'none';
      $('provKey').placeholder = 'sk-...';
      $('provFormTitle').textContent = 'Add a provider';
    }

    function provEdit(id) {
      const p = provState.providers.find((x) => x.id === id);
      if (!p) return;
      provState.editing = p;
      $('provName').value = p.name;
      $('provKind').value = p.kind;
      $('provUrl').value = p.base_url;
      $('provLocal').checked = p.local;
      $('provKey').value = '';
      $('provKey').placeholder = p.has_key ? '(saved - leave empty to keep it)' : 'sk-...';
      $('provClearKey').checked = false;
      $('provClearKeyRow').style.display = p.has_key ? '' : 'none';
      $('provFormTitle').textContent = `Edit ${p.name}`;
      $('provName').scrollIntoView({ block: 'center' });
    }

    function provFormBody() {
      const body = {
        name: $('provName').value.trim(),
        kind: $('provKind').value,
        base_url: $('provUrl').value.trim(),
        local: $('provLocal').checked,
      };
      if (provState.editing) body.id = provState.editing.id;
      const key = $('provKey').value.trim();
      if (key) body.api_key = key;
      else if (provState.editing && $('provClearKey').checked) body.api_key = '';
      return body;
    }

    async function provTest(body) {
      provShow('Testing...', true);
      try {
        const r = await provRequest('POST', '/api/studio/providers/test', body);
        if (r.ok) provShow(`Connected. ${r.count} model${r.count === 1 ? '' : 's'} found.`, true);
        else provShow(r.error || 'The test failed.', false);
        return r.ok;
      } catch (e) {
        provShow(e.message, false);
        return false;
      }
    }

    async function provSave() {
      const body = provFormBody();
      try {
        const r = await provRequest('POST', '/api/studio/providers', body);
        toast(`Saved ${body.name}`);
        provClearForm();
        await provTest({ id: r.provider.id });
        await provRefresh();
        provLoadModels(false);
      } catch (e) {
        provShow(e.message, false);
      }
    }

    $('btnProvSave').addEventListener('click', provSave);
    $('btnProvCancel').addEventListener('click', provClearForm);
    $('btnProvTestForm').addEventListener('click', () => provTest(provFormBody()));
    $('btnProvRefreshModels').addEventListener('click', async () => {
      provShow('Refreshing the model lists...', true);
      await provLoadModels(true);
      await provRefresh();
      provShow(provState.errors.length ? provState.errors.map((x) => `${x.provider_name}: ${x.error}`).join('\n') : `${provState.models.length} models from ${provState.providers.filter((p) => p.enabled).length} provider(s).`, !provState.errors.length);
    });
    $('settingsMenuNav').addEventListener('click', (e) => {
      if (e.target.closest('[data-spane="providers"]')) provRefresh();
    });
    $('provPresets').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-prov-preset]');
      const t = btn && provState.presets.find((x) => x.id === btn.dataset.provPreset);
      if (!t) return;
      provClearForm();
      $('provName').value = t.name;
      $('provKind').value = t.kind;
      $('provUrl').value = t.base_url;
      $('provLocal').checked = Boolean(t.local);
      $('provKey').focus();
    });
    $('provList').addEventListener('click', async (e) => {
      const btn = e.target.closest('[data-prov-act]');
      const row = e.target.closest('[data-prov]');
      if (!btn || !row || btn.dataset.provAct === 'toggle') return;
      const id = row.dataset.prov;
      try {
        if (btn.dataset.provAct === 'edit') provEdit(id);
        if (btn.dataset.provAct === 'test') {
          btn.disabled = true;
          await provTest({ id });
          btn.disabled = false;
          await provRefresh();
          provLoadModels(false);
        }
        if (btn.dataset.provAct === 'remove') {
          if (!confirm('Remove this provider and its saved key?')) return;
          await provRequest('DELETE', '/api/studio/providers', { id });
          await provRefresh();
          provLoadModels(false);
        }
      } catch (err) {
        toast(err.message);
      }
    });
    $('provList').addEventListener('change', async (e) => {
      const box = e.target.closest('[data-prov-act="toggle"]');
      const row = e.target.closest('[data-prov]');
      if (!box || !row) return;
      try {
        await provRequest('POST', '/api/studio/providers', { id: row.dataset.prov, enabled: box.checked });
        provLoadModels(false);
      } catch (err) {
        toast(err.message);
        provRefresh();
      }
    });

    provLoadModels(false);
