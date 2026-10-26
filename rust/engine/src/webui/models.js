    // ==========================================================================
    // Models: what is loaded, what is on this computer, search and download from Hugging Face,
    // and switching the loaded model (the server restarts itself with the new model).
    // ==========================================================================
    const nf = (n) => Number(n || 0).toLocaleString('en-US');
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    /** One line about the computer: processor, graphics cards and the memory a model can use. */
    function describeComputer(hw, hub) {
      if (!hw && state.hardwareError) return `The hardware could not be read: ${state.hardwareError}`;
      const devices = hw?.devices || [];
      const names = [];
      for (const d of devices) {
        if (d.backend === 'cpu') { names.unshift(d.name); continue; }
        const label = `${d.name}${d.dev_type === 'gpu' ? ` (${formatBytes(d.total_memory_bytes)})` : ''}`;
        if (!names.includes(label)) names.push(label);
      }
      const usable = Number(hub?.available_memory_bytes || 0);
      return `${names.join(' · ') || 'Unknown processor'} · ${usable ? `${formatBytes(usable)} usable for a model` : 'memory unknown'}`;
    }

    /** Plain text about this computer and this app, for a bug report. */
    function diagnostics(hw, hub) {
      const s = state.status || {};
      const lines = [`app: ${s.build_info || s.version || 'unknown'}`, `system: ${hw?.os || '?'} ${hw?.arch || '?'}, ${hw?.threads || '?'} threads`];
      lines.push(`backends: ${(hw?.compiled_backends || []).join(', ')}`);
      lines.push(`cpu: ${(hw?.cpu_capabilities || []).join(' ')} (tiers: ${(hw?.supported_cpu_tiers || []).join(', ')})`);
      lines.push(`memory: ${formatBytes(hw?.host_free_memory_bytes)} free, usable for a model ${formatBytes(hub?.available_memory_bytes)}`);
      for (const d of hw?.devices || []) lines.push(`device: ${d.id} ${d.name} [${d.dev_type}, ${d.api}] ${formatBytes(d.free_memory_bytes)} free of ${formatBytes(d.total_memory_bytes)}`);
      if (state.hardwareError) lines.push(`error: ${state.hardwareError}`);
      return lines.join('\n');
    }

    async function refreshHub() {
      try {

        const res = await apiFetch('/api/studio/models');
        if (!res.ok) return;
        state.modelsHub = await res.json();
        renderHub();
        pollDownloads();
        refreshAccount();
        if (!state.updateChecked) { state.updateChecked = true; checkUpdate(); }
        if (!state.hardware) { // looking at the hardware can take a moment: the page is shown first
          try {
            const hw = await apiFetch('/api/studio/hardware');
            if (hw.ok) state.hardware = await hw.json();
            else state.hardwareError = (await hw.json().catch(() => ({}))).error?.message || `HTTP ${hw.status}`;
          } catch (e) { state.hardwareError = e.message; }
          renderHub();
        }
      } catch { /* the server is restarting: the next call asks again */ }
    }

    /** Whether a model of `bytes` is likely to run in the memory that is free right now. */
    function fitBadge(bytes) {
      const free = Number(state.modelsHub?.available_memory_bytes || 0);
      if (!bytes || !free) return '';
      const ok = bytes * 1.15 < free;
      return `<span class="fit ${ok ? 'fit-ok' : 'fit-no'}" title="${ok ? 'Should fit into the memory that is free' : 'May not fit into the memory that is free'}">${ok ? 'fits in memory' : 'may not fit'}</span>`;
    }

    function renderRecommended(hub) {
      const byId = new Map((hub.catalog || []).map((e) => [e.id, e]));
      // the suggestions use the 4-bit quantization: nearly as good as the big ones, much smaller and faster
      const picks = (hub.recommended || []).map((id) => byId.get(id)).filter(Boolean).map((e) => {
        const v = (e.variants || []).find((x) => x.quant === 'Q4_K_M');
        return v ? { ...e, recommended_quant: v.quant, estimated_bytes: v.estimated_bytes } : e;
      });
      $('recommendedCards').innerHTML = picks.length ? `
        <p class="app-item-desc" style="margin-top:12px;">Models that run well here:</p>
        ${picks.map((entry) => `
        <div class="catalog-row">
          <div class="catalog-name">
            <strong>${escapeHtml(entry.display_name)}</strong>
            <span class="quant-chip">${escapeHtml(entry.recommended_quant)}</span>${entry.vision ? '<span class="quant-chip">vision</span>' : ''}
            <small class="mono">about ${(Number(entry.estimated_bytes || 0) / 1073741824).toFixed(1)} GB</small>
          </div>
          <div class="app-item-actions">
            <button class="btn btn-sm btn-primary" data-download="${escapeHtml(entry.id)}" data-quant="${escapeHtml(entry.recommended_quant)}" data-then-load="1">Download and start</button>
          </div>
        </div>`).join('')}` : '';
    }

    function renderHub() {
      const hub = state.modelsHub || {};
      $('computerSummary').textContent = describeComputer(state.hardware, hub);
      renderRecommended(hub);
      const m = state.status?.model || hub.loaded || {};
      const lo = m.load_options || {};
      const shownOptions = lo.device ? `device: ${lo.device} · GPU layers: ${lo.gpu_layers} · threads: ${lo.threads} · conversation memory: ${lo.kv_cache_type} · chats at once: ${m.max_sequences || lo.max_seqs}${lo.speculative && lo.speculative !== 'auto' ? ` · speculative decoding: ${lo.speculative}` : ''}` : '';
      $('loadedModelHubItem').innerHTML = `
        <div class="app-card-item">
          <div class="app-item-info">
            <div class="app-item-title">${escapeHtml(m.name || 'Model')} <span class="quant-chip">${escapeHtml(m.quantization || '')}</span></div>
            <div class="app-item-desc mono">${escapeHtml(m.arch || '')} · ${escapeHtml(m.parameter_size || '')} · ${formatBytes(m.size_bytes)} · ${nf(m.max_context)} tokens of context</div>
            ${shownOptions ? `<div class="app-item-desc mono">${escapeHtml(shownOptions)}</div>` : ''}
          </div>
          <div class="app-item-actions">
            <button class="btn btn-sm btn-primary" data-use-model="1">Chat</button>
            <button class="btn btn-sm btn-outline" data-unload="1" title="Stops the model and frees its memory">Unload</button>
          </div>
        </div>`;

      const cached = hub.cached || [];
      const local = (hub.local || []).map((l) => `
        <div class="app-card-item">
          <div class="app-item-info">
            <div class="app-item-title mono">${escapeHtml(l.name)} ${fitBadge(l.size_bytes)}</div>
            <div class="app-item-desc mono">${formatBytes(l.size_bytes)} · ${escapeHtml(l.path)}</div>
          </div>
          <div class="app-item-actions">
            <button class="btn btn-sm btn-secondary" data-load="${escapeHtml(l.path)}" data-label="${escapeHtml(l.name)}" data-key="${escapeHtml(l.name)}">Load</button>
            <button class="btn btn-sm btn-ghost" data-del-local="${escapeHtml(l.path)}" title="Takes it off this list; the file stays on disk">Remove</button>
          </div>
        </div>`).join('');
      $('cachedModelsHubList').innerHTML = local + (cached.length ? cached.map((c) => `
        <div class="app-card-item">
          <div class="app-item-info">
            <div class="app-item-title mono">${escapeHtml(c.name)} ${fitBadge(c.size_bytes)}</div>
            <div class="app-item-desc mono">${formatBytes(c.size_bytes)} · ${escapeHtml(c.path)}</div>
          </div>
          <div class="app-item-actions">
            <button class="btn btn-sm btn-secondary" data-load="${escapeHtml(c.name)}" data-key="${escapeHtml(baseName(c.path))}" title="Restart the server with this model">Load</button>
            <button class="btn btn-sm btn-ghost" data-del-cache="${escapeHtml(c.name)}">Delete</button>
          </div>
        </div>`).join('') : (local ? '' : '<p class="app-item-desc">No downloaded models yet. Search for one above.</p>'));

      const q = ($('hubSearchInput').value || '').toLowerCase().trim();
      const all = (hub.catalog || []).filter((item) => !q || item.id.toLowerCase().includes(q) || item.architecture.toLowerCase().includes(q));
      // what fits first, the biggest of those on top; models that are too large for this computer stay hidden until asked for
      const rank = (e) => (e.fits === false ? 1 : 0);
      all.sort((a, b) => rank(a) - rank(b) || Number(b.parameters_billion) - Number(a.parameters_billion));
      const tooLarge = all.filter((e) => e.fits === false).length;
      const cat = state.showLarge ? all : all.filter((e) => e.fits !== false);
      $('hubCatalogCards').innerHTML = cat.map((entry) => `
        <div class="catalog-row">
          <div class="catalog-name">
            <strong>${escapeHtml(entry.display_name)}</strong>
            <span class="quant-chip">${escapeHtml(entry.architecture)}</span>${entry.vision ? '<span class="quant-chip">vision</span>' : ''}
            <small class="mono">${escapeHtml(entry.recommended_quant)}, about ${(Number(entry.estimated_bytes || 0) / 1073741824).toFixed(1)} GB ${fitBadge(entry.estimated_bytes)}</small>
          </div>
          <div class="app-item-actions">
            <button class="btn btn-sm btn-outline" data-inspect="${escapeHtml(entry.default_repo || entry.gguf_repo || '')}">Files</button>
            <button class="btn btn-sm btn-secondary" data-download="${escapeHtml(entry.id)}" data-quant="${escapeHtml(entry.recommended_quant)}">Download</button>
            <button class="btn btn-sm btn-primary" data-download="${escapeHtml(entry.id)}" data-quant="${escapeHtml(entry.recommended_quant)}" data-then-load="1">Download and load</button>
          </div>
        </div>`).join('') + (tooLarge ? `<p class="app-item-desc" style="margin-top:10px;"><button class="btn btn-sm btn-ghost" data-toggle-large="1">${state.showLarge ? 'Hide' : 'Show'} ${tooLarge} models that are too large for this computer</button></p>` : '');
    }

    // ---- search and files -------------------------------------------------------------------------
    const looksLikeRepo = (text) => /^[\w.-]+\/[\w.-]+$/.test(text) || /^https?:\/\/(www\.)?(huggingface\.co|hf\.co)\//.test(text) || /^hf:\/\//.test(text);

    function repoOf(text) {
      return text.replace(/^https?:\/\/(www\.)?(huggingface\.co|hf\.co)\//, '').replace(/^hf:\/\//, '').split(/[/:]/).slice(0, 2).join('/');
    }

    async function searchHf() {
      const text = $('hfRepoInput').value.trim();
      if (!text) { toast('Type what to search for, or paste a repository.'); return; }
      if (looksLikeRepo(text)) { await showFiles(repoOf(text)); return; }
      const box = $('hfFilesBox');
      box.style.display = 'block';
      box.innerHTML = '<div class="app-item-desc">Searching Hugging Face ...</div>';
      const res = await apiFetch(`/api/studio/models/search?q=${encodeURIComponent(text)}`);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { box.innerHTML = `<div class="msg-error">${escapeHtml(data.error?.message || 'The search failed.')}</div>`; return; }
      const found = data.results || [];
      box.innerHTML = found.length ? `<div class="result-list">${found.map((r) => `
        <div class="catalog-row">
          <div class="catalog-name"><strong class="mono">${escapeHtml(r.id)}</strong><small>${nf(r.downloads)} downloads · ${nf(r.likes)} likes</small></div>
          <div class="app-item-actions"><button class="btn btn-sm btn-secondary" data-inspect="${escapeHtml(r.id)}">Files</button></div>
        </div>`).join('')}</div>` : '<div class="app-item-desc">Nothing found. Try other words.</div>';
    }

    async function showFiles(repo) {
      $('hfRepoInput').value = repo;
      const box = $('hfFilesBox');
      box.style.display = 'block';
      box.innerHTML = `<div class="app-item-desc">Reading the files of ${escapeHtml(repo)} ...</div>`;
      const res = await apiFetch(`/api/studio/hf-files?repo=${encodeURIComponent(repo)}`);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { box.innerHTML = `<div class="msg-error">${escapeHtml(data.error?.message || 'The repository could not be read.')}</div>`; return; }
      const files = data.gguf_files || [];
      const fitMark = (fits) => (fits === true ? '<span class="fit fit-ok" title="Should fit into the memory that is free">fits in memory</span>'
        : fits === false ? '<span class="fit fit-no" title="May not fit into the memory that is free">may not fit</span>' : '');
      box.innerHTML = (files.length ? `
        <div class="app-item-desc" id="hfCard" style="margin-bottom:10px;white-space:pre-wrap;">Reading the model card ...</div>
        <table class="app-table"><thead><tr><th>${escapeHtml(repo)}</th><th>Quality</th><th>Size</th><th></th></tr></thead><tbody>
          ${files.map((f) => `<tr>
            <td class="mono">${escapeHtml(f.filename)} ${fitMark(f.fits)}</td>
            <td class="mono">${f.quant ? `<span class="quant-chip">${escapeHtml(f.quant)}</span>` : ''}</td>
            <td class="mono tabular">${formatBytes(f.size_bytes)}</td>
            <td class="nowrap">
              <button class="btn btn-xs btn-secondary" data-download="hf://${escapeHtml(repo)}/${escapeHtml(f.filename)}">Download</button>
              <button class="btn btn-xs btn-primary" data-download="hf://${escapeHtml(repo)}/${escapeHtml(f.filename)}" data-then-load="1">Download and load</button>
            </td></tr>`).join('')}
        </tbody></table>` : '<div class="app-item-desc">This repository has no GGUF files.</div>');
      if (files.length) {
        // the model card is read when the files are opened
        apiFetch(`/api/studio/hf-readme?repo=${encodeURIComponent(repo)}`).then((r) => r.json().then((d) => ({ ok: r.ok, d }))).then(({ ok, d }) => {
          const card = $('hfCard');
          if (!card) return;
          if (ok && d.summary) card.textContent = d.summary;
          else card.remove();
        }).catch(() => $('hfCard')?.remove());
      }
    }

    // ---- download with a progress bar -------------------------------------------------------------------
    // The queue lives in the server, so the list is the same after a page reload; this code only shows it.
    const baseName = (p) => String(p || '').split(/[\\/]/).pop();
    const fmtDuration = (s) => (s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.round(s)} s`);
    let downloadsPolling = false;
    let seenDone = null;

    async function fetchDownloads() {
      const d = await (await apiFetch('/api/studio/models/progress')).json();
      state.downloads = d.downloads || [];
      return state.downloads;
    }

    function renderDownloads() {
      const list = state.downloads || [];
      $('downloadsCard').style.display = list.length ? '' : 'none';
      const label = { queued: 'Waiting', running: 'Downloading', paused: 'Paused', failed: 'Failed', done: 'Done' };
      $('downloadList').innerHTML = list.map((d) => {
        const pct = d.total ? Math.min(100, (d.written / d.total) * 100) : 0;
        const running = d.state === 'running';
        const detail = running && d.file
          ? `${d.total ? `${pct.toFixed(0)}% · ` : ''}${formatBytes(d.written)}${d.total ? ` of ${formatBytes(d.total)}` : ''} · ${Number(d.mb_per_s || 0).toFixed(1)} MB/s${d.eta_seconds ? ` · about ${fmtDuration(d.eta_seconds)} left` : ''}`
          : running ? (d.attempt > 1 ? `Connection lost, continuing (attempt ${d.attempt} of ${d.max_attempts}) ...` : 'Starting ...')
          : d.state === 'failed' ? d.error : '';
        const buttons = [];
        if (d.state === 'running' || d.state === 'queued') buttons.push(['pause', 'Pause']);
        if (d.state === 'paused') buttons.push(['resume', 'Resume']);
        if (d.state === 'failed') buttons.push(['resume', 'Try again']);
        buttons.push(d.state === 'done' || d.state === 'failed' ? ['clear', 'Remove from list'] : ['cancel', 'Cancel']);
        return `<div class="app-card-item column">
          <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;width:100%;">
            <div class="app-item-info"><div class="app-item-title mono">${escapeHtml(d.model)}${d.quant ? ` <span class="quant-chip">${escapeHtml(d.quant)}</span>` : ''}</div>
              <div class="app-item-desc">${label[d.state] || d.state}${detail ? ` · ${escapeHtml(detail)}` : ''}</div></div>
            <div class="app-item-actions">${buttons.map(([action, text]) => `<button class="btn btn-xs btn-outline" data-dl-action="${action}" data-dl-id="${d.id}">${text}</button>`).join('')}</div>
          </div>
          ${running && d.total ? `<div class="dl-track" style="width:100%;"><i style="width:${pct.toFixed(1)}%"></i></div>` : ''}
        </div>`;
      }).join('');
    }

    /** Follows the queue until nothing is running or waiting. */
    async function pollDownloads() {
      if (downloadsPolling) return;
      downloadsPolling = true;
      let misses = 0;
      try {
        for (;;) {
          try {
            const list = await fetchDownloads();
            misses = 0;
            renderDownloads();
            const done = new Set(list.filter((d) => d.state === 'done').map((d) => d.id));
            if (seenDone && [...done].some((id) => !seenDone.has(id))) { toast('Download finished'); refreshHub(); }
            seenDone = done;
            if (!list.some((d) => d.state === 'running' || d.state === 'queued')) break;
          } catch { if (++misses > 40) break; }
          await sleep(800);
        }
      } finally { downloadsPolling = false; }
    }

    async function downloadModel(spec, quant, thenLoad) {
      const res = await apiFetch('/api/studio/models/pull', { method: 'POST', body: { model: spec, quant: quant || '' } });
      const started = await res.json().catch(() => ({}));
      if (!res.ok) { toast(started.error?.message || 'The download could not be started.'); return false; }
      toast('Added to the downloads');
      await fetchDownloads().catch(() => {});
      renderDownloads();
      pollDownloads();
      if (!thenLoad) return true;
      // wait for this download, then load it
      for (;;) {
        await sleep(900);
        const entry = (await fetchDownloads().catch(() => state.downloads)).find((d) => d.id === started.id);
        if (!entry) return false; // cancelled
        if (entry.state === 'failed') { toast(entry.error || 'The download failed.'); return false; }
        if (entry.state === 'done') { await refreshHub(); await loadModel(entry.reference || spec, spec, baseName(entry.model_path)); return true; }
      }
    }

    // ---- switching the loaded model ---------------------------------------------------------------------------
    const optionFields = { context: 'optContext', gpu_layers: 'optGpuLayers', threads: 'optThreads', kv_cache_type: 'optKv', max_seqs: 'optSeqs', device: 'optDevice', npu_mode: 'optNpuMode', speculative: 'optSpeculative', draft_model: 'optDraftModel', speculative_tokens: 'optSpecTokens' };
    // the NPU choice only matters when the model runs on the NPU
    function syncNpuMode() {
      const onNpu = /^(npu|openvino)/.test($('optDevice').value);
      for (const id of ['optNpuMode', 'optNpuModeLabel']) $(id).style.display = onNpu ? '' : 'none';
    }
    $('optDevice').addEventListener('change', syncNpuMode);

    // speculative decoding: the choices that go with the mode (the texts are in sampler.js)
    function syncSpeculative() {
      const mode = $('optSpeculative').value;
      const show = (ids, on) => ids.forEach((id) => { $(id).style.display = on ? '' : 'none'; });
      show(['optDraftModelLabel', 'optDraftModel'], mode === 'draft');
      show(['optSpecTokensLabel', 'optSpecTokens'], mode === 'lookup' || mode === 'draft');
      const note = speculativeNote(mode);
      $('optSpeculativeNote').textContent = note;
      show(['optSpeculativeNote'], Boolean(note));
    }
    $('optSpeculative').addEventListener('change', syncSpeculative);

    /** The downloaded models that can serve as a draft model, the smallest first. */
    function fillDraftChoices(selected) {
      const hub = state.modelsHub || {};
      const files = [...(hub.local || []), ...(hub.cached || []).filter((c) => /\.gguf$/i.test(c.path || ''))];
      const seen = new Set();
      const rows = files.filter((f) => f.path && !seen.has(f.path) && seen.add(f.path)).sort((a, b) => (a.size_bytes || 0) - (b.size_bytes || 0));
      $('optDraftModel').innerHTML = `<option value="">Choose a model</option>${rows.map((f) => `<option value="${escapeHtml(f.path)}">${escapeHtml(f.name || baseName(f.path))} (${formatBytes(f.size_bytes)})</option>`).join('')}`;
      if (selected) $('optDraftModel').value = selected;
    }

    function readOptions() {
      const out = {};
      // a path keeps its case
      for (const [key, id] of Object.entries(optionFields)) out[key] = (key === 'draft_model' ? $(id).value.trim() : $(id).value.trim().toLowerCase()) || null;
      return out;
    }

    /** Restarts the server and reloads the page when it is back. `done(status)` tells whether the restart has finished. */
    async function waitForRestart(done, waiting) {
      let sawDown = false;
      const started = Date.now();
      while (Date.now() - started < 240000) {
        await sleep(1500);
        try {
          const r = await fetch('/api/studio/status', { cache: 'no-store' });
          if (r.ok && (sawDown || done(await r.json()))) { location.reload(); return; }
        } catch { sawDown = true; }
        $('switchText').textContent = sawDown ? waiting : 'Stopping the old server ...';
      }
      $('switchTitle').textContent = 'The server did not come back';
      $('switchText').textContent = 'Check the terminal where fLLM runs. If it stopped, start it again, then reload this page.';
    }

    /** Asks how to load the model (with the advanced options), then restarts the server with it. `key` finds the options saved for the model. */
    async function loadModel(reference, label, key) {
      const saved = state.modelsHub?.model_options?.[key] || {};
      fillDraftChoices(saved.draft_model);
      for (const [name, id] of Object.entries(optionFields)) $(id).value = saved[name] ?? '';
      syncSpeculative();
      $('loadAdvanced').open = Object.values(saved).some((v) => v !== null && v !== undefined);
      await fillDeviceChoices(saved.device);
      syncNpuMode();
      $('loadOptionsTitle').textContent = 'Load this model?';
      $('loadOptionsText').textContent = `The server restarts with ${label || reference}. This takes a few seconds; your chats stay in this browser.`;
      $('loadOptionsError').style.display = 'none';
      $('loadOptionsModal').classList.add('open');
      const before = state.status?.model?.name;
      try {
        for (;;) {
          const go = await new Promise((resolve) => { $('loadOptionsOk').onclick = () => resolve(true); $('loadOptionsCancel').onclick = () => resolve(false); });
          if (!go) return;
          const body = { model: reference };
          if ($('loadAdvanced').open) body.options = readOptions();
          $('loadOptionsOk').disabled = true;
          $('loadOptionsOk').textContent = 'Preparing ...';
          const res = await apiFetch('/api/studio/models/load', { method: 'POST', body });
          const data = await res.json().catch(() => ({}));
          $('loadOptionsOk').disabled = false;
          $('loadOptionsOk').textContent = 'Load';
          if (res.ok) break;
          $('loadOptionsError').textContent = data.error?.message || 'The model could not be loaded.';
          $('loadOptionsError').style.display = '';
        }
      } finally {
        $('loadOptionsOk').disabled = false;
        $('loadOptionsOk').textContent = 'Load';
        $('loadOptionsModal').classList.remove('open');
      }
      $('switchTitle').textContent = 'Switching model';
      $('switchText').textContent = 'The server restarts with the new model ...';
      $('switchOverlay').classList.add('open');
      await waitForRestart((d) => d.model?.name !== before, 'Loading the model ...');
    }

    async function unloadModel() {
      const ok = await askDialog({ title: 'Unload the model?', text: 'The model is removed from memory. You can load it again from Models.', ok: 'Unload' });
      if (!ok) return;
      const res = await apiFetch('/api/studio/models/unload', { method: 'POST' });
      if (!res.ok) { toast((await res.json().catch(() => ({}))).error?.message || 'The model could not be unloaded.'); return; }
      $('switchTitle').textContent = 'Unloading the model';
      $('switchText').textContent = 'The server restarts without a model ...';
      $('switchOverlay').classList.add('open');
      await waitForRestart((d) => Boolean(d.setup), 'Almost done ...');
    }

    // ---- Hugging Face account ------------------------------------------------------------------------------
    async function refreshAccount() {
      try {
        const d = await (await apiFetch('/api/studio/hf-account')).json();
        $('hfAccountText').textContent = d.signed_in
          ? `Signed in as ${d.name}${d.source === 'environment' ? ' (token from the environment)' : ''}.`
          : (d.error ? `The saved token does not work: ${d.error}` : 'Not signed in. Public models need no account.');
        $('btnHfSignOut').style.display = d.signed_in && d.source === 'saved' ? '' : 'none';
      } catch { $('hfAccountText').textContent = 'The account could not be checked.'; }
    }

    // ---- updates -------------------------------------------------------------------------------------------------
    async function checkUpdate() {
      try {
        const d = await (await apiFetch('/api/studio/update')).json();
        state.update = d;
        const canInstall = Boolean(d.installer);
        $('updateCard').style.display = d.newer ? '' : 'none';
        if (d.newer) {
          $('updateText').textContent = `Version ${d.latest} is available (you have ${d.current}).${canInstall ? '' : ' There is no installer for this system yet; open the release page.'}`;
          $('btnUpdateInstall').style.display = canInstall ? '' : 'none';
        }
        const i = d.install || {};
        $('updateProgress').textContent = i.state === 'downloading' ? `Downloading ${formatBytes(i.written)}${i.total ? ` of ${formatBytes(i.total)}` : ''} ...`
          : i.state === 'started' ? 'The installer has started. Close fLLM and follow its steps.'
          : i.state === 'failed' ? `Failed: ${i.error}` : '';
        return i.state;
      } catch { return 'idle'; }
    }
    // ---- events ---------------------------------------------------------------------------------------------------
    $('hubSearchInput').addEventListener('input', renderHub);
    $('btnCopyDiag').addEventListener('click', () => { navigator.clipboard.writeText($('computerDetails').textContent).then(() => toast('Copied')); });
    $('btnRefreshHub').addEventListener('click', refreshHub);
    async function addLocal() {
      const res = await apiFetch('/api/studio/models/local', { method: 'POST', body: { path: $('localPathInput').value } });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) { toast(d.error?.message || 'The file could not be added.'); return; }
      $('localPathInput').value = '';
      toast(d.added.length === 1 ? 'Model added' : `Added ${d.added.length} models`);
      refreshHub();
    }
    $('btnAddLocal').addEventListener('click', addLocal);

    /** Opens the file or folder dialog of the operating system; `null` when it was closed or cannot be shown. */
    async function browse(kind) {
      const res = await apiFetch('/api/studio/pick-file', { method: 'POST', body: { kind } });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) { toast(d.error?.message || 'The file dialog could not be opened.'); return null; }
      return d.path || null;
    }
    $('btnBrowseLocal').addEventListener('click', async () => {
      const path = await browse('gguf');
      if (path) { $('localPathInput').value = path; addLocal(); }
    });

    // ---- Settings: models folder, idle unload, logs, speed test ---------------------------------------------
    async function loadModelSettings() {
      try {
        const d = await (await apiFetch('/api/studio/settings')).json();
        $('inpModelsDir').value = d.models_dir || '';
        $('inpIdleMinutes').value = d.idle_unload_minutes || 0;
        $('modelsDirActive').textContent = `In use now: ${d.models_dir_active}`;
        $('inpModelsDir').placeholder = d.models_dir_default ? `Default: ${d.models_dir_default}` : 'Default folder';
      } catch { /* the server is restarting */ }
    }
    async function saveSettings(body, message) {
      const res = await apiFetch('/api/studio/settings', { method: 'POST', body });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) { toast(d.error?.message || 'The setting could not be saved.'); return; }
      toast(message);
      loadModelSettings();
    }
    $('btnSaveModelsDir').addEventListener('click', () => saveSettings({ models_dir: $('inpModelsDir').value }, 'Models folder saved. New downloads go there.'));
    $('btnBrowseModelsDir').addEventListener('click', async () => {
      const path = await browse('folder');
      if (path) $('inpModelsDir').value = path;
    });
    $('btnSaveIdle').addEventListener('click', () => saveSettings({ idle_unload_minutes: Number($('inpIdleMinutes').value || 0) }, 'Saved'));
    $('btnUnloadSettings').addEventListener('click', () => (state.setup ? toast('No model is loaded.') : unloadModel()));

    async function refreshLogs() {
      try {
        const d = await (await apiFetch('/api/studio/logs?lines=300')).json();
        const box = $('preLogs');
        const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 8;
        box.textContent = (d.lines || []).join('\n') || 'Nothing has been logged yet.';
        if (atEnd) box.scrollTop = box.scrollHeight;
      } catch { $('preLogs').textContent = 'The logs could not be read.'; }
    }
    $('btnRefreshLogs').addEventListener('click', refreshLogs);
    $('btnCopyLogs').addEventListener('click', () => { navigator.clipboard.writeText($('preLogs').textContent).then(() => toast('Copied')); });
    $('settingsMenuNav').addEventListener('click', (e) => {
      const pane = e.target.closest('[data-spane]')?.dataset.spane;
      if (pane === 'models') loadModelSettings();
      if (pane === 'logs') refreshLogs();
    });

    $('btnSpeedTest').addEventListener('click', async () => {
      if (state.setup) { toast('Load a model first.'); return; }
      const button = $('btnSpeedTest');
      const out = $('speedTestResult');
      button.disabled = true;
      out.textContent = 'Testing, this takes a few seconds ...';
      try {
        const text = 'The quick brown fox jumps over the lazy dog. '.repeat(24);
        const res = await apiFetch('/v1/chat/completions', { method: 'POST', body: {
          model: state.status?.model?.name, stream: false, temperature: 0, max_tokens: 64,
          messages: [{ role: 'user', content: `${text}\nNow write a short story about a fox.` }],
        } });
        const d = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(d.error?.message || 'The test failed.');
        const t = d.timings || {};
        if (!state.hardware) { try { state.hardware = await (await apiFetch('/api/studio/hardware')).json(); } catch { /* shown without the device name */ } }
        const device = state.status?.model?.load_options?.device || 'auto';
        const hw = state.hardware;
        const deviceName = device === 'auto' ? (hw?.devices || []).find((x) => x.id === hw.auto_choice)?.name : device;
        const rate = (value) => (Number(value) > 0 ? `${Number(value).toFixed(1)} tokens per second` : 'too short to measure');
        out.textContent = `Reading the prompt: ${rate(t.prompt_per_second)} (${nf(t.prompt_n)} tokens). Writing the answer: ${rate(t.predicted_per_second)} (${nf(t.predicted_n)} tokens). Device: ${deviceName || device}.`;
      } catch (e) {
        out.textContent = `The speed test failed: ${e.message}`;
      } finally { button.disabled = false; }
    });
    $('localPathInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') addLocal(); });
    $('btnHfSave').addEventListener('click', async () => {
      const res = await apiFetch('/api/studio/hf-account', { method: 'POST', body: { token: $('hfTokenInput').value } });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) { toast(d.error?.message || 'The token was not accepted.'); return; }
      $('hfTokenInput').value = '';
      toast(d.signed_in ? `Signed in as ${d.name}` : 'Signed out');
      refreshAccount();
    });
    $('btnHfSignOut').addEventListener('click', async () => {
      await apiFetch('/api/studio/hf-account', { method: 'POST', body: { token: '' } });
      refreshAccount();
    });
    const openPage = (url) => apiFetch('/api/studio/open-url', { method: 'POST', body: { url } });
    $('btnHfTokenPage').addEventListener('click', () => openPage('https://huggingface.co/settings/tokens'));
    $('btnUpdatePage').addEventListener('click', () => openPage(state.update?.page || 'https://github.com/Marco-Lu2508/fLLM/releases'));
    $('btnUpdateInstall').addEventListener('click', async () => {
      const res = await apiFetch('/api/studio/update/install', { method: 'POST' });
      if (!res.ok) { toast((await res.json().catch(() => ({}))).error?.message || 'The download could not be started.'); return; }
      while ((await checkUpdate()) === 'downloading') await sleep(800);
    });
    $('btnSearchHf').addEventListener('click', searchHf);
    $('hfRepoInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') searchHf(); });
    $('surface-hub').addEventListener('click', async (e) => {
      if (e.target.closest('[data-use-model]')) { switchSurface('chat'); return; }
      if (e.target.closest('[data-unload]')) { await unloadModel(); return; }
      const dlAction = e.target.closest('[data-dl-action]');
      if (dlAction) {
        const res = await apiFetch('/api/studio/models/downloads', { method: 'POST', body: { id: Number(dlAction.dataset.dlId), action: dlAction.dataset.dlAction } });
        if (!res.ok) toast((await res.json().catch(() => ({}))).error?.message || 'That did not work.');
        await fetchDownloads().catch(() => {});
        renderDownloads();
        pollDownloads();
        return;
      }
      if (e.target.closest('[data-toggle-large]')) { state.showLarge = !state.showLarge; renderHub(); return; }
      const inspect = e.target.closest('[data-inspect]');
      if (inspect) { await showFiles(inspect.dataset.inspect); return; }
      const dl = e.target.closest('[data-download]');
      if (dl) { await downloadModel(dl.dataset.download, dl.dataset.quant, Boolean(dl.dataset.thenLoad)); return; }
      const load = e.target.closest('[data-load]');
      if (load) { await loadModel(load.dataset.load, load.dataset.label || load.dataset.load, load.dataset.key); return; }
      const delLocal = e.target.closest('[data-del-local]');
      if (delLocal) { await apiFetch('/api/studio/models/local', { method: 'DELETE', body: { path: delLocal.dataset.delLocal } }); refreshHub(); return; }
      const del = e.target.closest('[data-del-cache]');
      if (del) {
        if (!(await askDialog({ title: 'Delete this download?', text: `${del.dataset.delCache} is removed from the model cache.`, ok: 'Delete', danger: true }))) return;
        const res = await apiFetch('/api/studio/models/cache', { method: 'DELETE', body: { name: del.dataset.delCache } });
        if (res.ok) { toast('Deleted from the cache'); refreshHub(); }
        else toast((await res.json().catch(() => ({}))).error?.message || 'Could not delete it.');
      }
    });
