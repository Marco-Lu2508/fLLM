    // ==========================================================================
    // Models: what is loaded, what is on this computer, search and download from Hugging Face,
    // and switching the loaded model (the server restarts itself with the new model).
    // ==========================================================================
    const nf = (n) => Number(n || 0).toLocaleString('en-US');
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    /** One line about the computer: processor, graphics cards and the memory a model can use. */
    function describeComputer(hw, hub) {
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

    async function refreshHub() {
      try {

        const res = await apiFetch('/api/studio/models');
        if (!res.ok) return;
        state.modelsHub = await res.json();
        renderHub();
        if (!state.hardware) { // looking at the hardware can take a moment: the page is shown first
          const hw = await apiFetch('/api/studio/hardware');
          if (hw.ok) { state.hardware = await hw.json(); renderHub(); }
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
      const picks = (hub.recommended || []).map((id) => byId.get(id)).filter(Boolean);
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
      $('loadedModelHubItem').innerHTML = `
        <div class="app-card-item">
          <div class="app-item-info">
            <div class="app-item-title">${escapeHtml(m.name || 'Model')} <span class="quant-chip">${escapeHtml(m.quantization || '')}</span></div>
            <div class="app-item-desc mono">${escapeHtml(m.arch || '')} · ${escapeHtml(m.parameter_size || '')} · ${formatBytes(m.size_bytes)} · ${nf(m.max_context)} tokens of context</div>
          </div>
          <div class="app-item-actions"><button class="btn btn-sm btn-primary" data-use-model="1">Chat</button></div>
        </div>`;

      const cached = hub.cached || [];
      $('cachedModelsHubList').innerHTML = cached.length ? cached.map((c) => `
        <div class="app-card-item">
          <div class="app-item-info">
            <div class="app-item-title mono">${escapeHtml(c.name)} ${fitBadge(c.size_bytes)}</div>
            <div class="app-item-desc mono">${formatBytes(c.size_bytes)} · ${escapeHtml(c.path)}</div>
          </div>
          <div class="app-item-actions">
            <button class="btn btn-sm btn-secondary" data-load="${escapeHtml(c.name)}" title="Restart the server with this model">Load</button>
            <button class="btn btn-sm btn-ghost" data-del-cache="${escapeHtml(c.name)}">Delete</button>
          </div>
        </div>`).join('') : '<p class="app-item-desc">No downloaded models yet. Search for one above.</p>';

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
      box.innerHTML = files.length ? `
        <table class="app-table"><thead><tr><th>${escapeHtml(repo)}</th><th>Size</th><th></th></tr></thead><tbody>
          ${files.map((f) => `<tr>
            <td class="mono">${escapeHtml(f.filename)} ${fitBadge(f.size_bytes)}</td>
            <td class="mono tabular">${formatBytes(f.size_bytes)}</td>
            <td class="nowrap">
              <button class="btn btn-xs btn-secondary" data-download="hf://${escapeHtml(repo)}/${escapeHtml(f.filename)}">Download</button>
              <button class="btn btn-xs btn-primary" data-download="hf://${escapeHtml(repo)}/${escapeHtml(f.filename)}" data-then-load="1">Download and load</button>
            </td></tr>`).join('')}
        </tbody></table>` : '<div class="app-item-desc">This repository has no GGUF files.</div>';
    }

    // ---- download with a progress bar -------------------------------------------------------------------
    let downloading = false;
    async function downloadModel(spec, quant, thenLoad) {
      if (downloading) { toast('A download is already running.'); return false; }
      downloading = true;
      const bar = $('downloadBar');
      bar.style.display = 'flex';
      $('dlFill').style.width = '0%';
      $('dlText').textContent = 'Starting ...';
      const poll = setInterval(async () => {
        try {
          const d = await (await apiFetch('/api/studio/models/progress')).json();
          if (d.active && d.file) {
            const pct = d.total ? Math.min(100, (d.written / d.total) * 100) : 0;
            $('dlFill').style.width = `${pct.toFixed(1)}%`;
            $('dlText').textContent = `${d.file}: ${d.total ? `${pct.toFixed(0)}%, ` : ''}${formatBytes(d.written)}${d.total ? ` of ${formatBytes(d.total)}` : ''}, ${Number(d.mb_per_s).toFixed(1)} MB/s`;
          }
        } catch { /* a missed poll only delays the bar */ }
      }, 700);
      let ok = false;
      try {
        const res = await apiFetch('/api/studio/models/pull', { method: 'POST', body: { model: spec, quant: quant || '' } });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error?.message || 'The download failed.');
        toast('Download finished');
        ok = true;
        await refreshHub();
        if (thenLoad) await loadModel(data.reference || spec, spec);
      } catch (e) {
        toast(e.message);
      } finally {
        clearInterval(poll);
        downloading = false;
        bar.style.display = 'none';
      }
      return ok;
    }

    // ---- switching the loaded model ---------------------------------------------------------------------------
    async function loadModel(reference, label) {
      const ok = await askDialog({ title: 'Switch the model?', text: `The server restarts with ${label || reference}. This takes a few seconds; your chats stay in this browser.`, ok: 'Switch' });
      if (!ok) return;
      $('switchTitle').textContent = 'Switching model';
      $('switchText').textContent = 'The server restarts with the new model ...';
      $('switchOverlay').classList.add('open');
      const before = state.status?.model?.name;
      try {
        const res = await apiFetch('/api/studio/models/load', { method: 'POST', body: { model: reference } });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error?.message || 'The model could not be loaded.');
      } catch (e) {
        $('switchOverlay').classList.remove('open');
        toast(e.message);
        return;
      }
      let sawDown = false;
      const started = Date.now();
      while (Date.now() - started < 240000) {
        await sleep(1500);
        try {
          const r = await fetch('/api/studio/status', { cache: 'no-store' });
          if (r.ok) {
            const d = await r.json();
            if (sawDown || d.model?.name !== before) { location.reload(); return; }
          }
        } catch { sawDown = true; }
        $('switchText').textContent = sawDown ? 'Loading the model ...' : 'Stopping the old server ...';
      }
      $('switchTitle').textContent = 'The server did not come back';
      $('switchText').textContent = 'Check the terminal where fLLM runs. If it stopped, start it again, then reload this page.';
    }

    // ---- events ---------------------------------------------------------------------------------------------------
    $('hubSearchInput').addEventListener('input', renderHub);
    $('btnRefreshHub').addEventListener('click', refreshHub);
    $('btnSearchHf').addEventListener('click', searchHf);
    $('hfRepoInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') searchHf(); });
    $('surface-hub').addEventListener('click', async (e) => {
      if (e.target.closest('[data-use-model]')) { switchSurface('chat'); return; }
      if (e.target.closest('[data-toggle-large]')) { state.showLarge = !state.showLarge; renderHub(); return; }
      const inspect = e.target.closest('[data-inspect]');
      if (inspect) { await showFiles(inspect.dataset.inspect); return; }
      const dl = e.target.closest('[data-download]');
      if (dl) { await downloadModel(dl.dataset.download, dl.dataset.quant, Boolean(dl.dataset.thenLoad)); return; }
      const load = e.target.closest('[data-load]');
      if (load) { await loadModel(load.dataset.load, load.dataset.load); return; }
      const del = e.target.closest('[data-del-cache]');
      if (del) {
        if (!(await askDialog({ title: 'Delete this download?', text: `${del.dataset.delCache} is removed from the model cache.`, ok: 'Delete', danger: true }))) return;
        const res = await apiFetch('/api/studio/models/cache', { method: 'DELETE', body: { name: del.dataset.delCache } });
        if (res.ok) { toast('Deleted from the cache'); refreshHub(); }
        else toast((await res.json().catch(() => ({}))).error?.message || 'Could not delete it.');
      }
    });
