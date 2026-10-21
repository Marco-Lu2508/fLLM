    // Appearance controls
    $('selThemeMode').addEventListener('change', () => {
      state.theme = $('selThemeMode').value;
      localStorage.setItem(STORAGE.theme, state.theme);
      applyAppearance();
    });
    $('selFontSize').addEventListener('change', () => {
      state.fontSize = $('selFontSize').value;
      localStorage.setItem(STORAGE.fontSize, state.fontSize);
      applyAppearance();
    });
    $('chkColoredBubble').addEventListener('change', () => {
      state.coloredBubble = $('chkColoredBubble').checked;
      localStorage.setItem(STORAGE.coloredBubble, String(state.coloredBubble));
      applyAppearance();
    });
    $('chkShowSpeed').addEventListener('change', () => {
      state.showSpeed = $('chkShowSpeed').checked;
      localStorage.setItem(STORAGE.showSpeed, String(state.showSpeed));
      renderChatArea();
    });
    $('chkAutoTitle').addEventListener('change', () => {
      state.autoTitle = $('chkAutoTitle').checked;
      localStorage.setItem(STORAGE.autoTitle, String(state.autoTitle));
    });
    $('accentPickerRow').addEventListener('click', (e) => {
      const sw = e.target.closest('[data-accent]');
      if (!sw) return;
      state.accent = sw.dataset.accent;
      localStorage.setItem(STORAGE.accent, state.accent);
      applyAppearance();
    });

    // Hub events
    $('hubSearchInput').addEventListener('input', renderHub);
    $('btnRefreshHub').addEventListener('click', refreshHub);
    $('btnInspectHfRepo').addEventListener('click', async () => {
      const repo = $('hfRepoInput').value.trim();
      if (!repo) { toast('Gib zuerst ein Hugging-Face-Repository an'); return; }
      const box = $('hfFilesBox');
      box.style.display = 'block';
      box.innerHTML = `<div class="app-item-desc">Fetching files from ${escapeHtml(repo)}...</div>`;
      const res = await apiFetch(`/api/studio/hf-files?repo=${encodeURIComponent(repo)}`);
      const data = await res.json();
      if (!res.ok) { box.innerHTML = `<div style="color:var(--destructive);font-size:12px;">${escapeHtml(data.error?.message || 'Error')}</div>`; return; }
      const files = data.gguf_files?.length ? data.gguf_files : (data.all_files || []);
      box.innerHTML = `
        <table class="app-table">
          <thead><tr><th>GGUF File</th><th>Size</th><th></th></tr></thead>
          <tbody>
            ${files.map((f) => `<tr><td class="mono">${escapeHtml(f.filename)}</td><td class="mono">${formatBytes(f.size_bytes)}</td><td><button class="btn btn-xs btn-primary" data-pull-hf="${escapeHtml(data.repo)}:${escapeHtml(f.filename)}">Download</button></td></tr>`).join('')}
          </tbody>
        </table>
      `;
    });
    $('btnPullHfModel').addEventListener('click', async () => {
      const model = $('hfRepoInput').value.trim();
      const quant = $('hfQuantInput').value.trim();
      if (!model) return;
      toast(`${model} wird geladen ...`);
      const res = await apiFetch('/api/studio/models/pull', { method: 'POST', body: { model, quant } });
      const d = await res.json();
      if (res.ok) { toast(`Geladen: ${d.model_path}`); refreshHub(); }
      else toast(d.error?.message || 'Download failed');
    });
    $('surface-hub').addEventListener('click', async (e) => {
      const useBtn = e.target.closest('[data-use-model]');
      if (useBtn) { switchSurface('chat'); return; }
      const insp = e.target.closest('[data-inspect-hf]');
      if (insp) { $('hfRepoInput').value = insp.dataset.inspectHf; $('btnInspectHfRepo').click(); return; }
      const pullCat = e.target.closest('[data-pull-cat]');
      if (pullCat) { $('hfRepoInput').value = pullCat.dataset.pullCat; $('hfQuantInput').value = pullCat.dataset.pullQ || ''; $('btnPullHfModel').click(); return; }
      const pullHf = e.target.closest('[data-pull-hf]');
      if (pullHf) { $('hfRepoInput').value = pullHf.dataset.pullHf; $('hfQuantInput').value = ''; $('btnPullHfModel').click(); return; }
      const delCache = e.target.closest('[data-del-cache]');
      if (delCache) {
        const res = await apiFetch('/api/studio/models/cache', { method: 'DELETE', body: { name: delCache.dataset.delCache } });
        if (res.ok) { toast('Aus dem Cache gelöscht'); refreshHub(); }
      }
    });

    // Local API Server & Keys events
    $('btnCreateKey').addEventListener('click', async () => {
      const label = $('inpKeyLabel').value.trim() || 'Studio Key';
      const key = $('inpKeyCustom').value.trim();
      const rpm = parseFloat($('inpKeyRpm').value) || null;
      const tpm = parseFloat($('inpKeyTpm').value) || null;
      const res = await apiFetch('/api/studio/keys', { method: 'POST', body: { label, key, rpm, tpm } });
      const data = await res.json();
      if (!res.ok) { toast(data.error?.message || 'Schlüssel konnte nicht angelegt werden'); return; }
      state.apiKey = data.secret;
      localStorage.setItem(STORAGE.apiKey, state.apiKey);
      $('newKeySecretText').textContent = data.secret;
      $('newKeyBanner').style.display = 'flex';
      $('inpKeyLabel').value = '';
      $('inpKeyCustom').value = '';
      refreshApiKeys();
      updateCliAndSdkSnippets();
      toast(`API-Schlüssel "${data.key.label}" angelegt`);
    });
    $('btnCopyNewKey').addEventListener('click', async () => {
      await navigator.clipboard.writeText($('newKeySecretText').textContent);
      toast('API-Schlüssel kopiert');
    });
    $('apiKeysTableContainer').addEventListener('click', async (e) => {
      const rev = e.target.closest('[data-revoke-key]');
      if (!rev) return;
      const res = await apiFetch(`/api/studio/keys/${encodeURIComponent(rev.dataset.revokeKey)}`, { method: 'DELETE' });
      if (res.ok) { toast('API-Schlüssel widerrufen'); refreshApiKeys(); }
    });
    $('btnSaveGlobalRate').addEventListener('click', async () => {
      const rpm = parseFloat($('inpGlobalRpm').value) || null;
      const tpm = parseFloat($('inpGlobalTpm').value) || null;
      const res = await apiFetch('/api/studio/config', { method: 'POST', body: { rate_limit: { requests_per_minute: rpm, tokens_per_minute: tpm } } });
      if (res.ok) { toast('Allgemeines Limit gespeichert'); refreshApiKeys(); }
    });
    $('btnCopyServerUrl').addEventListener('click', async () => {
      await navigator.clipboard.writeText($('serverEndpointUrlText').textContent);
      toast('Adresse kopiert');
    });
    $('selRouteTest').addEventListener('change', updateRouteTesterSample);
    $('btnRunRouteTest').addEventListener('click', async () => {
      const [method, path] = $('selRouteTest').value.split(':');
      const opts = { method };
      if (method === 'POST') opts.body = JSON.parse($('txtRouteReq').value || '{}');
      const res = await apiFetch(path, opts);
      const txt = await res.text();
      try { $('preRouteRes').textContent = JSON.stringify(JSON.parse(txt), null, 2); }
      catch { $('preRouteRes').textContent = txt; }
    });
    $('btnRefreshMetrics').addEventListener('click', refreshPrometheus);

    // Hardware & SIMD events
    $('selSimdTierCap').addEventListener('change', async () => {
      const res = await apiFetch('/api/studio/config', { method: 'POST', body: { cpu_tier_cap: $('selSimdTierCap').value } });
      if (res.ok) { toast('CPU-Stufe geändert'); refreshStudioStatus(); refreshHardware(false); }
    });
    $('chkCpuRepack').addEventListener('change', async () => {
      const res = await apiFetch('/api/studio/config', { method: 'POST', body: { cpu_repack: $('chkCpuRepack').checked } });
      if (res.ok) { toast('Umpacken der Gewichte geändert'); refreshStudioStatus(); }
    });
    $('btnMeasureBandwidth').addEventListener('click', () => {
      toast('Speicherbandbreite wird gemessen ...');
      refreshHardware(true);
    });
    $('btnRunSimdSelftest').addEventListener('click', async () => {
      toast('Selbsttest läuft, etwa 5 Sekunden ...');
      const res = await apiFetch('/api/studio/selftest-cpu', { method: 'POST', body: {} });
      const data = await res.json();
      $('simdSelftestCard').style.display = 'block';
      const okAll = data.ok && data.failed === 0;
      $('simdSelftestTitle').textContent = `SIMD Kernel Self-Test: ${data.checks - data.failed}/${data.checks} Checks bestanden · ${Math.round(data.duration_ms)} ms · ${data.skipped} uebersprungen (Tier nicht unterstuetzt)`;
      $('simdSelftestTableWrap').innerHTML = `
        <table class="app-table">
          <thead><tr><th>SIMD-Tier</th><th>Checks</th><th>Bestanden</th><th>Fehlgeschlagen</th><th>Status</th></tr></thead>
          <tbody>
            ${(data.tested_tiers || []).map((c) => `
              <tr>
                <td class="mono">${escapeHtml(c.tier)}</td>
                <td class="mono tabular">${c.checks}</td>
                <td class="mono tabular">${c.passed}</td>
                <td class="mono tabular">${c.failed}</td>
                <td><span class="quant-chip" style="color:${c.ok ? 'var(--primary)' : 'var(--destructive)'};">${c.ok ? 'PASS' : 'FAIL'}</span></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
        <p class="app-item-desc" style="margin-top:8px;">Jeder Kernel (Q4/Q6/IQ4, Attention, RMSNorm, ...) wird je Tier gegen eine f64-Referenz geprueft. ${okAll ? 'Alles korrekt.' : 'Es gibt Abweichungen!'}</p>
      `;
    });

    // Tokenizer & Jinja events
    $('tokTextInput').addEventListener('input', runTokenize);
    $('chkTokSpecial').addEventListener('change', runTokenize);
    $('btnRenderJinja').addEventListener('click', runJinjaTemplate);

    // Provider aliases
    $('btnAddAlias').addEventListener('click', async () => {
      const dest = $('inpNewAlias').value.trim();
      if (!dest) return;
      const src = state.status?.model?.name || 'tiny';
      const res = await apiFetch('/api/copy', { method: 'POST', body: { source: src, destination: dest } });
      if (res.ok) { $('inpNewAlias').value = ''; toast(`Alias "${dest}" hinzugefügt`); refreshStudioStatus(); }
    });
    $('providerAliasesList').addEventListener('click', async (e) => {
      const del = e.target.closest('[data-del-alias]');
      if (!del) return;
      const res = await apiFetch('/api/delete', { method: 'DELETE', body: { model: del.dataset.delAlias } });
      if (res.ok) { toast('Alias entfernt'); refreshStudioStatus(); }
    });

    // SDK & CLI copy
    $('sdkSnippetTabs').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-sdk]');
      if (!btn) return;
      state.sdkLang = btn.dataset.sdk;
      document.querySelectorAll('#sdkSnippetTabs .btn').forEach((b) => {
        b.className = b === btn ? 'btn btn-sm btn-secondary' : 'btn btn-sm btn-ghost';
      });
      updateCliAndSdkSnippets();
    });
    $('btnCopySdkSnippet').addEventListener('click', async () => {
      await navigator.clipboard.writeText($('preSdkSnippet').textContent);
      toast('Beispiel kopiert');
    });
    $('btnCopyCliCmd').addEventListener('click', async () => {
      await navigator.clipboard.writeText($('genCliPre').textContent);
      toast('Befehl kopiert');
    });
    $('btnCopyTomlCmd').addEventListener('click', async () => {
      await navigator.clipboard.writeText($('genTomlPre').textContent);
      toast('fllm.toml kopiert');
    });
    $('btnRefreshAllStatus').addEventListener('click', () => {
      refreshStudioStatus();
      toast('Vom Server aktualisiert');
    });

