    // Appearance controls
    $('selThemeMode').addEventListener('change', () => {
      state.theme = $('selThemeMode').value;
      localStorage.setItem(STORAGE.theme, state.theme);
      applyAppearance();
    });
    // with "Same as the system" the page follows the system when it switches between light and dark
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.theme === 'system') applyAppearance(); });
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

    // Local API Server & Keys events
    $('btnCreateKey').addEventListener('click', async () => {
      const label = $('inpKeyLabel').value.trim() || 'Studio Key';
      const key = $('inpKeyCustom').value.trim();
      const rpm = parseFloat($('inpKeyRpm').value) || null;
      const tpm = parseFloat($('inpKeyTpm').value) || null;
      const res = await apiFetch('/api/studio/keys', { method: 'POST', body: { label, key, rpm, tpm, admin: $('chkKeyAdmin').checked } });
      const data = await res.json();
      if (!res.ok) { toast(data.error?.message || 'The key could not be created'); return; }
      state.apiKey = data.secret;
      localStorage.setItem(STORAGE.apiKey, state.apiKey);
      $('newKeySecretText').textContent = data.secret;
      $('newKeyBanner').style.display = 'flex';
      $('inpKeyLabel').value = '';
      $('inpKeyCustom').value = '';
      $('chkKeyAdmin').checked = false;
      refreshApiKeys();
      updateCliAndSdkSnippets();
      toast(`API key "${data.key.label}" created`);
    });
    $('btnCopyNewKey').addEventListener('click', async () => {
      await navigator.clipboard.writeText($('newKeySecretText').textContent);
      toast('API key copied');
    });
    $('apiKeysTableContainer').addEventListener('click', async (e) => {
      const rev = e.target.closest('[data-revoke-key]');
      if (!rev) return;
      const res = await apiFetch(`/api/studio/keys/${encodeURIComponent(rev.dataset.revokeKey)}`, { method: 'DELETE' });
      if (res.ok) { toast('API key revoked'); refreshApiKeys(); }
    });
    $('btnSaveGlobalRate').addEventListener('click', async () => {
      const rpm = parseFloat($('inpGlobalRpm').value) || null;
      const tpm = parseFloat($('inpGlobalTpm').value) || null;
      const res = await apiFetch('/api/studio/config', { method: 'POST', body: { rate_limit: { requests_per_minute: rpm, tokens_per_minute: tpm } } });
      if (res.ok) { toast('Global limit saved'); refreshApiKeys(); }
    });
    $('btnCopyServerUrl').addEventListener('click', async () => {
      await navigator.clipboard.writeText($('serverEndpointUrlText').textContent);
      toast('Address copied');
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
      if (res.ok) { toast('CPU instruction set changed'); refreshStudioStatus(); refreshHardware(false); }
    });
    $('chkCpuRepack').addEventListener('change', async () => {
      const res = await apiFetch('/api/studio/config', { method: 'POST', body: { cpu_repack: $('chkCpuRepack').checked } });
      if (res.ok) { toast('Weight repacking changed'); refreshStudioStatus(); }
    });
    $('btnMeasureBandwidth').addEventListener('click', () => {
      toast('Measuring memory bandwidth ...');
      refreshHardware(true);
    });
    $('btnRunSimdSelftest').addEventListener('click', async () => {
      toast('Self test running, about 5 seconds ...');
      const res = await apiFetch('/api/studio/selftest-cpu', { method: 'POST', body: {} });
      const data = await res.json();
      $('simdSelftestCard').style.display = 'block';
      const okAll = data.ok && data.failed === 0;
      $('simdSelftestTitle').textContent = `Processor check: ${data.checks - data.failed} of ${data.checks} checks passed in ${Math.round(data.duration_ms)} ms, ${data.skipped} skipped (feature not supported by this processor)`;
      $('simdSelftestTableWrap').innerHTML = `
        <table class="app-table">
          <thead><tr><th>Feature level</th><th>Checks</th><th>Passed</th><th>Failed</th><th>Result</th></tr></thead>
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
        <p class="app-item-desc" style="margin-top:8px;">The calculations the model needs are checked against exact results for every processor feature level. ${okAll ? 'All correct.' : 'Some results deviate.'}</p>
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
      if (res.ok) { $('inpNewAlias').value = ''; toast(`Alias "${dest}" added`); refreshStudioStatus(); }
    });
    $('providerAliasesList').addEventListener('click', async (e) => {
      const del = e.target.closest('[data-del-alias]');
      if (!del) return;
      const res = await apiFetch('/api/delete', { method: 'DELETE', body: { model: del.dataset.delAlias } });
      if (res.ok) { toast('Alias removed'); refreshStudioStatus(); }
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
      toast('Example copied');
    });
    $('btnCopyCliCmd').addEventListener('click', async () => {
      await navigator.clipboard.writeText($('genCliPre').textContent);
      toast('Command copied');
    });
    $('btnCopyTomlCmd').addEventListener('click', async () => {
      await navigator.clipboard.writeText($('genTomlPre').textContent);
      toast('fllm.toml copied');
    });
    $('btnRefreshAllStatus').addEventListener('click', () => {
      refreshStudioStatus();
      toast('Refreshed from the server');
    });

