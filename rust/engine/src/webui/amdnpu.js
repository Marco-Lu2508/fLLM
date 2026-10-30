    // ==========================================================================
    // AMD NPU through FastFlowLM: the card in Settings, Hardware. fLLM has no kernels for the XDNA NPU of a Ryzen AI processor; FastFlowLM (a separate
    // program you install) has. The server finds it, downloads models, runs "flm serve" for one model and lists that model as the provider
    // "AMD NPU (FastFlowLM)" in the model picker. The state comes from GET /api/studio/npu/amd; the card is built from it by amdNpuModel (pure,
    // tested in node).
    // ==========================================================================

    // @pure-begin
    const AMD_NPU_NOTE = 'FastFlowLM is a separate program (its NPU kernels are AMD\'s, free to use). It has its own model files, so a model here is not the same file as one in your fLLM model list.';

    function amdNpuText(s, serving, problems) {
      const running = serving.state === 'running' ? serving.model : '';
      if (s.state === 'flm_missing') return 'An AMD NPU was found. fLLM runs models on it through FastFlowLM, which you install yourself (a free download).';
      if (s.state === 'needs_attention') return `FastFlowLM reports a problem with the NPU setup: ${problems.join('; ') || 'see the details'}`;
      if (serving.state === 'running') return `Running ${running} on the NPU. Pick "AMD NPU (FastFlowLM) · ${running}" in the model list of the chat.`;
      if (serving.state === 'starting') return `Starting ${serving.model} on the NPU ... (the first start of a model can take a few minutes)`;
      if (serving.state === 'failed') return `${serving.model} did not start.`;
      return 'Ready. Choose a model and run it on the NPU.';
    }

    /** The status line of fLLM's own (native) XDNA kernels: `n` is the "native" object of the API state. Empty when the build has none. */
    function amdNpuNativeText(n) {
      if (!n || !n.compiled) return '';
      if (n.available) return 'Native fLLM kernels: ready. Start the server with FLLM_NPU_HOST=1 and the long prompts of any model run on the NPU, the rest on the processor (or choose NPU under Advanced when you load a model).';
      return `Native fLLM kernels: not available${n.reason ? ` (${n.reason})` : ''}. FastFlowLM, below, does not need them.`;
    }

    /**
     * Everything the card shows, from the API state `s` and the model chosen in the list. No DOM here. `visible` is false when there is no AMD
     * NPU and no FastFlowLM on the computer.
     */
    function amdNpuModel(s, chosen) {
      if (!s || s.state === 'no_npu') return { visible: false };
      const flm = s.flm || {};
      const serving = s.serving || { state: 'idle' };
      const pull = s.pull || {};
      const problems = (s.validation && s.validation.problems) || [];
      const models = (s.models || []).slice().sort((a, b) => (b.installed - a.installed) || a.tag.localeCompare(b.tag));
      const running = serving.state === 'running' ? serving.model : null;
      const starting = serving.state === 'starting' ? serving.model : null;
      const tag = models.some((m) => m.tag === chosen) ? chosen : (running || (models.find((m) => m.installed) || models[0] || {}).tag || '');
      const picked = models.find((m) => m.tag === tag) || null;
      const ready = s.state === 'ready';
      const downloading = !!pull.busy;
      let error = pull.error || (serving.state === 'failed' ? serving.error : null) || s.error || null;
      if (pull.error === 'cancelled') error = null;
      return {
        visible: true,
        summary: amdNpuText(s, serving, problems),
        nativeText: amdNpuNativeText(s.native),
        nativeReady: !!(s.native && s.native.available),
        tone: running ? 'good' : (s.state === 'needs_attention' || error ? 'warn' : 'muted'),
        hardware: (s.hardware || []).map((h) => ({ label: h.name, problem: h.problem || null })),
        install: s.state === 'flm_missing' ? (flm.install || null) : null,
        problems,
        version: flm.version || null,
        models: models.map((m) => ({ tag: m.tag, label: m.tag + (m.installed ? '' : ' (not downloaded)'), installed: !!m.installed })),
        selected: tag,
        start: ready && !!picked && picked.installed && !downloading && running !== tag && starting !== tag,
        stop: !!running || !!starting,
        pull: ready && !!picked && !picked.installed && !downloading,
        cancelPull: downloading,
        busy: !!starting || downloading,
        pullText: downloading ? `Downloading ${pull.model}: ${pull.line || '...'}` : (pull.line === 'done' && pull.model ? `${pull.model} downloaded.` : ''),
        error,
        note: AMD_NPU_NOTE,
      };
    }
    // @pure-end

    // ---- the card ------------------------------------------------------------------------------------------------------------
    let amdNpuTimer = null;
    let amdNpuState = null;
    let amdNpuChosen = '';

    async function refreshAmdNpu() {
      let data;
      try {
        const res = await apiFetch('/api/studio/npu/amd');
        if (!res.ok) return;
        data = await res.json();
      } catch { return; }
      const was = amdNpuState && amdNpuState.serving && amdNpuState.serving.state;
      amdNpuState = data;
      renderAmdNpuCard();
      const now = data.serving && data.serving.state;
      // the running model is a provider: the model list learns about it (and forgets it when it stops)
      if (was !== now && (now === 'running' || was === 'running')) {
        if (typeof provRefresh === 'function') provRefresh();
        if (typeof provLoadModels === 'function') provLoadModels(true);
      }
      const m = amdNpuModel(data, amdNpuChosen);
      if (m.busy && !amdNpuTimer) amdNpuTimer = setInterval(refreshAmdNpu, 1500);
      if (!m.busy && amdNpuTimer) { clearInterval(amdNpuTimer); amdNpuTimer = null; }
    }

    function renderAmdNpuCard() {
      const card = $('amdNpuCard');
      if (!card) return;
      const m = amdNpuModel(amdNpuState, amdNpuChosen);
      card.style.display = m.visible ? '' : 'none';
      if (!m.visible) return;
      amdNpuChosen = m.selected;
      $('amdNpuSummary').textContent = m.summary;
      $('amdNpuSummary').style.color = m.tone === 'good' ? 'var(--primary)' : '';
      $('amdNpuNative').textContent = m.nativeText;
      $('amdNpuNative').style.display = m.nativeText ? '' : 'none';
      $('amdNpuNative').style.color = m.nativeReady ? 'var(--primary)' : '';
      $('amdNpuHardware').innerHTML = m.hardware.map((h) => `<div class="app-item-desc">${escapeHtml(h.label)}${h.problem ? ` - ${escapeHtml(h.problem)}` : ''}</div>`).join('');
      const install = $('amdNpuInstall');
      install.style.display = m.install ? '' : 'none';
      if (m.install) install.innerHTML = `<p class="app-item-desc">${escapeHtml(m.install.text)}</p><p class="app-item-desc"><a href="${escapeHtml(m.install.url)}" target="_blank" rel="noopener">${escapeHtml(m.install.url)}</a></p>`;
      $('amdNpuProblems').innerHTML = m.problems.map((p) => `<div class="msg-error" style="margin:4px 0;">${escapeHtml(p)}</div>`).join('');
      const select = $('amdNpuModel');
      select.style.display = m.models.length ? '' : 'none';
      select.innerHTML = m.models.map((o) => `<option value="${escapeHtml(o.tag)}">${escapeHtml(o.label)}</option>`).join('');
      select.value = m.selected;
      $('amdNpuStart').style.display = m.start ? '' : 'none';
      $('amdNpuStop').style.display = m.stop ? '' : 'none';
      $('amdNpuPull').style.display = m.pull ? '' : 'none';
      $('amdNpuCancelPull').style.display = m.cancelPull ? '' : 'none';
      $('amdNpuPullText').textContent = m.pullText;
      $('amdNpuPullText').style.display = m.pullText ? '' : 'none';
      $('amdNpuError').textContent = m.error || '';
      $('amdNpuError').style.display = m.error ? '' : 'none';
      $('amdNpuNote').textContent = m.note;
    }

    async function amdNpuAction(action) {
      try {
        const res = await apiFetch('/api/studio/npu/amd', { method: 'POST', body: { action, model: amdNpuChosen } });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) { toast((data.error && data.error.message) || 'The NPU action failed.'); return; }
        amdNpuState = data;
        renderAmdNpuCard();
      } catch { toast('The NPU action failed.'); }
      refreshAmdNpu();
    }

    $('amdNpuModel')?.addEventListener('change', (e) => { amdNpuChosen = e.target.value; renderAmdNpuCard(); });
    $('amdNpuStart')?.addEventListener('click', () => amdNpuAction('start'));
    $('amdNpuStop')?.addEventListener('click', () => amdNpuAction('stop'));
    $('amdNpuPull')?.addEventListener('click', () => amdNpuAction('pull'));
    $('amdNpuCancelPull')?.addEventListener('click', () => amdNpuAction('cancel_pull'));

    // The card refreshes when the Hardware pane is shown (the pane gets the class "active").
    if ($('spane-hardware')) {
      new MutationObserver(() => { if ($('spane-hardware').classList.contains('active')) refreshAmdNpu(); })
        .observe($('spane-hardware'), { attributes: true, attributeFilter: ['class'] });
    }
