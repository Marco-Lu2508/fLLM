    // ==========================================================================
    // NPU: the card in Settings, Hardware (what was found, install or remove the runtime that drives it) and the device list of the
    // load options. The state comes from GET /api/studio/npu; the card is built from it by npuCardModel (pure, tested in node).
    // Honest status: the NPU backend has not run on NPU hardware yet (test level L0), so everything here says "experimental".
    // ==========================================================================

    // @pure-begin
    const NPU_WARNING = 'Experimental: slower than the processor for many models; weights are held twice; dense models only.';

    const NPU_EXPLANATION = [
      'Today fLLM can use an Intel NPU only as a helper: the large matrix multiplications of quantized layers run on the NPU while reading a prompt, and everything else runs on the processor.',
      'No speed-up is expected when the model writes its answer, and the model is often slower than on the processor alone. The weights are kept twice in memory (once for the processor, once converted for the NPU). Only ordinary (dense) models work; mixture-of-experts and other special models stay on the processor.',
      'It is never chosen automatically. To use it, pick NPU under Advanced when you load a model, or start fLLM with --device npu. If the NPU cannot take a model, fLLM runs it on the processor and says why here.',
      'This path is experimental and has not been run on NPU hardware by its authors yet.',
    ];

    const NPU_VENDOR_NAMES = { intel: 'Intel', amd: 'AMD', qualcomm: 'Qualcomm', other: 'Unknown maker' };

    const NPU_SUMMARIES = {
      not_in_build: 'This build of fLLM has no NPU support.',
      no_npu: 'No NPU was found on this computer.',
      unsupported_hardware: 'An NPU was detected, but this kind of NPU is not supported yet.',
      setup_unavailable: 'An Intel NPU was found, but fLLM has no automatic setup for this system. Install the OpenVINO runtime yourself and fLLM will find it.',
      no_device: 'The OpenVINO runtime is installed but does not list an NPU. The NPU driver may be missing, or this user may not be allowed to open the device.',
      ready: 'Ready (experimental). Choose NPU under Advanced when you load a model.',
      installing: 'Installing the runtime ...',
    };

    function npuMegabytes(bytes) { return `${(Number(bytes || 0) / 1e6).toFixed(1)}`; }

    /** The progress bar and its text for a setup run (`progress` is the "setup.progress" object of the API). */
    function npuProgressModel(progress) {
      const p = progress || {};
      const labels = { downloading: 'Downloading', verifying: 'Checking the download', unpacking: 'Unpacking' };
      if (!p.busy) return { visible: false, percent: 0, text: '' };
      const total = Number(p.total_bytes || 0);
      const done = Number(p.done_bytes || 0);
      const percent = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
      const detail = total > 0 ? ` (${npuMegabytes(done)} of ${npuMegabytes(total)} MB)` : '';
      return { visible: true, percent, text: `${labels[p.phase] || 'Working'}${detail}` };
    }

    /**
     * Everything the card shows, from the API state: summary text, one row per NPU found, which buttons exist, the progress bar, the error and
     * the lists of formats. No DOM here.
     */
    function npuCardModel(npu) {
      if (!npu) return { summary: 'Checking for an NPU ...', tone: 'muted', hardware: [], button: null, cancel: false, remove: false, progress: { visible: false, percent: 0, text: '' }, error: null, formats: [], explanation: NPU_EXPLANATION };
      const setup = npu.setup || {};
      const progress = setup.progress || {};
      const runtime = npu.runtime || {};
      let summary = NPU_SUMMARIES[npu.state] || npu.state_text || '';
      if (npu.state === 'runtime_missing') {
        const mb = setup.download_bytes ? ` (a download of about ${Math.round(setup.download_bytes / 1e6)} MB)` : '';
        summary = `An Intel NPU was found. The runtime that drives it (OpenVINO ${setup.pinned_version || ''}${mb}) is not installed yet.`;
      }
      if (npu.state === 'ready' && runtime.version) summary = `Ready (experimental), OpenVINO ${runtime.version}. Choose NPU under Advanced when you load a model.`;
      const hardware = (npu.hardware || []).map((h) => ({
        label: `${h.name} (${NPU_VENDOR_NAMES[h.vendor] || NPU_VENDOR_NAMES.other})`,
        badge: h.supported ? 'Intel NPU' : 'detected, not supported yet',
        supported: !!h.supported,
        problem: h.problem || null,
      }));
      const busy = !!progress.busy;
      const retry = progress.phase === 'failed' || progress.phase === 'cancelled';
      const canInstall = !busy && (npu.state === 'runtime_missing');
      const megabytes = setup.download_bytes ? ` (about ${Math.round(setup.download_bytes / 1e6)} MB)` : '';
      const button = canInstall ? { action: 'install', label: retry ? 'Try again' : `Install the runtime${megabytes}` } : null;
      let error = null;
      if (progress.phase === 'failed' && progress.error) error = progress.error;
      else if (runtime.error && npu.state !== 'ready') error = runtime.error;
      else if (npu.last_error) error = npu.last_error;
      const tone = npu.state === 'ready' ? 'good' : (npu.state === 'no_device' || error ? 'warn' : 'muted');
      const q = npu.quantizations || {};
      const formats = [
        { title: 'Exact (same values as on the processor)', items: q.exact || [] },
        { title: 'Converted to 4 bit', items: q.converted_int4 || [] },
        { title: 'Converted to 8 bit', items: q.converted_int8 || [] },
        { title: 'Half precision', items: q.float16 || [] },
      ].filter((f) => f.items.length);
      return {
        summary,
        tone,
        hardware,
        button,
        cancel: busy,
        remove: !busy && !!runtime.managed,
        progress: npuProgressModel(progress),
        error,
        formats,
        explanation: NPU_EXPLANATION,
      };
    }

    /**
     * The entries of the device choice in the load options: automatic, the processor, every graphics device that reports memory, and the NPU
     * when it is ready. `current` (the device value the server runs with or the saved one) is kept in the list even when it is unknown here.
     */
    function deviceChoices(hw, npu, current) {
      const out = [
        { value: 'auto', label: 'Automatic (recommended)', warning: null },
        { value: 'cpu', label: 'Processor only', warning: null },
      ];
      const names = { cuda: 'NVIDIA CUDA', vulkan: 'Vulkan', hip: 'AMD ROCm', metal: 'Metal', dx12: 'DirectX 12', sycl: 'Intel SYCL' };
      for (const d of (hw && hw.devices) || []) {
        if (!d || d.backend === 'cpu' || d.backend === 'npu' || d.available === false || !(Number(d.total_memory_bytes) > 0)) continue;
        if (out.some((o) => o.value === d.id)) continue;
        out.push({ value: d.id, label: `${d.name} (${names[String(d.backend).toLowerCase()] || d.backend})`, warning: null });
      }
      if (npu && npu.state === 'ready') out.push({ value: 'npu', label: 'NPU (experimental)', warning: npu.warning || NPU_WARNING });
      const wanted = String(current || '').toLowerCase();
      if (wanted && !out.some((o) => o.value === wanted)) {
        out.push({ value: wanted, label: `${wanted} (current)`, warning: wanted.startsWith('npu') ? NPU_WARNING : null });
      }
      return out;
    }
    // @pure-end

    // ---- the card ------------------------------------------------------------------------------------------------------------
    let npuTimer = null;

    async function refreshNpu() {
      try {
        const res = await apiFetch('/api/studio/npu');
        if (!res.ok) return;
        state.npu = await res.json();
      } catch { return; }
      renderNpuCard();
      const busy = !!state.npu?.setup?.progress?.busy;
      if (busy && !npuTimer) npuTimer = setInterval(refreshNpu, 1000);
      if (!busy && npuTimer) { clearInterval(npuTimer); npuTimer = null; }
    }

    function renderNpuCard() {
      if (!$('npuCard')) return;
      const m = npuCardModel(state.npu);
      $('npuSummary').textContent = m.summary;
      $('npuSummary').style.color = m.tone === 'good' ? 'var(--primary)' : '';
      $('npuHardwareList').innerHTML = m.hardware.map((h) => `
        <div class="device-card" style="margin-bottom:6px;">
          <div style="font-weight:600;color:var(--foreground);">${escapeHtml(h.label)}</div>
          <div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;">
            <span class="quant-chip">${escapeHtml(h.badge)}</span>
            ${h.problem ? `<span class="quant-chip" style="color:var(--destructive,#c0392b);">${escapeHtml(h.problem)}</span>` : ''}
          </div>
        </div>`).join('');
      $('npuProgressWrap').style.display = m.progress.visible ? '' : 'none';
      $('npuProgress').style.width = `${m.progress.percent}%`;
      $('npuProgressText').textContent = m.progress.text;
      $('npuProgressText').style.display = m.progress.visible ? '' : 'none';
      $('npuError').textContent = m.error || '';
      $('npuError').style.display = m.error ? '' : 'none';
      $('npuSetupBtn').style.display = m.button ? '' : 'none';
      if (m.button) $('npuSetupBtn').textContent = m.button.label;
      $('npuCancelBtn').style.display = m.cancel ? '' : 'none';
      $('npuRemoveBtn').style.display = m.remove ? '' : 'none';
      $('npuExplain').innerHTML = m.explanation.map((t) => `<p class="app-item-desc">${escapeHtml(t)}</p>`).join('');
      $('npuFormats').innerHTML = m.formats.map((f) => `
        <div class="app-item-desc" style="margin-top:6px;"><b>${escapeHtml(f.title)}</b></div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px;">${f.items.map((i) => `<span class="quant-chip">${escapeHtml(i)}</span>`).join('')}</div>`).join('');
    }

    async function npuSetupAction(method, body) {
      try {
        const res = await apiFetch('/api/studio/npu/setup', { method, body });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) { toast(data.error?.message || 'The NPU action failed.'); return; }
        state.npu = data;
        renderNpuCard();
      } catch { toast('The NPU action failed.'); }
      refreshNpu();
    }

    $('npuSetupBtn')?.addEventListener('click', () => npuSetupAction('POST', { action: 'install' }));
    $('npuCancelBtn')?.addEventListener('click', () => npuSetupAction('POST', { action: 'cancel' }));
    $('npuRemoveBtn')?.addEventListener('click', () => {
      if (confirm('Remove the OpenVINO runtime? The NPU cannot be used until it is installed again.')) npuSetupAction('DELETE');
    });

    // The card refreshes when the Hardware pane is shown (the pane gets the class "active").
    if ($('spane-hardware')) {
      new MutationObserver(() => { if ($('spane-hardware').classList.contains('active')) refreshNpu(); })
        .observe($('spane-hardware'), { attributes: true, attributeFilter: ['class'] });
    }

    // ---- the device choice of the load options -------------------------------------------------------------------------------
    /** Fills the "Run on" choice of the load dialog; `saved` is the device saved for this model. */
    async function fillDeviceChoices(saved) {
      const select = $('optDevice');
      if (!select) return;
      if (!state.hardware) { try { state.hardware = await (await apiFetch('/api/studio/hardware')).json(); } catch { /* the list is shorter */ } }
      try { state.npu = await (await apiFetch('/api/studio/npu')).json(); } catch { /* no NPU entry */ }
      const current = saved || state.status?.model?.load_options?.device || 'auto';
      const choices = deviceChoices(state.hardware, state.npu, current);
      select.innerHTML = choices.map((c) => `<option value="${escapeHtml(c.value)}">${escapeHtml(c.label)}</option>`).join('');
      select.value = choices.some((c) => c.value === String(current).toLowerCase()) ? String(current).toLowerCase() : 'auto';
      showDeviceWarning();
    }

    function showDeviceWarning() {
      const select = $('optDevice');
      const warning = $('optDeviceWarning');
      if (!select || !warning) return;
      const isNpu = select.value === 'npu' || select.value.startsWith('npu:');
      warning.textContent = isNpu ? (state.npu?.warning || NPU_WARNING) : '';
      warning.style.display = isNpu ? '' : 'none';
    }
    $('optDevice')?.addEventListener('change', showDeviceWarning);
