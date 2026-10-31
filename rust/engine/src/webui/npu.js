    // ==========================================================================
    // NPU: the card in Settings, Hardware (what was found, install or remove the runtime that drives it) and the device list of the
    // load options. The state comes from GET /api/studio/npu; the card is built from it by npuCardModel (pure, tested in node).
    // Honest status: measured on one Intel Core Ultra 5 235U (NPU 3720): the NPU makes reading long prompts faster than the processor alone
    // (about 1.5 to 3 times) but not faster than the integrated graphics; answers are written at the processor's speed. Experimental.
    // ==========================================================================

    // @pure-begin
    const NPU_WARNING = 'Experimental: the NPU only speeds up reading long prompts (the processor writes the answer). The integrated graphics are usually faster. A converted copy of the weights is kept in memory.';

    const NPU_EXPLANATION = [
      'fLLM uses an Intel NPU as a helper: the model runs on the processor, and the large matrix multiplications of a long prompt (64 tokens or more) run on the NPU. Writing the answer is not sped up and stays on the processor.',
      'On a Core Ultra 5 235U the NPU read prompts 1.5 to 3 times faster than the processor alone, and about as fast as the integrated graphics, which need no setup. The NPU is worth it when the graphics are busy or absent, or to keep the processor free. The first start compiles the NPU programs (about a second each, shown as "still compiling"; the processor serves those meanwhile); later starts take them from a cache.',
      'The NPU keeps a converted 8-bit copy of the weights it computes with, so a model needs about its own size again in memory (a budget limits it). Only ordinary (dense) models are accelerated; mixture-of-experts and other special models stay on the processor.',
      'It is never chosen automatically. To use it, pick NPU under Advanced when you load a model, or start fLLM with --device npu (or FLLM_NPU_HOST=1). If the NPU cannot take a model, fLLM runs it on the processor and says why here.',
    ];

    const NPU_VENDOR_NAMES = { intel: 'Intel', amd: 'AMD', qualcomm: 'Qualcomm', other: 'Unknown maker' };

    const NPU_SUMMARIES = {
      not_in_build: 'This build of fLLM has no NPU support.',
      no_npu: 'No NPU was found on this computer.',
      unsupported_hardware: 'An NPU was detected, but this kind of NPU is not supported yet.',
      setup_unavailable: 'An Intel NPU was found, but fLLM has no automatic setup for this system. Install the OpenVINO runtime yourself and fLLM will find it.',
      no_device: 'The OpenVINO runtime is installed but does not list an NPU. The NPU driver may be missing, or this user may not be allowed to open the device.',
      ready: 'Ready (experimental). Choose NPU under Advanced when you load a model; long prompts then run on it.',
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
      if (!npu) return { summary: 'Checking for an NPU ...', offload: '', tone: 'muted', hardware: [], button: null, cancel: false, remove: false, progress: { visible: false, percent: 0, text: '' }, error: null, formats: [], explanation: NPU_EXPLANATION };
      const setup = npu.setup || {};
      const progress = setup.progress || {};
      const runtime = npu.runtime || {};
      let summary = NPU_SUMMARIES[npu.state] || npu.state_text || '';
      if (npu.state === 'runtime_missing') {
        const mb = setup.download_bytes ? ` (a download of about ${Math.round(setup.download_bytes / 1e6)} MB)` : '';
        summary = `An Intel NPU was found. The runtime that drives it (OpenVINO ${setup.pinned_version || ''}${mb}) is not installed yet.`;
      }
      if (npu.state === 'ready' && runtime.version) summary = `Ready (experimental), OpenVINO ${runtime.version}. Choose NPU under Advanced when you load a model; long prompts then run on it.`;
      const dev = (npu.devices || [])[0];
      if (npu.state === 'ready' && dev && dev.full_name) summary += ` ${dev.full_name}${dev.architecture ? ` (platform ${dev.architecture})` : ''}.`;
      if (npu.state === 'unsupported_hardware' && (npu.hardware || []).some((h) => h.vendor === 'amd')) {
        summary = 'An AMD NPU was found. fLLM drives it with its own kernels (long prompts) or through FastFlowLM: see the AMD NPU card below.';
      }
      const hardware = (npu.hardware || []).map((h) => ({
        label: `${h.name} (${NPU_VENDOR_NAMES[h.vendor] || NPU_VENDOR_NAMES.other})`,
        badge: h.supported ? 'Intel NPU' : (h.vendor === 'amd' ? 'AMD NPU: native kernels or FastFlowLM, see below' : 'detected, not supported yet'),
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
        { title: 'Computed as 8 bit with one scale per output row (the default)', items: q.eight_bit || [] },
        { title: 'Half precision', items: q.float16 || [] },
        { title: 'Exact (same values as on the processor) with FLLM_NPU_WEIGHTS=exact', items: q.exact || [] },
        { title: 'Kept at 4 bit with FLLM_NPU_WEIGHTS=int4', items: q.converted_int4 || [] },
      ].filter((f) => f.items.length);
      // what the prompt offload is doing now (the status line from the engine; empty when the offload is not in use)
      const offload = typeof npu.offload_line === 'string' ? npu.offload_line : '';
      return {
        summary,
        offload,
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

    const DEVICE_BACKEND_NAMES = { cuda: 'NVIDIA CUDA', vulkan: 'Vulkan', hip: 'AMD ROCm', metal: 'Metal', dx12: 'DirectX 12', sycl: 'Intel SYCL' };

    /** The name of a graphics device with the trademark marks and spacing removed, to compare the entries of several drivers. */
    function deviceKey(d) {
      return String((d && d.name) || '').replace(/\((R|TM)\)/gi, '').replace(/\s+/g, ' ').trim().toLowerCase();
    }

    /** Lower is better: the vendor's own interface first, then the native graphics API, the generic (wgpu) entries last. */
    function deviceRank(d) {
      if (d.api === 'wgpu') return 3;
      return ['cuda', 'hip', 'metal', 'sycl'].includes(String(d.backend).toLowerCase()) ? 0 : 1;
    }

    /**
     * One entry per physical device. The same graphics chip is listed once per driver interface (Vulkan, generic Vulkan through wgpu,
     * DirectX 12, CUDA ...); entries with the same name and a similar memory size are one device. Two identical cards stay two devices:
     * an entry joins a group only when the group has no entry of the same backend and interface yet. The processor and NPUs are never merged.
     * `wantedId` (the device the model runs on, or the automatic choice) becomes the entry that stands for its group.
     * Each result is the primary entry plus `members` (id, backend, api of every entry of the device) and `others` (the members besides it).
     */
    function mergeDevices(devices, wantedId) {
      const wanted = String(wantedId || '').toLowerCase();
      const groups = [];
      const list = (devices || []).filter((d) => d && d.id);
      const mergeable = (d) => !['cpu', 'npu'].includes(String(d.backend).toLowerCase()) && deviceKey(d);
      for (const d of list) {
        const group = mergeable(d) && groups.find((g) => {
          if (!g.mergeable || deviceKey(g.items[0]) !== deviceKey(d)) return false;
          if (g.items.some((o) => o.backend === d.backend && o.api === d.api)) return false;
          const a = Number(g.items[0].total_memory_bytes) || 0;
          const b = Number(d.total_memory_bytes) || 0;
          return a === 0 || b === 0 || Math.min(a, b) / Math.max(a, b) >= 0.5;
        });
        if (group) group.items.push(d);
        else groups.push({ mergeable: !!mergeable(d), items: [d] });
      }
      return groups.map((g) => {
        const ordered = g.items.slice().sort((a, b) => deviceRank(a) - deviceRank(b));
        const primary = g.items.find((d) => String(d.id).toLowerCase() === wanted) || ordered[0];
        const members = g.items.map((d) => ({ id: d.id, backend: d.backend, api: d.api }));
        const others = ordered.filter((d) => d !== primary).map((d) => ({ id: d.id, backend: d.backend, api: d.api, label: deviceBackendLabel(d) }));
        return { ...primary, members, others };
      });
    }

    /** How a driver interface is named for a person: "Vulkan", "Vulkan (generic)", "NVIDIA CUDA". */
    function deviceBackendLabel(d) {
      const b = String((d && d.backend) || '').toLowerCase();
      if (b === 'cpu') return 'Processor';
      return `${DEVICE_BACKEND_NAMES[b] || b}${d && d.api === 'wgpu' ? ' (generic)' : ''}`;
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
      const usable = ((hw && hw.devices) || []).filter((d) => d && d.backend !== 'cpu' && d.backend !== 'npu' && d.available !== false && Number(d.total_memory_bytes) > 0);
      const groups = mergeDevices(usable, current);
      for (const d of groups) {
        if (out.some((o) => o.value === d.id)) continue;
        out.push({ value: d.id, label: `${d.name} (${deviceBackendLabel(d)})`, warning: null });
      }
      // the same chip through another vendor interface (for example CUDA and Vulkan) can be chosen too; the generic entries are not offered
      for (const d of groups) {
        for (const o of d.others) {
          if (o.api === 'wgpu' || out.some((x) => x.value === o.id)) continue;
          out.push({ value: o.id, label: `${d.name} (${deviceBackendLabel(o)}, other backend)`, warning: null, alternative: true });
        }
      }
      if (npu && npu.state === 'ready') out.push({ value: 'npu', label: 'NPU (experimental)', warning: npu.warning || NPU_WARNING });
      const wanted = String(current || '').toLowerCase();
      if (wanted && !out.some((o) => o.value === wanted)) {
        out.push({ value: wanted, label: `${wanted} (current)`, warning: wanted.startsWith('npu') ? NPU_WARNING : null });
      }
      return out;
    }

    /**
     * What the footer says about where the model runs. `loadDevice` is the --device value the model was started with ("auto" when
     * the server chose), `hw` the hardware report (its `auto_choice` is the device the automatic choice picks), `tier` the CPU kernel tier.
     * Returns { label, title, auto }. Without a hardware report an automatic choice can only be shown as the processor part.
     */
    function runningDevice(loadDevice, hw, tier) {
      const want = String(loadDevice || 'auto').toLowerCase();
      const cpu = { label: `CPU · ${tier || 'simd'}`, title: 'The model runs on the processor.', auto: want === 'auto' };
      if (want === 'cpu') return cpu;
      const names = { cuda: 'CUDA', vulkan: 'Vulkan', hip: 'ROCm', metal: 'Metal', dx12: 'DirectX 12', sycl: 'SYCL', npu: 'NPU' };
      const id = want === 'auto' ? String((hw && hw.auto_choice) || 'cpu').toLowerCase() : want;
      if (id === 'cpu') return cpu;
      const dev = ((hw && hw.devices) || []).find((d) => d && String(d.id).toLowerCase() === id);
      const backend = String((dev && dev.backend) || id.split(':')[0]);
      const plain = dev ? String(dev.name).replace(/\((R|TM)\)/gi, '').replace(/\s+/g, ' ').trim() : id;
      const label = `${plain} · ${names[backend] || backend}`;
      const title = want === 'auto' ? `${dev ? dev.name : id}: the automatic choice (${id}); layers that do not fit run on the processor.` : `${dev ? dev.name : id}: chosen in the load options (${id}).`;
      return { label, title, auto: want === 'auto' };
    }

    /** Memory use of a device for the bar on its card: { known, pct (0-100), level: ok | warn | high }. A device without a size has no bar. */
    function deviceMemory(d) {
      const total = Number(d && d.total_memory_bytes) || 0;
      const free = Number(d && d.free_memory_bytes) || 0;
      if (total <= 0) return { known: false, pct: 0, level: 'ok' };
      const pct = Math.round(Math.min(1, Math.max(0, (total - free) / total)) * 100);
      return { known: true, pct, level: pct >= 90 ? 'high' : (pct >= 75 ? 'warn' : 'ok') };
    }

    /**
     * What a device does for the loaded model: 'running' (the model runs on it), 'auto' (no model is loaded; the automatic choice would use
     * it) or ''. `loadDevice` is the --device value of the loaded model ("auto" when the server chose).
     */
    function deviceRole(d, loadDevice, autoChoice, modelLoaded) {
      const ids = (d && Array.isArray(d.members) && d.members.length ? d.members.map((m) => m.id) : [d && d.id]).map((x) => String(x || '').toLowerCase());
      const want = String(loadDevice || 'auto').toLowerCase();
      const target = want === 'auto' ? String(autoChoice || 'cpu').toLowerCase() : want;
      if (!ids.includes(target)) return '';
      return modelLoaded ? 'running' : 'auto';
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
      // poll while a setup runs, and while the prompt offload is working (its counters change)
      const busy = !!state.npu?.setup?.progress?.busy || !!state.npu?.offload?.active;
      if (busy && !npuTimer) npuTimer = setInterval(() => { if ($('spane-hardware')?.classList.contains('active')) refreshNpu(); }, 2000);
      if (!busy && npuTimer) { clearInterval(npuTimer); npuTimer = null; }
    }

    function renderNpuCard() {
      if (!$('npuCard')) return;
      const m = npuCardModel(state.npu);
      $('npuSummary').textContent = m.summary;
      $('npuSummary').style.color = m.tone === 'good' ? 'var(--primary)' : '';
      $('npuOffload').textContent = m.offload;
      $('npuOffload').style.display = m.offload ? '' : 'none';
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
    $('npuRemoveBtn')?.addEventListener('click', async () => {
      if (await confirmDialog('Remove the OpenVINO runtime? The NPU cannot be used until it is installed again.')) npuSetupAction('DELETE');
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
