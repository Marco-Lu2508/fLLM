    // ==========================================================================
    // Is the GPU doing the work? Two things in one place:
    //  * the amber notice "Part of this model runs on the CPU" (footer, Settings > Hardware, Models page), built from `device_fallbacks` of
    //    GET /api/studio/status (placement.rs);
    //  * the "Test this GPU" button of every device card: POST /api/studio/gpu-check runs a quantized matrix product on the device and on the
    //    CPU and answers with one line (gpu_check.rs). The pure parts are tested in node (tests/gpucheck.test.js).
    // ==========================================================================

    // @pure-begin
    /**
     * What to show for the `device_fallbacks` object of the status. `show` is false when everything runs where it should (or no GPU is
     * involved); `level` is 'cpu' (the whole model), 'mostly' (half or more) or 'partial'.
     */
    function fallbackNotice(df) {
      if (!df || !df.fallback) return { show: false };
      const mode = df.mode;
      const where = df.device ? `${df.device}${df.backend ? ` (${df.backend})` : ''}` : 'the graphics card';
      let title;
      if (mode === 'cpu') title = 'The whole model runs on the CPU';
      else if (mode === 'mostly_cpu') title = 'Weights in GPU memory, compute mostly on the CPU';
      else title = 'Part of this model runs on the CPU';
      const reasons = (df.reasons || []).map((r) => `${r.reason} (${r.operations})`);
      const ops = (df.operation_fallbacks || []).filter((o) => !o.by_design).map((o) => `${o.operation}: ${o.reason} (${o.calls} calls)`);
      const detail = [df.headline, ...(df.notices || [])].filter(Boolean).join(' ');
      const lines = [];
      if (df.whole_model_on_cpu) lines.push(`${where} is not used for this model.`);
      else if (df.operations_on_cpu > 0) lines.push(`${df.operations_on_cpu} of ${df.operations_total} operations (attention or feed-forward blocks of the layers) run on the CPU. ${df.layers_on_device} layers run completely on ${where}.`);
      for (const r of reasons) lines.push(`Why: ${r}`);
      for (const o of ops) lines.push(`Operation on the CPU: ${o}`);
      return {
        show: true,
        level: mode === 'cpu' ? 'cpu' : (mode === 'mostly_cpu' ? 'mostly' : 'partial'),
        title,
        short: mode === 'cpu' ? 'Model runs on the CPU' : 'Part of the model runs on the CPU',
        summary: df.summary || title,
        detail,
        lines,
        footerTitle: `${title}. ${detail || ''} Open Settings, Hardware for the details, or run "fllm-rs doctor --gpu".`.replace(/\s+/g, ' ').trim(),
      };
    }

    /** One result line of the GPU check: the text, and 'ok' / 'warn' / 'bad' for its colour. */
    function gpuVerdictLine(r) {
      const cls = r.status === 'ok' ? 'ok' : (r.status === 'slow' ? 'warn' : 'bad');
      return { text: `${r.device}: ${r.verdict}`, cls };
    }
    // @pure-end

    const gpuCheckResults = {}; // card id -> [result]
    const gpuCheckBusy = {};

    /** The footer chip, the Hardware notice and the line on the Models page follow the status. */
    function renderFallbackNotice(df) {
      const n = fallbackNotice(df);
      const foot = $('footerFallback');
      if (foot) {
        foot.style.display = n.show ? '' : 'none';
        if (n.show) { foot.textContent = n.short; foot.title = n.footerTitle; foot.dataset.level = n.level; }
      }
      const hw = $('hwFallbackNotice');
      if (hw) {
        hw.style.display = n.show ? '' : 'none';
        if (n.show) {
          $('hwFallbackTitle').textContent = n.title;
          $('hwFallbackText').textContent = n.summary;
          $('hwFallbackLines').innerHTML = n.lines.map((l) => `<div>${escapeHtml(l)}</div>`).join('');
          $('hwFallbackDetail').textContent = n.detail;
        }
      }
      const models = $('modelsFallbackNotice');
      if (models) {
        models.style.display = n.show ? '' : 'none';
        if (n.show) models.textContent = `${n.title}. ${n.lines[0] || ''} Details: Settings, Hardware.`;
      }
    }

    /** The "Test this GPU" part of a device card (its button and the stored result lines). */
    function gpuCheckHtml(d) {
      if (!d || d.backend === 'cpu' || d.backend === 'npu') return '';
      const busy = gpuCheckBusy[d.id];
      const results = gpuCheckResults[d.id] || [];
      const lines = results.map((r) => { const l = gpuVerdictLine(r); return `<div class="gpu-verdict is-${l.cls}">${escapeHtml(l.text)}</div>`; }).join('');
      return `<div style="margin-top:8px;display:flex;flex-direction:column;gap:4px;">
        <div><button class="btn btn-xs btn-ghost" data-gpu-check="${escapeHtml(d.id)}"${busy ? ' disabled' : ''}>${busy ? 'Testing ...' : 'Test this GPU'}</button></div>
        ${lines}</div>`;
    }

    async function runGpuCheck(id) {
      const card = ((state.hardware && state.hardware.devices) || []).find((d) => d.id === id);
      const members = mergeDevices(((state.hardware && state.hardware.devices) || []).filter((d) => d.backend !== 'cpu'), id).find((g) => g.id === id)?.members || [{ id }];
      gpuCheckBusy[id] = true;
      gpuCheckResults[id] = [];
      refreshHardware();
      try {
        for (const m of members) {
          const res = await apiFetch('/api/studio/gpu-check', { method: 'POST', body: { device: m.id } });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { gpuCheckResults[id].push({ device: m.id, status: 'error', verdict: (data.error && data.error.message) || 'The check could not run.' }); continue; }
          gpuCheckResults[id].push(...(data.results || []));
        }
      } catch (e) {
        gpuCheckResults[id].push({ device: card ? card.id : id, status: 'error', verdict: `The check could not run (${e.message || e}).` });
      } finally {
        gpuCheckBusy[id] = false;
        refreshHardware();
      }
    }

    $('hwDevicesGrid')?.addEventListener('click', (e) => {
      const b = e.target.closest('[data-gpu-check]');
      if (b) runGpuCheck(b.dataset.gpuCheck);
    });
    $('footerFallback')?.addEventListener('click', () => { switchSurface('settings', 'hardware'); });
