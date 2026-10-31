    // --- Start screen: what the engine is doing right now ---
    const heroTrace = { samples: [], lastTokens: null, lastUptime: null, size: 60 };

    // @pure-begin
    /** The name of a model file for a person: the file name without folders and `.gguf` (an Ollama model keeps its `name:tag`). */
    function modelFileTitle(text) {
      const last = String(text || '').split(/[\\/]/).pop();
      return last.replace(/\.gguf$/i, '');
    }

    /**
     * What the start screen offers when no model is loaded. Every model that is on this computer counts: downloads, files the user added,
     * and files found in the usual places (Downloads, LM Studio, Ollama ...). `mode` is "load" when there is at least one, otherwise
     * "download" with the recommended model. The one to load first is the model used last, then the recommended one when it is on disk,
     * then the largest that fits into the free memory. A recommended model that is already on disk is never offered as a download.
     * Returns { mode, primary, others, download }.
     */
    function startPlan(hub, lastModel) {
      const h = hub || {};
      const free = Number(h.available_memory_bytes || 0);
      const catalog = h.catalog || [];
      const rec = catalog.find((c) => c.id === (h.recommended || [])[0]) || null;
      const recRef = rec && rec.installed ? rec.installed.ref : '';
      const last = String(lastModel || '');
      const lastTitle = modelFileTitle(last).toLowerCase();
      const seen = new Set();
      const all = [];
      const add = (ref, name, size, where) => {
        const key = String(ref || '').toLowerCase();
        if (!ref || seen.has(key)) return;
        seen.add(key);
        all.push({ ref, name, title: modelFileTitle(name), size: Number(size || 0), where });
      };
      for (const c of h.cached || []) add(c.name, c.path || c.name, c.size_bytes, 'downloaded');
      for (const l of h.local || []) add(l.path, l.name || l.path, l.size_bytes, 'added');
      for (const f of h.found || []) add(f.path, f.name, f.size_bytes, f.source || 'found');
      const lastEntry = catalog.find((c) => c.id === last);
      const lastRef = lastEntry && lastEntry.installed ? lastEntry.installed.ref : '';
      const isLast = (m) => !!last && (m.ref === last || m.ref === lastRef || m.title.toLowerCase() === lastTitle);
      const fits = (m) => !free || !m.size || m.size * 1.15 < free;
      const rank = (m) => (isLast(m) ? 0 : m.ref === recRef ? 1 : fits(m) ? 2 : 3);
      all.sort((x, y) => rank(x) - rank(y) || y.size - x.size || x.title.localeCompare(y.title));
      if (all.length) {
        const primary = Object.assign({}, all[0], { last: isLast(all[0]), fits: fits(all[0]) });
        return { mode: 'load', primary, others: all.slice(1), download: null };
      }
      return { mode: 'download', primary: null, others: [], download: rec };
    }
    // @pure-end

    let heroStartFor = '';
    let heroHubAsked = false;
    function sizeText(bytes) {
      const b = Number(bytes || 0);
      return b ? (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`) : '';
    }

    /** The empty start screen: load a model that is already here (one button), or download the recommended one. */
    function renderHeroStart() {
      const box = document.getElementById('heroStart');
      if (!box) return;
      const hub = state.modelsHub;
      if (!hub) {
        if (!heroHubAsked) { heroHubAsked = true; refreshHub(); }
        box.style.display = 'none';
        return;
      }
      const plan = startPlan(hub, state.status?.last_model);
      const key = JSON.stringify([plan.mode, plan.primary && plan.primary.ref, plan.others.map((o) => o.ref), plan.download && plan.download.id, plan.download && plan.download.default_quant]);
      if (heroStartFor === key) return; // the hero is redrawn on every status poll
      heroStartFor = key;
      const chooseFile = '<button class="btn btn-sm btn-ghost" id="btnHeroPick">Choose a file ...</button>';
      const browse = '<button class="btn btn-sm btn-ghost" id="btnHeroBrowse">Browse models</button>';
      box.style.display = '';
      if (plan.mode === 'load') {
        const p = plan.primary;
        box.innerHTML =
          '<div class="hero-start-main"><button class="btn btn-primary" id="btnHeroLoad" title="' + escapeHtml(p.ref) + '">Load ' + escapeHtml(p.title) + '</button>' +
          '<span class="app-item-desc mono hero-start-meta">' + escapeHtml([sizeText(p.size), p.last ? 'used last' : p.where].filter(Boolean).join(', ')) + (p.fits ? '' : ', may not fit into free memory') + '</span></div>' +
          (plan.others.length ? '<div class="hero-start-list">' + plan.others.slice(0, 4).map((o, i) =>
            '<div class="hero-start-row"><span class="mono hero-start-name" title="' + escapeHtml(o.ref) + '">' + escapeHtml(o.title) + '</span><span class="app-item-desc mono">' + escapeHtml(sizeText(o.size)) + '</span>' +
            '<button class="btn btn-xs btn-ghost" data-hero-load="' + i + '">Load</button></div>').join('') +
            (plan.others.length > 4 ? '<div class="app-item-desc">and ' + (plan.others.length - 4) + ' more on the Models page</div>' : '') + '</div>' : '') +
          '<div class="hero-start-more">' + chooseFile + browse + '</div>';
        box.querySelector('#btnHeroLoad').addEventListener('click', () => window.fllmLoadModelNow(p.ref, p.title));
        box.querySelectorAll('[data-hero-load]').forEach((btn) => btn.addEventListener('click', () => { const o = plan.others[Number(btn.dataset.heroLoad)]; window.fllmLoadModelNow(o.ref, o.title); }));
      } else if (plan.download) {
        const rec = plan.download;
        const quant = rec.default_quant || '';
        const variant = (rec.variants || []).find((v) => v.quant === quant) || (rec.variants || [])[0] || null;
        const bytes = Number((variant && variant.estimated_bytes) || rec.estimated_bytes || 0);
        const label = [rec.display_name || rec.id, quant, sizeText(bytes)].filter(Boolean).join(', ');
        const fitsMem = variant && variant.fits_memory !== undefined ? variant.fits_memory : rec.fits;
        box.innerHTML =
          '<div class="hero-start-main"><button class="btn btn-primary" id="btnHeroDownload">Download and start</button>' +
          '<span class="app-item-desc mono hero-start-meta">' + escapeHtml(label) + (fitsMem === false ? ', may not fit into free memory' : '') + '</span></div>' +
          '<div class="hero-start-more">' + chooseFile + browse + '</div>';
        const button = box.querySelector('#btnHeroDownload');
        button.addEventListener('click', async () => {
          button.disabled = true;
          button.textContent = 'Downloading ...';
          try {
            const ok = await window.fllmDownloadModel(rec.id, quant, true);
            if (!ok) { button.disabled = false; button.textContent = 'Download and start'; }
          } catch (e) {
            button.disabled = false;
            button.textContent = 'Download and start';
            if (typeof toast === 'function') toast((e && e.message) || 'The download failed.');
          }
        });
      } else {
        box.innerHTML = '<div class="hero-start-more">' + chooseFile + browse + '</div>';
      }
      const pick = box.querySelector('#btnHeroPick');
      if (pick) pick.addEventListener('click', () => window.fllmPickModelFile());
      const br = box.querySelector('#btnHeroBrowse');
      if (br) br.addEventListener('click', () => switchSurface('hub'));
    }

    function updateHero(data) {
      const m = data.model || {};
      const mt = data.metrics || {};
      const nf = (value, digits = 1) => value.toLocaleString('en-US', { maximumFractionDigits: digits });
      const params = Number(m.parameters || 0);
      const size = params >= 1e9 ? `${nf(params / 1e9)} billion parameters` : params > 0 ? `${nf(params / 1e6, 0)} million parameters` : '';
      const facts = [size, m.quantization, m.max_context ? `${nf(m.max_context, 0)} tokens of context` : ''].filter(Boolean);
      if (data.setup) {
        $('welcomeHeading').textContent = 'No model loaded';
        const plan = startPlan(state.modelsHub, data.last_model);
        $('welcomeSub').textContent = !state.modelsHub ? 'Looking for models on this computer ...'
          : plan.mode === 'load' ? 'A model on this computer is ready. Nothing to download.'
          : 'Download one, or choose a model file you already have.';
        renderHeroStart();
      } else {
        $('welcomeHeading').textContent = state.selectedModel || m.name || 'Ready';
        $('welcomeSub').textContent = `Running locally on this computer${facts.length ? ': ' + facts.join(', ') : ''}.`;
      }

      // tokens per second from the counters of the server (its own clock, so a slow poll does not skew it)
      const tokens = Number(mt.generated_tokens_total || 0);
      const uptime = Number(mt.uptime_secs || 0);
      let rate = 0;
      if (heroTrace.lastTokens !== null && uptime > heroTrace.lastUptime && tokens >= heroTrace.lastTokens) {
        rate = (tokens - heroTrace.lastTokens) / (uptime - heroTrace.lastUptime);
      }
      heroTrace.lastTokens = tokens;
      heroTrace.lastUptime = uptime;
      heroTrace.samples.push(rate);
      if (heroTrace.samples.length > heroTrace.size) heroTrace.samples.shift();

      const top = Math.max(10, ...heroTrace.samples);
      const step = 600 / (heroTrace.size - 1);
      const offset = heroTrace.size - heroTrace.samples.length;
      const points = heroTrace.samples.map((v, i) => `${((offset + i) * step).toFixed(1)},${(63 - Math.min(1, v / top) * 58).toFixed(1)}`);
      const trace = `0,63 ${points.join(' ')}`;
      $('heroTraceLine').setAttribute('points', trace);
      $('headerTraceLine').setAttribute('points', trace);
      const busy = Number(mt.slots_running || 0) > 0;
      $('heroTrace').classList.toggle('is-busy', busy);
      $('heroTps').textContent = busy ? nf(rate, 1) : '0';
      $('headerTps').textContent = busy ? nf(rate, 0) : '0';
      $('headerTrace').classList.toggle('is-busy', busy);
      $('heroTpsLabel').textContent = busy ? 'tokens per second, the model is answering' : 'tokens per second, the model is waiting for a question';
    }

    // --- Studio Status Sync ---
    let hardwareFetching = false; // one request for the hardware report from the footer at a time
    async function refreshStudioStatus() {
      try {
        const res = await apiFetch('/api/studio/status');
        if (!res.ok) return;
        const data = await res.json();
        state.status = data;
        renderThinkingChip(); // thinking chip follows the loaded model (sampler.js)
        const wasSetup = state.setup;
        state.setup = Boolean(data.setup);
        $('setupBanner').style.display = state.setup ? '' : 'none';
        $('setupError').style.display = data.setup_error ? '' : 'none';
        if (data.setup_error) $('setupError').textContent = `The model could not be loaded: ${data.setup_error}`;
        $('loadedModelCard').style.display = state.setup ? 'none' : '';
        // without a model the page opens on the model list, unless the last model loads with the first message
        if (state.setup && wasSetup === undefined && !data.last_model) switchSurface('hub');
        if (data.last_model && $('setupText')) {
          $('setupText').textContent = 'No model is loaded. The last model loads when you send your first message, or load another one below.';
        }
        updateHero(data);
        if (typeof updateContextMeter === 'function') updateContextMeter(); // the window of a model that was just loaded

        const m = data.model || {};
        if (!state.selectedModel) state.selectedModel = m.name || '';
        $('activeModelLabel').textContent = state.setup ? 'Choose a model' : state.selectedModel;
        $('activeQuantChip').textContent = m.quantization || 'GGUF';
        $('activeQuantChip').style.display = state.setup ? 'none' : '';
        // where the model runs: the device it was started on, or the one the automatic choice picks (npu.js)
        const where = runningDevice(m.load_options?.device, state.hardware, data.config?.active_cpu_tier);
        $('footerSimdLabel').textContent = state.setup ? 'No model loaded' : where.label;
        $('footerSimdLabel').title = state.setup ? 'Pick a model on the Models page to start.' : where.title;
        if (!state.hardware && !state.hardwareError && !hardwareFetching) {
          hardwareFetching = true; // the footer shows the processor until the report arrives, then the device
          apiFetch('/api/studio/hardware').then((r) => (r.ok ? r.json() : null)).then((hw) => { if (hw) { state.hardware = hw; refreshStudioStatus(); } }).catch(() => {}).finally(() => { hardwareFetching = false; });
        }
        // a mixture-of-experts model whose experts are read from the file: how much of them is cached and how much is read per token
        const es = m.expert_streaming;
        const expertNote = es ? ` · experts: ${es.mode}${es.streaming ? `, ${Math.round((es.hit_ratio || 0) * 100)}% cached, ${Math.round((es.read_bytes_per_token || 0) / 1048576)} MB read/token` : ''}` : '';
        $('footerSlotsLabel').textContent = `${data.metrics?.slots_running || 0}/${data.metrics?.slots_total || 8} busy${expertNote}`;
        $('settingsVersionBadge').textContent = `fllm-rs ${data.version || '0.1.2'}`;
        $('genBuildInfoText').textContent = data.build_info || '';
        $('serverEndpointUrlText').textContent = `${window.location.origin}/v1`;

        const allNames = [m.name, ...(m.aliases || [])].filter(Boolean);
        $('modelDropdownItems').innerHTML = allNames.map((name) => `
          <button class="nav-item ${name === state.selectedModel ? 'active' : ''}" data-pick-model="${escapeHtml(name)}">
            <svg viewBox="0 0 24 24" class="icon icon-sm" style="color:var(--primary);"><circle cx="12" cy="12" r="5"/></svg>
            <span class="mono">${escapeHtml(name)}</span>
            <span class="quant-chip" style="margin-left:auto;">${escapeHtml(m.parameter_size || '')}</span>
          </button>
        `).join('');

        // Provider facts in Settings > fllm.rs
        $('providerModelFacts').innerHTML = `
          <div class="app-card-item"><div class="app-item-info"><div class="app-item-title">Model</div><div class="app-item-desc mono">${escapeHtml(m.arch)} · ${escapeHtml(m.parameter_size)} (${Number(m.parameters || 0).toLocaleString()} parameters) · ${formatBytes(m.size_bytes)}</div></div><span class="quant-chip">${escapeHtml(m.quantization)}</span></div>
          <div class="app-card-item"><div class="app-item-info"><div class="app-item-title">Conversation length</div><div class="app-item-desc mono">${m.max_context} tokens now (the model was trained for ${m.trained_context}) · ${m.vocab_size} known word pieces</div></div></div>
        `;
        $('providerAliasesList').innerHTML = (m.aliases || []).map((a) => `
          <span class="backend-chip">${escapeHtml(a)} <button data-del-alias="${escapeHtml(a)}" style="color:var(--destructive);margin-left:4px;">✕</button></span>
        `).join('');
        if (window.fllmPickerHook) window.fllmPickerHook(); // providers hook: models of other providers in the picker
        $('preProviderTemplate').textContent = m.chat_template || '';

        if (!state._samplerLoaded) {
          state._samplerLoaded = true;
          setSamplerUI(data.config);
        }
        $('chkCpuRepack').checked = Boolean(data.config?.cpu_repack);
        $('selSimdTierCap').value = data.config?.cpu_tier_cap || 'auto';
        updateCliAndSdkSnippets();
        if (!state._routeSampleLoaded) {
          // the route tester starts with the name of the model that is really loaded
          state._routeSampleLoaded = true;
          updateRouteTesterSample();
        }
      } catch {}
    }

    // --- API Keys ---
    async function refreshApiKeys() {
      if (state.setup) {
        // without a model the server has no key routes (the keys belong to the running server)
        $('apiKeysTableContainer').innerHTML = '<div class="app-item-desc">Keys are made by the running server. Load a model first (Models page), then create a key here.</div>';
        return;
      }
      try {
        const res = await apiFetch('/api/studio/keys');
        if (!res.ok) return;
        state.keysData = await res.json();
        const rl = state.keysData.rate_limit || {};
        $('inpGlobalRpm').value = rl.requests_per_minute || '';
        $('inpGlobalTpm').value = rl.tokens_per_minute || '';
        const keys = state.keysData.keys || [];
        if (!keys.length) {
          $('apiKeysTableContainer').innerHTML = `<div class="app-item-desc">No API key is set up. The server is open, and it accepts changes only from this computer. Create a key above to require a Bearer token for requests.</div>`;
          return;
        }
        $('apiKeysTableContainer').innerHTML = `
          <table class="app-table">
            <thead><tr><th>ID</th><th>Name</th><th>Key</th><th>Limit</th><th>Requests</th><th>Tokens</th><th>Refused</th><th></th></tr></thead>
            <tbody>
              ${keys.map((k) => `
                <tr>
                  <td class="mono">${escapeHtml(k.id)}</td>
                  <td><strong>${escapeHtml(k.label)}</strong>${k.admin ? ' <span class="quant-chip">admin</span>' : ''}</td>
                  <td class="mono">${escapeHtml(k.masked)}</td>
                  <td class="mono">${k.rpm_limit ? k.rpm_limit + ' per minute' : 'Default'}</td>
                  <td class="mono tabular">${k.requests_accepted}</td>
                  <td class="mono tabular">${k.tokens_used}</td>
                  <td class="mono tabular">${k.rejected}</td>
                  <td><button class="btn btn-xs btn-destructive" data-revoke-key="${escapeHtml(k.id)}">Revoke</button></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `;
      } catch {}
    }

    // --- Hardware & SIMD ---
    async function refreshHardware(withBench = false) {
      try {
        const [hwRes, slotsRes] = await Promise.all([
          apiFetch(`/api/studio/hardware${withBench ? '?benchmark=true' : ''}`),
          apiFetch('/slots'),
        ]);
        if (hwRes.ok) {
          const hw = await hwRes.json();
          state.hardware = hw;
          const devs = hw.devices || [];
          const totalB = Number(devs[0]?.total_memory_bytes || 0);
          const freeB = Number(hw.host_free_memory_bytes || devs[0]?.free_memory_bytes || 0);
          const usedPct = totalB > 0 ? ((totalB - freeB) / totalB) * 100 : 0;
          const tiers = hw.supported_cpu_tiers || [];
          $('hwRamDesc').textContent = (hw.measured_cpu_bandwidth_gbs
            ? `Instruction set: ${hw.active_cpu_tier}. Measured memory bandwidth: ${Number(hw.measured_cpu_bandwidth_gbs).toFixed(1)} GB/s.`
            : `Instruction set: ${hw.active_cpu_tier} (available: ${tiers.join(', ')}).`) + ` ${hw.threads} threads.`;
          $('hwRamText').textContent = `${formatBytes(freeB)} free of ${formatBytes(totalB)}`;
          $('hwRamProgress').style.width = `${Math.min(100, Math.max(2, usedPct)).toFixed(0)}%`;
          const sel = $('selSimdTierCap');
          const wanted = ['auto', ...tiers];
          if (sel.options.length !== wanted.length || [...sel.options].some((o, i) => o.value !== wanted[i])) {
            const cur = state.status?.config?.cpu_tier_cap || 'auto';
            sel.innerHTML = wanted.map((t) => `<option value="${t}">${t === 'auto' ? 'Auto (highest supported)' : t}</option>`).join('');
            sel.value = wanted.includes(cur) ? cur : 'auto';
          }

          // a device that reports no memory cannot hold a model: it stays out of the list (the details for a bug report still name it).
          // The same chip is listed once per driver interface: it is one card, with the interface in use marked.
          const loadDev = String(state.status?.model?.load_options?.device || 'auto').toLowerCase();
          const target = loadDev === 'auto' ? hw.auto_choice : loadDev;
          const usable = mergeDevices(devs.filter((d) => d.backend === 'cpu' || Number(d.total_memory_bytes) > 0), target);
          const chosen = usable.find((d) => (d.members || []).some((m) => m.id === hw.auto_choice));
          $('hwChoiceText').textContent = hw.split
            ? `The model is split over ${hw.split.stages.length} stage(s) (tensor-parallel ${hw.split.tensor_parallel} x pipeline-parallel ${hw.split.pipeline_parallel}): ${hw.split.stages.map((s) => `${(s.devices || []).join(' + ')}${s.layers ? ` layers ${s.layers[0]}-${s.layers[1] - 1}` : ''}`).join(' | ')}.`
            : chosen
            ? `With the automatic choice fLLM runs models on: ${chosen.name} (${chosen.backend === 'cpu' ? 'the processor' : deviceBackendLabel(chosen)}). The processor is the fallback.`
            : 'The processor runs the models: no usable graphics device was found.';
          $('hwDevicesGrid').innerHTML = usable.map((d) => {
            const role = deviceRole(d, loadDev, hw.auto_choice, !state.setup);
            const mem = deviceMemory(d);
            return `
            <div class="device-card${role === 'running' ? ' is-running' : ''}">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
                <div>
                  <div style="font-weight:600;color:var(--foreground);">${escapeHtml(d.name)}</div>
                  <div style="margin-top:6px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
                    ${role === 'running' ? `<span class="quant-chip role-chip">runs the model${loadDev === 'auto' ? ' (automatic choice)' : ''}</span>` : (role === 'auto' ? '<span class="quant-chip role-chip">automatic choice</span>' : '')}
                    ${d.split ? `<span class="quant-chip" style="color:var(--primary);">stage ${Number(d.split.stage) + 1}: ${d.split.layers ? `layers ${d.split.layers[0]}-${d.split.layers[1] - 1}` : 'layers'}${d.split.share != null ? ` (${(Number(d.split.share) * 100).toFixed(0)} % of the model)` : ''}</span>` : ''}
                    <span class="backend-chip"><span class="backend-chip-dot"></span>${escapeHtml(deviceBackendLabel(d))}</span>
                    ${d.description ? `<span class="quant-chip">${escapeHtml(String(d.description).replace(/\bthreads\b/, 'threads'))}</span>` : ''}
                    ${d.unified_memory ? '<span class="quant-chip">shares system memory</span>' : ''}
                  </div>
                  ${(d.others || []).length ? `<details class="device-notes"><summary>Also reachable through ${(d.others || []).length === 1 ? 'another backend' : `${d.others.length} other backends`}</summary><div class="app-item-desc">${d.others.map((o) => `${escapeHtml(o.label)} (${escapeHtml(o.id)})`).join(', ')}. The same chip, listed once per driver interface. The card shows the one the model runs on, or the one fLLM prefers.</div></details>` : ''}
                  ${d.notes ? `<details class="device-notes"><summary>Technical notes</summary><div class="app-item-desc">${escapeHtml(d.notes)}</div></details>` : ''}
                </div>
                <span class="mono tabular" style="font-size:12px;white-space:nowrap;">${mem.known ? `${formatBytes(d.free_memory_bytes)} free of ${formatBytes(d.total_memory_bytes)}` : 'memory not reported'}</span>
              </div>
              ${mem.known ? `<div class="mem-bar is-${mem.level}" role="img" aria-label="${mem.pct}% of the memory is in use"><i style="width:${Math.max(2, mem.pct)}%"></i></div>` : ''}
            </div>
          `; }).join('');
        }
        if (slotsRes.ok) {
          const sData = await slotsRes.json();
          const slots = Array.isArray(sData) ? sData : (sData.slots || []);
          $('hwSlotsTableWrap').innerHTML = `
            <table class="app-table">
              <thead><tr><th>Slot</th><th>State</th><th>Max Context</th><th>Task</th></tr></thead>
              <tbody>
                ${slots.map((s) => `
                  <tr>
                    <td class="mono">#${s.id}</td>
                    <td><span class="quant-chip">${escapeHtml(s.state)}</span></td>
                    <td class="mono">${s.n_ctx} tok</td>
                    <td class="mono">${s.next_token !== null && s.next_token !== undefined ? "Running" : "Idle"}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          `;
        }
      } catch {}
    }

    // --- Tokenizer & Jinja ---
    async function runTokenize() {
      const content = $('tokTextInput').value;
      const res = await apiFetch('/tokenize', {
        method: 'POST',
        body: { content, add_special: $('chkTokSpecial').checked, with_pieces: true },
      });
      if (!res.ok) return;
      const data = await res.json();
      const tokens = data.tokens || [];
      $('tokStatsLabel').textContent = `${tokens.length} tokens · ${new Blob([content]).size} bytes`;
      const hues = [145, 200, 265, 25, 330, 175];
      $('tokVisualWall').innerHTML = tokens.map((tok, i) => {
        const id = typeof tok === 'object' ? tok.id : tok;
        const piece = typeof tok === 'object' ? tok.piece : String(tok);
        const h = hues[i % hues.length];
        return `<span class="tok-pill" style="background:hsla(${h},65%,45%,0.18);"><span>${escapeHtml(piece)}</span><small>#${id}</small></span>`;
      }).join('');
    }

    async function runJinjaTemplate() {
      try {
        const messages = JSON.parse($('jinjaMessagesInput').value);
        const res = await apiFetch('/apply-template', { method: 'POST', body: { messages } });
        const data = await res.json();
        $('jinjaOutputPre').textContent = data.prompt || JSON.stringify(data, null, 2);
      } catch (e) {
        $('jinjaOutputPre').textContent = 'Invalid JSON: ' + e.message;
      }
    }

    // --- Snippets & Route Tester ---
    function updateCliAndSdkSnippets() {
      const m = state.status?.model || {};
      const p = getSampler();
      const origin = window.location.origin;
      const model = m.name || 'fllm-studio-demo';
      const key = state.apiKey || 'fllm-sk-local';
      const aliases = [model, ...(m.aliases || [])].join(',');

      $('genCliPre').textContent = [
        `fllm-rs serve --model "${model}.gguf"`,
        `  --host 127.0.0.1 --port 8000`,
        `  --ctx ${m.max_context || 512} --max-seqs ${m.max_sequences || 8}`,
        `  --model-name "${aliases}"`,
      ].join(' \\\n');

      $('genTomlPre').textContent = `[server]\nmodel = "${model}.gguf"\nhost = "127.0.0.1"\nport = 8000\nctx = ${m.max_context || 512}\nmax_seqs = ${m.max_sequences || 8}\n`;

      const sdkMap = {
        claude: `export ANTHROPIC_BASE_URL="${origin}"\nexport ANTHROPIC_API_KEY="${key}"\n\n# Run Claude Code against your local fLLM Rust server:\nclaude --model ${model}`,
        python: `from openai import OpenAI\n\nclient = OpenAI(base_url="${origin}/v1", api_key="${key}")\nres = client.chat.completions.create(\n    model="${model}",\n    messages=[{"role": "user", "content": "Hello from fLLM!"}],\n    temperature=${p.temperature},\n)\nprint(res.choices[0].message.content)`,
        ts: `import OpenAI from 'openai';\n\nconst client = new OpenAI({ baseURL: '${origin}/v1', apiKey: '${key}' });\nconst res = await client.chat.completions.create({\n  model: '${model}',\n  messages: [{ role: 'user', content: 'Hello from fLLM!' }],\n});\nconsole.log(res.choices[0]?.message?.content);`,
        curl: `curl ${origin}/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer ${key}" \\\n  -d '{"model":"${model}","messages":[{"role":"user","content":"Hello!"}]}'`,
      };
      $('preSdkSnippet').textContent = sdkMap[state.sdkLang] || sdkMap.claude;
    }

    function updateRouteTesterSample() {
      const v = $('selRouteTest').value;
      const model = state.selectedModel || state.status?.model?.name || 'your-model';
      const map = {
        'POST:/v1/chat/completions': { model, messages: [{ role: 'user', content: 'Hello! Answer briefly.' }], max_tokens: 48 },
        'POST:/v1/messages': { model, max_tokens: 48, messages: [{ role: 'user', content: 'Hello via Anthropic API!' }] },
        'POST:/v1/responses': { model, input: 'Three facts about Rust.', max_output_tokens: 48 },
        'POST:/api/chat': { model, stream: false, messages: [{ role: 'user', content: 'Hello via Ollama API!' }] },
        'POST:/v1/embeddings': { model, input: ['Local AI in Rust', 'SIMD Kernels'] },
        'POST:/v1/rerank': { model, query: 'SIMD CPU acceleration', documents: ['AVX-512 accelerates dot products.', 'Apple pie recipe.'] },
      };
      $('txtRouteReq').value = map[v] ? JSON.stringify(map[v], null, 2) : '';
    }

    async function refreshPrometheus() {
      try {
        const res = await apiFetch('/metrics');
        $('prePrometheusMetrics').textContent = await res.text();
      } catch {}
    }

