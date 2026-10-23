    // --- Start screen: what the engine is doing right now ---
    const heroTrace = { samples: [], lastTokens: null, lastUptime: null, size: 60 };

    function updateHero(data) {
      const m = data.model || {};
      const mt = data.metrics || {};
      const nf = (value, digits = 1) => value.toLocaleString('de-DE', { maximumFractionDigits: digits });
      const params = Number(m.parameters || 0);
      const size = params >= 1e9 ? `${nf(params / 1e9)} Milliarden Parameter` : params > 0 ? `${nf(params / 1e6, 0)} Millionen Parameter` : '';
      const facts = [size, m.quantization, m.max_context ? `${nf(m.max_context, 0)} Token Kontext` : ''].filter(Boolean);
      $('welcomeHeading').textContent = state.selectedModel || m.name || 'Bereit';
      $('welcomeSub').textContent = `Running locally on this computer${facts.length ? ': ' + facts.join(', ') : ''}.`;

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
    async function refreshStudioStatus() {
      try {
        const res = await apiFetch('/api/studio/status');
        if (!res.ok) return;
        const data = await res.json();
        state.status = data;
        const wasSetup = state.setup;
        state.setup = Boolean(data.setup);
        $('setupBanner').style.display = state.setup ? '' : 'none';
        $('setupError').style.display = data.setup_error ? '' : 'none';
        if (data.setup_error) $('setupError').textContent = `The model could not be loaded: ${data.setup_error}`;
        $('loadedModelCard').style.display = state.setup ? 'none' : '';
        if (state.setup && wasSetup === undefined) switchSurface('hub');
        updateHero(data);

        const m = data.model || {};
        if (!state.selectedModel) state.selectedModel = m.name || 'tiny';
        $('activeModelLabel').textContent = state.selectedModel;
        $('activeQuantChip').textContent = m.quantization || 'GGUF';
        $('footerSimdLabel').textContent = `CPU · ${data.config?.active_cpu_tier || 'simd'}`;
        $('footerSlotsLabel').textContent = `${data.metrics?.slots_running || 0}/${data.metrics?.slots_total || 8} slots`;
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
          <div class="app-card-item"><div class="app-item-info"><div class="app-item-title">Architecture & Parameters</div><div class="app-item-desc mono">${escapeHtml(m.arch)} · ${escapeHtml(m.parameter_size)} (${Number(m.parameters || 0).toLocaleString()} params) · ${formatBytes(m.size_bytes)}</div></div><span class="quant-chip">${escapeHtml(m.quantization)}</span></div>
          <div class="app-card-item"><div class="app-item-info"><div class="app-item-title">Context Window & Vocabulary</div><div class="app-item-desc mono">${m.max_context} tokens (trained ${m.trained_context}) · ${m.vocab_size} vocab · Tool parser: ${escapeHtml(m.tool_parser)}</div></div></div>
        `;
        $('providerAliasesList').innerHTML = (m.aliases || []).map((a) => `
          <span class="backend-chip">${escapeHtml(a)} <button data-del-alias="${escapeHtml(a)}" style="color:var(--destructive);margin-left:4px;">✕</button></span>
        `).join('');
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
            <thead><tr><th>ID</th><th>Label</th><th>Key</th><th>Rate Limit</th><th>Requests</th><th>Tokens</th><th>429s</th><th></th></tr></thead>
            <tbody>
              ${keys.map((k) => `
                <tr>
                  <td class="mono">${escapeHtml(k.id)}</td>
                  <td><strong>${escapeHtml(k.label)}</strong></td>
                  <td class="mono">${escapeHtml(k.masked)}</td>
                  <td class="mono">${k.rpm_limit ? k.rpm_limit + ' pro Minute' : 'Standard'}</td>
                  <td class="mono tabular">${k.requests_accepted}</td>
                  <td class="mono tabular">${k.tokens_used}</td>
                  <td class="mono tabular">${k.rejected}</td>
                  <td><button class="btn btn-xs btn-destructive" data-revoke-key="${escapeHtml(k.id)}">Widerrufen</button></td>
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

          // a device that reports no memory cannot hold a model: it stays out of the list (the details for a bug report still name it)
          const usable = devs.filter((d) => d.backend === 'cpu' || Number(d.total_memory_bytes) > 0);
          const chosen = usable.find((d) => d.id === hw.auto_choice);
          $('hwChoiceText').textContent = chosen
            ? `With the automatic choice fLLM runs models on: ${chosen.name} (${({ cpu: 'the processor', vulkan: 'Vulkan', hip: 'AMD ROCm', cuda: 'NVIDIA CUDA', metal: 'Metal', dx12: 'DirectX 12' })[String(chosen.backend).toLowerCase()] || chosen.backend}${chosen.id.startsWith('wgpu') ? ', generic' : ''}). The processor is the fallback.`
            : 'The processor runs the models: no usable graphics device was found.';
          $('hwDevicesGrid').innerHTML = usable.map((d) => `
            <div class="device-card">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
                <div>
                  <div style="font-weight:600;color:var(--foreground);">${escapeHtml(d.name)}</div>
                  <div style="margin-top:6px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
                    ${d.id === hw.auto_choice ? '<span class="quant-chip" style="color:var(--primary);">in use (automatic choice)</span>' : ''}
                    <span class="backend-chip"><span class="backend-chip-dot"></span>${escapeHtml(({ cpu: 'Processor', cuda: 'NVIDIA CUDA', vulkan: 'Vulkan', hip: 'AMD ROCm', metal: 'Metal', dx12: 'DirectX 12' })[String(d.backend).toLowerCase()] || d.backend)}</span>
                    ${d.description ? `<span class="quant-chip">${escapeHtml(String(d.description).replace(/\bthreads\b/, 'threads'))}</span>` : ''}
                    ${d.unified_memory ? '<span class="quant-chip">shares system memory</span>' : ''}
                  </div>
                  ${d.notes ? `<details class="device-notes"><summary>Technical notes</summary><div class="app-item-desc">${escapeHtml(d.notes)}</div></details>` : ''}
                </div>
                <span class="mono tabular" style="font-size:12px;white-space:nowrap;">${formatBytes(d.free_memory_bytes)} free of ${formatBytes(d.total_memory_bytes)}</span>
              </div>
            </div>
          `).join('');
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
      const model = state.selectedModel || state.status?.model?.name || 'tiny';
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

