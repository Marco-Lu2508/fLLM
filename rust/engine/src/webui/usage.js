// ==========================================================================
    // Usage and cost: tokens per model for this browser, and what they cost with the prices the user entered.
    //  - Counted from the usage block every stream returns (prompt_tokens, completion_tokens).
    //  - Prices come from prices.json (Settings > Usage); nothing is prefilled, because a local model costs
    //    nothing and provider prices differ per plan. Unpriced models show tokens without a dollar amount.
    // Exposed on window: fllmUsageRecord(model, usage), fllmUsageSummary(), fllmUsageReset().
    // ==========================================================================

    (function usageModule() {
      const KEY = 'fllm.studio.usage';
      const MAX_MODELS = 60;

      function load() {
        try {
          const raw = JSON.parse(localStorage.getItem(KEY) || '{}');
          return raw && typeof raw === 'object' && raw.byModel ? raw : { byModel: {}, since: new Date().toISOString() };
        } catch {
          return { byModel: {}, since: new Date().toISOString() };
        }
      }
      let usage = load();
      let prices = [];

      function save() {
        try { localStorage.setItem(KEY, JSON.stringify(usage)); } catch { /* storage full */ }
      }

      const modelKey = (model) => String(model || 'unknown').slice(0, 120);

      /** Adds one answer's usage. `usage` is the object of the stream (prompt_tokens, completion_tokens). */
      function record(model, raw) {
        if (!raw || typeof raw !== 'object') return;
        const prompt = Number(raw.prompt_tokens) || 0;
        const completion = Number(raw.completion_tokens) || 0;
        if (!prompt && !completion) return;
        const key = modelKey(model);
        const entry = usage.byModel[key] || { calls: 0, prompt: 0, completion: 0 };
        entry.calls += 1;
        entry.prompt += prompt;
        entry.completion += completion;
        usage.byModel[key] = entry;
        const keys = Object.keys(usage.byModel);
        if (keys.length > MAX_MODELS) delete usage.byModel[keys[0]];
        save();
        renderUsage();
      }

      /** The price row that fits the model name best (longest pattern wins; `*` matches any tail). */
      function priceFor(model) {
        const name = String(model || '').toLowerCase();
        let best = null;
        let bestLength = 0;
        for (const row of prices) {
          const pattern = String(row.pattern || '').toLowerCase().trim();
          if (!pattern) continue;
          const hit = pattern.endsWith('*') ? name.startsWith(pattern.slice(0, -1)) : (name === pattern || name.includes(pattern));
          if (hit && pattern.length > bestLength) { best = row; bestLength = pattern.length; }
        }
        return best;
      }

      function costOf(model, entry) {
        const price = priceFor(model);
        if (!price) return null;
        const input = Number(price.input_per_million) || 0;
        const output = Number(price.output_per_million) || 0;
        return (entry.prompt / 1e6) * input + (entry.completion / 1e6) * output;
      }

      function totals() {
        let calls = 0, prompt = 0, completion = 0, cost = 0, priced = 0;
        for (const [model, entry] of Object.entries(usage.byModel)) {
          calls += entry.calls;
          prompt += entry.prompt;
          completion += entry.completion;
          const c = costOf(model, entry);
          if (c !== null) { cost += c; priced += 1; }
        }
        return { calls, prompt, completion, cost, priced };
      }

      const fmtTokens = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
      const fmtMoney = (n) => n === null ? '—' : `$${n < 0.01 ? n.toFixed(4) : n.toFixed(2)}`;

      let pane = null;

      function renderUsage() {
        if (!pane) return;
        const t = totals();
        const rows = Object.entries(usage.byModel).sort((a, b) => (b[1].prompt + b[1].completion) - (a[1].prompt + a[1].completion));
        pane.querySelector('#usageTotals').textContent =
          `${fmtTokens(t.prompt)} in / ${fmtTokens(t.completion)} out in ${t.calls} answer(s)` +
          (t.priced ? ` · about ${fmtMoney(t.cost)}` : ' · no prices entered, so no dollar amount');
        pane.querySelector('#usageTable').innerHTML = rows.length
          ? '<table class="tabular" style="width:100%;"><thead><tr><th>Model</th><th>Answers</th><th>Input</th><th>Output</th><th>Cost</th></tr></thead><tbody>' +
            rows.map(([model, entry]) => {
              const c = costOf(model, entry);
              return `<tr><td class="mono">${escapeHtml(model)}</td><td>${entry.calls}</td><td>${fmtTokens(entry.prompt)}</td><td>${fmtTokens(entry.completion)}</td><td>${fmtMoney(c)}</td></tr>`;
            }).join('') +
            `</tbody><tfoot><tr><th>All</th><th>${t.calls}</th><th>${fmtTokens(t.prompt)}</th><th>${fmtTokens(t.completion)}</th><th>${fmtMoney(t.priced ? t.cost : null)}</th></tr></tfoot></table>`
          : '<div class="app-item-desc">Nothing counted yet in this browser. Ask something in Chat or run the Code agent.</div>';
        renderPrices();
      }

      function renderPrices() {
        if (!pane) return;
        pane.querySelector('#priceRows').innerHTML = prices.map((row, i) =>
          '<div class="app-card-item" data-price="' + i + '" style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">' +
          '  <input class="ui-input mono" data-f="pattern" value="' + attr(row.pattern || '') + '" placeholder="model pattern, e.g. gpt-6*" style="width:200px;">' +
          '  <input class="ui-input mono" data-f="input_per_million" type="number" min="0" step="0.01" value="' + (row.input_per_million ?? 0) + '" style="width:110px;" title="Dollars per million input tokens">' +
          '  <input class="ui-input mono" data-f="output_per_million" type="number" min="0" step="0.01" value="' + (row.output_per_million ?? 0) + '" style="width:110px;" title="Dollars per million output tokens">' +
          '  <input class="ui-input" data-f="note" value="' + attr(row.note || '') + '" placeholder="note (optional)" style="flex:1;min-width:140px;">' +
          '  <button class="btn btn-sm btn-ghost" data-action="remove-price">Remove</button>' +
          '</div>').join('') || '<div class="app-item-desc">No price rows. Local models need none.</div>';
      }

      function attr(text) { return escapeHtml(text).replace(/"/g, '&quot;'); }

      function collectPrices() {
        const rows = [];
        pane.querySelectorAll('[data-price]').forEach((row) => {
          const value = (f) => row.querySelector(`[data-f="${f}"]`).value;
          const pattern = value('pattern').trim();
          if (!pattern) return;
          rows.push({
            pattern,
            input_per_million: Number(value('input_per_million')) || 0,
            output_per_million: Number(value('output_per_million')) || 0,
            note: value('note').trim(),
          });
        });
        return rows;
      }

      async function loadPrices() {
        try {
          const res = await apiFetch('/api/studio/prices');
          const data = await res.json();
          prices = Array.isArray(data.prices) ? data.prices : [];
        } catch { prices = []; }
        renderPrices();
      }

      async function savePrices() {
        const rows = collectPrices();
        try {
          const res = await apiFetch('/api/studio/prices', { method: 'POST', body: { prices: rows } });
          const data = await res.json();
          if (!res.ok) { say(data.error && data.error.message ? data.error.message : 'Could not save the prices.'); return; }
          prices = Array.isArray(data.prices) ? data.prices : [];
          renderPrices();
          renderUsage();
          say('Prices saved.');
        } catch (e) {
          say('Could not save: ' + ((e && e.message) || e));
        }
      }

      function say(text) {
        if (!pane) return;
        pane.querySelector('#usageStatus').textContent = text || '';
      }

      // ---- the settings pane -------------------------------------------------------------------------------
      const nav = document.getElementById('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      if (nav && content) {
        const button = document.createElement('button');
        button.className = 'settings-nav-btn';
        button.dataset.spane = 'usage';
        button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg><span>Usage and cost</span>';
        nav.appendChild(button);

        pane = document.createElement('div');
        pane.className = 'settings-pane';
        pane.id = 'spane-usage';
        pane.innerHTML =
          '<div class="app-card"><div class="app-card-title">Usage and cost</div>' +
          '<div class="app-item-desc" style="margin-bottom:10px;">Tokens are counted in this browser from the usage block of every answer. A local model costs nothing; for a model behind a provider, add its price below and the cost appears here and in <span class="mono">/cost</span>.</div>' +
          '<div class="app-item-desc" id="usageTotals" style="margin-bottom:10px;"></div>' +
          '<div id="usageTable" style="margin-bottom:10px;"></div>' +
          '<button class="btn btn-sm btn-ghost" id="btnUsageReset">Reset counters</button>' +
          '<div class="app-card-title" style="margin-top:18px;">Prices (dollars per million tokens)</div>' +
          '<div class="app-item-desc" style="margin-bottom:8px;">A pattern is matched against the model name; the longest match wins, <span class="mono">*</span> matches any tail (for example <span class="mono">gpt-6*</span>). Models without a row show tokens only.</div>' +
          '<div id="priceRows"></div>' +
          '<div style="display:flex;gap:8px;align-items:center;margin-top:8px;">' +
          '  <button class="btn btn-sm btn-ghost" id="btnPriceAdd">+ Add price</button>' +
          '  <button class="btn btn-sm" id="btnPriceSave">Save prices</button>' +
          '  <span class="app-item-desc mono" id="usageStatus" style="font-size:11px;"></span>' +
          '</div></div>';
        content.appendChild(pane);

        pane.addEventListener('click', (e) => {
          const action = e.target.dataset && e.target.dataset.action;
          if (action === 'remove-price') {
            const row = e.target.closest('[data-price]');
            if (row) { row.remove(); say('Removed. Press Save prices to write it.'); }
          }
        });
        pane.querySelector('#btnPriceAdd').addEventListener('click', () => {
          prices = collectPrices();
          prices.push({ pattern: '', input_per_million: 0, output_per_million: 0, note: '' });
          renderPrices();
        });
        pane.querySelector('#btnPriceSave').addEventListener('click', savePrices);
        pane.querySelector('#btnUsageReset').addEventListener('click', () => {
          usage = { byModel: {}, since: new Date().toISOString() };
          save();
          renderUsage();
          say('Counters reset.');
        });

        renderUsage();
        loadPrices();
      }

      // ---- the public surface ------------------------------------------------------------------------------
      window.fllmUsageRecord = record;
      window.fllmUsageSummary = () => {
        const t = totals();
        return {
          ...t,
          text: `${fmtTokens(t.prompt)} in / ${fmtTokens(t.completion)} out in ${t.calls} answer(s)` +
            (t.priced ? `, about ${fmtMoney(t.cost)}` : ' (no prices entered)'),
        };
      };
      window.fllmUsageReset = () => { usage = { byModel: {}, since: new Date().toISOString() }; save(); renderUsage(); };
    })();
