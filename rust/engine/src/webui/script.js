  (() => {
    'use strict';

    // Accent palette
    const ACCENT_COLORS = [
      { name: 'Patina',  value: 'patina',  thumb: '#43c0aa', primary: '#43c0aa', primaryLight: '#17806f', sidebar: { light: '#e3eae7', dark: '#0b1116' } },
      { name: 'Kupfer',  value: 'kupfer',  thumb: '#d9925a', primary: '#e0955c', primaryLight: '#b4672b', sidebar: { light: '#ece6df', dark: '#150f0b' } },
      { name: 'Emerald', value: 'emerald', thumb: '#38AB51', primary: '#38AB51', sidebar: { light: '#C4F3CE', dark: '#194D24' } },
      { name: 'Teal',    value: 'teal',    thumb: '#38AB8D', primary: '#38AB8D', sidebar: { light: '#C4F3E6', dark: '#194D3F' } },
      { name: 'Cyan',    value: 'cyan',    thumb: '#45BBDE', primary: '#45BBDE', sidebar: { light: '#C4E8F3', dark: '#0F4657' } },
      { name: 'Blue',    value: 'blue',    thumb: '#456BDE', primary: '#456BDE', sidebar: { light: '#C4D0F3', dark: '#0F2157' } },
      { name: 'Purple',  value: 'purple',  thumb: '#865EEA', primary: '#865EEA', sidebar: { light: '#D2C4F3', dark: '#220C5A' } },
      { name: 'Orange',  value: 'orange',  thumb: '#E9A23F', primary: '#E9A23F', sidebar: { light: '#F3DFC4', dark: '#5C3A0A' } },
      { name: 'Red',     value: 'red',     thumb: '#F0614B', primary: '#F0614B', sidebar: { light: '#F3CBC4', dark: '#5E1308' } },
      { name: 'Green',   value: 'green',   thumb: '#88BA42', primary: '#88BA42', sidebar: { light: '#DFF3C4', dark: '#374B1B' } },
      { name: 'Pink',    value: 'pink',    thumb: '#D55EF3', primary: '#D55EF3', sidebar: { light: '#FFDAE9', dark: '#4D075F' } },
      { name: 'Rose',    value: 'rose',    thumb: '#F655B8', primary: '#F655B8', sidebar: { light: '#F3C4E1', dark: '#61053E' } },
    ];

    const STORAGE = {
      threads: 'fllm.studio.threads.v2',
      activeThread: 'fllm.studio.activeThread.v2',
      theme: 'fllm.studio.theme.v2',
      accent: 'fllm.studio.accent.v2',
      fontSize: 'fllm.studio.fontSize.v2',
      coloredBubble: 'fllm.studio.coloredBubble.v2',
      showSpeed: 'fllm.studio.showSpeed.v2',
      apiKey: 'fllm.studio.apiKey.v2',
    };

    const state = {
      surface: 'chat', // 'chat' | 'hub' | 'settings'
      playMode: 'chat', // 'chat' | 'infill' | 'arena'
      thinkingEnabled: false,
      toolsEnabled: false,
      threads: [],
      activeThreadId: null,
      attachments: [],
      status: null,
      modelsHub: null,
      keysData: null,
      hardware: null,
      selectedModel: '',
      apiKey: localStorage.getItem(STORAGE.apiKey) || '',
      theme: localStorage.getItem(STORAGE.theme) || 'dark',
      accent: localStorage.getItem(STORAGE.accent) || 'patina',
      fontSize: localStorage.getItem(STORAGE.fontSize) || '15px',
      coloredBubble: localStorage.getItem(STORAGE.coloredBubble) !== 'false',
      showSpeed: localStorage.getItem(STORAGE.showSpeed) !== 'false',
      sdkLang: 'claude',
      abortController: null,
    };

    const $ = (id) => document.getElementById(id);

    function escapeHtml(str) {
      return String(str ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    function formatBytes(bytes) {
      const b = Number(bytes || 0);
      if (b <= 0) return '0 B';
      const units = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.min(Math.floor(Math.log(b) / Math.log(1024)), units.length - 1);
      return `${(b / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
    }

    function toast(msg) {
      const el = document.createElement('div');
      el.className = 'toast-card';
      el.textContent = msg;
      $('toastStack').appendChild(el);
      setTimeout(() => el.remove(), 3400);
    }

    async function apiFetch(path, opts = {}) {
      const headers = Object.assign({}, opts.headers || {});
      if (state.apiKey) {
        headers['Authorization'] = `Bearer ${state.apiKey}`;
      }
      if (opts.body && typeof opts.body === 'object' && !(opts.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(opts.body);
      }
      return fetch(path, Object.assign({}, opts, { headers }));
    }

    // --- Theme & Accent Colors ---
    function applyAppearance() {
      const isDark = state.theme === 'dark';
      document.documentElement.classList.toggle('dark', isDark);
      document.documentElement.style.setProperty('--font-size-base', state.fontSize);
      document.body.classList.toggle('colored-user-bubble', state.coloredBubble);

      const c = ACCENT_COLORS.find((x) => x.value === state.accent) || ACCENT_COLORS[0];
      const primary = isDark ? c.primary : (c.primaryLight || c.primary);
      // text on the accent: dark on light accents, light on dark ones
      const lum = [1, 3, 5].map((i) => parseInt(primary.slice(i, i + 2), 16) / 255).reduce((s, v, k) => s + v * [0.2126, 0.7152, 0.0722][k], 0);
      document.documentElement.style.setProperty('--primary', primary);
      document.documentElement.style.setProperty('--ring', primary);
      document.documentElement.style.setProperty('--primary-foreground', lum > 0.45 ? '#06161a' : '#f6fbfa');
      document.documentElement.style.setProperty('--sidebar', isDark ? c.sidebar.dark : c.sidebar.light);

      $('selThemeMode').value = state.theme;
      $('selFontSize').value = state.fontSize;
      $('chkColoredBubble').checked = state.coloredBubble;
      $('chkShowSpeed').checked = state.showSpeed;

      $('accentPickerRow').innerHTML = ACCENT_COLORS.map((col) => `
        <button
          type="button"
          class="accent-swatch ${col.value === state.accent ? 'active' : ''}"
          title="${col.name}"
          data-accent="${col.value}"
          style="background:${col.thumb};"
        ></button>
      `).join('');
    }

    // --- Surface Navigation (Chat / Hub / Settings) ---
    function switchSurface(surfaceName, settingsPane) {
      state.surface = surfaceName;
      ['chat', 'hub', 'settings'].forEach((s) => {
        $(`surface-${s}`).classList.toggle('active', s === surfaceName);
      });
      $('navHub').classList.toggle('active', surfaceName === 'hub');
      $('navSettings').classList.toggle('active', surfaceName === 'settings');

      if (surfaceName === 'hub') refreshHub();
      if (surfaceName === 'settings') {
        if (settingsPane) switchSettingsPane(settingsPane);
        refreshApiKeys();
        refreshHardware(false);
        updateCliAndSdkSnippets();
      }
    }

    function switchSettingsPane(paneId) {
      document.querySelectorAll('#settingsMenuNav .settings-nav-btn').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.spane === paneId);
      });
      document.querySelectorAll('.settings-pane').forEach((pane) => {
        pane.classList.toggle('active', pane.id === `spane-${paneId}`);
      });
      if (paneId === 'tokenizer') {
        runTokenize();
        runJinjaTemplate();
      }
      if (paneId === 'local-api-server') {
        refreshApiKeys();
        refreshPrometheus();
      }
      if (paneId === 'hardware') {
        refreshHardware(false);
      }
    }

    // --- Threads Management ---
    function loadThreads() {
      try {
        const raw = localStorage.getItem(STORAGE.threads);
        state.threads = raw ? JSON.parse(raw) : [];
      } catch {
        state.threads = [];
      }
      state.activeThreadId = localStorage.getItem(STORAGE.activeThread) || null;
      if (state.activeThreadId && !state.threads.some((t) => t.id === state.activeThreadId)) {
        state.activeThreadId = state.threads[0]?.id || null;
      }
    }

    function saveThreads() {
      localStorage.setItem(STORAGE.threads, JSON.stringify(state.threads));
      if (state.activeThreadId) {
        localStorage.setItem(STORAGE.activeThread, state.activeThreadId);
      } else {
        localStorage.removeItem(STORAGE.activeThread);
      }
    }

    function currentThread() {
      return state.threads.find((t) => t.id === state.activeThreadId) || null;
    }

    function startNewChat() {
      state.activeThreadId = null;
      saveThreads();
      switchSurface('chat');
      renderThreadList();
      renderChatArea();
      $('chatTextarea').focus();
    }

    function ensureActiveThread(initialTitle) {
      let t = currentThread();
      if (!t) {
        t = {
          id: 'thr_' + Date.now(),
          title: (initialTitle || 'New Chat').slice(0, 48),
          updatedAt: Date.now(),
          messages: [],
        };
        state.threads.unshift(t);
        state.activeThreadId = t.id;
        saveThreads();
        renderThreadList();
      }
      return t;
    }

    function renderThreadList() {
      const container = $('threadList');
      if (!state.threads.length) {
        container.innerHTML = `<div style="padding:8px 6px;font-size:12px;color:var(--muted-foreground);">Noch keine Chats</div>`;
        return;
      }
      container.innerHTML = state.threads.map((t) => `
        <div class="thread-row ${t.id === state.activeThreadId && state.surface === 'chat' ? 'active' : ''}" data-tid="${t.id}">
          <span class="thread-row-title">${escapeHtml(t.title)}</span>
          <div class="thread-row-actions">
            <button class="btn-icon btn-icon-xs" data-del-tid="${t.id}" title="Delete thread">
              <svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
            </button>
          </div>
        </div>
      `).join('');
    }

    // --- Markdown & Reasoning Parser ---
    function splitReasoning(text) {
      const openIdx = text.indexOf('<think>');
      if (openIdx === -1) return { reasoning: '', answer: text };
      const closeIdx = text.indexOf('</think>');
      if (closeIdx === -1) {
        return { reasoning: text.slice(openIdx + 7).trim(), answer: '' };
      }
      return {
        reasoning: text.slice(openIdx + 7, closeIdx).trim(),
        answer: (text.slice(0, openIdx) + text.slice(closeIdx + 8)).trim(),
      };
    }

    let codeBlockSeq = 0;
    function renderMarkdown(md) {
      if (!md) return '';
      const blocks = [];
      let text = String(md).replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const id = ++codeBlockSeq;
        blocks.push(`
          <div class="code-card">
            <div class="code-card-header">
              <span>${escapeHtml(lang || 'code')}</span>
              <button class="btn btn-xs btn-ghost" data-copy-code="code-${id}">Copy</button>
            </div>
            <pre id="code-${id}">${escapeHtml(code.replace(/\n$/, ''))}</pre>
          </div>
        `);
        return `\x00CODE${blocks.length - 1}\x00`;
      });

      text = escapeHtml(text)
        .replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*\n]+)\*/g, '<em>$1</em>');

      const paras = text.split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
      return paras.replace(/\x00CODE(\d+)\x00/g, (_, idx) => blocks[Number(idx)]);
    }

    function renderChatArea() {
      const t = currentThread();
      const hasMessages = t && Array.isArray(t.messages) && t.messages.length > 0;
      $('chatContainer').classList.toggle('is-empty', !hasMessages);
      if (!hasMessages) {
        $('messagesColumn').innerHTML = '';
        return;
      }

      $('messagesColumn').innerHTML = t.messages.map((m, idx) => {
        if (m.role === 'user') {
          const imgs = Array.isArray(m.images) && m.images.length
            ? `<div style="display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap;">${m.images.map((src) => `<img src="${src}" style="max-height:120px;border-radius:8px;">`).join('')}</div>`
            : '';
          return `
            <div class="msg-user-wrap">
              <div class="msg-user-bubble">${imgs}${escapeHtml(m.content)}</div>
              <div class="msg-actions-bar">
                <button class="btn-icon btn-icon-xs" data-msg-copy="${idx}" title="Copy"><svg viewBox="0 0 24 24" class="icon icon-sm"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg></button>
                <button class="btn-icon btn-icon-xs" data-msg-del="${idx}" title="Delete"><svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/></svg></button>
              </div>
            </div>
          `;
        }

        const parsed = splitReasoning(m.content || '');
        const reasoning = m.reasoning || parsed.reasoning;
        const answer = m.reasoning ? (m.content || '') : parsed.answer;

        const cotHtml = reasoning ? `
          <details class="cot-box" ${m.streaming && !answer ? 'open' : ''}>
            <summary>
              <svg viewBox="0 0 24 24" class="icon icon-sm" style="color:var(--primary);"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/></svg>
              <span>Chain of Thought (${reasoning.length} chars)</span>
            </summary>
            <div class="cot-body">${escapeHtml(reasoning)}</div>
          </details>
        ` : '';

        const lpHtml = Array.isArray(m.logprobs) && m.logprobs.length ? `
          <div class="logprobs-wrap">
            ${m.logprobs.map((lp) => {
              const p = Math.exp(Number(lp.logprob || 0));
              const hue = Math.round(p * 130);
              return `<span class="lp-tok" style="background:hsla(${hue},70%,42%,0.22);" title="p=${(p * 100).toFixed(1)}% (${Number(lp.logprob).toFixed(3)})">${escapeHtml(lp.token)}</span>`;
            }).join('')}
          </div>
        ` : '';

        const speedHtml = (state.showSpeed && m.telemetry) ? `
          <span class="speed-chip">
            ${m.telemetry.tokPerSec ? `<span>⚡ ${m.telemetry.tokPerSec} tok/s</span>` : ''}
            ${m.telemetry.ttftMs ? `<span>· TTFT ${m.telemetry.ttftMs}ms</span>` : ''}
            ${m.telemetry.tokens ? `<span>· ${m.telemetry.tokens} tok</span>` : ''}
          </span>
        ` : '';

        return `
          <div class="msg-assistant-wrap">
            ${cotHtml}
            <div class="msg-assistant-prose">
              ${answer ? renderMarkdown(answer) : (m.streaming ? '<span style="color:var(--muted-foreground);">Thinking…</span>' : '')}
            </div>
            ${lpHtml}
            <div class="msg-actions-bar">
              <button class="btn-icon btn-icon-xs" data-msg-copy="${idx}" title="Copy response">
                <svg viewBox="0 0 24 24" class="icon icon-sm"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
              </button>
              <button class="btn-icon btn-icon-xs" data-msg-regen="${idx}" title="Regenerate">
                <svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>
              </button>
              <button class="btn-icon btn-icon-xs" data-msg-del="${idx}" title="Delete">
                <svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/></svg>
              </button>
              ${speedHtml}
            </div>
          </div>
        `;
      }).join('');

      const sc = $('chatMessagesScroll');
      sc.scrollTop = sc.scrollHeight;
    }

    // --- Sampling Parameters Helper ---
    function getSampler() {
      return {
        temperature: parseFloat($('numTemp').value) || 0.7,
        top_p: parseFloat($('numTopP').value) || 0.95,
        top_k: parseInt($('numTopK').value, 10) || 0,
        min_p: parseFloat($('numMinP').value) || 0.0,
        max_tokens: parseInt($('numMaxTokens').value, 10) || 256,
        repetition_penalty: parseFloat($('numRepeatPen').value) || 1.0,
        presence_penalty: parseFloat($('numPresencePen').value) || 0.0,
        frequency_penalty: parseFloat($('numFreqPen').value) || 0.0,
        dry_multiplier: parseFloat($('numDryMult').value) || 0.0,
        mirostat: parseInt($('selMirostat').value, 10) || 0,
        seed: parseInt($('numSeed').value, 10),
        constraint: $('selConstraint').value,
        constraint_spec: $('txtConstraintSpec').value.trim(),
        speculative: $('chkSpeculative').checked,
        logprobs: $('chkLogprobs').checked,
        system_prompt: $('txtSystemPrompt').value.trim(),
        stop: $('inpStopWords').value.split(',').map((s) => s.trim()).filter(Boolean),
      };
    }

    function setSamplerUI(cfg) {
      if (!cfg) return;
      const pairs = [
        ['Temp', cfg.temperature],
        ['TopP', cfg.top_p],
        ['TopK', cfg.top_k],
        ['MinP', cfg.min_p],
        ['RepeatPen', cfg.repetition_penalty],
      ];
      pairs.forEach(([k, v]) => {
        if (v !== undefined && v !== null) {
          $(`num${k}`).value = v;
          $(`rng${k}`).value = v;
        }
      });
      if (cfg.presence_penalty !== undefined) $('numPresencePen').value = cfg.presence_penalty;
      if (cfg.frequency_penalty !== undefined) $('numFreqPen').value = cfg.frequency_penalty;
      if (cfg.dry_multiplier !== undefined) $('numDryMult').value = cfg.dry_multiplier;
      if (cfg.mirostat !== undefined) $('selMirostat').value = String(cfg.mirostat);
      if (cfg.seed !== undefined) $('numSeed').value = cfg.seed === null ? -1 : cfg.seed;
      if (cfg.speculative !== undefined) $('chkSpeculative').checked = Boolean(cfg.speculative);
      if (cfg.system_prompt !== undefined && !$('txtSystemPrompt').value) {
        $('txtSystemPrompt').value = cfg.system_prompt;
      }
    }

    // Bind sliders <-> number inputs
    [['Temp', 'Temp'], ['TopP', 'TopP'], ['TopK', 'TopK'], ['MinP', 'MinP'], ['MaxTokens', 'MaxTokens'], ['RepeatPen', 'RepeatPen']].forEach(([n, r]) => {
      $(`rng${r}`).addEventListener('input', () => { $(`num${n}`).value = $(`rng${r}`).value; });
      $(`num${n}`).addEventListener('input', () => { $(`rng${r}`).value = $(`num${n}`).value; });
    });

    // --- Send Message / Stream Completion ---
    async function sendChatPrompt() {
      if (state.abortController) {
        state.abortController.abort();
        return;
      }
      const p = getSampler();
      const model = state.selectedModel || state.status?.model?.name || 'tiny';

      if (state.playMode === 'infill') {
        const prefix = $('fimPrefixInput').value;
        const suffix = $('fimSuffixInput').value;
        const t = ensureActiveThread('FIM Code Infill');
        t.messages.push({ role: 'user', content: `[FIM Prefix]\n${prefix}\n[FIM Suffix]\n${suffix}` });
        const aMsg = { role: 'assistant', content: '', streaming: true };
        t.messages.push(aMsg);
        renderChatArea();
        const t0 = performance.now();
        try {
          const res = await apiFetch('/infill', {
            method: 'POST',
            body: { input_prefix: prefix, input_suffix: suffix, n_predict: p.max_tokens, temperature: p.temperature, stream: false },
          });
          const data = await res.json();
          if (!res.ok) throw new Error((data.error && data.error.message) || ('HTTP ' + res.status) + ' - Dieses Modell unterstuetzt keine Code-Vervollstaendigung (Fill-in-the-Middle). Nutze ein Code-Modell wie Qwen2.5-Coder oder CodeLlama.');
          const middle = data.content || data.choices?.[0]?.text || '';
          const elapsed = (performance.now() - t0) / 1000;
          aMsg.content = '```rust\n' + prefix + middle + suffix + '\n```';
          aMsg.streaming = false;
          aMsg.telemetry = { tokPerSec: (16 / Math.max(elapsed, 0.05)).toFixed(1), ttftMs: Math.round(elapsed * 1000) };
          saveThreads();
          renderChatArea();
        } catch (e) {
          aMsg.content = 'Fehler: ' + e.message + ' (Dieses Modell hat keine Fill-in-the-Middle-Tokens. Nutze ein Code-Modell wie Qwen2.5-Coder oder CodeLlama aus dem Hub.)';
          aMsg.streaming = false;
          renderChatArea();
        }
        return;
      }

      const text = $('chatTextarea').value.trim();
      if (!text && !state.attachments.length) return;
      $('chatTextarea').value = '';
      $('chatTextarea').style.height = 'auto';

      const t = ensureActiveThread(text);
      if (t.messages.length === 0 && text) t.title = text.slice(0, 46);
      t.messages.push({
        role: 'user',
        content: text,
        images: state.attachments.length ? [...state.attachments] : undefined,
      });
      state.attachments = [];
      renderAttachments();

      const aMsg = { role: 'assistant', content: '', reasoning: '', streaming: true, logprobs: [] };
      t.messages.push(aMsg);
      saveThreads();
      renderThreadList();
      renderChatArea();

      const apiMessages = [];
      if (p.system_prompt) apiMessages.push({ role: 'system', content: p.system_prompt });
      for (const m of t.messages.slice(0, -1)) {
        if (m.role === 'user' && Array.isArray(m.images) && m.images.length) {
          const parts = [];
          if (m.content) parts.push({ type: 'text', text: m.content });
          m.images.forEach((url) => parts.push({ type: 'image_url', image_url: { url } }));
          apiMessages.push({ role: 'user', content: parts });
        } else {
          apiMessages.push({ role: m.role, content: m.content || '' });
        }
      }

      const body = {
        model,
        messages: apiMessages,
        stream: true,
        temperature: p.temperature,
        top_p: p.top_p,
        top_k: p.top_k,
        min_p: p.min_p,
        max_tokens: p.max_tokens,
        repetition_penalty: p.repetition_penalty,
        presence_penalty: p.presence_penalty,
        frequency_penalty: p.frequency_penalty,
        speculative: p.speculative,
      };
      if (p.seed >= 0) body.seed = p.seed;
      if (p.stop.length) body.stop = p.stop;
      if (p.dry_multiplier > 0) body.dry_multiplier = p.dry_multiplier;
      if (p.mirostat > 0) body.mirostat = p.mirostat;
      if (p.logprobs) { body.logprobs = true; body.top_logprobs = 5; }
      if (state.thinkingEnabled) body.reasoning_effort = 'high';
      if (p.constraint === 'json_object') body.response_format = { type: 'json_object' };
      else if (p.constraint === 'json_schema' && p.constraint_spec) {
        try { body.response_format = { type: 'json_schema', json_schema: { name: 'schema', strict: true, schema: JSON.parse(p.constraint_spec) } }; } catch {}
      } else if (p.constraint === 'gbnf' && p.constraint_spec) {
        body.grammar = p.constraint_spec; body.grammar_type = 'gbnf';
      } else if (p.constraint === 'regex' && p.constraint_spec) {
        body.regex = p.constraint_spec;
      }

      state.abortController = new AbortController();
      const sendBtn = $('btnSendChat');
      sendBtn.classList.add('stop');
      sendBtn.innerHTML = `<svg viewBox="0 0 24 24" class="icon icon-sm"><rect width="10" height="10" x="7" y="7" rx="1" fill="currentColor"/></svg>`;

      const t0 = performance.now();
      let firstTokenAt = null;
      let tokCount = 0;

      try {
        const res = await apiFetch('/v1/chat/completions', {
          method: 'POST',
          body,
          signal: state.abortController.signal,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.error?.message || `HTTP ${res.status}`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() || '';
          for (const raw of lines) {
            const line = raw.trim();
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (payload === '[DONE]') break;
            try {
              const chunk = JSON.parse(payload);
              const delta = chunk.choices?.[0]?.delta;
              if (delta) {
                if (firstTokenAt === null && (delta.content || delta.reasoning_content)) {
                  firstTokenAt = performance.now();
                }
                if (delta.reasoning_content) aMsg.reasoning += delta.reasoning_content;
                if (delta.content) { aMsg.content += delta.content; tokCount++; }
              }
              const lp = chunk.choices?.[0]?.logprobs?.content;
              if (Array.isArray(lp)) aMsg.logprobs.push(...lp);
              renderChatArea();
            } catch {}
          }
        }
        const totalSec = Math.max((performance.now() - (firstTokenAt || t0)) / 1000, 0.02);
        aMsg.streaming = false;
        aMsg.telemetry = {
          tokPerSec: (Math.max(tokCount, 1) / totalSec).toFixed(1),
          ttftMs: firstTokenAt ? Math.round(firstTokenAt - t0) : Math.round(totalSec * 1000),
          tokens: tokCount,
        };
      } catch (e) {
        aMsg.streaming = false;
        if (e.name !== 'AbortError') aMsg.content += `\n\nError: ${e.message}`;
      } finally {
        state.abortController = null;
        sendBtn.classList.remove('stop');
        sendBtn.innerHTML = `<svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>`;
        saveThreads();
        renderChatArea();
        refreshStudioStatus();
      }
    }

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
      $('welcomeSub').textContent = `Läuft lokal auf diesem Rechner${facts.length ? ': ' + facts.join(', ') : ''}.`;

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
      $('heroTpsLabel').textContent = busy ? 'Token pro Sekunde, das Modell antwortet gerade' : 'Token pro Sekunde, das Modell wartet auf eine Frage';
    }

    // --- Studio Status Sync ---
    async function refreshStudioStatus() {
      try {
        const res = await apiFetch('/api/studio/status');
        if (!res.ok) return;
        const data = await res.json();
        state.status = data;
        updateHero(data);

        const m = data.model || {};
        if (!state.selectedModel) state.selectedModel = m.name || 'tiny';
        $('activeModelLabel').textContent = state.selectedModel;
        $('activeQuantChip').textContent = m.quantization || 'GGUF';
        $('footerSimdLabel').textContent = `CPU · ${data.config?.active_cpu_tier || 'simd'}`;
        $('footerSlotsLabel').textContent = `${data.metrics?.slots_running || 0}/${data.metrics?.slots_total || 8} Plätze`;
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

    // --- Hub ---
    async function refreshHub() {
      try {
        const res = await apiFetch('/api/studio/models');
        if (!res.ok) return;
        state.modelsHub = await res.json();
        renderHub();
      } catch {}
    }

    function renderHub() {
      const m = state.status?.model || {};
      $('loadedModelHubItem').innerHTML = `
        <div class="app-card-item">
          <div class="app-item-info">
            <div class="app-item-title">${escapeHtml(m.name || 'Loaded Model')} <span class="quant-chip" style="color:var(--primary);">In der Engine aktiv</span></div>
            <div class="app-item-desc mono">${escapeHtml(m.arch || 'llama')} · ${escapeHtml(m.parameter_size || '')} · ${formatBytes(m.size_bytes)} · ${m.max_context || 512} ctx</div>
          </div>
          <div class="app-item-actions">
            <button class="btn btn-sm btn-primary" data-use-model="${escapeHtml(m.name || '')}">Chat</button>
          </div>
        </div>
      `;

      const cached = state.modelsHub?.cached || [];
      $('cachedModelsHubList').innerHTML = cached.map((c) => `
        <div class="app-card-item">
          <div class="app-item-info">
            <div class="app-item-title mono">${escapeHtml(c.name)}</div>
            <div class="app-item-desc mono">${formatBytes(c.size_bytes)} · ${escapeHtml(c.path)}</div>
          </div>
          <div class="app-item-actions">
            <button class="btn btn-xs btn-destructive" data-del-cache="${escapeHtml(c.name)}">Löschen</button>
          </div>
        </div>
      `).join('');

      const q = ($('hubSearchInput').value || '').toLowerCase().trim();
      const cat = (state.modelsHub?.catalog || []).filter((item) => {
        if (!q) return true;
        return item.id.toLowerCase().includes(q) || item.display_name.toLowerCase().includes(q) || item.architecture.toLowerCase().includes(q);
      });

      $('hubCatalogCards').innerHTML = cat.map((entry) => `
        <div class="app-card">
          <div class="app-card-header" style="margin-bottom:8px;padding-bottom:8px;">
            <div>
              <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-weight:600;font-size:14.5px;color:var(--foreground);">${escapeHtml(entry.display_name)}</span>
                <span class="quant-chip">${escapeHtml(entry.architecture)}</span>
                <span class="quant-chip">${entry.context_length} ctx</span>
                ${entry.vision ? '<span class="quant-chip">Vision</span>' : ''}
              </div>
              <p class="app-item-desc mono" style="margin-top:3px;">${escapeHtml(entry.default_repo || entry.gguf_repo || '')}: für deinen Arbeitsspeicher empfohlen ist <strong>${escapeHtml(entry.recommended_quant)}</strong>, etwa ${(Number(entry.estimated_bytes || 0) / 1073741824).toFixed(1)} GB</p>
            </div>
            <div style="display:flex;gap:6px;">
              <button class="btn btn-sm btn-outline" data-inspect-hf="${escapeHtml(entry.default_repo || entry.gguf_repo || '')}">Dateien</button>
              <button class="btn btn-sm btn-primary" data-pull-cat="${escapeHtml(entry.id)}" data-pull-q="${escapeHtml(entry.recommended_quant)}">${escapeHtml(entry.recommended_quant)} laden</button>
            </div>
          </div>
        </div>
      `).join('');
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
          $('apiKeysTableContainer').innerHTML = `<div class="app-item-desc">Es ist kein API-Schlüssel eingerichtet. Der Server ist offen und nimmt Änderungen nur von diesem Rechner an. Lege oben einen Schlüssel an, um Anfragen mit Bearer-Token zu verlangen.</div>`;
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
            ? `Befehlssatz-Stufe: ${hw.active_cpu_tier}. Gemessene Speicherbandbreite: ${Number(hw.measured_cpu_bandwidth_gbs).toFixed(1)} GB/s.`
            : `Befehlssatz-Stufe: ${hw.active_cpu_tier} (möglich: ${tiers.join(', ')}).`) + ` ${hw.threads} Threads.`;
          $('hwRamText').textContent = `${formatBytes(freeB)} frei von ${formatBytes(totalB)}`;
          $('hwRamProgress').style.width = `${Math.min(100, Math.max(2, usedPct)).toFixed(0)}%`;
          const sel = $('selSimdTierCap');
          const wanted = ['auto', ...tiers];
          if (sel.options.length !== wanted.length || [...sel.options].some((o, i) => o.value !== wanted[i])) {
            const cur = state.status?.config?.cpu_tier_cap || 'auto';
            sel.innerHTML = wanted.map((t) => `<option value="${t}">${t === 'auto' ? 'Auto (hoechster unterstuetzter)' : t}</option>`).join('');
            sel.value = wanted.includes(cur) ? cur : 'auto';
          }

          $('hwDevicesGrid').innerHTML = devs.map((d) => `
            <div class="device-card">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
                <div>
                  <div style="font-weight:600;color:var(--foreground);">${escapeHtml(d.name)}</div>
                  <div style="margin-top:6px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
                    <span class="backend-chip"><span class="backend-chip-dot"></span>${escapeHtml(({ cpu: 'Prozessor', cuda: 'NVIDIA CUDA', vulkan: 'Vulkan', hip: 'AMD ROCm', metal: 'Metal', dx12: 'DirectX 12' })[String(d.backend).toLowerCase()] || d.backend)}</span>
                    ${d.description ? `<span class="quant-chip">${escapeHtml(String(d.description).replace(/\bthreads\b/, 'Threads'))}</span>` : ''}
                    ${d.unified_memory ? '<span class="quant-chip">teilt den Arbeitsspeicher</span>' : ''}
                  </div>
                  ${d.notes ? `<details class="device-notes"><summary>Technische Hinweise</summary><div class="app-item-desc">${escapeHtml(d.notes)}</div></details>` : ''}
                </div>
                <span class="mono tabular" style="font-size:12px;white-space:nowrap;">${formatBytes(d.free_memory_bytes)} frei von ${formatBytes(d.total_memory_bytes)}</span>
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
        'POST:/v1/chat/completions': { model, messages: [{ role: 'user', content: 'Hallo! Antworte kurz.' }], max_tokens: 48 },
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

    function renderAttachments() {
      $('attachmentsRow').innerHTML = state.attachments.map((src, i) => `
        <div class="att-item">
          <img src="${src}">
          <button class="att-remove" data-rm-att="${i}">✕</button>
        </div>
      `).join('');
    }

    // --- Event Listeners ---
    function toggleSidebar() {
      const collapsed = $('appShell').classList.toggle('sidebar-collapsed');
      ['btnExpandSidebarChat', 'btnExpandSidebarHub', 'btnExpandSidebarSettings'].forEach((id) => {
        $(id).style.display = collapsed ? 'inline-flex' : 'none';
      });
    }
    $('btnCollapseSidebar').addEventListener('click', toggleSidebar);
    $('btnExpandSidebarChat').addEventListener('click', toggleSidebar);
    $('btnExpandSidebarHub').addEventListener('click', toggleSidebar);
    $('btnExpandSidebarSettings').addEventListener('click', toggleSidebar);

    $('navNewChat').addEventListener('click', startNewChat);
    $('navHub').addEventListener('click', () => switchSurface('hub'));
    $('navSettings').addEventListener('click', () => switchSurface('settings'));
    $('btnQuickLocalServer').addEventListener('click', () => switchSurface('settings', 'local-api-server'));
    $('btnGenGoHub').addEventListener('click', () => switchSurface('hub'));
    $('btnModelMenuGoHub').addEventListener('click', () => {
      $('modelDropdownMenu').classList.remove('open');
      switchSurface('hub');
    });

    $('sidebarModeTabs').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-playmode]');
      if (!btn) return;
      state.playMode = btn.dataset.playmode;
      document.querySelectorAll('#sidebarModeTabs .nav-tab').forEach((b) => b.classList.toggle('active', b === btn));
      $('modeIndicatorBadge').textContent = state.playMode === 'infill' ? 'Code-Vervollstaendigung' : 'Chat';
      $('fimGrid').classList.toggle('active', state.playMode === 'infill');
      $('chatTextarea').style.display = state.playMode === 'infill' ? 'none' : 'block';
      switchSurface('chat');
    });

    $('settingsMenuNav').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-spane]');
      if (btn) switchSettingsPane(btn.dataset.spane);
    });

    // Model dropdown
    $('btnModelDropdown').addEventListener('click', (e) => {
      e.stopPropagation();
      $('modelDropdownMenu').classList.toggle('open');
    });
    $('modelDropdownItems').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-pick-model]');
      if (!btn) return;
      state.selectedModel = btn.dataset.pickModel;
      $('activeModelLabel').textContent = state.selectedModel;
      $('modelDropdownMenu').classList.remove('open');
      refreshStudioStatus();
    });

    // Popovers inside ChatInput
    $('btnToggleSamplerPop').addEventListener('click', (e) => {
      e.stopPropagation();
      $('assistantPopover').classList.remove('open');
      $('samplerPopover').classList.toggle('open');
      $('btnToggleSamplerPop').classList.toggle('active', $('samplerPopover').classList.contains('open'));
    });
    $('btnToggleAssistantPop').addEventListener('click', (e) => {
      e.stopPropagation();
      $('samplerPopover').classList.remove('open');
      $('assistantPopover').classList.toggle('open');
      $('btnToggleAssistantPop').classList.toggle('active', $('assistantPopover').classList.contains('open'));
    });
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#samplerPopover') && !e.target.closest('#btnToggleSamplerPop')) {
        $('samplerPopover').classList.remove('open');
        $('btnToggleSamplerPop').classList.remove('active');
      }
      if (!e.target.closest('#assistantPopover') && !e.target.closest('#btnToggleAssistantPop')) {
        $('assistantPopover').classList.remove('open');
        $('btnToggleAssistantPop').classList.remove('active');
      }
      if (!e.target.closest('#modelDropdownMenu') && !e.target.closest('#btnModelDropdown')) {
        $('modelDropdownMenu').classList.remove('open');
      }
    });

    $('selConstraint').addEventListener('change', () => {
      const v = $('selConstraint').value;
      $('txtConstraintSpec').style.display = (v === 'none' || v === 'json_object') ? 'none' : 'block';
    });

    $('btnToggleThinking').addEventListener('click', () => {
      state.thinkingEnabled = !state.thinkingEnabled;
      $('btnToggleThinking').classList.toggle('active', state.thinkingEnabled);
    });
    $('btnToggleTools').addEventListener('click', () => {
      state.toolsEnabled = !state.toolsEnabled;
      $('btnToggleTools').classList.toggle('active', state.toolsEnabled);
    });

    // Save Sampler to Rust
    const saveConfigToRust = async () => {
      const p = getSampler();
      const res = await apiFetch('/api/studio/config', {
        method: 'POST',
        body: {
          temperature: p.temperature, top_p: p.top_p, top_k: p.top_k, min_p: p.min_p,
          repetition_penalty: p.repetition_penalty, presence_penalty: p.presence_penalty,
          frequency_penalty: p.frequency_penalty, dry_multiplier: p.dry_multiplier,
          mirostat: p.mirostat, seed: p.seed < 0 ? null : p.seed,
          speculative: p.speculative, system_prompt: p.system_prompt,
        },
      });
      if (res.ok) {
        toast('Als Standard der Engine gespeichert');
        refreshStudioStatus();
      }
    };
    $('btnSaveSamplerToRust').addEventListener('click', saveConfigToRust);
    $('btnSaveSystemPromptRust').addEventListener('click', saveConfigToRust);
    $('btnResetSampler').addEventListener('click', () => {
      setSamplerUI({ temperature: 0.7, top_p: 0.95, top_k: 20, min_p: 0, repetition_penalty: 1.0, presence_penalty: 0, frequency_penalty: 0, dry_multiplier: 0, mirostat: 0, seed: -1, speculative: true });
    });

    // Chat Textarea & Live Token Counter
    let tokTimer = null;
    $('chatTextarea').addEventListener('input', () => {
      const el = $('chatTextarea');
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 200) + 'px';
      clearTimeout(tokTimer);
      tokTimer = setTimeout(async () => {
        const txt = el.value;
        const maxCtx = state.status?.model?.max_context || 512;
        if (!txt.trim()) {
          $('liveTokenCounter').textContent = `0 / ${maxCtx} tok`;
          return;
        }
        try {
          const r = await apiFetch('/tokenize', { method: 'POST', body: { content: txt } });
          if (r.ok) {
            const d = await r.json();
            $('liveTokenCounter').textContent = `${(d.tokens || []).length} / ${maxCtx} tok`;
          }
        } catch {}
      }, 220);
    });
    $('chatTextarea').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatPrompt();
      }
    });
    $('btnSendChat').addEventListener('click', sendChatPrompt);

    // Starter pills
    $('starterPills').addEventListener('click', (e) => {
      const pill = e.target.closest('.starter-pill');
      if (!pill) return;
      if (pill.dataset.prompt) {
        $('chatTextarea').value = pill.dataset.prompt;
        sendChatPrompt();
      } else if (pill.dataset.action === 'open-keys') {
        switchSurface('settings', 'local-api-server');
      } else if (pill.dataset.action === 'open-hub') {
        switchSurface('hub');
      }
    });

    // Thread list clicks
    $('threadList').addEventListener('click', (e) => {
      const del = e.target.closest('[data-del-tid]');
      if (del) {
        e.stopPropagation();
        state.threads = state.threads.filter((t) => t.id !== del.dataset.delTid);
        if (state.activeThreadId === del.dataset.delTid) state.activeThreadId = state.threads[0]?.id || null;
        saveThreads();
        renderThreadList();
        renderChatArea();
        return;
      }
      const row = e.target.closest('[data-tid]');
      if (row) {
        state.activeThreadId = row.dataset.tid;
        saveThreads();
        switchSurface('chat');
        renderThreadList();
        renderChatArea();
      }
    });

    // Messages actions
    $('messagesColumn').addEventListener('click', async (e) => {
      const copyCode = e.target.closest('[data-copy-code]');
      if (copyCode) {
        const pre = $(copyCode.dataset.copyCode);
        if (pre) { await navigator.clipboard.writeText(pre.textContent); toast('Code kopiert'); }
        return;
      }
      const copyMsg = e.target.closest('[data-msg-copy]');
      if (copyMsg) {
        const t = currentThread();
        const m = t?.messages?.[Number(copyMsg.dataset.msgCopy)];
        if (m) { await navigator.clipboard.writeText(m.content || ''); toast('Nachricht kopiert'); }
        return;
      }
      const delMsg = e.target.closest('[data-msg-del]');
      if (delMsg) {
        const t = currentThread();
        if (t) {
          t.messages.splice(Number(delMsg.dataset.msgDel), 1);
          saveThreads();
          renderChatArea();
        }
        return;
      }
      const regenMsg = e.target.closest('[data-msg-regen]');
      if (regenMsg) {
        const t = currentThread();
        const idx = Number(regenMsg.dataset.msgRegen);
        if (t && idx > 0) {
          const prevUser = t.messages[idx - 1];
          t.messages.splice(idx - 1);
          $('chatTextarea').value = prevUser?.content || '';
          sendChatPrompt();
        }
      }
    });

    $('btnClearCurrentChat').addEventListener('click', startNewChat);
    $('btnDeleteAllThreads').addEventListener('click', () => {
      state.threads = [];
      state.activeThreadId = null;
      saveThreads();
      renderThreadList();
      renderChatArea();
      toast('Alle Chats gelöscht');
    });
    $('btnFactoryResetThreads').addEventListener('click', () => $('btnDeleteAllThreads').click());
    $('btnExportThreads').addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(state.threads, null, 2)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `fllm-chats-${Date.now()}.json`;
      a.click();
    });

    // Image attachments
    $('btnAttachImg').addEventListener('click', () => $('fileImageInput').click());
    $('fileImageInput').addEventListener('change', (e) => {
      Array.from(e.target.files || []).forEach((f) => {
        const r = new FileReader();
        r.onload = () => { state.attachments.push(r.result); renderAttachments(); };
        r.readAsDataURL(f);
      });
      e.target.value = '';
    });
    $('attachmentsRow').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-rm-att]');
      if (btn) { state.attachments.splice(Number(btn.dataset.rmAtt), 1); renderAttachments(); }
    });

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

    // Search Modal (Ctrl+K)
    function openSearchModal() {
      $('searchModal').classList.add('open');
      $('searchModalInput').value = '';
      renderSearchResults('');
      $('searchModalInput').focus();
    }
    function renderSearchResults(q) {
      const query = q.toLowerCase().trim();
      const items = [
        ...state.threads.map((t) => ({ type: 'chat', title: t.title, id: t.id })),
        { type: 'nav', title: 'Settings → Local API Server & Keys', pane: 'local-api-server' },
        { type: 'nav', title: 'Settings → Hardware & SIMD Kernels', pane: 'hardware' },
        { type: 'nav', title: 'Settings → Appearance & Accent Colors', pane: 'appearance' },
        { type: 'nav', title: 'Settings → Tokenizer & Jinja Templates', pane: 'tokenizer' },
        { type: 'hub', title: 'Open Model Hub (Hugging Face & GGUF)', surface: 'hub' },
      ].filter((it) => !query || it.title.toLowerCase().includes(query));

      $('searchModalResults').innerHTML = items.map((it, i) => `
        <button class="nav-item" data-sres="${i}">
          <span>${escapeHtml(it.title)}</span>
          <span class="quant-chip" style="margin-left:auto;">${it.type}</span>
        </button>
      `).join('');
      $('searchModalResults')._items = items;
    }
    $('navOpenSearch').addEventListener('click', openSearchModal);
    $('searchModalInput').addEventListener('input', (e) => renderSearchResults(e.target.value));
    $('searchModal').addEventListener('click', (e) => {
      if (e.target === $('searchModal')) $('searchModal').classList.remove('open');
      const btn = e.target.closest('[data-sres]');
      if (btn) {
        const it = $('searchModalResults')._items[Number(btn.dataset.sres)];
        $('searchModal').classList.remove('open');
        if (it.type === 'chat') { state.activeThreadId = it.id; switchSurface('chat'); renderThreadList(); renderChatArea(); }
        else if (it.type === 'nav') switchSurface('settings', it.pane);
        else if (it.type === 'hub') switchSurface('hub');
      }
    });

    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearchModal(); }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'n') { e.preventDefault(); startNewChat(); }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') { e.preventDefault(); toggleSidebar(); }
      if (e.key === 'Escape') $('searchModal').classList.remove('open');
    });

    // Verhindert Flackern: DOM nur neu schreiben, wenn sich der HTML-Inhalt wirklich geaendert hat
    (function dedupeInnerHtml() {
      const d = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML');
      ['modelDropdownItems','providerModelFacts','providerAliasesList','apiKeysTableContainer','hwDevicesGrid','hwSlotsTableWrap','cachedModelsHubList','loadedModelHubItem','hubCatalogCards','threadList','messagesColumn'].forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        Object.defineProperty(el, 'innerHTML', {
          configurable: true,
          get() { return d.get.call(this); },
          set(v) { if (this._h !== v) { this._h = v; d.set.call(this, v); } },
        });
      });
    })();

    // --- Boot ---
    applyAppearance();
    loadThreads();
    renderThreadList();
    renderChatArea();
    updateRouteTesterSample();
    refreshStudioStatus();
    // on a phone the rail would cover the page: start with it closed
    if (window.matchMedia('(max-width: 760px)').matches) toggleSidebar();
    setInterval(refreshStudioStatus, 2000);
  })();
  
