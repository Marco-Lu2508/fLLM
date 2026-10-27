    // Accent palette
    // One accent per personality; the first is copper — the colour of the material (Rust, patina),
    // not the acid green every AI tool reaches for.
    const ACCENT_COLORS = [
      { name: 'Copper',  value: 'patina',  thumb: '#e08a4b', primary: '#e08a4b', primaryLight: '#b4541f', sidebar: { light: '#ffffff', dark: '#12151c' } },
      { name: 'Signal',  value: 'emerald', thumb: '#4ade80', primary: '#4ade80', primaryLight: '#15803d', sidebar: { light: '#ffffff', dark: '#12151c' } },
      { name: 'Indigo',  value: 'indigo',  thumb: '#8b8cf9', primary: '#8b8cf9', primaryLight: '#4338ca', sidebar: { light: '#ffffff', dark: '#12151c' } },
      { name: 'Cyan',    value: 'cyan',    thumb: '#22d3ee', primary: '#22d3ee', primaryLight: '#0e7490', sidebar: { light: '#ffffff', dark: '#12151c' } },
      { name: 'Rose',    value: 'rose',    thumb: '#fb7185', primary: '#fb7185', primaryLight: '#be123c', sidebar: { light: '#ffffff', dark: '#12151c' } },
      { name: 'Violet', value: 'purple',  thumb: '#c084fc', primary: '#c084fc', primaryLight: '#7e22ce', sidebar: { light: '#ffffff', dark: '#12151c' } },
      { name: 'Steel',   value: 'blue',    thumb: '#94a3b8', primary: '#94a3b8', primaryLight: '#334155', sidebar: { light: '#ffffff', dark: '#12151c' } },
      { name: 'Lime',    value: 'green',   thumb: '#a3e635', primary: '#a3e635', primaryLight: '#4d7c0f', sidebar: { light: '#ffffff', dark: '#12151c' } },
    ];

    const STORAGE = {
      threads: 'fllm.studio.threads.v3',
      activeThread: 'fllm.studio.activeThread.v3',
      autoTitle: 'fllm.studio.autoTitle.v1',
      theme: 'fllm.studio.theme.v2',
      accent: 'fllm.studio.accent.v2',
      fontSize: 'fllm.studio.fontSize.v2',
      coloredBubble: 'fllm.studio.coloredBubble.v2',
      showSpeed: 'fllm.studio.showSpeed.v2',
      apiKey: 'fllm.studio.apiKey.v2',
    };

    const state = {
      surface: 'chat', // 'chat' | 'code' | 'hub' | 'settings'
      tab: 'chat', // the tab of the rail: 'chat' | 'code'
      thinkingEnabled: false,
      preset: null,
      autoTitle: localStorage.getItem('fllm.studio.autoTitle.v1') !== 'false',
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
      // the answers describe the state of the app right now: never take them from the cache of the browser
      return fetch(path, Object.assign({ cache: 'no-store' }, opts, { headers }));
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
      $('chkAutoTitle').checked = state.autoTitle;

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
      if (surfaceName === 'chat' || surfaceName === 'code') state.tab = surfaceName;
      ['chat', 'code', 'hub', 'settings'].forEach((s) => {
        $(`surface-${s}`).classList.toggle('active', s === surfaceName);
      });
      document.querySelectorAll('#sidebarModeTabs .nav-tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === state.tab));
      $('navHub').classList.toggle('active', surfaceName === 'hub');
      $('navSettings').classList.toggle('active', surfaceName === 'settings');
      $('appShell').classList.toggle('tab-code', state.tab === 'code');
      $('threadGroupTitle').textContent = state.tab === 'code' ? 'Folders' : 'Chats';
      renderThreadList();

      if (surfaceName === 'code') { loadWorkspace(); setTimeout(() => $('codeTextarea').focus(), 30); }
      if (surfaceName === 'chat') setTimeout(() => $('chatTextarea').focus(), 30);
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

