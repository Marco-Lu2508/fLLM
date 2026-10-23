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

