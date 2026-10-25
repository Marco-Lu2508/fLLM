    // ==========================================================================
    // "Desktop app" section of the settings. Only inside the desktop app (the Tauri window injects
    // __TAURI_INTERNALS__); in a normal browser this file does nothing.
    // ==========================================================================
    (function desktopSettings() {
      const tauri = window.__TAURI_INTERNALS__;
      if (!tauri || typeof tauri.invoke !== 'function') return;
      const nav = document.getElementById('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      if (!nav || !content) return;
      const call = (cmd, args) => tauri.invoke(cmd, args || {});

      const item = (title, desc, actions) =>
        `<div class="app-card-item"><div class="app-item-info"><div class="app-item-title">${title}</div>` +
        `<div class="app-item-desc">${desc}</div></div><div class="app-item-actions">${actions}</div></div>`;
      const toggle = (id) => `<label class="ui-switch"><input type="checkbox" id="${id}"><span class="ui-switch-track"></span></label>`;

      const button = document.createElement('button');
      button.className = 'settings-nav-btn';
      button.dataset.spane = 'desktop';
      button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><rect width="20" height="14" x="2" y="3" rx="2"/><path d="M8 21h8"/><path d="M12 17v4"/></svg><span>Desktop app</span>';
      nav.insertBefore(button, nav.firstElementChild ? nav.children[1] : null);

      const pane = document.createElement('div');
      pane.className = 'settings-pane';
      pane.id = 'spane-desktop';
      pane.innerHTML =
        '<div class="app-card"><div class="app-card-title">Desktop app</div>' +
        item('Keep running in the tray', 'Closing the window hides it instead of quitting. Quit from the tray icon.', toggle('chkDesktopTray')) +
        item('Start fLLM when I sign in', 'Starts the app when you sign in to this computer.', toggle('chkDesktopAutostart')) +
        item('Quick entry hotkey', 'Works from any program: brings fLLM to the front with a new chat. Click the field and press the keys; it needs Ctrl, Alt or Cmd. It does not work on some Linux desktops (Wayland).',
          '<input type="text" readonly class="ui-input mono" id="inpQuickEntry" style="width:170px;" placeholder="Off"> <button class="btn btn-sm btn-ghost" id="btnQuickEntryClear">Off</button> <button class="btn btn-sm btn-ghost" id="btnQuickEntryDefault">Default</button>') +
        item('Server log', '<span class="mono" id="desktopLogFolder"></span>', '<button class="btn btn-sm btn-secondary" id="btnDesktopLogs">Open log folder</button>') +
        item('Server', 'Stops the server and starts it again. The page reloads when it is ready.', '<button class="btn btn-sm btn-secondary" id="btnDesktopRestart">Restart server</button>') +
        '<div class="app-card-item"><div class="app-item-desc" id="desktopStatus"></div></div></div>';
      content.appendChild(pane);

      const status = (text) => { pane.querySelector('#desktopStatus').textContent = text || ''; };
      const fail = (e) => status(String((e && e.message) || e));
      const refresh = () => call('get_desktop_settings').then((s) => {
        pane.querySelector('#chkDesktopTray').checked = !!s.keepInTray;
        pane.querySelector('#chkDesktopAutostart').checked = !!s.autostart;
        pane.querySelector('#desktopLogFolder').textContent = s.logFolder;
        pane.querySelector('#inpQuickEntry').value = s.quickEntry || '';
        defaultQuickEntry = s.defaultQuickEntry || '';
      }).catch(fail);
      // quick entry: the field records a key combination in the form the app understands ("Ctrl+Alt+Space")
      let defaultQuickEntry = '';
      const setQuickEntry = (value) => call('set_quick_entry', { value }).then(() => { status(''); return refresh(); }).catch((e) => { fail(e); refresh(); });
      const hotkeyText = (e) => {
        if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return '';
        let key = e.key;
        if (/^Key[A-Z]$/.test(e.code || '')) key = e.code.slice(3);
        else if (/^Digit[0-9]$/.test(e.code || '')) key = e.code.slice(5);
        else if (key === ' ') key = 'Space';
        else if (key.length === 1) key = key.toUpperCase();
        const mods = [e.ctrlKey && 'Ctrl', e.altKey && 'Alt', e.shiftKey && 'Shift', e.metaKey && 'Super'].filter(Boolean);
        if (!e.ctrlKey && !e.altKey && !e.metaKey) { status('Use at least one of Ctrl, Alt or Cmd.'); return ''; }
        return [...mods, key].join('+');
      };
      pane.querySelector('#inpQuickEntry').addEventListener('keydown', (e) => {
        if (e.key === 'Tab') return;
        e.preventDefault();
        e.stopPropagation();
        const text = hotkeyText(e);
        if (text) setQuickEntry(text);
      });
      pane.querySelector('#btnQuickEntryClear').addEventListener('click', () => setQuickEntry(''));
      pane.querySelector('#btnQuickEntryDefault').addEventListener('click', () => setQuickEntry(defaultQuickEntry));

      pane.querySelector('#chkDesktopTray').addEventListener('change', (e) => {
        call('set_keep_in_tray', { value: e.target.checked }).then(() => status('')).catch(fail);
      });
      pane.querySelector('#chkDesktopAutostart').addEventListener('change', (e) => {
        call('set_autostart', { value: e.target.checked }).then(() => status('')).catch((err) => { fail(err); refresh(); });
      });
      pane.querySelector('#btnDesktopLogs').addEventListener('click', () => call('open_log_folder').catch(fail));
      pane.querySelector('#btnDesktopRestart').addEventListener('click', () => call('restart_server').catch(fail));
      button.addEventListener('click', refresh);
      refresh();
    })();

