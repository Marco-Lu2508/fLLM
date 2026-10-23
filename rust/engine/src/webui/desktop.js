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
      }).catch(fail);

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

