    // ==========================================================================
    // Browser: the live view of the page the coding agent works with, on the Code page, and Settings > Browser.
    //  - the server drives a real Chromium-family browser over the DevTools protocol (see docs/user/studio.md, "The browser")
    //  - this panel shows the page as a stream of screenshots, forwards the user's clicks, typing and scrolling, and lets the
    //    user comment on elements: the comments (with the ref of the element) go with the next message to the agent
    //  - nothing here starts a browser by itself; it starts when the agent opens a page or the user types an address
    // ==========================================================================

    (function browserPanel() {
      const panel = document.getElementById('codeBrowser');
      const toggle = document.getElementById('btnCodeBrowser');
      if (!panel || !toggle) return;

      const ICONS = {
        back: '<path d="m15 18-6-6 6-6"/>',
        forward: '<path d="m9 18 6-6-6-6"/>',
        reload: '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>',
        comment: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
        stop: '<rect width="12" height="12" x="6" y="6" rx="1.5"/>',
        hide: '<path d="m18 6-12 12"/><path d="m6 6 12 12"/>',
      };
      const icon = (name) => `<svg viewBox="0 0 24 24" class="icon icon-sm">${ICONS[name]}</svg>`;

      const br = {
        open: false,
        status: null,
        running: false,
        key: '',
        view: { width: 1280, height: 800 },
        url: '',
        title: '',
        commenting: false,
        comments: [], // { ref, role, name, tag, text, url, note }
        draft: null, // the element being commented on
        timer: 0,
        ticks: 0,
        wheel: { dx: 0, dy: 0, x: 0, y: 0, timer: 0 },
        pending: false,
      };

      panel.innerHTML =
        '<div class="br-head"><strong>Browser</strong><span class="br-state" id="brState"></span><span class="br-spacer"></span>' +
        `<button class="btn-icon-xs" id="brComment" title="Comment on an element of the page" aria-pressed="false">${icon('comment')}</button>` +
        `<button class="btn-icon-xs" id="brStop" title="Close the browser (the logins stay in its profile)">${icon('stop')}</button>` +
        `<button class="btn-icon-xs" id="brHide" title="Hide this panel">${icon('hide')}</button></div>` +
        '<form class="br-bar" id="brBar">' +
        `<button type="button" class="btn-icon-xs" id="brBack" title="Back">${icon('back')}</button>` +
        `<button type="button" class="btn-icon-xs" id="brForward" title="Forward">${icon('forward')}</button>` +
        `<button type="button" class="btn-icon-xs" id="brReload" title="Reload">${icon('reload')}</button>` +
        '<input type="text" class="ui-input" id="brUrl" placeholder="Address, for example localhost:3000" spellcheck="false" autocomplete="off">' +
        '<button type="submit" class="btn btn-sm btn-secondary">Go</button></form>' +
        '<div class="br-stage" id="brStage" tabindex="0" role="application" aria-label="Live view of the browser page. Click, type and scroll to use it."><img id="brImg" alt="" hidden><div class="br-pin" id="brPin" hidden></div><div class="br-empty" id="brEmpty"></div></div>' +
        '<div class="br-note" id="brNote" hidden></div>' +
        '<div class="br-comments" id="brComments"></div>' +
        '<p class="br-foot" id="brFoot"></p>';

      const el = (id) => document.getElementById(id);
      const stage = el('brStage');
      const img = el('brImg');
      const urlInput = el('brUrl');

      const post = (path, body) => apiFetch(`/api/studio/agent/browser${path}`, { method: 'POST', body: body || {} }).then((r) => r.json().catch(() => ({})));

      // ---- what the panel says when there is nothing to show ----
      function installHelp() {
        const p = navigator.platform || '';
        if (/Win/i.test(p)) return 'Windows has Microsoft Edge built in; if it is missing, install Google Chrome.';
        if (/Mac/i.test(p)) return 'Install Google Chrome (or Microsoft Edge, Brave, Chromium) from its website.';
        return 'Install Chromium or Google Chrome with your package manager (for example "apt install chromium"), or name a browser program in Settings > Browser.';
      }

      function emptyHtml() {
        const s = br.status;
        if (!s) return 'Looking for the browser ...';
        if (!s.settings.enabled) {
          return 'The browser is switched off. When it is on, the agent can open pages, read them, click and type, and you see what it sees here and can take over.<button class="btn btn-sm btn-primary" id="brSettings">Open Browser settings</button>';
        }
        if (!s.browser) {
          return `No Chromium-based browser was found on this computer. ${installHelp()}<button class="btn btn-sm btn-secondary" id="brSettings">Open Browser settings</button>`;
        }
        if (br.running && br.undrawable) return `The browser shows a page that cannot be drawn here, probably an error page (the site is not reachable). Address: ${escapeHtml(String(br.undrawable))}. The address bar and the agent still work.`;
        return 'Nothing is open yet. The agent starts the browser when it opens a page; you can also type an address above.';
      }

      function renderState() {
        const s = br.status;
        const state = !s ? '' : !s.settings.enabled ? 'off' : br.running ? (s.settings.show_window ? 'running, window shown' : 'running') : 'ready';
        el('brState').textContent = state;
        toggle.querySelector('.br-live').hidden = !br.running;
        const showImage = br.running && img.getAttribute('src');
        img.hidden = !showImage;
        const empty = el('brEmpty');
        empty.hidden = Boolean(showImage);
        if (!showImage) empty.innerHTML = emptyHtml();
        for (const id of ['brBack', 'brForward', 'brReload', 'brStop']) el(id).disabled = !br.running;
        el('brComment').disabled = !br.running;
        urlInput.disabled = !s || !s.settings.enabled;
        el('brFoot').textContent = br.commenting
          ? 'Comment mode: click an element on the page, then write what the agent should do about it.'
          : br.running
            ? 'Click, type and scroll in the page to use it yourself. The agent sees the result.'
            : '';
        stage.classList.toggle('br-commenting', br.commenting);
        el('brComment').setAttribute('aria-pressed', br.commenting ? 'true' : 'false');
      }

      function renderComments() {
        const box = el('brComments');
        if (!br.comments.length) { box.innerHTML = ''; return; }
        box.innerHTML = `<h5>Comments for the agent (${br.comments.length})</h5>` + br.comments.map((c, i) =>
          `<div class="br-chip"><span><span class="mono">${escapeHtml(c.role)}${c.name ? ` "${escapeHtml(c.name.slice(0, 40))}"` : ''}</span><br>${escapeHtml(c.note)}</span><button class="btn-icon-xs" data-br-del="${i}" title="Remove this comment">${icon('hide')}</button></div>`
        ).join('') + '<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;"><button class="btn btn-sm btn-primary" id="brSend">Send to the agent</button><span class="br-foot">They are also added to your next message.</span></div>';
      }

      function renderNote() {
        const note = el('brNote');
        if (!br.draft) { note.hidden = true; note.innerHTML = ''; el('brPin').hidden = true; return; }
        const d = br.draft;
        note.hidden = false;
        note.innerHTML = `<div><strong>${escapeHtml(d.role)}</strong>${d.name ? ` "${escapeHtml(d.name.slice(0, 60))}"` : ''} <span class="br-foot">ref ${d.ref}, ${escapeHtml(d.tag || '')}</span></div>` +
          (d.text ? `<div class="br-foot">${escapeHtml(String(d.text).replace(/\s+/g, ' ').slice(0, 120))}</div>` : '') +
          '<textarea class="ui-textarea" id="brNoteText" placeholder="What should be changed or checked here?"></textarea>' +
          '<div class="br-note-actions"><button class="btn btn-sm btn-ghost" id="brNoteCancel">Cancel</button><button class="btn btn-sm btn-primary" id="brNoteAdd">Add comment</button></div>';
        const r = d.rect || {};
        const pin = el('brPin');
        const k = img.clientWidth / br.view.width;
        if (r.w > 0 && r.h > 0 && k > 0) {
          pin.hidden = false;
          Object.assign(pin.style, { left: `${r.x * k}px`, top: `${r.y * k}px`, width: `${r.w * k}px`, height: `${r.h * k}px` });
        }
        setTimeout(() => el('brNoteText')?.focus(), 0);
      }

      // ---- talking to the server ----
      async function loadStatus() {
        try {
          const res = await apiFetch('/api/studio/agent/browser');
          if (!res.ok) { br.status = null; return; }
          br.status = await res.json();
          br.running = Boolean(br.status.running);
          if (br.status.url) br.url = br.status.url;
        } catch { /* the server is not there: the next poll asks again */ }
        renderState();
      }

      async function pollFrame() {
        if (br.pending) return;
        br.pending = true;
        try {
          const data = await post('/frame', { since: br.key });
          br.running = Boolean(data.running);
          if (!br.running) { br.key = ''; img.removeAttribute('src'); }
          else if (data.undrawable) {
            br.undrawable = data.url || true; br.key = ''; img.removeAttribute('src');
            br.url = data.url || br.url;
            if (document.activeElement !== urlInput) urlInput.value = br.url;
          } else if (!data.error) {
            br.undrawable = false;
            if (data.image) { img.src = data.image; br.key = data.key; }
            if (data.width) br.view = { width: data.width, height: data.height || br.view.height };
            br.url = data.url || br.url;
            br.title = data.title || '';
            if (document.activeElement !== urlInput) urlInput.value = br.url === 'about:blank' ? '' : br.url;
          }
          if (br.draft) renderNote();
          renderState();
        } catch { /* the server is busy; try again */ } finally { br.pending = false; }
      }

      function visible() {
        const surface = document.getElementById('surface-code');
        return br.open && !document.hidden && surface && surface.classList.contains('active');
      }

      function schedule(ms) {
        clearTimeout(br.timer);
        br.timer = setTimeout(tick, ms);
      }

      async function tick() {
        if (!visible()) { schedule(1500); return; }
        if (!br.status || br.ticks++ % 12 === 0) await loadStatus();
        if (br.status && br.status.settings.enabled) await pollFrame();
        schedule(br.running ? 650 : 2000);
      }

      function poke() {
        if (br.open) { schedule(0); return; }
        loadStatus();
      }

      // one input after the other, so that fast typing arrives in the order it was typed
      let inputChain = Promise.resolve();
      function sendInput(body) {
        const run = async () => {
          try {
            const data = await post('/input', body);
            if (data && data.ok === false && data.error) toast(data.error);
            setTimeout(poke, 120);
            return data;
          } catch (e) {
            toast(`The browser did not answer: ${(e && e.message) || e}`);
            return null;
          }
        };
        const next = inputChain.then(run, run);
        inputChain = next.then(() => {}, () => {});
        return next;
      }

      // ---- open and close the panel ----
      function setOpen(open) {
        br.open = open;
        panel.hidden = !open;
        toggle.setAttribute('aria-pressed', open ? 'true' : 'false');
        if (open) { loadStatus().then(() => schedule(0)); }
        else clearTimeout(br.timer);
      }
      toggle.addEventListener('click', () => setOpen(!br.open));
      el('brHide').addEventListener('click', () => setOpen(false));
      el('brStop').addEventListener('click', async () => {
        if (!await confirmDialog('Close the browser? Pages that are open are lost; logins stay in its profile.')) return;
        await post('/input', { type: 'close' });
        br.running = false; br.key = ''; img.removeAttribute('src');
        loadStatus();
      });

      // ---- the address bar and the buttons ----
      el('brBar').addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = urlInput.value.trim();
        if (!text) return;
        await sendInput({ type: 'navigate', url: text });
        loadStatus();
      });
      el('brBack').addEventListener('click', () => sendInput({ type: 'back' }));
      el('brForward').addEventListener('click', () => sendInput({ type: 'forward' }));
      el('brReload').addEventListener('click', () => sendInput({ type: 'reload' }));

      // ---- the user uses the page ----
      /** A point of the preview as a point of the page. */
      function pagePoint(e) {
        const r = img.getBoundingClientRect();
        if (!r.width) return null;
        return { x: ((e.clientX - r.left) / r.width) * br.view.width, y: ((e.clientY - r.top) / r.height) * br.view.height };
      }

      stage.addEventListener('focus', () => stage.classList.add('br-focus'));
      stage.addEventListener('blur', () => stage.classList.remove('br-focus'));

      stage.addEventListener('click', async (e) => {
        if (e.target.closest('button')) return;
        if (!br.running || img.hidden) return;
        const p = pagePoint(e);
        if (!p) return;
        stage.focus();
        if (br.commenting) {
          const data = await sendInput({ type: 'locate', x: p.x, y: p.y });
          if (data && data.ok) { br.draft = data; renderNote(); }
          return;
        }
        sendInput({ type: 'click', x: p.x, y: p.y });
      });
      stage.addEventListener('dblclick', (e) => {
        if (br.commenting || !br.running || img.hidden) return;
        const p = pagePoint(e);
        if (p) sendInput({ type: 'dblclick', x: p.x, y: p.y });
      });

      stage.addEventListener('wheel', (e) => {
        if (!br.running || img.hidden) return;
        e.preventDefault();
        const p = pagePoint(e);
        if (!p) return;
        const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? br.view.height : 1;
        const w = br.wheel;
        w.dx += e.deltaX * unit; w.dy += e.deltaY * unit; w.x = p.x; w.y = p.y;
        if (!w.timer) {
          w.timer = setTimeout(() => {
            const send = { type: 'wheel', x: w.x, y: w.y, dx: w.dx, dy: w.dy };
            w.dx = 0; w.dy = 0; w.timer = 0;
            sendInput(send);
          }, 50);
        }
      }, { passive: false });

      const KEY_NAMES = new Set(['Enter', 'Tab', 'Escape', 'Backspace', 'Delete', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown', 'Insert']);
      stage.addEventListener('keydown', (e) => {
        if (!br.running || e.isComposing || ['Control', 'Shift', 'Alt', 'Meta', 'Dead', 'Process', 'CapsLock'].includes(e.key)) return;
        // the paste shortcut is handled by the paste event, which has the text
        if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'v') return;
        const single = [...e.key].length === 1;
        if (!single && !KEY_NAMES.has(e.key) && !/^F([1-9]|1[0-2])$/.test(e.key)) return;
        const mods = [];
        if (e.ctrlKey) mods.push('Control');
        if (e.altKey) mods.push('Alt');
        if (e.metaKey) mods.push('Meta');
        if (e.shiftKey && (!single || mods.length)) mods.push('Shift');
        const name = e.key === ' ' ? 'Space' : e.key;
        e.preventDefault();
        sendInput({ type: 'key', key: [...mods, name].join('+') });
      });
      stage.addEventListener('paste', (e) => {
        const text = e.clipboardData && e.clipboardData.getData('text/plain');
        if (!text || !br.running) return;
        e.preventDefault();
        sendInput({ type: 'text', text });
      });

      // ---- comments ----
      el('brComment').addEventListener('click', () => {
        br.commenting = !br.commenting;
        if (!br.commenting) { br.draft = null; renderNote(); }
        renderState();
      });
      panel.addEventListener('click', (e) => {
        const del = e.target.closest('[data-br-del]');
        if (del) { br.comments.splice(Number(del.dataset.brDel), 1); renderComments(); return; }
        if (e.target.closest('#brNoteCancel')) { br.draft = null; renderNote(); return; }
        if (e.target.closest('#brNoteAdd')) {
          const note = (el('brNoteText')?.value || '').trim();
          if (!note) { el('brNoteText')?.focus(); return; }
          const d = br.draft;
          br.comments.push({ ref: d.ref, role: d.role, name: d.name || '', tag: d.tag || '', text: String(d.text || '').replace(/\s+/g, ' ').slice(0, 160), url: d.url || br.url, note });
          br.draft = null;
          renderNote();
          renderComments();
          return;
        }
        if (e.target.closest('#brSend')) {
          const box = document.getElementById('codeTextarea');
          const send = document.getElementById('btnCodeSend');
          if (box && send) {
            if (!box.value.trim()) box.value = 'Please act on my comments on the browser page.';
            send.click();
          }
          return;
        }
        if (e.target.closest('#brSettings')) {
          document.getElementById('navSettings')?.click();
          setTimeout(() => document.querySelector('.settings-nav-btn[data-spane="browser"]')?.click(), 50);
        }
      });
      panel.addEventListener('keydown', (e) => {
        if (e.target.id === 'brNoteText' && e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); el('brNoteAdd')?.click(); }
        if (e.target.id === 'brNoteText' && e.key === 'Escape') { e.preventDefault(); el('brNoteCancel')?.click(); }
      });

      /** The comments as text for the agent; they are sent once. */
      window.fllmBrowserTakeComments = () => {
        if (!br.comments.length) return '';
        const lines = br.comments.map((c, i) => `${i + 1}. ${c.role}${c.name ? ` "${c.name}"` : ''} <${c.tag || 'element'}> (ref ${c.ref}${c.text && c.text !== c.name ? `, text: "${c.text}"` : ''}) on ${c.url || 'the page'}: ${c.note}`);
        br.comments = [];
        renderComments();
        return `[The user pointed at elements of the browser page and left comments. Act on them. The refs belong to the page that was open then; take a new browser_snapshot if a ref no longer works.]\n${lines.join('\n')}\n\n`;
      };
      window.fllmBrowserUrl = () => br.url || (br.status && br.status.url) || '';
      window.fllmBrowserPoke = poke;
      window.fllmBrowserChanged = () => { loadStatus(); if (typeof window.fllmReloadTools === 'function') window.fllmReloadTools(); };

      renderState();
      renderComments();
      loadStatus();
    })();

    // ==========================================================================
    // Settings > Browser
    // ==========================================================================
    (function browserSettings() {
      const nav = document.getElementById('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      if (!nav || !content) return;

      const button = document.createElement('button');
      button.className = 'settings-nav-btn';
      button.dataset.spane = 'browser';
      button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a14 14 0 0 1 0 18 14 14 0 0 1 0-18z"/></svg><span>Browser</span>';
      nav.appendChild(button);

      const pane = document.createElement('div');
      pane.className = 'settings-pane';
      pane.id = 'spane-browser';
      pane.innerHTML =
        '<div class="app-card"><div class="app-card-title">Browser for the coding agent</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;">With this on, the agent on the Code page can open a web page in a real browser (Chrome, Chromium, Edge or Brave on this computer), read it, click, type, scroll and take screenshots, for example to test the app it is building on a local dev server. You see the same page live in the Browser panel of the Code page and can take over, or leave comments on elements. Every browser action asks first, like the other network tools. It is off until you turn it on.</div>' +
        '<label class="check-row"><input type="checkbox" id="brsEnabled"> Let the agent use a browser</label>' +
        '<div class="app-item-desc mono" id="brsFound" style="margin-top:8px;font-size:11px;"></div>' +
        '</div>' +
        '<div class="app-card" style="margin-top:12px;"><div class="app-card-title">How it runs</div>' +
        '<label class="check-row"><input type="checkbox" id="brsShow"> Show the browser window</label>' +
        '<div class="app-item-desc" style="margin:4px 0 10px 24px;">Off: the browser runs without a window (headless) and you watch it in the panel. On: a real window opens on this computer, for example to log in to a site once. Needs a display (not for servers and containers).</div>' +
        '<label class="check-row"><input type="checkbox" id="brsKeep"> Keep logins and cookies between sessions</label>' +
        '<div class="app-item-desc" style="margin:4px 0 10px 24px;">The browser uses its own profile folder inside fLLM\'s data folder, never the profile of your own browser. Off: every start uses a fresh, throw-away profile.</div>' +
        '<label class="rules-label" for="brsLocal">Local addresses</label>' +
        '<select class="ui-select" id="brsLocal"><option value="auto">Only localhost, while a workspace is open (default)</option><option value="off">None</option><option value="all">Localhost and the local network (10.x, 192.168.x ...)</option></select>' +
        '<div class="app-item-desc" style="margin-top:4px;">Addresses on this computer or your network can hold things that must not be reached by accident, so they are refused unless allowed here. Link-local and cloud-metadata addresses (169.254.x.x) are never reachable. A page from the internet can never load local addresses, whatever you choose.</div>' +
        '<label class="rules-label" for="brsExe" style="margin-top:10px;display:block;">Browser program (optional)</label>' +
        '<input type="text" class="ui-input mono" id="brsExe" placeholder="Leave empty to look for Chrome, Edge, Brave or Chromium" spellcheck="false">' +
        '<div class="app-item-desc" style="margin-top:4px;">A path or a program name. The environment variable FLLM_BROWSER wins over this field.</div>' +
        '<div style="margin-top:10px;display:flex;gap:8px;align-items:center;"><button class="btn btn-sm btn-primary" id="brsSave">Save</button><span class="app-item-desc mono" id="brsMsg" style="font-size:11px;"></span></div>' +
        '</div>' +
        '<div class="app-card" style="margin-top:12px;"><div class="app-card-title">Safety</div>' +
        '<div class="app-item-desc">Only http and https pages can be opened (no files, no browser-internal pages). Downloads, camera, microphone, location and notifications are refused. The agent\'s browser can hold logged-in sessions, so its controls are only reachable from this computer, even when the server has an API key. If you log in to a site in the shown window, the agent can use that session: only do it for sites you are willing to let the agent act on.</div>' +
        '<div class="app-item-desc" id="brsProfile" style="margin-top:6px;font-size:11px;"></div></div>';
      content.appendChild(pane);

      const q = (id) => pane.querySelector(id);
      const say = (t) => { q('#brsMsg').textContent = t || ''; };

      function show(data) {
        const s = data.settings;
        q('#brsEnabled').checked = s.enabled;
        q('#brsShow').checked = s.show_window;
        q('#brsKeep').checked = s.persistent_profile;
        q('#brsLocal').value = s.local_addresses;
        q('#brsExe').value = s.executable;
        const found = q('#brsFound');
        if (data.browser) found.textContent = `Browser found: ${data.browser}${data.env_browser ? ' (from FLLM_BROWSER)' : ''}`;
        else found.textContent = `No browser found. ${data.browser_error || ''}`;
        found.style.color = data.browser ? '' : 'var(--destructive,#d33)';
        q('#brsProfile').textContent = `Profile folder: ${data.profile}`;
      }

      async function load() {
        try {
          const res = await apiFetch('/api/studio/agent/browser');
          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            q('#brsFound').textContent = (err.error && err.error.message) || 'The browser settings are available once a model is loaded (the coding agent starts with a model).';
            return;
          }
          show(await res.json());
        } catch (e) {
          q('#brsFound').textContent = `Could not read the settings: ${(e && e.message) || e}`;
        }
      }

      async function save(body, quiet) {
        try {
          const res = await apiFetch('/api/studio/agent/browser/settings', { method: 'POST', body });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { say((data.error && data.error.message) || 'Could not save it.'); return false; }
          show(data);
          if (!quiet) say('Saved.');
          if (typeof window.fllmBrowserChanged === 'function') window.fllmBrowserChanged();
          return true;
        } catch (e) {
          say(`Could not save it: ${(e && e.message) || e}`);
          return false;
        }
      }

      q('#brsEnabled').addEventListener('change', async (e) => {
        if (e.target.checked && !await confirmDialog('Let the agent open web pages in a real browser on this computer? Each browser action still asks first. Pages are limited to http and https; local addresses follow the setting below.')) {
          e.target.checked = false;
          return;
        }
        save({ enabled: e.target.checked });
      });
      q('#brsSave').addEventListener('click', () => save({
        show_window: q('#brsShow').checked,
        persistent_profile: q('#brsKeep').checked,
        local_addresses: q('#brsLocal').value,
        executable: q('#brsExe').value.trim(),
      }));
      button.addEventListener('click', load);
      load();
    })();
