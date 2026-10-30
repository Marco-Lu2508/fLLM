    // ==========================================================================
    // Command palette (Ctrl+K): commands, chats and settings pages in one list, reached by typing and the arrow keys.
    // ==========================================================================

    // @pure-begin
    /** Scores one text for a query: 0 = no match. Word starts count more than a hit inside a word; a spread-out match counts least. */
    function paletteScore(text, query) {
      const t = String(text || '').toLowerCase();
      const q = String(query || '').toLowerCase().trim();
      if (!q) return 1;
      if (t === q) return 120;
      if (t.startsWith(q)) return 100;
      const words = q.split(/\s+/).filter(Boolean);
      let total = 0;
      for (const w of words) {
        let s = 0;
        if (t.startsWith(w)) s = 90;
        else if (new RegExp(`(^|[^\\p{L}\\p{N}])${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'u').test(t)) s = 70;
        else if (t.includes(w)) s = 40;
        else if (w.length >= 3) {
          // letters in order, starting at the start of a word and close together ("kbd" finds "keyboard"); never scattered over the text
          for (let st = 0; st < t.length && !s; st++) {
            if (t[st] !== w[0] || (st > 0 && /[\p{L}\p{N}]/u.test(t[st - 1]))) continue;
            let i = 1;
            let at = st;
            while (i < w.length && at + 1 < t.length) { at++; if (t[at] === w[i]) i++; }
            if (i === w.length && at - st + 1 <= w.length * 3) s = 12;
          }
        }
        if (!s) return 0;
        total += s;
      }
      return Math.round(total / words.length);
    }

    /**
     * Orders palette items for a query. Items are { id, kind, title, sub?, keywords? } with kind "Command", "Chat" or "Setting".
     * Without a query the commands come first (the first few), then the chats in the order given. With a query every item whose title,
     * keywords or sub text matches is kept, best first; a title match beats a keyword match, which beats a text match.
     */
    function rankPalette(query, items, limit) {
      const q = String(query || '').trim();
      const cap = limit || 40;
      if (!q) {
        const commands = items.filter((i) => i.kind === 'Command').slice(0, 7);
        const chats = items.filter((i) => i.kind === 'Chat').slice(0, 6);
        return [...commands, ...chats].slice(0, cap);
      }
      const scored = [];
      items.forEach((item, order) => {
        const title = paletteScore(item.title, q);
        const kw = item.keywords ? paletteScore(item.keywords, q) : 0;
        const sub = item.sub ? paletteScore(item.sub, q) : 0;
        const score = item.weak ? 1 : Math.max(title * 3, kw * 2, sub);
        if (score) scored.push({ item, score, order });
      });
      scored.sort((a, b) => b.score - a.score || a.order - b.order);
      // one block per kind; the block with the best match comes first, a weak entry ("search the settings for ...") stays last in its block
      const best = new Map();
      for (const s of scored) if (!s.item.weak && !best.has(s.item.kind)) best.set(s.item.kind, s.score);
      const rank = (kind) => (best.has(kind) ? -best.get(kind) : 0);
      return scored.map((s, i) => ({ ...s, i })).sort((a, b) => rank(a.item.kind) - rank(b.item.kind) || (a.item.kind === b.item.kind ? a.i - b.i : String(a.item.kind).localeCompare(b.item.kind)))
        .slice(0, cap).map((s) => s.item);
    }

    /** The next active row for an arrow key: wraps around at both ends. */
    function paletteStep(index, count, key) {
      if (!count) return -1;
      if (key === 'Home') return 0;
      if (key === 'End') return count - 1;
      if (key === 'ArrowDown') return index < 0 ? 0 : (index + 1) % count;
      if (key === 'ArrowUp') return index <= 0 ? count - 1 : index - 1;
      return index;
    }
    // @pure-end

    (function commandPalette() {
      const modal = $('searchModal');
      const input = $('searchModalInput');
      const list = $('searchModalResults');
      let items = [];
      let active = -1;
      let sessionHits = [];
      let returnFocus = null;
      let stamp = 0;

      const key = (id) => (typeof shortcutLabel === 'function' ? shortcutLabel(id) : '');
      const goSettings = (pane) => () => switchSurface('settings', pane);

      /** The fixed commands plus the pages of the settings (read from the settings menu, so a new page shows up here by itself). */
      function commands() {
        const t = currentThread();
        const out = [
          { id: 'new', kind: 'Command', title: 'New chat', hint: key('newChat'), keywords: 'start conversation', run: () => SHORTCUT_ACTIONS.newChat() },
          { id: 'chat', kind: 'Command', title: 'Go to Chat', hint: key('surfaceChat'), run: () => surfaceTab('chat') },
          { id: 'code', kind: 'Command', title: 'Go to Code', hint: key('surfaceCode'), keywords: 'coding agent folder', run: () => surfaceTab('code') },
          { id: 'models', kind: 'Command', title: 'Open Models', hint: key('openModels'), keywords: 'download hugging face gguf catalog load', run: () => switchSurface('hub') },
          { id: 'settings', kind: 'Command', title: 'Open Settings', hint: key('openSettings'), run: () => switchSurface('settings') },
          { id: 'theme', kind: 'Command', title: document.documentElement.classList.contains('dark') ? 'Switch to the light theme' : 'Switch to the dark theme', keywords: 'appearance colors mode',
            run: () => { const sel = $('selThemeMode'); sel.value = document.documentElement.classList.contains('dark') ? 'light' : 'dark'; sel.dispatchEvent(new Event('change')); } },
          { id: 'sidebar', kind: 'Command', title: 'Show or hide the sidebar', hint: key('toggleSidebar'), run: toggleSidebar },
          { id: 'shortcuts', kind: 'Command', title: 'Keyboard shortcuts', hint: key('shortcuts'), keywords: 'keys hotkeys', run: openShortcuts },
          { id: 'tour', kind: 'Command', title: 'Show the welcome tour', run: () => $('btnShowTour').click() },
        ];
        if (t && state.surface === 'chat') {
          out.splice(1, 0,
            { id: 'regen', kind: 'Command', title: 'Regenerate the last answer', hint: key('regenerate'), keywords: 'again retry', run: regenerateLast },
            { id: 'export-md', kind: 'Command', title: 'Export this chat as Markdown', keywords: 'save download', run: () => chatAction('export-md') },
            { id: 'export-json', kind: 'Command', title: 'Export this chat as JSON', keywords: 'save download backup', run: () => chatAction('export-json') });
        }
        document.querySelectorAll('#settingsMenuNav [data-spane]').forEach((b) => {
          const label = b.textContent.trim();
          if (label) out.push({ id: `pane-${b.dataset.spane}`, kind: 'Setting', title: `Settings: ${label}`, run: goSettings(b.dataset.spane) });
        });
        return out;
      }

      function chatItems(query) {
        const q = query.toLowerCase().trim();
        return state.threads.slice().sort((a, b) => b.updatedAt - a.updatedAt).map((t) => {
          const inTitle = !q || t.title.toLowerCase().includes(q);
          const hit = !inTitle ? snippet(threadText(t), q) : '';
          return { id: `chat-${t.id}`, kind: 'Chat', title: t.title, sub: hit, noText: !hit, run: () => openThread(t.id), skip: !inTitle && !hit };
        }).filter((c) => !c.skip);
      }

      function build() {
        const q = input.value;
        const all = [...commands(), ...chatItems(q)];
        let ranked = rankPalette(q, all.map((i) => ({ ...i, sub: i.kind === 'Chat' ? '' : i.sub })), 40);
        if (q.trim()) {
          // chats are already filtered by their text; keep the ones the ranking dropped because only their text matched
          const seen = new Set(ranked.map((r) => r.id));
          ranked = ranked.concat(all.filter((i) => i.kind === 'Chat' && i.sub && !seen.has(i.id))).slice(0, 40);
          const seen2 = new Set(ranked.map((r) => r.id));
          ranked = ranked.concat(sessionHits.filter((h) => !seen2.has(h.id))).slice(0, 40);
          if (typeof window.fllmSettingsSearch === 'function') {
            ranked = rankPalette(q, [...ranked, { id: 'find-setting', kind: 'Setting', weak: true, title: `Search the settings for "${q.trim()}"`, run: () => window.fllmSettingsSearch(q.trim()) }], 41);
          }
        }
        items = ranked;
      }

      function render() {
        const labelOf = { Command: 'Commands', Chat: 'Chats', 'Code session': 'Code sessions', Setting: 'Settings' };
        const html = [];
        let lastKind = '';
        items.forEach((it, i) => {
          if (it.kind !== lastKind) { lastKind = it.kind; html.push(`<div class="palette-group" role="presentation">${labelOf[it.kind] || it.kind}</div>`); }
          html.push(`<div class="palette-item${i === active ? ' is-active' : ''}" role="option" id="pal-${i}" aria-selected="${i === active}" data-sres="${i}">` +
            `<span class="search-title">${escapeHtml(it.title)}${it.sub ? `<small>${escapeHtml(it.sub)}</small>` : ''}</span>` +
            (it.hint ? `<kbd class="palette-key">${escapeHtml(it.hint)}</kbd>` : '') + '</div>');
        });
        list.innerHTML = items.length ? html.join('') : `<div class="thread-empty" role="presentation">Nothing found for "${escapeHtml(input.value.trim())}".</div>`;
        list._items = items;
        input.setAttribute('aria-activedescendant', active >= 0 ? `pal-${active}` : '');
        $('searchModalCount').textContent = items.length ? `${items.length} result${items.length === 1 ? '' : 's'}` : 'No results';
        const row = active >= 0 ? $(`pal-${active}`) : null;
        if (row) row.scrollIntoView({ block: 'nearest' });
      }

      function refresh() {
        build();
        active = items.length ? 0 : -1;
        render();
        const q = input.value.trim();
        if (q.length >= 2 && window.fllmSessionSearch) {
          const mine = ++stamp;
          window.fllmSessionSearch(q.toLowerCase(), { limit: 8 }).then((hits) => {
            if (mine !== stamp || !modal.classList.contains('open')) return;
            sessionHits = (hits || []).map((h, n) => ({
              id: `hit-${n}-${h.threadId || h.title}`, kind: h.kind === 'code' ? 'Code session' : 'Chat', title: h.title || 'Untitled', sub: h.snippet || '',
              run: () => { if (h.threadId && state.threads.some((t) => t.id === h.threadId)) openThread(h.threadId); else if (h.kind === 'code') switchSurface('code'); else toast('This chat is only in the search index of the app, not in this browser.'); },
            }));
            const keep = active >= 0 && items[active] ? items[active].id : '';
            build();
            const i = items.findIndex((x) => x.id === keep);
            active = i >= 0 ? i : (items.length ? 0 : -1);
            render();
          }).catch(() => { /* the plain list stays */ });
        } else sessionHits = [];
      }

      function open() {
        returnFocus = document.activeElement;
        input.value = '';
        sessionHits = [];
        modal.classList.add('open');
        refresh();
        input.focus();
      }

      function run(i) {
        const it = items[i];
        if (!it) return;
        modal.classList.remove('open');
        it.run();
      }

      window.openSearchModal = open;
      new MutationObserver(() => {
        if (!modal.classList.contains('open') && returnFocus) {
          const target = returnFocus; returnFocus = null;
          // a command that moved the focus on purpose (the composer after "New chat") keeps it
          if (document.activeElement === document.body || modal.contains(document.activeElement)) { try { target.focus(); } catch { /* gone */ } }
        }
      }).observe(modal, { attributes: true, attributeFilter: ['class'] });

      $('navOpenSearch').addEventListener('click', open);
      input.addEventListener('input', refresh);
      input.addEventListener('keydown', (e) => {
        if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) {
          if ((e.key === 'Home' || e.key === 'End') && input.value) return; // the caret keys keep working in the text
          e.preventDefault();
          active = paletteStep(active, items.length, e.key);
          render();
        } else if (e.key === 'Enter') { e.preventDefault(); run(active); }
      });
      modal.addEventListener('click', (e) => {
        if (e.target === modal) { modal.classList.remove('open'); return; }
        const row = e.target.closest('[data-sres]');
        if (row) run(Number(row.dataset.sres));
      });
      list.addEventListener('mousemove', (e) => {
        const row = e.target.closest('[data-sres]');
        const i = row ? Number(row.dataset.sres) : -1;
        if (i >= 0 && i !== active) {
          active = i;
          list.querySelectorAll('.palette-item').forEach((r, n) => { r.classList.toggle('is-active', n === i); r.setAttribute('aria-selected', String(n === i)); });
          input.setAttribute('aria-activedescendant', `pal-${i}`);
        }
      });
    })();
