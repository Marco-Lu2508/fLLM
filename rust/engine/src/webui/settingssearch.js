    // ==========================================================================
    // Settings search: a box above the list of settings pages. It keeps the pages that mention every word,
    // best match first, and jumps to the matching row when a page is opened from the results.
    // ==========================================================================

    // @pure-begin
    /** Splits a query into lower-case words. */
    function searchWords(query) {
      return String(query || '').toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);
    }

    /**
     * Ranks pages for a query. `pages` is [{ id, label, text }]; every word must occur in the label or the text.
     * A word in the label counts most, a word at the start of a word in the text counts more than one inside a word.
     * Returns [{ id, score }] best first, ties in the original order.
     */
    function rankSettingsPages(query, pages) {
      const words = searchWords(query);
      if (!words.length) return pages.map((p, i) => ({ id: p.id, score: 0, order: i }));
      const out = [];
      pages.forEach((p, order) => {
        const label = String(p.label || '').toLowerCase();
        const text = String(p.text || '').toLowerCase();
        let score = 0;
        for (const w of words) {
          let s = 0;
          if (label.startsWith(w)) s = 100;
          else if (label.includes(w)) s = 60;
          else if (text.includes(` ${w}`) || text.startsWith(w)) s = 10;
          else if (text.includes(w)) s = 4;
          if (!s) { score = 0; break; }
          score += s;
        }
        if (score) out.push({ id: p.id, score, order });
      });
      return out.sort((a, b) => b.score - a.score || a.order - b.order);
    }
    // @pure-end

    (function settingsSearch() {
      const nav = document.getElementById('settingsMenuNav');
      if (!nav) return;
      const box = document.createElement('div');
      box.className = 'settings-search';
      box.innerHTML = '<input type="search" id="settingsSearch" class="ui-input" placeholder="Search settings" aria-label="Search settings" autocomplete="off" spellcheck="false">' +
        '<div class="settings-search-empty" id="settingsSearchEmpty" hidden>No setting matches.</div>';
      nav.insertBefore(box, nav.firstChild);
      const input = box.querySelector('input');
      const empty = box.querySelector('#settingsSearchEmpty');
      let pages = [];
      let lastQuery = '';

      function buildIndex() {
        pages = Array.from(nav.querySelectorAll('[data-spane]')).map((btn) => {
          const pane = document.getElementById(`spane-${btn.dataset.spane}`);
          return { id: btn.dataset.spane, label: btn.textContent.trim(), text: pane ? pane.textContent.replace(/\s+/g, ' ') : '' };
        });
      }

      function apply() {
        const q = input.value;
        lastQuery = q;
        const ranked = rankSettingsPages(q, pages);
        const rank = new Map(ranked.map((r, i) => [r.id, i]));
        const searching = searchWords(q).length > 0;
        nav.querySelectorAll('[data-spane]').forEach((btn) => {
          btn.hidden = searching && !rank.has(btn.dataset.spane);
          btn.style.order = searching && rank.has(btn.dataset.spane) ? String(rank.get(btn.dataset.spane)) : '';
        });
        nav.querySelectorAll('.settings-section-label').forEach((l) => { l.hidden = searching; });
        box.style.order = '-1';
        empty.hidden = !searching || ranked.length > 0;
      }

      /** Marks the first row of the open page that mentions the query, so the eye finds it. */
      function highlight(paneId) {
        const words = searchWords(lastQuery);
        const pane = document.getElementById(`spane-${paneId}`);
        if (!words.length || !pane) return;
        pane.querySelectorAll('.search-hit').forEach((e) => e.classList.remove('search-hit'));
        const rows = pane.querySelectorAll('.app-card-item, .app-card-title, .app-card > p, label');
        const hit = Array.from(rows).find((r) => { const t = r.textContent.toLowerCase(); return words.every((w) => t.includes(w)); })
          || Array.from(rows).find((r) => words.some((w) => r.textContent.toLowerCase().includes(w)));
        if (!hit) return;
        hit.classList.add('search-hit');
        hit.scrollIntoView({ block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
        setTimeout(() => hit.classList.remove('search-hit'), 3200);
      }

      input.addEventListener('focus', buildIndex);
      input.addEventListener('input', () => { if (!pages.length) buildIndex(); apply(); });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') { input.value = ''; apply(); return; }
        if (e.key === 'Enter') {
          const first = rankSettingsPages(input.value, pages)[0];
          if (first) { switchSettingsPane(first.id); highlight(first.id); }
        }
        if (e.key === 'ArrowDown') {
          const first = Array.from(nav.querySelectorAll('[data-spane]')).filter((b) => !b.hidden).sort((a, b) => Number(a.style.order || 0) - Number(b.style.order || 0))[0];
          if (first) { e.preventDefault(); first.focus(); }
        }
      });
      nav.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-spane]');
        if (btn && input.value) setTimeout(() => highlight(btn.dataset.spane), 60);
      });
      // other parts of the page (the command palette) can ask for a setting by name
      window.fllmSettingsSearch = (text) => {
        buildIndex();
        input.value = text;
        apply();
        const first = rankSettingsPages(text, pages)[0];
        if (first) { switchSurface('settings', first.id); setTimeout(() => highlight(first.id), 80); }
      };
    })();
