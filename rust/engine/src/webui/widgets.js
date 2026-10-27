    // ---- Own widgets instead of the browser's: drop-down lists and tooltips (the page should not look like a web page) ----

    /** Replaces the list of every <select> with a menu of the app. The select stays in the page (hidden) and keeps the value, so all other code is unchanged. */
    const customSelects = (() => {
      let open = null;

      function label(select) {
        const option = select.options[select.selectedIndex];
        return option ? option.textContent : '';
      }

      function close() {
        if (!open) return;
        open.list.remove();
        open.button.setAttribute('aria-expanded', 'false');
        document.removeEventListener('mousedown', open.outside, true);
        window.removeEventListener('resize', close);
        document.removeEventListener('scroll', open.scroll, true);
        open = null;
      }

      function choose(select, index) {
        if (select.selectedIndex !== index) {
          select.selectedIndex = index;
          select.dispatchEvent(new Event('input', { bubbles: true }));
          select.dispatchEvent(new Event('change', { bubbles: true }));
        }
        close();
      }

      function show(select, button) {
        if (open && open.select === select) { close(); return; }
        close();
        const list = document.createElement('div');
        list.className = 'popover-panel cs-list open';
        list.setAttribute('role', 'listbox');
        let active = select.selectedIndex;
        const rows = [];
        Array.from(select.children).forEach((child) => {
          const options = child.tagName === 'OPTGROUP' ? Array.from(child.children) : [child];
          if (child.tagName === 'OPTGROUP') {
            const head = document.createElement('div');
            head.className = 'cs-group';
            head.textContent = child.label;
            list.appendChild(head);
          }
          options.forEach((option) => {
            const row = document.createElement('div');
            row.className = 'menu-item cs-item';
            row.setAttribute('role', 'option');
            row.textContent = option.textContent;
            if (option.disabled || child.disabled) row.classList.add('cs-disabled');
            if (option.index === select.selectedIndex) row.classList.add('is-active');
            row.addEventListener('mousedown', (e) => e.preventDefault());
            row.addEventListener('click', () => { if (!row.classList.contains('cs-disabled')) choose(select, option.index); });
            row.dataset.index = option.index;
            list.appendChild(row);
            rows.push(row);
          });
        });
        document.body.appendChild(list);
        const rect = button.getBoundingClientRect();
        list.style.minWidth = `${rect.width}px`;
        const room = window.innerHeight - rect.bottom - 12;
        const above = rect.top - 12;
        const flip = room < 200 && above > room;
        list.style.maxHeight = `${Math.max(120, flip ? above : room)}px`;
        list.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - list.offsetWidth - 8))}px`;
        if (flip) list.style.bottom = `${window.innerHeight - rect.top + 4}px`;
        else list.style.top = `${rect.bottom + 4}px`;
        const mark = (i) => {
          rows.forEach((r) => r.classList.remove('cs-hover'));
          const r = rows.find((x) => Number(x.dataset.index) === i);
          if (r) { r.classList.add('cs-hover'); r.scrollIntoView({ block: 'nearest' }); }
        };
        mark(active);
        const step = (delta) => {
          const enabled = rows.filter((r) => !r.classList.contains('cs-disabled')).map((r) => Number(r.dataset.index));
          if (!enabled.length) return;
          const at = enabled.indexOf(active);
          active = enabled[Math.max(0, Math.min(enabled.length - 1, (at < 0 ? 0 : at) + delta))];
          mark(active);
        };
        open = {
          select, button, list,
          outside: (e) => { if (!list.contains(e.target) && !button.contains(e.target)) close(); },
          scroll: (e) => { if (!list.contains(e.target)) close(); },
          key: (e) => {
            if (e.key === 'ArrowDown') step(1);
            else if (e.key === 'ArrowUp') step(-1);
            else if (e.key === 'Home') step(-1e6);
            else if (e.key === 'End') step(1e6);
            else if (e.key === 'Enter' || e.key === ' ') choose(select, active);
            else if (e.key === 'Escape' || e.key === 'Tab') close();
            else if (e.key.length === 1) {
              const hit = rows.find((r) => !r.classList.contains('cs-disabled') && r.textContent.toLowerCase().startsWith(e.key.toLowerCase()));
              if (hit) { active = Number(hit.dataset.index); mark(active); }
            } else return;
            e.preventDefault();
            e.stopPropagation();
          },
        };
        document.addEventListener('mousedown', open.outside, true);
        document.addEventListener('scroll', open.scroll, true);
        window.addEventListener('resize', close);
        button.setAttribute('aria-expanded', 'true');
      }

      function enhance(select) {
        if (select._cs || select.multiple || select.size > 1 || select.dataset.native) return;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `${select.className} cs-btn`;
        button.setAttribute('aria-haspopup', 'listbox');
        button.setAttribute('aria-expanded', 'false');
        const text = document.createElement('span');
        text.className = 'cs-text';
        button.appendChild(text);
        const sync = () => {
          text.textContent = label(select);
          button.disabled = select.disabled;
          button.style.cssText = select.style.cssText;
          button.style.display = (select.style.display === 'none' || select.hidden) ? 'none' : '';
          if (select.title) button.title = select.title;
        };
        select._cs = { sync };
        select.classList.add('cs-native');
        select.tabIndex = -1;
        select.setAttribute('aria-hidden', 'true');
        select.insertAdjacentElement('afterend', button);
        button.addEventListener('click', () => show(select, button));
        button.addEventListener('keydown', (e) => {
          if (open && open.select === select) { open.key(e); return; }
          if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); show(select, button); }
        });
        new MutationObserver(sync).observe(select, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['style', 'hidden', 'disabled', 'title'] });
        select.addEventListener('change', sync);
        sync();
      }

      // code sets .value / .selectedIndex directly, which fires no event: keep the shown text in step
      for (const prop of ['value', 'selectedIndex']) {
        const desc = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, prop);
        Object.defineProperty(HTMLSelectElement.prototype, prop, {
          get: desc.get,
          set(v) { desc.set.call(this, v); if (this._cs) this._cs.sync(); },
          configurable: true,
        });
      }

      const scan = (root) => {
        if (root.tagName === 'SELECT') enhance(root);
        else if (root.querySelectorAll) root.querySelectorAll('select').forEach(enhance);
      };
      scan(document);
      new MutationObserver((records) => records.forEach((r) => r.addedNodes.forEach((n) => { if (n.nodeType === 1) scan(n); })))
        .observe(document.body, { childList: true, subtree: true });
      return { close };
    })();

    /** Tooltips of the app instead of the browser's yellow box: every `title` becomes a tip that appears after a short wait. */
    (() => {
      const tip = document.createElement('div');
      tip.className = 'app-tip';
      document.body.appendChild(tip);
      let timer = 0;
      let target = null;
      const hide = () => { clearTimeout(timer); tip.classList.remove('show'); target = null; };
      document.addEventListener('mouseover', (e) => {
        const el = e.target.closest && e.target.closest('[title], [data-tip]');
        if (!el || el === target) return;
        hide();
        if (el.hasAttribute('title')) {
          const text = el.getAttribute('title');
          el.removeAttribute('title');
          if (text) {
            el.dataset.tip = text;
            if (!el.getAttribute('aria-label') && !el.textContent.trim()) el.setAttribute('aria-label', text);
          }
        }
        if (!el.dataset.tip) return;
        target = el;
        timer = setTimeout(() => {
          tip.textContent = el.dataset.tip;
          const r = el.getBoundingClientRect();
          tip.classList.add('show');
          const w = tip.offsetWidth;
          tip.style.left = `${Math.max(6, Math.min(r.left + r.width / 2 - w / 2, window.innerWidth - w - 6))}px`;
          const below = r.bottom + 8 + tip.offsetHeight < window.innerHeight;
          tip.style.top = `${below ? r.bottom + 8 : r.top - tip.offsetHeight - 8}px`;
        }, 450);
      });
      document.addEventListener('mouseout', (e) => { if (target && !target.contains(e.relatedTarget)) hide(); });
      document.addEventListener('mousedown', hide, true);
      document.addEventListener('scroll', hide, true);
    })();

    /** The right-click menu of the app (cut, copy, paste, select all, copy link) instead of the browser's. */
    (() => {
      let menu = null;
      const close = () => { if (menu) { menu.remove(); menu = null; } };
      const isEditable = (el) => el && (el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && !['checkbox', 'radio', 'button', 'range', 'file', 'color'].includes(el.type)) || el.isContentEditable);
      const selectionText = (field) => (field && typeof field.selectionStart === 'number' && field.tagName !== 'SELECT'
        ? field.value.slice(field.selectionStart, field.selectionEnd)
        : String(window.getSelection() || ''));

      async function paste(field) {
        let text = '';
        try { text = await navigator.clipboard.readText(); } catch { toast('Paste is not allowed here. Use Ctrl+V.'); return; }
        field.focus();
        if (typeof field.selectionStart === 'number') {
          field.setRangeText(text, field.selectionStart, field.selectionEnd, 'end');
          field.dispatchEvent(new Event('input', { bubbles: true }));
        } else document.execCommand('insertText', false, text);
      }

      function actions(target) {
        const list = [];
        const field = isEditable(target) ? target : null;
        const text = selectionText(field);
        const link = target.closest && target.closest('a[href]');
        if (field) {
          const readOnly = field.readOnly || field.disabled;
          list.push({ label: 'Cut', key: 'Ctrl+X', off: readOnly || !text, run: () => { navigator.clipboard.writeText(text); field.focus(); document.execCommand('delete'); } });
          list.push({ label: 'Copy', key: 'Ctrl+C', off: !text, run: () => navigator.clipboard.writeText(text) });
          list.push({ label: 'Paste', key: 'Ctrl+V', off: readOnly, run: () => paste(field) });
          list.push({ label: 'Select all', key: 'Ctrl+A', run: () => { field.focus(); field.select ? field.select() : document.execCommand('selectAll'); } });
          return list;
        }
        if (text) list.push({ label: 'Copy', key: 'Ctrl+C', run: () => navigator.clipboard.writeText(text) });
        if (link) list.push({ label: 'Copy link', run: () => navigator.clipboard.writeText(link.href) });
        const block = target.closest && target.closest('pre, .msg-assistant-prose, .msg-user-bubble, .tool-cmd, td');
        if (block && !text) list.push({ label: block.tagName === 'PRE' ? 'Copy code' : 'Copy text', run: () => navigator.clipboard.writeText(block.innerText.trim()) });
        if (block) list.push({ label: 'Select all text here', run: () => { const r = document.createRange(); r.selectNodeContents(block); const s = window.getSelection(); s.removeAllRanges(); s.addRange(r); } });
        return list;
      }

      document.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        close();
        customSelects.close();
        const list = actions(e.target);
        if (!list.length) return;
        menu = document.createElement('div');
        menu.className = 'popover-panel cs-list open cm';
        list.forEach((a) => {
          const row = document.createElement('div');
          row.className = 'menu-item cs-item' + (a.off ? ' cs-disabled' : '');
          const name = document.createElement('span');
          name.textContent = a.label;
          row.appendChild(name);
          if (a.key) { const k = document.createElement('span'); k.className = 'cm-key'; k.textContent = a.key; row.appendChild(k); }
          row.addEventListener('mousedown', (ev) => ev.preventDefault());
          row.addEventListener('click', () => { if (a.off) return; close(); a.run(); });
          menu.appendChild(row);
        });
        document.body.appendChild(menu);
        menu.style.left = `${Math.max(6, Math.min(e.clientX, window.innerWidth - menu.offsetWidth - 6))}px`;
        menu.style.top = `${Math.max(6, Math.min(e.clientY, window.innerHeight - menu.offsetHeight - 6))}px`;
      });
      document.addEventListener('mousedown', (e) => { if (menu && !menu.contains(e.target)) close(); }, true);
      document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); }, true);
      document.addEventListener('scroll', close, true);
      window.addEventListener('blur', close);
    })();
