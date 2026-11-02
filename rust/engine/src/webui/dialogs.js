    // ---- The behaviour every dialog shares ------------------------------------------------------------------------
    // The look and the size rules of a dialog are in chat.css (.dialog, .dialog-actions) and style.css (.modal-backdrop): never taller
    // than the screen, scrolling inside, buttons always visible. This file adds what CSS cannot: the keyboard. While a dialog is open,
    // Tab stays inside it, Escape closes it (through its Cancel button, marked data-dialog-cancel), focus moves into it when it opens and
    // returns to where it was when it closes. It watches the page, so a dialog that code adds later gets the same behaviour.
    // @pure-begin
    /** The element that comes next (or before) `current` in a Tab loop over `items`; `null` when the browser may decide. */
    function dialogTabTarget(items, current, backwards) {
      if (!items.length) return null;
      const at = items.indexOf(current);
      if (at < 0) return backwards ? items[items.length - 1] : items[0];
      if (backwards && at === 0) return items[items.length - 1];
      if (!backwards && at === items.length - 1) return items[0];
      return null;
    }
    // @pure-end

    (function dialogBehaviour() {
      const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';
      const returnTo = new WeakMap();

      const visible = (el) => el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
      const isOpen = (back) => back.classList.contains('open') && getComputedStyle(back).display !== 'none';
      const openBackdrops = () => [...document.querySelectorAll('.modal-backdrop')].filter(isOpen);
      /** The dialog on top: the highest z-index, the one added last among equals. */
      function topmost() {
        const list = openBackdrops();
        let best = null;
        let bestZ = -Infinity;
        for (const back of list) {
          const z = Number.parseInt(getComputedStyle(back).zIndex, 10) || 0;
          if (z >= bestZ) { best = back; bestZ = z; }
        }
        return best;
      }
      const boxOf = (back) => back.querySelector('.dialog, .search-dialog') || back;
      const focusables = (back) => [...boxOf(back).querySelectorAll(FOCUSABLE)].filter(visible);

      function opened(back) {
        if (returnTo.has(back)) return;
        returnTo.set(back, document.activeElement);
        const box = boxOf(back);
        if (!box.hasAttribute('role')) box.setAttribute('role', 'dialog');
        box.setAttribute('aria-modal', 'true');
        if (box.contains(document.activeElement)) return;
        const items = focusables(back);
        const wanted = box.querySelector('[data-autofocus]') || items.find((el) => /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)) || items[items.length - 1] || box;
        if (wanted === box) box.tabIndex = -1;
        wanted.focus({ preventScroll: true });
      }
      function closed(back) {
        if (!returnTo.has(back)) return;
        const before = returnTo.get(back);
        returnTo.delete(back);
        if (before && before.isConnected && !openBackdrops().length && typeof before.focus === 'function') before.focus({ preventScroll: true });
      }
      function sync() {
        const open = new Set(openBackdrops());
        open.forEach(opened);
        document.querySelectorAll('.modal-backdrop').forEach((back) => { if (!open.has(back)) closed(back); });
      }
      new MutationObserver(sync).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ['class', 'style'] });

      document.addEventListener('keydown', (e) => {
        const back = topmost();
        if (!back) return;
        if (e.key === 'Tab') {
          const target = dialogTabTarget(focusables(back), document.activeElement, e.shiftKey);
          const outside = !boxOf(back).contains(document.activeElement);
          if (target || outside) {
            e.preventDefault();
            (target || focusables(back)[e.shiftKey ? focusables(back).length - 1 : 0] || boxOf(back)).focus();
          }
        } else if (e.key === 'Escape' && !back.hasAttribute('data-modal-static')) {
          // the handlers of the dialog itself run first; whatever is still open afterwards is closed here
          setTimeout(() => {
            if (!isOpen(back)) return;
            const cancel = back.querySelector('[data-dialog-cancel]');
            if (cancel) cancel.click(); else back.classList.remove('open');
          }, 0);
        }
      });
      sync();
    })();
