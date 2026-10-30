    // ==========================================================================
    // Navigation, popovers, shortcuts, search and start-up
    // ==========================================================================
    function toggleSidebar() {
      const collapsed = $('appShell').classList.toggle('sidebar-collapsed');
      ['btnExpandSidebarChat', 'btnExpandSidebarCode', 'btnExpandSidebarHub', 'btnExpandSidebarSettings'].forEach((id) => {
        $(id).style.display = collapsed ? 'inline-flex' : 'none';
      });
    }
    ['btnCollapseSidebar', 'btnExpandSidebarChat', 'btnExpandSidebarCode', 'btnExpandSidebarHub', 'btnExpandSidebarSettings'].forEach((id) => $(id).addEventListener('click', toggleSidebar));

    // On a phone the sidebar is a drawer over the page: closed at the start, closed again after a choice or a tap on the dimmed page.
    const narrowScreen = window.matchMedia('(max-width: 760px)');
    let closedForPhone = false; // the sidebar was closed by the narrow window, not by the reader: it comes back when the window is wide again
    function closeSidebarIfNarrow() {
      if (narrowScreen.matches && !$('appShell').classList.contains('sidebar-collapsed')) toggleSidebar();
    }
    narrowScreen.addEventListener('change', () => {
      if (narrowScreen.matches) { closedForPhone = !$('appShell').classList.contains('sidebar-collapsed'); closeSidebarIfNarrow(); }
      else if (closedForPhone) { closedForPhone = false; if ($('appShell').classList.contains('sidebar-collapsed')) toggleSidebar(); }
    });
    $('sidebarScrim').addEventListener('click', closeSidebarIfNarrow);
    document.querySelector('.sidebar-outer').addEventListener('click', (e) => {
      if (e.target.closest('.nav-item, .thread-row') && !e.target.closest('.thread-row-actions, .sel-box')) setTimeout(closeSidebarIfNarrow, 0);
    });

    $('navNewChat').addEventListener('click', startNewChat);
    $('navHub').addEventListener('click', () => switchSurface('hub'));
    $('navSettings').addEventListener('click', () => switchSurface('settings'));
    $('btnGenGoHub').addEventListener('click', () => switchSurface('hub'));
    $('btnModelMenuGoHub').addEventListener('click', () => { $('modelDropdownMenu').classList.remove('open'); switchSurface('hub'); });
    $('sidebarModeTabs').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-tab]');
      if (btn) switchSurface(btn.dataset.tab);
    });
    $('settingsMenuNav').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-spane]');
      if (btn) switchSettingsPane(btn.dataset.spane);
    });

    // model menu
    $('btnModelDropdown').addEventListener('click', (e) => { e.stopPropagation(); $('presetMenu').classList.remove('open'); const open = $('modelDropdownMenu').classList.toggle('open'); if (open) fillModelMenu(); });
    $('modelDropdownItems').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-pick-model]');
      if (!btn) return;
      state.selectedModel = btn.dataset.pickModel;
      $('activeModelLabel').textContent = state.selectedModel;
      $('modelDropdownMenu').classList.remove('open');
      renderThinkingChip();
      refreshStudioStatus();
    });

    // popovers of the composer
    function togglePopover(popId, btnId, otherPop, otherBtn) {
      $(otherPop).classList.remove('open');
      $(otherBtn).classList.remove('active');
      $(popId).classList.toggle('open');
      $(btnId).classList.toggle('active', $(popId).classList.contains('open'));
    }
    $('btnToggleSamplerPop').addEventListener('click', (e) => { e.stopPropagation(); togglePopover('samplerPopover', 'btnToggleSamplerPop', 'assistantPopover', 'btnToggleAssistantPop'); });
    $('btnToggleAssistantPop').addEventListener('click', (e) => { e.stopPropagation(); togglePopover('assistantPopover', 'btnToggleAssistantPop', 'samplerPopover', 'btnToggleSamplerPop'); });
    document.addEventListener('click', (e) => {
      const pairs = [
        ['samplerPopover', 'btnToggleSamplerPop'], ['assistantPopover', 'btnToggleAssistantPop'],
        ['modelDropdownMenu', 'btnModelDropdown'], ['presetMenu', 'btnPresets'], ['chatMenu', 'btnChatMenu'],
      ];
      for (const [pop, btn] of pairs) {
        if (!e.target.closest(`#${pop}`) && !e.target.closest(`#${btn}`)) {
          $(pop).classList.remove('open');
          $(btn).classList.remove('active');
        }
      }
    });
    $('selConstraint').addEventListener('change', () => {
      const v = $('selConstraint').value;
      $('txtConstraintSpec').style.display = (v === 'none' || v === 'json_object') ? 'none' : 'block';
    });
    // thinking: the chip steps through the levels the loaded model has (sampler.js)
    $('btnToggleThinking').addEventListener('click', () => {
      const caps = thinkingCaps();
      setThinkingLevel(nextThinkingLevel(caps, thinkingLevelNow()));
    });

    const saveConfigToRust = async () => {
      const p = getSampler();
      const res = await apiFetch('/api/studio/config', {
        method: 'POST',
        body: {
          temperature: p.temperature, top_p: p.top_p, top_k: p.top_k, min_p: p.min_p,
          repetition_penalty: p.repetition_penalty, presence_penalty: p.presence_penalty,
          frequency_penalty: p.frequency_penalty, dry_multiplier: p.dry_multiplier,
          mirostat: p.mirostat, seed: p.seed < 0 ? null : p.seed,
          system_prompt: p.system_prompt,
        },
      });
      if (res.ok) { toast('Saved as the engine default'); refreshStudioStatus(); }
    };
    $('btnSaveSamplerToRust').addEventListener('click', saveConfigToRust);
    $('btnSaveSystemPromptRust').addEventListener('click', saveConfigToRust);
    $('btnResetSampler').addEventListener('click', () => {
      setSamplerUI({ temperature: 0.7, top_p: 0.95, top_k: 20, min_p: 0, repetition_penalty: 1.0, presence_penalty: 0, frequency_penalty: 0, dry_multiplier: 0, mirostat: 0, seed: -1 });
    });

    // ---- search (Ctrl+K): the command palette is in palette.js ----
    function snippet(text, q) {
      const i = text.toLowerCase().indexOf(q);
      if (i < 0) return '';
      return `${i > 30 ? '…' : ''}${text.slice(Math.max(0, i - 30), i + q.length + 50).replace(/\s+/g, ' ')}…`;
    }
    // ---- keyboard shortcuts: one registry (ids, default keys, descriptions), rebindable, stored in localStorage ----
    // A key combination is a text such as "Mod+Shift+K": Mod is Ctrl (Cmd on macOS), then Ctrl (macOS only), Alt, Shift, and one key.
    // @pure-begin
    const SHORTCUTS = [
      { id: 'newChat', keys: 'Mod+N', desc: 'New chat (new session in Code)' },
      { id: 'search', keys: 'Mod+K', desc: 'Search chats and settings' },
      { id: 'focusComposer', keys: 'Shift+Escape', desc: 'Put the cursor in the message field' },
      { id: 'toggleSidebar', keys: 'Mod+B', desc: 'Show or hide the sidebar' },
      { id: 'stop', keys: 'Escape', desc: 'Stop the answer or the agent, close a window' },
      { id: 'surfaceChat', keys: 'Alt+1', desc: 'Go to Chat' },
      { id: 'surfaceCode', keys: 'Alt+2', desc: 'Go to Code' },
      { id: 'openModels', keys: 'Alt+3', desc: 'Open Models' },
      { id: 'openSettings', keys: 'Mod+,', desc: 'Open Settings' },
      { id: 'nextChat', keys: 'Alt+ArrowDown', desc: 'Next chat in the list' },
      { id: 'prevChat', keys: 'Alt+ArrowUp', desc: 'Previous chat in the list' },
      { id: 'regenerate', keys: 'Alt+R', desc: 'Regenerate the last answer' },
      { id: 'shortcuts', keys: 'Mod+/', desc: 'This overview (also ? outside a text field)' },
      { id: 'codeAgent', keys: 'Alt+B', desc: 'Code: switch between the build and the plan agent' },
      { id: 'codeSessions', keys: 'Alt+L', desc: 'Code: pick an earlier session of this folder' },
    ];
    /** Keys with a fixed meaning that no shortcut may take. */
    const FIXED_KEYS = [
      { keys: 'Enter', desc: 'Send' },
      { keys: 'Shift+Enter', desc: 'New line' },
      { keys: 'ArrowUp', desc: 'Edit your last message (empty field)' },
      { keys: '?', desc: 'This overview' },
    ];
    const MOD_ORDER = ['Mod', 'Ctrl', 'Alt', 'Shift'];

    /** Splits "Mod+Shift+K" into its parts; the key may be "+" ("Mod++"). Returns null for an empty text. */
    function parseCombo(text) {
      const s = String(text || '').trim();
      if (!s) return null;
      const out = { mods: [], key: '' };
      let rest = s;
      for (;;) {
        const i = rest.indexOf('+');
        if (i <= 0) break;
        const head = MOD_ORDER.find((m) => m.toLowerCase() === rest.slice(0, i).toLowerCase());
        if (!head) break;
        if (!out.mods.includes(head)) out.mods.push(head);
        rest = rest.slice(i + 1);
      }
      out.key = rest.length === 1 ? rest.toUpperCase() : rest;
      out.mods.sort((a, b) => MOD_ORDER.indexOf(a) - MOD_ORDER.indexOf(b));
      return out.key ? out : null;
    }
    /** The normal form of a combination text (modifiers in a fixed order, letters in upper case). */
    function normCombo(text) {
      const p = parseCombo(text);
      return p ? [...p.mods, p.key].join('+') : '';
    }
    /** The combination of a keyboard event, or '' while only a modifier key is down. */
    function comboFromEvent(e, isMac) {
      let key = e.key;
      if (!key || ['Control', 'Shift', 'Alt', 'Meta', 'AltGraph', 'Dead', 'Unidentified'].includes(key)) return '';
      // with Alt held, some systems report a special character: take the physical key then
      const code = e.code || '';
      if (e.altKey && /^Key[A-Z]$/.test(code)) key = code.slice(3);
      else if (e.altKey && /^Digit[0-9]$/.test(code)) key = code.slice(5);
      if (key === ' ') key = 'Space';
      const mods = [];
      const mod = isMac ? e.metaKey : e.ctrlKey;
      if (mod) mods.push('Mod');
      if (isMac && e.ctrlKey) mods.push('Ctrl');
      if (e.altKey) mods.push('Alt');
      // Shift is part of a typed symbol such as ? or +, so it only counts for letters, digits and named keys
      if (e.shiftKey && (key.length > 1 || /[A-Za-z0-9]/.test(key))) mods.push('Shift');
      return normCombo([...mods, key.length === 1 ? key.toUpperCase() : key].join('+'));
    }
    /** Shows a combination for people: "Ctrl+Shift+K", on macOS with the usual symbols. */
    function formatCombo(text, isMac) {
      const p = parseCombo(text);
      if (!p) return '';
      const names = { ArrowUp: 'Up', ArrowDown: 'Down', ArrowLeft: 'Left', ArrowRight: 'Right', Escape: 'Esc' };
      const mod = (m) => (m === 'Mod' ? (isMac ? 'Cmd' : 'Ctrl') : m === 'Alt' && isMac ? 'Option' : m);
      return [...p.mods.map(mod), names[p.key] || p.key].join(isMac ? ' ' : '+');
    }
    /** The keys in force: the defaults with the saved changes on top (changes for unknown ids are ignored). */
    function effectiveCombos(overrides) {
      const out = {};
      for (const s of SHORTCUTS) {
        const o = overrides && typeof overrides[s.id] === 'string' ? normCombo(overrides[s.id]) : '';
        out[s.id] = o || normCombo(s.keys);
      }
      return out;
    }
    /** Whether `combo` may be given to shortcut `id`. Returns { ok: true, combo } or { ok: false, error }. */
    function checkCombo(combos, id, text) {
      const combo = normCombo(text);
      const p = parseCombo(combo);
      if (!p) return { ok: false, error: 'Press a key combination.' };
      const named = p.key.length > 1;
      if (!p.mods.length && !(named && !['Enter', 'Tab', 'Space', 'Backspace', 'Delete'].includes(p.key))) {
        return { ok: false, error: 'Use at least one of Ctrl, Alt or Cmd with the key, so typing is not disturbed.' };
      }
      if (p.mods.length === 1 && p.mods[0] === 'Shift' && !named) return { ok: false, error: 'Shift alone is used for typing capital letters.' };
      for (const f of FIXED_KEYS) {
        if (normCombo(f.keys) === combo) return { ok: false, error: `${formatCombo(combo, false)} is fixed: ${f.desc}.` };
      }
      for (const s of SHORTCUTS) {
        if (s.id !== id && combos[s.id] === combo) return { ok: false, error: `${formatCombo(combo, false)} is already used for: ${s.desc}.` };
      }
      return { ok: true, combo };
    }
    /** The shortcut id that `combo` triggers, or ''. */
    function shortcutFor(combos, combo) {
      return (SHORTCUTS.find((s) => combos[s.id] === combo) || {}).id || '';
    }
    // @pure-end

    const SHORTCUTS_KEY = 'fllm.studio.shortcuts.v1';
    const isMacOs = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
    let shortcutOverrides = {};
    try { shortcutOverrides = JSON.parse(localStorage.getItem(SHORTCUTS_KEY) || '{}') || {}; } catch { shortcutOverrides = {}; }
    let shortcutCombos = effectiveCombos(shortcutOverrides);
    /** The keys of shortcut `id` as shown to people (used by the welcome tour as well). */
    function shortcutLabel(id) { return formatCombo(shortcutCombos[id], isMacOs); }
    function saveShortcuts() {
      for (const id of Object.keys(shortcutOverrides)) {
        if (!SHORTCUTS.some((s) => s.id === id) || shortcutOverrides[id] === normCombo(SHORTCUTS.find((s) => s.id === id).keys)) delete shortcutOverrides[id];
      }
      localStorage.setItem(SHORTCUTS_KEY, JSON.stringify(shortcutOverrides));
      shortcutCombos = effectiveCombos(shortcutOverrides);
    }

    let shortcutCapture = null; // the id being rebound
    let shortcutMessage = '';
    function renderShortcutsPanel() {
      const keyChip = (c) => `<kbd class="mono">${escapeHtml(formatCombo(c, isMacOs))}</kbd>`;
      $('shortcutsBody').innerHTML =
        SHORTCUTS.map((s) => {
          const changed = Boolean(shortcutOverrides[s.id]);
          return `<tr data-sc="${s.id}"><td>${escapeHtml(s.desc)}</td>
            <td class="sc-keys">${shortcutCapture === s.id ? '<span class="sc-wait">Press the new keys (Esc cancels)</span>' : keyChip(shortcutCombos[s.id])}</td>
            <td class="sc-actions"><button class="btn btn-xs btn-ghost" data-sc-change="${s.id}">${shortcutCapture === s.id ? 'Cancel' : 'Change'}</button>${changed ? `<button class="btn btn-xs btn-ghost" data-sc-reset="${s.id}">Reset</button>` : ''}</td></tr>`;
        }).join('') +
        FIXED_KEYS.filter((f) => f.keys !== '?').map((f) => `<tr class="sc-fixed"><td>${escapeHtml(f.desc)}</td><td class="sc-keys">${keyChip(f.keys)}</td><td></td></tr>`).join('');
      $('shortcutsMessage').textContent = shortcutMessage;
      $('btnShortcutsReset').style.display = Object.keys(shortcutOverrides).length ? '' : 'none';
    }
    function openShortcuts() {
      shortcutCapture = null;
      shortcutMessage = '';
      renderShortcutsPanel();
      $('helpModal').classList.add('open');
    }
    function closeShortcuts() { shortcutCapture = null; $('helpModal').classList.remove('open'); }
    $('helpClose').addEventListener('click', closeShortcuts);
    $('helpModal').addEventListener('click', (e) => { if (e.target === $('helpModal')) closeShortcuts(); });
    $('btnShortcutsReset').addEventListener('click', () => { shortcutOverrides = {}; shortcutMessage = 'All shortcuts are back to their defaults.'; saveShortcuts(); renderShortcutsPanel(); });
    $('shortcutsBody').addEventListener('click', (e) => {
      const change = e.target.closest('[data-sc-change]');
      const reset = e.target.closest('[data-sc-reset]');
      if (change) {
        shortcutCapture = shortcutCapture === change.dataset.scChange ? null : change.dataset.scChange;
        shortcutMessage = '';
      } else if (reset) {
        delete shortcutOverrides[reset.dataset.scReset];
        saveShortcuts();
        shortcutMessage = '';
      } else return;
      renderShortcutsPanel();
    });

    function surfaceTab(tab) { if (state.surface !== tab) switchSurface(tab); else setTimeout(() => $(tab === 'code' ? 'codeTextarea' : 'chatTextarea').focus(), 0); }
    function stepChat(delta) {
      const list = state.threads.slice().sort((a, b) => b.updatedAt - a.updatedAt);
      if (!list.length) return;
      const i = list.findIndex((t) => t.id === state.activeThreadId);
      openThread(list[Math.min(Math.max(i < 0 ? 0 : i + delta, 0), list.length - 1)].id);
    }
    function regenerateLast() {
      const t = currentThread();
      const last = t && pathOf(t).filter((n) => n.role === 'assistant').pop();
      if (last && !state.abortController) regenerate(last.id);
    }
    function closeOverlays() {
      ['searchModal', 'helpModal', 'tourModal'].forEach((id) => $(id) && $(id).classList.remove('open'));
      shortcutCapture = null;
    }
    const SHORTCUT_ACTIONS = {
      newChat: () => { if (state.surface === 'code') newCodeSession(); else startNewChat(); },
      search: () => window.openSearchModal(),
      focusComposer: () => { if (state.surface === 'code') $('codeTextarea').focus(); else if (state.surface === 'chat') $('chatTextarea').focus(); else switchSurface('chat'); },
      toggleSidebar,
      stop: () => {
        closeOverlays();
        if (state.surface === 'chat' && state.abortController) state.abortController.abort();
        if (state.surface === 'code' && code.running) stopAgent();
      },
      surfaceChat: () => surfaceTab('chat'),
      surfaceCode: () => surfaceTab('code'),
      openModels: () => switchSurface('hub'),
      openSettings: () => switchSurface('settings'),
      nextChat: () => stepChat(1),
      prevChat: () => stepChat(-1),
      regenerate: regenerateLast,
      shortcuts: openShortcuts,
      codeAgent: () => { if (state.surface !== 'code') switchSurface('code'); toggleAgent(); },
      codeSessions: () => { if (state.surface !== 'code') switchSurface('code'); runCodeCommand('/resume'); },
    };

    window.addEventListener('keydown', (e) => {
      const combo = comboFromEvent(e, isMacOs);
      if (!combo) return;
      if (shortcutCapture) { // rebinding: the next combination is the new one
        e.preventDefault();
        e.stopPropagation();
        if (combo === 'Escape') { shortcutCapture = null; shortcutMessage = ''; renderShortcutsPanel(); return; }
        const result = checkCombo(shortcutCombos, shortcutCapture, combo);
        if (!result.ok) { shortcutMessage = result.error; renderShortcutsPanel(); return; }
        shortcutOverrides[shortcutCapture] = result.combo;
        shortcutCapture = null;
        saveShortcuts();
        shortcutMessage = 'Saved.';
        renderShortcutsPanel();
        return;
      }
      if ($('dialogModal').classList.contains('open')) return;
      const t = e.target;
      const typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
      if (combo === '?' && !typing) { e.preventDefault(); openShortcuts(); return; }
      // a plain key or Shift+key goes to the text field; Escape and combinations with Ctrl, Alt or Cmd always count
      const plain = !/(^|\+)(Mod|Ctrl|Alt)\+/.test(combo) && combo !== 'Escape' && combo !== 'Shift+Escape';
      if (typing && plain) return;
      const id = shortcutFor(shortcutCombos, combo);
      if (id) { e.preventDefault(); SHORTCUT_ACTIONS[id](); return; }
      if (combo === 'Escape') closeOverlays(); // Escape closes windows even when it is not the stop key
    });

    // Rewrite a container only when its HTML really changed, so the page does not flicker on the status poll
    (function dedupeInnerHtml() {
      const d = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML');
      ['modelDropdownItems', 'providerModelFacts', 'providerAliasesList', 'apiKeysTableContainer', 'hwDevicesGrid', 'hwSlotsTableWrap', 'cachedModelsHubList', 'loadedModelHubItem', 'hubCatalogCards', 'threadList'].forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        Object.defineProperty(el, 'innerHTML', {
          configurable: true,
          get() { return d.get.call(this); },
          set(v) { if (this._h !== v) { this._h = v; d.set.call(this, v); } },
        });
      });
    })();

    // ---- start ----
    applyAppearance();
    loadThreads();
    renderThreadList();
    renderChatArea();
    prjInit(); // projects.js: assistants (migrated presets), projects, memory, prompt library
    updateRouteTesterSample();
    refreshStudioStatus().then(() => {
      if (state.setup) return;
      loadWorkspace();
      // the message that asked for the model to be loaded (models.js: loadLastModelFor) goes out now
      let pending = null;
      try { pending = JSON.parse(sessionStorage.getItem('fllm.pendingSend') || 'null'); } catch { /* not ours */ }
      if (!pending || !pending.text) return;
      sessionStorage.removeItem('fllm.pendingSend');
      setTimeout(() => {
        if (pending.surface === 'code') {
          switchSurface('code');
          $('codeTextarea').value = pending.text;
          $('btnCodeSend').click();
        } else {
          $('chatTextarea').value = pending.text;
          sendChatPrompt();
        }
      }, 400);
    });
    setTimeout(() => { state.updateChecked = true; checkUpdate().then(() => { if (state.update?.newer) toast(`Version ${state.update.latest} is available: see Models.`); }); }, 4000);
    if (window.matchMedia('(max-width: 760px)').matches) toggleSidebar(); // on a phone the rail would cover the page
    setInterval(() => { if (!document.hidden) refreshStudioStatus(); }, 2000); // a hidden tab asks nothing
