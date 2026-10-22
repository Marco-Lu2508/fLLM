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
    $('btnModelDropdown').addEventListener('click', (e) => { e.stopPropagation(); $('modelDropdownMenu').classList.toggle('open'); });
    $('modelDropdownItems').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-pick-model]');
      if (!btn) return;
      state.selectedModel = btn.dataset.pickModel;
      $('activeModelLabel').textContent = state.selectedModel;
      $('modelDropdownMenu').classList.remove('open');
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
    $('btnToggleThinking').addEventListener('click', () => {
      state.thinkingEnabled = !state.thinkingEnabled;
      $('btnToggleThinking').classList.toggle('active', state.thinkingEnabled);
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
          speculative: p.speculative, system_prompt: p.system_prompt,
        },
      });
      if (res.ok) { toast('Saved as the engine default'); refreshStudioStatus(); }
    };
    $('btnSaveSamplerToRust').addEventListener('click', saveConfigToRust);
    $('btnSaveSystemPromptRust').addEventListener('click', saveConfigToRust);
    $('btnResetSampler').addEventListener('click', () => {
      setSamplerUI({ temperature: 0.7, top_p: 0.95, top_k: 20, min_p: 0, repetition_penalty: 1.0, presence_penalty: 0, frequency_penalty: 0, dry_multiplier: 0, mirostat: 0, seed: -1, speculative: true });
    });

    // ---- search (Ctrl+K): chats by title and content, plus places in the app ----
    function openSearchModal() {
      $('searchModal').classList.add('open');
      $('searchModalInput').value = '';
      renderSearchResults('');
      $('searchModalInput').focus();
    }
    function snippet(text, q) {
      const i = text.toLowerCase().indexOf(q);
      if (i < 0) return '';
      return `${i > 30 ? '…' : ''}${text.slice(Math.max(0, i - 30), i + q.length + 50).replace(/\s+/g, ' ')}…`;
    }
    function renderSearchResults(q) {
      const query = q.toLowerCase().trim();
      const places = [
        { type: 'Page', title: 'Coding agent (Code)', run: () => switchSurface('code') },
        { type: 'Page', title: 'Models and Hugging Face', run: () => switchSurface('hub') },
        { type: 'Page', title: 'Settings: local API server and keys', run: () => switchSurface('settings', 'local-api-server') },
        { type: 'Page', title: 'Settings: hardware', run: () => switchSurface('settings', 'hardware') },
        { type: 'Page', title: 'Settings: appearance and colors', run: () => switchSurface('settings', 'appearance') },
        { type: 'Page', title: 'Settings: tokenizer and Jinja templates', run: () => switchSurface('settings', 'tokenizer') },
      ];
      const chats = state.threads.slice().sort((a, b) => b.updatedAt - a.updatedAt).map((t) => {
        const hit = query && !t.title.toLowerCase().includes(query) ? snippet(threadText(t), query) : '';
        return { type: 'Chat', title: t.title, sub: hit, match: !query || t.title.toLowerCase().includes(query) || Boolean(hit), run: () => openThread(t.id) };
      });
      const items = [...chats.filter((c) => c.match), ...places.filter((p) => !query || p.title.toLowerCase().includes(query))].slice(0, 30);
      $('searchModalResults').innerHTML = items.length ? items.map((it, i) => `
        <button class="nav-item" data-sres="${i}">
          <span class="search-title">${escapeHtml(it.title)}${it.sub ? `<small>${escapeHtml(it.sub)}</small>` : ''}</span>
          <span class="quant-chip" style="margin-left:auto;">${it.type}</span>
        </button>`).join('') : '<div class="thread-empty">Nothing found</div>';
      $('searchModalResults')._items = items;
    }
    $('navOpenSearch').addEventListener('click', openSearchModal);
    $('searchModalInput').addEventListener('input', (e) => renderSearchResults(e.target.value));
    $('searchModalInput').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') $('searchModalResults').querySelector('[data-sres]')?.click();
    });
    $('searchModal').addEventListener('click', (e) => {
      if (e.target === $('searchModal')) $('searchModal').classList.remove('open');
      const btn = e.target.closest('[data-sres]');
      if (btn) {
        const it = $('searchModalResults')._items[Number(btn.dataset.sres)];
        $('searchModal').classList.remove('open');
        it.run();
      }
    });
    $('helpClose').addEventListener('click', () => $('helpModal').classList.remove('open'));
    $('helpModal').addEventListener('click', (e) => { if (e.target === $('helpModal')) $('helpModal').classList.remove('open'); });

    window.addEventListener('keydown', (e) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearchModal(); }
      if (mod && e.key.toLowerCase() === 'n') { e.preventDefault(); if (state.surface === 'code') newCodeSession(); else startNewChat(); }
      if (mod && e.key.toLowerCase() === 'b') { e.preventDefault(); toggleSidebar(); }
      if (mod && e.key === '/') { e.preventDefault(); $('helpModal').classList.add('open'); }
      if (e.key === 'Escape') {
        if ($('dialogModal').classList.contains('open')) return;
        ['searchModal', 'helpModal'].forEach((id) => $(id).classList.remove('open'));
        if (state.surface === 'chat' && state.abortController) state.abortController.abort();
        if (state.surface === 'code' && code.running) stopAgent();
      }
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
    renderPresets();
    updateRouteTesterSample();
    refreshStudioStatus().then(() => { if (!state.setup) loadWorkspace(); });
    if (window.matchMedia('(max-width: 760px)').matches) toggleSidebar(); // on a phone the rail would cover the page
    setInterval(() => { if (!document.hidden) refreshStudioStatus(); }, 2000); // a hidden tab asks nothing
