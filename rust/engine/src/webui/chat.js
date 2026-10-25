    // ==========================================================================
    // Chat: every thread is a tree of messages (editing or regenerating adds a branch),
    // streamed answers update only their own message, threads are stored in the browser.
    // ==========================================================================
    const ROOT = 'root';
    const uid = (prefix) => `${prefix}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
    const PRESET_KEY = 'fllm.studio.presets.v1';
    const MAX_TEXT_ATTACHMENT = 20 * 1024 * 1024;
    const PASTE_AS_FILE = 6000;

    function newThread(title) {
      const now = Date.now();
      return {
        id: uid('thr'), title: title || 'New chat', autoTitle: true, createdAt: now, updatedAt: now, pinned: false, system: null,
        nodes: { [ROOT]: { id: ROOT, role: 'root', children: [], active: 0 } },
      };
    }

    /** Old threads kept a flat list of messages; make a chain of nodes out of it. */
    function migrateThread(raw) {
      if (!raw || typeof raw !== 'object') return null;
      if (raw.nodes && raw.nodes[ROOT]) return raw;
      const t = newThread(raw.title);
      t.id = raw.id || t.id;
      t.autoTitle = false;
      t.updatedAt = raw.updatedAt || t.updatedAt;
      let parent = ROOT;
      for (const m of raw.messages || []) {
        const node = { id: uid('m'), parent, role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content || '', reasoning: m.reasoning || '', images: m.images, telemetry: m.telemetry, children: [], active: 0, createdAt: t.updatedAt };
        t.nodes[node.id] = node;
        t.nodes[parent].children = [node.id];
        parent = node.id;
      }
      return t;
    }

    function loadThreads() {
      let list = [];
      try {
        const v3 = localStorage.getItem(STORAGE.threads);
        const v2 = localStorage.getItem('fllm.studio.threads.v2');
        list = JSON.parse(v3 || v2 || '[]');
      } catch {
        list = [];
      }
      state.threads = list.map(migrateThread).filter(Boolean);
      state.activeThreadId = localStorage.getItem(STORAGE.activeThread) || null;
      if (state.activeThreadId && !state.threads.some((t) => t.id === state.activeThreadId)) state.activeThreadId = null;
    }

    let saveTimer = null;
    let quotaWarned = false;
    function saveThreads() {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(saveThreadsNow, 200);
    }
    function saveThreadsNow() {
      clearTimeout(saveTimer);
      const strip = (t) => ({ ...t, nodes: Object.fromEntries(Object.entries(t.nodes).map(([id, n]) => [id, n.streaming ? { ...n, streaming: false } : n])) });
      try {
        localStorage.setItem(STORAGE.threads, JSON.stringify(state.threads.map(strip)));
        if (state.activeThreadId) localStorage.setItem(STORAGE.activeThread, state.activeThreadId);
        else localStorage.removeItem(STORAGE.activeThread);
      } catch {
        if (!quotaWarned) {
          quotaWarned = true;
          toast('The browser storage is full. Export or delete old chats, or new messages will be lost.');
        }
      }
    }
    window.addEventListener('pagehide', saveThreadsNow);

    const currentThread = () => state.threads.find((t) => t.id === state.activeThreadId) || null;

    /** The messages shown: from the root, always along the active branch. */
    function pathOf(t) {
      const out = [];
      let node = t.nodes[ROOT];
      while (node && node.children.length) {
        const next = t.nodes[node.children[Math.min(node.active, node.children.length - 1)]];
        if (!next) break;
        out.push(next);
        node = next;
      }
      return out;
    }

    function addChild(t, parentId, fields) {
      const node = { id: uid('m'), parent: parentId, children: [], active: 0, createdAt: Date.now(), ...fields };
      t.nodes[node.id] = node;
      const parent = t.nodes[parentId];
      parent.children.push(node.id);
      parent.active = parent.children.length - 1;
      t.updatedAt = Date.now();
      return node;
    }

    function removeSubtree(t, id) {
      const node = t.nodes[id];
      if (!node) return;
      node.children.slice().forEach((c) => removeSubtree(t, c));
      const parent = t.nodes[node.parent];
      if (parent) {
        parent.children = parent.children.filter((c) => c !== id);
        parent.active = Math.max(0, Math.min(parent.active, parent.children.length - 1));
      }
      delete t.nodes[id];
    }

    function copyThread(t) {
      const idMap = {};
      Object.keys(t.nodes).forEach((id) => { idMap[id] = id === ROOT ? ROOT : uid('m'); });
      const copy = { ...t, id: uid('thr'), title: `${t.title} (Kopie)`, createdAt: Date.now(), updatedAt: Date.now(), pinned: false, nodes: {} };
      for (const [id, n] of Object.entries(t.nodes)) {
        copy.nodes[idMap[id]] = { ...n, id: idMap[id], parent: n.parent ? idMap[n.parent] : n.parent, children: n.children.map((c) => idMap[c]) };
      }
      return copy;
    }

    function startNewChat() {
      state.activeThreadId = null;
      prjOnNewChat(); // projects.js: a new chat starts with the chosen assistant
      saveThreads();
      switchSurface('chat');
      renderThreadList();
      renderChatArea();
      $('chatTextarea').focus();
    }

    function ensureActiveThread(firstText) {
      let t = currentThread();
      if (!t) {
        t = newThread((firstText || 'New chat').replace(/\s+/g, ' ').slice(0, 48));
        t.system = $('txtSystemPrompt').value.trim() || null;
        t.preset = state.preset || null;
        prjOnNewThread(t); // projects.js: join the project the chat was started from
        state.threads.unshift(t);
        state.activeThreadId = t.id;
        saveThreads();
        renderThreadList();
      }
      return t;
    }

    function openThread(id) {
      state.activeThreadId = id;
      const t = currentThread();
      if (t && t.system !== null && t.system !== undefined) $('txtSystemPrompt').value = t.system;
      saveThreads();
      switchSurface('chat');
      renderThreadList();
      renderChatArea();
    }

    // ---- thread list --------------------------------------------------------
    function threadText(t) {
      return Object.values(t.nodes).map((n) => n.content || '').join(' ').toLowerCase();
    }

    function dayGroup(ts) {
      const day = (d) => new Date(d).setHours(0, 0, 0, 0);
      const diff = Math.round((day(Date.now()) - day(ts)) / 86400000);
      if (diff <= 0) return 'Today';
      if (diff === 1) return 'Yesterday';
      if (diff < 8) return 'Last 7 days';
      return 'Older';
    }

    function renderThreadList() {
      if (state.tab === 'code') { renderWorkspaceList(); return; }
      const container = $('threadList');
      const q = ($('threadSearch').value || '').toLowerCase().trim();
      // projects.js: chats of projects and archived chats are kept out of the main list (a search finds them all)
      const view = prjSidebarThreads(q);
      const list = view.main;
      if (!list.length && !view.archived.length && !view.archivedCount) {
        container.innerHTML = `<div class="thread-empty">${q ? 'No chat matches the search' : 'No chats yet'}</div>`;
        return;
      }
      const groups = [];
      const pinned = list.filter((t) => t.pinned);
      if (pinned.length) groups.push(['Pinned', pinned]);
      for (const t of list.filter((x) => !x.pinned)) {
        const name = dayGroup(t.updatedAt);
        const g = groups.find((x) => x[0] === name);
        if (g) g[1].push(t); else groups.push([name, [t]]);
      }
      if (view.archived.length) groups.push(['Archived', view.archived]);
      const icon = (d) => `<svg viewBox="0 0 24 24" class="icon icon-sm">${d}</svg>`;
      container.innerHTML = groups.map(([name, items]) => `
        <div class="thread-group">${escapeHtml(name)}</div>
        ${items.map((t) => `
          <div class="thread-row ${t.id === state.activeThreadId && state.surface === 'chat' ? 'active' : ''} ${state.prj.selected.has(t.id) ? 'selected' : ''}" data-tid="${t.id}" draggable="true">
            <input type="checkbox" class="sel-box" ${state.prj.selected.has(t.id) ? 'checked' : ''} tabindex="-1">
            <span class="thread-row-title">${t.pinned ? icon('<path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"/>') : ''}${escapeHtml(t.title)}</span>
            <div class="thread-row-actions">
              <button class="btn-icon btn-icon-xs" data-pin-tid="${t.id}" title="${t.pinned ? 'Unpin' : 'Pin'}">${icon('<path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"/>')}</button>
              <button class="btn-icon btn-icon-xs" data-ren-tid="${t.id}" title="Rename">${icon('<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>')}</button>
              <button class="btn-icon btn-icon-xs" data-arch-tid="${t.id}" title="${t.archived ? 'Unarchive' : 'Archive'}">${icon('<rect width="20" height="5" x="2" y="3" rx="1"/><path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8"/><path d="M10 12h4"/>')}</button>
              <button class="btn-icon btn-icon-xs" data-del-tid="${t.id}" title="Delete">${icon('<path d="M18 6 6 18"/><path d="m6 6 12 12"/>')}</button>
            </div>
          </div>`).join('')}
      `).join('') + prjArchiveToggleHtml(view.archivedCount, q);
    }

    // ---- dialogs ------------------------------------------------------------
    function askDialog({ title, text = '', input = null, ok = 'OK', danger = false }) {
      return new Promise((resolve) => {
        $('dialogTitle').textContent = title;
        $('dialogText').textContent = text;
        $('dialogText').style.display = text ? 'block' : 'none';
        const field = $('dialogInput');
        field.style.display = input === null ? 'none' : 'block';
        field.value = input ?? '';
        $('dialogOk').textContent = ok;
        $('dialogOk').className = `btn ${danger ? 'btn-destructive' : 'btn-primary'}`;
        $('dialogModal').classList.add('open');
        (input === null ? $('dialogOk') : field).focus();
        const done = (value) => {
          $('dialogModal').classList.remove('open');
          $('dialogOk').onclick = $('dialogCancel').onclick = field.onkeydown = null;
          resolve(value);
        };
        $('dialogOk').onclick = () => done(input === null ? true : field.value.trim());
        $('dialogCancel').onclick = () => done(null);
        field.onkeydown = (e) => {
          if (e.key === 'Enter') done(field.value.trim());
          if (e.key === 'Escape') done(null);
        };
      });
    }

    // ---- rendering ----------------------------------------------------------
    function splitReasoning(text) {
      const open = text.indexOf('<think>');
      if (open === -1) return { reasoning: '', answer: text };
      const close = text.indexOf('</think>');
      if (close === -1) return { reasoning: text.slice(open + 7).trim(), answer: text.slice(0, open).trim() };
      return { reasoning: text.slice(open + 7, close).trim(), answer: (text.slice(0, open) + text.slice(close + 8)).trim() };
    }

    const ICON = {
      copy: '<rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
      edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
      redo: '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
      trash: '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>',
      prev: '<path d="m15 18-6-6 6-6"/>',
      next: '<path d="m9 18 6-6-6-6"/>',
      file: '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5z"/><path d="M14 2v6h6"/>',
      brain: '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>',
    };
    const svgIcon = (name, cls = 'icon icon-sm') => `<svg viewBox="0 0 24 24" class="${cls}">${ICON[name]}</svg>`;

    function branchNav(t, node) {
      const parent = t.nodes[node.parent];
      if (!parent || parent.children.length < 2) return '';
      const i = parent.children.indexOf(node.id);
      return `<span class="branch-nav"><button class="btn-icon btn-icon-xs" data-branch="${node.id}" data-dir="-1" ${i === 0 ? 'disabled' : ''} title="Previous version">${svgIcon('prev')}</button>` +
        `<span class="tabular">${i + 1} / ${parent.children.length}</span>` +
        `<button class="btn-icon btn-icon-xs" data-branch="${node.id}" data-dir="1" ${i === parent.children.length - 1 ? 'disabled' : ''} title="Next version">${svgIcon('next')}</button></span>`;
    }

    function attachmentsHtml(m) {
      const imgs = (m.images || []).map((src) => `<img src="${src}" alt="Image" class="msg-img">`).join('');
      const files = (m.files || []).map((f) => `<span class="file-chip" title="${escapeHtml(f.name)}">${svgIcon('file')}<span>${escapeHtml(f.name)}</span></span>`).join('');
      return imgs || files ? `<div class="msg-attachments">${imgs}${files}</div>` : '';
    }

    function statsHtml(m) {
      if (!state.showSpeed || !m.telemetry) return '';
      const tel = m.telemetry;
      const bits = [];
      if (tel.tokPerSec) bits.push(`${tel.tokPerSec} tokens/s`);
      if (tel.tokens) bits.push(`${tel.tokens} tokens`);
      if (tel.ttftMs) bits.push(`first token after ${tel.ttftMs} ms`);
      if (tel.promptPerSec) bits.push(`prompt ${tel.promptPerSec} tokens/s`);
      if (m.finish === 'length') bits.push('stopped at the length limit');
      return bits.length ? `<span class="speed-chip">${bits.map(escapeHtml).join('  ·  ')}</span>` : '';
    }

    function assistantInner(m) {
      const parsed = splitReasoning(m.content || '');
      const reasoning = m.reasoning || parsed.reasoning;
      const answer = m.reasoning ? (m.content || '') : parsed.answer;
      const thinking = reasoning ? `
        <details class="cot-box" ${m.streaming && !answer ? 'open' : ''}>
          <summary>${svgIcon('brain')}<span>${m.streaming && !answer ? 'Thinking' : 'Thought process'} (${reasoning.length} characters)</span></summary>
          <div class="cot-body">${escapeHtml(reasoning)}</div>
        </details>` : '';
      const body = answer ? MD.render(answer) : (m.streaming ? '<span class="typing"><i></i><i></i><i></i></span>' : '');
      const err = m.error ? `<div class="msg-error">${escapeHtml(m.error)}</div>` : '';
      return `${thinking}<div class="msg-assistant-prose">${body}</div>${err}`;
    }

    function logprobsHtml(m) {
      if (!Array.isArray(m.logprobs) || !m.logprobs.length) return '';
      return `<div class="logprobs-wrap">${m.logprobs.map((lp) => {
        const p = Math.exp(Number(lp.logprob || 0));
        return `<span class="lp-tok" style="background:hsla(${Math.round(p * 130)},70%,42%,0.22);" title="Probability ${(p * 100).toFixed(1)}%">${escapeHtml(lp.token)}</span>`;
      }).join('')}</div>`;
    }

    function messageHtml(t, node, isLast) {
      if (node.role === 'user') {
        return `
          <div class="msg-user-wrap" data-node="${node.id}">
            <div class="msg-user-bubble">${attachmentsHtml(node)}${escapeHtml(node.content)}</div>
            ${ragChipsHtml(node)}
            <div class="msg-actions-bar">
              ${branchNav(t, node)}
              <button class="btn-icon btn-icon-xs" data-act="copy" title="Copy">${svgIcon('copy')}</button>
              <button class="btn-icon btn-icon-xs" data-act="edit" title="Edit and resend">${svgIcon('edit')}</button>
              <button class="btn-icon btn-icon-xs" data-act="delete" title="Delete this message and everything after it">${svgIcon('trash')}</button>
            </div>
          </div>`;
      }
      return `
        <div class="msg-assistant-wrap" data-node="${node.id}">
          <div class="msg-assistant-body">${assistantInner(node)}</div>
          ${logprobsHtml(node)}
          ${ragSourcesHtml(t, node)}
          <div class="msg-actions-bar">
            ${branchNav(t, node)}
            <button class="btn-icon btn-icon-xs" data-act="copy" title="Copy the answer">${svgIcon('copy')}</button>
            <button class="btn-icon btn-icon-xs" data-act="regen" title="Regenerate (the old answer stays as a version)">${svgIcon('redo')}</button>
            <button class="btn-icon btn-icon-xs" data-act="delete" title="Delete the answer">${svgIcon('trash')}</button>
            <span class="msg-stats" data-stats>${statsHtml(node)}</span>
          </div>
        </div>`;
    }

    function renderChatArea() {
      const t = currentThread();
      const path = t ? pathOf(t) : [];
      $('chatContainer').classList.toggle('is-empty', path.length === 0);
      $('messagesColumn').innerHTML = path.map((n, i) => messageHtml(t, n, i === path.length - 1)).join('');
      updateChatChrome();
      if (path.length) scrollChatToEnd(true);
    }

    /** Header: pin label, context fill. */
    function updateChatChrome() {
      const t = currentThread();
      $('menuPin').textContent = t && t.pinned ? 'Unpin' : 'Pin';
      prjChrome(); // projects.js: assistant label, project badge, suggestions
      updateContextMeter();
      renderAttachments();
    }

    const estimateTokens = (text) => Math.ceil(String(text || '').length / 3.6);

    function contextUsed(t) {
      if (!t) return 0;
      const path = pathOf(t);
      let total = estimateTokens(prjComposeSystem(t, $('txtSystemPrompt').value.trim())) + 8;
      const lastUser = path.filter((n) => n.role === 'user').pop();
      for (const n of path) total += estimateTokens(n.content) + 6 + (n.images ? n.images.length * 256 : 0) + (n.files || []).reduce((s, f) => s + estimateTokens(f.text), 0) + (n === lastUser ? ragTokens(n) : 0);
      return total;
    }

    function setMeter(fillId, textId, used, max) {
      const ratio = Math.min(1, used / Math.max(1, max));
      $(fillId).style.width = `${(ratio * 100).toFixed(0)}%`;
      $(fillId).parentElement.parentElement.classList.toggle('is-high', ratio > 0.8);
      $(textId).textContent = `${used.toLocaleString('de-DE')} / ${max.toLocaleString('de-DE')}`;
    }

    function updateContextMeter() {
      const max = state.status?.model?.max_context || 4096;
      setMeter('ctxFill', 'ctxText', contextUsed(currentThread()), max);
    }

    function scrollChatToEnd(force) {
      const sc = $('chatMessagesScroll');
      if (force || state.stick) sc.scrollTop = sc.scrollHeight;
    }
    state.stick = true;
    $('chatMessagesScroll').addEventListener('scroll', () => {
      const sc = $('chatMessagesScroll');
      state.stick = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 90;
    });

    /** Streaming: rewrite only the message that grows. */
    let paintQueued = false;
    function paintStreaming(node) {
      if (paintQueued) return;
      paintQueued = true;
      requestAnimationFrame(() => {
        paintQueued = false;
        const el = $('messagesColumn').querySelector(`[data-node="${node.id}"] .msg-assistant-body`);
        if (!el) return;
        const wasOpen = el.querySelector('details.cot-box')?.open;
        el.innerHTML = assistantInner(node);
        if (wasOpen === false) { const d = el.querySelector('details.cot-box'); if (d) d.open = false; }
        const stats = $('messagesColumn').querySelector(`[data-node="${node.id}"] [data-stats]`);
        if (stats) stats.innerHTML = statsHtml(node);
        scrollChatToEnd(false);
      });
    }

    // ---- building the request ------------------------------------------------
    function userContent(node, withContext) {
      const files = (node.files || []).map((f) => `File ${f.name}:\n\`\`\`\n${f.text}\n\`\`\`\n\n`).join('');
      const context = withContext ? ragPrefix(node) : '';
      const text = context + files + (node.content || (context ? 'Summarize the key points.' : ''));
      if (node.images && node.images.length) {
        const parts = [{ type: 'text', text: text || 'Describe the image.' }];
        node.images.forEach((url) => parts.push({ type: 'image_url', image_url: { url } }));
        return parts;
      }
      return text;
    }

    function apiMessages(t, upToNodeId) {
      const out = [];
      const system = prjComposeSystem(t, $('txtSystemPrompt').value.trim()); // projects.js: personalization + project + assistant
      if (system) out.push({ role: 'system', content: system });
      const path = pathOf(t);
      const end = upToNodeId ? path.findIndex((n) => n.id === upToNodeId) : -1;
      const used = end >= 0 ? path.slice(0, end + 1) : path;
      // passages from documents and the web only travel with the newest question
      const lastUser = used.filter((n) => n.role === 'user').pop();
      for (const n of used) out.push({ role: n.role, content: n.role === 'user' ? userContent(n, n === lastUser) : (n.content || '') });
      return out;
    }

    function requestBody(messages) {
      const p = getSampler();
      const body = {
        model: state.selectedModel || state.status?.model?.name || 'tiny',
        messages, stream: true, stream_options: { include_usage: true },
        temperature: p.temperature, top_p: p.top_p, top_k: p.top_k, min_p: p.min_p, max_tokens: p.max_tokens,
        repetition_penalty: p.repetition_penalty, presence_penalty: p.presence_penalty, frequency_penalty: p.frequency_penalty,
        speculative: p.speculative,
      };
      if (p.seed >= 0) body.seed = p.seed;
      if (p.stop.length) body.stop = p.stop;
      if (p.dry_multiplier > 0) body.dry_multiplier = p.dry_multiplier;
      if (p.mirostat > 0) body.mirostat = p.mirostat;
      if (p.logprobs) { body.logprobs = true; body.top_logprobs = 5; }
      if (state.thinkingEnabled) body.reasoning_effort = 'high';
      if (p.constraint === 'json_object') body.response_format = { type: 'json_object' };
      else if (p.constraint === 'json_schema' && p.constraint_spec) {
        try { body.response_format = { type: 'json_schema', json_schema: { name: 'schema', strict: true, schema: JSON.parse(p.constraint_spec) } }; } catch { /* the schema box holds invalid JSON: ignore it */ }
      } else if (p.constraint === 'gbnf' && p.constraint_spec) { body.grammar = p.constraint_spec; body.grammar_type = 'gbnf'; }
      else if (p.constraint === 'regex' && p.constraint_spec) body.regex = p.constraint_spec;
      return body;
    }

    /** Reads an OpenAI-style event stream; calls `onChunk(json)` for every data line. */
    async function readEventStream(res, onChunk) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        for (const raw of lines) {
          const line = raw.trim();
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') return;
          try { onChunk(JSON.parse(payload)); } catch { /* a broken line is skipped */ }
        }
      }
    }

    function setBusy(busy) {
      const btn = $('btnSendChat');
      btn.classList.toggle('stop', busy);
      btn.title = busy ? 'Stop the answer (Esc)' : 'Send (Enter)';
      btn.innerHTML = busy
        ? '<svg viewBox="0 0 24 24" class="icon icon-sm"><rect width="10" height="10" x="7" y="7" rx="1" fill="currentColor"/></svg>'
        : '<svg viewBox="0 0 24 24" class="icon icon-sm"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>';
    }

    /** Generate the answer to the user message `userId` as a new assistant node. */
    async function generateReply(t, userId) {
      const aNode = addChild(t, userId, { role: 'assistant', content: '', reasoning: '', streaming: true, logprobs: [], model: state.selectedModel || state.status?.model?.name || '' });
      saveThreads();
      renderThreadList();
      renderChatArea();
      state.abortController = new AbortController();
      setBusy(true);
      const t0 = performance.now();
      let first = null;
      let chunks = 0;
      try {
        const res = await apiFetch('/v1/chat/completions', { method: 'POST', body: requestBody(apiMessages(t, userId)), signal: state.abortController.signal });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.error?.message || `The server answered with error ${res.status}`);
        }
        await readEventStream(res, (chunk) => {
          const choice = chunk.choices?.[0];
          const delta = choice?.delta;
          if (delta) {
            if (first === null && (delta.content || delta.reasoning_content)) first = performance.now();
            if (delta.reasoning_content) aNode.reasoning += delta.reasoning_content;
            if (delta.content) { aNode.content += delta.content; chunks++; }
          }
          const lp = choice?.logprobs?.content;
          if (Array.isArray(lp)) aNode.logprobs.push(...lp);
          if (choice?.finish_reason) aNode.finish = choice.finish_reason;
          if (chunk.timings) {
            aNode.telemetry = {
              ...(aNode.telemetry || {}),
              tokPerSec: chunk.timings.predicted_per_second ? chunk.timings.predicted_per_second.toFixed(1) : undefined,
              promptPerSec: chunk.timings.prompt_per_second ? Math.round(chunk.timings.prompt_per_second) : undefined,
              tokens: chunk.timings.predicted_n || undefined,
            };
          }
          if (chunk.usage) aNode.usage = chunk.usage;
          paintStreaming(aNode);
        });
        const elapsed = Math.max((performance.now() - (first || t0)) / 1000, 0.02);
        aNode.telemetry = {
          ttftMs: first ? Math.round(first - t0) : undefined,
          tokens: Math.max(chunks, 1),
          tokPerSec: (Math.max(chunks, 1) / elapsed).toFixed(1),
          ...(aNode.telemetry || {}),
        };
        if (!aNode.telemetry.ttftMs && first) aNode.telemetry.ttftMs = Math.round(first - t0);
      } catch (e) {
        if (e.name !== 'AbortError') aNode.error = e.message;
      } finally {
        aNode.streaming = false;
        state.abortController = null;
        setBusy(false);
        t.updatedAt = Date.now();
        saveThreads();
        renderChatArea();
        refreshStudioStatus();
        if (aNode.usage?.total_tokens) setMeter('ctxFill', 'ctxText', aNode.usage.total_tokens, state.status?.model?.max_context || 4096);
        maybeTitle(t);
      }
    }

    /** Gives a new chat a short title made by the model itself. */
    async function maybeTitle(t) {
      if (!t.autoTitle || !state.autoTitle) return;
      const path = pathOf(t);
      const first = path.find((n) => n.role === 'user');
      const reply = path.find((n) => n.role === 'assistant' && n.content);
      if (!first || !reply) return;
      t.autoTitle = false;
      try {
        const res = await apiFetch('/v1/chat/completions', {
          method: 'POST',
          body: {
            model: state.selectedModel || state.status?.model?.name || 'tiny', stream: false, temperature: 0.3, max_tokens: 20,
            messages: [
              { role: 'system', content: 'You give conversations short titles. Answer with the title only, at most five words, without quotation marks and without punctuation at the end.' },
              { role: 'user', content: `Question: ${first.content.slice(0, 400)}\n\nAnswer: ${reply.content.slice(0, 400)}\n\nTitle:` },
            ],
          },
        });
        const data = await res.json();
        const title = String(data.choices?.[0]?.message?.content || '').replace(/<think>[\s\S]*?<\/think>/g, '').replace(/["*#\n]/g, ' ').replace(/\s+/g, ' ').trim().replace(/[.!?:]+$/, '');
        if (title && title.length >= 3 && title.length <= 60) {
          t.title = title;
          saveThreads();
          renderThreadList();
        }
      } catch { /* the first words of the question stay as the title */ }
    }

    async function sendChatPrompt() {
      if (state.abortController) { state.abortController.abort(); return; }
      const raw = $('chatTextarea').value;
      const text = raw.trim();
      if (!text && !state.attachments.length) return;
      if (state.setup) { toast('Load a model first.'); switchSurface('hub'); return; }
      if (text.startsWith('/') && state.attachments.length === 0 && (await runSlashCommand(text))) return;
      prjCaptureMemory(text); // projects.js: "remember that ..." saves a memory
      $('chatTextarea').value = '';
      $('chatTextarea').style.height = 'auto';
      const t = ensureActiveThread(text || state.attachments[0]?.name);
      const path = pathOf(t);
      const parent = path.length ? path[path.length - 1].id : ROOT;
      const images = state.attachments.filter((a) => a.kind === 'image').map((a) => a.dataUrl);
      const files = state.attachments.filter((a) => a.kind === 'text').map((a) => ({ name: a.name, text: a.text }));
      const userNode = addChild(t, parent, { role: 'user', content: text, images: images.length ? images : undefined, files: files.length ? files : undefined });
      const newDocs = state.attachments.filter((a) => a.kind === 'doc');
      if (newDocs.length) { ragStoreDocs(t, newDocs); userNode.ragFresh = true; }
      state.attachments = [];
      renderAttachments();
      renderChatArea();
      if (!(await ragPrepare(t, userNode))) return;
      await generateReply(t, userNode.id);
    }

    async function regenerate(nodeId) {
      const t = currentThread();
      const node = t?.nodes[nodeId];
      if (!t || !node || state.abortController) return;
      await generateReply(t, node.parent);
    }

    async function editUserMessage(nodeId, newText) {
      const t = currentThread();
      const old = t?.nodes[nodeId];
      if (!t || !old || state.abortController) return;
      const copy = addChild(t, old.parent, { role: 'user', content: newText, images: old.images, files: old.files, ragFresh: old.ragFresh });
      if (!(await ragPrepare(t, copy))) return;
      await generateReply(t, copy.id);
    }

    function startEditing(nodeId) {
      const t = currentThread();
      const node = t?.nodes[nodeId];
      const wrap = $('messagesColumn').querySelector(`[data-node="${nodeId}"]`);
      if (!node || !wrap || state.abortController) return;
      wrap.innerHTML = `<div class="msg-edit"><textarea class="ui-textarea" rows="3">${escapeHtml(node.content)}</textarea>` +
        `<div class="msg-edit-actions"><button class="btn btn-ghost btn-sm" data-edit-cancel>Cancel</button><button class="btn btn-primary btn-sm" data-edit-save="${nodeId}">Send</button></div></div>`;
      const ta = wrap.querySelector('textarea');
      ta.style.height = `${Math.min(ta.scrollHeight + 4, 320)}px`;
      ta.focus();
      ta.setSelectionRange(ta.value.length, ta.value.length);
      ta.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); wrap.querySelector('[data-edit-save]').click(); }
        if (e.key === 'Escape') { e.stopPropagation(); renderChatArea(); }
      });
    }

    async function deleteMessage(nodeId) {
      const t = currentThread();
      if (!t || state.abortController) return;
      const ok = await askDialog({ title: 'Delete the message?', text: 'The message and everything after it in this version will be deleted.', ok: 'Delete', danger: true });
      if (!ok) return;
      removeSubtree(t, nodeId);
      saveThreads();
      renderChatArea();
      renderThreadList();
    }

    function messageText(node) {
      return node.role === 'user' ? node.content : splitReasoning(node.content || '').answer || node.content || '';
    }

    // ---- attachments ------------------------------------------------------------
    function renderAttachments() {
      $('attachmentsRow').innerHTML = ragDocsChipsHtml() + state.attachments.map((a, i) => (a.kind === 'image'
        ? `<div class="att-item"><img src="${a.dataUrl}" alt="${escapeHtml(a.name)}"><button class="att-remove" data-rm-att="${i}" title="Remove">✕</button></div>`
        : `<div class="file-chip removable" title="${escapeHtml(a.name)} (${formatBytes(a.size)})${a.kind === 'doc' ? ': too long for the context, the passages that match your question are used' : ''}">${svgIcon('file')}<span>${escapeHtml(a.name)}</span><button class="att-x" data-rm-att="${i}" title="Remove">✕</button></div>`)).join('');
    }

    function shrinkImage(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = reject;
        reader.onload = () => {
          const img = new Image();
          img.onerror = () => resolve(reader.result);
          img.onload = () => {
            const scale = Math.min(1, 1280 / Math.max(img.width, img.height));
            if (scale === 1 && file.size < 300 * 1024) { resolve(reader.result); return; }
            const canvas = document.createElement('canvas');
            canvas.width = Math.round(img.width * scale);
            canvas.height = Math.round(img.height * scale);
            canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
            resolve(canvas.toDataURL('image/jpeg', 0.84));
          };
          img.src = reader.result;
        };
        reader.readAsDataURL(file);
      });
    }

    async function addFiles(fileList) {
      for (const file of Array.from(fileList || [])) {
        if (file.type.startsWith('image/')) {
          try { state.attachments.push({ kind: 'image', name: file.name || 'Image', dataUrl: await shrinkImage(file) }); } catch { toast(`${file.name}: the image could not be read`); }
        } else if (RAG_SERVER_EXT.test(file.name)) {
          toast(`Reading ${file.name}...`);
          try {
            const doc = await ragExtract(file);
            ragPlace(file.name, doc.text, file.size, { pages: doc.pages, truncated: doc.truncated });
          } catch (e) { toast(e.message); }
        } else if (file.size > MAX_TEXT_ATTACHMENT) {
          toast(`${file.name} is too large (more than ${formatBytes(MAX_TEXT_ATTACHMENT)}).`);
        } else {
          const text = await file.text();
          if (text.includes('\u0000')) { toast(`${file.name} is not a text file.`); continue; }
          ragPlace(file.name, text, file.size);
        }
      }
      renderAttachments();
    }

    // ---- presets ----------------------------------------------------------------
    const BUILTIN_PRESETS = [
      { id: 'standard', name: 'Standard', system: '', temperature: 0.7, top_p: 0.95, top_k: 20 },
      { id: 'precise', name: 'Precise and brief', system: 'Answer briefly and factually, and only with what you are sure about. Say so when you do not know something.', temperature: 0.2, top_p: 0.9, top_k: 20 },
      { id: 'creative', name: 'Creative', system: 'You are an inventive writing partner. Offer surprising ideas and vivid language.', temperature: 1.0, top_p: 0.95, top_k: 60 },
      { id: 'programmer', name: 'Programmer', system: 'You are an experienced software developer. Answer with working code and a short explanation. State your assumptions explicitly.', temperature: 0.2, top_p: 0.9, top_k: 20 },
      { id: 'translator', name: 'Translator English/German', system: "Translate the user's text between English and German. Output only the translation, without comments.", temperature: 0.1, top_p: 0.9, top_k: 20 },
      { id: 'editor', name: 'Editor', system: 'Improve the spelling, grammar and style of the text. Then show in a short list what you changed.', temperature: 0.3, top_p: 0.9, top_k: 20 },
    ];

    const customPresets = () => { try { return JSON.parse(localStorage.getItem(PRESET_KEY) || '[]'); } catch { return []; } };

    // Presets are assistants now (projects.js): the old custom presets are migrated once and shown as assistants,
    // the stored list under PRESET_KEY is only read for that migration.
    const renderPresets = () => asstRender();
    const applyPreset = (id) => asstApply(id);

    // ---- export, import ------------------------------------------------------------
    function download(name, mime, content) {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([content], { type: mime }));
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    }
    const fileSafe = (s) => String(s).replace(/[^\p{L}\p{N}\-. ]+/gu, '_').slice(0, 60) || 'chat';

    function threadMarkdown(t) {
      const lines = [`# ${t.title}`, ''];
      const system = t.system || '';
      if (system) lines.push(`> Systemprompt: ${system.replace(/\n/g, ' ')}`, '');
      for (const n of pathOf(t)) {
        lines.push(n.role === 'user' ? '## You' : '## Model', '', messageText(n), '');
      }
      return lines.join('\n');
    }

    async function chatAction(action) {
      const t = currentThread();
      if (!t) { toast('Open a chat first.'); return; }
      if (prjChatAction(action, t)) return; // projects.js: archive, move to a project, personalization
      if (action === 'rename') {
        const name = await askDialog({ title: 'Rename chat', input: t.title, ok: 'Rename' });
        if (name) { t.title = name.slice(0, 80); t.autoTitle = false; saveThreads(); renderThreadList(); }
      } else if (action === 'pin') {
        t.pinned = !t.pinned; saveThreads(); renderThreadList(); updateChatChrome();
      } else if (action === 'duplicate') {
        const copy = copyThread(t); state.threads.unshift(copy); openThread(copy.id); toast('Chat duplicated');
      } else if (action === 'export-md') {
        download(`${fileSafe(t.title)}.md`, 'text/markdown', threadMarkdown(t));
      } else if (action === 'export-json') {
        download(`${fileSafe(t.title)}.json`, 'application/json', JSON.stringify([t], null, 2));
      } else if (action === 'delete') {
        deleteThread(t.id);
      }
    }

    async function deleteThread(id) {
      const t = state.threads.find((x) => x.id === id);
      if (!t) return;
      if (!(await askDialog({ title: 'Delete chat?', text: `"${t.title}" will be deleted for good.`, ok: 'Delete', danger: true }))) return;
      state.threads = state.threads.filter((x) => x.id !== id);
      if (state.activeThreadId === id) state.activeThreadId = null;
      saveThreads();
      renderThreadList();
      renderChatArea();
    }

    function importThreads(text) {
      let list;
      try { list = JSON.parse(text); } catch { toast('The file is not valid JSON.'); return; }
      if (!Array.isArray(list)) list = [list];
      let added = 0;
      for (const raw of list) {
        const t = migrateThread(raw);
        if (!t) continue;
        if (state.threads.some((x) => x.id === t.id)) t.id = uid('thr');
        state.threads.push(t);
        added++;
      }
      saveThreads();
      renderThreadList();
      toast(added ? `${added} chat${added === 1 ? '' : 's'} imported` : 'No chats were found in the file.');
    }

    // ---- slash commands ----------------------------------------------------------------
    async function runSlashCommand(text) {
      const [cmd, ...rest] = text.slice(1).split(/\s+/);
      const arg = rest.join(' ').trim();
      const done = () => { $('chatTextarea').value = ''; $('chatTextarea').style.height = 'auto'; return true; };
      switch (cmd.toLowerCase()) {
        case 'clear': case 'new': startNewChat(); return done();
        case 'system':
          $('txtSystemPrompt').value = arg;
          { const t = currentThread(); if (t) { t.system = arg || null; saveThreads(); } }
          updateContextMeter();
          toast(arg ? 'System prompt set' : 'System prompt removed');
          return done();
        case 'export': await chatAction('export-md'); return done();
        case 'help': $('helpModal').classList.add('open'); return done();
        case 'retry': {
          const t = currentThread();
          const last = t && pathOf(t).filter((n) => n.role === 'assistant').pop();
          if (last) { done(); await regenerate(last.id); }
          return true;
        }
        default: return false;
      }
    }

    // ---- events of the chat surface ---------------------------------------------------------
    $('threadSearch').addEventListener('input', renderThreadList);

    $('threadList').addEventListener('click', async (e) => {
      if (prjThreadListClick(e)) return; // projects.js: select mode, archive
      const del = e.target.closest('[data-del-tid]');
      if (del) { e.stopPropagation(); deleteThread(del.dataset.delTid); return; }
      const pin = e.target.closest('[data-pin-tid]');
      if (pin) {
        e.stopPropagation();
        const t = state.threads.find((x) => x.id === pin.dataset.pinTid);
        if (t) { t.pinned = !t.pinned; saveThreads(); renderThreadList(); }
        return;
      }
      const ren = e.target.closest('[data-ren-tid]');
      if (ren) {
        e.stopPropagation();
        const t = state.threads.find((x) => x.id === ren.dataset.renTid);
        const name = t && await askDialog({ title: 'Rename chat', input: t.title, ok: 'Rename' });
        if (name) { t.title = name.slice(0, 80); t.autoTitle = false; saveThreads(); renderThreadList(); }
        return;
      }
      const row = e.target.closest('[data-tid]');
      if (row) openThread(row.dataset.tid);
    });

    $('messagesColumn').addEventListener('click', async (e) => {
      const copyCode = e.target.closest('[data-copy-code]');
      if (copyCode) {
        const pre = $(copyCode.dataset.copyCode);
        if (pre) { await navigator.clipboard.writeText(pre.textContent); toast('Code copied'); }
        return;
      }
      const wrapCode = e.target.closest('[data-wrap-code]');
      if (wrapCode) { $(wrapCode.dataset.wrapCode)?.classList.toggle('wrap'); return; }
      const branch = e.target.closest('[data-branch]');
      if (branch) {
        const t = currentThread();
        const node = t?.nodes[branch.dataset.branch];
        const parent = node && t.nodes[node.parent];
        if (parent && !state.abortController) {
          parent.active = Math.max(0, Math.min(parent.children.length - 1, parent.children.indexOf(node.id) + Number(branch.dataset.dir)));
          saveThreads();
          renderChatArea();
        }
        return;
      }
      const save = e.target.closest('[data-edit-save]');
      if (save) {
        const text = save.closest('.msg-edit').querySelector('textarea').value.trim();
        if (text) await editUserMessage(save.dataset.editSave, text);
        return;
      }
      if (e.target.closest('[data-edit-cancel]')) { renderChatArea(); return; }
      const act = e.target.closest('[data-act]');
      if (!act) return;
      const id = act.closest('[data-node]').dataset.node;
      const t = currentThread();
      const node = t?.nodes[id];
      if (!node) return;
      if (act.dataset.act === 'copy') { await navigator.clipboard.writeText(messageText(node)); toast('Copied'); }
      else if (act.dataset.act === 'edit') startEditing(id);
      else if (act.dataset.act === 'regen') await regenerate(id);
      else if (act.dataset.act === 'delete') await deleteMessage(id);
    });

    $('chatMenu').addEventListener('click', (e) => {
      const item = e.target.closest('[data-chat-action]');
      if (!item) return;
      $('chatMenu').classList.remove('open');
      chatAction(item.dataset.chatAction);
    });
    $('btnChatMenu').addEventListener('click', (e) => { e.stopPropagation(); $('chatMenu').classList.toggle('open'); });

    // (the assistant menu is wired in projects.js)

    $('txtSystemPrompt').addEventListener('input', () => {
      const t = currentThread();
      if (t) { t.system = $('txtSystemPrompt').value.trim() || null; t.preset = null; saveThreads(); }
      state.preset = null;
      renderPresets();
      updateContextMeter();
    });

    // composer
    let tokTimer = null;
    $('chatTextarea').addEventListener('input', () => {
      const el = $('chatTextarea');
      el.style.height = 'auto';
      el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
      clearTimeout(tokTimer);
      tokTimer = setTimeout(async () => {
        const max = state.status?.model?.max_context || 4096;
        if (!el.value.trim()) { $('liveTokenCounter').textContent = `0 / ${max} tokens`; return; }
        try {
          const r = await apiFetch('/tokenize', { method: 'POST', body: { content: el.value } });
          if (r.ok) $('liveTokenCounter').textContent = `${((await r.json()).tokens || []).length} / ${max} tokens`;
        } catch { $('liveTokenCounter').textContent = `about ${estimateTokens(el.value)} / ${max} tokens`; }
      }, 250);
    });
    $('chatTextarea').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendChatPrompt(); }
      if (e.key === 'ArrowUp' && !$('chatTextarea').value) {
        const t = currentThread();
        const last = t && pathOf(t).filter((n) => n.role === 'user').pop();
        if (last) { e.preventDefault(); startEditing(last.id); }
      }
    });
    $('chatTextarea').addEventListener('paste', async (e) => {
      const items = Array.from(e.clipboardData?.items || []);
      const files = items.filter((i) => i.kind === 'file').map((i) => i.getAsFile()).filter(Boolean);
      if (files.length) { e.preventDefault(); await addFiles(files); return; }
      const text = e.clipboardData?.getData('text/plain') || '';
      if (text.length > PASTE_AS_FILE) {
        e.preventDefault();
        ragPlace('Pasted text.txt', text, text.length);
        renderAttachments();
        toast('Long text attached as a file');
      }
    });
    $('btnSendChat').addEventListener('click', sendChatPrompt);

    $('starterPills').addEventListener('click', (e) => {
      const pill = e.target.closest('.starter-pill');
      if (!pill) return;
      if (pill.dataset.prompt) { $('chatTextarea').value = pill.dataset.prompt; sendChatPrompt(); }
      else if (pill.dataset.action === 'open-keys') switchSurface('settings', 'local-api-server');
      else if (pill.dataset.action === 'open-hub') switchSurface('hub');
    });

    $('btnAttach').addEventListener('click', () => $('fileAttachInput').click());
    $('fileAttachInput').addEventListener('change', async (e) => { await addFiles(e.target.files); e.target.value = ''; });
    $('attachmentsRow').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-rm-att]');
      if (btn) { state.attachments.splice(Number(btn.dataset.rmAtt), 1); renderAttachments(); }
    });

    // dropping files anywhere on the chat
    let dragDepth = 0;
    $('chatContainer').addEventListener('dragenter', (e) => { if (e.dataTransfer?.types?.includes('Files')) { dragDepth++; $('dropOverlay').classList.add('show'); } });
    $('chatContainer').addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $('dropOverlay').classList.remove('show'); });
    $('chatContainer').addEventListener('dragover', (e) => { if (e.dataTransfer?.types?.includes('Files')) e.preventDefault(); });
    $('chatContainer').addEventListener('drop', async (e) => {
      e.preventDefault();
      dragDepth = 0;
      $('dropOverlay').classList.remove('show');
      await addFiles(e.dataTransfer.files);
    });

    $('btnExportThreads').addEventListener('click', () => download(`fllm-chats-${new Date().toISOString().slice(0, 10)}.json`, 'application/json', JSON.stringify(state.threads, null, 2)));
    $('btnImportThreads').addEventListener('click', () => $('fileImportThreads').click());
    $('fileImportThreads').addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (file) importThreads(await file.text());
      e.target.value = '';
    });
    $('btnDeleteAllThreads').addEventListener('click', async () => {
      if (!state.threads.length) return;
      if (!(await askDialog({ title: 'Delete all chats?', text: `${state.threads.length} chats will be deleted for good. Export them first if you want to keep them.`, ok: 'Delete all', danger: true }))) return;
      state.threads = [];
      state.activeThreadId = null;
      saveThreads();
      renderThreadList();
      renderChatArea();
      toast('All chats deleted');
    });
    $('btnFactoryResetThreads').addEventListener('click', () => $('btnDeleteAllThreads').click());
