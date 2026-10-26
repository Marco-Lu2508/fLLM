    // ==========================================================================
    // Search over past chats and code sessions. Both live in the browser (localStorage); this file builds a BM25 index
    // over their messages in the page and keeps a text copy on the computer (POST /api/studio/sessions/index) so that
    // they can still be found after the browser cache was cleared.
    // Exposed: fllmSessionSearch(query, opts) -> Promise<hits>, fllmSessionIndexNow(), fllmSessionForget(id),
    // fllmSessionIndexInfo(). A hit is {id, threadId, kind: 'chat'|'code', title, snippet, time, score, source:
    // 'browser'|'server', workspace?}. Chat ids are thread ids; code sessions have the id "code:<session id>".
    // ==========================================================================

    // @pure-begin
    const SS_MAX_TEXT = 200000;

    /** Lowercase words of letters and digits, at least two characters. */
    function ssTokenize(text) {
      return String(text || '').toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((w) => w.length >= 2);
    }

    /** The text of a message content (a string, or a list of parts with `text`). */
    function ssMessageText(content) {
      if (typeof content === 'string') return content;
      if (Array.isArray(content)) return content.map((p) => (p && typeof p.text === 'string' ? p.text : '')).join(' ');
      return '';
    }

    /** `{id, kind, title, text, updated}` for a chat thread: every message of every branch, without reasoning. */
    function ssThreadDoc(t) {
      const nodes = Object.values(t.nodes || {}).filter((n) => n && (n.role === 'user' || n.role === 'assistant') && n.content)
        .sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
      let text = nodes.map((n) => ssMessageText(n.content)).join('\n');
      if (text.length > SS_MAX_TEXT) text = text.slice(0, SS_MAX_TEXT);
      return { id: t.id, kind: 'chat', title: t.title || '', text, updated: t.updatedAt || 0 };
    }

    /** The same for a saved code session of the workspace `root`. Tool results are left out. */
    function ssCodeDoc(s, root) {
      let text = (s.messages || []).filter((m) => m && (m.role === 'user' || m.role === 'assistant')).map((m) => ssMessageText(m.content)).filter(Boolean).join('\n');
      if (text.length > SS_MAX_TEXT) text = text.slice(0, SS_MAX_TEXT);
      return { id: `code:${s.id}`, kind: 'code', title: s.title || '', text, updated: s.time || 0, workspace: root || '' };
    }

    /** An index over `docs` for `ssSearch`: term counts per document (the title counts three times), document frequencies, average length. */
    function ssBuildIndex(docs) {
      const tfs = [];
      const df = new Map();
      const lens = [];
      for (const d of docs) {
        const tf = new Map();
        let len = 0;
        const words = ssTokenize(d.title);
        for (let i = 0; i < 3; i++) for (const w of words) { tf.set(w, (tf.get(w) || 0) + 1); len++; }
        for (const w of ssTokenize(d.text)) { tf.set(w, (tf.get(w) || 0) + 1); len++; }
        for (const w of tf.keys()) df.set(w, (df.get(w) || 0) + 1);
        tfs.push(tf);
        lens.push(len);
      }
      const avg = Math.max(1, lens.reduce((a, b) => a + b, 0) / Math.max(1, docs.length));
      return { docs, tfs, df, lens, avg };
    }

    /** BM25 (k1 1.5, b 0.75), best first: `[{doc, score}]`. */
    function ssSearch(index, query, limit = 20) {
      const terms = [...new Set(ssTokenize(query))];
      if (!terms.length || !index.docs.length) return [];
      const n = index.docs.length;
      const hits = [];
      index.docs.forEach((doc, i) => {
        let score = 0;
        for (const t of terms) {
          const f = index.tfs[i].get(t);
          if (!f) continue;
          const df = index.df.get(t) || 0;
          const idf = Math.log((n - df + 0.5) / (df + 0.5) + 1);
          score += (idf * f * 2.5) / (f + 1.5 * (0.25 + (0.75 * index.lens[i]) / index.avg));
        }
        if (score > 0) hits.push({ doc, score });
      });
      hits.sort((a, b) => b.score - a.score || (b.doc.updated || 0) - (a.doc.updated || 0));
      return hits.slice(0, limit);
    }

    /** About `width` characters of `text` around the first occurrence of a word of the query. */
    function ssSnippet(text, query, width = 160) {
      const lower = String(text || '').toLowerCase();
      let at = -1;
      for (const t of new Set(ssTokenize(query))) {
        const i = lower.indexOf(t);
        if (i >= 0 && (at < 0 || i < at)) at = i;
      }
      const start = Math.max(0, (at < 0 ? 0 : at) - Math.floor(width / 3));
      const end = Math.min(text.length, start + width);
      return `${start > 0 ? '...' : ''}${String(text).slice(start, end).replace(/\s+/g, ' ').trim()}${end < text.length ? '...' : ''}`;
    }

    /** Hits of this browser first, then hits that only the copy on the computer knows, without repeats. */
    function ssMerge(local, remote, limit = 20) {
      const seen = new Set(local.map((h) => h.id));
      const out = local.slice();
      for (const h of remote) if (!seen.has(h.id)) { seen.add(h.id); out.push(h); }
      return out.slice(0, limit);
    }
    // @pure-end

    // The saved code sessions of code.js: { workspace folder: [ { id, title, time, messages, ... } ] }.
    const SS_CODE_KEY = 'fllm.studio.codesessions.v2';
    const SS_ON_KEY = 'fllm.studio.sessionIndex.v1';
    const SS_SENT_KEY = 'fllm.studio.sessionIndexed.v1';
    let ssCache = { sig: '', index: null };

    const ssEnabled = () => localStorage.getItem(SS_ON_KEY) !== 'false';

    function ssCodeSessions() {
      try { return JSON.parse(localStorage.getItem(SS_CODE_KEY) || '{}'); } catch { return {}; }
    }

    /** Every searchable document of this browser. */
    function ssLocalDocs() {
      const docs = (state.threads || []).filter((t) => !t.temporary).map(ssThreadDoc);
      const all = ssCodeSessions();
      for (const root of Object.keys(all)) for (const s of all[root] || []) docs.push(ssCodeDoc(s, root));
      return docs;
    }

    function ssIndex() {
      const raw = localStorage.getItem(SS_CODE_KEY) || '';
      const sig = `${(state.threads || []).filter((t) => !t.temporary).map((t) => `${t.id}:${t.updatedAt}`).join(',')}|${raw.length}`;
      if (ssCache.sig !== sig || !ssCache.index) ssCache = { sig, index: ssBuildIndex(ssLocalDocs()) };
      return ssCache.index;
    }

    window.fllmSessionSearch = async (query, opts) => {
      const limit = (opts && opts.limit) || 20;
      const q = String(query || '').trim();
      if (!q) return [];
      const local = ssSearch(ssIndex(), q, limit).map(({ doc, score }) => ({
        id: doc.id, threadId: doc.id, kind: doc.kind, title: doc.title, snippet: ssSnippet(doc.text, q), time: doc.updated, score, source: 'browser', workspace: doc.workspace,
      }));
      if (!ssEnabled() || (opts && opts.localOnly)) return local;
      let remote = [];
      try {
        const res = await apiFetch(`/api/studio/sessions/search?q=${encodeURIComponent(q)}&limit=${limit}`);
        if (res.ok) remote = ((await res.json()).hits || []).map((h) => ({ id: h.id, threadId: h.id, kind: h.kind, title: h.title, snippet: h.snippet, time: h.updated, score: h.score, source: 'server' }));
      } catch { /* the server copy is optional */ }
      return ssMerge(local, remote, limit);
    };

    // ---- the copy on the computer ---------------------------------------------------------------------------
    function ssSent() {
      try { return JSON.parse(localStorage.getItem(SS_SENT_KEY) || '{}'); } catch { return {}; }
    }
    const ssStreaming = (t) => Object.values(t.nodes || {}).some((n) => n && n.streaming);
    let ssBusy = false;

    /** Sends the sessions that changed since the last time (a few per call) and removes the copy of deleted chats. Resolves to the number sent. */
    async function ssPush() {
      if (ssBusy || !ssEnabled() || !Array.isArray(state.threads)) return 0;
      ssBusy = true;
      let sent = 0;
      try {
        const done = ssSent();
        const threads = state.threads.filter((t) => !t.temporary);
        const pending = [];
        for (const t of threads) if (done[t.id] !== t.updatedAt && !ssStreaming(t)) pending.push(t);
        const all = ssCodeSessions();
        const codePending = [];
        for (const root of Object.keys(all)) for (const s of all[root] || []) if (done[`code:${s.id}`] !== s.time) codePending.push([s, root]);
        const docs = pending.slice(0, 4).map(ssThreadDoc).concat(codePending.slice(0, 2).map(([s, root]) => ssCodeDoc(s, root)));
        for (const d of docs) {
          if (!d.text.trim() && !d.title) { done[d.id] = d.updated; continue; }
          const res = await apiFetch('/api/studio/sessions/index', { method: 'POST', body: { id: d.id, title: d.title, text: d.text, kind: d.kind, updated: d.updated } });
          if (!res.ok) break;
          done[d.id] = d.updated;
          sent++;
        }
        const live = new Set(threads.map((t) => t.id));
        for (const id of Object.keys(done)) {
          if (id.startsWith('code:') || live.has(id)) continue;
          const res = await apiFetch(`/api/studio/sessions/index?id=${encodeURIComponent(id)}`, { method: 'DELETE' });
          if (res.ok) delete done[id]; else break;
        }
        localStorage.setItem(SS_SENT_KEY, JSON.stringify(done));
      } catch { /* the server is not reachable: try again later */ } finally { ssBusy = false; }
      return sent;
    }

    window.fllmSessionIndexNow = async () => {
      let total = 0;
      for (let i = 0; i < 200; i++) { const n = await ssPush(); total += n; if (!n) break; }
      return total;
    };

    window.fllmSessionForget = async (id) => {
      try {
        await apiFetch(`/api/studio/sessions/index?id=${encodeURIComponent(id)}`, { method: 'DELETE' });
        const done = ssSent();
        delete done[id];
        localStorage.setItem(SS_SENT_KEY, JSON.stringify(done));
      } catch { /* nothing to forget */ }
    };

    window.fllmSessionIndexInfo = () => {
      const el = $('ssInfo');
      if (!el) return;
      $('chkSessionIndex').checked = ssEnabled();
      el.textContent = `${ssLocalDocs().length} chats and code sessions in this browser, ${Object.keys(ssSent()).length} copied to the computer. ${ssEnabled() ? '' : 'The copy is switched off.'}`;
    };

    // ---- the pane: a search box and the switch --------------------------------------------------------------
    async function ssRunSearch() {
      const q = $('ssQuery').value.trim();
      const out = $('ssResults');
      if (!q) { out.innerHTML = ''; return; }
      const hits = await window.fllmSessionSearch(q, { limit: 12 });
      out.innerHTML = hits.length ? hits.map((h, i) => `
        <button class="nav-item" data-ss-hit="${i}" style="text-align:left;height:auto;padding:6px 8px;">
          <span class="search-title">${escapeHtml(h.title || '(untitled)')}<small>${escapeHtml(h.snippet)}</small></span>
          <span class="quant-chip" style="margin-left:auto;">${h.kind}${h.source === 'server' ? ' - copy' : ''}</span>
        </button>`).join('') : '<div class="app-item-desc">Nothing found.</div>';
      out._hits = hits;
    }

    if ($('ssQuery')) {
      let timer = null;
      $('ssQuery').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(ssRunSearch, 250); });
      $('ssResults').addEventListener('click', (e) => {
        const btn = e.target.closest('[data-ss-hit]');
        const hit = btn && $('ssResults')._hits[Number(btn.dataset.ssHit)];
        if (hit && hit.kind === 'chat' && hit.source === 'browser' && typeof openThread === 'function') { switchSurface('chat'); openThread(hit.id); }
        else if (hit) toast(hit.source === 'server' ? 'This session is only in the copy on the computer; it is not in this browser.' : 'Open the Code page to continue this session.');
      });
      $('chkSessionIndex').addEventListener('change', (e) => { localStorage.setItem(SS_ON_KEY, e.target.checked ? 'true' : 'false'); window.fllmSessionIndexInfo(); if (e.target.checked) ssPush(); });
      $('btnSessionIndexNow').addEventListener('click', async () => { const n = await window.fllmSessionIndexNow(); toast(`${n} sessions copied`); window.fllmSessionIndexInfo(); });
      $('btnSessionIndexClear').addEventListener('click', async () => {
        if (!(await askDialog({ title: 'Delete the copy?', text: 'The text copy of your chats on this computer is deleted. The chats in the browser stay.', ok: 'Delete the copy', danger: true }))) return;
        try { await apiFetch('/api/studio/sessions/index?all=1', { method: 'DELETE' }); localStorage.removeItem(SS_SENT_KEY); toast('Copy deleted'); } catch (err) { toast(err.message); }
        window.fllmSessionIndexInfo();
      });
      window.setInterval(ssPush, 20000);
      window.setTimeout(ssPush, 5000);
    }
