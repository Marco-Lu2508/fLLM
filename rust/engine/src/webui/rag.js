    // ==========================================================================
    // Documents and the web in chat. A document that does not fit the context is cut into passages
    // (about 500 tokens, with overlap) and ranked against the question with BM25 in the browser: no embedding
    // model is needed. The best passages go into the prompt with a source label. The web is reached through the
    // model's own tool calls (tools.js), not here.
    // ==========================================================================
    // @pure-begin
    const RAG_CHUNK_CHARS = 1800; // about 500 tokens
    const RAG_OVERLAP_CHARS = 240;
    const RAG_MAX_DOC_CHARS = 600000; // kept per document in the browser storage
    const RAG_MAX_CHAT_CHARS = 1500000; // kept per chat
    const RAG_FIT_SHARE = 0.35; // a document goes in whole when it takes less than this share of the context
    const RAG_VEC_MIN = 0.3; // cosine a passage needs to count as a hit without sharing a word with the question
    const RAG_STOP = new Set(('a an the and or but if of to in on at by for with from as is are was were be been it its this that these those ' +
      'i you he she we they me my your our their what which who whom how why when where do does did not no so than then there here can could ' +
      'will would should may might about into over under also just very der die das und oder aber ist sind war ein eine einer den dem des ' +
      'zu von mit auf fuer bei als wie le la les un une des du de et en est').split(' '));

    function ragTokenize(text) {
      const words = String(text || '').toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
      const out = [];
      for (let w of words) {
        if (RAG_STOP.has(w)) continue;
        if (w.length > 3 && w.endsWith('s') && !w.endsWith('ss')) w = w.slice(0, -1);
        if (w.length > 1 || /\d/.test(w)) out.push(w);
      }
      return out;
    }

    /** Cuts text into passages of about `size` characters; each passage starts with the tail of the previous one. */
    function ragChunk(text, size = RAG_CHUNK_CHARS, overlap = RAG_OVERLAP_CHARS) {
      const units = [];
      for (const para of String(text || '').split(/\n\s*\n/)) {
        const p = para.trim();
        if (!p) continue;
        if (p.length <= size) { units.push({ t: p, sep: '\n\n' }); continue; }
        let first = true;
        for (const sentence of p.split(/(?<=[.!?])\s+|\n/)) {
          let rest = sentence.trim();
          while (rest.length > size) {
            let cut = rest.lastIndexOf(' ', size);
            if (cut < size * 0.5) cut = size;
            units.push({ t: rest.slice(0, cut).trim(), sep: first ? '\n\n' : ' ' });
            first = false;
            rest = rest.slice(cut).trim();
          }
          if (rest) { units.push({ t: rest, sep: first ? '\n\n' : ' ' }); first = false; }
        }
      }
      const chunks = [];
      let cur = '';
      for (const u of units) {
        if (cur && cur.length + u.sep.length + u.t.length > size) {
          chunks.push(cur.trim());
          let tail = overlap > 0 ? cur.slice(-overlap) : '';
          const space = tail.indexOf(' ');
          if (space >= 0 && space < tail.length - 1) tail = tail.slice(space + 1);
          cur = tail ? `${tail} ${u.t}` : u.t;
        } else {
          cur += (cur ? u.sep : '') + u.t;
        }
      }
      if (cur.trim()) chunks.push(cur.trim());
      return chunks;
    }

    /** Index over documents `[{id, name, text}]`: passages with term counts and the document frequencies for BM25. */
    function ragBuildIndex(docs) {
      const chunks = [];
      const df = new Map();
      for (const doc of docs) {
        const parts = ragChunk(doc.text);
        parts.forEach((text, i) => {
          const tf = new Map();
          const terms = ragTokenize(text);
          for (const w of terms) tf.set(w, (tf.get(w) || 0) + 1);
          for (const w of tf.keys()) df.set(w, (df.get(w) || 0) + 1);
          chunks.push({ docId: doc.id, doc: doc.name, n: i + 1, m: parts.length, text, tf, len: terms.length || 1 });
        });
      }
      const avg = chunks.reduce((s, c) => s + c.len, 0) / Math.max(1, chunks.length);
      return { chunks, df, avg };
    }

    /** The `k` best passages for the query, best first (only passages that share a word with it). With `queryVec` and
     * vectors in the index the ranking is hybrid: 60 % keyword (BM25, scaled to its best hit) and 40 % meaning (cosine),
     * so an exact word still wins over a neighbouring meaning. */
    function ragSearch(index, query, k, queryVec) {
      const terms = [...new Set(ragTokenize(query))];
      const vec = Array.isArray(queryVec) && index.vecs && index.vecs.length === index.chunks.length ? queryVec : null;
      if ((!terms.length && !vec) || !index.chunks.length) return [];
      const N = index.chunks.length;
      const k1 = 1.5;
      const b = 0.75;
      const rows = [];
      let best = 0;
      index.chunks.forEach((c, i) => {
        let bm = 0;
        for (const w of terms) {
          const f = c.tf.get(w);
          if (!f) continue;
          const d = index.df.get(w) || 0;
          const idf = Math.log(1 + (N - d + 0.5) / (d + 0.5));
          bm += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * c.len / index.avg));
        }
        if (bm > best) best = bm;
        rows.push({ c, i, bm });
      });
      const scored = [];
      for (const row of rows) {
        if (!vec) {
          if (row.bm > 0) scored.push({ c: row.c, score: row.bm });
          continue;
        }
        const cos = ragCosine(vec, index.vecs[row.i]);
        // a passage without a shared word has to be clearly close in meaning, otherwise everything would match
        if (row.bm <= 0 && cos < RAG_VEC_MIN) continue;
        const keyword = best > 0 ? row.bm / best : 0;
        scored.push({ c: row.c, score: 0.6 * keyword + 0.4 * Math.max(0, cos) });
      }
      scored.sort((x, y) => y.score - x.score);
      return scored.slice(0, k).map((s) => ({ docId: s.c.docId, doc: s.c.doc, n: s.c.n, m: s.c.m, text: s.c.text, score: s.score }));
    }

    /** Cosine similarity of two vectors (0 when one of them is empty or all zeros). */
    function ragCosine(a, b) {
      if (!Array.isArray(a) || !Array.isArray(b) || !a.length || a.length !== b.length) return 0;
      let dot = 0;
      let na = 0;
      let nb = 0;
      for (let i = 0; i < a.length; i += 1) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      if (!na || !nb) return 0;
      return dot / Math.sqrt(na * nb);
    }

    /** The first passages of every document: used when the question names nothing to search for ("summarize this"). */
    function ragOpening(index, k) {
      const docIds = [...new Set(index.chunks.map((c) => c.docId))];
      const per = Math.max(1, Math.ceil(k / Math.max(1, docIds.length)));
      const out = [];
      for (const id of docIds) {
        for (const c of index.chunks.filter((x) => x.docId === id).slice(0, per)) out.push({ docId: c.docId, doc: c.doc, n: c.n, m: c.m, text: c.text, score: 0 });
      }
      return out.slice(0, k);
    }

    // @pure-end

    /** The embeddings server set in Settings > Search, read once (an empty string means: none). */
    async function ragEmbedUrl() {
      if (typeof state.embedUrl === 'string') return state.embedUrl;
      try {
        const res = await apiFetch('/api/studio/embeddings');
        const data = await res.json();
        state.embedUrl = data.url || '';
      } catch {
        state.embedUrl = '';
      }
      return state.embedUrl;
    }

    /** Vectors for a batch of texts from the route (which talks to the embeddings server). */
    async function ragEmbedTexts(texts) {
      const out = [];
      for (let i = 0; i < texts.length; i += 32) {
        const slice = texts.slice(i, i + 32);
        const res = await apiFetch('/api/studio/embed', { method: 'POST', body: { input: slice } });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error((data.error && data.error.message) || 'The embeddings server did not answer.');
        for (const vector of data.vectors || []) out.push(vector);
      }
      return out;
    }

    /** Fills `index.vecs` once per index (the index itself is cached per chat, so this costs once). */
    async function ragFillVectors(index) {
      if (index.vecs || index.vecsFailed) return index.vecs;
      try {
        index.vecs = await ragEmbedTexts(index.chunks.map((c) => c.text));
        if (index.vecs.length !== index.chunks.length) index.vecs = null;
      } catch (e) {
        index.vecsFailed = (e && e.message) || String(e);
        index.vecs = null;
      }
      return index.vecs;
    }

    /** Keyword search, or hybrid search when an embeddings server is set and answers. */
    async function ragSearchBest(index, query, k) {
      const url = await ragEmbedUrl();
      if (!url) return ragSearch(index, query, k);
      try {
        const filled = await ragFillVectors(index);
        if (!filled) return ragSearch(index, query, k);
        const [vector] = await ragEmbedTexts([query]);
        return ragSearch(index, query, k, vector);
      } catch {
        return ragSearch(index, query, k);
      }
    }

    /** How many passages fit: about 40 % of the context, between 2 and 8. */
    function ragPassageCount() {
      const max = state.status?.model?.max_context || 4096;
      return Math.max(2, Math.min(8, Math.floor(max * 0.4 / (RAG_CHUNK_CHARS / 3.6))));
    }

    /** Does a document of this text go into the message whole? `usedTokens` counts what other attachments already take. */
    function ragFits(text, usedTokens = 0) {
      const max = state.status?.model?.max_context || 4096;
      return estimateTokens(text) + usedTokens <= max * RAG_FIT_SHARE;
    }

    // ---- documents kept in a chat ---------------------------------------------------
    const ragIndexCache = new Map();

    // The index covers the documents of the chat and, unless `chatOnly`, the knowledge of its project (projects.js).
    function ragThreadIndex(t, chatOnly) {
      const docs = prjCombinedDocs(state.prj.projects, t, chatOnly);
      const sig = docs.map((d) => `${d.id}:${d.text.length}`).join('|');
      const key = chatOnly ? `${t.id}:chat` : t.id;
      const hit = ragIndexCache.get(key);
      if (hit && hit.sig === sig) return hit.index;
      const index = ragBuildIndex(docs);
      ragIndexCache.set(key, { sig, index });
      return index;
    }

    /** Stores the extracted text of new documents in the chat, within the size limits; says so when a limit is hit. */
    function ragStoreDocs(t, pending) {
      t.docs = t.docs || [];
      for (const p of pending) {
        let text = p.text;
        const used = t.docs.reduce((s, d) => s + d.text.length, 0);
        const room = Math.min(RAG_MAX_DOC_CHARS, RAG_MAX_CHAT_CHARS - used);
        if (room < 2000) {
          toast(`${p.name} was not added: the documents of this chat already use the storage limit (${RAG_MAX_CHAT_CHARS.toLocaleString('en-US')} characters). Remove one first.`);
          continue;
        }
        if (text.length > room) {
          text = text.slice(0, room);
          toast(`${p.name} is long: only its first ${room.toLocaleString('en-US')} characters are kept in this chat.`);
        } else if (p.truncated) {
          toast(`${p.name} is very long: only the first part was read.`);
        }
        t.docs.push({ id: uid('doc'), name: p.name, text, pages: p.pages, chars: text.length });
      }
      ragIndexCache.delete(t.id);
      ragIndexCache.delete(`${t.id}:chat`);
    }

    function ragRemoveDoc(t, id) {
      t.docs = (t.docs || []).filter((d) => d.id !== id);
      ragIndexCache.delete(t.id);
      ragIndexCache.delete(`${t.id}:chat`);
      saveThreads();
      renderAttachments();
    }

    // ---- the web ------------------------------------------------------------------------
    function ragWebError(status, detail) {
      if (status === 404) return 'Web search is not available: this server has no web tools (the route was not found).';
      if (status === 403) return `Web access is blocked${detail ? `: ${detail}` : ' by this server'}.`;
      return `Web search failed${detail ? `: ${detail}` : ` (error ${status})`}.`;
    }

    async function ragErrorDetail(res) {
      try { const j = await res.json(); return j.error?.message || j.error || j.message || ''; } catch { return ''; }
    }

    // ---- preparing a message ----------------------------------------------------------------
    /** Adds the passages of the chat's documents and the web results to a user message. Returns false when stopped. */
    async function ragPrepare(t, node) {
      node.passages = undefined;
      node.web = undefined;
      // (the web is not searched here any more: with "Allow web access" the model decides, see tools.js)
      const hasDocs = prjCombinedDocs(state.prj.projects, t, false).length > 0; // the chat's documents and its project's knowledge
      if (!hasDocs) return true;
      const controller = new AbortController();
      state.abortController = controller;
      setBusy(true);
      try {
        if (hasDocs) {
          node.ctxStatus = 'Searching the documents';
          renderChatArea();
          const index = ragThreadIndex(t);
          const k = ragPassageCount();
          let found = await ragSearchBest(index, node.content || '', k);
          if (!found.length && node.ragFresh) found = ragOpening(ragThreadIndex(t, true), k); // "summarize this": only the chat's own documents
          found.sort((a, b) => (a.docId === b.docId ? a.n - b.n : 0));
          node.passages = found.length ? found.map((p) => ({ doc: p.doc, n: p.n, m: p.m, text: p.text })) : undefined;
        }
        return true;
      } catch (e) {
        if (e.name === 'AbortError') return false;
        throw e;
      } finally {
        node.ctxStatus = undefined;
        if (state.abortController === controller) state.abortController = null;
        setBusy(false);
        saveThreads();
        renderChatArea();
      }
    }

    /** The text put in front of the question: passages with their source labels. */
    function ragPrefix(node) {
      let out = '';
      if (node.passages && node.passages.length) {
        out += 'Passages from the attached documents. Use them to answer, and name the source in brackets, for example [report.pdf, chunk 3/42], when you rely on one.\n\n';
        out += node.passages.map((p) => `[${p.doc}, chunk ${p.n}/${p.m}]\n${p.text}`).join('\n\n');
        out += '\n\n';
      }
      const w = node.web;
      if (w && w.passages && w.passages.length) {
        out += `Web search results for "${w.query}". Answer from them where they help, cite them by number such as [1], and do not invent sources.\n\n`;
        out += w.passages.map((p) => `[${p.n}] ${p.title} (${p.url})\n${p.text}`).join('\n\n');
        out += '\n\n';
      }
      return out ? `${out}---\n` : '';
    }

    function ragTokens(node) {
      return estimateTokens(ragPrefix(node));
    }

    // ---- display -------------------------------------------------------------------------------
    function ragSnippet(text) {
      const t = String(text).replace(/\s+/g, ' ').trim();
      return t.length > 700 ? `${t.slice(0, 700)}...` : t;
    }

    function ragNames(names) {
      const unique = [...new Set(names)];
      return unique.length > 2 ? `${unique.slice(0, 2).join(', ')} and ${unique.length - 2} more` : unique.join(' and ');
    }

    /** Chips under a user message: what was searched and which passages were used. */
    function ragChipsHtml(node) {
      const out = [];
      if (node.ctxStatus) out.push(`<div class="rag-chip busy">${escapeHtml(node.ctxStatus)}...</div>`);
      if (node.passages && node.passages.length) {
        const p = node.passages;
        out.push(`<details class="rag-chip"><summary>Using ${p.length} passage${p.length === 1 ? '' : 's'} from ${escapeHtml(ragNames(p.map((x) => x.doc)))}</summary>` +
          `<div class="rag-list">${p.map((x) => `<div class="rag-passage"><b>${escapeHtml(x.doc)}, chunk ${x.n}/${x.m}</b><span>${escapeHtml(ragSnippet(x.text))}</span></div>`).join('')}</div></details>`);
      }
      const w = node.web;
      if (w && w.passages && w.passages.length) {
        out.push(`<details class="rag-chip"><summary>Using ${w.passages.length} passage${w.passages.length === 1 ? '' : 's'} from the web</summary>` +
          `<div class="rag-list">${w.passages.map((x) => `<div class="rag-passage"><b>[${x.n}] ${escapeHtml(x.title)}</b><span>${escapeHtml(ragSnippet(x.text))}</span></div>`).join('')}</div></details>`);
      }
      if (w && w.error) out.push(`<div class="rag-chip warn">${escapeHtml(w.error)}</div>`);
      return out.length ? `<div class="rag-chips">${out.join('')}</div>` : '';
    }

    /** Sources under an answer that was given with web results. */
    function ragSourcesHtml(t, node) {
      const w = t.nodes[node.parent]?.web;
      if (!w || !w.sources || !w.sources.length || !(w.passages || []).length) return '';
      const links = w.sources.map((s) => `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(s.url)}">[${s.n}] ${escapeHtml(s.title)}</a>`).join('');
      return `<div class="rag-sources"><span>Sources</span>${links}</div>`;
    }

    /** Chips above the composer for the documents of this chat. */
    function ragDocsChipsHtml() {
      const t = currentThread();
      return (t?.docs || []).map((d) => `<div class="file-chip removable doc-chip" title="${escapeHtml(d.name)}: ${d.chars.toLocaleString('en-US')} characters${d.pages ? `, ${d.pages} pages` : ''}. Searched for every question in this chat.">${svgIcon('file')}<span>${escapeHtml(d.name)}</span><button class="att-x" data-rm-doc="${d.id}" title="Remove the document from this chat">✕</button></div>`).join('');
    }

    // ---- reading files ---------------------------------------------------------------------------
    const RAG_SERVER_EXT = /\.(pdf|docx|xlsx|epub|html?|xhtml)$/i;
    const RAG_MAX_UPLOAD = 50 * 1024 * 1024;

    /** Text of a document through the server (PDF, DOCX, XLSX, EPUB, HTML). */
    async function ragExtract(file) {
      if (file.size > RAG_MAX_UPLOAD) throw new Error(`${file.name} is too large (more than 50 MB).`);
      const res = await apiFetch(`/api/studio/extract?name=${encodeURIComponent(file.name)}`, { method: 'POST', body: file, headers: { 'Content-Type': 'application/octet-stream' } });
      if (!res.ok) {
        const detail = await ragErrorDetail(res);
        throw new Error(res.status === 404 ? 'This server cannot read documents (the extract route was not found).' : (detail || `The server could not read ${file.name} (error ${res.status}).`));
      }
      return res.json();
    }

    /** Adds text as an attachment: whole when it fits the context, else as a searchable document. */
    function ragPlace(name, text, size, extra = {}) {
      const wholeTokens = state.attachments.filter((a) => a.kind === 'text').reduce((s, a) => s + estimateTokens(a.text), 0);
      if (ragFits(text, wholeTokens)) {
        state.attachments.push({ kind: 'text', name, text, size });
      } else {
        state.attachments.push({ kind: 'doc', name, text, size, pages: extra.pages, truncated: extra.truncated });
        toast(`${name} is too long for the context: the passages that match your question are used.`);
      }
    }

    // ---- the "Allow web access" switch ----------------------------------------------------------------
    // On: the model is offered the tools web_search and web_fetch and decides itself when to use them (tools.js).
    state.useWeb = false;
    $('btnToggleWeb').addEventListener('click', () => {
      state.useWeb = !state.useWeb;
      $('btnToggleWeb').classList.toggle('active', state.useWeb);
      if (state.useWeb) toast('The model may now search the web and read pages when it needs to. Questions it can answer itself are not searched.');
      updateContextMeter();
    });
    $('attachmentsRow').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-rm-doc]');
      const t = currentThread();
      if (btn && t) ragRemoveDoc(t, btn.dataset.rmDoc);
    });

    // ---- Settings > Search: the embeddings server for the document search --------
    (function searchSettings() {
      const nav = document.getElementById('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      if (!nav || !content) return;

      const button = document.createElement('button');
      button.className = 'settings-nav-btn';
      button.dataset.spane = 'search';
      button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg><span>Search</span>';
      nav.appendChild(button);

      const pane = document.createElement('div');
      pane.className = 'settings-pane';
      pane.id = 'spane-search';
      pane.innerHTML =
        '<div class="app-card"><div class="app-card-title">Document search</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;">Documents are cut into passages and ranked by keyword (BM25) in this browser. With an embeddings server the ranking becomes hybrid: half keyword, half meaning, so a passage is found even when the question uses another word for the same thing. Nothing to install: run <span class="mono">fllm-rs serve --encoder &lt;model&gt;</span> or point it at any OpenAI-compatible <span class="mono">/v1/embeddings</span>. Leave the field empty to keep keyword-only search.</div>' +
        '<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">' +
        '  <input class="ui-input mono" id="embedUrl" placeholder="http://127.0.0.1:8100" style="flex:1;min-width:260px;" autocomplete="off">' +
        '  <button class="btn btn-sm" id="btnEmbedSave">Save</button>' +
        '  <button class="btn btn-sm btn-ghost" id="btnEmbedTest">Test</button>' +
        '</div>' +
        '<div class="app-item-desc mono" id="embedStatus" style="margin-top:8px;font-size:11px;white-space:pre-wrap;"></div>' +
        '</div>';
      content.appendChild(pane);

      const input = pane.querySelector('#embedUrl');
      const status = pane.querySelector('#embedStatus');
      const say = (text) => { status.textContent = text || ''; };

      async function load() {
        try {
          const res = await apiFetch('/api/studio/embeddings');
          const data = await res.json();
          input.value = data.url || '';
          say(data.configured ? 'Hybrid search is on.' : 'Keyword search only.');
        } catch (e) {
          say('Could not read the setting: ' + ((e && e.message) || e));
        }
      }

      pane.querySelector('#btnEmbedSave').addEventListener('click', async () => {
        try {
          const res = await apiFetch('/api/studio/embeddings', { method: 'POST', body: { url: input.value.trim() } });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { say((data.error && data.error.message) || 'Could not save it.'); return; }
          state.embedUrl = data.url || '';
          // indexes built so far hold no vectors yet
          for (const entry of ragIndexCache.values()) { entry.index.vecs = null; entry.index.vecsFailed = null; }
          say(data.configured ? 'Saved. The next question searches hybrid.' : 'Saved. Keyword search only.');
        } catch (e) {
          say('Could not save it: ' + ((e && e.message) || e));
        }
      });

      pane.querySelector('#btnEmbedTest').addEventListener('click', async () => {
        say('Asking the server ...');
        try {
          const res = await apiFetch('/api/studio/embed', { method: 'POST', body: { input: ['the quick brown fox', 'a fast brown fox'] } });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { say((data.error && data.error.message) || 'The server did not answer.'); return; }
          const vectors = data.vectors || [];
          const close = vectors.length === 2 ? ragCosine(vectors[0], vectors[1]) : 0;
          say(`${vectors.length} vector(s), ${data.dims || 0} numbers each${data.model ? ', model ' + data.model : ''}. Similarity of two similar sentences: ${close.toFixed(3)}.`);
        } catch (e) {
          say('The server did not answer: ' + ((e && e.message) || e));
        }
      });

      load();
    })();
