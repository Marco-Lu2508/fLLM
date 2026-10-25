    // ==========================================================================
    // Documents and the web in chat. A document that does not fit the context is cut into passages
    // (about 500 tokens, with overlap) and ranked against the question with BM25 in the browser: no embedding
    // model is needed. The best passages go into the prompt with a source label. "Use the web" searches for
    // the question, reads the first pages found and ranks their passages the same way.
    // ==========================================================================
    const RAG_CHUNK_CHARS = 1800; // about 500 tokens
    const RAG_OVERLAP_CHARS = 240;
    const RAG_MAX_DOC_CHARS = 600000; // kept per document in the browser storage
    const RAG_MAX_CHAT_CHARS = 1500000; // kept per chat
    const RAG_FIT_SHARE = 0.35; // a document goes in whole when it takes less than this share of the context
    const RAG_WEB_PAGES = 3;
    const RAG_WEB_PAGE_CHARS = 80000;
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

    /** The `k` best passages for the query, best first (only passages that share a word with it). */
    function ragSearch(index, query, k) {
      const terms = [...new Set(ragTokenize(query))];
      if (!terms.length || !index.chunks.length) return [];
      const N = index.chunks.length;
      const k1 = 1.5;
      const b = 0.75;
      const scored = [];
      for (const c of index.chunks) {
        let score = 0;
        for (const w of terms) {
          const f = c.tf.get(w);
          if (!f) continue;
          const d = index.df.get(w) || 0;
          const idf = Math.log(1 + (N - d + 0.5) / (d + 0.5));
          score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * c.len / index.avg));
        }
        if (score > 0) scored.push({ c, score });
      }
      scored.sort((x, y) => y.score - x.score);
      return scored.slice(0, k).map((s) => ({ docId: s.c.docId, doc: s.c.doc, n: s.c.n, m: s.c.m, text: s.c.text, score: s.score }));
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

    /** Searches the web for the question and keeps the best passages of the first pages found on the message. */
    async function ragWeb(node, signal) {
      const query = (node.content || '').trim();
      node.web = { query, sources: [], passages: [] };
      if (!query) return;
      let results;
      try {
        const res = await apiFetch('/api/studio/web/search', { method: 'POST', body: { query, limit: 6 }, signal });
        if (!res.ok) { node.web.error = ragWebError(res.status, String(await ragErrorDetail(res))); return; }
        results = (await res.json()).results || [];
      } catch (e) {
        if (e.name === 'AbortError') throw e;
        node.web.error = `Web search failed: ${e.message}`;
        return;
      }
      const top = results.filter((r) => /^https?:\/\//i.test(r.url || '')).slice(0, RAG_WEB_PAGES);
      if (!top.length) { node.web.error = 'The web search found nothing for this question.'; return; }
      const pages = await Promise.all(top.map(async (r) => {
        try {
          const res = await apiFetch('/api/studio/web/fetch', { method: 'POST', body: { url: r.url, max_chars: RAG_WEB_PAGE_CHARS }, signal });
          if (!res.ok) return '';
          return String((await res.json()).text || '');
        } catch (e) {
          if (e.name === 'AbortError') throw e;
          return '';
        }
      }));
      const docs = [];
      top.forEach((r, i) => {
        const title = r.title || r.url;
        node.web.sources.push({ n: i + 1, title, url: r.url, fetched: !!pages[i] });
        const text = pages[i] || [r.title, r.snippet].filter(Boolean).join('\n\n');
        if (text.trim()) docs.push({ id: i + 1, name: title, text });
      });
      const index = ragBuildIndex(docs);
      const found = ragSearch(index, query, ragPassageCount()).length ? ragSearch(index, query, ragPassageCount()) : ragOpening(index, 3);
      node.web.passages = found.map((p) => ({ n: p.docId, title: p.doc, url: top[p.docId - 1].url, text: p.text }));
      if (!node.web.passages.length) node.web.error = 'The pages found had no readable text.';
    }

    // ---- preparing a message ----------------------------------------------------------------
    /** Adds the passages of the chat's documents and the web results to a user message. Returns false when stopped. */
    async function ragPrepare(t, node) {
      node.passages = undefined;
      node.web = undefined;
      const wantsWeb = state.useWeb && (node.content || '').trim();
      const hasDocs = prjCombinedDocs(state.prj.projects, t, false).length > 0; // the chat's documents and its project's knowledge
      if (!hasDocs && !wantsWeb) return true;
      const controller = new AbortController();
      state.abortController = controller;
      setBusy(true);
      try {
        if (hasDocs) {
          node.ctxStatus = 'Searching the documents';
          renderChatArea();
          const index = ragThreadIndex(t);
          const k = ragPassageCount();
          let found = ragSearch(index, node.content || '', k);
          if (!found.length && node.ragFresh) found = ragOpening(ragThreadIndex(t, true), k); // "summarize this": only the chat's own documents
          found.sort((a, b) => (a.docId === b.docId ? a.n - b.n : 0));
          node.passages = found.length ? found.map((p) => ({ doc: p.doc, n: p.n, m: p.m, text: p.text })) : undefined;
        }
        if (wantsWeb) {
          node.ctxStatus = 'Searching the web';
          renderChatArea();
          await ragWeb(node, controller.signal);
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

    // ---- the "Use the web" switch ------------------------------------------------------------------
    state.useWeb = false;
    $('btnToggleWeb').addEventListener('click', () => {
      state.useWeb = !state.useWeb;
      $('btnToggleWeb').classList.toggle('active', state.useWeb);
      if (state.useWeb) toast('Your questions are searched on the web; the pages found are given to the model.');
    });
    $('attachmentsRow').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-rm-doc]');
      const t = currentThread();
      if (btn && t) ragRemoveDoc(t, btn.dataset.rmDoc);
    });
