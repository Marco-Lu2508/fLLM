    // ==========================================================================
    // Research mode in the chat. One question becomes a small research project that a local model with a
    // small context can finish: plan search queries, search the web, read the best pages in parallel, take
    // notes per page, ask whether the notes are enough (one more round at most), then write a cited report.
    // Everything that was done stays inspectable in the "Research log" under the answer. Stop cancels every
    // pending request (they all share one AbortController).
    // ==========================================================================
    const RESEARCH_MAX_ROUNDS = 2;
    const RESEARCH_CONCURRENCY = 4;
    const RESEARCH_NOTE_SHARE = 0.6; // notes may take this share of the context when the report is written
    const RESEARCH_PAGE_CHARS = 60000;

    // @pure-begin
    /** Removes reasoning blocks and code fences from a model answer. */
    function researchClean(text) {
      return String(text || '').replace(/<think>[\s\S]*?(<\/think>|$)/g, '').replace(/```[a-z]*\n?/gi, '').trim();
    }

    /** Search queries from the model's plan. Strict JSON first, then the first JSON value in the text, then list lines. */
    function parsePlan(text, max = 6) {
      const s = researchClean(text);
      let list = null;
      const take = (v) => {
        if (Array.isArray(v)) return v;
        if (v && typeof v === 'object') return v.queries || v.searches || v.search_queries || null;
        return null;
      };
      try { list = take(JSON.parse(s)); } catch { /* not pure JSON */ }
      if (!list) {
        const m = s.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
        if (m) { try { list = take(JSON.parse(m[0])); } catch { /* fall through to lines */ } }
      }
      if (!list) {
        list = s.split('\n').map((l) => l.replace(/^\s*(?:[-*•]|\d+[.)])\s*/, '').replace(/^["'`]|["'`,]+$/g, '').trim())
          .filter((l) => l && !/^[{}\[\]]/.test(l) && !/^queries\b/i.test(l));
      }
      const seen = new Set();
      const out = [];
      for (const q of list) {
        const t = String(q || '').replace(/\s+/g, ' ').trim();
        const key = t.toLowerCase();
        if (t.length < 3 || t.length > 200 || seen.has(key)) continue;
        seen.add(key);
        out.push(t);
        if (out.length >= max) break;
      }
      return out;
    }

    /** The model's verdict after reading the notes: `{enough, queries}`. Unreadable answers count as "enough". */
    function parseReflection(text) {
      const s = researchClean(text);
      let v = null;
      try { v = JSON.parse(s); } catch { const m = s.match(/\{[\s\S]*\}/); if (m) { try { v = JSON.parse(m[0]); } catch { /* unreadable */ } } }
      if (!v || typeof v !== 'object') return { enough: true, queries: [] };
      const queries = parsePlan(JSON.stringify({ queries: Array.isArray(v.queries) ? v.queries : [] }), 3);
      const enough = typeof v.enough === 'boolean' ? v.enough : !queries.length;
      return { enough, queries: enough ? [] : queries };
    }

    const RESEARCH_JUNK_HOST = /(^|\.)(facebook\.com|instagram\.com|tiktok\.com|pinterest\.[a-z.]+|twitter\.com|x\.com|linkedin\.com|youtube\.com|youtu\.be|vimeo\.com|spotify\.com)$/;
    const RESEARCH_JUNK_EXT = /\.(pdf|zip|gz|tar|7z|png|jpe?g|gif|svg|webp|mp[34]|avi|mov|exe|dmg|iso|docx?|xlsx?|pptx?)$/i;
    const RESEARCH_JUNK_PATH = /\/(login|log-in|signin|sign-in|signup|sign-up|register|account|cart|checkout)(\/|$)/i;

    /** Identity of a page for deduplication: host without "www", path without a trailing slash or index file; no query, no fragment. */
    function urlKey(url) {
      try {
        const u = new URL(url);
        if (u.protocol !== 'http:' && u.protocol !== 'https:') return null;
        const path = u.pathname.replace(/\/index\.(html?|php)$/i, '').replace(/\/+$/, '');
        return `${u.hostname.toLowerCase().replace(/^www\./, '')}${path.toLowerCase()}`;
      } catch { return null; }
    }

    /**
     * The best distinct pages from the results of several queries (`[[{title,url,snippet}]]`): no junk, no page twice,
     * at most `perHost` per site, pages found by several queries first. `seen` holds the keys of pages already read.
     */
    function selectUrls(resultsByQuery, { limit = 6, perHost = 2, seen = new Set() } = {}) {
      const found = new Map();
      resultsByQuery.forEach((results, qi) => {
        (results || []).forEach((r, rank) => {
          const key = urlKey(r && r.url);
          if (!key || seen.has(key)) return;
          let u;
          try { u = new URL(r.url); } catch { return; }
          if (RESEARCH_JUNK_HOST.test(u.hostname.toLowerCase()) || RESEARCH_JUNK_EXT.test(u.pathname) || RESEARCH_JUNK_PATH.test(u.pathname)) return;
          const hit = found.get(key);
          if (hit) { hit.hits++; hit.rank = Math.min(hit.rank, rank); return; }
          found.set(key, { key, url: r.url, title: r.title || r.url, snippet: r.snippet || '', hits: 1, rank, order: qi * 100 + rank, host: u.hostname.toLowerCase().replace(/^www\./, '') });
        });
      });
      const sorted = [...found.values()].sort((a, b) => b.hits - a.hits || a.rank - b.rank || a.order - b.order);
      const perHostCount = new Map();
      const out = [];
      for (const c of sorted) {
        const n = perHostCount.get(c.host) || 0;
        if (n >= perHost) continue;
        perHostCount.set(c.host, n + 1);
        out.push(c);
        if (out.length >= limit) break;
      }
      return out;
    }

    /** A publication date in the opening of a page ("2025-03-14", "14 March 2025", "March 14, 2025"), or ''. */
    function findDate(text) {
      const head = String(text || '').slice(0, 3000);
      const iso = head.match(/\b((?:19|20)\d\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b/);
      if (iso) return iso[0];
      const mon = '(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)';
      const a = head.match(new RegExp(`\\b(\\d{1,2})\\s+${mon}\\.?,?\\s+((?:19|20)\\d\\d)\\b`));
      if (a) return a[0];
      const b = head.match(new RegExp(`\\b${mon}\\.?\\s+\\d{1,2},?\\s+((?:19|20)\\d\\d)\\b`));
      return b ? b[0] : '';
    }

    /**
     * Shrinks notes (`[{n, text}]`, best first) to `limitTokens`: every note is cut in proportion (not below `minChars`),
     * and when even that is too much the last notes are dropped. Returns `{notes, cut, dropped}`.
     */
    function fitNotes(notes, limitTokens, estimate, minChars = 240) {
      const cost = (list) => list.reduce((s, x) => s + estimate(x.text) + 8, 0);
      if (cost(notes) <= limitTokens) return { notes: notes.slice(), cut: false, dropped: 0 };
      let list = notes.slice();
      let dropped = 0;
      for (;;) {
        const fixed = list.length * 8;
        const body = Math.max(0, limitTokens - fixed);
        const total = list.reduce((s, x) => s + estimate(x.text), 0) || 1;
        const ratio = Math.min(1, body / total);
        const shrunk = list.map((x) => {
          const keep = Math.max(minChars, Math.floor(x.text.length * ratio * 0.95) - 4);
          if (x.text.length <= keep) return x;
          const slice = x.text.slice(0, keep);
          const nl = slice.lastIndexOf('\n');
          return { ...x, text: `${nl > keep * 0.5 ? slice.slice(0, nl) : slice.trimEnd()} ...` };
        });
        if (cost(shrunk) <= limitTokens || list.length <= 1) return { notes: shrunk, cut: true, dropped };
        list = list.slice(0, -1);
        dropped++;
      }
    }

    /** How many tokens the notes may take when the report is written, and whether they must be condensed first. */
    function noteBudget(notes, maxContext, fixedTokens, estimate, reserveOut = 1200) {
      const total = notes.reduce((s, x) => s + estimate(x.text) + 8, 0);
      const margin = 32 + Math.ceil(maxContext * 0.02);
      const limit = Math.max(0, Math.floor(Math.min(maxContext * RESEARCH_NOTE_SHARE, maxContext - fixedTokens - reserveOut - margin)));
      return { limit, total, needsCondense: total > limit };
    }

    /** Tokens the answer may use: the wanted size cut to what is left of the context, 0 when under 200 remain. */
    function researchMaxTokens(maxContext, promptTokens, wanted = 1800) {
      const margin = 32 + Math.ceil(maxContext * 0.02);
      const n = Math.min(wanted, Math.floor(maxContext - promptTokens - margin));
      return n >= 200 ? n : 0;
    }
    // @pure-end

    // ---- requests to the model and the web ----------------------------------------------------------
    state.research = false;
    const researchCtx = () => state.status?.model?.max_context || 4096;
    const researchModel = () => state.selectedModel || state.status?.model?.name || 'tiny';
    const researchAbortError = () => Object.assign(new Error('Stopped'), { name: 'AbortError' });

    /** Runs `fn` for every item, at most `limit` at a time. Errors that `fn` does not catch (a stop) end the whole run. */
    async function researchPool(items, limit, fn, signal) {
      const out = new Array(items.length);
      let next = 0;
      const worker = async () => {
        for (;;) {
          if (signal.aborted) throw researchAbortError();
          const k = next++;
          if (k >= items.length) return;
          out[k] = await fn(items[k], k);
        }
      };
      await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
      return out;
    }

    /** One non-streaming completion. With a schema the answer is constrained to JSON, if the server takes it. */
    async function researchChat(messages, { signal, maxTokens = 400, schema = null } = {}) {
      const body = { model: researchModel(), messages, stream: false, temperature: 0.2, max_tokens: maxTokens };
      if (schema && !state.researchNoSchema) body.response_format = { type: 'json_schema', json_schema: { name: 'research', strict: true, schema } };
      let res = await apiFetch('/v1/chat/completions', { method: 'POST', body: { ...body }, signal });
      if (!res.ok && body.response_format && res.status === 400) {
        state.researchNoSchema = true; // this server does not take the schema: the tolerant parser reads plain text
        delete body.response_format;
        res = await apiFetch('/v1/chat/completions', { method: 'POST', body, signal });
      }
      if (!res.ok) throw new Error((await ragErrorDetail(res)) || `The server answered with error ${res.status}`);
      const data = await res.json();
      return researchClean(data.choices?.[0]?.message?.content || '');
    }

    class ResearchWebError extends Error {}

    async function researchSearch(query, signal) {
      const res = await apiFetch('/api/studio/web/search', { method: 'POST', body: { query, limit: 8 }, signal });
      if (!res.ok) {
        const msg = ragWebError(res.status, String(await ragErrorDetail(res)));
        if (res.status === 404 || res.status === 403) throw new ResearchWebError(msg);
        throw new Error(msg);
      }
      return (await res.json()).results || [];
    }

    async function researchFetch(url, signal) {
      const res = await apiFetch('/api/studio/web/fetch', { method: 'POST', body: { url, max_chars: RESEARCH_PAGE_CHARS }, signal });
      if (!res.ok) throw new Error((await ragErrorDetail(res)) || `error ${res.status}`);
      const j = await res.json();
      const text = String(j.text || '').trim();
      if (!text) throw new Error('the page has no readable text');
      return text;
    }

    // ---- the log and the display -----------------------------------------------------------------------
    function researchEvent(r, text) {
      r.events.push({ s: Math.round((performance.now() - r.t0) / 100) / 10, text });
      if (r.events.length > 80) r.events.shift();
    }

    function researchPhase(node, phase, done, total) {
      const r = node.research;
      r.phase = phase;
      r.done = done;
      r.total = total;
      researchPaint(node);
    }

    function researchProgressHtml(node) {
      const r = node.research;
      if (!r) return '';
      if (r.notice) return `<div class="research-progress warn">${escapeHtml(r.notice)}</div>`;
      if (r.running) {
        const count = r.total ? ` (${r.done}/${r.total})` : '';
        return `<div class="research-progress busy"><b>Research</b> · round ${r.round} of ${RESEARCH_MAX_ROUNDS} · step ${r.step} of ${r.steps}: ${escapeHtml(r.phase)}${count}</div>`;
      }
      return r.stopped ? '<div class="research-progress warn">Research stopped.</div>' : '';
    }

    function researchLogHtml(node) {
      const r = node.research;
      if (!r) return '';
      const pages = r.pages || [];
      const ok = pages.filter((p) => p.ok).length;
      const bad = pages.length - ok;
      const secs = r.running ? Math.round((performance.now() - r.t0) / 1000) : r.seconds;
      const summary = `Research log · ${r.queries.length} ${r.queries.length === 1 ? 'query' : 'queries'}, ${ok} ${ok === 1 ? 'page' : 'pages'} read${bad ? `, ${bad} failed` : ''}, ${secs || 0} s`;
      const queries = r.queries.length ? `<div class="research-sec"><b>Queries</b><ol>${r.queries.map((q) => `<li>${escapeHtml(q)}</li>`).join('')}</ol></div>` : '';
      const list = pages.length ? `<div class="research-sec"><b>Pages</b>${pages.map((p) => {
        const head = `${p.n ? `[${p.n}] ` : ''}<a href="${escapeHtml(p.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(p.title)}</a>${p.date ? ` <span class="research-date">${escapeHtml(p.date)}</span>` : ''}`;
        if (!p.ok) return `<div class="research-page fail">${head}<span>${escapeHtml(p.error || 'failed')}</span></div>`;
        return `<details class="research-page"><summary>${head}${p.snippetOnly ? ' <span class="research-date">search snippet only</span>' : ''}</summary><pre>${escapeHtml(p.notes || '(nothing relevant)')}</pre></details>`;
      }).join('')}</div>` : '';
      const failures = (r.failures || []).length ? `<div class="research-sec"><b>Problems</b><ul>${r.failures.map((f) => `<li>${escapeHtml(f)}</li>`).join('')}</ul></div>` : '';
      const events = r.events.length ? `<div class="research-sec"><b>Steps</b><ul>${r.events.map((e) => `<li><span class="research-date">${e.s} s</span> ${escapeHtml(e.text)}</li>`).join('')}</ul></div>` : '';
      return `<details class="research-log" ${r.logOpen ? 'open' : ''}><summary>${escapeHtml(summary)}</summary>${queries}${list}${failures}${events}</details>`;
    }

    /** The slot above the answer (progress) and below it (log) of an assistant message. */
    function researchSlotsHtml(node, which) {
      if (!node.research) return '';
      return `<div class="research-${which}">${which === 'top' ? researchProgressHtml(node) : researchLogHtml(node)}</div>`;
    }

    function researchPaint(node) {
      const wrap = $('messagesColumn').querySelector(`[data-node="${node.id}"]`);
      if (!wrap) return;
      const top = wrap.querySelector('.research-top');
      const bottom = wrap.querySelector('.research-bottom');
      if (top) top.innerHTML = researchProgressHtml(node);
      if (bottom) {
        const open = bottom.querySelector('details.research-log')?.open;
        if (open !== undefined) node.research.logOpen = open;
        bottom.innerHTML = researchLogHtml(node);
      }
      scrollChatToEnd(false);
    }

    document.addEventListener('toggle', (e) => {
      if (!e.target.classList || !e.target.classList.contains('research-log')) return;
      const id = e.target.closest('[data-node]')?.dataset.node;
      const t = currentThread();
      if (id && t?.nodes[id]?.research) t.nodes[id].research.logOpen = e.target.open;
    }, true);

    // ---- the prompts ---------------------------------------------------------------------------------------
    const RESEARCH_PLAN_SYSTEM = 'You plan web research. Given a question, write 3 to 6 focused web search queries that look at it from different angles. ' +
      'Use short keyword queries, not full sentences, and do not repeat the same query in other words. ' +
      'Answer with JSON only, in the form {"queries": ["first query", "second query"]}.';
    const RESEARCH_PLAN_SCHEMA = { type: 'object', properties: { queries: { type: 'array', items: { type: 'string' }, minItems: 1, maxItems: 6 } }, required: ['queries'], additionalProperties: false };
    const RESEARCH_REFLECT_SYSTEM = 'You check research notes. Decide whether the notes contain enough to answer the question well. ' +
      'If they do not, name up to 3 new web search queries that would fill the gaps. ' +
      'Answer with JSON only, in the form {"enough": true, "queries": []} or {"enough": false, "queries": ["new query"]}.';
    const RESEARCH_REFLECT_SCHEMA = { type: 'object', properties: { enough: { type: 'boolean' }, queries: { type: 'array', items: { type: 'string' }, maxItems: 3 } }, required: ['enough', 'queries'], additionalProperties: false };

    /** The system prompt for taking notes. It is the same for every page of a run, so the server can reuse its prefix cache. */
    const researchNoteSystem = (question) => 'You take research notes from one web source at a time.\n' +
      `Research question: ${question}\n` +
      'Write 3 to 5 short bullet points with the facts in the source that help answer the question. Where you can, add a short exact quote in quotation marks. ' +
      'Use only what the source says. If the source has nothing relevant to the question, answer exactly NONE.';

    const researchReportSystem = () => 'You write research reports from notes. Use only the notes you are given; never add facts or sources of your own.\n' +
      'Structure the report with these Markdown headings: "## Summary" (2 to 4 sentences), "## Findings" (bullet points or short paragraphs, each claim followed by the numbers of its sources, for example [1] or [2][3]), ' +
      '"## Open questions" (what the notes do not settle). Cite only numbers that appear in the notes. If sources disagree, say so. Answer in the language of the question. Do not write a list of sources: it is added afterwards.';

    // ---- one run ----------------------------------------------------------------------------------------------
    /** Streams a completion into the assistant node (the report, or a normal answer when research cannot run). */
    async function researchStream(node, messages, signal, { plain = false, wanted = 1800 } = {}) {
      const body = requestBody(messages);
      if (!plain) {
        delete body.response_format; delete body.grammar; delete body.grammar_type; delete body.regex;
        const prompt = messages.reduce((s, m) => s + estimateTokens(m.content), 0) * 1.1;
        body.max_tokens = Math.max(200, researchMaxTokens(researchCtx(), prompt, wanted) || 200);
      }
      const t0 = performance.now();
      let first = null;
      let chunks = 0;
      const res = await apiFetch('/v1/chat/completions', { method: 'POST', body, signal });
      if (!res.ok) throw new Error((await ragErrorDetail(res)) || `The server answered with error ${res.status}`);
      await readEventStream(res, (chunk) => {
        const choice = chunk.choices?.[0];
        const delta = choice?.delta;
        if (delta) {
          if (first === null && (delta.content || delta.reasoning_content)) first = performance.now();
          if (delta.reasoning_content) node.reasoning = (node.reasoning || '') + delta.reasoning_content;
          if (delta.content) { node.content += delta.content; chunks++; }
        }
        if (choice?.finish_reason) node.finish = choice.finish_reason;
        if (chunk.usage) node.usage = chunk.usage;
        paintStreaming(node);
      });
      const elapsed = Math.max((performance.now() - (first || t0)) / 1000, 0.02);
      node.telemetry = { ttftMs: first ? Math.round(first - t0) : undefined, tokens: Math.max(chunks, 1), tokPerSec: (Math.max(chunks, 1) / elapsed).toFixed(1) };
    }

    function researchSourcesMarkdown(pages) {
      const used = pages.filter((p) => p.n);
      if (!used.length) return '';
      return `\n\n## Sources\n\n${used.map((p) => `${p.n}. [${p.title.replace(/[\[\]]/g, '')}](${p.url})${p.date ? `, ${p.date}` : ''}`).join('\n')}\n`;
    }

    /** Searches, reads and takes notes for `queries`. Pages go to `r.pages`; returns the notes made in this round. */
    async function researchRound(node, question, queries, signal, seen, wanted) {
      const r = node.research;
      researchPhase(node, 'searching the web', 0, queries.length);
      let done = 0;
      const results = await researchPool(queries, RESEARCH_CONCURRENCY, async (q) => {
        try { return await researchSearch(q, signal); } catch (e) {
          if (e.name === 'AbortError' || e instanceof ResearchWebError) throw e;
          r.failures.push(`Search "${q}": ${e.message}`);
          return [];
        } finally { researchPhase(node, 'searching the web', ++done, queries.length); }
      }, signal);
      const picks = selectUrls(results, { limit: wanted, seen });
      researchEvent(r, `Round ${r.round}: ${results.reduce((s, x) => s + x.length, 0)} results, ${picks.length} new pages chosen.`);
      if (!picks.length) return [];
      picks.forEach((p) => seen.add(p.key));

      researchPhase(node, 'reading pages', 0, picks.length);
      done = 0;
      const query = [question, ...queries].join(' ');
      const k = Math.max(2, Math.min(6, Math.floor(researchCtx() * 0.3 / (RAG_CHUNK_CHARS / 3.6))));
      const entries = await researchPool(picks, RESEARCH_CONCURRENCY, async (p) => {
        const entry = { title: p.title, url: p.url, ok: false, round: r.round };
        try {
          let text;
          try { text = await researchFetch(p.url, signal); } catch (e) {
            if (e.name === 'AbortError') throw e;
            if (!p.snippet) throw e;
            text = `${p.title}\n\n${p.snippet}`; // the page failed: the search snippet is all there is
            entry.snippetOnly = true;
            r.failures.push(`${p.url}: ${e.message} (using the search snippet)`);
          }
          entry.date = findDate(text);
          const index = ragBuildIndex([{ id: 1, name: p.title, text }]);
          let found = ragSearch(index, query, k);
          if (!found.length) found = ragOpening(index, Math.min(k, 3));
          found.sort((a, b) => a.n - b.n);
          entry.passages = found.map((x) => x.text).join('\n\n');
          entry.ok = !!entry.passages;
          if (!entry.ok) entry.error = 'no passage matched the question';
        } catch (e) {
          if (e.name === 'AbortError') throw e;
          entry.error = e.message;
          r.failures.push(`${p.url}: ${e.message}`);
        }
        researchPhase(node, 'reading pages', ++done, picks.length);
        return entry;
      }, signal);
      r.pages.push(...entries);

      const readable = entries.filter((e) => e.ok);
      researchPhase(node, 'taking notes', 0, readable.length);
      done = 0;
      const system = researchNoteSystem(question);
      const room = Math.max(200, Math.min(500, researchMaxTokens(researchCtx(), estimateTokens(system) + k * 500 + 100, 500) || 200));
      await researchPool(readable, RESEARCH_CONCURRENCY, async (e) => {
        try {
          const text = await researchChat([{ role: 'system', content: system }, { role: 'user', content: `Source: ${e.title} (${e.url})\n\n${e.passages}` }], { signal, maxTokens: room });
          e.notes = /^\W*NONE\W*$/i.test(text) ? '' : text;
        } catch (err) {
          if (err.name === 'AbortError') throw err;
          e.ok = false;
          e.error = `notes failed: ${err.message}`;
          r.failures.push(`${e.url}: ${e.error}`);
        }
        delete e.passages;
        researchPhase(node, 'taking notes', ++done, readable.length);
      }, signal);
      return entries;
    }

    /** Runs the research for a user message and writes the report into a new assistant message. */
    async function researchRun(t, userNode) {
      const question = (userNode.content || '').trim();
      const r = { running: true, round: 1, step: 1, steps: 5, phase: 'planning', done: 0, total: 0, queries: [], pages: [], failures: [], events: [], t0: performance.now(), logOpen: false };
      const node = addChild(t, userNode.id, { role: 'assistant', content: '', reasoning: '', streaming: true, logprobs: [], model: researchModel(), research: r });
      const controller = new AbortController();
      state.abortController = controller;
      setBusy(true);
      saveThreads();
      renderThreadList();
      renderChatArea();
      const signal = controller.signal;
      const fallback = async (why) => {
        r.notice = `Research could not run: ${why} The model answers on its own instead.`;
        researchEvent(r, `Fell back to a normal answer: ${why}`);
        researchPaint(node);
        await researchStream(node, apiMessages(t, userNode.id), signal, { plain: true });
      };
      try {
        const today = new Date().toISOString().slice(0, 10);
        researchEvent(r, 'Planning the search queries.');
        let queries = [];
        try {
          queries = parsePlan(await researchChat([{ role: 'system', content: `${RESEARCH_PLAN_SYSTEM} Today is ${today}.` }, { role: 'user', content: question }], { signal, maxTokens: 300, schema: RESEARCH_PLAN_SCHEMA }));
        } catch (e) {
          if (e.name === 'AbortError') throw e;
          r.failures.push(`Planning: ${e.message}`);
        }
        if (!queries.length) { queries = [question.slice(0, 200)]; researchEvent(r, 'The plan was unreadable: the question itself is searched.'); }
        r.queries.push(...queries);

        const seen = new Set();
        let roundQueries = queries;
        let reflected = null;
        for (r.round = 1; r.round <= RESEARCH_MAX_ROUNDS; r.round++) {
          r.step = 2;
          const wanted = r.round === 1 ? (researchCtx() <= 8192 ? 4 : 6) : 3;
          try { await researchRound(node, question, roundQueries, signal, seen, wanted); } catch (e) {
            if (e instanceof ResearchWebError) { await fallback(e.message); reflected = 'web'; break; }
            throw e;
          }
          const notes = r.pages.filter((p) => p.ok && p.notes);
          if (r.round >= RESEARCH_MAX_ROUNDS || !notes.length) break;
          r.step = 3;
          researchPhase(node, 'checking whether the notes are enough', 0, 0);
          const digest = notes.map((p) => `- ${p.title}: ${p.notes.replace(/\s+/g, ' ').slice(0, 300)}`).join('\n');
          const room = Math.floor(researchCtx() * 0.5 * 3.6);
          let verdict = { enough: true, queries: [] };
          try {
            verdict = parseReflection(await researchChat([{ role: 'system', content: RESEARCH_REFLECT_SYSTEM }, { role: 'user', content: `Question: ${question}\n\nNotes:\n${digest.slice(0, room)}` }], { signal, maxTokens: 200, schema: RESEARCH_REFLECT_SCHEMA }));
          } catch (e) {
            if (e.name === 'AbortError') throw e;
            r.failures.push(`Reflection: ${e.message}`);
          }
          const fresh = verdict.queries.filter((q) => !r.queries.some((x) => x.toLowerCase() === q.toLowerCase()));
          if (verdict.enough || !fresh.length) { researchEvent(r, 'The notes are enough.'); break; }
          researchEvent(r, `Gaps found, one more round: ${fresh.join('; ')}`);
          r.queries.push(...fresh);
          roundQueries = fresh;
        }

        r.round = Math.min(r.round, RESEARCH_MAX_ROUNDS);
        if (reflected !== 'web') {
          const usable = r.pages.filter((p) => p.ok && p.notes);
          if (!usable.length) {
            await fallback(r.pages.length ? 'none of the pages had anything relevant to the question.' : 'the web search found no usable pages.');
          } else {
            r.step = 4;
            usable.forEach((p, i) => { p.n = i + 1; });
            let list = usable.map((p) => ({ n: p.n, text: p.notes }));
            const system = researchReportSystem();
            const fixed = estimateTokens(system) + estimateTokens(question) + 60;
            const budget = noteBudget(list, researchCtx(), fixed, estimateTokens);
            if (budget.needsCondense) {
              researchEvent(r, `Notes take ${budget.total} of ${budget.limit} allowed tokens: condensing.`);
              const order = list.map((x, i) => i).sort((a, b) => list[b].text.length - list[a].text.length);
              let total = budget.total;
              let done = 0;
              const targets = [];
              for (const i of order) { targets.push(i); total -= Math.floor(estimateTokens(list[i].text) / 2); if (total <= budget.limit) break; }
              researchPhase(node, 'condensing notes', 0, targets.length);
              await researchPool(targets, RESEARCH_CONCURRENCY, async (i) => {
                try {
                  const text = await researchChat([{ role: 'system', content: 'You shorten research notes. Keep the numbers, names and exact quotes; cut everything else. Answer with at most 2 bullet points.' }, { role: 'user', content: list[i].text }], { signal, maxTokens: 200 });
                  if (text) { list[i] = { ...list[i], text }; usable[i].notes = text; }
                } catch (e) { if (e.name === 'AbortError') throw e; }
                researchPhase(node, 'condensing notes', ++done, targets.length);
              }, signal);
            }
            const fit = fitNotes(list, budget.limit, estimateTokens);
            if (fit.cut) researchEvent(r, `Notes were cut to fit the context${fit.dropped ? ` and ${fit.dropped} were left out` : ''}.`);
            list = fit.notes;
            const keep = new Set(list.map((x) => x.n));
            usable.forEach((p) => { if (!keep.has(p.n)) p.n = 0; });
            r.step = 5;
            researchPhase(node, 'writing the report', 0, 0);
            researchEvent(r, 'Writing the report.');
            const notesText = list.map((x) => `[${x.n}] ${usable[x.n - 1].title} (${usable[x.n - 1].url})\n${x.text}`).join('\n\n');
            await researchStream(node, [{ role: 'system', content: system }, { role: 'user', content: `Question: ${question}\n\nNotes:\n\n${notesText}\n\nWrite the report.` }], signal);
            node.content += researchSourcesMarkdown(r.pages);
          }
        }
      } catch (e) {
        if (e.name === 'AbortError') r.stopped = true;
        else node.error = e.message;
      } finally {
        r.running = false;
        r.seconds = Math.round((performance.now() - r.t0) / 1000);
        researchEvent(r, r.stopped ? 'Stopped.' : 'Done.');
        node.streaming = false;
        state.abortController = null;
        setBusy(false);
        t.updatedAt = Date.now();
        saveThreads();
        renderChatArea();
        refreshStudioStatus();
        maybeTitle(t);
      }
    }

    // ---- the composer switch ------------------------------------------------------------------------------------
    $('btnToggleResearch').addEventListener('click', () => {
      state.research = !state.research;
      $('btnToggleResearch').classList.toggle('active', state.research);
      if (state.research) toast('Research: your question is searched on the web in several steps and answered with a cited report. It takes longer than a normal answer.');
    });
