    // ==========================================================================
    // The chat decides for itself: the model is offered tools and calls them when it needs them.
    //   web_search, web_fetch  only with "Allow web access" (the routes of agent_net.rs, safe against private targets)
    //   memory                 add / update / remove / list short facts about the user (projects.js keeps them)
    //   search_chats           looks through the user's earlier chats; the page asks once per chat
    // The loop: ask the model, run the calls it made, show a compact card for each, feed the results back, repeat
    // (at most CT_MAX_ROUNDS rounds of calls, then one more answer without tools). A model that never calls a tool just
    // answers; a server or template that refuses tools gets a plain request instead.
    // ==========================================================================
    const CT_MAX_ROUNDS = 5;
    const CT_MAX_CALLS = 4; // calls honoured in one round

    // @pure-begin
    const CT_EXAMPLE = {
      web_search: '{"query": "latest stable version of Rust"}',
      web_fetch: '{"url": "https://example.com/page"}',
      memory: '{"action": "add", "text": "Prefers metric units"}',
      search_chats: '{"query": "trip to Lisbon"}',
      skill: '{"action": "read", "name": "release-checklist"}',
      schedule: '{"name": "Morning news", "prompt": "Summarise the top technology news", "when": "daily at 08:00"}',
    };

    const CT_TEXT = {
      web_search: ['Search the web. Use only for current, specific or unknown facts. Returns titles, links and snippets.', 'Search the web for current or unknown facts.'],
      web_fetch: ['Open a web page and read it as text. Use it on a result of web_search, or on a link the user gave.', 'Read a web page as text.'],
      memory: ['Save, change or delete a short fact about the user that stays useful in later chats (name, preferences, projects, constraints). Never trivia or secrets.', 'Save durable facts about the user.'],
      search_chats: ['Search the title and text of the user\'s earlier chats and return matching snippets. Use when the user refers to an earlier conversation.', 'Search the user\'s earlier chats.'],
    };

    /** The tool definitions (OpenAI format) for the flags { web, memory, chats }. */
    function ctDefs(flags, compact) {
      const d = (name, props, required) => ({
        type: 'function',
        function: { name, description: CT_TEXT[name][compact ? 1 : 0], parameters: { type: 'object', properties: props, required } },
      });
      const out = [];
      if (flags.web) {
        out.push(d('web_search', { query: { type: 'string', description: 'What to search for, as short search keywords' } }, ['query']));
        out.push(d('web_fetch', { url: { type: 'string', description: 'The full http or https address' } }, ['url']));
      }
      if (flags.memory) {
        out.push(d('memory', {
          action: { type: 'string', enum: ['add', 'update', 'remove', 'list'] },
          text: { type: 'string', description: 'The fact, one short sentence (add, update)' },
          category: { type: 'string', description: 'profile, preference, project, constraint or other' },
          match: { type: 'string', description: 'Part of the existing entry to update or remove' },
          id: { type: 'string', description: 'Id of the entry, from list (instead of match)' },
        }, ['action']));
      }
      if (flags.chats) out.push(d('search_chats', { query: { type: 'string', description: 'Keywords to look for' } }, ['query']));
      return out;
    }

    /** Arguments of a call: a JSON object, also when wrapped in a code fence; null when unusable. */
    function ctParseArgs(text) {
      const raw = String(text ?? '').trim();
      if (!raw) return {};
      for (const candidate of [raw, raw.replace(/^```(?:json)?\s*|\s*```$/g, '')]) {
        try {
          const v = JSON.parse(candidate);
          return v && typeof v === 'object' && !Array.isArray(v) ? v : null;
        } catch { /* try the next form */ }
      }
      return null;
    }

    /** Calls written as text, <tool_call>{"name": ..., "arguments": {...}}</tool_call>, for models whose output the server did not parse. */
    function ctTextCalls(content, names) {
      const out = [];
      for (const m of String(content || '').matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g)) {
        try {
          const j = JSON.parse(m[1]);
          const fn = j.function || j;
          const args = fn.arguments ?? fn.parameters ?? {};
          if (fn && names.includes(fn.name)) out.push({ id: '', name: fn.name, args: typeof args === 'string' ? args : JSON.stringify(args) });
        } catch { /* not a call */ }
      }
      return out;
    }

    /** Room for tool results, in characters, from the context size: one result and all results of an answer. */
    function ctBudget(maxContext) {
      const chars = Math.floor((Number(maxContext) || 4096) * 3.4);
      return { per: Math.max(1200, Math.min(12000, Math.floor(chars * 0.15))), total: Math.floor(chars * 0.4) };
    }

    function ctClip(text, max) {
      const s = String(text ?? '');
      return s.length > max ? `${s.slice(0, Math.max(0, max))}\n... [${s.length - max} characters cut]` : s;
    }

    const ctHost = (url) => { try { return new URL(url).host; } catch { return String(url).slice(0, 40); } };
    const ctShort = (s, n) => { const t = String(s || '').replace(/\s+/g, ' ').trim(); return t.length > n ? `${t.slice(0, n - 1)}...` : t; };

    function ctHttpError(status, detail) {
      if (status === 404) return 'Web access is not available: this server has no web tools.';
      if (status === 403) return `Web access is blocked${detail ? `: ${detail}` : ' by this server'}.`;
      return `The web request failed${detail ? `: ${detail}` : ` (error ${status})`}.`;
    }

    /** The number [n] of a source: an existing one for the same address, else the next free one. */
    function ctSourceNumber(sources, url, title, fetched) {
      const hit = sources.find((s) => s.url === url);
      if (hit) { if (fetched) hit.fetched = true; return hit.n; }
      const n = sources.length + 1;
      sources.push({ n, title: title || url, url, fetched: Boolean(fetched) });
      return n;
    }

    /**
     * Words of a search for earlier chats and the chats that match them best.
     * `threads`: the chat objects of the page. Returns [{ id, title, updatedAt, snippets }], best first.
     */
    function ctChatSearch(threads, query, exceptId, limit) {
      const terms = [...new Set(String(query || '').toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((w) => w.length >= 2))].slice(0, 8);
      if (!terms.length) return [];
      const hits = [];
      for (const t of threads) {
        if (t.temporary || t.id === exceptId) continue;
        const title = String(t.title || '').toLowerCase();
        let score = 0;
        const matched = new Set();
        for (const w of terms) if (title.includes(w)) { score += 5; matched.add(w); }
        const msgs = [];
        for (const n of Object.values(t.nodes || {})) {
          if ((n.role !== 'user' && n.role !== 'assistant') || typeof n.content !== 'string') continue;
          const text = n.content.replace(/<think>[\s\S]*?<\/think>/g, '').replace(/\s+/g, ' ').trim();
          const low = text.toLowerCase();
          let here = 0;
          let at = -1;
          for (const w of terms) {
            const i = low.indexOf(w);
            if (i >= 0) { here++; matched.add(w); if (at < 0 || i < at) at = i; }
          }
          if (here) msgs.push({ here, at, text, role: n.role });
        }
        if (!matched.size) continue;
        score += matched.size * 3 + Math.min(5, msgs.length);
        msgs.sort((a, b) => b.here - a.here);
        const snippets = msgs.slice(0, 2).map((m) => {
          const from = Math.max(0, m.at - 80);
          return `${m.role === 'user' ? 'User' : 'Assistant'}: ${from > 0 ? '...' : ''}${m.text.slice(from, from + 260)}${m.text.length > from + 260 ? '...' : ''}`;
        });
        hits.push({ id: t.id, title: t.title || 'Untitled', updatedAt: t.updatedAt || 0, score, snippets });
      }
      return hits.sort((a, b) => b.score - a.score || b.updatedAt - a.updatedAt).slice(0, limit || 4);
    }

    const ctLabel = {
      web_search: (a, done) => `${done ? 'Searched' : 'Searching'}: ${ctShort(a.query, 80)}`,
      web_fetch: (a, done) => `${done ? 'Read' : 'Reading'}: ${ctShort(String(a.url || '').replace(/^https?:\/\//, ''), 80)}`,
      memory: (a, done) => (done ? 'Memory' : 'Updating memory'),
      search_chats: (a, done) => `${done ? 'Searched your chats' : 'Searching your chats'}: ${ctShort(a.query, 60)}`,
      skill: (a, done) => `${done ? 'Skill' : 'Using a skill'}: ${ctShort(`${a.action || ''} ${a.name || ''}`.trim(), 60)}`,
      schedule: (a, done) => `${done ? 'Schedule' : 'Setting up a schedule'}: ${ctShort(a.name || a.prompt, 60)}`,
    };

    /**
     * Runs one tool call. env: { apiFetch, signal, budget: { per, left }, sources: [], memory: { apply(args) }, chats: { allow(), search(query) } }.
     * Returns { text (for the model), card: { label, status: 'done' | 'error', detail?, change? } }.
     */
    async function ctRunTool(call, env) {
      const name = call.name;
      const args = ctParseArgs(call.args);
      const label = (a, done) => (ctLabel[name] ? ctLabel[name](a || {}, done) : name);
      const fail = (text) => ({ text, card: { label: label(args, true), status: 'error', detail: text } });
      if (!ctLabel[name]) return { text: `Error: unknown tool '${name}'.`, card: { label: name, status: 'error', detail: 'unknown tool' } };
      if (args === null) return fail(`Error: the arguments of ${name} are not valid JSON. Call it again with a JSON object such as ${CT_EXAMPLE[name]}.`);
      const room = () => Math.min(env.budget.per, env.budget.left);
      const spend = (text) => { env.budget.left -= text.length; return text; };
      if ((name === 'web_search' || name === 'web_fetch') && env.budget.left < 400) {
        return fail('Skipped: the room for web results in this answer is used up. Answer with what you have.');
      }

      if (name === 'web_search') {
        const query = String(args.query || '').trim();
        if (!query) return fail('Error: "query" is required.');
        let json;
        try {
          const res = await env.apiFetch('/api/studio/web/search', { method: 'POST', body: { query, limit: 6 }, signal: env.signal });
          if (!res.ok) return fail(`Error: ${ctHttpError(res.status, '')}`);
          json = await res.json();
        } catch (e) {
          if (e.name === 'AbortError') throw e;
          return fail(`Error: the web search failed: ${e.message}`);
        }
        if (json.ok === false) return fail(`Error: ${json.error || 'the web search failed'}`);
        const results = (json.results || []).filter((r) => /^https?:\/\//i.test(r.url || ''));
        if (!results.length) return { text: `No results for "${query}". Try other keywords or answer without the web.`, card: { label: label(args, true), status: 'done', detail: 'no results' } };
        const lines = results.map((r) => {
          const n = ctSourceNumber(env.sources, r.url, r.title, false);
          return `[${n}] ${ctShort(r.title, 120) || r.url} (${r.url})\n${ctShort(r.snippet, 300)}`;
        });
        const text = spend(ctClip(`Search results for "${query}". Cite sources as [n].\n\n${lines.join('\n\n')}`, room()));
        return { text, card: { label: label(args, true), status: 'done', detail: `${results.length} result${results.length === 1 ? '' : 's'}` } };
      }

      if (name === 'web_fetch') {
        const url = String(args.url || '').trim();
        if (!/^https?:\/\/\S+$/i.test(url)) return fail('Error: "url" must be a full http or https address.');
        let json;
        try {
          const res = await env.apiFetch('/api/studio/web/fetch', { method: 'POST', body: { url, max_chars: Math.max(2000, room()) }, signal: env.signal });
          if (!res.ok) return fail(`Error: ${ctHttpError(res.status, '')}`);
          json = await res.json();
        } catch (e) {
          if (e.name === 'AbortError') throw e;
          return fail(`Error: the page could not be read: ${e.message}`);
        }
        if (json.ok === false) return fail(`Error: ${json.error || 'the page could not be read'}`);
        const page = String(json.text || '').trim();
        if (!page) return fail('Error: the page has no readable text.');
        const n = ctSourceNumber(env.sources, json.url || url, ctHost(url), true);
        const text = spend(ctClip(`[${n}] ${json.url || url}\n${page}`, room()));
        return { text, card: { label: label(args, true), status: 'done', detail: `${page.length.toLocaleString('en-US')} characters` } };
      }

      if (name === 'memory') {
        if (!env.memory) return fail('Error: memory is switched off in this chat.');
        const r = env.memory.apply(args);
        const c = r.change;
        if (!c) {
          const refused = r.text.startsWith('Not saved');
          const quiet = String(args.action || '').toLowerCase() === 'list' || r.text === 'Already remembered.';
          return { text: r.text, card: { label: refused ? 'Memory: not saved (looks like a secret)' : (quiet ? 'Memory checked' : `Memory: ${ctShort(r.text, 80)}`), status: refused || r.text.startsWith('Error') ? 'error' : 'done', quiet: quiet && !refused } };
        }
        const labelText = c.kind === 'add' ? `Saved to memory: ${c.after.text}` : c.kind === 'update' ? `Updated memory: ${c.after.text}` : `Removed from memory: ${c.before.text}`;
        return { text: r.text, card: { label: ctShort(labelText, 140), status: 'done', change: c } };
      }

      // skills and scheduled tasks (skills.js): they check their own arguments and the schedule asks the user first
      if (name === 'skill' || name === 'schedule') {
        const run = name === 'skill' ? window.fllmSkillToolRun : window.fllmScheduleToolRun;
        const check = name === 'skill' ? window.fllmSkillCheckArgs : window.fllmScheduleCheckArgs;
        if (!run) return fail(`Error: the tool ${name} is not available.`);
        const problem = check ? check(args) : '';
        if (problem) return fail(`Error: ${problem}`);
        try {
          const text = String(await run(args, {}));
          return { text: ctClip(text, room()), card: { label: label(args, true), status: 'done', detail: ctShort(text.split('\n')[0], 80) } };
        } catch (e) {
          if (e.name === 'AbortError') throw e;
          return fail(`Error: ${e.message}`);
        }
      }

      // search_chats
      const query = String(args.query || '').trim();
      if (!query) return fail('Error: "query" is required.');
      if (!(await env.chats.allow())) return fail('The user did not allow searching their earlier chats. Say so and continue without it.');
      const hits = env.chats.search(query);
      if (!hits.length) return { text: `No earlier chat matches "${query}".`, card: { label: label(args, true), status: 'done', detail: 'nothing found' } };
      const text = spend(ctClip(hits.map((h, i) => `${i + 1}. "${h.title}" (${h.updatedAt ? new Date(h.updatedAt).toISOString().slice(0, 10) : 'earlier'})\n${h.snippets.join('\n')}`).join('\n\n'), room()));
      return { text, card: { label: label(args, true), status: 'done', detail: `${hits.length} chat${hits.length === 1 ? '' : 's'}` } };
    }

    /**
     * The loop. o: { messages, tools, stream(messages, { tools, round, final }) -> { content, calls: [{ id, name, args }] },
     *                run(call) -> { text }, isStopped(), maxRounds }.
     * Returns { endedBy: 'answer' | 'limit' | 'stopped', rounds }.
     */
    async function ctLoop(o) {
      const messages = o.messages.slice();
      const max = o.maxRounds === undefined ? CT_MAX_ROUNDS : o.maxRounds;
      const seen = new Set();
      let counter = 0;
      for (let round = 0; ; round++) {
        const final = round >= max;
        if (final) messages.push({ role: 'user', content: 'Answer the user now with what you have found. Do not call any tool.' });
        const r = await o.stream(messages, { tools: final ? [] : o.tools, round, final });
        if (o.isStopped()) return { endedBy: 'stopped', rounds: round };
        const calls = final ? [] : (r.calls || []).slice(0, CT_MAX_CALLS).map((c) => ({ ...c, id: c.id || `call_${round}_${counter++}` }));
        if (!calls.length) return { endedBy: final ? 'limit' : 'answer', rounds: round };
        messages.push({ role: 'assistant', content: r.content || '', tool_calls: calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args || '{}' } })) });
        for (const c of calls) {
          if (o.isStopped()) return { endedBy: 'stopped', rounds: round };
          const key = `${c.name}\n${String(c.args || '').replace(/\s+/g, '')}`;
          let text;
          if (seen.has(key)) text = 'You already made this exact call. Use the earlier result and answer the user now.';
          else { seen.add(key); text = (await o.run(c)).text; }
          messages.push({ role: 'tool', tool_call_id: c.id, content: text });
        }
      }
    }
    // @pure-end

    // ---- what is offered in a chat ------------------------------------------------------------------------
    /** Which tools chat `t` (null: a chat about to start) is offered. Also read by prompts.js to describe them. */
    function ctOfferedFlags(t) {
      const none = { web: false, memory: false, chats: false };
      const model = state.selectedModel || state.status?.model?.name || '';
      if (state.setup || (state.ctBroken && state.ctBroken[model])) return none;
      const P = state.prj && state.prj.personal;
      const priv = Boolean(t && (t.temporary || t.noPersonal));
      return {
        web: Boolean(state.useWeb),
        memory: Boolean(P && P.useMemory && !priv && !(t && t.noMemory)),
        chats: !priv && state.threads.some((x) => !x.temporary && !(t && x.id === t.id)),
      };
    }

    function ctOffered(t) {
      const flags = ctOfferedFlags(t);
      const remote = Boolean(window.fllmRemoteSelected && window.fllmRemoteSelected());
      const compact = !remote && spSize(state.status?.model) === 'compact';
      const defs = ctDefs(flags, compact);
      // skills and schedules: not for temporary chats, and not for small models (every extra tool makes them less reliable)
      if (!compact && !(t && (t.temporary || t.noPersonal)) && !state.setup && !(state.ctBroken && state.ctBroken[state.selectedModel || state.status?.model?.name || ''])) {
        // a skill tool only when there is a skill to use, the schedule tool only when the message sounds like a reminder
        const haveSkills = Boolean(window.fllmSkillIndex && String(window.fllmSkillIndex() || '').trim());
        const lastUser = (() => { try { const path = t ? pathOf(t) : []; for (let i = path.length - 1; i >= 0; i--) if (path[i].role === 'user') return String(path[i].content || ''); } catch { /* none */ } return ''; })();
        if (haveSkills && window.fllmSkillToolDef) defs.push(window.fllmSkillToolDef);
        if (window.fllmScheduleToolDef && /\b(remind|schedule|every\s+(day|morning|evening|week|hour|\d+\s*min)|daily|weekly|hourly|each\s+(day|morning|week)|cron)\b/i.test(lastUser)) defs.push(window.fllmScheduleToolDef);
      }
      return defs;
    }

    // ---- the answer of a chat with tools -------------------------------------------------------------------
    function ctPaint(aNode) { paintStreaming(aNode); }

    async function ctGenerate(t, userId, aNode, offered, stats) {
      const signal = state.abortController.signal;
      const names = offered.map((d) => d.function.name);
      const model = state.selectedModel || state.status?.model?.name || '';
      const remote = Boolean(window.fllmRemoteSelected && window.fllmRemoteSelected());
      aNode.tools = [];
      const sources = [];
      const env = {
        apiFetch, signal, sources,
        budget: (() => { const b = ctBudget(remote ? 16384 : state.status?.model?.max_context); return { per: b.per, left: b.total }; })(),
        memory: names.includes('memory') ? {
          apply: (args) => {
            const P = state.prj.personal;
            const r = memToolApply(P.memories, args, Date.now(), () => uid('mem'));
            if (r.list !== P.memories) { P.memories = r.list; prjSavePersonal(); memRender(); updateContextMeter(); }
            return r;
          },
        } : null,
        chats: {
          allow: async () => {
            if (t.chatSearch === undefined) {
              t.chatSearch = Boolean(await askDialog({ title: 'Search your earlier chats?', text: 'The assistant wants to look through your earlier chats on this computer to answer you. Allow this in this chat?', ok: 'Allow' }));
              saveThreads();
            }
            return t.chatSearch;
          },
          search: (query) => ctChatSearch(state.threads, query, t.id, 4),
        },
      };
      let grammar = !state.thinkingEnabled; // the grammar switches thinking off (see "Constrained tool calls" in the server API docs)
      const stream = async (msgs, opts) => {
        aNode.content = '';
        aNode.reasoning = '';
        ctPaint(aNode);
        let r;
        for (let attempt = 0; ; attempt++) {
          const sendTools = opts.tools.length > 0;
          try {
            r = await chatStream(aNode, msgs, signal, stats, sendTools ? { tools: opts.tools, ...(grammar ? { tool_call_grammar: true } : {}) } : undefined);
            break;
          } catch (e) {
            // a server or template that does not take tools (or the grammar): try again with less, once per step
            if (e.name === 'AbortError' || e.status !== 400 || !sendTools || attempt >= 2 || /context|too long|maximum|exceed/i.test(e.message)) throw e;
            aNode.content = '';
            aNode.reasoning = '';
            if (grammar) { grammar = false; continue; }
            state.ctBroken = { ...(state.ctBroken || {}), [model]: true };
            toast('This model or server does not accept tools. It answers without them.');
            opts = { ...opts, tools: [] };
          }
        }
        let calls = r.calls.map((c) => ({ ...c }));
        if (!calls.length && opts.tools.length && aNode.content.includes('<tool_call>')) {
          calls = ctTextCalls(aNode.content, names);
          if (calls.length) aNode.content = aNode.content.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();
        }
        return { content: splitReasoning(aNode.content).answer, calls: calls.filter((c) => c.name) };
      };
      const run = async (call) => {
        const card = { id: uid('tc'), name: call.name, label: (ctLabel[call.name] || (() => call.name))(ctParseArgs(call.args) || {}, false), status: 'running' };
        aNode.tools.push(card);
        ctPaint(aNode);
        let out;
        try {
          out = names.includes(call.name) ? await ctRunTool(call, env) : { text: `Error: the tool '${call.name}' is not available. Available tools: ${names.join(', ')}.`, card: { label: call.name, status: 'error' } };
        } catch (e) {
          card.status = 'error';
          throw e;
        }
        Object.assign(card, out.card);
        if (sources.length) aNode.sources = sources.map((s) => ({ ...s }));
        ctPaint(aNode);
        saveThreads();
        return out;
      };
      try {
        await ctLoop({ messages: apiMessages(t, userId), tools: offered, stream, run, isStopped: () => signal.aborted });
      } finally {
        aNode.tools = (aNode.tools || []).map((c) => (c.status === 'running' ? { ...c, status: 'error', detail: 'stopped' } : c));
        if (!aNode.tools.length) aNode.tools = undefined;
      }
      if (!signal.aborted && !aNode.content.trim() && !aNode.error) aNode.error = 'The model did not write an answer.';
    }

    // ---- display -------------------------------------------------------------------------------------------------
    const CT_ICON = {
      web_search: '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
      web_fetch: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15 15 0 0 1 0 20a15 15 0 0 1 0-20"/>',
      memory: '<path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>',
      search_chats: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    };

    /** Compact cards above an answer: what the model searched, read or saved. */
    function ctCardsHtml(m) {
      const cards = (m.tools || []).filter((c) => !c.quiet);
      if (!cards.length) return '';
      return `<div class="tool-cards">${cards.map((c) => `
        <div class="tool-card is-${c.status}${c.undone ? ' undone' : ''}" title="${escapeHtml(c.detail || '')}">
          <svg viewBox="0 0 24 24" class="icon icon-sm">${CT_ICON[c.name] || CT_ICON.web_search}</svg>
          <span class="tool-card-label">${escapeHtml(c.undone ? `${c.label} (undone)` : c.label)}</span>
          ${c.status === 'running' ? '<span class="tool-card-spin"></span>' : ''}
          ${c.change && !c.undone && !m.streaming ? `<button class="tool-card-undo" data-ct-undo="${escapeHtml(c.id)}">Undo</button>` : ''}
        </div>`).join('')}</div>`;
    }

    /** Sources under an answer that used the web tools. */
    function ctSourcesHtml(node) {
      const list = node.sources;
      if (!list || !list.length) return '';
      const links = list.map((s) => `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(s.url)}">[${s.n}] ${escapeHtml(ctShort(s.title, 70))}</a>`).join('');
      return `<div class="rag-sources"><span>Sources</span>${links}</div>`;
    }

    $('messagesColumn').addEventListener('click', (e) => {
      const b = e.target.closest('[data-ct-undo]');
      if (!b) return;
      const t = currentThread();
      const node = t?.nodes[b.closest('[data-node]')?.dataset.node];
      const card = node?.tools?.find((c) => c.id === b.dataset.ctUndo);
      if (!card || !card.change || card.undone) return;
      const P = state.prj.personal;
      const next = memUndo(P.memories, card.change);
      if (!next) { toast('That memory entry has changed since; it was left as it is.'); return; }
      P.memories = next;
      card.undone = true;
      prjSavePersonal();
      memRender();
      updateContextMeter();
      saveThreads();
      renderChatArea();
      toast('Memory change undone');
    });
