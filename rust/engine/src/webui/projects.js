    // ==========================================================================
    // Assistants, projects, memory and the prompt library.
    //  * Assistants: a name, icon and color, instructions, default sampling and an optional model. The old presets
    //    are migrated into assistants (same ids, so chats that remember a preset keep working).
    //  * Projects: a group of chats with its own instructions and a knowledge base. The extracted text of the
    //    documents is kept in the browser and searched (BM25, see rag.js) for every question of every chat of the project.
    //  * Personalization: "about me" text and a small memory, both put in front of the system prompt.
    //  * Prompt library: saved prompts inserted from "/" in the composer.
    // Everything is stored in the browser (localStorage) and never leaves it.
    // ==========================================================================
    const PRJ_KEYS = {
      assistants: 'fllm.studio.assistants.v1',
      migrated: 'fllm.studio.assistants.migrated.v1',
      current: 'fllm.studio.assistant.v1',
      projects: 'fllm.studio.projects.v1',
      personal: 'fllm.studio.personal.v1',
      prompts: 'fllm.studio.prompts.v1',
    };

    // @pure-begin
    const PRJ_LIMITS = {
      name: 60, system: 8000, starter: 140, model: 120,
      doc: 600000, project: 2000000, total: 3500000, minRoom: 2000, instructions: 8000,
      about: 2000, memory: 400, memories: 100, memoryPrompt: 3000, memoryPromptCompact: 1200, category: 24, sysText: 6000,
      promptTitle: 60, promptText: 8000, prompts: 100,
    };
    const ASST_ICONS = ['bot', 'pen', 'code', 'book', 'globe', 'spark', 'chat', 'star'];
    const ASST_COLORS = ['patina', 'kupfer', 'emerald', 'teal', 'cyan', 'blue', 'purple', 'orange', 'red', 'green', 'pink', 'rose'];
    const SLASH_COMMANDS = /^(clear|new|system|export|help|retry)(\s|$)/i;

    function prjClamp(v, lo, hi, dflt) {
      if (v === null || v === undefined || v === '') return dflt;
      const n = Number(v);
      return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
    }

    /** A clean assistant from untrusted data (storage, an imported file), or null when it has no name. */
    function asstSanitize(raw, makeId) {
      if (!raw || typeof raw !== 'object') return null;
      const name = String(raw.name ?? '').replace(/\s+/g, ' ').trim().slice(0, PRJ_LIMITS.name);
      if (!name) return null;
      const starters = (Array.isArray(raw.starters) ? raw.starters : [])
        .map((s) => String(s ?? '').replace(/\s+/g, ' ').trim().slice(0, PRJ_LIMITS.starter)).filter(Boolean).slice(0, 4);
      return {
        id: typeof raw.id === 'string' && raw.id ? raw.id.slice(0, 64) : makeId(),
        name,
        icon: ASST_ICONS.includes(raw.icon) ? raw.icon : 'bot',
        color: ASST_COLORS.includes(raw.color) ? raw.color : 'patina',
        system: String(raw.system ?? '').slice(0, PRJ_LIMITS.system),
        temperature: prjClamp(raw.temperature, 0, 2, 0.7),
        top_p: prjClamp(raw.top_p, 0, 1, 0.95),
        top_k: Math.round(prjClamp(raw.top_k, 0, 1000, 20)),
        model: String(raw.model ?? '').trim().slice(0, PRJ_LIMITS.model),
        starters,
        // how this assistant treats the default system prompt (prompts.js): inherit the global choice, replace it or add to it
        sysMode: ['override', 'append'].includes(raw.sysMode) ? raw.sysMode : 'inherit',
        sysText: String(raw.sysText ?? '').slice(0, PRJ_LIMITS.sysText),
      };
    }

    /**
     * Turns presets (the older feature) into assistants. A preset is migrated once (its id is remembered in `done`),
     * keeps its id, and never replaces an assistant that already has that id.
     */
    function asstMigrate(presets, assistants, done) {
      const have = new Set(assistants.map((a) => a.id));
      const seen = new Set(done);
      const out = assistants.slice();
      const doneOut = done.slice();
      let added = 0;
      for (const p of presets || []) {
        if (!p || typeof p.id !== 'string' || !p.id || seen.has(p.id)) continue;
        seen.add(p.id);
        doneOut.push(p.id);
        if (have.has(p.id)) continue;
        const a = asstSanitize({ id: p.id, name: p.name, system: p.system, temperature: p.temperature, top_p: p.top_p, top_k: p.top_k }, () => p.id);
        if (a) { out.push(a); have.add(a.id); added++; }
      }
      return { assistants: out, done: doneOut, added };
    }

    /** The built-in presets as read-only assistants (`meta` gives icon, color and starters by id). */
    function asstBuiltins(presets, meta) {
      const out = [];
      for (const p of presets) {
        const m = meta[p.id] || {};
        const a = asstSanitize({ ...p, ...m }, () => p.id);
        if (a) out.push({ ...a, builtin: true });
      }
      return out;
    }

    /** Four suggestion chips: the ones written for the assistant first, then generic ones that mention its name. */
    function asstStarters(a) {
      const own = (a.starters || []).map((s) => String(s).trim()).filter(Boolean);
      const generic = [
        `What can you help me with as "${a.name}"?`,
        'Ask me three questions so you understand my task, then help me with it.',
        'Show me a short example of what you do best.',
        'Summarize the text I paste next in five bullet points.',
      ];
      return own.concat(generic).slice(0, 4);
    }

    function asstExport(a) {
      const { id, builtin, ...rest } = a;
      return JSON.stringify({ format: 'fllm-assistant', version: 1, assistant: rest }, null, 2);
    }

    function asstParseImport(text, makeId) {
      let data;
      try { data = JSON.parse(text); } catch { return { error: 'The file is not valid JSON.' }; }
      const raw = data && data.format === 'fllm-assistant' ? data.assistant : null;
      if (!raw || typeof raw !== 'object') return { error: 'This file is not an exported assistant.' };
      const a = asstSanitize({ ...raw, id: undefined }, makeId);
      return a ? { assistant: a } : { error: 'The assistant in the file has no name.' };
    }

    // ---- memory -------------------------------------------------------------------
    /** "remember that ..." (or "remember: ...") at the start of a message gives the thing to remember, else null. */
    function memExtract(text) {
      const m = /^\s*(?:please\s+)?remember\s*(?:that\b|:)\s*([\s\S]+)$/i.exec(String(text || ''));
      if (!m) return null;
      const t = m[1].split(/\n\s*\n/)[0].replace(/\s+/g, ' ').trim().slice(0, PRJ_LIMITS.memory);
      return t || null;
    }

    /** Does a text look like a password, an access key, a card number or similar? Such things are never saved to memory. */
    function memSecret(text) {
      const t = String(text || '');
      if (/\b(?:pass(?:word|wd|phrase|code)?|pin|secret|api[ _-]?key|access[ _-]?key|private[ _-]?key|token|credentials?|cvv|cvc)\b\s*(?:is|are|:|=)\s*(?:\S*\d\S*|\S{6,})/i.test(t)) return true;
      if (/-----BEGIN [A-Z ]*PRIVATE KEY-----/.test(t)) return true;
      if (/\b(?:sk|pk|rk|ghp|gho|ghs|xox[abp]|AIza|hf)[-_][A-Za-z0-9_-]{16,}/.test(t) || /\bAKIA[0-9A-Z]{16}\b/.test(t)) return true;
      if (/(?:^|\s)[A-Za-z0-9+/_-]{32,}={0,2}(?=\s|$|[.,;])/.test(t)) return true; // a long token without spaces
      for (const m of t.matchAll(/\b(?:\d[ -]?){13,19}\b/g)) {
        const digits = m[0].replace(/\D/g, '');
        if (digits.length < 13 || digits.length > 19) continue;
        let sum = 0;
        for (let i = 0; i < digits.length; i++) {
          let d = Number(digits[digits.length - 1 - i]);
          if (i % 2 === 1) { d *= 2; if (d > 9) d -= 9; }
          sum += d;
        }
        if (sum % 10 === 0) return true;
      }
      return /\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?\b/.test(t); // IBAN
    }

    const memCategory = (c) => String(c ?? '').toLowerCase().replace(/[^a-z0-9 _-]/g, '').replace(/\s+/g, ' ').trim().slice(0, PRJ_LIMITS.category);
    const memNorm = (text) => String(text || '').replace(/\s+/g, ' ').trim().slice(0, PRJ_LIMITS.memory);

    /** Clean memory entries from untrusted data (storage, the server copy). Older entries only had id, text and createdAt. */
    function memSanitize(raw, makeId) {
      return (Array.isArray(raw) ? raw : []).filter((m) => m && typeof m.text === 'string' && m.text.trim()).slice(0, PRJ_LIMITS.memories).map((m) => {
        const createdAt = Number(m.createdAt) || 0;
        const e = { id: String(m.id || makeId()).slice(0, 64), text: m.text.slice(0, PRJ_LIMITS.memory), createdAt, updatedAt: Number(m.updatedAt) || createdAt, uses: Math.max(0, Math.min(1000, Math.round(Number(m.uses) || 0))) };
        const c = memCategory(m.category);
        if (c) e.category = c;
        return e;
      });
    }

    /** Adds a memory unless it is empty, a secret, a duplicate or the list is full. */
    function memAdd(list, text, now, makeId, category) {
      const t = memNorm(text);
      if (!t) return { list, status: 'empty' };
      if (memSecret(t)) return { list, status: 'secret' };
      if (list.some((m) => m.text.toLowerCase() === t.toLowerCase())) return { list, status: 'duplicate' };
      if (list.length >= PRJ_LIMITS.memories) return { list, status: 'full' };
      const entry = { id: makeId(), text: t, createdAt: now, updatedAt: now, uses: 0 };
      const c = memCategory(category);
      if (c) entry.category = c;
      return { list: list.concat([entry]), status: 'added', entry };
    }

    /** Entries that the model may name: by id, by position in the list (1 = first) or by a part of the text. */
    function memFind(list, args) {
      const id = args.id !== undefined && args.id !== null ? String(args.id).trim() : '';
      if (id) {
        const byId = list.filter((m) => m.id === id);
        if (byId.length) return byId;
        if (/^\d+$/.test(id) && list[Number(id) - 1]) return [list[Number(id) - 1]];
        return [];
      }
      const part = String(args.match ?? args.old_text ?? '').toLowerCase().replace(/\s+/g, ' ').trim();
      if (!part) return [];
      return list.filter((m) => m.text.toLowerCase().includes(part));
    }

    /**
     * The `memory` tool of the chat. args: { action: add | update | remove | list, text, category, id, match }.
     * Returns { list, text (what the model is told), change (what happened, for the undo card) | null }.
     */
    function memToolApply(list, args, now, makeId) {
      const a = args && typeof args === 'object' ? args : {};
      const action = String(a.action || '').toLowerCase();
      const out = (text, nextList, change) => ({ list: nextList || list, text, change: change || null });
      if (action === 'list') {
        return out(list.length ? list.map((m, i) => `${i + 1}. [${m.id}]${m.category ? ` (${m.category})` : ''} ${m.text}`).join('\n') : 'The memory is empty.');
      }
      if (action === 'add') {
        const r = memAdd(list, a.text, now, makeId, a.category);
        if (r.status === 'empty') return out('Error: "text" is required for add.');
        if (r.status === 'secret') return out('Not saved: this looks like a password, key or card number, which must never be stored.');
        if (r.status === 'full') return out(`Error: the memory is full (${PRJ_LIMITS.memories} entries). Remove an entry first.`);
        if (r.status === 'duplicate') {
          const next = list.map((m) => (m.text.toLowerCase() === memNorm(a.text).toLowerCase() ? { ...m, uses: (m.uses || 0) + 1 } : m));
          return out('Already remembered.', next);
        }
        return out('Saved.', r.list, { kind: 'add', id: r.entry.id, after: r.entry });
      }
      if (action === 'update' || action === 'remove') {
        const found = memFind(list, a);
        if (!found.length) return out('Error: no matching entry. Use action "list" to see the entries, then pass "id" or part of the old text as "match".');
        if (found.length > 1) return out(`Error: ${found.length} entries match; be more specific or pass "id". Matches:\n${found.slice(0, 5).map((m) => `[${m.id}] ${m.text}`).join('\n')}`);
        const before = found[0];
        if (action === 'remove') return out('Removed.', list.filter((m) => m.id !== before.id), { kind: 'remove', id: before.id, before });
        const text = memNorm(a.text);
        if (!text) return out('Error: "text" is required for update.');
        if (memSecret(text)) return out('Not saved: this looks like a password, key or card number, which must never be stored.');
        const after = { ...before, text, updatedAt: now, uses: (before.uses || 0) + 1 };
        const c = a.category !== undefined ? memCategory(a.category) : '';
        if (c) after.category = c;
        return out('Updated.', list.map((m) => (m.id === before.id ? after : m)), { kind: 'update', id: before.id, before, after });
      }
      return out('Error: "action" must be add, update, remove or list.');
    }

    /** Takes back one change of memToolApply (the card's Undo). Returns the new list, or null when it no longer applies. */
    function memUndo(list, change) {
      if (!change) return null;
      if (change.kind === 'add') return list.some((m) => m.id === change.id) ? list.filter((m) => m.id !== change.id) : null;
      if (change.kind === 'update') return list.some((m) => m.id === change.id) ? list.map((m) => (m.id === change.id ? change.before : m)) : null;
      if (change.kind === 'remove') return list.some((m) => m.id === change.id) || list.length >= PRJ_LIMITS.memories ? null : list.concat([change.before]);
      return null;
    }

    /**
     * The memories that go into the prompt, within `cap` characters. Priority: entries that were come back to
     * (uses >= 2) first, then the rest, each group newest first; the chosen ones keep their original order.
     */
    function memSelect(memories, cap) {
      const rank = memories.map((m, i) => ({ m, i })).sort((a, b) => {
        const ha = (a.m.uses || 0) >= 2 ? 1 : 0;
        const hb = (b.m.uses || 0) >= 2 ? 1 : 0;
        return hb - ha || b.i - a.i;
      });
      const chosen = [];
      let used = 0;
      for (const r of rank) {
        const cost = r.m.text.length + 3;
        if (used + cost > cap) continue;
        chosen.push(r);
        used += cost;
      }
      chosen.sort((a, b) => a.i - b.i);
      return { lines: chosen.map((r) => r.m.text), dropped: memories.length - chosen.length, used };
    }

    /**
     * The system prompt of a request: the default prompt (prompts.js), personalization (about me, memory), the
     * instructions of the project, then the instructions of the assistant. `personal` is null when personalization
     * is off for the chat; `memoryOff` keeps only the memory out; `memoryCap` caps the memory block.
     */
    function composeSystem(o) {
      const parts = [];
      if (o.defaults && o.defaults.trim()) parts.push(o.defaults.trim());
      const p = o.personal;
      if (p) {
        if (p.useAbout && String(p.about || '').trim()) parts.push(`About the user and how they want answers (in their own words):\n${p.about.trim().slice(0, PRJ_LIMITS.about)}`);
        if (p.useMemory && !o.memoryOff) {
          const m = memSelect(p.memories || [], o.memoryCap || PRJ_LIMITS.memoryPrompt);
          if (m.lines.length) parts.push(`Saved memory about the user (use it when it is relevant, do not bring it up otherwise):\n${m.lines.map((x) => `- ${x}`).join('\n')}`);
        }
      }
      if (o.project && String(o.project.instructions || '').trim()) {
        parts.push(`Instructions for the project "${o.project.name}":\n${o.project.instructions.trim().slice(0, PRJ_LIMITS.instructions)}`);
      }
      if (o.base && o.base.trim()) parts.push(o.base.trim());
      return parts.join('\n\n');
    }

    // ---- projects -------------------------------------------------------------------
    const prjUsed = (project) => (project.docs || []).reduce((s, d) => s + d.text.length, 0);

    /** Can this document go into the project knowledge? Cuts it to the room left, or refuses with the reason. */
    function prjFitDoc(projects, project, name, text) {
      const t = String(text || '');
      if (!t.trim()) return { error: `${name} has no readable text.` };
      const roomProject = PRJ_LIMITS.project - prjUsed(project);
      const roomTotal = PRJ_LIMITS.total - projects.reduce((s, p) => s + prjUsed(p), 0);
      if (roomProject < PRJ_LIMITS.minRoom) return { error: `${name} was not added: the knowledge of this project is full (${PRJ_LIMITS.project.toLocaleString('en-US')} characters). Remove a document first.` };
      if (roomTotal < PRJ_LIMITS.minRoom) return { error: `${name} was not added: the browser storage kept for projects is full (${PRJ_LIMITS.total.toLocaleString('en-US')} characters in all projects). Remove a document first.` };
      const room = Math.min(PRJ_LIMITS.doc, roomProject, roomTotal);
      if (t.length > room) return { text: t.slice(0, room), note: `${name} is long: only its first ${room.toLocaleString('en-US')} characters are kept in the project.` };
      return { text: t };
    }

    /** The documents searched for a question in a chat: the ones of its project, then the ones of the chat itself. */
    function prjCombinedDocs(projects, thread, chatOnly) {
      const own = (thread.docs || []).map((d) => ({ id: d.id, name: d.name, text: d.text }));
      if (chatOnly || !thread.project) return own;
      const project = projects.find((p) => p.id === thread.project);
      const shared = project ? (project.docs || []).map((d) => ({ id: d.id, name: d.name, text: d.text })) : [];
      return shared.concat(own);
    }

    /**
     * The chats of the sidebar. Without a search: chats outside projects and not archived (the archived ones only
     * when asked for). With a search: every chat that matches, archived ones and chats of projects included.
     */
    function prjSidebarView(threads, opts) {
      const q = String(opts.q || '').toLowerCase().trim();
      const known = new Set(opts.projectIds || []);
      const inProject = (t) => Boolean(t.project) && known.has(t.project);
      const hit = (t) => !q || t.title.toLowerCase().includes(q) || opts.textOf(t).includes(q);
      const sorted = threads.slice().sort((a, b) => b.updatedAt - a.updatedAt);
      const main = [];
      const archived = [];
      for (const t of sorted) {
        if (q) { if (hit(t)) (t.archived ? archived : main).push(t); continue; }
        if (inProject(t)) continue;
        if (t.archived) archived.push(t); else main.push(t);
      }
      const archivedCount = q ? archived.length : sorted.filter((t) => t.archived && !inProject(t)).length;
      return { main, archived: q || opts.showArchived ? archived : [], archivedCount };
    }

    // ---- prompt library ------------------------------------------------------------
    /** Saved prompts that match what is typed after a leading "/" (the built-in commands are left alone). */
    function plibMatch(prompts, text) {
      if (!String(text).startsWith('/')) return [];
      const q = text.slice(1).toLowerCase().trim();
      if (SLASH_COMMANDS.test(q)) return [];
      return prompts.filter((p) => !q || p.title.toLowerCase().includes(q))
        .sort((a, b) => Number(b.title.toLowerCase().startsWith(q)) - Number(a.title.toLowerCase().startsWith(q)))
        .slice(0, 8);
    }
    // @pure-end

    // ---- storage -------------------------------------------------------------------
    function prjRead(key, fallback) {
      try { const v = JSON.parse(localStorage.getItem(key) || 'null'); return v === null || v === undefined ? fallback : v; } catch { return fallback; }
    }
    function prjWrite(key, value, what) {
      try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch {
        toast(`The browser storage is full: ${what} could not be saved. Delete old chats or documents, or export them first.`);
        return false;
      }
    }

    state.prj = {
      assistants: [], projects: [], prompts: [],
      personal: { about: '', useAbout: true, useMemory: true, memories: [] },
      open: new Set(), selectMode: false, selected: new Set(), showArchived: false, pending: null, view: null,
    };

    const PRJ_SVG = {
      folder: '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
      edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
      x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
      archive: '<rect width="20" height="5" x="2" y="3" rx="1"/><path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8"/><path d="M10 12h4"/>',
      chevron: '<path d="m9 18 6-6-6-6"/>',
      bookmark: '<path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v21z"/>',
      list: '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    };
    const ASST_PATHS = {
      bot: '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
      pen: PRJ_SVG.edit,
      code: '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
      book: '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
      globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15 15 0 0 1 0 20a15 15 0 0 1 0-20"/>',
      spark: '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/>',
      chat: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
      star: '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    };
    const prjIcon = (path, cls = 'icon icon-sm') => `<svg viewBox="0 0 24 24" class="${cls}">${path}</svg>`;

    const ASST_META = {
      standard: { icon: 'chat', color: 'patina', starters: ['Explain in simple words how a language model writes a text word by word.', 'Write a fast Rust function for the dot product of two f32 slices.', 'Help me plan my week: ask me what I need to get done.', 'Summarize a topic I name in five bullet points.'] },
      precise: { icon: 'book', color: 'teal', starters: ['What is the difference between a process and a thread?', 'Give me the three most important facts about photosynthesis.', 'Is this claim correct, and how sure are you? I will paste it next.', 'Define "idempotent" in two sentences.'] },
      creative: { icon: 'spark', color: 'purple', starters: ['Write the opening paragraph of a mystery set on a night train.', 'Give me ten names for a small coffee shop by the sea.', 'Describe a city that only exists at night.', 'Invent a board game and explain its rules in a few lines.'] },
      programmer: { icon: 'code', color: 'blue', starters: ['Review this function and point out bugs. I will paste it next.', 'Write a unit test for a function that parses ISO dates.', 'Explain the difference between a mutex and a semaphore with a short example.', 'Show me how to read a file line by line in Rust.'] },
      translator: { icon: 'globe', color: 'orange', starters: ['Translate to French: "The meeting has been moved to Thursday."', 'Translate a sentence I paste next into Spanish and explain any tricky word choices.','Translate the text I paste next and keep its tone.', 'Which is the more natural English phrasing: "I am agree" or "I agree"?'] },
      editor: { icon: 'pen', color: 'rose', starters: ['Improve the style of the paragraph I paste next.', 'Fix the spelling and grammar of my text and list what you changed.', 'Make this email shorter and more polite.', 'Rewrite this text for a reader who is new to the topic.'] },
    };

    // ---- load -----------------------------------------------------------------------
    const prjMakeId = (p) => () => uid(p);

    function prjLoad() {
      const P = state.prj;
      P.assistants = (prjRead(PRJ_KEYS.assistants, []) || []).map((a) => asstSanitize(a, prjMakeId('as'))).filter(Boolean);
      const mig = asstMigrate(customPresets(), P.assistants, prjRead(PRJ_KEYS.migrated, []) || []);
      if (mig.added || mig.done.length) {
        P.assistants = mig.assistants;
        prjWrite(PRJ_KEYS.assistants, P.assistants, 'the migrated assistants');
        prjWrite(PRJ_KEYS.migrated, mig.done, 'the migrated assistants');
      }
      P.projects = (prjRead(PRJ_KEYS.projects, []) || []).filter((p) => p && typeof p.id === 'string').map((p) => ({
        id: p.id, name: String(p.name || 'Project').slice(0, PRJ_LIMITS.name), instructions: String(p.instructions || '').slice(0, PRJ_LIMITS.instructions),
        createdAt: p.createdAt || Date.now(), updatedAt: p.updatedAt || Date.now(),
        docs: (Array.isArray(p.docs) ? p.docs : []).filter((d) => d && typeof d.text === 'string').map((d) => ({ id: String(d.id), name: String(d.name || 'document'), text: d.text, pages: d.pages, chars: d.text.length, addedAt: d.addedAt || 0 })),
      }));
      const per = prjRead(PRJ_KEYS.personal, {}) || {};
      P.personal = {
        about: String(per.about || '').slice(0, PRJ_LIMITS.about), useAbout: per.useAbout !== false, useMemory: per.useMemory !== false,
        memories: memSanitize(per.memories, () => uid('mem')),
      };
      P.prompts = (Array.isArray(prjRead(PRJ_KEYS.prompts, [])) ? prjRead(PRJ_KEYS.prompts, []) : []).filter((p) => p && p.title && typeof p.text === 'string')
        .map((p) => ({ id: String(p.id || uid('pr')), title: String(p.title).slice(0, PRJ_LIMITS.promptTitle), text: p.text.slice(0, PRJ_LIMITS.promptText) })).slice(0, PRJ_LIMITS.prompts);
    }
    const prjSaveProjects = () => prjWrite(PRJ_KEYS.projects, state.prj.projects, 'the projects');
    const prjSavePersonal = () => { const ok = prjWrite(PRJ_KEYS.personal, state.prj.personal, 'your personalization'); if (ok) memSyncSoon(); return ok; };

    // ---- server copy of the memory (memory.json next to studio.json) --------------------------------
    let memSyncTimer = null;
    let memSynced = null;
    function memDoc() { return { version: 1, memories: state.prj.personal.memories }; }
    function memSyncSoon() {
      if (state.memSyncOff) return;
      clearTimeout(memSyncTimer);
      memSyncTimer = setTimeout(async () => {
        const body = memDoc();
        const text = JSON.stringify(body);
        if (text === memSynced) return;
        try {
          const res = await apiFetch('/api/studio/memory', { method: 'PUT', body });
          if (res.ok) memSynced = text; else state.memSyncOff = true; // this server has no such route, or it is not reachable from here
        } catch { /* try again with the next change */ }
      }, 800);
    }
    /** First run in a browser without stored personalization: take the memory from the copy on the server. */
    async function memImportFromServer() {
      if (localStorage.getItem(PRJ_KEYS.personal) !== null) { memSyncSoon(); return; }
      try {
        const res = await apiFetch('/api/studio/memory');
        if (!res.ok) { state.memSyncOff = true; return; }
        const doc = (await res.json()).memory;
        const list = memSanitize(doc && doc.memories, () => uid('mem'));
        if (list.length) {
          state.prj.personal.memories = list;
          prjWrite(PRJ_KEYS.personal, state.prj.personal, 'your personalization');
          memSynced = JSON.stringify(memDoc());
          memRender();
          toast(`${list.length} remembered fact${list.length === 1 ? '' : 's'} restored from the copy on this computer.`);
        }
      } catch { /* no server copy */ }
    }
    const prjSavePrompts = () => prjWrite(PRJ_KEYS.prompts, state.prj.prompts, 'the prompt library');
    const asstSaveList = () => prjWrite(PRJ_KEYS.assistants, state.prj.assistants, 'the assistants');

    // ---- assistants -----------------------------------------------------------------
    const asstAll = () => [...asstBuiltins(BUILTIN_PRESETS, ASST_META), ...state.prj.assistants];
    const asstById = (id) => asstAll().find((a) => a.id === id) || null;
    const asstThumb = (a) => (ACCENT_COLORS.find((c) => c.value === a.color) || ACCENT_COLORS[0]).thumb;
    const asstBadge = (a) => `<span class="as-badge" style="color:${asstThumb(a)};background:color-mix(in srgb, ${asstThumb(a)} 20%, transparent)">${prjIcon(ASST_PATHS[a.icon] || ASST_PATHS.bot)}</span>`;

    /** The assistant shown as active: the one of the open chat, else the one chosen for new chats; null means custom text. */
    function asstEffectiveId() {
      const t = currentThread();
      const id = t ? t.preset : state.preset;
      if (id && asstById(id)) return id;
      return $('txtSystemPrompt').value.trim() ? null : 'standard';
    }

    function asstRender() {
      const cur = asstEffectiveId();
      const a = cur ? asstById(cur) : null;
      $('presetLabel').textContent = a ? a.name : 'Custom';
      $('presetBadge').innerHTML = asstBadge(a || { icon: 'pen', color: 'patina' });
      $('btnPresets').title = `Assistant: ${a ? a.name : 'custom instructions'}. Click to choose another.`;
      $('presetItems').innerHTML = asstAll().map((x) => `
        <div class="as-row ${x.id === cur ? 'active' : ''}" data-preset="${escapeHtml(x.id)}" title="${escapeHtml(x.system || 'No instructions')}">
          ${asstBadge(x)}
          <span class="as-name"><b>${escapeHtml(x.name)}</b><small>${escapeHtml(x.system ? x.system.replace(/\s+/g, ' ').slice(0, 70) : 'No instructions')}${x.model ? ` · ${escapeHtml(x.model)}` : ''}</small></span>
          <span class="quant-chip">T ${x.temperature}</span>
          <button class="btn-icon btn-icon-xs" data-as-edit="${escapeHtml(x.id)}" title="${x.builtin ? 'Make an editable copy' : 'Edit'}">${prjIcon(PRJ_SVG.edit)}</button>
        </div>`).join('');
    }

    function asstModelKnown(name) {
      return Array.from($('modelDropdownItems').querySelectorAll('[data-pick-model]')).some((b) => b.dataset.pickModel === name);
    }

    function asstApply(id, opts = {}) {
      const a = asstById(id);
      if (!a) return;
      state.preset = id;
      localStorage.setItem(PRJ_KEYS.current, id);
      $('txtSystemPrompt').value = a.system || '';
      setSamplerUI({ temperature: a.temperature, top_p: a.top_p, top_k: a.top_k });
      if (a.model) {
        if (asstModelKnown(a.model)) {
          state.selectedModel = a.model;
          $('activeModelLabel').textContent = a.model;
        } else if (!opts.silent) {
          toast(`The model "${a.model}" of this assistant is not available here; the current model stays.`);
        }
      }
      const t = currentThread();
      if (t && !opts.noThread) { t.system = a.system || null; t.preset = id; saveThreads(); }
      asstRender();
      prjRenderStarters();
      updateContextMeter();
      if (!opts.silent) toast(`Assistant "${a.name}" selected`);
    }

    async function asstSaveCurrent() {
      const name = await askDialog({ title: 'Save as assistant', text: 'Saves the current instructions and sampling values as a new assistant.', input: '', ok: 'Save' });
      if (!name) return;
      const s = getSampler();
      const a = asstSanitize({ name, system: s.system_prompt, temperature: s.temperature, top_p: s.top_p, top_k: s.top_k, model: '' }, prjMakeId('as'));
      if (!a) return;
      state.prj.assistants = state.prj.assistants.filter((x) => x.name !== a.name).concat([a]);
      asstSaveList();
      state.preset = a.id;
      localStorage.setItem(PRJ_KEYS.current, a.id);
      const t = currentThread();
      if (t) { t.preset = a.id; saveThreads(); }
      asstRender();
      prjRenderStarters();
      toast(`Assistant "${a.name}" saved`);
    }

    // editor
    const asEd = { id: null, icon: 'bot', color: 'patina' };
    function asstEditorPaint() {
      $('asIconRow').innerHTML = ASST_ICONS.map((i) => `<button type="button" class="as-pick ${i === asEd.icon ? 'active' : ''}" data-as-icon="${i}">${prjIcon(ASST_PATHS[i])}</button>`).join('');
      $('asColorRow').innerHTML = ACCENT_COLORS.map((c) => `<button type="button" class="accent-swatch ${c.value === asEd.color ? 'active' : ''}" data-as-color="${c.value}" title="${c.name}" style="background:${c.thumb};"></button>`).join('');
    }
    function asstOpenEditor(a) {
      const copy = a && a.builtin;
      asEd.id = a && !copy ? a.id : null;
      asEd.icon = a ? a.icon : 'bot';
      asEd.color = a ? a.color : 'patina';
      $('asTitle').textContent = !a ? 'New assistant' : copy ? `Copy of "${a.name}"` : 'Edit assistant';
      $('asName').value = a ? (copy ? `${a.name} (copy)` : a.name) : '';
      $('asSystem').value = a ? a.system : '';
      $('asTemp').value = a ? a.temperature : 0.7;
      $('asTopP').value = a ? a.top_p : 0.95;
      $('asTopK').value = a ? a.top_k : 20;
      const names = Array.from($('modelDropdownItems').querySelectorAll('[data-pick-model]')).map((b) => b.dataset.pickModel);
      if (a && a.model && !names.includes(a.model)) names.push(a.model);
      $('asModel').innerHTML = `<option value="">Any (keep the current model)</option>${names.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('')}`;
      $('asModel').value = a ? a.model : '';
      $('asStarters').value = a ? (a.starters || []).join('\n') : '';
      $('asSysMode').value = a && a.sysMode ? a.sysMode : 'inherit';
      $('asSysText').value = a && a.sysText ? a.sysText : '';
      $('asDelete').style.display = asEd.id ? '' : 'none';
      asstEditorPaint();
      $('presetMenu').classList.remove('open');
      $('assistantModal').classList.add('open');
      $('asName').focus();
    }
    function asstEditorRead() {
      return asstSanitize({
        id: asEd.id || undefined, name: $('asName').value, icon: asEd.icon, color: asEd.color, system: $('asSystem').value,
        temperature: $('asTemp').value, top_p: $('asTopP').value, top_k: $('asTopK').value, model: $('asModel').value,
        starters: $('asStarters').value.split('\n'), sysMode: $('asSysMode').value, sysText: $('asSysText').value,
      }, prjMakeId('as'));
    }
    function asstEditorSave() {
      const a = asstEditorRead();
      if (!a) { toast('Give the assistant a name.'); return; }
      const list = state.prj.assistants;
      const i = list.findIndex((x) => x.id === a.id);
      if (i >= 0) list[i] = a; else list.push(a);
      if (!asstSaveList()) return;
      $('assistantModal').classList.remove('open');
      if (state.preset === a.id || asEd.id === null) asstApply(a.id, { silent: true });
      asstRender();
      prjRenderStarters();
      toast(`Assistant "${a.name}" saved`);
    }
    async function asstEditorDelete() {
      const a = asstById(asEd.id);
      if (!a || !(await askDialog({ title: 'Delete assistant?', text: `"${a.name}" will be deleted. Chats that used it keep their instructions.`, ok: 'Delete', danger: true }))) return;
      state.prj.assistants = state.prj.assistants.filter((x) => x.id !== a.id);
      asstSaveList();
      if (state.preset === a.id) { state.preset = null; localStorage.removeItem(PRJ_KEYS.current); }
      $('assistantModal').classList.remove('open');
      asstRender();
      prjRenderStarters();
    }
    function asstImportText(text) {
      const r = asstParseImport(text, prjMakeId('as'));
      if (r.error) { toast(r.error); return; }
      state.prj.assistants.push(r.assistant);
      if (asstSaveList()) { asstRender(); toast(`Assistant "${r.assistant.name}" imported`); }
    }

    $('btnPresets').addEventListener('click', (e) => { e.stopPropagation(); $('modelDropdownMenu').classList.remove('open'); $('presetMenu').classList.toggle('open'); });
    $('presetItems').addEventListener('click', (e) => {
      const edit = e.target.closest('[data-as-edit]');
      if (edit) { e.stopPropagation(); asstOpenEditor(asstById(edit.dataset.asEdit)); return; }
      const row = e.target.closest('[data-preset]');
      if (row) { asstApply(row.dataset.preset); $('presetMenu').classList.remove('open'); }
    });
    $('btnPresetSave').addEventListener('click', asstSaveCurrent);
    $('btnAsNew').addEventListener('click', () => asstOpenEditor(null));
    $('btnAsImport').addEventListener('click', () => $('fileImportAssistant').click());
    $('fileImportAssistant').addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (file) asstImportText(await file.text());
      e.target.value = '';
    });
    $('asIconRow').addEventListener('click', (e) => { const b = e.target.closest('[data-as-icon]'); if (b) { asEd.icon = b.dataset.asIcon; asstEditorPaint(); } });
    $('asColorRow').addEventListener('click', (e) => { const b = e.target.closest('[data-as-color]'); if (b) { asEd.color = b.dataset.asColor; asstEditorPaint(); } });
    $('asSave').addEventListener('click', asstEditorSave);
    $('asDelete').addEventListener('click', asstEditorDelete);
    $('asCancel').addEventListener('click', () => $('assistantModal').classList.remove('open'));
    $('asExport').addEventListener('click', () => {
      const a = asstEditorRead();
      if (a) download(`${fileSafe(a.name)}.assistant.json`, 'application/json', asstExport(a));
      else toast('Give the assistant a name first.');
    });

    // ---- starter prompts ------------------------------------------------------------
    function prjRenderStarters() {
      const id = asstEffectiveId();
      const a = (id && asstById(id)) || { name: 'Assistant', starters: [] };
      $('starterPills').innerHTML = asstStarters(a).map((s) => `<button class="starter-pill" data-prompt="${escapeHtml(s)}"><span>${escapeHtml(s.length > 110 ? `${s.slice(0, 108)}…` : s)}</span></button>`).join('') +
        (id && !asstById(id).builtin ? `<button class="starter-pill starter-edit" data-action="edit-starters" title="Edit the suggestions of this assistant"><span>Edit suggestions</span></button>` : '');
    }
    $('starterPills').addEventListener('click', (e) => {
      if (e.target.closest('[data-action="edit-starters"]')) asstOpenEditor(asstById(asstEffectiveId()));
    });

    // ---- projects -------------------------------------------------------------------
    const prjById = (id) => state.prj.projects.find((p) => p.id === id) || null;
    const prjProjectOf = (t) => (t && t.project ? prjById(t.project) : null);
    const prjChatsOf = (pid) => state.threads.filter((t) => t.project === pid).sort((a, b) => b.updatedAt - a.updatedAt);

    function prjThreadRow(t) {
      return `<div class="thread-row prj-chat ${t.id === state.activeThreadId && state.surface === 'chat' ? 'active' : ''} ${state.prj.selected.has(t.id) ? 'selected' : ''}" data-tid="${t.id}" draggable="true">` +
        `<input type="checkbox" class="sel-box" ${state.prj.selected.has(t.id) ? 'checked' : ''} tabindex="-1"><span class="thread-row-title">${escapeHtml(t.title)}</span></div>`;
    }

    function prjRenderSidebar() {
      const P = state.prj;
      const box = $('projectList');
      if (!P.projects.length) { box.innerHTML = '<div class="thread-empty">No projects yet</div>'; return; }
      box.innerHTML = P.projects.map((p) => {
        const chats = prjChatsOf(p.id).filter((t) => !t.archived);
        const open = P.open.has(p.id);
        return `<div class="prj-row" data-pid="${p.id}" data-drop="project" title="Open the project page. Drop a chat here to move it into the project.">
            <button class="prj-chev ${open ? 'open' : ''}" data-prj-toggle="${p.id}" title="${open ? 'Hide' : 'Show'} the chats">${prjIcon(PRJ_SVG.chevron)}</button>
            <span class="prj-name">${prjIcon(PRJ_SVG.folder)}<span>${escapeHtml(p.name)}</span></span>
            <span class="prj-count">${chats.length}</span>
            <div class="thread-row-actions">
              <button class="btn-icon btn-icon-xs" data-prj-ren="${p.id}" title="Rename">${prjIcon(PRJ_SVG.edit)}</button>
              <button class="btn-icon btn-icon-xs" data-prj-del="${p.id}" title="Delete the project">${prjIcon(PRJ_SVG.x)}</button>
            </div>
          </div>${open ? (chats.map(prjThreadRow).join('') || '<div class="thread-empty prj-empty">No chats yet</div>') : ''}`;
      }).join('');
    }

    async function prjCreate() {
      const name = await askDialog({ title: 'New project', text: 'A project groups chats and gives them shared instructions and documents.', input: '', ok: 'Create' });
      if (!name) return;
      const p = { id: uid('prj'), name: name.slice(0, PRJ_LIMITS.name), instructions: '', docs: [], createdAt: Date.now(), updatedAt: Date.now() };
      state.prj.projects.push(p);
      state.prj.open.add(p.id);
      if (prjSaveProjects()) { prjRenderSidebar(); prjOpenPage(p.id); }
    }
    async function prjRename(id) {
      const p = prjById(id);
      const name = p && await askDialog({ title: 'Rename project', input: p.name, ok: 'Rename' });
      if (!name) return;
      p.name = name.slice(0, PRJ_LIMITS.name);
      prjSaveProjects();
      prjRenderSidebar();
      if (state.prj.view === id) prjRenderPage();
      updateChatChrome();
    }
    async function prjDelete(id) {
      const p = prjById(id);
      if (!p) return;
      const n = prjChatsOf(id).length;
      if (!(await askDialog({ title: 'Delete project?', text: `"${p.name}" and its ${p.docs.length} document${p.docs.length === 1 ? '' : 's'} will be deleted. Its ${n} chat${n === 1 ? '' : 's'} stay in your list.`, ok: 'Delete', danger: true }))) return;
      state.threads.forEach((t) => { if (t.project === id) t.project = null; });
      state.prj.projects = state.prj.projects.filter((x) => x.id !== id);
      ragIndexCache.clear();
      prjSaveProjects();
      saveThreads();
      if (state.prj.view === id) $('projectModal').classList.remove('open');
      prjRenderSidebar();
      renderThreadList();
      updateChatChrome();
    }

    function prjMove(ids, projectId) {
      let n = 0;
      for (const t of state.threads) if (ids.includes(t.id)) { t.project = projectId || null; n++; }
      if (projectId) state.prj.open.add(projectId);
      ragIndexCache.clear();
      saveThreads();
      prjRenderSidebar();
      renderThreadList();
      updateChatChrome();
      if (n) toast(projectId ? `${n} chat${n === 1 ? '' : 's'} moved to "${prjById(projectId)?.name}"` : `${n} chat${n === 1 ? '' : 's'} taken out of the project`);
    }

    /** A small dialog to choose a project; resolves with its id, '' for "no project" or null when cancelled. */
    function prjPick(title) {
      return new Promise((resolve) => {
        $('pickTitle').textContent = title;
        $('pickList').innerHTML = `<button class="nav-item" data-pick-prj=""><span>No project</span></button>` +
          state.prj.projects.map((p) => `<button class="nav-item" data-pick-prj="${p.id}">${prjIcon(PRJ_SVG.folder)}<span>${escapeHtml(p.name)}</span></button>`).join('');
        $('pickModal').classList.add('open');
        const done = (v) => { $('pickModal').classList.remove('open'); $('pickList').onclick = $('pickCancel').onclick = null; resolve(v); };
        $('pickList').onclick = (e) => { const b = e.target.closest('[data-pick-prj]'); if (b) done(b.dataset.pickPrj); };
        $('pickCancel').onclick = () => done(null);
      });
    }

    // project page
    function prjOpenPage(id) {
      state.prj.view = id;
      prjRenderPage();
      $('projectModal').classList.add('open');
    }
    function prjRenderPage() {
      const p = prjById(state.prj.view);
      if (!p) return;
      $('prjTitle').textContent = p.name;
      if (document.activeElement !== $('prjInstr')) $('prjInstr').value = p.instructions;
      const used = prjUsed(p);
      $('prjUsage').textContent = `${used.toLocaleString('en-US')} of ${PRJ_LIMITS.project.toLocaleString('en-US')} characters used. Only the extracted text is stored, in this browser.`;
      $('prjDocs').innerHTML = p.docs.length ? p.docs.map((d) => `
        <div class="prj-doc">${prjIcon('<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5z"/><path d="M14 2v6h6"/>')}
          <span class="prj-doc-name" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</span>
          <span class="prj-doc-size">${formatBytes(d.text.length)}${d.pages ? `, ${d.pages} pages` : ''}</span>
          <button class="btn-icon btn-icon-xs" data-prj-rmdoc="${d.id}" title="Remove from the project">${prjIcon(PRJ_SVG.x)}</button>
        </div>`).join('') : '<div class="thread-empty">No documents yet. Add files that every chat of this project should be able to search.</div>';
      const chats = prjChatsOf(p.id);
      $('prjChats').innerHTML = chats.length ? chats.map((t) => `
        <button class="nav-item" data-prj-chat="${t.id}"><span class="prj-chat-title">${escapeHtml(t.title)}</span><span class="quant-chip" style="margin-left:auto;">${t.archived ? 'archived · ' : ''}${new Date(t.updatedAt).toLocaleDateString('en-US')}</span></button>`).join('')
        : '<div class="thread-empty">No chats in this project yet.</div>';
    }

    let prjInstrTimer = null;
    $('prjInstr').addEventListener('input', () => {
      const p = prjById(state.prj.view);
      if (!p) return;
      p.instructions = $('prjInstr').value.slice(0, PRJ_LIMITS.instructions);
      p.updatedAt = Date.now();
      clearTimeout(prjInstrTimer);
      prjInstrTimer = setTimeout(() => { prjSaveProjects(); updateContextMeter(); }, 400);
    });
    $('prjInstr').addEventListener('blur', () => { clearTimeout(prjInstrTimer); prjSaveProjects(); });

    async function prjAddFiles(fileList) {
      const p = prjById(state.prj.view);
      if (!p) return;
      for (const file of Array.from(fileList || [])) {
        let text;
        let pages;
        try {
          if (file.type.startsWith('image/')) { toast(`${file.name}: images cannot be searched. Add text, PDF or Office files.`); continue; }
          if (RAG_SERVER_EXT.test(file.name)) {
            toast(`Reading ${file.name}...`);
            const doc = await ragExtract(file);
            text = doc.text;
            pages = doc.pages;
          } else if (file.size > MAX_TEXT_ATTACHMENT) {
            toast(`${file.name} is too large (more than ${formatBytes(MAX_TEXT_ATTACHMENT)}).`);
            continue;
          } else {
            text = await file.text();
            if (text.includes('\u0000')) { toast(`${file.name} is not a text file.`); continue; }
          }
        } catch (e) { toast(e.message); continue; }
        const fit = prjFitDoc(state.prj.projects, p, file.name, text);
        if (fit.error) { toast(fit.error); continue; }
        if (fit.note) toast(fit.note);
        const doc = { id: uid('pdoc'), name: file.name, text: fit.text, pages, chars: fit.text.length, addedAt: Date.now() };
        p.docs.push(doc);
        p.updatedAt = Date.now();
        if (!prjSaveProjects()) { p.docs.pop(); continue; }
        ragIndexCache.clear();
        prjRenderPage();
        toast(`${file.name} added to "${p.name}"`);
      }
    }
    $('prjAddDocs').addEventListener('click', () => $('prjFileInput').click());
    $('prjFileInput').addEventListener('change', async (e) => { await prjAddFiles(e.target.files); e.target.value = ''; });
    $('prjDrop').addEventListener('dragover', (e) => { if (e.dataTransfer?.types?.includes('Files')) { e.preventDefault(); $('prjDrop').classList.add('over'); } });
    $('prjDrop').addEventListener('dragleave', () => $('prjDrop').classList.remove('over'));
    $('prjDrop').addEventListener('drop', async (e) => { e.preventDefault(); $('prjDrop').classList.remove('over'); await prjAddFiles(e.dataTransfer.files); });
    $('prjDocs').addEventListener('click', (e) => {
      const b = e.target.closest('[data-prj-rmdoc]');
      const p = prjById(state.prj.view);
      if (!b || !p) return;
      p.docs = p.docs.filter((d) => d.id !== b.dataset.prjRmdoc);
      ragIndexCache.clear();
      prjSaveProjects();
      prjRenderPage();
    });
    $('prjChats').addEventListener('click', (e) => {
      const b = e.target.closest('[data-prj-chat]');
      if (b) { $('projectModal').classList.remove('open'); openThread(b.dataset.prjChat); }
    });
    $('prjRename').addEventListener('click', () => prjRename(state.prj.view));
    $('prjDelete').addEventListener('click', () => prjDelete(state.prj.view));
    $('prjClose').addEventListener('click', () => { prjSaveProjects(); $('projectModal').classList.remove('open'); prjRenderSidebar(); });
    $('prjNewChat').addEventListener('click', () => {
      const id = state.prj.view;
      $('projectModal').classList.remove('open');
      startNewChat();
      state.prj.pending = id;
      state.prj.open.add(id);
      updateChatChrome();
      prjRenderSidebar();
    });
    $('btnNewProject').addEventListener('click', prjCreate);
    $('projectList').addEventListener('click', (e) => {
      const tog = e.target.closest('[data-prj-toggle]');
      if (tog) { const id = tog.dataset.prjToggle; if (state.prj.open.has(id)) state.prj.open.delete(id); else state.prj.open.add(id); prjRenderSidebar(); return; }
      const ren = e.target.closest('[data-prj-ren]');
      if (ren) { prjRename(ren.dataset.prjRen); return; }
      const del = e.target.closest('[data-prj-del]');
      if (del) { prjDelete(del.dataset.prjDel); return; }
      if (prjThreadListClick(e)) return;
      const chat = e.target.closest('[data-tid]');
      if (chat) { openThread(chat.dataset.tid); return; }
      const row = e.target.closest('[data-pid]');
      if (row) prjOpenPage(row.dataset.pid);
    });

    /** Called when a chat is created: it joins the project the user started it from. */
    function prjOnNewThread(t) {
      if (state.prj.pending && prjById(state.prj.pending)) t.project = state.prj.pending;
      state.prj.pending = null;
      prjRenderSidebar();
    }
    /** Called when a new empty chat is started: it starts with the chosen assistant. */
    function prjOnNewChat() {
      state.prj.pending = null;
      if (state.preset && asstById(state.preset)) asstApply(state.preset, { silent: true, noThread: true });
    }

    /** The system prompt of a request for chat `t` (or the chat about to start): default prompt, personalization, project and assistant. */
    function prjComposeSystem(t, base) {
      const P = state.prj;
      const project = t ? prjProjectOf(t) : prjById(P.pending);
      const remote = Boolean(window.fllmRemoteSelected && window.fllmRemoteSelected());
      const compact = !remote && spSize(state.status?.model) === 'compact';
      const system = composeSystem({
        defaults: spForChat(t), base, personal: t && t.noPersonal ? null : P.personal, project,
        memoryOff: Boolean(t && t.noMemory), memoryCap: compact ? PRJ_LIMITS.memoryPromptCompact : PRJ_LIMITS.memoryPrompt,
      });
      // the saved skills (skills.js): names and when to use them, so that the model can read one with the skill tool
      const skills = compact || (t && (t.noPersonal || t.temporary)) ? '' : String((window.fllmSkillIndex && window.fllmSkillIndex()) || '').trim();
      return skills ? `${system}\n\n${skills}`.trim() : system;
    }

    // ---- chat actions, archive, header ---------------------------------------------------
    function prjChrome() {
      const t = currentThread();
      if (t) state.prj.pending = null;
      const p = t ? prjProjectOf(t) : prjById(state.prj.pending);
      const badge = $('prjBadge');
      badge.style.display = p ? '' : 'none';
      if (p) { badge.dataset.pid = p.id; $('prjBadgeName').textContent = p.name; }
      $('menuArchive').textContent = t && t.archived ? 'Unarchive' : 'Archive';
      $('menuPersonal').textContent = t && t.noPersonal ? 'Personalization: off' : 'Personalization: on';
      $('menuMemory').textContent = t && t.noMemory ? 'Memory: off' : 'Memory: on';
      asstRender();
      prjRenderStarters();
      prjDecorateMessages();
    }
    $('prjBadge').addEventListener('click', () => prjOpenPage($('prjBadge').dataset.pid));

    /** Handles the chat-menu actions added by projects. Returns true when it handled the action. */
    function prjChatAction(action, t) {
      if (action === 'archive') {
        t.archived = !t.archived;
        if (t.archived) t.pinned = false;
        saveThreads(); renderThreadList(); prjRenderSidebar(); updateChatChrome();
        toast(t.archived ? 'Chat archived. Find it under "Archived" or with the search.' : 'Chat restored');
        return true;
      }
      if (action === 'project') {
        prjPick('Move the chat to a project').then((id) => { if (id !== null) prjMove([t.id], id); });
        return true;
      }
      if (action === 'personal') {
        t.noPersonal = !t.noPersonal;
        saveThreads(); updateChatChrome(); updateContextMeter();
        toast(t.noPersonal ? 'Your instructions and memory are not used in this chat.' : 'Your instructions and memory are used in this chat.');
        return true;
      }
      if (action === 'memory') {
        t.noMemory = !t.noMemory;
        saveThreads(); updateChatChrome(); updateContextMeter();
        toast(t.noMemory ? 'Memory is not used or changed in this chat.' : 'Memory is used in this chat.');
        return true;
      }
      return false;
    }

    // ---- thread list helpers (used by renderThreadList) --------------------------------------
    function prjSidebarThreads(q) {
      const P = state.prj;
      return prjSidebarView(state.threads.filter((t) => !t.temporary), { q, showArchived: P.showArchived, projectIds: P.projects.map((p) => p.id), textOf: threadText });
    }
    function prjArchiveToggleHtml(count, q) {
      if (!count || q) return '';
      return `<button class="thread-archive-toggle" data-archive-toggle>${prjIcon(PRJ_SVG.archive)}<span>${state.prj.showArchived ? 'Hide' : 'Show'} archived (${count})</span></button>`;
    }
    /** Clicks in the thread list that belong to projects (called first by the click handler of the list). True when handled. */
    function prjThreadListClick(e) {
      if (e.target.closest('[data-archive-toggle]')) { state.prj.showArchived = !state.prj.showArchived; renderThreadList(); return true; }
      const arch = e.target.closest('[data-arch-tid]');
      if (arch) {
        const t = state.threads.find((x) => x.id === arch.dataset.archTid);
        if (t) prjChatAction('archive', t);
        return true;
      }
      return prjSelectClick(e);
    }

    // select mode (bulk actions)
    function prjSelectClick(e) {
      if (!state.prj.selectMode) return false;
      const row = e.target.closest('[data-tid]');
      if (!row) return false;
      e.stopPropagation();
      const S = state.prj.selected;
      if (S.has(row.dataset.tid)) S.delete(row.dataset.tid); else S.add(row.dataset.tid);
      renderThreadList();
      prjRenderSidebar();
      prjBulkPaint();
      return true;
    }
    function prjBulkPaint() {
      const P = state.prj;
      $('bulkBar').style.display = P.selectMode ? '' : 'none';
      $('bulkCount').textContent = `${P.selected.size} selected`;
      $('btnSelectMode').classList.toggle('active', P.selectMode);
      $('sidebarMain').classList.toggle('select-mode', P.selectMode);
    }
    function prjSetSelectMode(on) {
      state.prj.selectMode = on;
      if (!on) state.prj.selected.clear();
      prjBulkPaint(); renderThreadList(); prjRenderSidebar();
    }
    $('btnSelectMode').addEventListener('click', () => prjSetSelectMode(!state.prj.selectMode));
    $('bulkDone').addEventListener('click', () => prjSetSelectMode(false));
    $('bulkBar').addEventListener('click', async (e) => {
      const b = e.target.closest('[data-bulk]');
      const ids = Array.from(state.prj.selected);
      if (!b || !ids.length) { if (b) toast('Select some chats first.'); return; }
      const action = b.dataset.bulk;
      if (action === 'move') {
        const id = await prjPick(`Move ${ids.length} chat${ids.length === 1 ? '' : 's'} to a project`);
        if (id !== null) { prjMove(ids, id); prjSetSelectMode(false); }
      } else if (action === 'archive') {
        const picked = state.threads.filter((t) => ids.includes(t.id));
        const archive = picked.some((t) => !t.archived);
        picked.forEach((t) => { t.archived = archive; if (archive) t.pinned = false; });
        saveThreads(); prjSetSelectMode(false); updateChatChrome();
        toast(`${picked.length} chat${picked.length === 1 ? '' : 's'} ${archive ? 'archived' : 'restored'}`);
      } else if (action === 'export') {
        download(`fllm-chats-selected-${new Date().toISOString().slice(0, 10)}.json`, 'application/json', JSON.stringify(state.threads.filter((t) => ids.includes(t.id)), null, 2));
      } else if (action === 'delete') {
        if (!(await askDialog({ title: `Delete ${ids.length} chat${ids.length === 1 ? '' : 's'}?`, text: 'They will be deleted for good. Export them first if you want to keep them.', ok: 'Delete', danger: true }))) return;
        state.threads = state.threads.filter((t) => !ids.includes(t.id));
        if (ids.includes(state.activeThreadId)) state.activeThreadId = null;
        saveThreads(); prjSetSelectMode(false); renderChatArea();
      }
    });

    // drag and drop: onto a project to move, onto "pin" to pin, onto "no project" to take out
    document.addEventListener('dragstart', (e) => {
      const row = e.target.closest ? e.target.closest('[data-tid][draggable="true"]') : null;
      if (!row) return;
      const S = state.prj.selected;
      const ids = state.prj.selectMode && S.has(row.dataset.tid) ? Array.from(S) : [row.dataset.tid];
      e.dataTransfer.setData('application/x-fllm-threads', JSON.stringify(ids));
      e.dataTransfer.setData('text/plain', ids.join(','));
      e.dataTransfer.effectAllowed = 'move';
      $('sidebarMain').classList.add('dragging');
    });
    document.addEventListener('dragend', () => {
      $('sidebarMain').classList.remove('dragging');
      document.querySelectorAll('[data-drop].over').forEach((el) => el.classList.remove('over'));
    });
    document.addEventListener('dragover', (e) => {
      const z = e.target.closest ? e.target.closest('[data-drop]') : null;
      if (z && e.dataTransfer?.types?.includes('application/x-fllm-threads')) { e.preventDefault(); z.classList.add('over'); }
    });
    document.addEventListener('dragleave', (e) => { const z = e.target.closest ? e.target.closest('[data-drop]') : null; if (z) z.classList.remove('over'); });
    document.addEventListener('drop', (e) => {
      const z = e.target.closest ? e.target.closest('[data-drop]') : null;
      if (!z || !e.dataTransfer?.types?.includes('application/x-fllm-threads')) return;
      e.preventDefault();
      z.classList.remove('over');
      $('sidebarMain').classList.remove('dragging');
      let ids = [];
      try { ids = JSON.parse(e.dataTransfer.getData('application/x-fllm-threads')); } catch { return; }
      if (z.dataset.drop === 'project') prjMove(ids, z.dataset.pid);
      else if (z.dataset.drop === 'unfile') prjMove(ids, null);
      else if (z.dataset.drop === 'pin') {
        state.threads.forEach((t) => { if (ids.includes(t.id)) { t.pinned = true; t.archived = false; } });
        saveThreads(); renderThreadList(); prjRenderSidebar();
        toast(ids.length === 1 ? 'Chat pinned' : `${ids.length} chats pinned`);
      }
    });

    // ---- memory -----------------------------------------------------------------------
    function prjCaptureMemory(text) {
      const P = state.prj;
      if (!P.personal.useMemory) return;
      const t = currentThread();
      if (t && (t.noPersonal || t.noMemory)) return;
      const m = memExtract(text);
      if (m) memRemember(m);
    }
    function memRemember(text) {
      const P = state.prj;
      const r = memAdd(P.personal.memories, text, Date.now(), prjMakeId('mem'));
      if (r.status === 'duplicate') { toast('Already remembered.'); return; }
      if (r.status === 'secret') { toast('That looks like a password, key or card number. It is not saved to memory.'); return; }
      if (r.status === 'full') { toast(`The memory is full (${PRJ_LIMITS.memories} entries). Delete some in Settings > Personalization.`); return; }
      if (r.status !== 'added') return;
      P.personal.memories = r.list;
      if (prjSavePersonal()) { memRender(); toast('Remembered. See Settings > Personalization.'); updateContextMeter(); }
    }

    /** Adds a "Remember" button to the action bar of every message. */
    function prjDecorateMessages() {
      $('messagesColumn').querySelectorAll('.msg-actions-bar').forEach((bar) => {
        if (bar.querySelector('[data-prj-remember]')) return;
        const b = document.createElement('button');
        b.className = 'btn-icon btn-icon-xs';
        b.dataset.prjRemember = '1';
        b.title = 'Remember something from this message';
        b.innerHTML = prjIcon(PRJ_SVG.bookmark);
        const stats = bar.querySelector('.msg-stats');
        bar.insertBefore(b, stats);
      });
    }
    $('messagesColumn').addEventListener('click', async (e) => {
      const b = e.target.closest('[data-prj-remember]');
      if (!b) return;
      const t = currentThread();
      const node = t?.nodes[b.closest('[data-node]')?.dataset.node];
      if (!node) return;
      if (!state.prj.personal.useMemory) { toast('Memory is switched off in Settings > Personalization.'); return; }
      const text = await askDialog({ title: 'Remember', text: 'This is added to the instructions of every chat. Edit it first if you like. Memories stay in this browser.', input: messageText(node).replace(/\s+/g, ' ').trim().slice(0, 300), ok: 'Remember' });
      if (text) memRemember(text);
    });

    // settings: Personalization
    /** Settings > Personalization > System prompt: the choice and the prompt as it is sent now. */
    function spRender() {
      const stored = spStored();
      if (document.activeElement !== $('spText')) $('spText').value = stored.text;
      $('spMode').value = stored.mode;
      const t = currentThread();
      $('spPreview').value = spForChat(t);
      const model = state.status?.model || {};
      $('spInfo').textContent = window.fllmRemoteSelected && window.fllmRemoteSelected()
        ? 'The selected model runs at an external provider: the full prompt is used and it does not say that the model runs locally.'
        : `Model family: ${spFamily(model)}. Size: ${spSize(model)}${spSize(model) === 'compact' ? ' (small model or small context)' : ''}. Lines about tools appear when web access, memory or chat search are on.`;
    }
    $('spMode').addEventListener('change', () => { spSaveStored({ mode: $('spMode').value, text: $('spText').value }); spRender(); updateContextMeter(); });
    $('spText').addEventListener('input', () => { spSaveStored({ mode: $('spMode').value, text: $('spText').value.slice(0, SP_LIMITS.custom) }); $('spPreview').value = spForChat(currentThread()); updateContextMeter(); });

    function memRender() {
      const P = state.prj;
      spRender();
      const per = P.personal;
      if (document.activeElement !== $('memAbout')) $('memAbout').value = per.about;
      $('memAboutCount').textContent = `${per.about.length} / ${PRJ_LIMITS.about}`;
      $('chkUseAbout').checked = per.useAbout;
      $('chkUseMemory').checked = per.useMemory;
      const sel = memSelect(per.memories, PRJ_LIMITS.memoryPrompt);
      $('memUsage').textContent = per.memories.length
        ? `${per.memories.length} of ${PRJ_LIMITS.memories} entries. ${sel.dropped ? `The ${sel.dropped} oldest do not fit into the ${PRJ_LIMITS.memoryPrompt.toLocaleString('en-US')} characters sent to the model and are skipped.` : 'All of them are sent to the model.'}`
        : 'Nothing remembered yet.';
      $('memList').innerHTML = per.memories.map((m) => `
        <div class="mem-row"><input type="text" class="ui-input" maxlength="${PRJ_LIMITS.memory}" data-mem-edit="${m.id}" value="${escapeHtml(m.text)}" title="${escapeHtml(`${m.category ? `${m.category}, ` : ''}saved ${m.createdAt ? new Date(m.createdAt).toLocaleDateString('en-US') : 'earlier'}${m.updatedAt && m.updatedAt !== m.createdAt ? `, changed ${new Date(m.updatedAt).toLocaleDateString('en-US')}` : ''}`)}">
          ${m.category ? `<span class="quant-chip">${escapeHtml(m.category)}</span>` : ''}<button class="btn-icon btn-icon-xs" data-mem-del="${m.id}" title="Delete">${prjIcon(PRJ_SVG.x)}</button></div>`).join('');
      $('plibList').innerHTML = P.prompts.length ? P.prompts.map((p) => `
        <div class="plib-row"><input type="text" class="ui-input" maxlength="${PRJ_LIMITS.promptTitle}" data-plib-title="${p.id}" value="${escapeHtml(p.title)}" placeholder="Title">
          <textarea class="ui-textarea" rows="2" maxlength="${PRJ_LIMITS.promptText}" data-plib-text="${p.id}" placeholder="Prompt">${escapeHtml(p.text)}</textarea>
          <button class="btn-icon btn-icon-xs" data-plib-del="${p.id}" title="Delete">${prjIcon(PRJ_SVG.x)}</button></div>`).join('')
        : '<div class="thread-empty">No saved prompts yet.</div>';
    }
    $('memAbout').addEventListener('input', () => {
      state.prj.personal.about = $('memAbout').value.slice(0, PRJ_LIMITS.about);
      $('memAboutCount').textContent = `${state.prj.personal.about.length} / ${PRJ_LIMITS.about}`;
      prjSavePersonal();
      updateContextMeter();
    });
    $('chkUseAbout').addEventListener('change', () => { state.prj.personal.useAbout = $('chkUseAbout').checked; prjSavePersonal(); updateContextMeter(); });
    $('chkUseMemory').addEventListener('change', () => { state.prj.personal.useMemory = $('chkUseMemory').checked; prjSavePersonal(); updateContextMeter(); });
    $('memAdd').addEventListener('click', () => {
      const text = $('memNew').value;
      if (!text.trim()) return;
      memRemember(text);
      $('memNew').value = '';
    });
    $('memNew').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('memAdd').click(); });
    $('memList').addEventListener('change', (e) => {
      const inp = e.target.closest('[data-mem-edit]');
      const m = inp && state.prj.personal.memories.find((x) => x.id === inp.dataset.memEdit);
      if (!m) return;
      const text = inp.value.replace(/\s+/g, ' ').trim().slice(0, PRJ_LIMITS.memory);
      if (text && memSecret(text)) { toast('That looks like a password, key or card number. It is not saved to memory.'); memRender(); return; }
      if (text) { m.text = text; m.updatedAt = Date.now(); } else state.prj.personal.memories = state.prj.personal.memories.filter((x) => x !== m);
      prjSavePersonal(); memRender(); updateContextMeter();
    });
    $('memList').addEventListener('click', (e) => {
      const d = e.target.closest('[data-mem-del]');
      if (!d) return;
      state.prj.personal.memories = state.prj.personal.memories.filter((x) => x.id !== d.dataset.memDel);
      prjSavePersonal(); memRender(); updateContextMeter();
    });
    $('memClear').addEventListener('click', async () => {
      if (!state.prj.personal.memories.length) return;
      if (!(await askDialog({ title: 'Forget everything?', text: 'All remembered facts will be deleted.', ok: 'Delete all', danger: true }))) return;
      state.prj.personal.memories = [];
      prjSavePersonal(); memRender(); updateContextMeter();
    });

    // prompt library
    $('plibAdd').addEventListener('click', () => {
      if (state.prj.prompts.length >= PRJ_LIMITS.prompts) { toast(`The library holds at most ${PRJ_LIMITS.prompts} prompts.`); return; }
      state.prj.prompts.push({ id: uid('pr'), title: `Prompt ${state.prj.prompts.length + 1}`, text: '' });
      prjSavePrompts(); memRender();
    });
    $('plibList').addEventListener('change', (e) => {
      const ti = e.target.closest('[data-plib-title]');
      const tx = e.target.closest('[data-plib-text]');
      const p = state.prj.prompts.find((x) => x.id === (ti ? ti.dataset.plibTitle : tx && tx.dataset.plibText));
      if (!p) return;
      if (ti) p.title = ti.value.trim().slice(0, PRJ_LIMITS.promptTitle) || p.title;
      if (tx) p.text = tx.value.slice(0, PRJ_LIMITS.promptText);
      prjSavePrompts();
    });
    $('plibList').addEventListener('click', (e) => {
      const d = e.target.closest('[data-plib-del]');
      if (!d) return;
      state.prj.prompts = state.prj.prompts.filter((x) => x.id !== d.dataset.plibDel);
      prjSavePrompts(); memRender();
    });

    // "/" in the composer lists the saved prompts
    const slash = { items: [], index: 0 };
    function plibUpdateSlash() {
      const text = $('chatTextarea').value;
      const items = text.startsWith('/') && !text.includes('\n') ? plibMatch(state.prj.prompts, text) : [];
      slash.items = items;
      slash.index = Math.min(slash.index, Math.max(0, items.length - 1));
      const menu = $('slashMenu');
      menu.classList.toggle('open', items.length > 0);
      menu.innerHTML = items.map((p, i) => `<button class="slash-item ${i === slash.index ? 'active' : ''}" data-slash="${i}"><b>/${escapeHtml(p.title)}</b><small>${escapeHtml(p.text.replace(/\s+/g, ' ').slice(0, 80))}</small></button>`).join('');
    }
    function plibPick(i) {
      const p = slash.items[i];
      if (!p) return;
      const ta = $('chatTextarea');
      ta.value = p.text;
      slash.items = [];
      $('slashMenu').classList.remove('open');
      ta.dispatchEvent(new Event('input'));
      ta.focus();
    }
    $('chatTextarea').addEventListener('input', plibUpdateSlash);
    $('chatTextarea').addEventListener('keydown', (e) => {
      if (!$('slashMenu').classList.contains('open')) return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        slash.index = (slash.index + (e.key === 'ArrowDown' ? 1 : slash.items.length - 1)) % slash.items.length;
        plibUpdateSlash();
      } else if ((e.key === 'Enter' && !e.shiftKey) || e.key === 'Tab') {
        plibPick(slash.index);
      } else if (e.key === 'Escape') {
        $('slashMenu').classList.remove('open');
        slash.items = [];
      } else return;
      e.preventDefault();
      e.stopImmediatePropagation();
    }, true);
    $('slashMenu').addEventListener('mousedown', (e) => {
      const b = e.target.closest('[data-slash]');
      if (b) { e.preventDefault(); plibPick(Number(b.dataset.slash)); }
    });

    // closing the dialogs of this file
    ['assistantModal', 'projectModal', 'pickModal'].forEach((id) => {
      $(id).addEventListener('mousedown', (e) => { if (e.target === $(id)) { if (id === 'projectModal') prjSaveProjects(); $(id).classList.remove('open'); } });
    });
    window.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape' || $('dialogModal').classList.contains('open')) return;
      ['pickModal', 'assistantModal', 'projectModal'].some((id) => {
        if (!$(id).classList.contains('open')) return false;
        $(id).classList.remove('open');
        return true;
      });
    });
    $('settingsMenuNav').addEventListener('click', (e) => { if (e.target.closest('[data-spane="personalization"]')) memRender(); });

    // ---- start (called by boot.js after the chats are loaded) -------------------------------------
    function prjInit() {
      prjLoad();
      const saved = localStorage.getItem(PRJ_KEYS.current);
      if (saved && saved !== 'standard' && asstById(saved)) {
        asstApply(saved, { silent: true, noThread: true });
        state._samplerLoaded = true; // the sampling values of the assistant win over the defaults of the server
      }
      asstRender();
      prjRenderStarters();
      prjRenderSidebar();
      prjBulkPaint();
      memRender();
      memImportFromServer();
    }
