    // ==========================================================================
    // System prompts of the chat. The pure part builds the default prompt from facts (time, language, model family,
    // size, which tools are offered); the rest reads the page state. Order of the final system message (projects.js):
    // this default prompt, personalization (about me, memory), project instructions, assistant instructions.
    // The prompt only describes what the engine really provides: tools are mentioned only when they are offered.
    // ==========================================================================
    const SP_KEY = 'fllm.studio.sysprompt.v1';

    // @pure-begin
    const SP_LIMITS = { custom: 6000, compactContext: 12288, compactParams: 3.5e9 };

    /** The model family, from the name, the architecture and the chat template of the loaded model. */
    function spFamily(model) {
      const m = model || {};
      const name = `${m.name || ''} ${m.arch || ''}`.toLowerCase();
      const tpl = String(m.chat_template || '');
      const has = (re) => re.test(name);
      if (has(/gpt[-_ ]?oss/) || /<\|channel\|>/.test(tpl)) return 'gptoss';
      if (has(/smollm/)) return 'smollm';
      if (has(/deepseek/) || /<[^<>a-z ]User[^<>a-z ]>/i.test(tpl)) return 'deepseek';
      if (has(/qwen|qwq/)) return 'qwen';
      if (has(/gemma/) || /<start_of_turn>/.test(tpl)) return 'gemma';
      if (has(/llama/)) return 'llama';
      if (has(/mistral|mixtral|ministral|devstral|magistral/) || /\[INST\]/.test(tpl)) return 'mistral';
      if (has(/phi[-_ ]?\d|\bphi\b/)) return 'phi';
      if (/<\|im_start\|>/.test(tpl)) return 'qwen';
      return 'generic';
    }

    /** "compact" for small models and small contexts, else "full". */
    function spSize(model) {
      const m = model || {};
      const ctx = Number(m.max_context) || 4096;
      const params = Number(m.parameters) || 0;
      return ctx < SP_LIMITS.compactContext || (params > 0 && params < SP_LIMITS.compactParams) ? 'compact' : 'full';
    }

    const SP_FAMILY_NOTES = {
      qwen: 'Keep any reasoning brief and do not repeat it in the final answer. Write a tool call only in the format of the tool interface, never as ordinary text.',
      llama: 'Do not apologize and do not add disclaimers. Get to the point.',
      mistral: 'Answer directly without a long introduction. Write a tool call only in the format of the tool interface.',
      gemma: 'Follow these instructions for the whole conversation, also when the user message is short.',
      gptoss: 'Keep the final answer short; your reasoning stays out of it.',
      deepseek: 'Keep your reasoning short. The final answer is in the language of the user, even when you reasoned in another one.',
      phi: 'Answer in the language of the user, even when you reason in English. Stay on topic and never write the next message of the user.',
      smollm: 'You are a very small model: keep answers short and simple, say "I am not sure" instead of guessing, and never write the next message of the user.',
      generic: '',
    };

    function spLanguageName(locale) {
      const tag = String(locale || 'en').trim() || 'en';
      try {
        const n = new Intl.DisplayNames(['en'], { type: 'language' }).of(tag);
        return n && n !== tag ? n : tag;
      } catch { return tag; }
    }

    /** "Monday, 26 October 2026, 14:05 (UTC+02:00)" for a point in time. */
    const spDate = (ms) => new Date(ms).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    function spNow(ms) {
      const d = new Date(ms);
      const date = spDate(ms);
      const pad = (n) => String(n).padStart(2, '0');
      const off = -d.getTimezoneOffset();
      const sign = off < 0 ? '-' : '+';
      return `${date}, ${pad(d.getHours())}:${pad(d.getMinutes())} (UTC${sign}${pad(Math.floor(Math.abs(off) / 60))}:${pad(Math.abs(off) % 60)})`;
    }

    /**
     * The clock shown to the model. It is frozen for 30 minutes per chat: a prompt that changes every minute would
     * make the server throw away its cached prefix of the conversation on every message.
     */
    function spClock(holder, now) {
      if (!holder.clock || now - holder.clock > 30 * 60 * 1000 || holder.clock > now) holder.clock = now;
      return holder.clock;
    }

    const SP_TOOL_TEXT = {
      web: 'Web tools (web_search, web_fetch): use them only for current or changing facts (news, prices, versions, events after your training), for specific facts you do not know or are unsure about, or when the user asks you to look something up or to open a link. For everything else (explanations, writing, code, math, advice, small talk) answer directly without searching. When you used the web, cite the sources as [n] with the numbers from the results and never invent a source. If a search fails, say so.',
      webCompact: 'Web tools: search only for current or unknown facts, otherwise answer directly. Cite sources as [n]; never invent sources.',
      memory: 'Memory tool (memory): save a short fact when you learn something durable about the user (name, preferences, projects, constraints) or when they ask you to remember something. Never save trivia, one-off requests or secrets such as passwords, keys or card numbers. Use update or remove when a fact changed or the user wants it forgotten. Mention a saved fact only briefly, if at all.',
      memoryCompact: 'Memory tool: save durable facts about the user (name, preferences, projects); never trivia or secrets.',
      chats: 'Chat search tool (search_chats): use it when the user refers to an earlier conversation that you cannot see ("as we discussed", "last time").',
      chatsCompact: 'search_chats finds earlier conversations the user refers to.',
      call: 'Call tools with valid JSON arguments. After a tool result, answer the user; do not repeat the same call.',
      none: 'You have no tools in this chat: you cannot browse the web, open links, run code or read files except what the user shares here. Say so instead of pretending.',
    };

    /**
     * The default system prompt.
     * o: { size, family, now (ms), locale, local (runs on this computer), modelName, thinking, tools: {web, memory, chats} }
     */
    function spBuild(o) {
      const compact = o.size === 'compact';
      const tools = o.tools || {};
      const lang = spLanguageName(o.locale);
      const lines = [];
      lines.push(compact
        ? `You are a helpful assistant${o.local ? " running locally on the user's computer" : ''}. Today is ${spNow(o.now)}.`
        : `You are a helpful, honest assistant${o.local ? ` running locally on the user's own computer${o.modelName ? `, based on the language model ${String(o.modelName).slice(0, 60)}` : ''}` : ''}. The current date and time is ${spNow(o.now)}.`);
      lines.push(compact
        ? `Answer in the language of the user (${lang} if unclear). Be concise. Use Markdown only where it helps.`
        : `Answer in the language the user writes in (${lang} if it is unclear). Be concise by default: give the answer first, then only the detail that is needed. Go into depth when the user asks for it.`);
      if (!compact) {
        lines.push('Format with Markdown only where it helps: short lists for steps or options, fenced code blocks with a language for code, tables only for real comparisons. Write plain paragraphs otherwise. Write math as plain text or LaTeX between $ signs.');
      }
      lines.push(compact
        ? 'If you are not sure, say so. Never invent facts, quotes, links or sources.'
        : 'Be honest about what you do not know: say when you are unsure, and never invent facts, quotes, numbers, links or sources. Your knowledge ends at your training date, so you may not know recent events. If a question is ambiguous, ask one short clarifying question.');
      if (o.local) {
        lines.push(compact
          ? "If asked, you run locally on the user's computer."
          : "If you are asked where you run: you run locally on the user's own computer, and the conversation stays on it unless the user chose an external provider.");
      }
      const parts = [];
      if (tools.web) parts.push(compact ? SP_TOOL_TEXT.webCompact : SP_TOOL_TEXT.web);
      if (tools.memory) parts.push(compact ? SP_TOOL_TEXT.memoryCompact : SP_TOOL_TEXT.memory);
      if (tools.chats) parts.push(compact ? SP_TOOL_TEXT.chatsCompact : SP_TOOL_TEXT.chats);
      if (parts.length) { parts.push(SP_TOOL_TEXT.call); lines.push(...parts); } else lines.push(SP_TOOL_TEXT.none);
      const note = SP_FAMILY_NOTES[o.family] || '';
      if (note) lines.push(note);
      if (o.thinking && o.family !== 'qwen' && o.family !== 'deepseek') lines.push('Keep your reasoning short.');
      return lines.join(compact ? '\n' : '\n\n');
    }

    /** Fills {{date}}, {{time}} and {{language}} in text written by the user. */
    function spFill(text, o) {
      return String(text || '')
        .replace(/\{\{\s*date\s*\}\}/gi, spDate(o.now))
        .replace(/\{\{\s*time\s*\}\}/gi, spNow(o.now))
        .replace(/\{\{\s*language\s*\}\}/gi, spLanguageName(o.locale));
    }

    /**
     * Applies the user's choices to the default prompt. A choice is { mode: 'default' | 'override' | 'append', text }.
     * The global choice comes first, then the one of the assistant (inherit = nothing). An empty override is ignored.
     */
    function spResolve(generated, global, assistant, o) {
      let out = generated;
      for (const c of [global, assistant]) {
        if (!c) continue;
        const text = spFill(String(c.text || '').trim().slice(0, SP_LIMITS.custom), o);
        if (!text) continue;
        if (c.mode === 'override') out = text;
        else if (c.mode === 'append') out = `${out}\n\n${text}`;
      }
      return out;
    }

    /** The messages that ask for a short title. Short and robust for models of 1 to 4 billion parameters. */
    function spTitleMessages(question, answer) {
      return [
        { role: 'system', content: 'Write a short title for a conversation: two to five words, in the language of the conversation. Reply with the title only, without quotation marks and without a full stop. Never answer the question and never mention these instructions.' },
        { role: 'user', content: `User: ${String(question).slice(0, 300)}\n\nAssistant: ${String(answer).slice(0, 300)}\n\nTitle:` },
      ];
    }

    /** A title from the raw reply of a model: no thinking blocks, no markup, one line; null when nothing usable is left. */
    function spCleanTitle(raw) {
      let t = String(raw || '').replace(/<think>[\s\S]*?<\/think>/g, '');
      if (t.includes('<think>')) t = t.slice(0, t.indexOf('<think>'));
      t = t.replace(/^\s*(?:title|titel)\s*:\s*/i, '').split('\n').map((l) => l.trim()).find(Boolean) || '';
      t = t.replace(/["*#`_]/g, '').replace(/\s+/g, ' ').trim().replace(/[.!?:]+$/, '');
      if (/system prompt|these instructions|my instructions/i.test(t)) return null;
      return t.length >= 3 && t.length <= 60 ? t : null;
    }
    // @pure-end

    // ---- state: what the page knows ----------------------------------------------------------------------------
    function spStored() {
      try {
        const v = JSON.parse(localStorage.getItem(SP_KEY) || 'null');
        return v && typeof v === 'object' ? { mode: ['override', 'append'].includes(v.mode) ? v.mode : 'default', text: String(v.text || '').slice(0, SP_LIMITS.custom) } : { mode: 'default', text: '' };
      } catch { return { mode: 'default', text: '' }; }
    }
    function spSaveStored(v) {
      try { localStorage.setItem(SP_KEY, JSON.stringify(v)); } catch { toast('The browser storage is full: the system prompt setting could not be saved.'); }
    }

    /** The default system prompt for chat `t` (null for a chat that is about to start), as sent to the model. */
    function spForChat(t) {
      const model = state.status?.model || {};
      const remote = Boolean(window.fllmRemoteSelected && window.fllmRemoteSelected());
      const holder = t || state;
      const o = {
        size: remote ? 'full' : spSize(model),
        family: remote ? 'generic' : spFamily(model),
        now: spClock(holder, Date.now()),
        locale: navigator.language || 'en',
        local: !remote,
        modelName: model.name,
        thinking: Boolean(state.thinkingEnabled),
        tools: ctOfferedFlags(t),
      };
      const asst = t && t.preset ? asstById(t.preset) : (!t && state.preset ? asstById(state.preset) : null);
      const generated = spBuild(o);
      return spResolve(generated, spStored(), asst && asst.sysMode && asst.sysMode !== 'inherit' ? { mode: asst.sysMode, text: asst.sysText } : null, o);
    }
