    // ==========================================================================
    // Artifacts and answers: a preview panel for HTML/SVG/Markdown/CSV code blocks (with versions),
    // a sandboxed JavaScript runner, message actions, response styles and temporary chats.
    // ==========================================================================

    // @pure-begin  (no DOM access in this block: it is loaded by tests/artifacts.test.js)
    const STYLES = [
      { id: 'default', label: 'Default', prompt: '' },
      { id: 'concise', label: 'Concise', prompt: 'Keep answers short and to the point. Skip preamble and repetition.' },
      { id: 'explanatory', label: 'Explanatory', prompt: 'Explain your reasoning step by step with clear examples, and define terms that may be unfamiliar.' },
      { id: 'formal', label: 'Formal', prompt: 'Use a formal, professional tone and precise wording.' },
      { id: 'casual', label: 'Casual', prompt: 'Use a relaxed, friendly, conversational tone.' },
    ];

    /** The system prompt of a chat plus the instruction of its response style. */
    function withStylePrompt(base, styleId) {
      const style = STYLES.find((s) => s.id === styleId);
      if (!style || !style.prompt) return base;
      return base ? `${base}\n\n${style.prompt}` : style.prompt;
    }

    const KIND_BY_LANG = { html: 'html', htm: 'html', xhtml: 'html', svg: 'svg', md: 'markdown', markdown: 'markdown', csv: 'csv', tsv: 'csv' };
    const EXT_BY_LANG = { python: 'py', py: 'py', javascript: 'js', js: 'js', typescript: 'ts', ts: 'ts', rust: 'rs', rs: 'rs', go: 'go', java: 'java', c: 'c', cpp: 'cpp', sh: 'sh', bash: 'sh', sql: 'sql', json: 'json', css: 'css', yaml: 'yaml', yml: 'yaml', toml: 'toml' };
    const KIND_EXT = { html: 'html', svg: 'svg', markdown: 'md', csv: 'csv' };
    const KIND_MIME = { html: 'text/html', svg: 'image/svg+xml', markdown: 'text/markdown', csv: 'text/csv', code: 'text/plain' };

    /** What a fenced block can be previewed as: html, svg, markdown, csv; anything else is plain code. */
    function artifactKind(lang) {
      return KIND_BY_LANG[String(lang || '').toLowerCase()] || 'code';
    }

    /** Reads `title="..."` (also `artifact=` or `name=`, quoted or bare) from the info string of a fence. */
    function parseInfo(info) {
      const m = /(?:^|\s)(?:title|artifact|name)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))/i.exec(String(info || ''));
      return { title: m ? (m[1] ?? m[2] ?? m[3] ?? '').trim() : '' };
    }

    /** Closed fenced blocks of a message that are previewable or carry a title. */
    function extractArtifacts(text) {
      const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
      const found = [];
      for (let i = 0; i < lines.length; i++) {
        const m = lines[i].match(/^\s*(`{3,}|~{3,})\s*([\w+#.-]*)([^\n]*)$/);
        if (!m) continue;
        const fence = m[1];
        const body = [];
        let closed = false;
        for (i++; i < lines.length; i++) {
          if (lines[i].trim().startsWith(fence[0].repeat(fence.length)) && /^\s*[`~]+\s*$/.test(lines[i])) { closed = true; break; }
          body.push(lines[i]);
        }
        if (!closed) continue;
        const kind = artifactKind(m[2]);
        const title = parseInfo(m[3]).title;
        if (kind !== 'code' || title) found.push({ lang: m[2].toLowerCase(), kind, title, code: body.join('\n') });
      }
      return found;
    }

    /**
     * Groups the artifacts of a conversation (assistant messages, oldest first: `{id, text}`) into
     * artifacts with versions. A block with a title belongs to the artifact of the same title and kind.
     * An untitled block belongs to the artifact with identical code, else to the most recently changed
     * artifact of the same kind, else it starts a new one. One message changes an artifact at most once.
     */
    function buildFamilies(messages) {
      const families = [];
      let clock = 0;
      for (const msg of messages) {
        const touched = new Set();
        extractArtifacts(msg.text).forEach((a, index) => {
          let fam = null;
          if (a.title) {
            fam = families.find((f) => f.kind === a.kind && f.title.toLowerCase() === a.title.toLowerCase() && !touched.has(f)) || null;
          } else {
            const same = families.filter((f) => f.kind === a.kind);
            fam = same.find((f) => f.versions[f.versions.length - 1].code === a.code)
              || same.filter((f) => !touched.has(f)).sort((x, y) => y.touched - x.touched)[0] || null;
          }
          if (!fam) {
            fam = { id: `${msg.id}#${index}`, kind: a.kind, title: a.title, lang: a.lang, versions: [], touched: 0 };
            families.push(fam);
          }
          const last = fam.versions[fam.versions.length - 1];
          if (!last || last.code !== a.code) fam.versions.push({ n: fam.versions.length + 1, msgId: msg.id, code: a.code });
          fam.touched = ++clock;
          touched.add(fam);
        });
      }
      return families;
    }

    function fileNameFor(a) {
      const base = String(a.title || '').replace(/[^A-Za-z0-9._ -]+/g, '').trim().replace(/\s+/g, '-').slice(0, 60) || 'artifact';
      const ext = KIND_EXT[a.kind] || EXT_BY_LANG[String(a.lang || '').toLowerCase()] || 'txt';
      return `${base}.${ext}`;
    }

    const PREVIEW_CSP = "default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'";

    /** The document shown in the sandboxed iframe. Its own CSP forbids every network request. */
    function previewDocument(kind, code) {
      const head = `<!doctype html><meta http-equiv="Content-Security-Policy" content="${PREVIEW_CSP}"><meta charset="utf-8">` +
        '<style>:root{color-scheme:light}html{background:#fff;color:#111}</style>';
      if (kind === 'svg') {
        return `${head}<style>html,body{height:100%;margin:0}body{display:flex;align-items:center;justify-content:center}svg{max-width:100%;max-height:100%}</style>${code}`;
      }
      return head + code;
    }

    /** Comma, semicolon or tab separated text -> rows. Quoted fields may hold the separator and line breaks. */
    function parseCsv(text) {
      const src = String(text || '').replace(/\r\n?/g, '\n');
      const first = src.split('\n')[0] || '';
      const count = (c) => first.split(c).length;
      const delim = [',', ';', '\t'].reduce((best, c) => (count(c) > count(best) ? c : best), ',');
      const rows = [];
      let row = [];
      let cell = '';
      let quoted = false;
      for (let i = 0; i < src.length; i++) {
        const c = src[i];
        if (quoted) {
          if (c === '"' && src[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') quoted = false; else cell += c;
        } else if (c === '"' && !cell) quoted = true;
        else if (c === delim) { row.push(cell); cell = ''; } else if (c === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; } else cell += c;
      }
      if (cell || row.length) { row.push(cell); rows.push(row); }
      return rows.filter((r) => r.length > 1 || r[0] !== '');
    }

    // @pure-end

    // ---- state and helpers ---------------------------------------------------------
    const ART = { open: false, famId: null, ver: 0, tab: 'preview', threadId: null, count: 0, sig: '', large: false, fixed: null };
    const KIND_LABEL = { html: 'HTML page', svg: 'SVG image', markdown: 'Markdown document', csv: 'Table (CSV)', code: 'Code' };
    const RUN_TIMEOUT_MS = 5000;

    Object.assign(ICON, {
      up: '<path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/>',
      down: '<path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/>',
      speak: '<path d="M11 5 6 9H2v6h4l5 4V5Z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>',
      more: '<path d="m6 17 5-5-5-5"/><path d="m13 17 5-5-5-5"/>',
      check: '<path d="M20 6 9 17l-5-5"/>',
      ghost: '<path d="M9 10h.01"/><path d="M15 10h.01"/><path d="M12 2a8 8 0 0 0-8 8v12l3-3 2.5 2.5L12 19l2.5 2.5L17 19l3 3V10a8 8 0 0 0-8-8z"/>',
    });

    /** Shows "Copied" on a button for a moment. */
    function flashCopied(btn) {
      if (!btn || btn.dataset.flash) return;
      const original = btn.innerHTML;
      btn.dataset.flash = '1';
      btn.classList.add('is-copied');
      btn.innerHTML = `${svgIcon('check')}<span class="copied-label">Copied</span>`;
      setTimeout(() => {
        btn.innerHTML = original;
        btn.classList.remove('is-copied');
        delete btn.dataset.flash;
      }, 1300);
    }

    async function copyText(text, btn) {
      try {
        await navigator.clipboard.writeText(text);
        flashCopied(btn);
      } catch { toast('The browser did not allow copying.'); }
    }

    /** The visible text of an answer (no Markdown marks), for "Copy" and read-aloud. */
    function plainTextOf(nodeId, fallback) {
      const el = $('messagesColumn').querySelector(`[data-node="${nodeId}"] .msg-assistant-prose`);
      return (el && el.innerText.trim()) || fallback;
    }

    // ---- code block actions (Preview, Run) -------------------------------------------
    const isJsLang = (lang) => ['js', 'javascript', 'mjs', 'node'].includes(String(lang || '').toLowerCase());
    MD.hooks.actions = (lang, info, code, id) => {
      const kind = artifactKind(lang);
      const title = parseInfo(info).title;
      let html = '';
      if (kind !== 'code' || title) {
        html += `<button class="btn btn-xs btn-ghost" data-art-open="${id}" data-kind="${kind}" data-title="${escapeHtml(title)}" data-lang="${escapeHtml(lang || '')}" title="Open in the side panel">${kind === 'code' ? 'Open' : 'Preview'}</button>`;
      }
      if (isJsLang(lang)) html += `<button class="btn btn-xs btn-ghost" data-run-code="${id}" title="Run in a sandboxed worker (no network, 5 s limit)">Run</button>`;
      return html;
    };

    // ---- artifact panel -----------------------------------------------------------------
    function artFamilies() {
      const t = currentThread();
      if (!t) return [];
      return buildFamilies(pathOf(t).filter((n) => n.role === 'assistant' && !n.streaming).map((n) => ({ id: n.id, text: messageText(n) })));
    }

    const currentFamily = () => (ART.famId === 'adhoc' ? ART.fixed : artFamilies().find((f) => f.id === ART.famId) || null);

    function ensurePanel() {
      let p = $('artPanel');
      if (p) return p;
      p = document.createElement('aside');
      p.id = 'artPanel';
      p.className = 'art-panel';
      p.setAttribute('aria-label', 'Artifact preview');
      p.innerHTML = `
        <div class="art-head">
          <div class="art-title"><strong data-art-title></strong><small data-art-kind></small></div>
          <span class="art-versions" data-art-versions>
            <button class="btn-icon btn-icon-xs" data-art="prev" title="Previous version">${svgIcon('prev')}</button>
            <span class="tabular" data-art-vlabel></span>
            <button class="btn-icon btn-icon-xs" data-art="next" title="Next version">${svgIcon('next')}</button>
          </span>
          <button class="btn btn-xs btn-ghost" data-art="close" title="Close the panel">Close</button>
        </div>
        <div class="art-toolbar">
          <span class="seg"><button data-art="tab-preview">Preview</button><button data-art="tab-code">Code</button></span>
          <span class="art-tools">
            <button class="btn btn-xs btn-ghost" data-art="reload" title="Run the preview again">Reload</button>
            <button class="btn btn-xs btn-ghost" data-art="copy">Copy</button>
            <button class="btn btn-xs btn-ghost" data-art="download">Download</button>
            <button class="btn btn-xs btn-ghost" data-art="large" title="Larger view">Larger</button>
          </span>
        </div>
        <div class="art-body" data-art-body></div>`;
      $('chatContainer').appendChild(p);
      p.addEventListener('click', (e) => {
        const b = e.target.closest('[data-art]');
        if (b) artAction(b.dataset.art, b);
      });
      return p;
    }

    function closePanel() {
      ART.open = false;
      ART.large = false;
      ART.famId = null;
      ART.sig = '';
      $('chatContainer').classList.remove('art-open');
      $('artPanel')?.classList.remove('is-large');
      const body = $('artPanel')?.querySelector('[data-art-body]');
      if (body) body.textContent = '';
    }

    function openFamily(fam, ver) {
      ART.open = true;
      ART.famId = fam.id;
      ART.fixed = fam.id === 'adhoc' ? fam : null;
      ART.ver = Math.max(0, Math.min(ver, fam.versions.length - 1));
      ART.count = fam.versions.length;
      ART.tab = fam.kind === 'code' ? 'code' : 'preview';
      ART.threadId = currentThread()?.id || null;
      $('chatContainer').classList.add('art-open');
      renderPanel(true);
    }

    function renderPanel(force) {
      const fam = currentFamily();
      if (!fam) { closePanel(); return; }
      ART.ver = Math.max(0, Math.min(ART.ver, fam.versions.length - 1));
      const v = fam.versions[ART.ver];
      const p = ensurePanel();
      if (fam.kind === 'code') ART.tab = 'code';
      p.querySelector('[data-art-title]').textContent = fam.title || KIND_LABEL[fam.kind];
      p.querySelector('[data-art-kind]').textContent = fam.title ? KIND_LABEL[fam.kind] : '';
      p.querySelector('[data-art-versions]').style.display = fam.versions.length > 1 ? 'inline-flex' : 'none';
      p.querySelector('[data-art-vlabel]').textContent = `v${v.n} / ${fam.versions.length}`;
      p.querySelector('[data-art="prev"]').disabled = ART.ver === 0;
      p.querySelector('[data-art="next"]').disabled = ART.ver === fam.versions.length - 1;
      p.querySelector('[data-art="tab-preview"]').style.display = fam.kind === 'code' ? 'none' : '';
      p.querySelector('[data-art="tab-preview"]').classList.toggle('active', ART.tab === 'preview');
      p.querySelector('[data-art="tab-code"]').classList.toggle('active', ART.tab === 'code');
      p.querySelector('[data-art="reload"]').style.display = ART.tab === 'preview' && (fam.kind === 'html' || fam.kind === 'svg') ? '' : 'none';
      const sig = `${fam.id}|${ART.ver}|${ART.tab}|${v.code}`;
      if (!force && sig === ART.sig) return;
      ART.sig = sig;
      const body = p.querySelector('[data-art-body]');
      body.textContent = '';
      if (ART.tab === 'code') {
        const pre = document.createElement('pre');
        pre.className = 'art-code';
        pre.innerHTML = `<code>${MD.highlight(v.code, fam.lang)}</code>`;
        body.appendChild(pre);
      } else if (fam.kind === 'html' || fam.kind === 'svg') {
        const frame = document.createElement('iframe');
        frame.className = 'art-frame';
        frame.setAttribute('sandbox', 'allow-scripts');
        frame.setAttribute('referrerpolicy', 'no-referrer');
        frame.setAttribute('title', 'Preview');
        frame.srcdoc = previewDocument(fam.kind, v.code);
        body.appendChild(frame);
      } else if (fam.kind === 'markdown') {
        const doc = document.createElement('div');
        doc.className = 'art-doc msg-assistant-prose';
        doc.innerHTML = MD.render(v.code);
        body.appendChild(doc);
      } else if (fam.kind === 'csv') {
        const rows = parseCsv(v.code);
        const shown = rows.slice(0, 500);
        const cell = (tag, c) => `<${tag}>${escapeHtml(c)}</${tag}>`;
        const doc = document.createElement('div');
        doc.className = 'art-doc md-table';
        doc.innerHTML = shown.length
          ? `<table><thead><tr>${shown[0].map((c) => cell('th', c)).join('')}</tr></thead><tbody>${shown.slice(1).map((r) => `<tr>${r.map((c) => cell('td', c)).join('')}</tr>`).join('')}</tbody></table>` +
            (rows.length > shown.length ? `<p class="art-note">Showing the first ${shown.length} of ${rows.length} rows.</p>` : '')
          : '<p class="art-note">The table is empty.</p>';
        body.appendChild(doc);
      }
    }

    function artAction(action, btn) {
      const fam = currentFamily();
      if (!fam) return;
      const v = fam.versions[ART.ver];
      if (action === 'close') closePanel();
      else if (action === 'prev') { ART.ver--; renderPanel(); } else if (action === 'next') { ART.ver++; renderPanel(); } else if (action === 'tab-preview') { ART.tab = 'preview'; renderPanel(); } else if (action === 'tab-code') { ART.tab = 'code'; renderPanel(); } else if (action === 'reload') renderPanel(true);
      else if (action === 'copy') copyText(v.code, btn);
      else if (action === 'download') download(fileNameFor(fam), KIND_MIME[fam.kind] || 'text/plain', v.code);
      else if (action === 'large') {
        ART.large = !ART.large;
        $('artPanel').classList.toggle('is-large', ART.large);
        btn.textContent = ART.large ? 'Smaller' : 'Larger';
      }
    }

    /** "Preview" on a code block: open the artifact (and the version) that block belongs to. */
    function openFromBlock(btn) {
      const code = $(btn.dataset.artOpen)?.textContent ?? '';
      const nodeId = btn.closest('[data-node]')?.dataset.node;
      const families = artFamilies();
      let hit = null;
      for (const f of families) {
        const i = f.versions.findIndex((v) => v.code === code && v.msgId === nodeId);
        const j = i >= 0 ? i : f.versions.findIndex((v) => v.code === code);
        if (j >= 0 && (!hit || i >= 0)) hit = { fam: f, ver: j };
        if (i >= 0) break;
      }
      if (hit) { openFamily(hit.fam, hit.ver); return; }
      openFamily({ id: 'adhoc', kind: btn.dataset.kind || 'code', title: btn.dataset.title || '', lang: btn.dataset.lang || '', versions: [{ n: 1, msgId: nodeId, code }] }, 0);
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && ART.large) { ART.large = false; $('artPanel').classList.remove('is-large'); $('artPanel').querySelector('[data-art="large"]').textContent = 'Larger'; }
    });

    // ---- JavaScript runner ------------------------------------------------------------
    const RUNS = new WeakMap();

    function stopRun(card, why) {
      const run = RUNS.get(card);
      if (!run) return;
      clearTimeout(run.timer);
      run.worker.terminate();
      RUNS.delete(card);
      card.querySelector('[data-run-code]')?.replaceChildren(document.createTextNode('Run'));
      const status = card.querySelector('.run-status');
      if (status && why) { status.textContent = why; status.classList.add('is-warn'); }
    }

    function runBlock(btn) {
      const card = btn.closest('.code-card');
      if (!card) return;
      if (RUNS.has(card)) { stopRun(card, 'Stopped'); return; }
      const code = $(btn.dataset.runCode)?.textContent ?? '';
      let out = card.querySelector('.run-out');
      if (!out) { out = document.createElement('div'); out.className = 'run-out'; card.appendChild(out); }
      out.innerHTML = '<div class="run-status">Running...</div><pre class="run-lines"></pre>';
      const status = out.querySelector('.run-status');
      const lines = out.querySelector('.run-lines');
      let worker;
      try {
        // A script of the server, not a Blob: it comes with its own CSP (eval allowed, network forbidden),
        // so the page itself never needs 'unsafe-eval'.
        worker = new Worker('/studio/runner.js');
      } catch (e) {
        status.textContent = `The runner could not start: ${e.message}`;
        status.classList.add('is-warn');
        return;
      }
      const t0 = performance.now();
      const run = { worker, timer: setTimeout(() => stopRun(card, `Stopped after ${RUN_TIMEOUT_MS / 1000} s (time limit)`), RUN_TIMEOUT_MS) };
      RUNS.set(card, run);
      btn.textContent = 'Stop';
      const addLine = (cls, text) => {
        const line = document.createElement('span');
        line.className = `run-line ${cls}`;
        line.textContent = text;
        lines.appendChild(line);
        lines.appendChild(document.createTextNode('\n'));
      };
      worker.onmessage = (e) => {
        const m = e.data || {};
        if (m.t === 'log') { addLine(`is-${m.level}`, m.text); return; }
        if (m.t !== 'done') return;
        if (m.error) addLine('is-error', m.error);
        else if (m.value !== undefined) addLine('is-value', `-> ${m.value}`);
        stopRun(card, null);
        status.textContent = `${m.error ? 'Failed' : 'Done'} in ${Math.round(performance.now() - t0)} ms`;
        status.classList.toggle('is-warn', Boolean(m.error));
      };
      worker.onerror = (e) => {
        e.preventDefault();
        addLine('is-error', e.message || 'The worker failed.');
        stopRun(card, 'Failed');
      };
      worker.postMessage({ code });
    }

    // ---- message actions -----------------------------------------------------------------
    /** Extra buttons for the action bar of an answer (called from chat.js). */
    function assistantExtraActions(node, isLast) {
      const btn = (act, icon, title, on) => `<button class="btn-icon btn-icon-xs${on ? ' is-on' : ''}" data-act="${act}" title="${escapeHtml(title)}">${icon}</button>`;
      let h = btn('copy-md', '<span class="act-text">MD</span>', 'Copy as Markdown', false);
      if (typeof window.fllmSpeak === 'function') h += btn('speak', svgIcon('speak'), 'Read aloud', false);
      h += btn('up', svgIcon('up'), node.rating === 'up' && node.ratingNote ? `Good answer: ${node.ratingNote}` : 'Good answer', node.rating === 'up');
      h += btn('down', svgIcon('down'), node.rating === 'down' && node.ratingNote ? `Bad answer: ${node.ratingNote}` : 'Bad answer', node.rating === 'down');
      if (isLast && node.finish === 'length' && !node.streaming) h += btn('continue', `${svgIcon('more')}<span class="copied-label">Continue</span>`, 'The answer was cut by the length limit: continue it', false);
      return h;
    }

    async function rateMessage(node, value, wrap) {
      if (node.rating === value) {
        delete node.rating;
        delete node.ratingNote;
      } else {
        node.rating = value;
        const note = await askDialog({ title: value === 'up' ? 'What was good?' : 'What was wrong?', text: 'Optional note. It is stored only in this browser, with the chat, and is part of the chat export.', input: node.ratingNote || '', ok: 'Save' });
        if (note !== null) { if (note) node.ratingNote = note; else delete node.ratingNote; }
      }
      saveThreads();
      for (const v of ['up', 'down']) {
        const b = wrap.querySelector(`[data-act="${v}"]`);
        if (!b) continue;
        b.classList.toggle('is-on', node.rating === v);
        b.title = node.rating === v && node.ratingNote ? `${v === 'up' ? 'Good' : 'Bad'} answer: ${node.ratingNote}` : `${v === 'up' ? 'Good' : 'Bad'} answer`;
      }
    }

    $('messagesColumn').addEventListener('click', async (e) => {
      const open = e.target.closest('[data-art-open]');
      if (open) { openFromBlock(open); return; }
      const run = e.target.closest('[data-run-code]');
      if (run) { runBlock(run); return; }
      const act = e.target.closest('[data-act]');
      if (!act) return;
      const wrap = act.closest('[data-node]');
      const t = currentThread();
      const node = wrap && t?.nodes[wrap.dataset.node];
      if (!node) return;
      const kind = act.dataset.act;
      if (kind === 'copy') flashCopied(act);
      else if (kind === 'copy-md') await copyText(messageText(node), act);
      else if (kind === 'up' || kind === 'down') await rateMessage(node, kind, wrap);
      else if (kind === 'speak') window.fllmSpeak?.(plainTextOf(node.id, messageText(node)));
      else if (kind === 'continue') { if (!state.abortController) await generateReply(t, node.parent, node); }
    });

    // ---- response style ---------------------------------------------------------------------
    const styleOf = (t) => (t ? t.style : state.newStyle) || 'default';

    function renderStyleMenu() {
      const current = styleOf(currentThread());
      $('styleMenu').innerHTML = STYLES.map((s) => `<button class="menu-item${s.id === current ? ' is-current' : ''}" data-style="${s.id}" title="${escapeHtml(s.prompt || 'No extra instruction')}">${escapeHtml(s.label)}</button>`).join('');
      const style = STYLES.find((s) => s.id === current) || STYLES[0];
      $('styleLabel').textContent = current === 'default' ? 'Style' : `Style: ${style.label}`;
      $('btnStyle').classList.toggle('active', current !== 'default');
    }

    $('btnStyle').addEventListener('click', (e) => { e.stopPropagation(); $('styleMenu').classList.toggle('open'); });
    $('styleMenu').addEventListener('click', (e) => {
      const item = e.target.closest('[data-style]');
      if (!item) return;
      const t = currentThread();
      if (t) { t.style = item.dataset.style; saveThreads(); } else state.newStyle = item.dataset.style;
      $('styleMenu').classList.remove('open');
      renderStyleMenu();
    });
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#styleMenu') && !e.target.closest('#btnStyle')) $('styleMenu').classList.remove('open');
    });

    // ---- temporary chats ------------------------------------------------------------------------
    /** Called when the chat changes: a temporary chat that is left is gone for good. */
    function leaveThread(keepId) {
      const before = state.threads.length;
      state.threads = state.threads.filter((t) => !t.temporary || t.id === keepId);
      if (state.threads.length !== before && !keepId) state.activeThreadId = null;
      if (!keepId || !state.threads.find((t) => t.id === keepId)?.temporary) state.tempNext = false;
    }

    async function toggleTemporary() {
      const t = currentThread();
      if (t && t.temporary) {
        if (pathOf(t).length && !(await askDialog({ title: 'Close the temporary chat?', text: 'It was never saved and will be deleted now.', ok: 'Close it', danger: true }))) return;
        startNewChat();
      } else if (state.tempNext && !t) {
        state.tempNext = false;
      } else {
        startNewChat();
        state.tempNext = true;
      }
      chatExtrasSync();
    }
    $('btnTemporary').addEventListener('click', toggleTemporary);

    // ---- keeping everything in step with the chat ---------------------------------------------------
    /** Called after the chat area is drawn: panel, style chip and temporary-chat banner follow the active chat. */
    function chatExtrasSync() {
      const t = currentThread();
      if (ART.open) {
        if (!t || t.id !== ART.threadId) closePanel();
        else if (ART.famId !== 'adhoc') {
          const fam = artFamilies().find((f) => f.id === ART.famId);
          if (!fam) closePanel();
          else {
            if (fam.versions.length > ART.count) ART.ver = fam.versions.length - 1;
            ART.count = fam.versions.length;
            renderPanel();
          }
        }
      }
      renderStyleMenu();
      const temp = Boolean(t ? t.temporary : state.tempNext);
      $('tempBanner').hidden = !temp;
      $('btnTemporary').classList.toggle('active', temp);
      $('btnTemporary').setAttribute('aria-pressed', temp ? 'true' : 'false');
    }
    chatExtrasSync();
