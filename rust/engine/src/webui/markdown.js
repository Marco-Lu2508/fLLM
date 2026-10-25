    // ==========================================================================
    // Markdown: block and inline parser with safe HTML output, and a small code highlighter.
    // All text is escaped first; only the tags produced here reach the page.
    // ==========================================================================
    const MD = (() => {
      const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

      // ---- highlighter -------------------------------------------------------
      const KEYWORDS = {
        rust: 'as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while',
        js: 'async await break case catch class const continue default delete do else export extends false finally for from function if import in instanceof let new null of return static super switch this throw true try typeof undefined var void while with yield',
        ts: 'abstract as async await break case catch class const continue declare default do else enum export extends false finally for from function if implements import in instanceof interface let namespace new null of private protected public readonly return static super switch this throw true try type typeof undefined var void while yield',
        py: 'and as assert async await break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield',
        go: 'break case chan const continue default defer else fallthrough false for func go goto if import interface map nil package range return select struct switch true type var',
        c: 'auto break case char const continue default do double else enum extern float for goto if inline int long register return short signed sizeof static struct switch typedef union unsigned void volatile while class namespace new delete template typename public private protected virtual override nullptr true false bool using',
        java: 'abstract boolean break byte case catch char class const continue default do double else enum extends false final finally float for if implements import instanceof int interface long native new null package private protected public return short static super switch synchronized this throw throws true try void volatile while var record',
        sh: 'if then else elif fi for while do done case esac function in return export local echo cd exit set unset source alias test',
        sql: 'select from where insert into values update set delete create table drop alter join left right inner outer on group by order having limit as and or not null is in like distinct union primary key foreign references index',
        css: '',
        json: 'true false null',
        toml: 'true false',
        yaml: 'true false null yes no',
      };
      const ALIASES = {
        rs: 'rust', rust: 'rust', js: 'js', javascript: 'js', jsx: 'js', mjs: 'js', node: 'js', ts: 'ts', tsx: 'ts', typescript: 'ts',
        py: 'py', python: 'py', python3: 'py', go: 'go', golang: 'go', c: 'c', h: 'c', cpp: 'c', 'c++': 'c', cc: 'c', hpp: 'c', cs: 'c', csharp: 'c',
        java: 'java', kotlin: 'java', kt: 'java', swift: 'java', sh: 'sh', bash: 'sh', shell: 'sh', zsh: 'sh', powershell: 'sh', ps1: 'sh', bat: 'sh', cmd: 'sh',
        sql: 'sql', css: 'css', scss: 'css', json: 'json', jsonc: 'json', toml: 'toml', yaml: 'yaml', yml: 'yaml',
      };
      const keywordSets = {};
      const keywordSet = (lang) => keywordSets[lang] || (keywordSets[lang] = new Set((KEYWORDS[lang] || '').split(' ').filter(Boolean)));

      function highlight(code, langName) {
        const lang = ALIASES[String(langName || '').toLowerCase()];
        if (!lang || code.length > 40000) return esc(code);
        const hashComment = lang === 'py' || lang === 'sh' || lang === 'toml' || lang === 'yaml';
        const slashComment = !hashComment && lang !== 'json' && lang !== 'sql';
        const parts = [];
        if (slashComment) parts.push('\\/\\*[\\s\\S]*?(?:\\*\\/|$)', '\\/\\/[^\\n]*');
        if (hashComment) parts.push('#[^\\n]*');
        if (lang === 'sql') parts.push('--[^\\n]*');
        parts.push('"(?:\\\\.|[^"\\\\\\n])*"?', "'(?:\\\\.|[^'\\\\\\n])*'?");
        if (lang === 'js' || lang === 'ts' || lang === 'go') parts.push('`(?:\\\\.|[^`\\\\])*`?');
        parts.push('\\b0x[0-9a-fA-F_]+\\b', '\\b\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b', '[A-Za-z_][A-Za-z0-9_]*');
        const re = new RegExp(parts.join('|'), 'g');
        const words = keywordSet(lang);
        const ci = lang === 'sql';
        let out = '';
        let last = 0;
        let m;
        while ((m = re.exec(code))) {
          out += esc(code.slice(last, m.index));
          const tok = m[0];
          const c = tok[0];
          let cls = '';
          if (c === '/' && (tok[1] === '/' || tok[1] === '*')) cls = 'tok-c';
          else if (c === '#' && hashComment) cls = 'tok-c';
          else if (c === '-' && tok[1] === '-') cls = 'tok-c';
          else if (c === '"' || c === "'" || c === '`') cls = 'tok-s';
          else if (/^\d/.test(tok)) cls = 'tok-n';
          else if (words.has(ci ? tok.toLowerCase() : tok)) cls = 'tok-k';
          else if (/^[A-Z]/.test(tok) && lang !== 'sh' && lang !== 'json') cls = 'tok-t';
          else if (code[m.index + tok.length] === '(' || (lang === 'rust' && code[m.index + tok.length] === '!')) cls = 'tok-f';
          out += cls ? `<span class="${cls}">${esc(tok)}</span>` : esc(tok);
          last = m.index + tok.length;
          if (tok.length === 0) re.lastIndex++;
        }
        return out + esc(code.slice(last));
      }

      // ---- math: a small, safe LaTeX subset (no library) -----------------------
      // Fractions, powers, subscripts, greek letters, roots, sums, integrals and common symbols.
      // Everything is escaped; anything not understood makes the caller show the source in a chip.
      const GREEK = {
        alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ',
        lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', varpi: 'ϖ', rho: 'ρ', varrho: 'ϱ', sigma: 'σ', varsigma: 'ς', tau: 'τ', upsilon: 'υ', phi: 'φ', varphi: 'φ',
        chi: 'χ', psi: 'ψ', omega: 'ω',
      };
      const GREEK_UP = { Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ', Upsilon: 'Υ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω' };
      const TEX_SYM = {
        cdot: '·', times: '×', div: '÷', pm: '±', mp: '∓', leq: '≤', le: '≤', geq: '≥', ge: '≥', neq: '≠', ne: '≠', approx: '≈', equiv: '≡', sim: '∼', propto: '∝',
        infty: '∞', partial: '∂', nabla: '∇', to: '→', rightarrow: '→', leftarrow: '←', gets: '←', Rightarrow: '⇒', Leftarrow: '⇐', Leftrightarrow: '⇔', leftrightarrow: '↔',
        mapsto: '↦', implies: '⟹', iff: '⟺', in: '∈', notin: '∉', subset: '⊂', subseteq: '⊆', supset: '⊃', cup: '∪', cap: '∩', forall: '∀', exists: '∃', ldots: '…',
        cdots: '⋯', dots: '…', circ: '∘', ell: 'ℓ', angle: '∠', neg: '¬', land: '∧', lor: '∨', wedge: '∧', vee: '∨', emptyset: '∅', varnothing: '∅', mid: '|', ast: '∗',
        star: '⋆', oplus: '⊕', otimes: '⊗', langle: '⟨', rangle: '⟩', lbrace: '{', rbrace: '}', lVert: '‖', rVert: '‖', lvert: '|', rvert: '|', prime: '′', degree: '°',
        ',': ' ', ';': ' ', ':': ' ', '!': '', quad: ' ', qquad: '  ', ' ': ' ', '{': '{', '}': '}', '%': '%', '&': '&', '#': '#', _: '_', $: '$', '|': '‖',
        '(': '(', ')': ')', '[': '[', ']': ']', '/': '/', '<': '<', '>': '>', '.': '',
      };
      const TEX_BIG = { sum: '∑', prod: '∏', coprod: '∐', bigcup: '⋃', bigcap: '⋂', int: '∫', iint: '∬', oint: '∮' };
      const TEX_FN = new Set('sin cos tan cot sec csc sinh cosh tanh arcsin arccos arctan log ln lg exp lim max min sup inf det gcd arg dim ker deg mod Pr'.split(' '));
      const TEX_ACCENT = { hat: '̂', widehat: '̂', bar: '̄', vec: '⃗', dot: '̇', ddot: '̈', tilde: '̃', widetilde: '̃' };
      const BB = { R: 'ℝ', N: 'ℕ', Z: 'ℤ', Q: 'ℚ', C: 'ℂ', P: 'ℙ', H: 'ℍ' };

      /** LaTeX source -> safe HTML, or null when something is not understood. */
      function texToHtml(src, display) {
        const s = String(src);
        if (!s.trim() || s.length > 2000) return null;
        let i = 0;
        let depth = 0;
        const fail = () => { throw new Error('tex'); };
        const skipWs = () => { while (i < s.length && /\s/.test(s[i])) i++; };
        const rawGroup = () => {
          skipWs();
          if (s[i] !== '{') fail();
          let level = 0;
          const start = ++i;
          for (; i < s.length; i++) {
            if (s[i] === '\\') { i++; continue; }
            if (s[i] === '{') level++;
            else if (s[i] === '}') { if (!level) { const out = s.slice(start, i); i++; return out; } level--; }
          }
          return fail();
        };
        function seq(end) {
          if (++depth > 30) fail();
          let html = '';
          for (;;) {
            skipWs();
            if (i >= s.length) { if (end) fail(); break; }
            if (end && s[i] === end) { i++; break; }
            html += scripted();
          }
          depth--;
          return html;
        }
        function arg() {
          skipWs();
          if (i >= s.length) fail();
          if (s[i] === '{') { i++; return seq('}'); }
          return atom().h;
        }
        function scripted() {
          const base = '^_'.includes(s[i]) ? { h: '' } : atom();
          let sup = '';
          let sub = '';
          for (;;) {
            skipWs();
            if (s[i] === '^' && !sup) { i++; sup = arg(); } else if (s[i] === '_' && !sub) { i++; sub = arg(); } else break;
          }
          if (!sup && !sub) return base.h;
          if (base.limits && display) {
            return `<span class="tx-lim"><span class="tx-top">${sup}</span><span class="tx-big">${base.h}</span><span class="tx-bot">${sub}</span></span>`;
          }
          if (sup && sub) return `${base.h}<span class="tx-ss"><sup>${sup}</sup><sub>${sub}</sub></span>`;
          return sup ? `${base.h}<sup>${sup}</sup>` : `${base.h}<sub>${sub}</sub>`;
        }
        function atom() {
          const c = s[i++];
          if (c === '{') return { h: seq('}') };
          if (c === '\\') return command();
          if (c === '}' || c === '^' || c === '_') return fail();
          if (/[A-Za-z]/.test(c)) return { h: `<i>${c}</i>` };
          if (c === '-') return { h: '−' };
          if (c === '*') return { h: '∗' };
          if (c === "'") return { h: '′' };
          if (c === '~' || c === '&') return { h: ' ' };
          return { h: esc(c) };
        }
        function delimiter() {
          skipWs();
          if (s[i] === '\\') { i++; const m = /^[A-Za-z]+|^./.exec(s.slice(i)); i += m[0].length; return esc(TEX_SYM[m[0]] ?? fail()); }
          return esc(s[i++] === '.' ? '' : s[i - 1]);
        }
        function command() {
          const m = /^[A-Za-z]+|^[\s\S]/.exec(s.slice(i));
          if (!m) return fail();
          const name = m[0];
          i += name.length;
          if (GREEK[name]) return { h: `<i>${GREEK[name]}</i>` };
          if (GREEK_UP[name]) return { h: GREEK_UP[name] };
          if (name === '\\') return { h: '<br>' };
          if (TEX_BIG[name]) return { h: `<span class="tx-op">${TEX_BIG[name]}</span>`, limits: name !== 'int' && name !== 'iint' && name !== 'oint' };
          if (TEX_FN.has(name)) return { h: `<span class="tx-fn">${name}</span>`, limits: name === 'lim' || name === 'max' || name === 'min' };
          if (Object.prototype.hasOwnProperty.call(TEX_SYM, name)) return { h: esc(TEX_SYM[name]) };
          switch (name) {
            case 'frac': case 'dfrac': case 'tfrac': { const a = arg(); const b = arg(); return { h: `<span class="tx-frac"><span class="tx-num">${a}</span><span class="tx-den">${b}</span></span>` }; }
            case 'sqrt': {
              let idx = '';
              if (s[i] === '[') { i++; idx = seq(']'); }
              const body = arg();
              return { h: `${idx ? `<sup class="tx-rootn">${idx}</sup>` : ''}<span class="tx-sqrt"><span class="tx-rad">√</span><span class="tx-rcnt">${body}</span></span>` };
            }
            case 'text': case 'mathrm': case 'textrm': case 'mbox': case 'operatorname': return { h: `<span class="tx-rm">${esc(rawGroup())}</span>` };
            case 'textbf': return { h: `<b>${esc(rawGroup())}</b>` };
            case 'mathbf': case 'boldsymbol': case 'bm': return { h: `<b>${arg()}</b>` };
            case 'mathbb': return { h: esc(rawGroup().replace(/[A-Z]/g, (ch) => BB[ch] || ch)) };
            case 'mathcal': case 'mathit': case 'mathsf': case 'mathtt': case 'displaystyle': case 'textstyle': case 'limits': case 'nolimits': return name.startsWith('math') ? { h: arg() } : { h: '' };
            case 'left': case 'right': case 'big': case 'Big': case 'bigg': case 'Bigg': return { h: delimiter() };
            case 'overline': return { h: `<span class="tx-over">${arg()}</span>` };
            case 'underline': return { h: `<span class="tx-under">${arg()}</span>` };
            default:
              if (TEX_ACCENT[name]) return { h: `${arg()}${TEX_ACCENT[name]}` };
              return fail();
          }
        }
        try {
          const html = seq(null);
          return html ? html : null;
        } catch {
          return null;
        }
      }

      /** Math as a styled span; the plain source in a chip when it cannot be typeset. */
      function mathSpan(src, display) {
        const html = texToHtml(src, display);
        if (html === null) return `<span class="math-inline" title="Shown as source: not a supported LaTeX expression">${esc(src)}</span>`;
        return `<span class="tex${display ? ' tex-display' : ''}">${html}</span>`;
      }

      // ---- inline ------------------------------------------------------------
      const safeUrl = (u) => /^(https?:\/\/|mailto:|#|\/)/i.test(u.trim()) ? u.trim() : '';

      function inline(src) {
        const stash = [];
        const keep = (html) => `\u0000${stash.push(html) - 1}\u0000`;
        let s = String(src);
        // code spans first: their content is literal
        s = s.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, (_, __, code) => keep(`<code class="inline-code">${esc(code.trim())}</code>`));
        // math: typeset when the small renderer understands it, otherwise shown as source in a chip
        s = s.replace(/\$\$([^$\n]+)\$\$|\\\[([^\n]+?)\\\]|\\\(([^\n]+?)\\\)|(?<![\w$\\])\$(?!\s)([^$\n]*?[^\s$\\])\$(?![\w$])/g,
          (_, a, b, c, d) => keep(mathSpan(a || b || c || d, Boolean(a || b))));
        // links and images
        s = s.replace(/!?\[([^\]\n]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (all, label, url) => {
          const href = safeUrl(url);
          if (!href) return all;
          const text = all.startsWith('!') ? (label || 'Image') : label;
          return keep(`<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${inlineText(text)}</a>`);
        });
        s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?])/g, (_, pre, url) => pre + keep(`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(url)}</a>`));
        s = inlineText(s);
        return s.replace(/\u0000(\d+)\u0000/g, (_, i) => stash[Number(i)]);
      }

      function inlineText(text) {
        return esc(text)
          .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
          .replace(/(?<![\w])__([^_\n]+?)__(?![\w])/g, '<strong>$1</strong>')
          .replace(/(?<![*\w])\*([^*\s][^*\n]*?)\*(?![*\w])/g, '<em>$1</em>')
          .replace(/(?<![\w])_([^_\s][^_\n]*?)_(?![\w])/g, '<em>$1</em>')
          .replace(/~~([^~\n]+?)~~/g, '<del>$1</del>');
      }

      // ---- blocks ------------------------------------------------------------
      let codeSeq = 0;
      // Other scripts (artifacts.js) set `hooks.actions` to add buttons such as Preview or Run to a block.
      const hooks = { actions: () => '' };
      function codeBlock(lang, code, closed, info) {
        const id = `cb${++codeSeq}`;
        const label = esc(lang || 'text');
        let extra = '';
        try { extra = closed ? hooks.actions(lang, info || '', code, id) || '' : ''; } catch { extra = ''; }
        return `<div class="code-card${closed ? '' : ' is-open'}" data-lang="${esc(String(lang || '').toLowerCase())}"><div class="code-card-header"><span>${label}</span>` +
          `<span class="code-card-actions">${extra}<button class="btn btn-xs btn-ghost" data-wrap-code="${id}" title="Toggle line wrapping">Wrap</button>` +
          `<button class="btn btn-xs btn-ghost" data-copy-code="${id}">Copy</button></span></div>` +
          `<pre id="${id}"><code>${highlight(code, lang)}</code></pre></div>`;
      }

      const isTableSep = (l) => /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(l) && l.includes('-') && l.includes('|');
      const splitRow = (l) => l.trim().replace(/^\|/, '').replace(/\|$/, '').split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, '|'));
      const indentOf = (l) => l.match(/^\s*/)[0].replace(/\t/g, '    ').length;
      const listMarker = (l) => l.match(/^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/);

      function render(md) {
        if (!md) return '';
        const lines = String(md).replace(/\r\n?/g, '\n').split('\n');
        return blocks(lines);
      }

      function blocks(lines) {
        let html = '';
        let i = 0;
        let para = [];
        const flush = () => {
          if (para.length) {
            html += `<p>${para.map(inline).join('<br>')}</p>`;
            para = [];
          }
        };
        while (i < lines.length) {
          const line = lines[i];
          let m;
          if ((m = line.match(/^\s*(`{3,}|~{3,})\s*([\w+#.-]*)([^\n]*)$/))) {
            flush();
            const fence = m[1];
            const body = [];
            i++;
            let closed = false;
            while (i < lines.length) {
              if (lines[i].trim().startsWith(fence[0].repeat(fence.length)) && /^\s*[`~]+\s*$/.test(lines[i])) { closed = true; i++; break; }
              body.push(lines[i]);
              i++;
            }
            html += codeBlock(m[2], body.join('\n'), closed, m[3]);
            continue;
          }
          if (/^\s*$/.test(line)) { flush(); i++; continue; }
          if ((m = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/))) {
            flush();
            html += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`;
            i++;
            continue;
          }
          if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); html += '<hr>'; i++; continue; }
          if ((m = line.match(/^\s*(\$\$|\\\[)\s*$/))) {
            flush();
            const close = m[1] === '$$' ? /^\s*\$\$\s*$/ : /^\s*\\\]\s*$/;
            const body = [];
            i++;
            while (i < lines.length && !close.test(lines[i])) { body.push(lines[i]); i++; }
            i++;
            const tex = body.join('\n');
            const typeset = texToHtml(tex, true);
            html += typeset === null ? `<div class="math-block">${esc(tex)}</div>` : `<div class="math-block tex tex-display">${typeset}</div>`;
            continue;
          }
          if (/^\s*>/.test(line)) {
            flush();
            const quote = [];
            while (i < lines.length && /^\s*>/.test(lines[i])) { quote.push(lines[i].replace(/^\s*> ?/, '')); i++; }
            html += `<blockquote>${blocks(quote)}</blockquote>`;
            continue;
          }
          if (line.includes('|') && i + 1 < lines.length && isTableSep(lines[i + 1]) && splitRow(line).length > 1) {
            flush();
            const head = splitRow(line);
            const aligns = splitRow(lines[i + 1]).map((c) => (c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : ''));
            i += 2;
            const rows = [];
            while (i < lines.length && lines[i].includes('|') && !/^\s*$/.test(lines[i])) { rows.push(splitRow(lines[i])); i++; }
            const cell = (tag, text, k) => `<${tag}${aligns[k] ? ` style="text-align:${aligns[k]}"` : ''}>${inline(text || '')}</${tag}>`;
            html += `<div class="md-table"><table><thead><tr>${head.map((c, k) => cell('th', c, k)).join('')}</tr></thead>` +
              `<tbody>${rows.map((r) => `<tr>${head.map((_, k) => cell('td', r[k], k)).join('')}</tr>`).join('')}</tbody></table></div>`;
            continue;
          }
          if (listMarker(line)) {
            flush();
            const result = list(lines, i);
            html += result.html;
            i = result.next;
            continue;
          }
          para.push(line);
          i++;
        }
        flush();
        return html;
      }

      function list(lines, start) {
        const first = listMarker(lines[start]);
        const base = indentOf(first[1]);
        const ordered = /\d/.test(first[2]);
        const tag = ordered ? 'ol' : 'ul';
        const startNumber = ordered ? parseInt(first[2], 10) : 1;
        let html = `<${tag}${ordered && startNumber !== 1 ? ` start="${startNumber}"` : ''}>`;
        let i = start;
        while (i < lines.length) {
          const m = listMarker(lines[i]);
          if (!m || indentOf(m[1]) !== base || /\d/.test(m[2]) !== ordered) break;
          let text = m[3];
          let task = '';
          const t = text.match(/^\[([ xX])\]\s+(.*)$/);
          if (t) {
            task = `<input type="checkbox" disabled${t[1] !== ' ' ? ' checked' : ''}> `;
            text = t[2];
          }
          i++;
          const inner = [];
          while (i < lines.length) {
            const l = lines[i];
            if (/^\s*$/.test(l)) {
              // a blank line continues the item only when an indented line follows
              if (i + 1 < lines.length && indentOf(lines[i + 1]) > base && !/^\s*$/.test(lines[i + 1])) { inner.push(''); i++; continue; }
              break;
            }
            const nm = listMarker(l);
            if (indentOf(l) > base) { inner.push(l); i++; continue; }
            if (nm && indentOf(nm[1]) === base) break;
            if (!nm && indentOf(l) <= base) { inner.push(l); i++; continue; }
            break;
          }
          const inside = inner.length ? blocks(inner.map((l) => l.replace(new RegExp(`^\\s{0,${base + 4}}`), ''))) : '';
          html += `<li>${task}${inline(text)}${inside}</li>`;
        }
        return { html: `${html}</${tag}>`, next: i };
      }

      return { render, inline, highlight, esc, texToHtml, hooks };
    })();
