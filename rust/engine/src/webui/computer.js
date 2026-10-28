    // ==========================================================================
    // Computer use in the Code page: the agent looks at the screen and operates the mouse and keyboard of THIS computer,
    // only after the user switched it on in Settings > Runtime and with a question before every action.
    //  - the banner (always visible while it is on) with the big STOP button, and the panic keys (Esc three times, Ctrl+Alt+Shift+Esc);
    //  - the cards: every action shows the last screenshot with a marker where the click, drag or scroll goes;
    //  - the screenshots that go back to the model (the last two stay in the conversation, older ones are dropped, none is saved).
    // The server enforces the switch, the limits and the stop as well (agent_computer.rs); this file is the view.
    // ==========================================================================

    // @pure-begin  (no DOM access in this block: it is loaded by tests/computer.test.js)
    const cuNum = (v) => (Number.isFinite(Number(v)) ? Math.round(Number(v) * 10) / 10 : 0);
    const cuPt = (p) => (p && typeof p === 'object' ? `${cuNum(p.x ?? p[0])},${cuNum(p.y ?? p[1])}` : '?');

    /** One line that names an action; the permission rules (`computer: computer_click*`) match against it. */
    function computerSubject(name, args) {
      const a = args || {};
      switch (name) {
        case 'computer_click': return `${name} ${cuNum(a.x)},${cuNum(a.y)}${a.button && a.button !== 'left' ? ` ${a.button}` : ''}${a.double ? ' double' : ''}`;
        case 'computer_move': return `${name} ${cuNum(a.x)},${cuNum(a.y)}`;
        case 'computer_drag': return `${name} ${cuPt(a.from)} -> ${cuPt(a.to)}`;
        case 'computer_type': return `${name} ${JSON.stringify(String(a.text ?? ''))}`;
        case 'computer_key': return `${name} ${String(a.keys ?? '')}`;
        case 'computer_scroll': return `${name} ${cuNum(a.x)},${cuNum(a.y)} dx=${cuNum(a.dx)} dy=${cuNum(a.dy)}`;
        case 'computer_wait': return `${name} ${cuNum(a.ms ?? 1000)}`;
        default: return String(name);
      }
    }

    /** The question on the permission card. */
    function computerQuestionText(name, args) {
      const a = args || {};
      switch (name) {
        case 'computer_screenshot': return 'Let the agent look at your screen?';
        case 'computer_click': return `${a.double ? 'Double-click' : `${a.button && a.button !== 'left' ? `${a.button}-click` : 'Click'}`} at the marked point?`;
        case 'computer_move': return 'Move the mouse pointer to the marked point?';
        case 'computer_drag': return 'Drag with the mouse along the marked line?';
        case 'computer_type': return `Type ${JSON.stringify(String(a.text ?? '').slice(0, 80))}${String(a.text ?? '').length > 80 ? ' ...' : ''} on the keyboard?`;
        case 'computer_key': return `Press ${String(a.keys ?? '')}?`;
        case 'computer_scroll': return 'Scroll at the marked point?';
        default: return 'Allow this?';
      }
    }

    /** Marks to draw on the screenshot an action refers to, in pixels of that screenshot. */
    function computerMarks(name, args) {
      const a = args || {};
      const x = Number(a.x);
      const y = Number(a.y);
      const ok = Number.isFinite(x) && Number.isFinite(y);
      if ((name === 'computer_click' || name === 'computer_move') && ok) return [{ kind: 'cross', x, y, double: Boolean(a.double) }];
      if (name === 'computer_scroll' && ok) return [{ kind: 'cross', x, y }, { kind: 'arrow', x, y, dx: Number(a.dx) || 0, dy: Number(a.dy) || 0 }];
      if (name === 'computer_drag') {
        const f = a.from || {};
        const t = a.to || {};
        const fx = Number(f.x ?? f[0]);
        const fy = Number(f.y ?? f[1]);
        const tx = Number(t.x ?? t[0]);
        const ty = Number(t.y ?? t[1]);
        if ([fx, fy, tx, ty].every(Number.isFinite)) return [{ kind: 'line', x1: fx, y1: fy, x2: tx, y2: ty }, { kind: 'cross', x: fx, y: fy }, { kind: 'cross', x: tx, y: ty }];
      }
      return [];
    }

    /** SVG (same coordinates as the screenshot) with a crosshair for every mark: white underneath, red on top, so it shows on any background. */
    function marksSvg(width, height, marks) {
      if (!marks.length || !(width > 0) || !(height > 0)) return '';
      const s = Math.min(40, Math.max(14, Math.max(width, height) / 50));
      const w = Math.max(2, Math.round(Math.max(width, height) / 450));
      const f = (n) => Math.round(n * 10) / 10;
      const layer = (stroke, extra) => marks.map((m) => {
        if (m.kind === 'cross') {
          const r = s * 0.55;
          const ring = m.double ? `<circle cx="${f(m.x)}" cy="${f(m.y)}" r="${f(r * 1.7)}"/>` : '';
          return `<g stroke="${stroke}" stroke-width="${w + extra}" fill="none" stroke-linecap="round"><circle cx="${f(m.x)}" cy="${f(m.y)}" r="${f(r)}"/>${ring}<path d="M${f(m.x - s * 1.1)} ${f(m.y)}H${f(m.x - r * 0.6)}M${f(m.x + r * 0.6)} ${f(m.y)}H${f(m.x + s * 1.1)}M${f(m.x)} ${f(m.y - s * 1.1)}V${f(m.y - r * 0.6)}M${f(m.x)} ${f(m.y + r * 0.6)}V${f(m.y + s * 1.1)}"/></g>`;
        }
        if (m.kind === 'line') return `<path d="M${f(m.x1)} ${f(m.y1)}L${f(m.x2)} ${f(m.y2)}" stroke="${stroke}" stroke-width="${w + extra}" fill="none" stroke-linecap="round" stroke-dasharray="${f(s * 0.5)} ${f(s * 0.35)}"/>`;
        if (m.kind === 'arrow') {
          const len = Math.hypot(m.dx, m.dy) || 1;
          const ex = m.x + (m.dx / len) * s * 3;
          const ey = m.y + (m.dy / len) * s * 3;
          return `<path d="M${f(m.x)} ${f(m.y)}L${f(ex)} ${f(ey)}" stroke="${stroke}" stroke-width="${w + extra}" fill="none" stroke-linecap="round"/>`;
        }
        return '';
      }).join('');
      return `<svg class="cu-mark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">${layer('#fff', 3)}${layer('#e5231b', 0)}</svg>`;
    }

    /** Takes the pictures out of all but the last `keep` messages that carry some; returns how many were dropped. */
    function dropOldImages(messages, keep = 2) {
      const carrying = messages.filter((m) => Array.isArray(m.images) && m.images.length);
      let dropped = 0;
      for (const m of carrying.slice(0, Math.max(0, carrying.length - keep))) {
        dropped += m.images.length;
        delete m.images;
        m.content = `${m.content}\n[The picture was removed from the conversation to save space. Take a new screenshot if you need to see the screen.]`;
      }
      return dropped;
    }

    /** A copy of the messages without pictures (for saving a session, and for summaries). */
    function withoutImages(messages) {
      return messages.map((m) => {
        if (!Array.isArray(m.images) || !m.images.length) return m;
        const { images, ...rest } = m;
        return { ...rest, content: `${m.content}\n[${images.length === 1 ? 'a screenshot was' : `${images.length} screenshots were`} shown here and are not kept]` };
      });
    }

    /** The messages in the OpenAI form for the server: a message with `images` becomes text plus image parts. */
    function toWireMessages(messages) {
      return messages.map((m) => {
        if (!Array.isArray(m.images) || !m.images.length) return m;
        const { images, ...rest } = m;
        return { ...rest, content: [{ type: 'text', text: m.content }, ...images.map((url) => ({ type: 'image_url', image_url: { url } }))] };
      });
    }

    /** Escape pressed three times within `windowMs`: `times` is the list of earlier presses, returns the new list and whether to stop. */
    function escapeStreak(times, now, windowMs = 1500, needed = 3) {
      const recent = [...times.filter((t) => now - t <= windowMs), now];
      return recent.length >= needed ? { times: [], fire: true } : { times: recent, fire: false };
    }

    /** Ctrl+Alt+Shift+Esc, the same chord the server watches for. */
    const isPanicChord = (e) => Boolean(e && e.key === 'Escape' && e.ctrlKey && e.altKey && e.shiftKey);
    // @pure-end

    const computer = { status: null, timer: 0, esc: [], busy: '', warnedVision: false, stopping: false };
    const computerOn = () => Boolean(computer.status && computer.status.enabled && computer.status.available);
    const computerQuestion = (item) => computerQuestionText(item.name, item.args);

    function fireComputerEvent() {
      window.dispatchEvent(new CustomEvent('fllm:computer', { detail: computer.status }));
    }

    async function refreshComputer() {
      const was = computerOn();
      try {
        const res = await apiFetch('/api/studio/agent/computer');
        computer.status = res.ok ? await res.json() : null;
      } catch {
        computer.status = null;
      }
      const now = computerOn();
      if (was && !now && !computer.stopping) {
        // switched off behind the page's back: the panic chord on the keyboard, or the setting in another window
        if (code.running) stopAgent();
        codeNotice('Computer use was stopped and switched off. Switch it on again in Settings > Runtime if you want to continue.');
      }
      if (was !== now) {
        if (code.registry && code.registry.has('computer_screenshot') !== now) loadRegistry();
        fireComputerEvent();
      }
      renderComputerBanner();
      scheduleComputerPoll();
    }

    function scheduleComputerPoll() {
      clearTimeout(computer.timer);
      computer.timer = setTimeout(refreshComputer, computerOn() ? 1500 : 15000);
    }

    /** The stop button, the panic keys and the stop route: cancel the running action, switch the permission off, end the run. */
    async function stopComputer(why) {
      computer.stopping = true;
      try {
        await apiFetch('/api/studio/agent/computer/stop', { method: 'POST', body: {} });
      } catch { /* the server may be gone; the page still stops */ }
      if (code.running) stopAgent();
      code.lastShot = null;
      computer.busy = '';
      await refreshComputer();
      computer.stopping = false;
      codeNotice(`${why || 'Stopped.'} Computer use is switched off; the agent cannot see or touch the screen any more. Switch it on again in Settings > Runtime.`);
      toast('Computer use stopped and switched off');
    }

    function renderComputerBanner() {
      let el = document.getElementById('cuBanner');
      const show = computerOn();
      if (!show) { if (el) el.remove(); document.body.dataset.computer = '0'; return; }
      document.body.dataset.computer = '1';
      if (!el) {
        el = document.createElement('div');
        el.id = 'cuBanner';
        el.className = 'cu-banner';
        el.setAttribute('role', 'alert');
        el.addEventListener('click', (e) => { if (e.target.closest('[data-cu-stop]')) stopComputer('You pressed STOP.'); });
        const main = document.querySelector('.code-main');
        if (main) main.insertBefore(el, main.firstChild);
        else document.body.appendChild(el);
      }
      const s = computer.status || {};
      const turbo = Boolean(code.turbo);
      const line = computer.busy
        ? `<b>${escapeHtml(computer.busy)}</b>`
        : `<b>Computer use is ON.</b> The agent can see your screen and use your mouse and keyboard${turbo ? ' <b>WITHOUT ASKING (turbo)</b>' : ', and asks you before every action'}.`;
      el.innerHTML = `<span class="cu-dot" aria-hidden="true"></span><span class="cu-text">${line}<span class="cu-keys"> Emergency stop: press Esc three times, or Ctrl+Alt+Shift+Esc anywhere. ${s.steps_used ? `${s.steps_used} of ${s.max_steps} actions used in this task.` : ''}</span></span><button class="btn cu-stop" data-cu-stop title="Stop the agent now and switch computer use off">STOP</button>`;
      el.classList.toggle('is-turbo', turbo);
    }

    document.addEventListener('keydown', (e) => {
      if (!computerOn()) return;
      if (isPanicChord(e)) { e.preventDefault(); stopComputer('You pressed Ctrl+Alt+Shift+Esc.'); return; }
      if (e.key !== 'Escape' || e.repeat) return;
      const streak = escapeStreak(computer.esc, Date.now());
      computer.esc = streak.times;
      if (streak.fire) stopComputer('You pressed Esc three times.');
    }, true);

    // ---- cards -------------------------------------------------------------------------------------------------

    /** Before the question: remember which screenshot the action refers to. Returns an error text when the action cannot work yet. */
    function computerPrepare(item) {
      if (COMPUTER_LOOK.has(item.name)) return '';
      if (!code.lastShot) return 'Take a screenshot first (computer_screenshot): coordinates refer to the last screenshot you took.';
      item.shot = code.lastShot;
      return '';
    }

    function computerDone(item, r, ctx) {
      computer.busy = '';
      if (!(r && r.ok && r.image && r.image.data)) { renderComputerBanner(); return; }
      const url = `data:${r.image.mime || 'image/png'};base64,${r.image.data}`;
      const shot = { url, width: r.image.width, height: r.image.height };
      item.image = shot;
      code.lastShot = shot;
      (ctx.shots || (ctx.shots = [])).push({ url, width: shot.width, height: shot.height, text: r.output });
      if (!computer.warnedVision && state.status?.model?.capabilities?.vision === false && !(typeof window.fllmRemoteSelected === 'function' && window.fllmRemoteSelected())) {
        computer.warnedVision = true;
        codeNotice('The loaded model cannot look at pictures, so it will not see the screenshots. Choose a vision model (the model list marks them with "vision"), or use a provider model that can see images.');
      }
      renderComputerBanner();
    }

    /** After the tool replies of one step: the screenshots go to the model as one message with pictures. */
    function computerAttachImages(ctx) {
      if (!ctx.shots || !ctx.shots.length) return;
      const shots = ctx.shots.splice(0);
      ctx.messages.push({ role: 'user', content: shots.map((s) => `[Screenshot from the computer tools: ${s.width}x${s.height} px. ${String(s.text || '').slice(0, 300)}]`).join('\n'), images: shots.map((s) => s.url) });
      dropOldImages(ctx.messages, 2);
    }

    function computerBodyHtml(item) {
      const shot = item.image || item.shot;
      if (!shot || !shot.url) return item.gone ? '<div class="cu-gone">(the screenshot is not kept after the session was saved)</div>' : '';
      const marks = item.image ? '' : marksSvg(shot.width, shot.height, computerMarks(item.name, item.args));
      return `<div class="cu-shot"><img src="${shot.url}" alt="Screenshot" width="${shot.width}" height="${shot.height}">${marks}</div>`;
    }

    /** What an action is doing right now, for the banner. */
    function computerActivity(item, waiting) {
      const [verb, target] = toolLabel(item);
      computer.busy = waiting ? `Waiting for your answer: ${verb} ${target}` : `Operating your computer: ${verb} ${target}`;
      renderComputerBanner();
    }

    /** At the start of a task: the step count and the remembered screenshot start again, and the tool list follows the setting. */
    async function computerBegin() {
      code.lastShot = null;
      computer.busy = '';
      await refreshComputer();
      if (code.registry && code.registry.has('computer_screenshot') !== computerOn()) await loadRegistry();
      if (computerOn()) {
        try { await apiFetch('/api/studio/agent/computer/begin', { method: 'POST', body: {} }); } catch { /* the server counts anyway */ }
      }
    }

    setTimeout(refreshComputer, 600);
