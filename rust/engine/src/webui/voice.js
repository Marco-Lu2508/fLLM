    // ==========================================================================
    // Everyday features: dictation (microphone), reading aloud, notifications when a long answer is done,
    // the welcome tour, and the hook for the quick-entry hotkey of the desktop app.
    // The keyboard shortcuts live in boot.js. Nothing here sends audio or text anywhere: the browser does the speech work.
    // ==========================================================================
    (function voiceAndAlerts() {
      // @pure-begin
      /** Turns Markdown into plain text that sounds right when read aloud: no code, no marks, no web addresses. */
      function stripForSpeech(markdown) {
        let s = String(markdown || '');
        s = s.replace(/<think>[\s\S]*?(<\/think>|$)/gi, ' ');
        let fence = '';
        s = s.split('\n').filter((line) => { // fenced code blocks (an unfinished one runs to the end)
          const m = /^[ \t]*(```|~~~)/.exec(line);
          if (fence) { if (m && m[1] === fence) fence = ''; return false; }
          if (m) { fence = m[1]; return false; }
          return true;
        }).join('\n');
        s = s.replace(/<[^>\n]+>/g, ' ');
        s = s.replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1');
        s = s.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');
        s = s.replace(/\bhttps?:\/\/[^\s)]+/g, ' ');
        s = s.replace(/`+([^`\n]*)`+/g, '$1');
        s = s.replace(/^[ \t]{0,3}#{1,6}[ \t]+/gm, '');
        s = s.replace(/^[ \t]{0,3}>[ \t]?/gm, '');
        s = s.replace(/^[ \t]*([-*+]|\d+[.)])[ \t]+/gm, '');
        s = s.replace(/^[ \t]*([-*_][ \t]*){3,}$/gm, '');
        s = s.replace(/^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$/gm, ''); // table separator rows
        s = s.replace(/\|/g, ' ');
        s = s.replace(/(\*\*|__)(.+?)\1/g, '$2');
        s = s.replace(/(^|[\s(])[*_]([^*_\n]+?)[*_](?=[\s).,!?;:]|$)/g, '$1$2');
        s = s.replace(/~~/g, '').replace(/\*+/g, '');
        return s.split('\n').map((l) => l.replace(/[ \t]+/g, ' ').trim()).filter(Boolean).join('\n');
      }
      /** Cuts plain text into pieces of at most `max` characters at sentence ends (long sentences at commas or spaces). */
      function chunkForSpeech(text, max) {
        const limit = max || 220;
        const out = [];
        let cur = '';
        const flush = () => { if (cur.trim()) out.push(cur.trim()); cur = ''; };
        const sentences = [];
        for (const line of String(text || '').split('\n')) {
          const parts = line.match(/[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$/g) || [];
          for (const p of parts) sentences.push(p);
          sentences.push('\n');
        }
        for (let s of sentences) {
          if (s === '\n') { flush(); continue; }
          while (s.length > limit) {
            let cut = s.lastIndexOf(',', limit);
            if (cut < limit / 3) cut = s.lastIndexOf(' ', limit);
            if (cut < 1) cut = limit;
            if (cur) flush();
            out.push(s.slice(0, cut + 1).trim());
            s = s.slice(cut + 1);
          }
          if (cur.length + s.length > limit) flush();
          cur += s;
        }
        flush();
        return out.filter(Boolean);
      }
      // @pure-end

      const KEY = 'fllm.studio.voice.v1';
      const TOUR_KEY = 'fllm.studio.tour.v1';
      const settings = Object.assign({ lang: '', voice: '', rate: 1, pitch: 1, auto: false, notify: true, sound: false, asked: false }, (() => {
        try { return JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch { return {}; }
      })());
      const save = () => localStorage.setItem(KEY, JSON.stringify(settings));
      const LANGUAGES = [['', 'Browser language'], ['en-US', 'English (US)'], ['en-GB', 'English (UK)'], ['de-DE', 'German'], ['fr-FR', 'French'], ['es-ES', 'Spanish'],
        ['it-IT', 'Italian'], ['pt-BR', 'Portuguese (Brazil)'], ['nl-NL', 'Dutch'], ['pl-PL', 'Polish'], ['ru-RU', 'Russian'], ['tr-TR', 'Turkish'],
        ['ja-JP', 'Japanese'], ['ko-KR', 'Korean'], ['zh-CN', 'Chinese (Mandarin)'], ['hi-IN', 'Hindi'], ['ar-SA', 'Arabic']];
      const speechLang = () => settings.lang || navigator.language || 'en-US';

      // ---- read aloud ----
      const synth = window.speechSynthesis;
      let speakGeneration = 0;
      let pending = 0;
      const updateSpeakButton = () => { $('btnStopSpeaking').style.display = pending > 0 ? '' : 'none'; };
      window.fllmStopSpeaking = () => {
        speakGeneration++;
        pending = 0;
        if (synth) synth.cancel();
        updateSpeakButton();
      };
      window.fllmSpeak = (text) => {
        if (!synth || typeof SpeechSynthesisUtterance === 'undefined') { toast('This browser cannot read aloud.'); return; }
        window.fllmStopSpeaking();
        const parts = chunkForSpeech(stripForSpeech(text));
        if (!parts.length) return;
        const generation = ++speakGeneration;
        const chosen = synth.getVoices().find((v) => v.voiceURI === settings.voice);
        const done = () => { if (generation === speakGeneration) { pending = Math.max(0, pending - 1); updateSpeakButton(); } };
        for (const part of parts) {
          const u = new SpeechSynthesisUtterance(part);
          u.lang = chosen ? chosen.lang : speechLang();
          if (chosen) u.voice = chosen;
          u.rate = Number(settings.rate) || 1;
          u.pitch = Number(settings.pitch) || 1;
          u.onend = done;
          u.onerror = done;
          pending++;
          synth.speak(u);
        }
        updateSpeakButton();
      };
      $('btnStopSpeaking').addEventListener('click', window.fllmStopSpeaking);
      window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && pending > 0) window.fllmStopSpeaking(); });

      // ---- dictation ----
      const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      let recognition = null;
      const micButton = $('btnMic');
      function stopDictation(hard) {
        if (!recognition) return;
        const r = recognition;
        recognition = null;
        micButton.classList.remove('recording');
        try { if (hard) r.abort(); else r.stop(); } catch { /* already stopped */ }
      }
      function startDictation() {
        const field = $('chatTextarea');
        const base = field.value;
        const r = new Recognition();
        r.lang = speechLang();
        r.interimResults = true;
        r.continuous = true;
        r.onresult = (ev) => {
          let text = '';
          for (let i = 0; i < ev.results.length; i++) text += ev.results[i][0].transcript;
          field.value = base + (base && text && !/\s$/.test(base) ? ' ' : '') + text.trimStart();
          field.dispatchEvent(new Event('input', { bubbles: true }));
        };
        r.onerror = (ev) => {
          const messages = {
            'not-allowed': 'The microphone is blocked. Allow it for this page in the browser or system settings.',
            'service-not-allowed': 'Speech recognition is not allowed here. Check the privacy settings of the browser or system.',
            'audio-capture': 'No microphone was found.',
            network: 'The speech service of the browser could not be reached. Most browsers need an internet connection for it.',
            'language-not-supported': 'This browser cannot recognize the chosen language.',
          };
          if (ev.error !== 'no-speech' && ev.error !== 'aborted') toast(messages[ev.error] || `Dictation stopped (${ev.error}).`);
        };
        r.onend = () => { if (recognition === r) { recognition = null; micButton.classList.remove('recording'); } };
        recognition = r;
        micButton.classList.add('recording');
        field.focus();
        try { r.start(); } catch (e) { recognition = null; micButton.classList.remove('recording'); toast(`Dictation could not start: ${e.message}`); }
      }
      if (Recognition) {
        micButton.style.display = '';
        micButton.title = 'Dictate with the microphone (the speech service of your browser does the recognition)';
        micButton.addEventListener('click', () => { if (recognition) stopDictation(false); else startDictation(); });
        // sending ends the dictation, or a late result would bring the sent text back
        $('chatTextarea').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) stopDictation(true); }, true);
        $('btnSendChat').addEventListener('click', () => stopDictation(true), true);
      }

      // ---- notifications and sound ----
      function chime() {
        try {
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          [[880, 0], [1175, 0.14]].forEach(([hz, at]) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.value = hz;
            gain.gain.setValueAtTime(0.0001, ctx.currentTime + at);
            gain.gain.exponentialRampToValueAtTime(0.12, ctx.currentTime + at + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + at + 0.2);
            osc.connect(gain).connect(ctx.destination);
            osc.start(ctx.currentTime + at);
            osc.stop(ctx.currentTime + at + 0.22);
          });
          setTimeout(() => ctx.close(), 800);
        } catch { /* no audio available */ }
      }
      const canNotify = typeof Notification !== 'undefined';
      function askPermission() {
        if (!canNotify || Notification.permission !== 'default') return;
        settings.asked = true;
        save();
        try { Promise.resolve(Notification.requestPermission()).catch(() => {}); } catch { /* not allowed here */ }
      }
      function announce(body) {
        if (document.hasFocus() && !document.hidden) return;
        if (settings.sound) chime();
        if (settings.notify && canNotify && Notification.permission === 'granted') {
          try {
            const n = new Notification('fLLM', { body, tag: 'fllm-done' });
            n.onclick = () => { window.focus(); n.close(); };
          } catch { /* the system refused the notification */ }
        }
      }
      /** Calls `done(ms)` each time the stop state of a send button ends; the page marks a running answer with the class "stop". */
      function watchRun(buttonId, done) {
        const button = $(buttonId);
        let started = 0;
        new MutationObserver(() => {
          const busy = button.classList.contains('stop');
          if (busy && !started) {
            started = Date.now();
            if (settings.notify && !settings.asked) askPermission(); // the click that sent the message still counts as a gesture
          } else if (!busy && started) {
            const ms = Date.now() - started;
            started = 0;
            done(ms);
          }
        }).observe(button, { attributes: true, attributeFilter: ['class'] });
      }
      watchRun('btnSendChat', (ms) => {
        const t = currentThread();
        const last = t && pathOf(t).filter((n) => n.role === 'assistant').pop();
        const finished = last && last.finish && !last.error && last.content;
        if (finished && settings.auto) window.fllmSpeak(last.content);
        if (finished && ms >= 20000) announce('Your answer is ready.');
      });
      watchRun('btnCodeSend', (ms) => { if (ms >= 5000) announce('The agent has finished.'); });

      // ---- settings page: Voice and alerts ----
      const nav = $('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      const row = (title, desc, control) =>
        `<div class="app-card-item"><div class="app-item-info"><div class="app-item-title">${title}</div>` +
        `<div class="app-item-desc">${desc}</div></div><div class="app-item-actions">${control}</div></div>`;
      const sw = (id) => `<label class="ui-switch"><input type="checkbox" id="${id}"><span class="ui-switch-track"></span></label>`;
      if (nav && content) {
        const button = document.createElement('button');
        button.className = 'settings-nav-btn';
        button.dataset.spane = 'voice';
        button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><path d="M11 5 6 9H2v6h4l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M19 5a10 10 0 0 1 0 14"/></svg><span>Voice and alerts</span>';
        const after = nav.querySelector('[data-spane="appearance"]');
        nav.insertBefore(button, after ? after.nextSibling : null);
        const pane = document.createElement('div');
        pane.className = 'settings-pane';
        pane.id = 'spane-voice';
        const dictation = Recognition
          ? 'Dictation uses the speech recognition of your browser. Depending on the browser, the audio may be sent to the speech service of its maker; fLLM itself never receives or stores any audio.'
          : 'Dictation is not available here: this browser, or the web view of the desktop app on this system, has no speech recognition, so the microphone button is hidden. Chrome, Edge and Safari have it.';
        pane.innerHTML =
          '<div class="app-card"><div class="app-card-title">Dictation</div>' +
          row('Microphone', dictation, '') +
          row('Language', 'Used for dictation, and for reading aloud when the voice does not set one.', `<select id="selVoiceLang" class="ui-select">${LANGUAGES.map(([v, n]) => `<option value="${v}">${n}</option>`).join('')}</select>`) +
          '</div>' +
          '<div class="app-card"><div class="app-card-title">Reading aloud</div>' +
          row('Voice', synth ? 'The voices come from your system or browser.' : 'This browser cannot read aloud.', '<select id="selVoiceName" class="ui-select"><option value="">Default</option></select>') +
          row('Speed', '<span id="valVoiceRate"></span>', '<input type="range" id="rngVoiceRate" min="0.5" max="2" step="0.1">') +
          row('Pitch', '<span id="valVoicePitch"></span>', '<input type="range" id="rngVoicePitch" min="0.5" max="2" step="0.1">') +
          row('Read answers automatically', 'Reads each finished answer aloud. The speaker button under an answer reads one answer.', sw('chkVoiceAuto')) +
          row('Try it', 'Code and Markdown marks are left out when reading.', '<button class="btn btn-sm btn-secondary" id="btnVoiceTest">Read a sample</button>') +
          '</div>' +
          '<div class="app-card"><div class="app-card-title">Alerts</div>' +
          row('System notification', 'When a long answer (over 20 seconds) or an agent run finishes while this window is not in front.', sw('chkNotify')) +
          row('Sound', 'A short tone at the same moment.', sw('chkNotifySound') + ' <button class="btn btn-sm btn-ghost" id="btnChimeTest">Play</button>') +
          '<div class="app-card-item"><div class="app-item-desc" id="notifyStatus"></div></div></div>';
        content.appendChild(pane);
        const $$ = (id) => pane.querySelector(`#${id}`);
        const fillVoices = () => {
          if (!synth) return;
          const list = synth.getVoices().slice().sort((a, b) => a.lang.localeCompare(b.lang) || a.name.localeCompare(b.name));
          $$('selVoiceName').innerHTML = '<option value="">Default</option>' + list.map((v) => `<option value="${escapeHtml(v.voiceURI)}">${escapeHtml(v.name)} (${escapeHtml(v.lang)})</option>`).join('');
          $$('selVoiceName').value = settings.voice;
        };
        const notifyNote = () => {
          $$('notifyStatus').textContent = !canNotify ? 'This browser has no system notifications.'
            : Notification.permission === 'denied' ? 'Notifications are blocked for this page. Allow them in the browser or system settings.' : '';
        };
        const show = () => {
          $$('selVoiceLang').value = settings.lang;
          $$('rngVoiceRate').value = settings.rate;
          $$('rngVoicePitch').value = settings.pitch;
          $$('valVoiceRate').textContent = `${Number(settings.rate).toFixed(1)}x`;
          $$('valVoicePitch').textContent = Number(settings.pitch).toFixed(1);
          $$('chkVoiceAuto').checked = settings.auto;
          $$('chkNotify').checked = settings.notify;
          $$('chkNotifySound').checked = settings.sound;
          fillVoices();
          notifyNote();
        };
        if (synth) synth.addEventListener('voiceschanged', fillVoices);
        $$('selVoiceLang').addEventListener('change', (e) => { settings.lang = e.target.value; save(); });
        $$('selVoiceName').addEventListener('change', (e) => { settings.voice = e.target.value; save(); });
        $$('rngVoiceRate').addEventListener('input', (e) => { settings.rate = Number(e.target.value); $$('valVoiceRate').textContent = `${settings.rate.toFixed(1)}x`; save(); });
        $$('rngVoicePitch').addEventListener('input', (e) => { settings.pitch = Number(e.target.value); $$('valVoicePitch').textContent = settings.pitch.toFixed(1); save(); });
        $$('chkVoiceAuto').addEventListener('change', (e) => { settings.auto = e.target.checked; save(); });
        $$('chkNotify').addEventListener('change', (e) => {
          settings.notify = e.target.checked;
          save();
          if (settings.notify) askPermission();
          setTimeout(notifyNote, 600);
        });
        $$('chkNotifySound').addEventListener('change', (e) => { settings.sound = e.target.checked; save(); });
        $$('btnChimeTest').addEventListener('click', chime);
        $$('btnVoiceTest').addEventListener('click', () => window.fllmSpeak('This is how **fLLM** sounds. Code blocks are skipped, and so are `marks` and [links](https://example.com).'));
        button.addEventListener('click', show);
        show();
      }

      // ---- the welcome tour ----
      const tour = { step: 0, timer: 0 };
      const STEPS = ['Welcome to fLLM', 'Your computer', 'Pick a model', 'Three tips'];
      function tourBody() {
        if (tour.step === 0) {
          return '<p>fLLM runs language models on your own computer, on the processor or on any graphics card. Your chats, files and models stay on this machine.</p>' +
            '<p>It is a chat, a coding agent and a model manager in one program, and it speaks the same API as the big services, so other tools can use it too.</p>';
        }
        if (tour.step === 1) {
          if (state.hardware) {
            const usable = Number(state.modelsHub?.available_memory_bytes || 0);
            return `<p class="tour-box">${escapeHtml(describeComputer(state.hardware, state.modelsHub))}</p>` +
              `<p>${usable ? 'This is the memory a model can use. Bigger models are smarter but need more memory and are slower; the Models page tells you which ones fit.' : 'The Models page tells you which models fit.'}</p>`;
          }
          if (state.hardwareError) return `<p class="tour-box">${escapeHtml(describeComputer(null, null))}</p><p>You can still pick a model on the Models page.</p>`;
          return '<p>Looking at your computer ...</p>';
        }
        if (tour.step === 2) {
          const loaded = state.status?.model?.name;
          const byId = new Map((state.modelsHub?.catalog || []).map((e) => [e.id, e]));
          const names = (state.modelsHub?.recommended || []).map((id) => byId.get(id)?.display_name).filter(Boolean).slice(0, 4);
          return (loaded ? `<p>A model is already loaded: <strong>${escapeHtml(loaded)}</strong>. You can start chatting, or look for another one.</p>`
            : '<p>You need a model before the first answer. The Models page suggests ones that run well on this computer and downloads them with one click.</p>') +
            (names.length ? `<p class="tour-box">Suggested: ${names.map(escapeHtml).join(', ')}</p>` : '') +
            '<p><button class="btn btn-secondary" id="tourOpenModels">Open Models</button></p>';
        }
        return '<ul class="tour-tips">' +
          `<li><strong>Shortcuts.</strong> Press ${escapeHtml(shortcutLabel('shortcuts'))} or ? to see all keys and change them. ${escapeHtml(shortcutLabel('search'))} searches, ${escapeHtml(shortcutLabel('newChat'))} starts a new chat.</li>` +
          '<li><strong>Code mode.</strong> Switch to Code in the sidebar: an agent that reads and edits the files of a folder you choose and runs commands, asking you first.</li>' +
          '<li><strong>Documents.</strong> Attach a PDF, Word, Excel or text file with the plus button, or drop it into the chat, and ask about it.</li></ul>';
      }
      function renderTour() {
        clearTimeout(tour.timer);
        $('tourTitle').textContent = STEPS[tour.step];
        $('tourBody').innerHTML = tourBody();
        $('tourDots').innerHTML = STEPS.map((_, i) => `<span class="${i === tour.step ? 'on' : ''}"></span>`).join('');
        $('tourBack').style.visibility = tour.step ? 'visible' : 'hidden';
        $('tourNext').textContent = tour.step === STEPS.length - 1 ? 'Get started' : 'Next';
        $('tourSkip').style.visibility = tour.step === STEPS.length - 1 ? 'hidden' : 'visible';
        const open = $('tourOpenModels');
        if (open) open.addEventListener('click', () => { closeTour(); switchSurface('hub'); });
        if (tour.step === 1 && !state.hardware && !state.hardwareError) tour.timer = setTimeout(renderTour, 700);
      }
      function closeTour() { clearTimeout(tour.timer); $('tourModal').classList.remove('open'); }
      function openTour() {
        localStorage.setItem(TOUR_KEY, '1');
        tour.step = 0;
        if (!state.hardware && !state.hardwareError) refreshHub(); // the Models page code fetches the hardware; nothing is fetched twice
        renderTour();
        $('tourModal').classList.add('open');
      }
      $('tourNext').addEventListener('click', () => { if (tour.step >= STEPS.length - 1) closeTour(); else { tour.step++; renderTour(); } });
      $('tourBack').addEventListener('click', () => { if (tour.step) { tour.step--; renderTour(); } });
      $('tourSkip').addEventListener('click', closeTour);
      $('tourModal').addEventListener('click', (e) => { if (e.target === $('tourModal')) closeTour(); });
      $('btnShowTour').addEventListener('click', () => { switchSurface('chat'); openTour(); });
      $('btnShowShortcuts').addEventListener('click', openShortcuts);
      // first start only: wait until the hardware is known (the Models code fetches it), then show the tour once
      setTimeout(() => {
        if (localStorage.getItem(TOUR_KEY)) return;
        if (state.threads.length) { localStorage.setItem(TOUR_KEY, '1'); return; } // an existing user does not need it
        if (!state.hardware && !state.hardwareError) refreshHub();
        let tries = 0;
        const wait = setInterval(() => {
          if (state.hardware || state.hardwareError || ++tries > 30) { clearInterval(wait); if (!localStorage.getItem(TOUR_KEY)) openTour(); }
        }, 500);
      }, 1500);

      // ---- quick entry (desktop app): the global hotkey asks the page for a new chat ----
      window.fllmQuickEntry = () => {
        window.fllmStopSpeaking();
        ['searchModal', 'helpModal', 'tourModal'].forEach((id) => $(id).classList.remove('open'));
        startNewChat();
        setTimeout(() => $('chatTextarea').focus(), 60);
      };
    })();
