// ==========================================================================
    // Settings > Runtime: how freely the app works — the sandbox for agent commands and the turbo switch.
    //  - sandbox: off, workspace (root read-only, workspace writable) or strict (also no network)
    //  - turbo: no permission questions at all and no sandbox; the switch is red because it is meant to be loud
    //  - the server says whether sandboxing is even possible here, so nothing is promised that the machine cannot do
    // ==========================================================================

    (function runtimeSettings() {
      const nav = document.getElementById('settingsMenuNav');
      const content = document.querySelector('.settings-content-scroll');
      if (!nav || !content) return;

      const button = document.createElement('button');
      button.className = 'settings-nav-btn';
      button.dataset.spane = 'runtime';
      button.innerHTML = '<svg viewBox="0 0 24 24" class="icon"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/></svg><span>Runtime</span>';
      nav.appendChild(button);

      const pane = document.createElement('div');
      pane.className = 'settings-pane';
      pane.id = 'spane-runtime';
      pane.innerHTML =
        '<div class="app-card"><div class="app-card-title">Sandbox for agent commands</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;">Approvals decide whether a command may run; the sandbox decides what it can reach once it runs. <span class="mono">workspace</span>: the system is read-only, the workspace writable, private /tmp, own PID namespace. <span class="mono">strict</span>: also no network. <span class="mono">off</span>: a plain shell. How it is done depends on the system: bubblewrap on Linux, a restricted low-integrity token in a job object on Windows (<span class="mono">strict</span> is unavailable there), Seatbelt on macOS. Windows and macOS are implemented but not yet verified on real machines, so they stay <span class="mono">off</span> until you choose a mode. When the mechanism does not work, the command runs unwrapped and the status below says so.</div>' +
        '<div id="rtSandboxRow" style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;">' +
        '  <label style="display:flex;gap:6px;align-items:center;"><input type="radio" name="rtSandbox" value="off"> off</label>' +
        '  <label style="display:flex;gap:6px;align-items:center;"><input type="radio" name="rtSandbox" value="workspace"> workspace</label>' +
        '  <label style="display:flex;gap:6px;align-items:center;"><input type="radio" name="rtSandbox" value="strict"> strict</label>' +
        '</div>' +
        '<div class="app-item-desc mono" id="rtSandboxState" style="margin-top:8px;font-size:11px;"></div>' +
        '<div class="app-item-desc" id="rtSandboxDetails" style="margin-top:6px;font-size:11px;"></div>' +
        '</div>' +
        '<div class="app-card" id="rtComputerCard" style="margin-top:12px;border:1px solid var(--destructive,#d33);">' +
        '<div class="app-card-title" style="color:var(--destructive,#d33);">Computer use</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;"><b>Warning.</b> When this is on, the coding agent can look at your whole screen (everything that is open on it, including private messages, other windows and passwords you can see) and can move the mouse, click and type on this computer as if you did it. It can do anything you can do in the programs that are open. A mistake of the model can send a message, delete a file or pay for something. Every action asks you first, a red banner shows while it is on, and you can stop it at any time with the STOP button, by pressing <span class="mono">Esc</span> three times, or with <span class="mono">Ctrl+Alt+Shift+Esc</span> anywhere. It is meant for programs that have no command line and no API; for web pages use the browser tool. It only works with a model that can see pictures and find things on them, and most small local models are bad at that. It is off until you switch it on here, and it can only be used from this computer.</div>' +
        '<label style="display:flex;gap:8px;align-items:center;font-size:13px;"><input type="checkbox" id="rtComputer"> Let the agent see the screen and use mouse and keyboard</label>' +
        '<div id="rtComputerLimits" style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-top:10px;font-size:12px;">' +
        '  <label style="display:flex;gap:6px;align-items:center;">At most <input type="number" class="ui-input" id="rtComputerRate" min="1" max="600" style="width:76px;"> actions per minute</label>' +
        '  <label style="display:flex;gap:6px;align-items:center;">At most <input type="number" class="ui-input" id="rtComputerSteps" min="1" max="5000" style="width:76px;"> actions per task</label>' +
        '</div>' +
        '<div class="app-item-desc mono" id="rtComputerState" style="margin-top:8px;font-size:11px;white-space:pre-wrap;"></div>' +
        '</div>' +
        '<div class="app-card" style="margin-top:12px;border:1px solid var(--destructive,#d33);">' +
        '<div class="app-card-title" style="color:var(--destructive,#d33);">Turbo</div>' +
        '<div class="app-item-desc" style="margin-bottom:10px;">No permission questions at all: every tool call runs, the sandbox is off, and the agent does not stop to ask. Meant for a machine you trust and work you can throw away. Deny rules are ignored too, because that is what “no restrictions” means — turn it off again when the job is done.</div>' +
        '<label style="display:flex;gap:8px;align-items:center;font-size:13px;"><input type="checkbox" id="rtTurbo"> Enable turbo</label>' +
        '<div class="app-item-desc mono" id="rtStatus" style="margin-top:8px;font-size:11px;"></div>' +
        '</div>';
      content.appendChild(pane);

      const say = (text) => { pane.querySelector('#rtStatus').textContent = text || ''; };
      const sandboxState = () => pane.querySelector('#rtSandboxState');

      function show(data) {
        for (const radio of pane.querySelectorAll('input[name=rtSandbox]')) radio.checked = radio.value === data.sandbox;
        pane.querySelector('#rtTurbo').checked = Boolean(data.turbo);
        for (const radio of pane.querySelectorAll('input[name=rtSandbox]')) radio.disabled = Boolean(data.turbo) || Boolean(data.sandbox_from_env);
        const parts = [`sandbox: ${data.sandbox}`];
        if (data.sandbox_from_env) parts.push('set by FLLM_SANDBOX for this server and wins over this page');
        const st = data.sandbox_status || {};
        const mechanism = st.mechanism || 'the sandbox';
        if (st.active) parts.push(`${mechanism} is active`);
        else if (data.sandbox === 'off') parts.push('sandbox is off');
        else parts.push(`${mechanism} is not active — ${(st.sandbox && st.sandbox.reason) || 'it does not work here'}${st.platform === 'windows' && data.sandbox === 'strict' ? ' (commands are refused in this mode)' : ' (commands run unwrapped)'}`);
        if (data.turbo) parts.push('turbo is on: no questions, sandbox off');
        sandboxState().textContent = parts.join(' · ');
        // what is and is not enforced here, and how well that is verified
        const details = pane.querySelector('#rtSandboxDetails');
        if (details) {
          const lines = [];
          for (const item of st.enforced || []) lines.push('Enforced: ' + item);
          for (const item of st.not_enforced || []) lines.push('Not enforced: ' + item);
          if (st.verification) lines.push('Verification: ' + st.verification);
          details.textContent = lines.join('\n');
          details.style.whiteSpace = 'pre-wrap';
        }
        document.body.dataset.turbo = data.turbo ? '1' : '0';
        showComputer(data.computer);
      }

      /** Computer use: the switch, the two limits and what this machine can do. */
      function showComputer(c) {
        if (!c) return;
        pane.querySelector('#rtComputer').checked = Boolean(c.enabled);
        pane.querySelector('#rtComputer').disabled = Boolean(c.forbidden) || !c.compiled;
        const rate = pane.querySelector('#rtComputerRate');
        const steps = pane.querySelector('#rtComputerSteps');
        if (document.activeElement !== rate) rate.value = c.rate_per_minute;
        if (document.activeElement !== steps) steps.value = c.max_steps;
        const lines = [];
        if (!c.compiled) lines.push('This build has no computer use (it was built without the computer-use feature).');
        else if (c.forbidden) lines.push('Computer use is forbidden on this server (FLLM_NO_COMPUTER_USE).');
        else if (c.available) lines.push(`Possible on this computer (${c.platform}).${c.enabled ? ' It is ON.' : ''}`);
        else lines.push(`Not possible right now: ${c.reason || 'unknown reason'}`);
        pane.querySelector('#rtComputerState').textContent = lines.join('\n');
      }

      async function load() {
        try {
          const res = await apiFetch('/api/studio/runtime');
          show(await res.json());
        } catch (e) {
          say('Could not read the setting: ' + ((e && e.message) || e));
        }
      }

      async function save(body) {
        try {
          const res = await apiFetch('/api/studio/runtime', { method: 'POST', body });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { say((data.error && data.error.message) || 'Could not save it.'); return; }
          show(data);
          if (typeof window.fllmRuntimeChanged === 'function') window.fllmRuntimeChanged(data);
          say('Saved.');
        } catch (e) {
          say('Could not save it: ' + ((e && e.message) || e));
        }
      }

      pane.querySelectorAll('input[name=rtSandbox]').forEach((radio) => radio.addEventListener('change', () => save({ sandbox: radio.value })));
      pane.querySelector('#rtComputer').addEventListener('change', async (e) => {
        if (e.target.checked && !await confirmDialog('Let the agent see your whole screen and use your mouse and keyboard? It can click and type in any open program as if you did it. Every action asks you first, and Esc three times (or Ctrl+Alt+Shift+Esc) stops it at once.')) {
          e.target.checked = false;
          return;
        }
        save({ computer_use: e.target.checked });
      });
      for (const id of ['rtComputerRate', 'rtComputerSteps']) {
        pane.querySelector('#' + id).addEventListener('change', (e) => {
          const n = Math.round(Number(e.target.value));
          if (n >= 1) save(id === 'rtComputerRate' ? { computer_rate: n } : { computer_max_steps: n });
        });
      }
      // the STOP button, Esc three times or the panic chord switch it off elsewhere: the switch here follows
      window.addEventListener('fllm:computer', (e) => showComputer(e.detail));
      pane.querySelector('#rtTurbo').addEventListener('change', async (e) => {
        if (e.target.checked && !await confirmDialog('Turbo removes every permission question and the sandbox. Continue?')) {
          e.target.checked = false;
          return;
        }
        save({ turbo: e.target.checked });
      });

      load();
    })();
