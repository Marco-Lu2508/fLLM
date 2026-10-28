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
      pane.querySelector('#rtTurbo').addEventListener('change', async (e) => {
        if (e.target.checked && !await confirmDialog('Turbo removes every permission question and the sandbox. Continue?')) {
          e.target.checked = false;
          return;
        }
        save({ turbo: e.target.checked });
      });

      load();
    })();
