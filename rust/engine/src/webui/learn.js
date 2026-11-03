    // ======================================================================================================
    //  learn.js: the self-improvement loop of the coding agent.
    //  After a Code session (when "Learn from this session" is on) the server notes how the skills fared, may draft a revised
    //  skill, and asks the model what is worth remembering. All of it waits for approval here:
    //    Settings > Skills           skill revisions with the reason and a diff, usage numbers, versions and rollback
    //    Settings > Personalization  the profile of the user (a plain text file you can edit), learned memory entries and
    //                                the entries and profile lines that wait for a yes
    //  The work runs in the server in the background, after the answer was delivered; nothing is applied without a yes
    //  (unless "Save memory without asking" is on, which still dedups, redacts and caps).
    // ======================================================================================================
    // @pure-begin
    const LN_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

    /** The names of the skills the model read (skill tool, action "read") in a list of chat messages. */
    function lnSkillsRead(messages) {
      const out = [];
      for (const m of messages || []) {
        for (const call of (m && m.tool_calls) || []) {
          if (!call || !call.function || call.function.name !== 'skill') continue;
          let args = call.function.arguments;
          if (typeof args === 'string') { try { args = JSON.parse(args); } catch { args = null; } }
          if (args && args.action === 'read' && LN_NAME_RE.test(String(args.name || '')) && !out.includes(args.name)) out.push(args.name);
        }
      }
      return out;
    }

    const lnText = (c) => (typeof c === 'string' ? c : Array.isArray(c) ? c.map((p) => (p && p.type !== 'image_url' && p.text) || '').join('') : '');

    /** The messages of one run reduced to what the server needs: roles, text and tool calls (no pictures), each part cut short. */
    function lnRunMessages(messages, from, maxMessages = 300, maxChars = 3000) {
      const list = (messages || []).slice(Math.max(0, from || 0)).filter((m) => m && ['user', 'assistant', 'tool'].includes(m.role));
      return list.slice(-maxMessages).map((m) => {
        const out = { role: m.role, content: lnText(m.content).slice(0, maxChars) };
        if (Array.isArray(m.tool_calls) && m.tool_calls.length) {
          out.tool_calls = m.tool_calls.map((c) => ({ id: c.id, function: { name: c.function && c.function.name, arguments: typeof (c.function && c.function.arguments) === 'string' ? c.function.arguments.slice(0, 800) : JSON.stringify((c.function && c.function.arguments) || {}).slice(0, 800) } }));
        }
        return out;
      });
    }

    /** A run counts as failed when it ended in an error or was stopped by the step limit. */
    function lnRunFailed({ error, hitLimit }) { return Boolean(error) || Boolean(hitLimit); }

    /** A unified diff (as the server writes it) as HTML with the changed lines coloured. `esc` escapes HTML. */
    function lnDiffHtml(diff, esc) {
      return String(diff || '').split('\n').filter((l, i, a) => l !== '' || i < a.length - 1).map((line) => {
        const color = line.startsWith('+ ') ? 'var(--success,#2a8a4a)' : line.startsWith('- ') ? 'var(--destructive,#c33)' : 'var(--muted-foreground)';
        return `<span style="color:${color};">${esc(line)}</span>`;
      }).join('\n');
    }

    /** "5 uses, 1 failed, thumbs 2 up / 0 down" for a skill row of GET /api/studio/learn/skills. */
    function lnStatText(s) {
      const parts = [`${s.uses || 0} use${s.uses === 1 ? '' : 's'}`];
      if (s.failed) parts.push(`${s.failed} failed`);
      if (s.up || s.down) parts.push(`thumbs ${s.up || 0} up / ${s.down || 0} down`);
      if (s.retries) parts.push(`${s.retries} retries`);
      return parts.join(', ');
    }

    /** A skill row that has something to show: used at least once, a revision waits, or it has older versions. */
    function lnSkillRelevant(s) { return Boolean(s && ((s.uses || 0) > 0 || s.pending || (s.versions && s.versions.length) || s.due)); }

    /** The sections of the profile as `## Title` blocks; used to tell whether the text has the shape the loop edits. */
    function lnProfileShape(text) {
      const sections = [...String(text || '').matchAll(/^## (.+)$/gm)].map((m) => m[1].trim());
      const lines = (String(text || '').match(/^- .+$/gm) || []).length;
      return { sections, lines };
    }

    /** A short sentence for the counters of GET /api/studio/learn. */
    function lnPendingText(o) {
      if (!o) return '';
      const parts = [];
      const n = (a, one, many) => { if (a && a.length) parts.push(`${a.length} ${a.length === 1 ? one : many}`); };
      n(o.skills, 'skill revision', 'skill revisions');
      n(o.memory, 'memory entry', 'memory entries');
      n(o.profile, 'profile line', 'profile lines');
      return parts.length ? `Waiting for a decision: ${parts.join(', ')}.` : 'Nothing waits for a decision.';
    }

    /** A line for an entry of the log of automatic changes. */
    function lnLogTitle(e) {
      const name = e.name || '';
      switch (e.kind) {
        case 'skill_create': return `Wrote the skill "${name}"`;
        case 'skill_revise': return `Revised the skill "${name}"`;
        case 'skill_rollback': return `Put back the previous text of "${name}"`;
        case 'memory_add': return `Remembered: ${e.text || ''}`;
        case 'memory_remove': return `Forgot a repeated entry: ${e.text || ''}`;
        case 'profile_add': return `Added to your profile: ${e.text || ''}`;
        default: return String(e.kind || 'change');
      }
    }

    /** The "What I learned" feed as HTML: every automatic change with time, session, reason, diff and an Undo button. */
    function lnFeedHtml(entries, esc, when) {
      if (!entries || !entries.length) return '<div class="app-item-desc">Nothing was learned automatically yet.</div>';
      return entries.map((e) => `<div class="app-card-item column" data-ln-chg="${esc(e.id)}" style="border-bottom:1px solid var(--border);padding-bottom:6px;">
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;"><span class="app-item-title">${esc(lnLogTitle(e))}</span><span style="flex:1"></span>${e.undone ? '<span class="quant-chip">undone</span>' : '<button class="btn btn-xs btn-ghost" data-ln-undo>Undo</button>'}</div>
        <div class="app-item-desc">${esc(when(e.t))}${e.session ? ` · session ${esc(e.session)}` : ''}${e.reason ? ` · ${esc(e.reason)}` : ''}</div>
        ${e.diff ? `<details><summary class="app-item-desc" style="cursor:pointer;">Changes</summary><pre class="mono" style="max-height:220px;overflow:auto;font-size:11.5px;margin:4px 0;white-space:pre-wrap;">${lnDiffHtml(e.diff, esc)}</pre></details>` : ''}
      </div>`).join('');
    }

    /** The sentence for the toast after a run: the server's summary of what was learned, or '' when nothing was. */
    function lnToastText(lastRun) { return lastRun && lastRun.summary ? `Learned: ${lastRun.summary}` : ''; }
    // @pure-end

    const lnWorkspace = () => { try { return (typeof code !== 'undefined' && code.workspace && code.workspace.root) || ''; } catch { return ''; } };
    const lnQuery = (o) => Object.entries(o).filter(([, v]) => v).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');

    async function lnRequest(method, path, body) {
      const res = await apiFetch(path, body === undefined ? { method } : { method, body });
      let data = {};
      try { data = await res.json(); } catch { /* no body */ }
      if (!res.ok) throw new Error((data.error && data.error.message) || `Request failed (${res.status})`);
      return data;
    }

    /**
     * Called by the Code page after a run was answered. The server does the rest in the background (it returns at once), so the
     * user is never kept waiting. `from` is the index of the first message of the run in `messages`.
     */
    window.fllmLearnAfterRun = async ({ messages, from, session, workspace, failed }) => {
      const run = lnRunMessages(messages, from);
      if (!run.length) return { queued: false };
      const skills = lnSkillsRead(run);
      window.fllmLearnLast = { session: session || 'studio', workspace: workspace || '', skills };
      try {
        const r = await lnRequest('POST', '/api/studio/learn/after-run', {
          session: session || 'studio', workspace: workspace || '', outcome: failed ? 'failed' : 'done', skills, messages: run,
        });
        if (r && r.queued) lnWatchRun(Date.now() / 1000 - 2);
        return r;
      } catch { return { queued: false }; }
    };

    /** A thumb for the skills of the last learned run: `value` is "up" or "down". */
    window.fllmLearnFeedback = async (value) => {
      const last = window.fllmLearnLast;
      if (!last || !last.skills.length) { toast('No skill was used in the last run.'); return false; }
      try {
        const r = await lnRequest('POST', '/api/studio/learn/feedback', { value, skills: last.skills, session: last.session, workspace: last.workspace });
        toast(value === 'up' ? `Noted: ${last.skills.join(', ')} worked.` : `Noted: ${last.skills.join(', ')} did not work. A revision is drafted on the next run${r.notes && r.notes.length ? `; ${r.notes.join(' ')}` : ''}.`);
        lnRefreshAll();
        return r.noted > 0;
      } catch (e) { toast(e.message); return false; }
    };

    window.fllmLearnStatus = async () => { try { return await lnRequest('GET', `/api/studio/learn?${lnQuery({ workspace: lnWorkspace() })}`); } catch { return null; } };

    /** The global switch of the toggle; a project's own setting is cleared so that the toggle and the setting agree. */
    window.fllmLearnSetEnabled = async (on) => {
      try { await lnRequest('POST', '/api/studio/learn/settings', { enabled: on, workspace: lnWorkspace(), project: null }); return true; } catch (e) { toast(e.message); return false; }
    };

    /** The one-sentence first-run notice (once), or ''. */
    window.fllmLearnNotice = async () => {
      const o = await window.fllmLearnStatus();
      if (!o || !o.notice_pending) return '';
      try { await lnRequest('POST', '/api/studio/learn/settings', { notice: true }); } catch { /* shown again next time */ }
      return o.notice;
    };

    /** Waits (in the background) for the learning step that follows a run and shows a toast with what it learned. */
    function lnWatchRun(since) {
      let tries = 0;
      const poll = async () => {
        tries += 1;
        const o = await window.fllmLearnStatus();
        const done = o && !o.running && o.last_run && o.last_run.t >= since;
        if (done) {
          const text = lnToastText(o.last_run);
          if (text) toast(text);
          lnRefreshAll();
          return;
        }
        if (tries < 60) window.setTimeout(poll, 5000);
      };
      window.setTimeout(poll, 4000);
    }

    // ---- Settings > Skills: revisions ---------------------------------------------------------------------
    const lnWhen = (t) => (t ? new Date(t * 1000).toLocaleString() : '');

    async function lnRefreshSkills() {
      const box = $('lnSkillList');
      if (!box) return;
      try {
        const data = await lnRequest('GET', `/api/studio/learn/skills?${lnQuery({ workspace: ($('skWorkspace') && $('skWorkspace').value.trim()) || lnWorkspace() })}`);
        const rows = (data.skills || []).filter(lnSkillRelevant);
        box.innerHTML = rows.length ? rows.map((s) => {
          const key = `data-ln-name="${escapeHtml(s.name)}" data-ln-scope="${escapeHtml(s.scope)}" data-ln-ws="${escapeHtml(s.workspace || '')}"`;
          const versions = (s.versions || []);
          return `<div class="app-card-item column" ${key} style="border-bottom:1px solid var(--border);padding-bottom:8px;">
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
              <span class="app-item-title">${escapeHtml(s.name)}</span>
              <span class="quant-chip">${escapeHtml(lnStatText(s))}</span>
              ${s.due && !s.pending ? `<span class="quant-chip" title="${escapeHtml(s.due)}">review due</span>` : ''}
              <span style="flex:1"></span>
              ${s.pending ? '' : '<button class="btn btn-xs btn-ghost" data-ln-act="improve" title="Ask the loaded model to draft a revision from how the skill was used">Improve now</button>'}
              ${versions.length ? `<select class="ui-input" data-ln-version style="width:auto;font-size:12px;">${versions.slice().reverse().map((v) => `<option value="${v.version}">v${v.version}: ${escapeHtml(v.action)}, ${escapeHtml(lnWhen(v.t))}</option>`).join('')}</select><button class="btn btn-xs btn-ghost" data-ln-act="rollback" title="Put this older text back (the current text is kept as a new version)">Roll back</button>` : ''}
            </div>
            ${s.revision ? `<div class="app-item-desc"><strong>Draft revision</strong> (${escapeHtml(s.revision.trigger || '')}): ${escapeHtml(s.revision.reason)}</div>
              <pre class="mono" style="max-height:260px;overflow:auto;font-size:11.5px;margin:4px 0;white-space:pre-wrap;">${lnDiffHtml(s.revision.diff, escapeHtml)}</pre>
              <div style="display:flex;gap:6px;"><button class="btn btn-xs btn-primary" data-ln-act="approve">Approve</button><button class="btn btn-xs btn-ghost" data-ln-act="reject">Reject</button><span class="app-item-desc">The skill stays as it is until you approve.</span></div>` : ''}
            ${(s.recent || []).length ? `<details><summary class="app-item-desc" style="cursor:pointer;">Recent uses</summary>${s.recent.map((u) => `<div class="app-item-desc">${escapeHtml(lnWhen(u.t))} · ${escapeHtml(u.outcome)}${u.feedback ? ` · thumbs ${escapeHtml(u.feedback)}` : ''}${u.verify === false ? ' · checks failed' : ''}${u.note ? ` · ${escapeHtml(u.note)}` : ''}</div>`).join('')}</details>` : ''}
          </div>`;
        }).join('') : '<div class="app-item-desc">Nothing to show yet. Switch on "Learn from this session" in the Code page: when the agent reads a skill, how it went is noted here, and after a failure or a few uses it can draft a better version.</div>';
      } catch (e) { box.textContent = e.message; }
      lnRefreshCounters();
    }

    async function lnSkillAction(row, action) {
      const name = row.dataset.lnName;
      const body = { action, name, scope: row.dataset.lnScope, workspace: row.dataset.lnWs };
      if (action === 'rollback') {
        const sel = row.querySelector('[data-ln-version]');
        body.version = sel ? Number(sel.value) : undefined;
        if (!(await askDialog({ title: `Roll back "${name}"?`, text: `The text of version ${body.version} is put back. The current text is kept as a new version, so this can be undone.`, ok: 'Roll back' }))) return;
      }
      if (action === 'approve' && !(await askDialog({ title: `Apply the revision of "${name}"?`, text: 'The current text is kept as a version; you can roll back at any time.', ok: 'Apply' }))) return;
      try {
        const r = await lnRequest('POST', '/api/studio/learn/skills', body);
        if (action === 'improve') { toast('Drafting a revision in the background. It appears here when it is ready.'); window.setTimeout(lnRefreshAll, 20000); return; }
        toast(action === 'approve' ? 'Revision applied.' : action === 'reject' ? 'Revision dropped.' : `Rolled back (the replaced text is version ${r.version}).`);
        if (typeof window.fllmSkillsReload === 'function') window.fllmSkillsReload();
      } catch (e) { toast(e.message); }
      lnRefreshAll();
    }

    // ---- Settings > Personalization: profile and learned memory ---------------------------------------------
    async function lnRefreshProfile() {
      if (!$('lnProfileText')) return;
      try {
        const p = await lnRequest('GET', '/api/studio/learn/profile');
        if (document.activeElement !== $('lnProfileText')) $('lnProfileText').value = p.text || '';
        $('lnProfileOn').checked = p.in_prompt !== false;
        $('lnProfileCount').textContent = `${($('lnProfileText').value || '').length} / ${p.max_chars}`;
        const box = $('lnProfileProposals');
        box.innerHTML = (p.proposals || []).length ? p.proposals.map((x) => `<div class="app-card-item" data-ln-prof="${escapeHtml(x.id)}"><div class="app-item-info"><div class="app-item-title">${escapeHtml(x.text)}</div><div class="app-item-desc">${escapeHtml(x.section.replace('_', ' '))}${x.session ? ` · from session ${escapeHtml(x.session)}` : ''}</div></div><button class="btn btn-xs btn-primary" data-ln-prof-act="accept">Add</button><button class="btn btn-xs btn-ghost" data-ln-prof-act="reject">Drop</button></div>`).join('') : '<div class="app-item-desc">No profile line waits for a yes.</div>';
      } catch (e) { $('lnProfileProposals').textContent = e.message; }
    }

    async function lnRefreshMemory() {
      const box = $('lnMemList');
      if (!box) return;
      try {
        const ws = lnWorkspace();
        const m = await lnRequest('GET', `/api/studio/learn/memory?${lnQuery({ workspace: ws })}`);
        const rows = (scope, list) => list.map((e) => `<div class="app-card-item" data-ln-mem-scope="${scope}" data-ln-mem-index="${e.index}"><div class="app-item-info"><div class="app-item-title">${escapeHtml(e.text)}</div><div class="app-item-desc">${scope === 'global' ? 'every project' : 'this project'}${e.source ? ` · learned in session ${escapeHtml(e.source.session || '?')}${e.source.created ? `, ${escapeHtml(lnWhen(e.source.created))}` : ''}` : ' · written by hand or by the memory tool'}</div></div><button class="btn btn-xs btn-ghost" data-ln-mem-act="edit">Edit</button><button class="btn btn-xs btn-ghost" data-ln-mem-act="delete">Delete</button></div>`).join('');
        box.innerHTML = (rows('global', m.global || []) + rows('workspace', m.workspace || [])) || '<div class="app-item-desc">The agent memory is empty.</div>';
        const pr = $('lnMemProposals');
        pr.innerHTML = (m.proposals || []).length ? m.proposals.map((p) => `<div class="app-card-item" data-ln-memp="${escapeHtml(p.id)}"><div class="app-item-info"><div class="app-item-title">${p.action === 'remove' ? 'Forget: ' : ''}${escapeHtml(p.text)}</div><div class="app-item-desc">${escapeHtml(p.scope === 'global' ? 'every project' : 'this project')} · ${escapeHtml(p.kind)}${p.session ? ` · from session ${escapeHtml(p.session)}` : ''}${p.reason ? ` · ${escapeHtml(p.reason)}` : ''}</div></div><button class="btn btn-xs btn-primary" data-ln-memp-act="accept">${p.action === 'remove' ? 'Forget it' : 'Remember'}</button><button class="btn btn-xs btn-ghost" data-ln-memp-act="reject">Drop</button></div>`).join('') : '<div class="app-item-desc">No memory entry waits for a yes.</div>';
      } catch (e) { box.textContent = e.message; }
    }

    async function lnRefreshCounters() {
      const o = await window.fllmLearnStatus();
      const text = lnPendingText(o);
      for (const id of ['lnPendingSkills', 'lnPendingPersonal']) if ($(id)) $(id).textContent = `${text}${o && o.running ? ' A learning step is running now.' : ''}`;
      const last = o && o.last_run && o.last_run.lines;
      for (const id of ['lnLastSkills', 'lnLastPersonal']) if ($(id)) $(id).textContent = last && last.length ? `Last learning step (${last.length}): ${last.slice(0, 4).join(' ')}` : '';
    }

    async function lnRefreshFeed() {
      if (!$('lnFeedPersonal')) return;
      try {
        const data = await lnRequest('GET', '/api/studio/learn/log?limit=40');
        const all = data.log || [];
        $('lnFeedPersonal').innerHTML = lnFeedHtml(all, escapeHtml, lnWhen);
        if ($('lnFeedSkills')) $('lnFeedSkills').innerHTML = lnFeedHtml(all.filter((e) => String(e.kind).startsWith('skill')), escapeHtml, lnWhen);
      } catch (e) { $('lnFeedPersonal').textContent = e.message; }
      const o = await window.fllmLearnStatus();
      if (o) {
        if ($('lnEnabled')) $('lnEnabled').checked = o.enabled !== false;
        if ($('lnAskFirst')) $('lnAskFirst').checked = Boolean(o.ask_first);
        if ($('lnProjectNote')) $('lnProjectNote').textContent = o.project_override === false ? 'Learning is off for the open project (its .fllm/learn.json).' : o.project_override === true ? 'Learning is on for the open project (its .fllm/learn.json).' : '';
      }
    }

    function lnRefreshAll() { lnRefreshSkills(); lnRefreshProfile(); lnRefreshMemory(); lnRefreshFeed(); }

    async function lnPost(path, body, done) {
      try { const r = await lnRequest('POST', path, body); if (done) toast(done); return r; } catch (e) { toast(e.message); return null; } finally { lnRefreshAll(); }
    }

    async function lnEditMemory(row) {
      const scope = row.dataset.lnMemScope;
      const index = Number(row.dataset.lnMemIndex);
      const current = row.querySelector('.app-item-title').textContent;
      const text = await askDialog({ title: 'Edit this memory entry', text: 'An empty text deletes the entry.', input: current, ok: 'Save' });
      if (text === null) return;
      await lnPost('/api/studio/learn/memory', { action: text.trim() ? 'edit' : 'delete', scope, index, text, workspace: lnWorkspace() }, 'Saved.');
    }

    // ---- wiring ----------------------------------------------------------------------------------------------
    if ($('lnSkillList')) {
      $('settingsMenuNav').addEventListener('click', (e) => { if (e.target.closest('[data-spane="skills"],[data-spane="personalization"]')) lnRefreshAll(); });
      $('lnSkillList').addEventListener('click', (e) => {
        const btn = e.target.closest('[data-ln-act]');
        const row = e.target.closest('[data-ln-name]');
        if (btn && row) lnSkillAction(row, btn.dataset.lnAct);
      });
      $('lnConsolidate').addEventListener('click', async () => {
        const r = await lnPost('/api/studio/learn/consolidate', { workspace: lnWorkspace() });
        if (r) toast('Consolidating in the background: revisions that are due are drafted and repeated memory entries are proposed for removal.');
      });
      $('lnProfileSave').addEventListener('click', () => lnPost('/api/studio/learn/profile', { action: 'save', text: $('lnProfileText').value }, 'Profile saved.'));
      $('lnProfileText').addEventListener('input', () => { $('lnProfileCount').textContent = `${$('lnProfileText').value.length} / 3000`; });
      $('lnProfileOn').addEventListener('change', (e) => lnPost('/api/studio/learn/profile', { action: 'prompt', on: e.target.checked }));
      $('lnProfileExport').addEventListener('click', () => {
        const blob = new Blob([$('lnProfileText').value], { type: 'text/markdown' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'user-profile.md';
        a.click();
        window.setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      });
      $('lnProfileDelete').addEventListener('click', async () => {
        if (await askDialog({ title: 'Delete your profile?', text: 'The profile file and the profile lines that wait for a yes are deleted. Memory entries and skills stay.', ok: 'Delete', danger: true })) lnPost('/api/studio/learn/profile', { action: 'delete' }, 'Profile deleted.');
      });
      $('lnProfileProposals').addEventListener('click', (e) => {
        const btn = e.target.closest('[data-ln-prof-act]');
        const row = e.target.closest('[data-ln-prof]');
        if (btn && row) lnPost('/api/studio/learn/profile', { action: btn.dataset.lnProfAct, id: row.dataset.lnProf });
      });
      $('lnMemProposals').addEventListener('click', (e) => {
        const btn = e.target.closest('[data-ln-memp-act]');
        const row = e.target.closest('[data-ln-memp]');
        if (btn && row) lnPost('/api/studio/learn/memory', { action: btn.dataset.lnMempAct, id: row.dataset.lnMemp, workspace: lnWorkspace() });
      });
      $('lnMemList').addEventListener('click', async (e) => {
        const btn = e.target.closest('[data-ln-mem-act]');
        const row = e.target.closest('[data-ln-mem-scope]');
        if (!btn || !row) return;
        if (btn.dataset.lnMemAct === 'edit') { lnEditMemory(row); return; }
        if (await askDialog({ title: 'Delete this memory entry?', text: row.querySelector('.app-item-title').textContent, ok: 'Delete', danger: true })) {
          lnPost('/api/studio/learn/memory', { action: 'delete', scope: row.dataset.lnMemScope, index: Number(row.dataset.lnMemIndex), workspace: lnWorkspace() }, 'Deleted.');
        }
      });
      $('lnForgetAll').addEventListener('click', async () => {
        if (await askDialog({ title: 'Forget everything that was learned?', text: 'Removes every memory entry the learning loop wrote and the memory entries that wait for a yes. Entries you wrote yourself stay.', ok: 'Forget', danger: true })) {
          lnPost('/api/studio/learn/memory', { action: 'forget', all_learned: true, text: '', workspace: lnWorkspace() }, 'Forgotten.');
        }
      });
      $('lnEnabled').addEventListener('change', async (e) => {
        if (await window.fllmLearnSetEnabled(e.target.checked)) {
          toast(e.target.checked ? 'Learning is on.' : 'Learning is off. What was learned stays; nothing new is saved.');
          if (typeof window.fllmLearnChanged === 'function') window.fllmLearnChanged(e.target.checked);
        }
        lnRefreshFeed();
      });
      $('lnAskFirst').addEventListener('change', async (e) => {
        try { await lnRequest('POST', '/api/studio/learn/settings', { ask_first: e.target.checked }); toast(e.target.checked ? 'The agent now proposes and waits for your yes.' : 'The agent applies what it learns and logs it.'); } catch (err) { toast(err.message); }
      });
      const onFeed = async (e) => {
        const btn = e.target.closest('[data-ln-undo]');
        const row = e.target.closest('[data-ln-chg]');
        if (!btn || !row) return;
        const r = await lnPost('/api/studio/learn/undo', { id: row.dataset.lnChg });
        if (r && r.message) toast(`Undone: ${r.message}`);
        if (typeof window.fllmSkillsReload === 'function') window.fllmSkillsReload();
      };
      $('lnFeedPersonal').addEventListener('click', onFeed);
      $('lnFeedSkills').addEventListener('click', onFeed);
      lnRefreshFeed();
      lnRefreshCounters();
    }
