// ==========================================================================
    // Skills that come with the app, and importing one from a URL.
    //  - The library ships inside the binary; installing writes a normal skill file the user can edit.
    //  - Import takes any http(s) address that serves a SKILL.md (for example a raw GitHub link).
    // ==========================================================================

    (function skillLibrary() {
      const pane = document.getElementById('spane-skills');
      if (!pane) return;

      const card = document.createElement('div');
      card.className = 'app-card';
      card.style.marginTop = '12px';
      card.innerHTML =
        '<div class="app-card-title">Skills that come with the app</div>' +
        '<div class="app-item-desc" style="margin-bottom:8px;">One click writes them into your skills folder — as a normal file you can change afterwards.</div>' +
        '<div id="skLibList"><div class="app-item-desc">Loading ...</div></div>' +
        '<div style="display:flex;gap:6px;align-items:center;margin-top:12px;flex-wrap:wrap;">' +
        '  <input class="ui-input mono" id="skImportUrl" placeholder="https://raw.githubusercontent.com/you/repo/main/SKILL.md" style="flex:1;min-width:260px;" autocomplete="off">' +
        '  <button class="btn btn-sm" id="btnSkillImport">Import from URL</button>' +
        '</div>' +
        '<div class="app-item-desc mono" id="skLibStatus" style="margin-top:8px;font-size:11px;white-space:pre-wrap;"></div>';
      pane.appendChild(card);

      // Skills from the open directory (skills.sh) and any other GitHub repository: type owner/repo, pick, install.
      const finder = document.createElement('div');
      finder.className = 'app-card';
      finder.style.marginTop = '12px';
      finder.innerHTML =
        '<div class="app-card-title">Find skills on GitHub</div>' +
        '<div class="app-item-desc" style="margin-bottom:8px;">Skills are folders with a SKILL.md in a GitHub repository, for example <span class="mono">anthropics/skills</span> or <span class="mono">vercel-labs/skills</span>. ' +
        'Type the repository (or paste the command of a skills directory) to see what it offers. A skill is a text file that tells the assistant how to do something: read it before you install it. ' +
        '<a href="#" id="skFindWeb">Browse the directory at skills.sh</a></div>' +
        '<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">' +
        '  <input class="ui-input mono" id="skRepo" placeholder="owner/repo" style="flex:1;min-width:220px;" autocomplete="off">' +
        '  <button class="btn btn-sm" id="skRepoGo">Show skills</button>' +
        '</div>' +
        '<div id="skRepoList" style="margin-top:8px;"></div>' +
        '<div class="app-item-desc mono" id="skRepoStatus" style="margin-top:8px;font-size:11px;white-space:pre-wrap;"></div>';
      pane.appendChild(finder);
      const fsay = (text) => { finder.querySelector('#skRepoStatus').textContent = text || ''; };
      const flist = finder.querySelector('#skRepoList');

      finder.querySelector('#skFindWeb').addEventListener('click', (e) => {
        e.preventDefault();
        apiFetch('/api/studio/open-url', { method: 'POST', body: { url: 'https://www.skills.sh/' } });
      });

      async function showRepo() {
        const repo = finder.querySelector('#skRepo').value.trim();
        if (!repo) { fsay('Type a repository like owner/repo.'); return; }
        fsay('Looking ...');
        flist.innerHTML = '';
        try {
          const res = await apiFetch('/api/studio/skills/github?repo=' + encodeURIComponent(repo));
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { fsay((data.error && data.error.message) || 'Could not read the repository.'); return; }
          const skills = data.skills || [];
          fsay(skills.length ? `${skills.length} skill${skills.length === 1 ? '' : 's'} in ${data.repo}${data.truncated ? ' (the list is cut off)' : ''}.` : `${data.repo} has no SKILL.md files.`);
          flist.innerHTML = skills.map((k) =>
            '<div style="display:flex;gap:8px;align-items:flex-start;border-bottom:1px solid var(--border);padding:6px 0;">' +
            '  <div style="flex:1;"><div class="app-item-title" style="font-size:12px;">' + escapeHtml(k.name) + '</div>' +
            '    <div class="app-item-desc">' + escapeHtml(k.description || k.path) + '</div></div>' +
            '  <button class="btn btn-xs ' + (k.installed ? 'btn-ghost' : 'btn-secondary') + '" data-gh-install="' + escapeHtml(k.url) + '" data-gh-name="' + escapeHtml(k.name) + '"' + (k.installed ? ' disabled' : '') + '>' + (k.installed ? 'installed' : 'Install') + '</button>' +
            '</div>').join('');
        } catch (err) {
          fsay('Could not read the repository: ' + ((err && err.message) || err));
        }
      }
      finder.querySelector('#skRepoGo').addEventListener('click', showRepo);
      finder.querySelector('#skRepo').addEventListener('keydown', (e) => { if (e.key === 'Enter') showRepo(); });
      flist.addEventListener('click', async (e) => {
        const btn = e.target.closest('[data-gh-install]');
        if (!btn || btn.disabled) return;
        fsay(`Installing ${btn.dataset.ghName} ...`);
        try {
          const res = await apiFetch('/api/studio/skills/import', { method: 'POST', body: { url: btn.dataset.ghInstall, name: btn.dataset.ghName } });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { fsay((data.error && data.error.message) || 'Could not install it.'); return; }
          btn.textContent = 'installed';
          btn.disabled = true;
          fsay(`Installed ${data.name}. It is in your skills folder now and can be edited there. Extra files of the skill (scripts, references) are not copied.`);
          await load();
          if (typeof window.fllmSkillsReload === 'function') window.fllmSkillsReload();
        } catch (err) {
          fsay('Could not install it: ' + ((err && err.message) || err));
        }
      });

      const list = card.querySelector('#skLibList');
      const say = (text) => { card.querySelector('#skLibStatus').textContent = text || ''; };

      async function load() {
        try {
          const res = await apiFetch('/api/studio/skills/library');
          const data = await res.json();
          const skills = data.skills || [];
          list.innerHTML = skills.map((s) =>
            '<div style="display:flex;gap:8px;align-items:flex-start;border-bottom:1px solid var(--border);padding:6px 0;" data-lib="' + escapeHtml(s.name) + '">' +
            '  <div style="flex:1;">' +
            '    <div class="app-item-title" style="font-size:12px;">' + escapeHtml(s.name) + '</div>' +
            '    <div class="app-item-desc">' + escapeHtml(s.description) + '</div>' +
            '    <div class="app-item-desc" style="font-size:11px;opacity:.75;">when: ' + escapeHtml(s.when) + '</div>' +
            '  </div>' +
            '  <button class="btn btn-xs ' + (s.installed ? 'btn-ghost' : 'btn-secondary') + '" data-lib-install="' + escapeHtml(s.name) + '"' + (s.installed ? ' disabled' : '') + '>' +
            (s.installed ? 'installed' : 'Install') + '</button>' +
            '</div>').join('');
          if (!skills.length) list.innerHTML = '<div class="app-item-desc">The library is empty.</div>';
        } catch (e) {
          list.innerHTML = '<div class="app-item-desc">Could not read the library: ' + escapeHtml((e && e.message) || String(e)) + '</div>';
        }
      }

      list.addEventListener('click', async (e) => {
        const btn = e.target.closest('[data-lib-install]');
        if (!btn || btn.disabled) return;
        const name = btn.dataset.libInstall;
        say(`Installing ${name} ...`);
        try {
          const res = await apiFetch('/api/studio/skills/library', { method: 'POST', body: { name } });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { say((data.error && data.error.message) || 'Could not install it.'); return; }
          say(`Installed ${name}. It is in your skills folder now and can be edited there.`);
          await load();
          if (typeof window.fllmSkillsReload === 'function') window.fllmSkillsReload();
        } catch (err) {
          say('Could not install it: ' + ((err && err.message) || err));
        }
      });

      card.querySelector('#btnSkillImport').addEventListener('click', async () => {
        const url = card.querySelector('#skImportUrl').value.trim();
        if (!url) { say('Give the address of a SKILL.md.'); return; }
        say('Fetching ...');
        try {
          const res = await apiFetch('/api/studio/skills/import', { method: 'POST', body: { url } });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) { say((data.error && data.error.message) || 'Could not import it.'); return; }
          card.querySelector('#skImportUrl').value = '';
          say(`Imported ${data.name} (${data.bytes} bytes).`);
          await load();
          if (typeof window.fllmSkillsReload === 'function') window.fllmSkillsReload();
        } catch (err) {
          say('Could not import it: ' + ((err && err.message) || err));
        }
      });

      load();
    })();
