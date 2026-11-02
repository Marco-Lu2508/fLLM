    // ==========================================================================
    // Choose a model: a file (.gguf, the first part of a split file, .safetensors), a Hugging Face folder, or a Hugging Face name.
    // The server looks at the choice first (POST /api/studio/models/inspect) and says what it found, or why it is not a model.
    // ==========================================================================
    // @pure-begin
    /** What a person reads about the multi-token prediction heads of a model (the speed-up that needs no second model). */
    function mtpSummary(facts) {
      if (!facts) return '';
      if (facts.speculation_blocked) return facts.speculation_blocked;
      if (facts.mtp === true) return 'Built-in multi-token prediction heads were found. They can make answers faster (Load options, Speculative decoding), on the processor or a CUDA graphics card.';
      if (facts.mtp === false) return 'This model has no multi-token prediction heads (MTP), so that option cannot speed it up. Prompt lookup and a draft model work without them. Heads are part of the model files: a model converted without them cannot get them back.';
      return 'Whether it has multi-token prediction heads (MTP) is known once the files are on this computer.';
    }

    /** The facts of an inspection as label and value pairs for the dialog (empty values are left out). */
    function factRows(facts) {
      if (!facts || !facts.ok) return [];
      const size = Number(facts.size_bytes || 0);
      const gb = (n) => (n >= 1073741824 ? `${(n / 1073741824).toFixed(1)} GB` : `${Math.max(1, Math.round(n / 1048576))} MB`);
      const fit = facts.fits === true ? 'Fits into the free memory' : facts.fits === false ? 'May not fit into the free memory' : '';
      return [
        ['Format', facts.format],
        ['Size', size ? `${gb(size)}${facts.files > 1 ? ` in ${facts.files} files` : ''}` : ''],
        ['Architecture', facts.architecture],
        ['Quantization', facts.quantization],
        ['Context length', facts.context_length ? `${Number(facts.context_length).toLocaleString('en-US')} tokens` : ''],
        ['Memory', fit],
      ].filter((row) => row[1]);
    }

    /** A path that was typed with quotes or around spaces. */
    function cleanPathText(text) {
      return String(text || '').trim().replace(/^["']+|["']+$/g, '').trim();
    }
    // @pure-end

    const mp = { facts: null, busy: false, browsing: false, dir: '' };

    function mpEl(id) { return document.getElementById(id); }

    function mpSetButtons() {
      const ok = Boolean(mp.facts && mp.facts.ok) && !mp.busy;
      mpEl('mpAdd').disabled = !ok || mp.facts.kind === 'repo';
      mpEl('mpLoad').disabled = !ok;
      mpEl('mpLoad').textContent = mp.facts && mp.facts.kind === 'repo' ? 'Download and load' : 'Load';
    }

    function mpRender() {
      const box = mpEl('mpResult');
      const facts = mp.facts;
      if (mp.busy) { box.innerHTML = '<p class="app-item-desc">Looking at it ...</p>'; mpSetButtons(); return; }
      if (!facts) { box.innerHTML = ''; mpSetButtons(); return; }
      let html = '';
      if (facts.problem) {
        html += `<div class="mp-problem" role="alert"><b>${escapeHtml(facts.problem)}</b>${facts.fix ? `<span>${escapeHtml(facts.fix)}</span>` : ''}</div>`;
      }
      if (facts.ok) {
        html += `<div class="mp-found"><div class="mp-title">${escapeHtml(facts.title || '')}</div><dl class="mp-facts">${factRows(facts).map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd class="mono">${escapeHtml(v)}</dd>`).join('')}</dl>`;
        html += `<p class="app-item-desc mp-mtp">${escapeHtml(mtpSummary(facts))}</p>`;
        for (const note of facts.notes || []) html += `<p class="app-item-desc">${escapeHtml(note)}</p>`;
        html += '</div>';
      }
      if ((facts.candidates || []).length) {
        html += `<div class="mp-candidates" role="list">${facts.candidates.map((c) => `<button type="button" class="mp-candidate mono" role="listitem" data-mp-path="${escapeHtml(c.path)}">${escapeHtml(c.label)}</button>`).join('')}</div>`;
      }
      box.innerHTML = html;
      mpSetButtons();
    }

    async function mpCheck(text) {
      const value = cleanPathText(text === undefined ? mpEl('mpPath').value : text);
      mpEl('mpPath').value = value;
      if (!value) { mp.facts = null; mpRender(); return; }
      mp.busy = true;
      mpRender();
      try {
        const res = await apiFetch('/api/studio/models/inspect', { method: 'POST', body: { path: value } });
        const data = await res.json().catch(() => ({}));
        mp.facts = res.ok ? data : { ok: false, kind: 'not_a_model', problem: data.error?.message || 'The model could not be checked.', fix: 'Try again, or restart fLLM if this keeps happening.' };
      } catch (e) {
        mp.facts = { ok: false, kind: 'not_a_model', problem: `The model could not be checked: ${e.message}`, fix: 'The server may be restarting. Try again in a moment.' };
      }
      mp.busy = false;
      mpRender();
    }

    // ---- the list of folders of this computer (for the browser version, where the system dialog is on the other screen) ----
    async function mpList(dir) {
      const list = mpEl('mpList');
      list.innerHTML = '<p class="app-item-desc">Reading the folder ...</p>';
      try {
        const res = await apiFetch(`/api/studio/fs/browse?path=${encodeURIComponent(dir || '')}`);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) { list.innerHTML = `<p class="app-item-desc">${escapeHtml(data.error?.message || 'The folder could not be read.')}</p>`; return; }
        mp.dir = data.path;
        const roots = (data.roots || []).map((r) => `<button type="button" class="btn btn-xs btn-ghost" data-mp-dir="${escapeHtml(r.path)}">${escapeHtml(r.label)}</button>`).join('');
        const up = data.parent ? `<button type="button" class="mp-entry" data-mp-dir="${escapeHtml(data.parent)}"><span aria-hidden="true">..</span><span>Up one folder</span></button>` : '';
        const rows = (data.entries || []).map((e) => e.kind === 'folder'
          ? `<button type="button" class="mp-entry" data-mp-dir="${escapeHtml(e.path)}"><span aria-hidden="true">/</span><span class="mp-name">${escapeHtml(e.name)}</span>${e.model_folder ? '<span class="quant-chip">model folder</span>' : ''}</button>`
          : `<button type="button" class="mp-entry" data-mp-pick="${escapeHtml(e.path)}"><span aria-hidden="true">&middot;</span><span class="mp-name">${escapeHtml(e.name)}</span><span class="app-item-desc">${escapeHtml(e.format)}, ${escapeHtml(formatBytes(e.size_bytes))}</span></button>`).join('');
        list.innerHTML = `<div class="mp-roots">${roots}</div><div class="mp-here mono" title="${escapeHtml(data.path)}">${escapeHtml(data.path)}</div>`
          + `<div class="mp-entries">${up}${rows || '<p class="app-item-desc">No folders or model files here.</p>'}${data.truncated ? '<p class="app-item-desc">Only the first entries are shown.</p>' : ''}</div>`
          + '<button type="button" class="btn btn-sm btn-secondary" id="mpUseFolder">Use this folder</button>';
      } catch (e) {
        list.innerHTML = `<p class="app-item-desc">The folder could not be read: ${escapeHtml(e.message)}</p>`;
      }
    }

    function mpToggleBrowse(force) {
      mp.browsing = force === undefined ? !mp.browsing : force;
      mpEl('mpList').hidden = !mp.browsing;
      mpEl('mpBrowse').setAttribute('aria-expanded', String(mp.browsing));
      if (mp.browsing) {
        const typed = cleanPathText(mpEl('mpPath').value);
        mpList(/^[A-Za-z]:|^[\\/]/.test(typed) ? typed.replace(/[\\/][^\\/]*$/, '') || typed : '');
      }
    }

    function closeModelPicker() { mpEl('modelPickerModal').classList.remove('open'); }

    /** Opens the dialog; `initial` fills the path box and checks it. */
    window.fllmChooseModel = (initial) => {
      mp.facts = null;
      mp.busy = false;
      mpEl('mpPath').value = initial || '';
      mpEl('mpList').hidden = true;
      mp.browsing = false;
      mpRender();
      mpEl('modelPickerModal').classList.add('open');
      if (initial) mpCheck(initial);
    };
    // the start screen calls this one ("Choose a file ...")
    window.fllmPickModelFile = () => window.fllmChooseModel('');

    mpEl('mpCheck').addEventListener('click', () => mpCheck());
    mpEl('mpPath').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); mpCheck(); } });
    mpEl('mpPickFile').addEventListener('click', async () => { const p = await browse('model'); if (p) mpCheck(p); });
    mpEl('mpPickFolder').addEventListener('click', async () => { const p = await browse('folder'); if (p) mpCheck(p); });
    mpEl('mpBrowse').addEventListener('click', () => mpToggleBrowse());
    mpEl('mpCancel').addEventListener('click', closeModelPicker);
    mpEl('modelPickerModal').addEventListener('mousedown', (e) => { if (e.target === mpEl('modelPickerModal')) closeModelPicker(); });
    mpEl('mpList').addEventListener('click', (e) => {
      const dir = e.target.closest('[data-mp-dir]');
      if (dir) { mpList(dir.dataset.mpDir); return; }
      const file = e.target.closest('[data-mp-pick]');
      if (file) { mpCheck(file.dataset.mpPick); return; }
      if (e.target.closest('#mpUseFolder')) mpCheck(mp.dir);
    });
    mpEl('mpResult').addEventListener('click', (e) => {
      const pick = e.target.closest('[data-mp-path]');
      if (pick) mpCheck(pick.dataset.mpPath);
    });

    async function mpAddCurrent(quiet) {
      const res = await apiFetch('/api/studio/models/local', { method: 'POST', body: { path: mp.facts.path } });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) { if (!quiet) toast(d.error?.message || 'The model could not be added.'); return null; }
      if (!quiet) toast('Model added');
      refreshHub();
      return d.added && d.added[0];
    }
    mpEl('mpAdd').addEventListener('click', async () => {
      if (!mp.facts || !mp.facts.ok) return;
      if (await mpAddCurrent(false)) closeModelPicker();
    });
    mpEl('mpLoad').addEventListener('click', async () => {
      const facts = mp.facts;
      if (!facts || !facts.ok) return;
      closeModelPicker();
      if (facts.kind === 'repo') { await downloadModel(facts.path, '', true); return; }
      const added = await mpAddCurrent(true);
      const reference = added || facts.path;
      await loadModel(reference, facts.title || baseName(reference), baseName(reference));
    });
