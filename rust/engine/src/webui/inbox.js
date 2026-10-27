// ==========================================================================
    // The session mailbox: notes one session leaves for another (or for you).
    //  - The agent writes them with the `session_note` tool, a chat can add one, and the notes wait in
    //    .fllm/messages.json of the workspace until they are read here.
    //  - The chat header shows how many wait; one click puts a note into the composer, another marks it done.
    // ==========================================================================

    (function sessionInbox() {
      const headerRight = document.querySelector('.header-right');
      if (!headerRight) return;

      const button = document.createElement('button');
      button.className = 'btn btn-xs btn-ghost';
      button.id = 'btnInbox';
      button.title = 'Notes from other sessions';
      button.style.cssText = 'display:none;margin-left:6px;';
      button.textContent = 'Inbox';
      headerRight.insertBefore(button, headerRight.firstChild);

      let panel = null;
      let notes = [];
      let chatKey = '';

      const currentChat = () => {
        try {
          const t = typeof currentThread === 'function' ? currentThread() : null;
          return t ? String(t.id || '') : '';
        } catch {
          return '';
        }
      };

      function hide() {
        if (panel) { panel.remove(); panel = null; }
      }

      function render() {
        hide();
        if (!notes.length) return;
        panel = document.createElement('div');
        panel.className = 'ui-card';
        panel.style.cssText = 'position:fixed;top:64px;right:16px;width:min(420px,92vw);max-height:60vh;overflow:auto;z-index:60;padding:12px;border:1px solid var(--border);background:var(--card,#111);border-radius:10px;box-shadow:0 8px 30px rgba(0,0,0,.35);';
        panel.innerHTML =
          '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">' +
          '  <strong style="font-size:13px;">Notes from other sessions</strong>' +
          '  <span style="flex:1"></span>' +
          '  <button class="btn btn-xs btn-ghost" data-inbox-close>Close</button>' +
          '</div>' +
          notes.map((n) =>
            '<div style="border-top:1px solid var(--border);padding:8px 0;" data-note="' + n.id + '">' +
            '  <div style="font-size:11px;color:var(--muted-foreground);">#' + n.id + ' · ' + escapeHtml(n.from || 'a session') + (n.to ? ' → ' + escapeHtml(n.to) : '') + '</div>' +
            '  <div style="font-size:13px;white-space:pre-wrap;margin:4px 0 6px;">' + escapeHtml(n.text) + '</div>' +
            '  <button class="btn btn-xs btn-secondary" data-inbox-insert="' + n.id + '">Insert into the message</button> ' +
            '  <button class="btn btn-xs btn-ghost" data-inbox-done="' + n.id + '">Done</button>' +
            '</div>').join('');
        document.body.appendChild(panel);
      }

      async function refresh() {
        chatKey = currentChat();
        if (!chatKey) {
          // no chat is open yet: nothing to ask the server about
          notes = [];
          button.style.display = 'none';
          hide();
          return;
        }
        try {
          const res = await apiFetch('/api/studio/agent/messages?chat=' + encodeURIComponent(chatKey));
          const data = await res.json();
          notes = Array.isArray(data.messages) ? data.messages : [];
        } catch {
          notes = [];
        }
        button.style.display = notes.length ? '' : 'none';
        button.textContent = notes.length ? `Inbox (${notes.length})` : 'Inbox';
        if (!notes.length) hide();
      }

      button.addEventListener('click', () => {
        if (panel) { hide(); return; }
        render();
      });

      document.addEventListener('click', async (e) => {
        const target = e.target;
        if (panel && !panel.contains(target) && target !== button) { hide(); return; }
        if (target.closest('[data-inbox-close]')) { hide(); return; }
        const insert = target.closest('[data-inbox-insert]');
        if (insert) {
          const note = notes.find((n) => String(n.id) === insert.dataset.inboxInsert);
          const box = document.querySelector('#chatTextarea');
          if (note && box) {
            box.value = box.value ? `${box.value}\n\n${note.text}` : note.text;
            box.dispatchEvent(new Event('input', { bubbles: true }));
            box.focus();
          }
          hide();
          return;
        }
        const done = target.closest('[data-inbox-done]');
        if (done) {
          const id = Number(done.dataset.inboxDone);
          try {
            await apiFetch('/api/studio/agent/messages', { method: 'DELETE', body: { ids: [id] } });
          } catch { /* it stays, the next look shows it again */ }
          await refresh();
          return;
        }
      });

      // the badge has to follow the chat the user looks at
      const originalRender = typeof renderChatArea === 'function' ? renderChatArea : null;
      if (originalRender) {
        window.fllmInboxRefresh = refresh;
      }
      let last = '';
      setInterval(() => {
        const key = currentChat();
        if (key !== last) { last = key; refresh(); }
      }, 1500);
      refresh();
    })();
