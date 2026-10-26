# Personal-agent parts of the Studio: skills, session search, scheduled tasks

Hermes Agent (Nous Research) is a personal agent with persistent memory, self-written skills, session search,
a cron scheduler with delivery, sub-agents, a messaging gateway, MCP and several terminal backends. This note
says which of these parts fit a local desktop app, what was built, and what was left out on purpose.

| Hermes feature | Here |
|---|---|
| Persistent memory | Not part of this change (the chat memory work is separate). |
| Skills the agent writes and improves | Built: `skill` tool, Settings pane, skill index for the system prompt, "propose a skill" helper. |
| Session search | Built: BM25 in the page plus a persistent text copy on the computer. |
| Cron with delivery | Built, local: a clock thread in the server, results list in the app, optional desktop notification. |
| Sub-agent delegation | Exists on the code page (`spawn_agent`); see "Delegation" below. |
| Messaging gateway | Not built (see below). |
| MCP | Exists (Settings, Tools). |
| Terminal backends | Not built: the agent runs on this computer only. Docker, SSH and cloud sandboxes need accounts or an always-on host. |

## Skills

A skill is `<name>/SKILL.md`: a front matter (`name`, `description`, `when`) and the steps. User skills live in
`skills/` next to `studio.json`; workspace skills in `<folder>/.fllm/skills/`. Names are lowercase slugs
(`[a-z0-9][a-z0-9_-]{0,63}`), so a name cannot contain a path; links are never followed; a skill is at most 64 KB.

Server (loopback only, like the agent routes, reads included): `GET/POST/DELETE /api/studio/skills`,
`GET /api/studio/skills/read`. A workspace is passed as an absolute folder in `workspace`.

Page API (`skills.js`):
- `fllmSkillIndex({workspace?})`: synchronous text for the system prompt (names, descriptions, when; capped at
  about 1800 characters and 40 skills); refreshes in the background. `await fllmSkillIndexRefresh(workspace)` for a
  current list.
- `fllmSkillRead(name, {workspace?, scope?})`: text or `null`; a workspace skill wins over a user skill.
- `fllmSkillTool` (`{name, permission: 'edit', description, properties, required}`, the shape of `CLIENT_TOOLS` in
  `code.js`), `fllmSkillToolDef` (OpenAI function schema) and `fllmSkillToolRun(args, {workspace?})` which resolves to
  the text for the model. Actions: list, read, create, update, delete.
- `fllmSkillPropose({messages | transcript, workspace?})`: asks the selected model to write a skill from a finished
  session (prompt template `skProposalPrompt`) and opens an approval dialog; nothing is saved without Save.

How to wire it: append `fllmSkillIndex()` to the system prompt, add `fllmSkillTool` to the tool list and call
`fllmSkillToolRun` as its executor, and show a "Save as skill" button (or a notice) after a long successful task that
calls `fllmSkillPropose`.

## Session search

`fllmSessionSearch(query, {limit?, localOnly?})` resolves to hits `{id, threadId, kind, title, snippet, time, score,
source, workspace?}` (`kind` chat or code; code sessions have the id `code:<session id>`; `source` is `browser` or
`server` for sessions that only the copy knows). The Ctrl+K modal in `boot.js` can call it and merge the hits into
its list; this change does not edit `boot.js`.

The copy: the page posts each changed chat and code session to `POST /api/studio/sessions/index`
(`{id, title, text, kind, updated}`) every 20 seconds, never temporary chats and never a message that is still
streaming. The server appends to `sessions.jsonl` next to `studio.json`; past 24 MB it keeps the newest sessions
that fit into 12 MB (and at most 5000). `GET /api/studio/sessions/search?q=` ranks the newest record of each
session with the same BM25 as the page. `DELETE ...?id=` and `?all=1` remove copies; a deleted chat is removed from
the copy by the page. The switch in Settings turns the copy off. The copy is plain text on disk, like the browser's
own storage.

## Scheduled tasks

A task has a name, a prompt, a schedule, a kind and a notify flag. Schedules (local time): `every 30 minutes`,
`every 2 hours`, `hourly`, `hourly at :15`, `daily at 08:30` (also `8:30pm`), `weekdays at 09:00`,
`weekly on mon,thu at 18:00`, or a cron line with five fields (lists, ranges, steps, names of weekdays and months;
day of month and weekday combine with OR like cron).

Daylight saving: intervals count real time. A wall-clock time that does not exist on a day (spring) runs at the first
moment after the gap; one that exists twice (autumn) runs once, at its first occurrence. An hourly line therefore skips
the repeated hour.

Server: a thread ticks every 5 seconds, plans each task, and runs due tasks on their own thread; it calls the
server's own chat-completions handler in process (the same code as `POST /v1/chat/completions`, no port or key
needed). One task never runs twice at once (a "run now" while it runs answers 409). A run that is more than an hour
late (the computer slept) is skipped, and runs missed while the app was closed are not made up. Tasks run only while
a model is loaded: with no model the scheduler is not running (the pane says so) and tasks can still be defined.

Kinds: `chat` is one answer. `agent` runs a tool loop (at most 12 steps) in an absolute folder with a fixed set:
`list_dir`, `read_file`, `grep`, `glob`, `web_fetch`, `web_search`. The server hands the model only those schemas and
refuses every other tool name, so a task cannot write files or run commands. `FLLM_NO_AGENT` switches agent tasks off.

Storage: `schedule.json` (tasks) and `schedule-results.json` (the last 20 results per task, 20,000 characters each).
Routes (loopback only): `GET/POST/DELETE /api/studio/schedule`, `POST .../run`, `GET .../results?id=&since=`,
`GET .../check?when=` (validates and shows the next runs). Delivery: the list in Settings and, while a page is open,
a toast and, if the task asked for it and the browser allows, a desktop notification (the page polls once a minute).

## Delegation and the `schedule` tool

The code page's `spawn_agent` starts a sub-agent with a fresh context inside the page's loop. The new tools are
ordinary tools of the same loop: a sub-agent in `full` mode may be given `skill` (read it before a job, create after);
a `read-only` sub-agent should get `skill` with the read and list actions only (`fllmSkillCheckArgs` can refuse the
others). Scheduled agent runs are a different thing: they run in the server without the page, with the fixed
read-only set.

`fllmScheduleTool` / `fllmScheduleToolDef` / `fllmScheduleToolRun(args)` is the tool for "remind me ..." and "every
morning summarise ...": the model supplies `{name, prompt, when, mode?, folder?, notify?}`; the page validates the
schedule with the server, shows a confirmation dialog with the next run time, and only then creates the task. The
prompt must be self-contained, because the task runs without the conversation.

## Left out on purpose: messaging gateways

Telegram, Discord, Slack and similar gateways need an external account and token, an always-on process and a public
or outbound connection, and they let remote messages drive an agent on this computer. A minimal safe design, if it
is ever built: an opt-in sidecar process (not part of the server), outbound connections only (long polling), an
allow-list of sender ids, the same fixed read-only tool set as scheduled tasks, no shell or file writes, tokens in a
file that the page never receives, and the results delivery of scheduled tasks as the only outbound path
(`schedule-results.json` plus a send-to-chat step). Nothing of this exists yet.

## Not verified

The JavaScript was syntax-checked and its pure parts tested in node; the Settings pane, the approval dialog,
notifications and the in-process model call of a scheduled run were not exercised in a browser or with a real model.
