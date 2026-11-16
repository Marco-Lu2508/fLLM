// Fixtures of `/skill`: the @pure block of skills.js (`skTranscript`, `skProposalPrompt`,
// `skParseProposal`) runs in a vm. Run: node tests/fixtures/skillpropose_fixtures.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const src = fs.readFileSync(path.join(WEBUI, 'skills.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('the pure block of skills.js was not found');
const zone = src.slice(src.indexOf('\n', begin) + 1, src.lastIndexOf('\n', end));
const api = vm.runInNewContext(`${zone}\n;({ skTranscript, skProposalPrompt, skParseProposal })`, { Math, String, Number, Array, Object, RegExp, JSON }, { timeout: 5000 });

const long = (n, c = 'x') => c.repeat(n);
const SESSIONS = {
  empty: [],
  only_system: [{ role: 'system', content: 'be good' }],
  simple: [{ role: 'user', content: 'fix the bug' }, { role: 'assistant', content: 'done' }],
  calls: [
    { role: 'user', content: 'run it' },
    { role: 'assistant', content: '', tool_calls: [{ function: { name: 'bash', arguments: '{"command":"npm test"}' } }] },
    { role: 'tool', content: 'ok 3 passed' },
    { role: 'assistant', content: 'passed', tool_calls: [{ function: { name: 'read_file', arguments: long(300, 'a') } }, { function: { name: 'edit', arguments: '{}' } }] },
  ],
  clipped_text: [{ role: 'user', content: long(1200, 'b') }, { role: 'assistant', content: `${long(800, 'c')}${long(400, 'd')}${long(300, 'e')}` }],
  empty_texts_skipped: [{ role: 'user', content: '   ' }, { role: 'assistant', content: '' }, { role: 'user', content: 'hi' }, { role: 'tool', content: null }],
  unicode: [{ role: 'user', content: `${long(798, 'a')}😀${long(400, 'z')}` }, { role: 'assistant', content: 'ünï' }],
  too_long: Array.from({ length: 40 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `${i} ${long(600, String(i % 10))}` })),
};
const REPLIES = {
  none: 'NONE', none_dot: 'none.', empty: '   ', think_only: '<think>hm</think>',
  good: '---\nname: run-tests\ndescription: Runs the tests\nwhen: after a change\n---\n# Title\n\n1. npm test\n',
  fenced: '```markdown\n---\nname: run-tests\ndescription: Runs the tests\nwhen: after a change\n---\n# Title\n\n1. npm test\n```',
  fenced_plain: '```\n---\nname: x\ndescription: d\n---\nbody\n```',
  with_think: '<think>planning</think>\n---\nname: run-tests\ndescription: Runs the tests\n---\nbody here\n',
  preamble: 'Here is the skill:\n---\nname: run-tests\ndescription: Runs\n---\nbody\n',
  no_front_matter: '# Just a title\n\n1. a step\n',
  no_description: '---\nname: a\n---\nbody\n',
  no_body: '---\nname: a\ndescription: d\n---\n   \n',
  bad_name: '---\nname: Run The Tests!\ndescription: Runs the tests\n---\nbody\n',
  no_name: '---\ndescription: Runs the unit tests of a project quickly now\n---\nbody\n',
  unusable_name: '---\nname: ***\ndescription: ***\n---\nbody\n',
  long_name: `---\nname: ${long(80, 'n')}\ndescription: d\n---\nbody\n`,
  crlf: '---\r\nname: crlf-skill\r\ndescription: d\r\n---\r\nbody\r\n',
  trailing_space: '---\nname: ok\ndescription: d\n---\nbody   \n\n\n',
};
const out = {
  transcripts: Object.entries(SESSIONS).map(([name, messages]) => ({ name, messages, got: api.skTranscript(messages) })),
  small_cap: ['simple', 'too_long'].map((name) => ({ name, max: 200, messages: SESSIONS[name], got: api.skTranscript(SESSIONS[name], 200) })),
  prompts: ['', 'a\nb', 'SESSION'].map((t) => ({ transcript: t, got: api.skProposalPrompt(t) })),
  replies: Object.entries(REPLIES).map(([name, reply]) => ({ name, reply, got: api.skParseProposal(reply) })),
  null_reply: api.skParseProposal(null),
};
fs.writeFileSync(path.join(__dirname, 'skillpropose_fixtures.json'), JSON.stringify(out, null, 1));
console.log('written', out.transcripts.length, out.replies.length);
