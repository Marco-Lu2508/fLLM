// Generates tests/fixtures/promptlib_fixtures.json: the expected results of the pure prompt library
// search of the Studio (rust/engine/src/webui/projects.js, plibMatch inside the @pure block), produced
// BY that code, so the Rust port in src/promptlib.rs can be checked against the original.
// Run from this directory: node promptlib_fixtures.js > fixtures/promptlib_fixtures.json
'use strict';
const fs = require('fs');
const path = require('path');

const projectsPath = path.resolve(__dirname, '../../../../rust/engine/src/webui/projects.js');
const src = fs.readFileSync(projectsPath, 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('markers not found in projects.js');
const pure = src.slice(src.lastIndexOf('\n', begin) + 1, src.lastIndexOf('\n', end) + 1);

const sandbox = new Function(pure + '\n  return { plibMatch, SLASH_COMMANDS };')();
const { plibMatch } = sandbox;

// ---- fixtures -----------------------------------------------------------------------------
const prompts = [
  { id: 'pr1', title: 'Email draft', text: 'Write a short email about: ' },
  { id: 'pr2', title: 'email me a plan', text: 'Plan my week around: ' },
  { id: 'pr3', title: 'Email reply to a customer', text: 'Write a polite answer: ' },
  { id: 'pr4', title: 'Summarize this text', text: 'Summarize in three sentences: ' },
  { id: 'pr5', title: 'summary for my boss', text: 'One paragraph for a manager: ' },
  { id: 'pr6', title: 'Translate to German', text: 'Translate to German: ' },
  { id: 'pr7', title: 'Explain like I am five', text: 'Explain it simply: ' },
  { id: 'pr8', title: 'Code review', text: 'Review this code: ' },
  { id: 'pr9', title: 'code cleanup', text: 'Clean up this code: ' },
  { id: 'pr10', title: 'Daily note', text: 'A note for today: ' },
  { id: 'pr11', title: 'Clear my inbox', text: 'Sort these messages: ' },
];
const texts = [
  '/email',
  '/EMAIL',
  '/email ',
  '/',
  '/sum',
  '/summarize',
  '/plan',
  '/c',
  '/clear',        // the built-in command wins: the library is left alone
  '/clear now',    // ... also when words follow it
  '/cle',          // not the built-in: a title of the library starts with "cle"
  '/NEW',
  '/  summarize  ',
  'email',
  'no slash at all',
  '/xyz',
];

const out = {
  prompts,
  cases: texts.map((text) => ({ text, indices: plibMatch(prompts, text) })),
};
process.stdout.write(JSON.stringify(out, null, 2) + '\n');
