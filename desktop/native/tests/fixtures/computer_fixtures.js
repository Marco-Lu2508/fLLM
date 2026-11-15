#!/usr/bin/env node
// Row 86 fixtures: the `@pure` block of the page's computer.js (the subjects and questions of the
// computer tools, the marks drawn on a screenshot, the pictures kept in a conversation, the
// panic keys) run in a vm over the cases below; the Rust port in src/computer.rs must give the
// same answers.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEBUI = path.resolve(__dirname, '../../../../rust/engine/src/webui');
const src = fs.readFileSync(path.join(WEBUI, 'computer.js'), 'utf8');
const a = src.indexOf('// @pure-begin');
const b = src.indexOf('// @pure-end');
if (a < 0 || b < 0) throw new Error('the pure block of computer.js is gone');
const context = { console, JSON, Math, Set, Map, Array, Object, String, Number, Boolean };
vm.createContext(context);
vm.runInContext(src.slice(a, b), context, { filename: 'computer.pure.js' });
const run = (expr, vars) => { Object.assign(context, vars); return JSON.parse(JSON.stringify(vm.runInContext(`(() => { const r = ${expr}; return r === undefined ? { __undefined: true } : r; })()`, context))); };

const NAMES = ['computer_screenshot', 'computer_click', 'computer_move', 'computer_drag', 'computer_type', 'computer_key', 'computer_scroll', 'computer_wait', 'computer_cursor', 'computer_displays', 'something_else'];
const ARGS = [
  null, {}, { x: 10, y: 20 }, { x: 10.04, y: 20.05 }, { x: 1.25, y: -3.75 }, { x: '5', y: '6' }, { x: 'a', y: null },
  { x: 1, y: 2, button: 'right' }, { x: 1, y: 2, button: 'left' }, { x: 1, y: 2, double: true }, { x: 1, y: 2, button: 'middle', double: true },
  { from: { x: 1, y: 2 }, to: { x: 30, y: 40 } }, { from: [1, 2], to: [3, 4] }, { from: { x: 1 }, to: 'x' }, { from: null, to: null },
  { text: 'hello' }, { text: 'a'.repeat(100) }, { text: 'a'.repeat(80) }, { text: 'a'.repeat(81) }, { text: 'quote " and \\ slash\nline' }, { text: 12 }, { text: null },
  { keys: 'ctrl+c' }, { keys: ['a'] }, { keys: null },
  { x: 5, y: 6, dx: 0, dy: 120 }, { x: 5, y: 6, dx: -3.3, dy: 0.04 }, { x: 5, y: 6 }, { dx: 1, dy: 2 },
  { ms: 250 }, { ms: 0 }, { ms: null }, {},
];
const subjects = [];
const questions = [];
const marks = [];
for (const name of NAMES) {
  for (const args of ARGS) {
    subjects.push({ name, args, result: run('computerSubject(__n, __a)', { __n: name, __a: args }) });
    questions.push({ name, args, result: run('computerQuestionText(__n, __a)', { __n: name, __a: args }) });
    marks.push({ name, args, result: run('computerMarks(__n, __a)', { __n: name, __a: args }) });
  }
}

const SVG_CASES = [
  { w: 1280, h: 800, name: 'computer_click', args: { x: 100, y: 200 } },
  { w: 1280, h: 800, name: 'computer_click', args: { x: 100.55, y: 200.45, double: true } },
  { w: 1920, h: 1080, name: 'computer_scroll', args: { x: 500, y: 500, dx: 0, dy: 300 } },
  { w: 1920, h: 1080, name: 'computer_scroll', args: { x: 500, y: 500, dx: 0, dy: 0 } },
  { w: 800, h: 600, name: 'computer_drag', args: { from: { x: 10, y: 20 }, to: { x: 300, y: 400 } } },
  { w: 400, h: 300, name: 'computer_move', args: { x: 5, y: 5 } },
  { w: 3840, h: 2160, name: 'computer_click', args: { x: 3000, y: 2000 } },
  { w: 100, h: 100, name: 'computer_click', args: { x: 50, y: 50 } },
  { w: 0, h: 800, name: 'computer_click', args: { x: 1, y: 1 } },
  { w: 800, h: 0, name: 'computer_click', args: { x: 1, y: 1 } },
  { w: 800, h: 600, name: 'computer_type', args: { text: 'x' } },
];
const svgs = SVG_CASES.map((c) => ({ ...c, result: run('marksSvg(__w, __h, computerMarks(__n, __a))', { __w: c.w, __h: c.h, __n: c.name, __a: c.args }) }));

const img = (n) => ({ role: 'user', content: `shot ${n}`, images: Array.from({ length: n }, (_, i) => `data:image/png;base64,${i}`) });
const MESSAGE_SETS = [
  { name: 'none', messages: [] },
  { name: 'no pictures', messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'yo' }] },
  { name: 'one carrier', messages: [{ role: 'user', content: 'hi' }, img(1)] },
  { name: 'two carriers', messages: [img(1), { role: 'assistant', content: 'x' }, img(2)] },
  { name: 'three carriers', messages: [img(1), img(2), img(1), { role: 'user', content: 'end' }] },
  { name: 'four carriers keep two', messages: [img(1), img(1), img(3), img(1)] },
  { name: 'an empty images list is no carrier', messages: [img(1), { role: 'user', content: 'e', images: [] }, img(1), img(1)] },
];
const images = MESSAGE_SETS.map((s) => {
  const keep2 = JSON.parse(JSON.stringify(s.messages));
  const dropped = run('dropOldImages(__m, 2)', { __m: keep2 });
  const keep1 = JSON.parse(JSON.stringify(s.messages));
  const dropped1 = run('dropOldImages(__m, 1)', { __m: keep1 });
  return {
    ...s,
    dropped, afterKeep2: keep2, dropped1, afterKeep1: keep1,
    without: run('withoutImages(__m)', { __m: JSON.parse(JSON.stringify(s.messages)) }),
    wire: run('toWireMessages(__m)', { __m: JSON.parse(JSON.stringify(s.messages)) }),
  };
});

const STREAKS = [
  { times: [], now: 1000 }, { times: [900], now: 1000 }, { times: [800, 900], now: 1000 }, { times: [100, 900], now: 1000 },
  { times: [-500, 900], now: 1000 }, { times: [-500, -400, 900], now: 1000 }, { times: [0, 800], now: 1500 }, { times: [0, 800], now: 1501 },
  { times: [1, 2, 3, 4], now: 5 },
];
const streaks = STREAKS.map((s) => ({ ...s, result: run('escapeStreak(__t, __n)', { __t: s.times, __n: s.now }) }));
const streaks2 = STREAKS.map((s) => ({ ...s, windowMs: 100, needed: 2, result: run('escapeStreak(__t, __n, 100, 2)', { __t: s.times, __n: s.now }) }));
const KEYS = [
  { key: 'Escape', ctrlKey: true, altKey: true, shiftKey: true }, { key: 'Escape', ctrlKey: true, altKey: true, shiftKey: false },
  { key: 'Escape' }, { key: 'a', ctrlKey: true, altKey: true, shiftKey: true }, null, {}, { key: 'Esc', ctrlKey: true, altKey: true, shiftKey: true },
];
const panic = KEYS.map((e) => ({ e, result: run('isPanicChord(__e)', { __e: e }) }));

fs.writeFileSync(path.join(__dirname, 'computer_fixtures.json'), JSON.stringify({ subjects, questions, marks, svgs, images, streaks, streaks2, panic }, null, 1));
console.log(`subjects ${subjects.length}, questions ${questions.length}, marks ${marks.length}, svgs ${svgs.length}, images ${images.length}`);
console.log('FIXTURES WRITTEN');
