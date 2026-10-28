// Tests for the pure logic of computer use in the Code page (computer.js): action subjects for the permission rules, the
// crosshair markers, the pictures that go to the model and are never saved, and the panic keys.
// Run from the repository root:  node rust/engine/src/webui/tests/computer.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '..', 'computer.js'), 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
assert(begin >= 0 && end > begin, 'pure block markers not found in computer.js');
const C = new Function(`${src.slice(begin, end)}
return { computerSubject, computerQuestionText, computerMarks, marksSvg, dropOldImages, withoutImages, toWireMessages, escapeStreak, isPanicChord };`)();

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

test('every action has a one-line subject that rules can match', () => {
  assert.strictEqual(C.computerSubject('computer_click', { x: 10, y: 20.55 }), 'computer_click 10,20.6');
  assert.strictEqual(C.computerSubject('computer_click', { x: 1, y: 2, button: 'right', double: true }), 'computer_click 1,2 right double');
  assert.strictEqual(C.computerSubject('computer_drag', { from: { x: 1, y: 2 }, to: [3, 4] }), 'computer_drag 1,2 -> 3,4');
  assert.strictEqual(C.computerSubject('computer_type', { text: 'a "b"\nc' }), 'computer_type "a \\"b\\"\\nc"');
  assert.strictEqual(C.computerSubject('computer_key', { keys: 'ctrl+s' }), 'computer_key ctrl+s');
  assert.strictEqual(C.computerSubject('computer_scroll', { x: 5, y: 6, dy: 3 }), 'computer_scroll 5,6 dx=0 dy=3');
  assert.strictEqual(C.computerSubject('computer_screenshot', {}), 'computer_screenshot');
  assert.strictEqual(C.computerSubject('computer_click', {}), 'computer_click 0,0', 'missing numbers do not throw');
});

test('the permission question names the action', () => {
  assert.match(C.computerQuestionText('computer_click', { double: true }), /Double-click/);
  assert.match(C.computerQuestionText('computer_click', { button: 'right' }), /right-click/);
  assert.match(C.computerQuestionText('computer_type', { text: 'x'.repeat(200) }), /\.\.\./);
  assert.match(C.computerQuestionText('computer_key', { keys: 'enter' }), /enter/);
  assert.match(C.computerQuestionText('computer_screenshot', {}), /look at your screen/);
});

test('markers: a cross for click and move, a line with two crosses for a drag, nothing for typing', () => {
  assert.deepStrictEqual(C.computerMarks('computer_click', { x: 5, y: 6 }), [{ kind: 'cross', x: 5, y: 6, double: false }]);
  assert.strictEqual(C.computerMarks('computer_move', { x: 5, y: 6 })[0].kind, 'cross');
  const drag = C.computerMarks('computer_drag', { from: { x: 1, y: 2 }, to: { x: 30, y: 40 } });
  assert.deepStrictEqual(drag.map((m) => m.kind), ['line', 'cross', 'cross']);
  assert.deepStrictEqual([drag[0].x1, drag[0].y1, drag[0].x2, drag[0].y2], [1, 2, 30, 40]);
  assert.deepStrictEqual(C.computerMarks('computer_scroll', { x: 1, y: 2, dy: 3 }).map((m) => m.kind), ['cross', 'arrow']);
  assert.deepStrictEqual(C.computerMarks('computer_type', { text: 'x' }), []);
  assert.deepStrictEqual(C.computerMarks('computer_click', { x: 'a', y: 1 }), [], 'bad numbers draw nothing');
  assert.deepStrictEqual(C.computerMarks('computer_drag', { from: { x: 1 }, to: { x: 2, y: 2 } }), []);
});

test('the crosshair svg uses the pixels of the screenshot and is drawn twice for contrast', () => {
  const svg = C.marksSvg(1568, 882, C.computerMarks('computer_click', { x: 784, y: 441 }));
  assert.ok(svg.startsWith('<svg class="cu-mark" viewBox="0 0 1568 882"'));
  assert.ok(svg.includes('cx="784" cy="441"'));
  assert.strictEqual((svg.match(/stroke="#fff"/g) || []).length, 1);
  assert.strictEqual((svg.match(/stroke="#e5231b"/g) || []).length, 1);
  assert.strictEqual(C.marksSvg(100, 100, []), '');
  assert.strictEqual(C.marksSvg(0, 100, [{ kind: 'cross', x: 1, y: 1 }]), '');
  const double = C.marksSvg(800, 600, C.computerMarks('computer_click', { x: 10, y: 10, double: true }));
  assert.strictEqual((double.match(/<circle/g) || []).length, 4, 'a double click has a second ring (in both layers)');
  const arrow = C.marksSvg(800, 600, C.computerMarks('computer_scroll', { x: 100, y: 100, dy: 3 }));
  assert.ok(arrow.includes('<path'), 'the scroll arrow');
});

test('only the last two pictures stay in the conversation', () => {
  const shot = (i) => ({ role: 'user', content: `shot ${i}`, images: [`data:image/png;base64,${i}`] });
  const messages = [{ role: 'user', content: 'go' }, shot(1), { role: 'assistant', content: 'a' }, shot(2), shot(3), shot(4)];
  assert.strictEqual(C.dropOldImages(messages, 2), 2);
  assert.strictEqual(messages.filter((m) => m.images).length, 2);
  assert.deepStrictEqual(messages.filter((m) => m.images).map((m) => m.content), ['shot 3', 'shot 4']);
  assert.match(messages[1].content, /removed from the conversation/);
  assert.strictEqual(C.dropOldImages(messages, 2), 0, 'nothing more to drop');
  assert.strictEqual(C.dropOldImages([], 2), 0);
});

test('saved sessions and summaries carry no pictures, the original messages are untouched', () => {
  const messages = [{ role: 'user', content: 'shot', images: ['data:image/png;base64,AAAA'] }, { role: 'user', content: 'plain' }];
  const clean = C.withoutImages(messages);
  assert.ok(!JSON.stringify(clean).includes('AAAA'));
  assert.ok(clean[0].content.includes('not kept'));
  assert.strictEqual(clean[1], messages[1]);
  assert.ok(messages[0].images, 'the live conversation keeps its picture');
});

test('the server gets image parts in the OpenAI form', () => {
  const wire = C.toWireMessages([{ role: 'user', content: 'see', images: ['data:image/png;base64,QQ=='] }, { role: 'assistant', content: 'ok' }]);
  assert.deepStrictEqual(wire[0], { role: 'user', content: [{ type: 'text', text: 'see' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,QQ==' } }] });
  assert.deepStrictEqual(wire[1], { role: 'assistant', content: 'ok' });
  assert.ok(!('images' in wire[0]));
});

test('Escape three times within a moment stops, slower or fewer does not', () => {
  let s = C.escapeStreak([], 1000);
  assert.deepStrictEqual(s, { times: [1000], fire: false });
  s = C.escapeStreak(s.times, 1400);
  assert.strictEqual(s.fire, false);
  s = C.escapeStreak(s.times, 1900);
  assert.strictEqual(s.fire, true);
  assert.deepStrictEqual(s.times, []);
  // too slow: the first press is forgotten
  let t = C.escapeStreak([], 0);
  t = C.escapeStreak(t.times, 1000);
  t = C.escapeStreak(t.times, 2200);
  assert.strictEqual(t.fire, false);
  t = C.escapeStreak(t.times, 2400);
  assert.strictEqual(t.fire, true);
});

test('the panic chord needs all four keys', () => {
  assert.strictEqual(C.isPanicChord({ key: 'Escape', ctrlKey: true, altKey: true, shiftKey: true }), true);
  assert.strictEqual(C.isPanicChord({ key: 'Escape', ctrlKey: true, altKey: true, shiftKey: false }), false);
  assert.strictEqual(C.isPanicChord({ key: 'Escape' }), false);
  assert.strictEqual(C.isPanicChord({ key: 'a', ctrlKey: true, altKey: true, shiftKey: true }), false);
  assert.strictEqual(C.isPanicChord(null), false);
});

console.log(`${n} tests passed`);
