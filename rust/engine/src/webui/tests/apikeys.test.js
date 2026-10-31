// The "create key" flow never asks for the secret: no field for it in the page, and the request carries name, limits and the admin right only.
// Run from the repository root:  node rust/engine/src/webui/tests/apikeys.test.js
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const dir = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(dir, 'shell.html'), 'utf8');
const js = fs.readFileSync(path.join(dir, 'listeners-panels.js'), 'utf8');

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`ok - ${name}`); };

test('the page has no input for the value of a key', () => {
  assert(!/inpKeyCustom/.test(html) && !/inpKeyCustom/.test(js));
  assert(!/Your own key/i.test(html));
  const card = html.slice(html.indexOf('<!-- API Keys Card -->'), html.indexOf('<!-- Live Endpoint Tester -->'));
  const inputs = [...card.matchAll(/<input[^>]*id="([^"]+)"/g)].map((m) => m[1]);
  assert.deepStrictEqual(inputs, ['inpKeyLabel', 'inpKeyRpm', 'inpKeyTpm', 'chkKeyAdmin']);
});

test('the create request sends no secret and shows the one the server made', () => {
  const start = js.indexOf("$('btnCreateKey').addEventListener");
  const handler = js.slice(start, js.indexOf("$('btnCopyNewKey')"));
  const call = handler.match(/apiFetch\('\/api\/studio\/keys', \{ method: 'POST', body: \{([^}]*)\}/);
  assert(call, 'create call found');
  const fields = call[1].split(',').map((s) => s.trim().split(':')[0].trim());
  assert.deepStrictEqual(fields, ['label', 'rpm', 'tpm', 'admin']);
  assert(/newKeySecretText'\)\.textContent = data\.secret/.test(handler));
});

console.log(`${n} tests passed`);
