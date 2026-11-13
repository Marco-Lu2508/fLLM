// Generates tests/fixtures/research_fixtures.json: the expected results of the pure research
// helpers of the Studio (rust/engine/src/webui/research.js, the part between @pure-begin and
// @pure-end), produced BY that code, so the Rust port in src/research.rs can be checked against
// the original.
// Run from this directory: node research_fixtures.js > fixtures/research_fixtures.json
'use strict';
const fs = require('fs');
const path = require('path');

const researchPath = path.resolve(__dirname, '../../../../rust/engine/src/webui/research.js');
const src = fs.readFileSync(researchPath, 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('markers not found in research.js');
// the four limits stand just above the marker
const limits = src.indexOf('const RESEARCH_MAX_ROUNDS');
const start = Math.min(src.lastIndexOf('\n', begin) + 1, limits < 0 ? Infinity : src.lastIndexOf('\n', limits) + 1);
const pure = src.slice(start, src.lastIndexOf('\n', end) + 1);

const sandbox = new Function(
  pure + '\n  return { RESEARCH_MAX_ROUNDS, RESEARCH_CONCURRENCY, RESEARCH_NOTE_SHARE, RESEARCH_PAGE_CHARS, researchClean, parsePlan, parseReflection, urlKey, selectUrls, findDate, fitNotes, noteBudget, researchMaxTokens };',
)();
const {
  RESEARCH_MAX_ROUNDS, RESEARCH_CONCURRENCY, RESEARCH_NOTE_SHARE, RESEARCH_PAGE_CHARS,
  researchClean, parsePlan, parseReflection, urlKey, selectUrls, findDate, fitNotes, noteBudget, researchMaxTokens,
} = sandbox;

const estimate = (t) => Math.ceil(String(t || '').length / 4);

// ---- fixtures -----------------------------------------------------------------------------
const out = {
  limits: { rounds: RESEARCH_MAX_ROUNDS, concurrency: RESEARCH_CONCURRENCY, noteShare: RESEARCH_NOTE_SHARE, pageChars: RESEARCH_PAGE_CHARS },
  clean: ['plain text', ' ```rust\nfn x() {}\n```  ', 'reason  think block  visible', '  spaced   out  ', ''].map((t) => researchClean(t)),
  plans: [
    { text: '{"queries": ["rust release cadence", "stable rust version"]}', max: undefined },
    { text: 'Here is the plan:\n{"queries": ["one", "two", "three"]}\nthanks', max: undefined },
    { text: '- first query\n* second query\n3. third query\nqueries: not this\n[]', max: undefined },
    { text: '{"searches": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]}', max: undefined },
    { text: '   ', max: undefined },
    { text: 'queries', max: undefined },
  ].map(({ text, max }) => ({ text, max: max === undefined ? null : max, out: parsePlan(text, max === undefined ? undefined : max) })),
  reflections: [
    '{"enough": true, "queries": []}',
    '{"enough": false, "queries": ["new angle", "other angle"]}',
    'not json at all',
    '{"enough": "maybe", "queries": ["kept anyway"]}',
    '{"queries": ["only queries"]}',
  ].map((text) => ({ text, out: parseReflection(text) })),
  urls: [
    'https://www.Example.com/index.html',
    'http://example.com/path/',
    'https://example.com/path/page',
    'ftp://files.test/x',
    'https://facebook.com/someone',
    'https://site.test/report.pdf',
    'https://site.test/login',
    'https://site.test/page?utm=1#frag',
    'not a url',
  ].map((u) => ({ url: u, out: urlKey(u) })),
  selects: [
    // two queries, duplicates across them, junk, a host with three pages
    [
      [{ title: 'A', url: 'https://one.test/a', snippet: 's' }, { title: 'B', url: 'https://two.test/b', snippet: '' }],
      [{ title: 'A again', url: 'https://one.test/a', snippet: '' }, { title: 'C', url: 'https://three.test/c', snippet: '' }],
    ],
    [
      [{ title: 'Social', url: 'https://www.facebook.com/x' }, { title: 'Pdf', url: 'https://four.test/doc.pdf' }],
      [{ title: 'Good', url: 'https://five.test/good' }],
    ],
    [[{ title: 'Same host 1', url: 'https://six.test/1' }, { title: 'Same host 2', url: 'https://six.test/2' }, { title: 'Same host 3', url: 'https://six.test/3' }, { title: 'Other', url: 'https://seven.test/o' }]],
  ].map((byQuery) => ({
    byQuery,
    out: selectUrls(byQuery).map((c) => ({ key: c.key, url: c.url, title: c.title, hits: c.hits })),
    seen: selectUrls(byQuery, { seen: new Set([selectUrls(byQuery)[0] && selectUrls(byQuery)[0].key].filter(Boolean)) }).map((c) => c.key),
  })),
  dates: [
    'The post from 2025-03-14 says hi.',
    'Published 14 March 2025 by someone.',
    'Dated March 14, 2025 — old news.',
    'Nothing here.',
    '2026-11-13 is the date at position beyond three thousand characters but only after the head: ' + 'x'.repeat(3100),
  ].map((t) => findDate(t)),
  fits: [
    { notes: [{ n: 1, text: 'x'.repeat(5) }], limit: 100 },
    { notes: [{ n: 1, text: 'x'.repeat(2000) }, { n: 2, text: 'x'.repeat(2000) }], limit: 300 },
    { notes: [{ n: 1, text: 'x'.repeat(5000) }, { n: 2, text: 'x'.repeat(5000) }, { n: 3, text: 'x'.repeat(5000) }], limit: 400 },
  ].map(({ notes, limit }) => {
    const r = fitNotes(notes, limit, estimate);
    return { notes: notes.map((n) => ({ n: n.n, len: n.text.length })), limit, out: { cut: r.cut, dropped: r.dropped, lens: r.notes.map((n) => n.text.length), tails: r.notes.map((n) => n.text.slice(-5)) } };
  }),
  noteBudgets: [
    { notes: [{ text: 'x'.repeat(400) }], max: 8192, fixed: 500 },
    { notes: [{ text: 'x'.repeat(60000) }], max: 8192, fixed: 500 },
    { notes: [{ text: 'short' }], max: 4096, fixed: 10000 },
  ].map(({ notes, max, fixed }) => {
    const r = noteBudget(notes, max, fixed, estimate);
    return { notes: notes.map((n) => n.text.length), max, fixed, out: r };
  }),
  maxTokens: [[8192, 1000, 1800], [8192, 8000, 1800], [4096, 4000, 1800], [2048, 100, 1800]].map(([c, p, w]) => researchMaxTokens(c, p, w)),
};

process.stdout.write(JSON.stringify(out, null, 2) + '\n');
