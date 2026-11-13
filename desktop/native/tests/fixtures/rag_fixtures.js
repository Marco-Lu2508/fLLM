// Generates tests/fixtures/rag_fixtures.json: the expected results of the pure document search of the
// Studio (rust/engine/src/webui/rag.js, the part between @pure-begin and @pure-end), produced BY that
// code, so the Rust port in src/rag.rs can be checked against the original.
// Run from this directory: node rag_fixtures.js > fixtures/rag_fixtures.json
'use strict';
const fs = require('fs');
const path = require('path');

const ragPath = path.resolve(__dirname, '../../../../rust/engine/src/webui/rag.js');
const src = fs.readFileSync(ragPath, 'utf8');
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('markers not found in rag.js');
const pure = src.slice(src.lastIndexOf('\n', begin) + 1, src.lastIndexOf('\n', end) + 1);

const sandbox = new Function(pure + `
  return { ragTokenize, ragChunk, ragBuildIndex, ragSearch, ragCosine, ragOpening, RAG_CHUNK_CHARS, RAG_OVERLAP_CHARS };
`)();
const { ragTokenize, ragChunk, ragBuildIndex, ragSearch, ragCosine, ragOpening } = sandbox;

// ---- fixtures -----------------------------------------------------------------------------
const docs = [
  {
    id: 'doc-a',
    name: 'report.pdf',
    text: [
      'The quarterly report covers revenue growth of twelve percent. Costs stayed flat, and the team shipped three features on time.',
      '',
      'Servers are the backbone of the platform. A server rack holds many machines, and each machine runs several services. When a server fails, the services move to another server. Our servers run in two data centers, and the operators check the servers every night. Documents about the servers live in the knowledge base.',
      '',
      'A short one.',
    ].join('\n'),
  },
  {
    id: 'doc-b',
    name: 'notes.md',
    text: (() => {
      // one long paragraph: sentences longer than the chunk size force the sentence/word cutting with overlap
      const sentence = 'The quick brown fox jumps over the lazy dog while the parser splits this sentence into passages of about eighteen hundred characters. ';
      const long = sentence.repeat(14); // > 1800 chars
      return 'Intro line about tokens and chunking.\n\n' + long + '\n\nTail paragraph with the word chunk twice: chunk one, chunk two.';
    })(),
  },
  {
    id: 'doc-c',
    name: 'kurz.txt',
    text: 'Zusammenfassung: Der Bericht behandelt Umsatz, Kosten und Server. Lesen Sie die Dokumente für die Details.',
  },
];

const queries = [
  'How many servers are in the data center?',
  'what does the quarterly report say about revenue?',
  'chunk overlap passages',
  'Dokumente und Server',
  'zzzz nothing matches this at all',
  'the of and', // stopwords only: no terms
  'Servers servers SERVERS chunking', // case and plural folding
  '42', // digits only
];

const out = { docs: docs.map((d) => ({ id: d.id, name: d.name, text: d.text })), cases: [] };

for (const q of queries) {
  out.cases.push({ kind: 'tokenize', query: q, tokens: ragTokenize(q) });
}

// chunking of the long document, per document
for (const d of docs) {
  out.cases.push({ kind: 'chunk', id: d.id, chunks: ragChunk(d.text) });
}

// index facts + search results
const index = ragBuildIndex(docs);
out.index = { n: index.chunks.length, avg: index.avg, df: Object.fromEntries([...index.df.entries()].sort()) };
for (const q of queries) {
  const hits = ragSearch(index, q, 4);
  out.cases.push({
    kind: 'search',
    query: q,
    hits: hits.map((h) => ({ docId: h.docId, doc: h.doc, n: h.n, m: h.m, score: h.score, text: h.text })),
  });
}
out.cases.push({ kind: 'opening', k: 5, hits: ragOpening(index, 5).map((h) => ({ docId: h.docId, n: h.n, m: h.m })) });
out.cases.push({ kind: 'opening', k: 2, hits: ragOpening(index, 2).map((h) => ({ docId: h.docId, n: h.n, m: h.m })) });

// cosine
out.cases.push({ kind: 'cosine', a: [1, 2, 3], b: [1, 2, 3], v: ragCosine([1, 2, 3], [1, 2, 3]) });
out.cases.push({ kind: 'cosine', a: [1, 0], b: [0, 1], v: ragCosine([1, 0], [0, 1]) });
out.cases.push({ kind: 'cosine', a: [0, 0], b: [1, 1], v: ragCosine([0, 0], [1, 1]) });
out.cases.push({ kind: 'cosine', a: [1, 2], b: [2, 4], v: ragCosine([1, 2], [2, 4]) });

// the prefix of the prompt, exactly as the Studio builds it
const found = ragSearch(index, 'servers in the data center', 3);
found.sort((a, b) => (a.docId === b.docId ? a.n - b.n : 0));
const passages = found.map((p) => ({ doc: p.doc, n: p.n, m: p.m, text: p.text }));
let prefix = '';
if (passages.length) {
  prefix += 'Passages from the attached documents. Use them to answer, and name the source in brackets, for example [report.pdf, chunk 3/42], when you rely on one.\n\n';
  prefix += passages.map((p) => `[${p.doc}, chunk ${p.n}/${p.m}]\n${p.text}`).join('\n\n');
  prefix += '\n\n';
}
prefix = prefix ? prefix + '---\n' : '';
out.cases.push({ kind: 'prefix', passages, prefix });

process.stdout.write(JSON.stringify(out, null, 1) + '\n');
