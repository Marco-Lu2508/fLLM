// The fixtures of the automatic approval are produced by code.js itself: the @pure-begin..
// @pure-end block of the Studio is run in a sandbox and every function of it is asked the cases
// the Rust tests ask. Run: node tests/fixtures/autoapprove_fixtures.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const src = fs.readFileSync(
  path.join(__dirname, '../../../../rust/engine/src/webui/code.js'),
  'utf8',
);
const begin = src.indexOf('// @pure-begin');
const end = src.indexOf('// @pure-end');
if (begin < 0 || end < 0) throw new Error('the pure block of code.js was not found');
const zone = src.slice(src.indexOf('\n', begin) + 1, src.lastIndexOf('\n', end));

const api = vm.runInNewContext(
  `${zone}\n;({ HARMLESS, GIT_DANGER, DANGER_WORDS, commandParts, commandHead, classifyCommand, autoVerdict })`,
  {},
  { timeout: 5000 },
);
const { commandParts, commandHead, classifyCommand, autoVerdict } = api;

const out = {};
const differences = [];

// Each case carries what this round believes the Studio answers beside the command; the answer
// that goes into the JSON is the one the Studio itself gives, and every difference between the
// two is printed — the list has to be empty before the Rust tests are wired to the JSON.
function recorded(list, run, kind) {
  return list.map((row) => {
    const args = row.slice(0, run.length);
    const expected = row.slice(run.length);
    const got = run(...args);
    if (expected.length) {
      const [verdict, reason] = expected;
      if (got.verdict !== verdict || got.reason !== reason) {
        differences.push(
          `${kind} ${JSON.stringify(args)}: expected [${verdict}, ${reason}] got [${got.verdict}, ${got.reason}]`,
        );
      }
    }
    return [...args, got.verdict, got.reason];
  });
}

// The same for the two helpers that answer with plain arrays.
function recordedPlain(list, run, kind) {
  return list.map((row) => {
    const args = row.slice(0, row.length - 1);
    const expected = row[row.length - 1];
    const got = run(...args);
    if (JSON.stringify(got) !== JSON.stringify(expected)) {
      differences.push(
        `${kind} ${JSON.stringify(args)}: expected ${JSON.stringify(expected)} got ${JSON.stringify(got)}`,
      );
    }
    return [...args, got];
  });
}

// classifyCommand: every hard rule of the page and every special head of the table
out.classify = recorded([
  ['', 'risky', 'the command is empty'],
  ['ls\ncat x', 'risky', 'it has more than one line'],
  ['ls &', 'risky', 'it starts a background job'],
  ['(ls)', 'risky', 'it uses a group, a subshell or an expansion'],
  ['{ ls; }', 'risky', 'it uses a group, a subshell or an expansion'],
  ['echo ${HOME}', 'risky', 'it uses a group, a subshell or an expansion'],
  ['echo hi > notes.txt', 'risky', 'it writes into notes.txt through a redirect'],
  ['ls > /dev/null', 'safe', 'reads, builds or tests only'],
  // the rule of a second line stands before the one of a heredoc, so this is what the page says
  ['cat <<EOF\nhi\nEOF', 'risky', 'it has more than one line'],
  ['echo hi <<EOF', 'risky', 'it uses a heredoc, which writes a file or feeds a program'],
  ['echo $(ls)', 'safe', 'reads, builds or tests only'],
  ['echo `ls -la`', 'safe', 'reads, builds or tests only'],
  ['echo $(rm x)', 'risky', 'inside the substitution: `rm` is not on the list of harmless programs'],
  ['foo bar', 'risky', '`foo` is not on the list of harmless programs'],
  ['ls -la', 'safe', 'reads, builds or tests only'],
  ['ls', 'safe', 'reads, builds or tests only'],
  ['cat a b', 'safe', 'reads, builds or tests only'],
  ['grep ln notes.txt', 'risky', '`grep` is given something that changes files or the system'],
  ['cargo test', 'safe', 'reads, builds or tests only'],
  ['cargo build --release', 'safe', 'reads, builds or tests only'],
  ['cargo publish', 'risky', '`cargo publish` is not on the list of harmless subcommands'],
  ['cargo --version', 'safe', 'reads, builds or tests only'],
  ['npm run build', 'safe', 'reads, builds or tests only'],
  ['npm install', 'risky', '`npm install` is not on the list of harmless subcommands'],
  ['git status', 'safe', 'reads, builds or tests only'],
  ['git diff -- src', 'safe', 'reads, builds or tests only'],
  ['git push origin main', 'risky', '`git push` changes history or the working tree'],
  ['git -c core.pager=cat log', 'risky', '`git` is given an option before its command'],
  ['git diff --output=patch.txt', 'risky', 'a `git` option writes a file or starts a program'],
  ['git diff -o patch.txt', 'risky', 'a `git` option writes a file or starts a program'],
  ["sed -n '3,9p' notes.txt", 'safe', 'reads, builds or tests only'],
  ['sed -i s/a/b/ file', 'risky', "`sed` is only approved for `sed -n '3,9p' file`"],
  ['sed s/a/b/ file', 'risky', "`sed` is only approved for `sed -n '3,9p' file`"],
  ['sort names.txt', 'safe', 'reads, builds or tests only'],
  ['sort -o out.txt names.txt', 'risky', '`sort` would write a file or start a program'],
  ['rg foo src', 'safe', 'reads, builds or tests only'],
  ['rg --pre=cat foo src', 'risky', '`rg` would start a program'],
  ['find . -name "*.rs"', 'safe', 'reads, builds or tests only'],
  ['find . -delete', 'risky', '`find` would run a program or write or delete files'],
  ["awk '{print $1}' names.txt", 'safe', 'reads, builds or tests only'],
  ["awk 'BEGIN{system(\"ls\")}'", 'risky', '`awk` would start a program or write a file'],
  ['python3 -m pytest', 'safe', 'reads, builds or tests only'],
  ['python3 -m unittest test_x', 'safe', 'reads, builds or tests only'],
  ['python3 --version', 'safe', 'reads, builds or tests only'],
  ['python3 main.py', 'risky', '`python3` would run code that is no test'],
  ['node --check app.js', 'safe', 'reads, builds or tests only'],
  ['node app.js', 'risky', '`node` would run code that is no syntax check'],
  ['ls && grep x file', 'safe', 'reads, builds or tests only'],
  ['ls | grep x', 'safe', 'reads, builds or tests only'],
  ['cargo test || echo failed', 'safe', 'reads, builds or tests only'],
  ['FOO=1 ls -la', 'safe', 'reads, builds or tests only'],
  ['/bin/ls -la', 'safe', 'reads, builds or tests only'],
  ['rm -rf /', 'risky', '`rm` is not on the list of harmless programs'],
  // no subcommand: the sentence keeps the space the template writes, `git ` with nothing behind it
  ['git', 'risky', '`git ` is not on the list of harmless subcommands'],
  ['git log -1', 'safe', 'reads, builds or tests only'],
], (command) => classifyCommand(command), 'classifyCommand');

// autoVerdict: only a plain shell command is ever approved on its own
out.auto = recorded([
  ['mcp', 'fetch_page', 'https://example.com', 'risky', 'a mcp call is never approved on its own'],
  ['network', 'browser_open', 'https://example.com', 'risky', 'a network call is never approved on its own'],
  ['bash', 'edit_file', 'README.md', 'risky', '`edit_file` is not a plain shell command'],
  ['bash', 'bash', 'ls -la', 'safe', 'reads, builds or tests only'],
  ['bash', 'bash', 'git push', 'risky', '`git push` changes history or the working tree'],
], (category, name, subject) => autoVerdict(category, name, subject), 'autoVerdict');

// commandParts: the parts that run on their own
out.parts = recordedPlain([
  ['ls', ['ls']],
  ['ls && cat x', ['ls', 'cat x']],
  ['ls; cat x', ['ls', 'cat x']],
  ['cat a || cat b', ['cat a', 'cat b']],
  ['ls | grep x', ['ls', 'grep x']],
  ['  ls   &&   ', ['ls']],
  ['', []],
], (command) => commandParts(command), 'commandParts');

// commandHead: the program, without NAME=value pairs and without its path
out.head = recordedPlain([
  ['ls -la', ['ls', ['-la']]],
  ['/bin/ls -la', ['ls', ['-la']]],
  ['FOO=1 BAR=2 ls -la', ['ls', ['-la']]],
  ['', ['', []]],
], (part) => { const h = commandHead(part); return [h.head, h.args]; }, 'commandHead');

const target = path.join(__dirname, 'autoapprove_fixtures.json');
fs.writeFileSync(target, `${JSON.stringify(out, null, 1)}\n`);
if (differences.length) {
  console.log(`DIFF (${differences.length}):`);
  for (const line of differences) console.log(`  ${line}`);
  process.exit(1);
}
console.log(`wrote ${path.basename(target)} — no differences`);
