// Does "fold, then compare with the case inverted" order the tags like `localeCompare`?
const pairs = [
  ["a", "B"], ["B", "a"], ["qwen", "Qwen"], ["Qwen", "qwen"],
  ["ab", "aB"], ["aB", "ab"], ["Qwen/Q2", "qwen/q1"], ["qwen/Q2", "Qwen/q1"],
  ["10", "9"], ["9", "10"], ["qwen-7b", "qwen-8b"], ["qwen", "qwen-7b"],
  ["model (x)", "model"], ["z/model", "a/model"], ["Zed", "apple"],
  ["Apple", "banana"], ["Banana", "apple"], ["a1", "A2"], ["a2", "A1"],
  ["Qwen/Qwen2.5-7B", "qwen/qwen2.5-7b"], ["", "a"], ["a", ""],
];

const inv = (s) =>
  [...s].map((ch) => {
    const c = ch.charCodeAt(0);
    if (c >= 97 && c <= 122) return c - 97; // a-z first
    if (c >= 65 && c <= 90) return 26 + (c - 65); // A-Z after
    return 1000 + c;
  });

const cmpInv = (a, b) => {
  const x = inv(a);
  const y = inv(b);
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    if (x[i] !== y[i]) return x[i] < y[i] ? -1 : 1;
  }
  return x.length === y.length ? 0 : x.length < y.length ? -1 : 1;
};

const rule = (a, b) => {
  const f = a.toLowerCase();
  const g = b.toLowerCase();
  if (f !== g) return f < g ? -1 : 1;
  return cmpInv(a, b);
};

let bad = 0;
for (const [a, b] of pairs) {
  const lc = a.localeCompare(b);
  const mine = rule(a, b);
  if (lc !== mine) bad++;
  console.log(
    `${JSON.stringify(a)} vs ${JSON.stringify(b)}: locale=${lc} rule=${mine} ${lc === mine ? "" : "  <-- DIFF"}`,
  );
}
console.log(bad === 0 ? "ALL MATCH" : `${bad} DIFFERENCES`);
