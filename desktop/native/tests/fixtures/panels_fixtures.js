/**
 * Fixtures for the native start plan: the pure heart of the Studio's hero block (`panels.js`),
 * loaded from the engine's own file so the ports are checked against the original.
 */
"use strict";
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const RAW = fs
  .readFileSync(
    path.join(__dirname, "../../../../rust/engine/src/webui/panels.js"),
    "utf8",
  )
  .replace(/\r/g, "");

function between(s, a, b) {
  const start = s.indexOf(a);
  const end = s.indexOf(b, start);
  if (start < 0 || end < 0) throw new Error("pure zone not found");
  return s.slice(start, end + b.length);
}

const context = {};
vm.createContext(context);
// @ts-expect-error JavaScript
vm.runInContext(between(RAW, "// @pure-begin", "// @pure-end"), context);

const hub = (over = {}) =>
  Object.assign(
    {
      available_memory_bytes: 0,
      catalog: [],
      recommended: [],
      cached: [],
      local: [],
      found: [],
    },
    over,
  );

const cat = (id, ref) => ({ id, installed: ref ? { ref } : undefined });

const cases = [
  ["modelFileTitle", ["/models/qwen/Qwen3-4B-Q4_K_M.gguf"], "Qwen3-4B-Q4_K_M"],
  ["modelFileTitle", ["C:\\Users\\a\\qwen.gguf"], "qwen"],
  ["modelFileTitle", ["qwen3.5:4b"], "qwen3.5:4b"],
  ["modelFileTitle", ["model.GGUF"], "model", "an upper-case extension goes too"],
  ["modelFileTitle", [""], ""],
  ["modelFileTitle", [null], ""],

  [
    "startPlan",
    [hub(), ""],
    { mode: "download", primary: null, others: [], download: null },
    "nothing anywhere: the plan is the download",
  ],
  [
    "startPlan",
    [hub({ catalog: [cat("qwen3-4b")], recommended: ["qwen3-4b"] }), ""],
    { mode: "download", primary: null, others: [], download: { id: "qwen3-4b" } },
    "the recommended model from the catalog",
  ],
  [
    "startPlan",
    [
      hub({
        cached: [{ name: "used.gguf", path: "/c/used.gguf", size_bytes: 1_000_000_000 }],
        catalog: [cat("used", "/c/used.gguf")],
        recommended: ["qwen"],
      }),
      "/c/used.gguf",
    ],
    {
      mode: "load",
      primary: { ref: "used.gguf", title: "used", size: 1_000_000_000, where: "downloaded", last: true, fits: true },
      others: [],
      download: null,
    },
    "the model used last is the first one",
  ],
  [
    "startPlan",
    [
      hub({
        memory: 3_000_000_000,
        local: [
          { name: "big.gguf", path: "/m/big.gguf", size_bytes: 4_000_000_000 },
          { name: "used.gguf", path: "/m/used.gguf", size_bytes: 1_000_000_000 },
        ],
        found: [{ name: "rec.gguf", path: "/o/rec.gguf", size_bytes: 2_000_000_000, source: "ollama" }],
        catalog: [cat("used", "/m/used.gguf"), cat("rec", "/o/rec.gguf")],
        recommended: ["rec"],
      }),
      "/m/used.gguf",
    ],
    {
      mode: "load",
      primary: { ref: "/m/used.gguf", title: "used", size: 1_000_000_000, where: "added", last: true, fits: true },
      others: [
        { ref: "/o/rec.gguf", title: "rec", size: 2_000_000_000, where: "ollama" },
        { ref: "/m/big.gguf", title: "big", size: 4_000_000_000, where: "added" },
      ],
      download: null,
    },
    "used first, then the recommended one, the big one that does not fit last",
  ],
  [
    "startPlan",
    [
      hub({
        local: [
          { name: "a.gguf", path: "/M/A.GGUF", size_bytes: 1_000_000_000 },
          { name: "a.gguf", path: "/m/a.gguf", size_bytes: 999 },
        ],
      }),
      "",
    ],
    {
      mode: "load",
      primary: { ref: "/M/A.GGUF", title: "a", size: 1_000_000_000, where: "added", last: false, fits: true },
      others: [],
      download: null,
    },
    "the same reference counts once, in lower case (the Set of the Studio)",
  ],
  [
    "startPlan",
    [
      hub({
        cached: [{ name: "only-name", path: "/elsewhere", size_bytes: 10 }],
        catalog: [cat("id-elsewhere", "/elsewhere")],
        recommended: [],
      }),
      "id-elsewhere",
    ],
    {
      mode: "load",
      primary: { ref: "only-name", title: "elsewhere", size: 10, where: "downloaded", last: false, fits: true },
      others: [],
      download: null,
    },
    "a model only named by its hub id is not the one used last (its file name decides beside it)",
  ],
];

const out = cases.map(([fn, args, expected, note]) => {
  const got = context[fn](...args);
  const comparable =
    fn === "startPlan"
      ? {
          mode: got.mode,
          primary: got.primary && { ref: got.primary.ref, title: got.primary.title, size: got.primary.size, where: got.primary.where, last: got.primary.last, fits: got.primary.fits },
          others: got.others.map((o) => ({ ref: o.ref, title: o.title, size: o.size, where: o.where, last: o.last, fits: o.fits })),
          download: got.download && { id: got.download.id },
        }
      : got;
  return { fn, args, expected, got: comparable, note: note || "" };
});

fs.writeFileSync(
  path.join(__dirname, "panels_fixtures.json"),
  JSON.stringify(out, null, 2),
);
console.log(`panels_fixtures.json: ${out.length} cases`);
