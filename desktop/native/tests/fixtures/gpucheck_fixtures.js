/**
 * Fixtures for the native hardware round: the pure heart of the Studio's GPU check
 * (`gpucheck.js`), loaded from the engine's own file so the ports are checked against it.
 */
"use strict";
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const RAW = fs
  .readFileSync(
    path.join(__dirname, "../../../../rust/engine/src/webui/gpucheck.js"),
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

const cases = [
  ["fallbackNotice", [null], "show: false"],
  ["fallbackNotice", [{}], "show: false"],
  ["fallbackNotice", [{ fallback: false, mode: "cpu" }], "the flag decides"],
  [
    "fallbackNotice",
    [{ fallback: true, mode: "cpu", device: "NVIDIA GPU", backend: "cuda", whole_model_on_cpu: true }],
    "the whole model on the CPU",
  ],
  ["fallbackNotice", [{ fallback: true, mode: "mostly_cpu" }], "the middle mode without a device"],
  [
    "fallbackNotice",
    [
      {
        fallback: true,
        mode: "layer_offload",
        device: "GPU",
        backend: "vulkan",
        operations_on_cpu: 12,
        operations_total: 48,
        layers_on_device: 30,
        reasons: [{ reason: "no kernel", operations: "attention" }],
        operation_fallbacks: [
          { operation: "rms_norm", reason: "unsupported", calls: 4, by_design: false },
          { operation: "rope", reason: "by design", calls: 9, by_design: true },
        ],
        headline: "Some blocks run on the processor.",
        notices: ["The card has 4 GB."],
        summary: "Half or more of the work is on the CPU.",
      },
    ],
    "counts, reasons, operations and the two texts",
  ],
  [
    "fallbackNotice",
    [{ fallback: true, mode: "other", device: "Radeon", notices: [] }],
    "a device without a backend and without counts",
  ],
  ["gpuVerdictLine", [{ device: "cuda:0", status: "ok", verdict: "the card does the work" }], "green"],
  ["gpuVerdictLine", [{ device: "cuda:0", status: "slow", verdict: "half of it" }], "a warning"],
  ["gpuVerdictLine", [{ device: "cpu", status: "error", verdict: "The check could not run." }], "red"],
  ["gpuVerdictLine", [{ device: "cuda:0", status: "weird", verdict: "?" }], "anything else is red"],
];

const out = cases.map(([fn, args, note]) => ({
  fn,
  args,
  got: context[fn](...args),
  note,
}));

fs.writeFileSync(
  path.join(__dirname, "gpucheck_fixtures.json"),
  JSON.stringify(out, null, 2),
);
console.log(`gpucheck_fixtures.json: ${out.length} cases`);
