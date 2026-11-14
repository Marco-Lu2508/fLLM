/**
 * Fixtures for the AMD NPU card: the pure heart of the Studio's `amdnpu.js` (the `@pure` zone),
 * loaded from the engine's own file so the ports are checked against it.
 */
"use strict";
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const RAW = fs
  .readFileSync(
    path.join(__dirname, "../../../../rust/engine/src/webui/amdnpu.js"),
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

const model = (tag, installed) => ({ tag, installed });
const serving = (state, extra = {}) => ({ state, ...extra });

const cases = [
  ["amdNpuText", [{ state: "flm_missing" }, serving("idle"), []], "FastFlowLM is missing"],
  ["amdNpuText", [{ state: "needs_attention" }, serving("idle"), ["the driver is missing", "no device"]], "two problems, joined"],
  ["amdNpuText", [{ state: "needs_attention" }, serving("idle"), []], "no problems: the Studio's own words"],
  ["amdNpuText", [{ state: "ready" }, serving("running", { model: "qwen-7b-q4" }), []], "running, the model named twice"],
  ["amdNpuText", [{ state: "ready" }, serving("starting", { model: "qwen-7b-q4" }), []], "starting"],
  ["amdNpuText", [{ state: "ready" }, serving("failed", { model: "qwen-7b-q4" }), []], "did not start"],
  ["amdNpuText", [{ state: "ready" }, serving("idle"), []], "ready and waiting"],
  ["amdNpuText", [{ state: "flm_missing" }, serving("running", { model: "m" }), []], "the state of the setup speaks first"],

  ["amdNpuNativeText", [null], "no native kernels in the build"],
  ["amdNpuNativeText", [{}], "no compiled flag"],
  ["amdNpuNativeText", [{ compiled: false, available: true }], "not compiled: nothing to say"],
  ["amdNpuNativeText", [{ compiled: true, available: true }], "ready"],
  ["amdNpuNativeText", [{ compiled: true, available: false }], "not available"],
  ["amdNpuNativeText", [{ compiled: true, available: false, reason: "no NPU driver" }], "not available, with the reason"],

  ["amdNpuModel", [null, ""], "nothing was asked"],
  ["amdNpuModel", [{ state: "no_npu" }, ""], "this machine has none"],
  ["amdNpuModel", [{ state: "no_npu", models: [model("m", true)] }, "m"], "none, even with a list"],
  ["amdNpuModel", [{ state: "flm_missing", flm: { install: { text: "Install FastFlowLM", url: "https://example.com/flm" } } }, ""], "the card that offers the install"],
  ["amdNpuModel", [{ state: "ready", flm: { version: "0.4.1" }, models: [model("a", true), model("b", false)] }, ""], "the first that is on disk is chosen"],
  ["amdNpuModel", [{ state: "ready", models: [model("a", true), model("b", true)] }, "b"], "what was chosen stays"],
  ["amdNpuModel", [{ state: "ready", models: [model("a", true), model("b", true)] }, "gone"], "a choice the list does not have"],
  ["amdNpuModel", [{ state: "ready", models: [model("z", false), model("a", true)] }, ""], "installed first, then by tag"],
  ["amdNpuModel", [{ state: "ready", models: [{ tag: "z" }, { tag: "a" }] }, ""], "a list without the flag sorts by tag"],
  ["amdNpuModel", [{ state: "ready", models: [model("Qwen/Q2", false), model("qwen/q1", false)] }, ""], "the case decides only when the tags are otherwise equal"],
  ["amdNpuModel", [{ state: "ready", models: [model("qwen", false), model("Qwen", false)] }, ""], "lower case before upper, as localeCompare does"],
  ["amdNpuModel", [{ state: "ready" }, ""], "no models at all"],
  ["amdNpuModel", [{ state: "ready", serving: serving("running", { model: "qwen" }), models: [model("qwen", true)] }, ""], "the running model is the one in the list"],
  ["amdNpuModel", [{ state: "ready", serving: serving("running", { model: "qwen" }), models: [model("other", true)] }, ""], "the running model is not in the list"],
  ["amdNpuModel", [{ state: "ready", serving: serving("starting", { model: "qwen" }), models: [model("qwen", true)] }, ""], "starting: no start, but a stop"],
  ["amdNpuModel", [{ state: "ready", serving: serving("failed", { model: "qwen", error: "it died" }), models: [model("qwen", true)] }, ""], "a failed start says so"],
  ["amdNpuModel", [{ state: "ready", pull: { busy: true, model: "qwen", line: "12%" }, models: [model("qwen", false)] }, "qwen"], "downloading: the line and the cancel"],
  ["amdNpuModel", [{ state: "ready", pull: { busy: false, model: "qwen", line: "done" }, models: [model("qwen", true)] }, "qwen"], "the download is done"],
  ["amdNpuModel", [{ state: "ready", pull: { error: "cancelled" }, error: "kept from before", models: [model("qwen", true)] }, "qwen"], "a cancelled download says nothing"],
  ["amdNpuModel", [{ state: "ready", pull: { error: "the network is gone" }, models: [model("qwen", true)] }, "qwen"], "a failed download"],
  ["amdNpuModel", [{ state: "needs_attention", validation: { problems: ["the NPU is busy"] } }, ""], "the problems of the setup"],
  ["amdNpuModel", [{ state: "ready", error: "FastFlowLM said no", models: [model("qwen", true)] }, "qwen"], "an error of the card itself"],
  ["amdNpuModel", [{
    state: "ready",
    flm: { version: "0.4.1", install: { text: "Install", url: "https://example.com" } },
    hardware: [{ name: "Ryzen AI 9 HX 370" }, { name: "XDNA", problem: "busy" }],
    native: { compiled: true, available: false, reason: "no toolkit" },
    models: [model("qwen-7b", true), model("llama-8b", false)],
  }, "llama-8b"], "the whole card at once"],
];

const out = cases.map(([fn, args, note]) => ({
  fn,
  args,
  got: context[fn](...args),
  note,
}));

fs.writeFileSync(
  path.join(__dirname, "amdnpu_fixtures.json"),
  JSON.stringify(out, null, 2),
);
console.log(`amdnpu_fixtures.json: ${out.length} cases`);
