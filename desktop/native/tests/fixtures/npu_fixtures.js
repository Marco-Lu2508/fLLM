/**
 * Fixtures for the native NPU and device round: the pure heart of the Studio's `npu.js`
 * (the `@pure` zone), loaded from the engine's own file so the ports are checked against it.
 */
"use strict";
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const RAW = fs
  .readFileSync(
    path.join(__dirname, "../../../../rust/engine/src/webui/npu.js"),
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

const dev = (id, name, backend, api, bytes, available = true) => ({
  id, name, backend, api,
  dev_type: "",
  total_memory_bytes: bytes,
  available,
  has_id: !!id,
});

const GPU = dev("cuda:0", "GeForce RTX 4060(R)", "cuda", "cuda", 8_000_000_000);
const GPU_VULKAN = dev("vulkan:1", "GeForce RTX 4060", "vulkan", "vulkan", 8_100_000_000);
const GPU_WGPU = dev("vulkan:2", "GeForce RTX 4060", "vulkan", "wgpu", 8_100_000_000);
const CPU = dev("cpu0", "Intel Core i5", "cpu", "cpu", 0);
const NPU = dev("npu0", "Intel AI Boost", "npu", "npu", 0);

const cases = [
  ["npuMegabytes", [0], "zero"],
  ["npuMegabytes", [1_500_000_000], "a billion and a half"],
  ["npuMegabytes", [null], "nothing"],

  ["npuProgressModel", [null], "no progress at all"],
  ["npuProgressModel", [{ busy: false }], "not busy: no bar"],
  ["npuProgressModel", [{ busy: true, phase: "downloading", done_bytes: 500_000_000, total_bytes: 2_000_000_000 }], "a quarter downloaded"],
  ["npuProgressModel", [{ busy: true, phase: "verifying" }], "no totals: no detail"],
  ["npuProgressModel", [{ busy: true, phase: "elsewhere" }], "an unknown phase says Working"],
  ["npuProgressModel", [{ busy: true, phase: "unpacking", done_bytes: 9, total_bytes: 5 }], "more done than total: a hundred"],

  ["npuCardModel", [null], "checking"],
  ["npuCardModel", [{ state: "not_in_build" }], "no support in this build"],
  ["npuCardModel", [{ state: "no_npu" }], "no NPU was found"],
  ["npuCardModel", [{ state: "unsupported_hardware" }], "detected but unsupported"],
  ["npuCardModel", [{ state: "setup_unavailable" }], "no automatic setup"],
  ["npuCardModel", [{ state: "no_device" }], "runtime without a device"],
  ["npuCardModel", [{ state: "ready" }], "ready without a version"],
  ["npuCardModel", [{ state: "installing" }], "installing"],
  ["npuCardModel", [{ state: "other_state", state_text: "Some text." }], "an unknown state uses state_text"],
  ["npuCardModel", [{ state: "runtime_missing", setup: { download_bytes: 140_000_000, pinned_version: "2025.0.1" } }], "the runtime to install"],
  ["npuCardModel", [{ state: "runtime_missing", setup: { download_bytes: 140_000_000 }, progress: { busy: true, phase: "failed" } }], "failed: Try again"],
  ["npuCardModel", [{
    state: "ready",
    runtime: { version: "2025.1.0", managed: true },
    devices: [{ full_name: "Intel AI Boost", architecture: "Arrow Lake" }],
    offload_line: "reading on the NPU",
    quantizations: { eight_bit: ["q8_0"], float16: ["f16"], exact: [], converted_int4: ["int4"] },
  }], "ready with version, device, offload and formats"],
  ["npuCardModel", [{ state: "unsupported_hardware", hardware: [{ name: "XDNA", vendor: "amd" }] }], "an AMD machine points at its card"],
  ["npuCardModel", [{
    state: "no_npu",
    hardware: [
      { name: "Intel AI Boost", vendor: "intel", supported: true },
      { name: "Ryzen AI", vendor: "amd", supported: false, problem: "needs FastFlowLM" },
      { name: "Something", vendor: "qualcomm", supported: false },
    ],
    last_error: "the driver said no",
  }], "three hardware rows and an error"],
  ["npuCardModel", [{
    state: "no_device",
    runtime: { error: "cannot open the device" },
    progress: { busy: true, phase: "downloading", done_bytes: 1, total_bytes: 4, error: "lost the network" },
  }], "the failed phase has the first word on the error"],

  ["deviceKey", [{ name: "GeForce RTX™ 4060(R)" }], "marks in parentheses go"],
  ["deviceKey", [{ name: "  spaced   name " }], "spacing folds"],
  ["deviceKey", [{ name: "Radeon (TM) RX 7800" }], "the TM mark goes"],
  ["deviceKey", [null], "nothing"],

  ["deviceRank", [GPU], "the vendor's own interface first"],
  ["deviceRank", [GPU_VULKAN], "the native API second"],
  ["deviceRank", [GPU_WGPU], "the generic last"],

  ["mergeDevices", [[CPU, GPU, GPU_VULKAN, GPU_WGPU, NPU], ""], "five entries become three"],
  ["mergeDevices", [[dev("cuda:0", "RTX 4090", "cuda", "cuda", 24_000_000_000), dev("cuda:1", "RTX 4090", "cuda", "cuda", 24_000_000_000)], ""], "two identical cards stay two"],
  ["mergeDevices", [[dev("a", "Same Card", "cuda", "cuda", 10_000_000_000), dev("b", "Same Card", "vulkan", "vulkan", 2_000_000_000)], ""], "under half the size: another device"],
  ["mergeDevices", [[], ""], "nothing"],
  ["mergeDevices", [[GPU, GPU_VULKAN], "vulkan:1"], "the wanted one stands for its group"],

  ["deviceBackendLabel", [CPU], "the processor"],
  ["deviceBackendLabel", [GPU], "NVIDIA CUDA"],
  ["deviceBackendLabel", [GPU_WGPU], "the generic gets its word"],
  ["deviceBackendLabel", [dev("x", "y", "weird", "weird", 0)], "an unknown backend is itself"],

  // deviceChoices reads the hardware report as `hw.devices` (the object of GET /api/studio/hardware)
  ["deviceChoices", [{ devices: [GPU, GPU_VULKAN, CPU, dev("ghost", "No Memory GPU", "vulkan", "vulkan", 0)] }, null, ""], "the list of the load options"],
  ["deviceChoices", [{ devices: [GPU] }, { state: "ready" }, ""], "the NPU when it is ready"],
  ["deviceChoices", [{ devices: [GPU] }, null, "not-here"], "the current one is kept"],
  ["deviceChoices", [{ devices: [GPU] }, null, "npu1"], "an unknown current with an npu name warns"],

  // runningDevice takes (loadDevice, hw, tier): auto_choice and the devices are hw's own fields
  ["runningDevice", ["cpu", { devices: [GPU], auto_choice: "cuda:0" }, "avx2-fma"], "the processor with its tier"],
  ["runningDevice", ["auto", { devices: [GPU], auto_choice: "cuda:0" }, ""], "the automatic choice"],
  ["runningDevice", ["cuda:0", { devices: [GPU], auto_choice: "" }, ""], "chosen in the load options"],
  ["runningDevice", ["auto", {}, ""], "without a report only the processor part"],

  // deviceMemory reads free_memory_bytes from the device (its one argument)
  ["deviceMemory", [{ ...GPU, free_memory_bytes: 9_000_000_000 }], "total under free: nothing used"],
  ["deviceMemory", [{ ...GPU, free_memory_bytes: 2_000_000_000 }], "three quarters full: a warning"],
  ["deviceMemory", [{ ...GPU, free_memory_bytes: 500_000_000 }], "ninety-four percent: high"],
  ["deviceMemory", [{ ...GPU, free_memory_bytes: 8_000_000_000 }], "all free: nothing used"],
  ["deviceMemory", [CPU], "a device without a size has no bar"],

  // deviceRole takes (d, loadDevice, autoChoice, modelLoaded): the device carries its members
  ["deviceRole", [{ id: "cuda:0", members: [{ id: "cuda:0", backend: "cuda", api: "cuda" }, { id: "vulkan:1", backend: "vulkan", api: "vulkan" }] }, "auto", "cuda:0", true], "the running one"],
  ["deviceRole", [{ id: "cuda:0", members: [{ id: "cuda:0", backend: "cuda", api: "cuda" }] }, "auto", "cuda:0", false], "the automatic choice without a model"],
  ["deviceRole", [{ id: "cuda:0", members: [{ id: "cuda:0", backend: "cuda", api: "cuda" }] }, "vulkan:9", "cuda:0", true], "another device: nothing"],
  ["deviceRole", [{ id: "cpu0" }, "auto", "cpu", true], "the processor when the choice is the processor"],
];

const out = cases.map(([fn, args, note]) => ({
  fn,
  args,
  got: context[fn](...args),
  note,
}));

fs.writeFileSync(
  path.join(__dirname, "npu_fixtures.json"),
  JSON.stringify(out, null, 2),
);
console.log(`npu_fixtures.json: ${out.length} cases`);
