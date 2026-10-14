# Vulkan shaders

GLSL compute shaders, compiled to SPIR-V by `build.rs` (feature `vulkan`, needs `glslc`).

- Every `*.comp` file is compiled once, output name = file stem, unless it is named in a `variants*.list`.
- `variants<suffix>.list`: one line per output `<output-name> <source.comp> [-DNAME=VALUE ...]` (`#` starts a comment). Package 2 uses `variants_quant.list`, package 3 `variants_ops.list`.
- Files starting with `_` and `*.glsl` / `*.inc` are include files (not compiled alone). `glslc` is called with `-I` on this directory, `--target-env=vulkan1.2` and `-O`.
- The generated `vulkan_shaders.rs` (`OUT_DIR`) lists name and bytes; `src/backend/vulkan/shaders.rs` exposes `spirv(name)`.
