# Metal shaders

MSL sources (`*.metal`). `build.rs` (feature `metal`, any host) embeds them into `metal_sources.rs`; the backend compiles them at start with `newLibraryWithSource` (a precompiled `.metallib` through `xcrun metal` is optional and done by package 6 on macOS). Shared code goes to `common.metal` (listed first).
