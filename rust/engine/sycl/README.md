# Intel-GPU-Backend über SYCL / oneAPI (Cargo-Feature `sycl`)

**Prüfstufe: L0.** Der Code ist geschrieben und von Front-End und SPIR-V-Codegenerierung des DPC++-Compilers sowie von `cargo check/clippy`
akzeptiert. Er ist **nie auf einer Intel-GPU gelaufen**. Kein Satz in diesem Ordner behauptet, das Backend „laufe“ oder sei schnell; die
Zahlen in `EVALUATION.md` sind Erwartungen, keine Messungen. Besitzer: Paket 9 (`p09-sycl-intel`).

## Dateien

| Datei | Inhalt |
|---|---|
| `fllm_sycl.h` | C-Schnittstelle von `libfllmsycl` (38 Funktionen `fllm_sycl_*`, rohe USM-Zeiger); Spiegel in `src/backend/sycl/ffi.rs` |
| `common.hpp` | Geräte-/Stream-Typen, Fehlerbehandlung, Weitergabe der Startfunktionen |
| `device.cpp` | Geräteaufzählung (Level Zero zuerst), XMX-Erkennung (`matrix_combinations`), USM-Speicher, In-Order-Queues, Staging-Arena für kleine Host-Arrays |
| `weights.cpp` | Gerätelayouts der Gewichte und Umpacken der GGUF-Bytes (Host, parallel); `fllm_sycl_pack_weight` |
| `formats.hpp` | Gerätedecoder je Format (`F::decode`, `F::pos`); eine Einheit = 8–32 Werte pro Work-Item |
| `linear_kernels.hpp`, `linear_small.cpp`, `linear_tiled.cpp`, `linear_xmx.cpp`, `linear_api.cpp` | Linear-Kernel: Sub-Group-Matvec (Batch ≤ 8, Decode, auch MoE), Shared-Memory-Kachel in FP32 (ab Batch 9), `joint_matrix`/XMX in fp16 (ab Batch 16 auf XMX-Geräten) |
| `attention.cpp` | Paged Attention (f16- und q8_0-Cache, GQA, Softcap, Fenster/Chunk, Sinks, ALiBi), Split-KV in zwei Stufen |
| `elementwise.cpp` | RMSNorm, LayerNorm, RoPE, SwiGLU/GeGLU/OAI-SwiGLU, Residual, KV-Schreiben, Argmax, MoE-Routing/-Combine, Gather/Scatter, Q/Gate-Split |
| `build_sycl.sh` | Baut `libfllmsycl.a` ohne cargo (Übersetzen, `-fsycl-link`, `ar`) |
| `build_rs.patch` | Fertiger Patch für `rust/engine/build.rs` (Funktion `sycl_kernels`); gehört dem Integrator, siehe unten |
| `tests/decode_check.cpp`, `tests/run_decode_check.sh` | Host-Test (kein Gerät): packt zufällige GGUF-Blöcke und vergleicht die Gerätedecoder mit einer Referenz nach dem CPU-Decoder des Engines |
| `EVALUATION.md` | Bewertung Vulkan gegen SYCL für Intel-GPUs, Empfehlung, Messplan |

## Formate

Nativ gelesen (15 Decoder, 16 GGUF-Typen): F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS
(beim Umpacken zu IQ4_NL-Blöcken), MXFP4. Alles andere wird beim Hochladen auf dem Host umgewandelt und meldet es einmalig:
FP8, NVFP4 und die 4-Bit-Gruppenformate (Q4Z/Q4ZP, AWQ/GPTQ) werden zu f16, die übrigen Formate mit 256 breiten Zeilen (IQ1/IQ2/IQ3,
Q8_K ...) zu Q5_K, der Rest zu f32 (wie im CUDA-Backend). **Nicht** vorhanden: DeltaNet (`delta_layer`), `host_linear`, CUDA-Graphen,
`linear_f16_out`-Familie, Device-Sampling (läuft auf der CPU).

Gerätelayout (alles 16-Byte-ausgerichtet, damit die Kernel 8/16-Byte-Blöcke laden): Skalen und Quants einer Zeile getrennt
(`[d][m][qh][qs]`, Blockzahl auf Vielfache von 8 aufgefüllt), Q6_K 224, Q2_K 96, Q3_K 112 Byte je Superblock, Float-Zeilen auf 8
Werte aufgefüllt. Siehe Kopf von `weights.cpp`.

## Bauen

Voraussetzung: oneAPI-DPC++-Compiler (`icpx`; Windows `icx`) mit Level-Zero-Laufzeit. `cargo build --release --features sycl` findet den
Compiler über `FLLM_SYCL_CXX`, `ONEAPI_ROOT`, `PATH` oder `/opt/intel/oneapi`. Ohne Compiler **warnt** `build.rs` nur: dann gibt es kein
`libfllmsycl`, `ffi.rs` benutzt Rust-Stubs, die Crate baut, `--device sycl` meldet „kein SYCL-Laufzeit im Build“ und `fllm-rs devices`
zeigt kein SYCL-Gerät. `FLLM_SYCL_AOT=acm-g10,pvc,bmg-g21` übersetzt vorab für Intel-GPUs (braucht `ocloc`; Standard: SPIR-V, der Treiber
übersetzt beim ersten Aufruf), `FLLM_SYCL_EXTRA_FLAGS` hängt Flags an, `FLLM_SKIP_GPU_KERNELS=1` überspringt den Bau.
Wichtig: Die Bibliothek wird **als Ganzes** gelinkt (`static:+whole-archive`), denn das vorgelinkte Geräteobjekt enthält nur einen
Konstruktor, den ein normaler Archiv-Link verwerfen würde (dann fehlten die Kernel zur Laufzeit).

Der Patch für `build.rs` ist noch nicht eingespielt (gemeinsame Datei): `git apply rust/engine/sycl/build_rs.patch`. Ohne ihn setzt
niemand die cfg `fllm_sycl_native`; das Backend baut dann immer als Stub.

## Laufzeit

`libsycl` (aus dem oneAPI-Verzeichnis, `RUNPATH` wird gesetzt), Level-Zero-Lader (`libze_loader.so.1`, im Linux-Paket `libze1`) und der
Intel-Compute-Runtime-Treiber. `ONEAPI_DEVICE_SELECTOR=level_zero:0` wählt wie bei jedem SYCL-Programm; `FLLM_SYCL_ALL_BACKENDS=1` listet auch
die OpenCL-Duplikate. Einzelallokationen sind auf Level Zero standardmäßig auf ca. 4 GiB begrenzt: für größere Gewichte oder KV-Pools
`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` setzen. `FLLM_SYCL_EXACT=1` schaltet die fp16-Matrixengine-Kernel ab (alles in FP32).

## Prüfen

* Host-Test der Decoder (kein Gerät, ca. 30 s): `tests/run_decode_check.sh` (`CXX` und `LD_LIBRARY_PATH` auf den DPC++-Compiler setzen).
* Frontend + SPIR-V ohne Gerät: `~/fllm-tools/syclcheck.sh rust/engine/sycl/<datei>.cpp` mit `SYCLCHECK_INC=-I<dieser Ordner>`.
* Auf einer Intel-GPU: `fllm-rs selftest-gpu --device sycl --json out.json` (Batch ≤ 5, FP32-exakt), danach
  `cargo test --release -p fllm-engine --features sycl --lib linear_matches_cpu` (deckt alle Batch-Pfade inkl. XMX ab), dann ein echtes
  Modell gegen den Greedy-Text der CPU (L2) und der Messplan aus `EVALUATION.md` (L3).
