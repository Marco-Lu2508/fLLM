# Backend-Matrix: Hardware × Format × Prüfstufe

Stand: Berichte der Mega-Runde (Welle 1 und 2), Branches auf Basis `247fb37`. Quelle jeder Zelle ist der Bericht des Pakets
(Berichte der einzelnen Pakete, Kürzel `p02` usw.) oder die bestehende Doku (`docs/gpu-backends.md`,
`docs/cpu-kernels.md`, `README.md`). **Alles Neue ist L0: nichts davon ist auf der Zielhardware gelaufen.** Wo diese Seite
„Backend X kann Y" schreibt, heißt das „der Code dafür existiert und wurde übersetzt bzw. syntaxgeprüft", nie „es läuft".

Anleitung zum Prüfen auf fremder Hardware: [backends-verify.md](backends-verify.md). Gerätewahl, `--device auto` und Selbsttest:
`docs/device-selection.md` (Paket p15, noch nicht auf `master`).

## 1. Prüfstufen und Zellen-Kürzel

| Stufe | Bedeutung | Wie man sie erreicht |
| --- | --- | --- |
| **L0** | übersetzt und/oder syntaxgeprüft; ggf. zusätzlich Unit-Tests ohne Hardware oder CPU-Emulation des Kerneltexts (bleibt L0) | `cargo check/clippy`, `xcheck.sh`, `hipcheck.sh`, `syclcheck.sh`, `mslcheck.sh`, `glslc` + `spirv-val` |
| **L1** | Selbsttest auf echter Hardware bestanden | `fllm-rs selftest-gpu --device <name> --json out.json` mit `failed = 0` (bzw. `selftest-cpu`, die `#[ignore]`-Hardwaretests des Pakets) |
| **L2** | Greedy-Text (Temperatur 0) eines Modells ist auf dieser Hardware identisch mit der CPU-Referenz | Textvergleich aus [backends-verify.md](backends-verify.md) |
| **L3** | Geschwindigkeit gemessen und mit einer Vergleichsimplementierung auf derselben Hardware verglichen | `fllm-rs bench` plus Vergleichslauf |

Der Cross-Check (`xcheck.sh`, t00) beweist nur die Übersetzbarkeit des Rust-Codes, nicht Linkbarkeit oder Lauf.

| Kürzel | Bedeutung |
| --- | --- |
| **B** | Bestand: Code vor dieser Runde, laut bestehender Doku auf Hardware gelaufen; in dieser Runde nicht erneut geprüft |
| **B‡** | Bestand, aber laut `docs/cpu-kernels.md` „not run" (kompiliert, Tests geschrieben, nie auf passender Hardware ausgeführt); effektiv L0 |
| **L0** | neu, native Kernel/Pfad vorhanden |
| **L0†** | neu, nur über Umwandlung beim Upload (Host dequantisiert oder quantisiert neu); kein eigener Kernel für das Format |
| **–** | nicht vorhanden; die Operation läuft auf dem Host/der CPU oder meldet `Unsupported` |
| **?** | der Bericht sagt dazu nichts; nicht geraten |

## 2. Gewichtsformate

Spalten: **L** = Q4_0 Q4_1 Q5_0 Q5_1 Q8_0; **K** = Q2_K Q3_K Q4_K Q5_K Q6_K; **IQ4** = IQ4_NL; **MX** = MXFP4; **NV** = NVFP4;
**Q4Z** = AWQ/GPTQ-4-Bit (Q4Z32/64/128, Q4ZP mit act-order); **F** = F16/BF16 (F32).
Zahlen in eckigen Klammern verweisen auf die Anmerkungen unter der Tabelle.

| Hardware | L | K | IQ4 | MX | NV | Q4Z | F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU x86 AVX2 / AVX-VNNI | B [1] | Q4_K–Q6_K: B; Q2_K, Q3_K: B‡ [1] | B‡ [1] | B‡ [1] | – [2] | – [2] | B [1] |
| CPU x86 AVX-512 (BW/VNNI/BF16) | B‡ [1] + L0 (16-Zeilen-VNNI-GEMM) [p14] | Q4_K–Q6_K: B‡; Q2_K, Q3_K, IQ4_XS: L0 [p14] | L0 [p14] | L0 [p14] | – [2] | – [2] | L0 (F32/F16/BF16, BF16 mit `vdpbf16ps`) [p14] |
| CPU x86 AMX (INT8/BF16-Tiles) | L0 (Q8_0 Q4_0 Q5_0 Q4_1 Q5_1) [p14] | L0 (Q2_K–Q6_K, IQ4_XS) [p14] | L0 [p14] | L0 [p14] | – | – | BF16: L0 (`cols % 32 == 0`); F16 nicht über AMX [p14] |
| CPU ARM NEON | B [3] | Q4_K–Q6_K: B; Q2_K, Q3_K: L0 [p13] | L0 (`tbl`) [p13] | L0 [p13] | – [2] | – [2] | L0 (F32/F16/BF16) [p13] |
| CPU ARM dotprod (`sdot`) | B [3] | wie NEON | L0 [p13] | L0 [p13] | – | – | L0 [p13] |
| CPU ARM i8mm (`smmla`, `bfdot`) | B [3] | wie NEON | L0 [p13] | L0 [p13] | – | – | L0, BF16 mit `bfdot` [p13] |
| CPU ARM SVE (opt-in `FLLM_ARM_SVE=1`) | nur Q8_0-Decode: L0 [p13]; sonst NEON-Tier | – | – | – | – | – | – |
| CUDA | B [4] | B (auch IQ1–IQ3, IQ4_XS) [4] | B [4] | B [4] | B (FP4-Kerne ab Blackwell) [4] | B (Marlin, ab sm_75) [4] | B [4] |
| wgpu (Vulkan/Metal/DX12) | B [5] | Q4_K, Q5_K, Q6_K: B; Q2_K, Q3_K: F32-Expansion beim Upload [5] | F32-Expansion [5] | F32-Expansion [5] | F32-Expansion [5] | F32-Expansion [5] | B [5] |
| Vulkan nativ | L0 [p02, p04] | L0 [p02, p04] | L0 [p02, p04] | L0 [p02, p04] | L0 [p02, p04] | L0 (Q4Z32/64/128; Q4ZP mit Vorschritt) [p02] | L0 (`lin_float`, p04) [p04] |
| Metal nativ | L0 [p06a] | L0 [p06a] | L0 [p06a] | L0 (planares Layout) [p06a] | L0 [p06a] | L0 (Q4ZP mit Vorschritt) [p06a] | L0 (F32/F16/BF16, FP8) [p06a] |
| ROCm/HIP RDNA3 (gfx11) | L0 [p08] [6] | L0 [p08] [6] | L0 [p08] [6] | L0 [p08] [6] | ? [7] | L0 (`hip_q4z.cu`, Int4-Skinny) [p08] | L0 (Skinny 1–4 Token + WMMA) [p08] |
| ROCm/HIP RDNA4 (gfx12) | L0 [6] | L0 [6] | L0 [6] | L0 [6] | ? [7] | L0 (Int4-Skinny + WMMA-gfx12) | L0 |
| ROCm/HIP CDNA (gfx908/90a/942/950) | L0 [6] | L0 [6] | L0 [6] | L0 [6] | ? [7] | L0 (eigener GEMV + MFMA-GEMM; Int4-Skinny nur RDNA3/4) | L0 (Skinny + MFMA) |
| Intel SYCL (Level Zero, Arc/Xe) | L0 [p09] | L0 [p09] | L0 [p09] | L0 [p09] | L0† (→ f16) [p09] | L0† (→ f16) [p09] | L0 [p09] |
| Intel NPU (OpenVINO) | Q4_0, Q8_0: L0 exakt (i4/i8, Gruppe 32); Q4_1, Q5_x: L0† (neu quantisiert) [p10] | L0† [p10] | L0† | L0† | L0† | L0† (Q4ZP nicht registriert) | L0 (→ FP16) [p10] |
| AMD XDNA NPU | nur Probe (ein MatMul über ONNX Runtime/VitisAI) [p11] | – | – | – | – | – | – |
| Qualcomm Hexagon NPU | nur Probe (FP16-MatMul über QNN HTP; Genie-Wrapper) [p11] | – | – | – | – | – | – |
| Apple Neural Engine | nur Probe (Linear über Core ML) [p11] | – | – | – | – | – | – |
| Google TPU (PJRT/StableHLO) | L0† (Host-Dekodierung, Upload als bf16/f32/int8) [p12] | L0† | L0† | L0† | L0† | L0† (Q4Z/Q4ZP/FP8M über `compute::dequantize`) | L0 [p12] |

Anmerkungen:

1. **x86-CPU:** `docs/cpu-kernels.md` („What is tested") führt AVX2-/AVX-VNNI-Projektionen als „run on hardware by us"; die Kerne für IQ4_NL, MXFP4
   und Q2_K/Q3_K/IQ4_XS stehen dort als „unit tests written, not run here" (daher B‡), die AVX-512-BW/VNNI-Kerne als „not run" (B‡). Die
   neuen Kerne aus p14 sind L0; p14 hat auf einer CPU-Maschine (nur AVX2/AVX-VNNI) die Unit-Tests ausgeführt (247 bestanden mit Feature `avx512`) und
   `selftest-cpu` über die dort vorhandenen Tiers; AVX-512 und AMX selbst sind nie gelaufen. Das AMX-Modell (`EmuTiles`) stammt vom selben Autor wie die Treiber.
2. NVFP4 und Q4Z haben auf der CPU keinen SIMD-Kern (Zeilen-Dekodierung in `compute.rs`, `docs/cpu-kernels.md`).
3. **ARM-CPU:** `docs/cpu-kernels.md`: „NEON `sdot` and `smmla` paths: compiled for aarch64-unknown-linux-gnu and run on hardware by us" (Q8_0, Q4_0/1, Q5_0/1,
   Q4_K, Q5_K, Q6_K). Vor p13 hatten Q2_K, Q3_K, IQ4_NL, IQ4_XS, MXFP4 und die Float-Formate auf ARM keinen Kern (f32-Zeilen-Dekodierung bzw. skalar).
4. **CUDA:** `docs/gpu-backends.md` führt die nativen Formate (`F32 F16 BF16 Q4_0 … Q6_K IQ1_S … IQ3_S IQ4_NL IQ4_XS MXFP4 NVFP4`); Q4Z über die
   Marlin-Kerne aus vLLM ab Compute-Capability 7.5 (`rust/engine/src/backend/cuda.rs`). Die Messungen im `README.md` stammen von einer Blackwell-Karte;
   `docs/device-selection.md` (p15) nennt Blackwell, Ada und Turing als die Hardware, auf der CUDA gelaufen ist.
5. **wgpu:** native WGSL-Kernel für `F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K`, alles andere wird beim Upload auf F32 expandiert
   (`docs/gpu-backends.md` §1). Physisch getestet ist nur Intel Arrow Lake über Vulkan; NVIDIA, AMD, Apple/Metal und DX12 sind nicht validiert
   (`docs/inference-features.md`).
6. **HIP:** das Backend ist `CudaBackend` mit der HIP-Übersetzung der CUDA-Kernel (`hip_kernels()` in `build.rs`, 10 Quellen plus die neuen `hip/*.cu`).
   p08 berichtet `hip_gemm.cu` (Decode-in-LDS-GEMM) „für alle GGUF-Kinds"; welche Formate im Einzelnen über welchen Pfad laufen, ist nicht Einzelfall für
   Einzelfall berichtet. Geprüft wurde Syntax und Codegen bis GCN-Assembler für gfx908, gfx90a, gfx942, gfx950, gfx1030, gfx1100, gfx1151, gfx1201 — keine Ausführung.
   Q4Z auf AMD verlangt `rows % 64 == 0` und `cols % 64 == 0` und den `cuda.rs`-Patch aus dem p08-Bericht (sonst Expansion nach F32).
7. HIP-NVFP4: der Build läuft mit `-DFLLM_NO_FP4`; „NVFP4/FP8-Marlin bleiben Stubs" (p08). Ob NVFP4-Gewichte auf AMD über den allgemeinen Pfad laufen, steht nicht im Bericht.

## 3. Operationen

Spalten: **Attention** = `attention_paged` und `kv_cache_write` (KV-Cache f16 und q8_0 hinter `DeviceKvCache`); **MoE** = Routing, expert-indiziertes Matvec,
Combine, Gather/Scatter; **DeltaNet** = Gated DeltaNet (`delta_layer`, Qwen3.5/3.6); **Mamba** = Mamba-1/-2 (`mamba_layer`, derzeit nirgends Teil des `Stream`-Traits).

| Hardware | Attention | MoE | DeltaNet | Mamba |
| --- | --- | --- | --- | --- |
| CPU x86 AVX2 / AVX-VNNI | B (f16 und q8_0) | B (fusioniertes `linear_experts`, nur x86) [p13] | B (`cpu_delta.rs`) | B (Host-Pfad `mamba_forward`) |
| CPU x86 AVX-512 | B‡ (Decode-Attention 512 Bit) | L0 (Prompts je Experte über die neuen 512-Bit-Kerne) [p14] | L0 (512-Bit-Zustandsupdate, bitgleich zu AVX2) [p14] | – (kein neuer Kern) |
| CPU x86 AMX | – | – (MoE-Prompts nicht über AMX) [p14] | – | – |
| CPU ARM NEON / dotprod / i8mm | B (`sdot` + NEON) | Einzelpfad; kein fusioniertes `linear_experts` [p13] | L0 (NEON-Schritt, bitgleich zum Skalar) [p13] | B (Host-Pfad) |
| CPU ARM SVE | – | – | – | – |
| CUDA | B (Kopfbreite ≤ 512, breitere → CPU) | Experten auf dem Gerät, `moe_route`/`moe_expert_gemm` auf dem Host (`supports_moe: false`) | vorhanden (`deltanet.cu`, `supports_delta`); Stufe nicht berichtet | Host |
| wgpu | B (Kopfbreite ≤ 512) | Host-Fallback | Host (`docs/gpu-backends.md` §2) | Host |
| Vulkan nativ | L0 (Decode mit Split-KV, Prompt mit Kacheln; Kopfbreite ≤ 512) [p03] | L0 (`moe_topk`, `moe_linear` in 12 Formaten, Combine, Gather/Scatter; `moe_route` auf der CPU; `moe_expert_gemm` = `Unsupported`) [p03, p04] | L0 (HK 32/64/96/128, HV ≤ 128) [p03, p04] | Shader und Pläne L0 [p03]; `VulkanStream::mamba2_layer` fertig, aber ohne Trait-Hook ungenutzt [p04] |
| Metal nativ | L0 (Decode Split-KV, Prompt mit `simdgroup_matrix`; Prefill-Kopfbreite ≤ 256) [p06b] | L0 (`moe_linear` in 10 Formaten, Matvec ohne gruppierten GEMM) [p06c, p07] | L0 [p06c, p07] | Kernel und Pläne L0, keine Stream-Methode [p06c, p07] |
| ROCm/HIP (RDNA3/RDNA4/CDNA) | L0 (CUDA-Attention-Kerne unter hipcc; DPP-Reduktion in `attention4.cu`; vLLM-Paged-Attention nicht portiert) [p08] | CUDA-Pfad unter hipcc; nicht einzeln berichtet | – (`deltanet.cu` wird für AMD nicht gebaut, `fllm_delta_supported` liefert 0, Host-Pfad) | Host |
| Intel SYCL | L0 (skalar FP32, Split-KV; Kopfbreite ≤ 512) [p09] | L0 (`moe_linear`, Top-k, Combine, Gather/Scatter; `moe_route` auf der CPU) [p09] | – (`supports_delta` = false) [p09] | – |
| Intel NPU (OpenVINO) | CPU-Referenz-Stream (Hybrid-Backend, nur dichte Modelle) [p10] | CPU | CPU | CPU |
| AMD XDNA / Qualcomm Hexagon / Apple ANE | – (nur Probe) [p11] | – | – | – |
| Google TPU | L0 (StableHLO, nur f16-KV, **kein q8_0**) [p12] | L0 (Top-k, Combine, Gather/Scatter); `moe_linear`, `upload_pointer_table`, `moe_expert_gemm` = `Unsupported` [p12] | – (Host) [p12] | – |

Alle Spalten gelten als L0, sofern nicht „B" dasteht. Die Metal-Kernel sind zusätzlich als C++ auf der CPU emuliert worden (160 + 139 + 2436 Fälle gegen das CPU-Backend bzw. die f64-Referenz,
[p06a, p06b, p06c]), die Vulkan-Shader-Teile (`emu/`, `ops/mirror.rs`) ebenfalls (175 Fälle [p02], Spiegeltests [p03]). Das belegt Index-/Layout-Logik, nicht Compiler-, Treiber- oder GPU-Verhalten.

## 4. Plattformen und Build (p15, t00)

| Ziel | Stand |
| --- | --- |
| Linux x86-64 (CPU, CUDA, wgpu) | Bestand |
| `x86_64-pc-windows-gnu` | Cross-Check (`check`/`clippy`) für alle neuen Features grün [t00, p01–p15]; MinGW + CUDA bricht in `build.rs` mit Meldung ab [p15] |
| Windows MSVC (CPU, wgpu, nativ Vulkan, CUDA, HIP) | `build.rs`-Zweig, `scripts/build-windows.ps1`, `install.ps1`, Workflows: L0; `x86_64-pc-windows-msvc` ist auf der CPU-Maschine nicht prüfbar, nichts lief auf Windows [p15] |
| `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin` | `check`/`clippy` grün; ARM-Codegen nur für Linux geprüft (kein Apple-SDK) [t00, p13] |
| `aarch64-pc-windows-msvc` | nicht übersetzt [p13, p15] |
| Workflows (`.github/`) | geparst, nie gelaufen [p15] |
| `--device auto`, Gerätenummern (`vulkan:N` = N-tes Gerät des Backends), `devices --json` | Auswahllogik durch Unit-Tests und Läufe ohne GPU abgedeckt; auf Hardware L0 [p15] |
| Selbsttest-Harness (Gruppen, `--only`, `--quants`, `--list`, `--strict`, `--json`) | L1 nur für die CPU gegen sich selbst (`selftest-gpu --device cpu`: 220 Prüfungen, 0 fehlgeschlagen, 35 übersprungen); das prüft den Test, kein GPU-Backend [p15] |
| Autotuner (`src/autotune.rs`) | L0, wird von nichts aufgerufen, nichts gemessen [p15] |

## 5. Bekannte Lücken

Nach Gebiet geordnet. „Integrator" meint die Besitzer der gemeinsamen Dateien (`Cargo.toml`, `build.rs`, `lib.rs`, `backend/mod.rs`, `devices.rs`, `bin/serve.rs`,
`selftest.rs`, `rust_decoder.rs`).

### 5.1 Indexsemantik der Geräte

* **Regel (p15):** `--device <backend>:<N>` zählt die Geräte *dieses Backends* ab 0, nicht die Nummer `[n]` aus `fllm-rs devices`. `VkContext::new(i)` [p01],
  `SYCL create(i)` [p09] und `metal::device_info::open(i)` [p05, p07] erwarten genau diese Zählung.
* **Vor p15** liefert `select_device("auto")` für Vulkan und Metal den *globalen* Inventarindex, `create(N)` erwartet die Nummer unter den nativen Geräten [p01, p04, p09]. Auf einem Rechner mit
  CPU-/CUDA-/wgpu-Einträgen wählt `auto` dann das falsche Gerät oder fällt auf wgpu zurück. p15 behebt das in `devices.rs`/`select_device` (auf dem p15-Branch, ohne Konflikt mit den anderen Paketen
  gemerged); **ungeprüft** ist es mit mehr als einem Adapter derselben API (Nummern > 0 nie ausgeführt [p15]).
* **Metal-Heuristik (p05):** `metal::device_info::open(n)` liest `n` als *globale* Nummer, wenn sie in `[first_index, first_index + Anzahl)` liegt, sonst als Position. Mit p15 wird immer die lokale Nummer übergeben;
  bei zwei Metal-GPUs (Intel-Mac mit eGPU) würde `metal:1` falsch gelesen. p15 empfiehlt, die Heuristik zu streichen. Die Empfehlung aus p05/p07, in `auto` das native Metal vor den wgpu-Adapter zu ziehen, ist mit p15 überflüssig.
* `auto` wählt SYCL, NPU und TPU nie (nur ausdrücklich) [p09, p15].

### 5.2 Doppelte native/wgpu-Einträge in `devices`

* Mit `--features rust-gpu,vulkan` (oder `metal`) steht dieselbe GPU zweimal im Inventar: als wgpu-Adapter und als nativer Eintrag. Native Einträge beginnen mit `native, Vulkan x.y.z, …` [p01] bzw. `Metal (native), …` [p05].
* p15 etikettiert jedes Gerät mit Backend und API (`native`/`wgpu`) und listet die wgpu-Adapter als `wgpu-vulkan:N`, `wgpu-metal:N`, `wgpu-dx12:N`; erzwingen lässt sich ein Weg mit `vulkan-native`/`metal-native` bzw.
  `wgpu-<api>`. Diese Ausgabe wurde nur auf einer Maschine (ein Adapter) gesehen [p15].
* Unter `--features hip` zeigt die CUDA-Liste die AMD-Karten bereits (Beschreibung „CUDA/ROCm"); es gibt keinen zweiten Eintrag [p08].
* `fllm-rs devices` und `serve` öffnen unter `tpu-pjrt` schon beim Auflisten den PJRT-Client; die TPU gehört dem Prozess bis zum Ende, ein zweiter Prozess scheitert [p12].

### 5.3 Mamba: es fehlt die `Stream`-Methode

* Mamba-1/-2 laufen auf **keinem** Gerätebackend im Decoder, weil `trait Stream` keinen Hook hat; der Decoder rechnet Mamba-Schichten auf dem Host.
* Vorhanden, aber ungenutzt: Shader und Pläne für Vulkan (`plan_mamba2_mixer`, `plan_mamba_conv`, `plan_mamba1_scan`, `plan_mamba1_bc_norms`) [p03], `VulkanStream::mamba2_layer` [p04],
  MSL-Kernel und Pläne für Metal (`mamba_conv`, `mamba_scan1/2`, `mamba_gate_norm`, `mamba_bc_norm`) [p06c].
* Der Vorschlag steht im p03-Bericht (Typen `MambaShape`, `MambaWeights`, Methoden `supports_mamba`, `mamba_layer` in `backend/mod.rs`). Dazu müsste `rust_decoder.rs` Gerätezustände für Mamba-Schichten anlegen
  und `mamba_forward` auf den Hook umleiten. Der `impl`-Block für Vulkan ist wegen der fehlenden Trait-Typen nicht kompiliert [p04]. Mamba-1 braucht zwischen den Teilen zwei Linears (`x_proj`, `dt_proj`) [p03].

### 5.4 Konflikte p13 (ARM) ↔ p14 (x86)

* Beide ändern `rust/engine/src/cpu_kernels.rs` und `rust/engine/src/cpu_selftest.rs`; p15 fand beim Probe-Merge genau diesen einen Konflikt (alle anderen Zweige mergen einzeln und zusammen) [p15].
* `cpu_kernels.rs`: Enum `CpuTier`, `ALL`, `level()`, `native()`, `name()`, `supported()`. p13 fügt `NeonSve` hinzu (`ALL: [CpuTier; 9]`, Stufe 4), p14 fügt `Avx512Bf16` (Stufe 5) und `Amx` (Stufe 6) hinzu (`ALL` 8 → 10).
  Nach beiden Änderungen hätte `ALL` 11 Einträge (aus den Berichten gerechnet, nicht gebaut). Die Leitern sind je Architektur getrennt; Stufe 4 darf doppelt vorkommen [p13, p14].
* `cpu_selftest.rs`: `check_tier` (p13: Schleifen für K-Gruppe und Float-Formen, DeltaNet-Block; p14: Float-Fälle 96 Spalten/20 Token, neue Tiers), `run`, Feature-Liste, Hilfetext.
* `devices.rs` (`cpu_features`/`cpu_capabilities`): p13 ergänzt die ARM-Probe um `bf16`, `fhm`; p14 gibt Klasse/Policy aus. Beides sind kleine, getrennte Hunks.
* Außerdem: `scripts/check-arm-cpu.sh` (p13) und `scripts/check_x86_ext.sh` (p14) setzen voraus, dass der jeweilige Zweig gemerged ist.

### 5.5 Nicht portierte HIP-Kernel und HIP-Lücken [p08]

* **Nicht portiert:** vLLM `attention.cu` (Paged Attention; erwartet vLLMs Cache-Layout, der `DeviceKvCache` hat ein anderes), `q_gemm_rdna3.cu`, `q_gemm_rdna3_wmma.cu`, `moe_q_gemm_rdna3.cu` (GPTQ-Layout vs. Marlin-permutiertes Q4Z-Layout;
  fLLM hat keinen Q4Z-MoE-Pfad), LLMM1, `wvSplitK_hf_big_`, `wvSplitKrc`, `wvSplitKQ` (FP8), die MFMA-bf16-Variante für gfx94x. Als Ersatz dienen `hip_q4z.cu` (MMA-GEMM + Int4-Skinny) und DPP in `attention4.cu`.
* DeltaNet wird für AMD nicht gebaut (`hip_stubs.cu`: `fllm_delta_supported` liefert 0); `fllm_cuda_graph_*` sind Platzhalter, NVFP4/FP8-Marlin und `fllm_q4z_gptq_repack` sind Stubs.
* **Ohne die Patches der GEMEINSAMEN Dateien fehlen Symbole:** `build.rs` muss die Quellen aus `rust/engine/hip/` bauen (angepasst an p15: `~/fllm-tools/patches/p15-p08-build-rs-adapted.patch`), `cuda.rs` den Q4Z-Pfad für ROCm zulassen
  (`~/fllm-tools/patches/p08-rocm-hip-common.patch`, nur der `cuda.rs`-Teil; der `selftest.rs`-Teil ist in p15 schon enthalten) [p08, p15].
* Alle Lane-Layouts (MFMA/WMMA) und DPP-Kontrollcodes stammen aus Dokumentation und rocWMMA-Headern; die Python-Emulation prüft nur die Kernel-Indizes gegen diese Layouts [p08].

### 5.6 Fehlende Metal-Teile [p05, p06a–c, p07]

* Mamba: keine `Stream`-Methode (siehe 5.3).
* Nicht vorhanden: `linear_f16_out`, `glu_f16`, `rope_kv_write_f16`, Graph-Capture, `host_linear`; GPU-Sampling (`supports_device_sampling` = false, Temperatur über das CPU-Backend); `quantized_linear_add` nur mit `scale == 1.0`.
* Embedding läuft für IQ4_XS, NVFP4, Q4Z, FP8 und skalierte Tabellen auf dem Host (langsam bei langen Prompts).
* Formate ohne Kernel (IQ1/IQ2/IQ3, Q8_1, Q8_K) werden beim Upload auf dem Host nach F16 gewandelt; `moe_linear` kennt nur 10 Formate (kein Q2_K, Q3_K, NVFP4, Q4Z, F16/BF16).
* Gewichte über `maxBufferLength` werden nicht in Zeilenscheiben zerlegt. Der Decoder ruft `register_host_mapping` nicht auf (kein No-Copy-GGUF; jedes Gewicht wird einmal kopiert).
* `x` und `out` dürfen sich bei `quantized_linear` nicht überlappen. Die Hazard-Deklarationen der Pläne sind gegen keine GPU geprüft (`FLLM_METAL_SERIAL=1` als erster Test).
* Der Metal-Compiler hat die Kernel nie gesehen: `mslcheck.sh` ist ein C++-Frontend mit selbst geschriebenem Shim und kennt keine Adressraum-, Attribut- oder Threadgroup-Speicher-Regeln; im Verzeichnismodus gibt es Fehlalarme [t00, p07].
* Alle Bandbreitenzahlen der Chips M1–M5 sind Schätzwerte, besonders M3 Max, M4 Max und M5 [p07].

### 5.7 Fehlende Vulkan-Teile [p01–p04]

* Mamba: siehe 5.3. `moe_expert_gemm` = `Unsupported`; `moe_linear` ist ein Matvec (kein gruppierter GEMM); `lin_float` ist nicht gekachelt.
* Embedding für NVFP4, Q4Z, IQ1–IQ3, IQ4_XS, Q8_K, FP8 hat keinen Shader (Host-Dekodierung); `sampling_logits` mit Temperatur, `moe_route` und Attention mit Kopfbreite > 512 laufen auf der CPU [p03, p04].
* MoE-Experten liegen doppelt im Speicher (Arena plus Einzelgewichte); Arena und `host_linear`-Scratch stehen nicht im `MemoryPlan`, bei fast vollem Speicher ist OOM möglich [p04].
* Gesplittete KV-Pools (über `maxBufferSize`) funktionieren nur für Attention und KV-Schreiben, nicht für `upload`, `download`, `copy_buffer`, `zero_buffer` oder CPU-Attention [p04].
* Cooperative Matrix wird nur bei 16×16×16 f16/f32 geplant und nur mit pinnbarer Subgruppengröße [p02, p04]; `VK_NV_cooperative_matrix` und bf16-/FP8-Shader-Erweiterungen werden nicht genutzt [p01].
* Eine Queue, kein asynchroner Transfer, keine Buffer-Device-Address (die `_bda`-Varianten sind fertig, aber ungenutzt), kein Descriptor-Indexing [p01, p03].
* Der Selftest-Patch `p04-selftest-vulkan.patch` (exakter Durchlauf, dann schneller) liegt nur als Patch vor; ohne ihn besteht `selftest-gpu` nur mit `FLLM_VULKAN_EXACT=1` (die schnellen Varianten `_i`/`_c` runden Aktivierungen auf int8 bzw. rechnen Kacheln in f16) [p02, p04].
* Standardmäßig laufen die schnellen Varianten, Greedy-Text kann von der CPU abweichen; für Vergleiche `FLLM_VULKAN_EXACT=1` [p04].

### 5.8 Weitere Lücken

* **SYCL:** kein DeltaNet, kein `host_linear`, keine Graphen, keine fused Operationen; Attention skalar FP32; nativ nur nach `git apply rust/engine/sycl/build_rs.patch` auf `build.rs` (ohne den Patch ein Stub, der „kein SYCL-Gerät" meldet);
  `sycl:N` zählt die SYCL-Geräte. Verifiziert nur mit dem Open-Source-DPC++-Nightly, nicht mit dem Release-`icpx`; `joint_matrix`-Namen können abweichen [p09].
  p09 empfiehlt Vulkan als Intel-Standard und SYCL als Option (Begründung und Messplan in `rust/engine/sycl/EVALUATION.md`; von p16 nicht nachgeprüft).
* **NPU (OpenVINO):** kein Decode-Gewinn erwartet (bandbreitengebunden); der Hybrid-Stream rechnet alles außer `quantized_linear` einkernig auf der Referenz-CPU, Gewichte liegen doppelt; der Host-Haken (`p10-npu-openvino-host-hook.patch`
  für `compute.rs`, aktiv mit `FLLM_NPU_HOST=1`) ist der brauchbarere Weg, aber nur als Patch übersetzt; Raten des Kostenmodells sind Platzhalter [p10].
* **XDNA/Hexagon/ANE:** nur Proben, kein Rechenpfad; `create()` scheitert immer mit einer Zustandsmeldung [p11]. Die Qualcomm-QAIRT-Header sind „Confidential and Proprietary" und nicht im Repository.
* **TPU:** kein q8_0-KV, kein DeltaNet, jede neue Form kompiliert ~12 Programme, kein Persistent-Cache; `model.rs` (ganzer Decoderblock als ein Programm) wird vom Decoder nicht aufgerufen (kein Hook); die Selbsttestschranken setzen f32-Genauigkeit voraus,
  erwartet sind einzelne FAIL-Zeilen bei `swiglu`, `geglu`, `attention`, `rms_norm` [p12].
* **Selbsttest (p15):** `SKIPPED (unsupported)` ändert die Zählung bei Backends, die Operationen nicht anbieten; der Standard von `selftest-gpu --device` ist jetzt `auto`. Das CPU-Backend selbst meldet im Selbsttest 7 fehlgeschlagene q8-Attention-Prüfungen (schon vor dieser Runde) [p10, p15].
* **Autotuner (p15):** `src/autotune.rs` wird von nichts aufgerufen.
* **Zahlen:** alle Geschwindigkeitsangaben in den Berichten sind Erwartungen, keine Messungen; die L3-Spalte ist für alle neuen Backends leer.

## 6. Berichtsstand (Branch-Köpfe)

| Paket | Branch | Kopf | Inhalt |
| --- | --- | --- | --- |
| t00 | `t00-toolchain` | `247fb37` (kein Commit; Werkzeuge liegen in `~/fllm-tools`) | `xcheck.sh`, `hipcheck.sh`, `syclcheck.sh`, `mslcheck.sh` |
| p01 | `p01-vulkan-core` | `bc07dd0` | Vulkan-Kern (ash) |
| p02 | `p02-vulkan-shader-a` | `df9971f` | quantisierte Matvec/Matmul-Shader |
| p03 | `p03-vulkan-shader-b` | `75c43f8` | Nicht-Matmul-Shader und Pläne |
| p04 | `p04-vulkan-integration` | `b5e3386` | `VulkanBackend`/`VulkanStream` (enthält p01–p03) |
| p05 | `p05-metal-core` | `cbe9b93` | Metal-Kern (objc2-metal) |
| p06a | `p06a-metal-quant` | `886e567` | MSL-Quantkernel, Layouts, Planer |
| p06b | `p06b-metal-ops` | `998de62` | MSL Attention/Norm/RoPE/Aktivierung/Embedding/KV |
| p06c | `p06c-metal-recurrent` | `8a85a94` | MSL MoE/Gating/DeltaNet/Mamba |
| p07 | `p07-metal-integration` | `0f1468e` | `MetalBackend`/`MetalStream` (enthält p05–p06c) |
| p08 | `p08-rocm-hip` | `17b5c96` | HIP: MFMA/WMMA-GEMM, Skinny, Q4Z, `HipBackend` |
| p09 | `p09-sycl-intel` | `d9f3552` | SYCL-Backend |
| p10 | `p10-npu-openvino` | `e496f89` | Intel-NPU über OpenVINO |
| p11 | `p11-npu-other` | `a957552` | XDNA/Hexagon/ANE (Studien und Proben) |
| p12 | `p12-tpu-pjrt` | `7dd23d1` | TPU über PJRT/StableHLO |
| p13 | `p13-arm-cpu` | `3594818` | ARM-CPU-Kerne |
| p14 | `p14-x86-ext` | `d58ebb0` | AVX-512, AMX, CPU-Familien-Policy |
| p15 | `p15-windows-conformance` | `a893644` | Windows, Konformanz, `auto`, Selbsttest |

Alle Berichte lagen beim Schreiben vor; kein Bericht fehlt. Der Branch `p06-metal-shaders` (das ursprüngliche, zu große Paket) ist durch p06a/b/c ersetzt und wird hier nicht ausgewertet.
