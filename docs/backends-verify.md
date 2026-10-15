# Prüfanleitung für fremde Hardware

Diese Anleitung bringt die neuen Backends der Mega-Runde von **L0** (übersetzt) auf **L1/L2/L3** (siehe [backends.md](backends.md), Abschnitt 1).
Sie besteht aus fertigen Kopierbefehlen je Plattform und aus Skripten (`scripts/verify-<backend>.sh` und `.ps1`), die denselben Ablauf
automatisch fahren und das Ergebnis als `report.json` ablegen.

**Ehrlicher Stand:** Weder diese Befehle noch die Skripte sind je auf der Zielhardware gelaufen. Die Skripte sind mit `bash -n` bzw. dem
PowerShell-Parser geprüft, sonst nichts. Wo ein Befehl aus dem Bericht eines Pakets stammt, steht das Paket dabei (`[p07]`). Ein erster
Lauf mit Fehlern ist erwartet und gewollt; die Fehlerbilder in Abschnitt 4 sagen, wo man zuerst sucht.

## 0. Vorbereitung

* Quelltext: ein Checkout, in dem alle gewünschten Pakete gemerged sind (die Verify-Skripte rufen die Skripte der Pakete auf, soweit sie im Baum liegen; sonst
  melden sie „skipped" mit dem Paketnamen). Die Patches der gemeinsamen Dateien aus den Berichten müssen angewendet sein (HIP: `build.rs` und `cuda.rs` [p08, p15]; SYCL: `rust/engine/sycl/build_rs.patch` [p09]; optional NPU:
  `p10-npu-openvino-host-hook.patch` für `compute.rs`; Vulkan-Selbsttest: `p04-selftest-vulkan.patch`). Ohne diese Patches baut das Feature, aber das Backend ist ein Stub oder fehlt Symbole.
* Rust (rustup, `cargo`). Je Backend zusätzlich: siehe Abschnitt der Plattform.
* Ein **kleines Modell**: das Beispiel der Berichte ist `hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf` [p12]. Für Q4-Formate daraus `fllm-rs quantize IN.gguf OUT.gguf q4_0`
  (bzw. `q4_k_m`) erzeugen. Für MoE, DeltaNet (Qwen3.5/3.6) und Mamba je ein eigenes Modell; welche Formate ein GGUF enthält, zeigt `fllm-rs inspect MODEL.gguf`.
  `fllm-rs pull hf://owner/repo/datei.gguf` lädt ein Modell und gibt den Pfad aus.
* Nicht gleichzeitig mehrere Prozesse auf demselben Gerät (TPU: ein Prozess pro Chip).

Die Skripte schreiben nach `verify-out/<backend>/`: ein Log je Schritt, `report.json`, Server-Logs, die Texte. **Zurückschicken:** das ganze Verzeichnis (bzw. die Dateien, die
im jeweiligen Abschnitt genannt sind) plus die Zeile „Gerät" aus `fllm-rs devices`.

## 1. Gemeinsamer Ablauf (von Hand)

Die fünf Schritte sind für alle GPU-/Beschleuniger-Backends gleich. `<F>` ist das Cargo-Feature, `<DEV>` die Geräteangabe (`vulkan:0`, `metal:0`, `hip:0`, `sycl:0`, `npu`, `tpu`, `auto`).
Seit p15 zählt `<backend>:N` die Geräte *dieses* Backends ab 0; die Nummer `[n]` aus `fllm-rs devices` ist **nicht** gemeint, die Zeile `Select:` nennt den richtigen Wert
(`docs/device-selection.md`). Mit p15 erzwingen `vulkan-native:N`/`metal-native:N` das native Backend (kein stiller Rückfall auf wgpu), `wgpu-vulkan:N` usw. das wgpu-Backend.

### 1.1 Bauen

```sh
cargo build --release -p fllm-engine --features <F> --bin fllm-rs      # Binary: target/release/fllm-rs (Windows: fllm-rs.exe)
```

### 1.2 Gerät sichtbar?

```sh
target/release/fllm-rs devices                 # Zeile "Select:" nennt den Wert für --device; mit p15 auch: devices --json [--measure]
```
Erwartung: ein Eintrag pro Gerät; nativ erkennbar an `native, Vulkan x.y.z, …` [p01] bzw. `Metal (native), …` [p05]; AMD unter HIP als „CUDA/ROCm, N CUs" [p08]. Kein Eintrag = Treiber/Laufzeit fehlt (kein Fehler).

### 1.3 Selbsttest

```sh
target/release/fllm-rs selftest-gpu --device <DEV> --json out.json
```
Erwartung: Exit 0, `failed = 0`. Jede Zeile nennt den größten Fehler von Gerät und CPU relativ zur Fehlerskala des Elements, die Toleranz und die Zeit [`docs/gpu-backends.md` §7].
`out.json` hat (p15) `checks`, `failed`, `skipped`, `ok`, `error` und jede einzelne Prüfung; mit p15: `--list` (Gruppen), `--only GRUPPE[,…]`, `--quants q4_0,q8_0`, `--strict`.
`SKIPPED (unsupported)` heißt „Backend bietet es nicht an" und ist **kein** Bestehen.
Zweiter Durchlauf mit den schnellen Kernen, wo es sie gibt (CUDA/HIP macht ihn selbst; für Vulkan siehe Abschnitt 2.2, für Metal `FLLM_METAL_EXACT=1` gegen Standard).

### 1.4 Text gegen die CPU-Referenz (L2)

```sh
BIN=target/release/fllm-rs; MODEL=/pfad/modell.gguf
ASK='{"model":"m","messages":[{"role":"user","content":"Write one sentence about the sea."}],"temperature":0,"max_tokens":64}'
$BIN serve --model "$MODEL" --device cpu --port 18001 --ctx 2048 --max-seqs 2 & PID=$!
until curl -sf http://127.0.0.1:18001/v1/models >/dev/null; do sleep 2; done
curl -s http://127.0.0.1:18001/v1/chat/completions -H 'Content-Type: application/json' -d "$ASK" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > cpu.txt
kill $PID

$BIN serve --model "$MODEL" --device <DEV> --port 18002 --ctx 2048 --max-seqs 2 & PID=$!
until curl -sf http://127.0.0.1:18002/v1/models >/dev/null; do sleep 2; done      # beim ersten Start dauert das (Pipelines/Shader/Programme)
for i in 1 2; do                                                                    # die erste Anfrage wärmt auf
  curl -s http://127.0.0.1:18002/v1/chat/completions -H 'Content-Type: application/json' -d "$ASK" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > dev.txt
done
kill $PID
cmp cpu.txt dev.txt && echo "IDENTISCH" || echo "UNTERSCHIEDLICH"
```
Windows (PowerShell), je Server nach dem Start:
```powershell
$body = '{"model":"m","messages":[{"role":"user","content":"Write one sentence about the sea."}],"temperature":0,"max_tokens":64}'
(Invoke-RestMethod -Uri http://127.0.0.1:18002/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body).choices[0].message.content | Set-Content dev.txt
```
Erwartung: **im exakten Modus identisch** (Vulkan `FLLM_VULKAN_EXACT=1`, Metal `FLLM_METAL_EXACT=1`, HIP `FLLM_CUDA_EXACT=1`, SYCL `FLLM_SYCL_EXACT=1`, TPU `FLLM_TPU_EXACT=1`). Im schnellen Standardmodus
kann ein späteres Token abweichen (Vulkan: int8-Aktivierungen etwa 0,4 % und f16-Kacheln [p04]; Metal: Half-Kacheln und schnelle `exp`/`tanh` [p06a, p06b]; HIP: FP16-Rundung der Matrixkern-Pfade [p08];
SYCL/XMX: fp16-Eingänge [p09]). Dann die Position des ersten Unterschieds notieren (`cmp` nennt das Byte), nicht „fehlgeschlagen" melden.
Ein Treffer des Textes allein beweist nicht, dass das Backend benutzt wurde: Server-Log lesen (bei `vulkan`/`metal` fällt `Decoder::set_device` bei Fehler auf wgpu zurück; die Verify-Skripte vermerken
`log_mentions_wgpu` im Report, wenn das Log „wgpu" erwähnt — nur ein Indiz).

### 1.5 Benchmark

```sh
$BIN serve --model "$MODEL" --device <DEV> --port 18002 --ctx 4096 --max-seqs 2 &        # wie oben
$BIN bench --url http://127.0.0.1:18002 --scenarios prefill,decode-single --prompt-lens 512 --gen-len 128 \
           --warmup 1 --repeats 3 --label "<Hardware>" --json bench-<hardware>.json
$BIN bench-report bench-cpu.json bench-<hardware>.json                                  # Markdown-Vergleich
```
Szenarien laut `fllm-rs bench --help`: `prefill`, `decode-single`, `decode-context`, `concurrency`. **L3** braucht zusätzlich einen Lauf der Vergleichsimplementierung auf *derselben* Hardware (llama.cpp mit
Vulkan/Metal/HIP/SYCL; vLLM auf ROCm [p08]); ohne ihn bleibt es bei einer Messung, nicht bei L3. Das `tok_s` im `report.json` der Skripte ist ein grober Wert aus einer Chat-Anfrage (inklusive Prompt), kein Benchmark.

### 1.6 Die Skripte

| Backend | Skript (`.sh` und `.ps1`) | Features | ruft auf (falls im Baum) |
| --- | --- | --- | --- |
| Vulkan nativ | `verify-vulkan` | `vulkan` | `shaders/vulkan/check_all.sh`, Smoke- und Hardware-Tests |
| Metal | `verify-metal` | `metal` | `metal_core_smoke`, `metal_quant_kernels`, `metal_ops_kernels` |
| ROCm/HIP | `verify-hip` | `hip` | `scripts/hip-verify.sh` |
| Intel SYCL | `verify-sycl` | `sycl` | Test `linear_matches_cpu` |
| Intel NPU (OpenVINO) | `verify-npu-openvino` | `npu-openvino` | Python-Gerätecheck, `npu_hardware_selfcheck`, optional die Proben von p11 (`V_NPU_PROBES`) |
| TPU | `verify-tpu` | `tpu-pjrt` | `scripts/colab/tpu_check.py` |
| ARM-CPU | `verify-arm` | – | `scripts/check-arm-cpu.sh` |
| x86 AVX-512/AMX | `verify-avx512` | `avx512` | `scripts/check_x86_ext.sh` |
| Windows | `verify-windows` | `rust-gpu,vulkan` (über `build-windows.ps1`) | `scripts/build-windows.ps1` |

Gemeinsame Optionen (bash: `--model`, PowerShell: `-Model`; beide Schreibweisen funktionieren in den `.ps1`):
`--model FILE`, `--model-uri URI`, `--bin PATH` (fertiges Binary, überspringt den Build), `--device NAME`, `--out DIR`, `--port N`, `--tokens N`, `--prompt TEXT`, `--skip build,devices,pre,selftest,text,bench`, `--dry-run` (druckt den Plan und führt nichts aus). Die Phasen und ihre Reihenfolge stehen im Kopf von `scripts/verify-common.sh` (bash 3.2-kompatibel für macOS) und `scripts/verify-common.ps1`.

`report.json` enthält: `backend`, `device`, `commit`, `tree_dirty`, `command`, `features`, `model`, `host`, `date_utc`, `exit_code`, `duration_seconds`, `tok_s` und `steps[]` mit je `name`, `command`, `exit_code`, `seconds`, `log`, `note`
(Textfälle zusätzlich `text_vs_cpu` und `request` mit Sekunden, Tokens und `tok_s`). Der Exit-Code des Skripts ist 1, sobald ein Schritt scheiterte; ein abweichender Text zählt nicht als Fehlschlag (siehe 1.4).
Schlägt der Selbsttest fehl, wiederholt das Skript ihn je Schalter aus der Schalterliste des Backends (`selftest-with-<SCHALTER>`): **notieren, welcher Schalter den Fehler beseitigt** (so verlangen es die Berichte p05, p07, p08).

## 2. Plattformen

### 2.1 Mac (Metal, Apple Silicon)

* Voraussetzungen: macOS (für Residency-Sets und MSL 3.2 ab macOS 15 [p05]), Rust, das Apple-SDK (Xcode oder Command Line Tools; t00: ohne SDK ist Metal nicht linkbar). M1 bis M5; eine Zeile pro Chip.
* Build: `cargo build --release -p fllm-engine --features metal --bin fllm-rs`
* Skript: `scripts/verify-metal.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf`

Reihenfolge von Hand [p07]:

| # | Befehl | Erwartung |
| --- | --- | --- |
| a | `cargo test --release -p fllm-engine --features metal metal_core_smoke -- --nocapture` | 9 Zeilen `[ok]`/`[skip]`/`[FAIL]` [p05] |
| b+c | `FLLM_METAL_PREWARM=1 FLLM_METAL_DEBUG=1 cargo test --release -p fllm-engine --features metal metal_quant_kernels -- --nocapture` | zuerst die **Compilermeldungen** je `.metal`-Datei lesen (der wahrscheinlichste erste Fehler: der Metal-Compiler hat die Quellen nie gesehen [p06a, p06b]); danach je Zeile `[ok]`/`[FAIL] <Format> batch <n> <Kernel>: max rel err …` |
| d | `cargo test --release -p fllm-engine --features metal metal_ops_kernels -- --nocapture` | `[ok]` je Operation; Zeile `device: …` notieren |
| e | `target/release/fllm-rs devices` und `selftest-gpu --device metal:0 --json metal.json` | Eintrag „Metal (native), …" mit Chip, Familien, Arbeitsspeicher |
| f | Text gegen CPU (1.4) mit einem dichten Q8_0-/Q4_0-Modell, dann `FLLM_METAL_NO_NOCOPY=1` gegen Standard (echtes Datei-Mapping, Annahme von p05) | identisch im exakten Modus (`FLLM_METAL_EXACT=1`) |
| g | MoE- und DeltaNet-Modell gegen die CPU | wie f |

Bei Fehlern der Reihe nach diese Schalter versuchen und notieren, welcher den Fehler beseitigt [p07, p05]: `FLLM_METAL_SERIAL=1` (schließt Barrierenfehler aus), `FLLM_METAL_FAST_MATH=0`, `FLLM_METAL_EXACT=1` (Float- statt Half-Kacheln),
`FLLM_METAL_NO_NOCOPY=1`, `FLLM_METAL_SINGLE_LIB=1` (Quell-Layout), `FLLM_METAL_MSL=3.0` oder `3.1`, `FLLM_METAL_RESIDENCY=0`.
Fehlermuster [p06a]: nur `_h`-Kernel falsch ⇒ Half-Kacheln/`simdgroup_matrix`; nur ein Format falsch ⇒ Decoder (läuft in `metal/emu/` auf der CPU richtig); nur Batch > 8 ⇒ Matmul; alles falsch ⇒ Parameterblock/Bindung.
Nur Prefill-Fälle der Attention falsch ⇒ die Annahme zu `simdgroup_load(…, transpose=true)` [p06b]. CPU-Emulation zum Gegenprüfen: `rust/engine/metal/emu/`, `emu_ops/`, `emu_recurrent/` (je mit README; braucht `python3`, clang++ und den MSL-Shim aus `~/fllm-tools/tools/msl-shim`).

Je Chip festhalten: Name, `summary()` (die Zeile „Metal (native), …" aus `devices`), `recommendedMaxWorkingSetSize`, Ergebnis der drei Tests, Tokens/s. Die Bandbreitenzahlen in der Beschreibung sind Schätzungen (M3 Max, M4 Max, M5 unsicher) [p07].

### 2.2 AMD (Vulkan und ROCm/HIP)

**Vulkan nativ** (Linux RADV, Windows AMD-Treiber). Voraussetzungen: Vulkan-Treiber, `glslc` (Vulkan SDK) beim Bauen; für `check_all.sh` zusätzlich `spirv-val`.
```sh
cargo build --release -p fllm-engine --features vulkan --bin fllm-rs
scripts/verify-vulkan.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
```
Reihenfolge von Hand, alles mit `FLLM_VULKAN_VALIDATE=1` [p04]:

| # | Befehl | Erwartung |
| --- | --- | --- |
| a | `rust/engine/shaders/vulkan/check_all.sh` | `modules: 171 … FAIL=0` (übersetzt und validiert alle Shader mit dem lokalen `glslc`) |
| b | `FLLM_VULKAN_VALIDATE=1 FLLM_VULKAN_VERBOSE=1 cargo test --release -p fllm-engine --features vulkan --lib backend::vulkan::instance::smoke -- --ignored --nocapture --test-threads=1` | 7 Tests grün, Validation-Layer ohne Fehler [p01]; `context_reports_capabilities` druckt Gerät, Caps, Coop-Matrix-Formen |
| c | dasselbe mit `backend::vulkan::backend::hw_tests` | 7 Tests grün (u. a. `a_decode_step_is_recorded_into_one_submission`) [p04] |
| d | `fllm-rs devices` | ein Eintrag `native, Vulkan …` je Gerät [p01] |
| e | `FLLM_VULKAN_EXACT=1 fllm-rs selftest-gpu --device vulkan:0 --json exact.json`, danach ohne `FLLM_VULKAN_EXACT` | exakt: `failed = 0`. Der schnelle Durchlauf (`_i` Integer-Dot, `_c` Coopmat) besteht die FP32-Toleranz nicht zwingend; ein eigener Durchlauf mit gelockerter Toleranz liegt als `p04-selftest-vulkan.patch` vor (Vorschlag p02: 1 % für `_i`, 0,5 % für `_c`). Ob `_i`/`_c` mit `FAST_TOL` bestehen, zeigt erst die Hardware [p04] |
| f | kleines Q4_0/Q8_0-Modell, Greedy, gegen die CPU (1.4): zuerst mit `FLLM_VULKAN_EXACT=1`, dann ohne | exakt identisch; schnell ggf. abweichend |
| g | Eingrenzen: `FLLM_VULKAN_NO_COOP=1`, `FLLM_VULKAN_NO_IDOT=1` | der Schalter, der den Fehler beseitigt, nennt die Variantenfamilie |

Pro Gerät festhalten [p01, p04]: `VkCaps::describe()` (kommt mit `FLLM_VULKAN_VERBOSE=1` beim Öffnen), `push_descriptor`, `unified_memory`, `external_memory_host`, `int_dot`, `subgroup_size`, `max_storage_buffer_range`, Anzahl der Coop-Matrix-Formen
(`vulkaninfo | grep -A6 CooperativeMatrix` für die Formen), Dauer der ersten Pipeline-Erzeugung, Anzahl Submits eines Dekodierschritts (`VulkanStream::recorder_stats()`), und dass die Datei `vk-pipeline-v1-<vendor>-<device>-<uuid>.bin` im Cache-Verzeichnis
entsteht und ein zweiter Lauf sie lädt. Verdächtigenliste bei Fehlern [p03, p04]: Pipelineerstellung (Spezialisierungskonstanten, Subgruppen-Pinning), Bindungsprüfung in `Recorder::set_buffers`, `lin_float`, Ringe, KV-Fenster,
`attn_prefill_*`, `kv_write_q8`, `moe_linear_*`/`embed_*` (Layout-Annahme), `delta_conv`/`delta_rec_*`, Experten-Arena. Shader-Fehler ohne GPU nachstellen: `rust/engine/shaders/vulkan/emu/run_kernels.sh` (siehe `emu/README.md`).
Test der Scheiben-Logik für große Gewichte: ein Gerät mit nur 128 MiB `maxStorageBufferRange`, z. B. lavapipe mit `FLLM_ALLOW_SOFTWARE_GPU=1` [p04].

**ROCm/HIP** (RDNA3 gfx11, RDNA4 gfx12, CDNA gfx908/90a/942/950; ROCm 6.2 oder neuer, empfohlen 7.x; gfx950 braucht 6.4 oder neuer und gehört in `FLLM_HIP_ARCHS`) [p08]:
```sh
FLLM_HIP=1 ROCM_PATH=/opt/rocm FLLM_HIP_ARCHS="gfx1100;gfx90a" cargo build --release -p fllm-engine --features hip --bin fllm-rs
scripts/hip-verify.sh MODEL.gguf target/release/fllm-rs hip:0                                       # Skript von p08
scripts/verify-hip.sh --model MODEL.gguf                                                            # Rahmen: Build, devices, Bench, report.json
```
Erwartung [p08]: (1) `devices` zeigt „CUDA/ROCm, N CUs" mit Name `… (gfx….)`, keinen zweiten Eintrag; beim Start Logzeile `[fllm-rs] hip: device 0 is gfx… (waveNN, WMMA|MFMA)`. (2) `selftest-gpu --device hip` (exakt, dann schnell). (3) Greedy-Text CPU gegen GPU mit allen Schaltern:
`FLLM_HIP_NO_MMA=1` (rocWMMA/FP32 statt MFMA/WMMA), `FLLM_HIP_NO_SKINNY=1`, `FLLM_HIP_NO_Q4Z=1` (AWQ/GPTQ als F32-Referenz), `FLLM_CUDA_NO_ATTN4=1`, `FLLM_CUDA_EXACT=1`. Je Kernelfamilie ein Modell: F16-GGUF (Skinny + MMA),
Q4_K/Q8_0 (MMA ab 16 Token), AWQ/GPTQ (Q4Z), ein GQA-Modell mit G = 4 oder 8 (DPP-Attention). Verdachtsstellen [p08]: Lane-Layouts in `hip_mma.cuh`, DPP-Kontrollcodes (0xB1, 0x4E, 0x141, 0x140), Nibble-Reihenfolge in `hip_q4z.cu`, LDS-Größe der Skinny-Kernel,
Registerdruck (`-Rpass-analysis=kernel-resource-usage`). Vorab auf einem Entwicklerrechner: `HIPCHECK_CODEGEN=1 ~/fllm-tools/hipcheck.sh <worktree>`. **Das Skript `hip-verify.sh` wurde nie ausgeführt** (der AMD-Lauf ist sein erster).

### 2.3 NVIDIA (Vulkan nativ, mit CUDA als Gegenprobe)

Das native Vulkan-Backend ist der einzige Weg dieser Runde, der auf NVIDIA läuft; CUDA ist Bestand. Voraussetzungen wie 2.2 (Vulkan). Die Gegenprobe vergleicht drei Wege auf derselben Karte:
```sh
cargo build --release -p fllm-engine --features vulkan,cuda,rust-gpu --bin fllm-rs
fllm-rs selftest-gpu --device vulkan-native:0 --json vk.json     # nativ (mit p15; sonst vulkan:0)
fllm-rs selftest-gpu --device wgpu-vulkan:0   --json wgpu.json   # das vorhandene wgpu-Backend (mit p15)
fllm-rs selftest-gpu --device cuda:0          --json cuda.json   # Bestand
scripts/verify-vulkan.sh --model MODEL.gguf
```
Hinweise aus den Berichten: Coopmat nur mit 16×16×16 f16/f32 im Subgruppen-Scope; `VK_NV_cooperative_matrix` (ältere NVIDIA-Treiber) wird nicht benutzt [p01, p02]. Schalter wie in 2.2. Geschwindigkeit (L3): gegen CUDA und gegen llama.cpp-Vulkan auf derselben Karte.

### 2.4 Intel: iGPU (Vulkan, SYCL) und NPU

**Intel-iGPU/Arc über Vulkan** (Linux ANV, Windows): wie 2.2 (`scripts/verify-vulkan.sh`). Auf einer iGPU ist die CPU auf x86 oft der schnellere Weg, das weiß das Inventar nicht (`FLLM_AUTO_IGPU=0`, [p15]).

**Intel GPU über SYCL** [p09]: oneAPI (`source /opt/intel/oneapi/setvars.sh`) und Level-Zero-Treiber; `git apply rust/engine/sycl/build_rs.patch` ist Voraussetzung für das native Backend (sonst Stub).
```sh
cargo build --release -p fllm-engine --features sycl --bin fllm-rs
fllm-rs devices                                              # muss das Gerät samt XMX-Form zeigen
fllm-rs selftest-gpu --device sycl:0 --json sycl.json
cargo test --release -p fllm-engine --features sycl --lib linear_matches_cpu   # deckt alle Batch-Pfade (Kachel, XMX) ab; der Selbsttest trifft nur Batch ≤ 5
scripts/verify-sycl.sh --model MODEL.gguf
```
Fehlersuche: `ONEAPI_DEVICE_SELECTOR=level_zero:0` (erster Lauf), `FLLM_SYCL_EXACT=1` (XMX aus), `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` (Einzelallokation > 4 GiB; Name der Variable ist eine Annahme nach Fremddokumentation),
`FLLM_SYCL_ALL_BACKENDS=1`, `FLLM_VULKAN_VALIDATE=1` (Vulkan-Gegenprobe). Offene Annahmen [p09]: XMX-Formen (8x16x16 / Sub-Group 16 und 8x8x16 / Sub-Group 8), VNNI-Layout der B-Kachel, Register-Spills bei Q6_K/Q3_K/Q2_K.
Windows: der `icx`-Zweig von `build.rs` ist nie gelaufen. Keine DeltaNet-Modelle (`supports_delta` = false). Messplan für L3: `rust/engine/sycl/EVALUATION.md` (Vulkan gegen SYCL).

**Intel NPU über OpenVINO** (Core Ultra mit NPU, Linux oder Windows) [p10]:
```sh
python3 -c "import openvino as ov; print(ov.Core().available_devices)"       # NPU muss erscheinen
cargo test --release -p fllm-engine --features npu-openvino --lib npu_hardware_selfcheck -- --ignored --nocapture   # 192×512-Q4_0 gegen Host, Schwelle 2 %
fllm-rs devices
FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1 fllm-rs selftest-gpu --device npu --json npu.json
scripts/verify-npu-openvino.sh --model MODEL.gguf
```
Erwartung: dieselben **7 fehlgeschlagenen q8-Attention-Prüfungen wie beim CPU-Backend** (Toleranz enger als der q8-Cache-Fehler, schon vor dieser Runde), sonst `ok`. Ohne die gesenkten Schwellen (Standard `FLLM_NPU_MIN_BATCH` = 64, `FLLM_NPU_MIN_ELEMS` = 1 Mi) erreicht der Selbsttest die NPU gar nicht;
jede Form kompiliert ein eigenes Modell (dauert). L2: Text mit `FLLM_NPU=0` gegen `--device npu` bzw. (nach dem Host-Hook-Patch) `FLLM_NPU_HOST=1`; f16-Rechnung kann Logits leicht verschieben. Bei Fehlern: `FLLM_NPU_DUMP_IR=/tmp/ir`, dann
`benchmark_app -m x.xml -d NPU` oder `ov.Core().compile_model(ov.Core().read_model("x.xml"), "NPU")`, `OV_NPU_LOG_LEVEL=LOG_INFO`. Das Pip-Paket `openvino` genügt für einen reinen Parser-Test ohne NPU
(`cargo test … -- --ignored openvino_reader`). Erwarteter Gewinn: nur Prefill, kein Decode-Gewinn; die Raten des Kostenmodells sind Platzhalter, L3 ersetzt sie (Messplan `OPENVINO.md` §8).

**AMD XDNA, Qualcomm Hexagon, Apple ANE** sind nur Proben (ein MatMul durch den Hersteller-Stack), kein Rechenpfad; „läuft" darf nirgends stehen [p11]:
```sh
cargo run --release -p fllm-engine --example npu_probe --features npu-xdna -- list          # AMD Ryzen AI: Ryzen AI Software installieren; dann: -- xdna 1 4096 4096 und -- xdna 128 4096 4096
cargo run --release -p fllm-engine --example npu_probe --features npu-qnn  -- hexagon 1 4096 4096   # Snapdragon: QAIRT SDK, QNN_SDK_ROOT (oder FLLM_QNN_HTP_LIB, FLLM_GENIE_LIB)
cargo run --release -p fllm-engine --example npu_probe --features npu-ane  -- ane 1 4096 4096       # Apple Silicon
```
(XDNA: `FLLM_XDNA_ORT_LIB`, `FLLM_XDNA_TARGET`, `FLLM_XDNA_CACHE_DIR`, `FLLM_XDNA_CONFIG`; Platzierung mit `XLNX_ONNX_EP_REPORT_FILE`.) Die Ausgabe (ProbeReport) in die Matrix-Zeile übernehmen; erst danach über die Empfehlungen der Studien entscheiden. `V_NPU_PROBES="xdna hexagon ane"` lässt `verify-npu-openvino` sie mitlaufen.

### 2.5 ARM-CPU (Apple M, Graviton, Raspberry Pi, Ampere, Snapdragon X)

```sh
scripts/check-arm-cpu.sh                  # Paket p13: Build, Unit-Tests, selftest-cpu je Tier, FLLM_CPU_ISA-Deckel; --quick = nur selftest; Logs in target/check-arm-cpu/
scripts/verify-arm.sh --model MODEL.gguf  # Rahmen: Text je Tier gegen FLLM_CPU_ISA=neon, Bench, report.json
```
Erwartung [p13]: `check-arm-cpu.sh` endet mit Exit 0 und „ok" je Tier; die Zeile „SVE vector length" erscheint auf SVE-Kernen. Zielgeräte laut Bericht: Apple M1 (dotprod), M2 oder neuer (i8mm, bf16), Graviton 3 (SVE 256 Bit, `FLLM_ARM_SVE=1`),
Raspberry Pi 5 (Cortex-A76) und eine A72-Maschine (nur NEON: prüft, dass nichts Falsches dispatcht wird). Keine Leistungsbehauptung ohne Messung; wenn `neon-sve` besteht *und* gemessen ist, kann `cpu_arm::sve::DEFAULT_ON` auf `true` (eine Zeile).
`FLLM_CPU_ISA=neon-dotprod` schaltet `bfdot` ab. Die NEON-Intrinsics (`vshlq_u8` mit negativen Schiebezahlen, `vqtbl1q_s8`, `vcvt_f16_f32`, die `bfdot`-Paarung) sind die wahrscheinlichsten ersten Fehlerstellen.

### 2.6 x86: AVX-512 und AMX

```sh
scripts/check_x86_ext.sh                  # Paket p14: Build mit --features avx512, Unit-Tests, selftest-cpu je Tier und Policy-Overrides; Exit 0 = alles bestanden
scripts/verify-avx512.sh --model MODEL.gguf
```
Testmatrix [p14]: (a) Zen 4, (b) Zen 5, (c) Intel Ice/Tiger/Rocket Lake (Client, ein 512-Bit-Port), (d) Sapphire/Emerald/Granite Rapids für AMX (Cloud, z. B. AWS m7i/c7i, GCP C3; **Linux ≥ 5.16**, AMX nur Linux), (e) Arrow/Lunar Lake für AVX-VNNI-INT8.
Die Reihenfolge: `check_x86_ext.sh` (L1), dann Greedy-Text gegen `FLLM_CPU_ISA=avx2` (L2), dann Vergleich mit llama.cpp (L3). Die ersten Fehler sind laut p14 zu erwarten bei: Tile-Konfiguration/`#UD`/SIGILL, Lane-Reihenfolgen in `codebook`/`kextra`, Paarreihenfolge bei `vdpbf16ps`,
der Berechtigungsanfrage (`arch_prctl`) in Container/VM. Policy-Schalter: `FLLM_X86_WIDE_FLOAT`, `FLLM_X86_WIDE_INT`, `FLLM_X86_BF16`, `FLLM_VNNI_GEMM`, `FLLM_AMX_MIN_BATCH`, `FLLM_CPU_ISA`.
Auf GitHub-gehosteten Runnern gibt es oft AVX-512-VNNI (Xeon Platinum 8370C), aber kein AMX [p14].

### 2.7 Windows

Voraussetzungen: Rust (MSVC-Toolchain), Visual Studio 2022 Build Tools („Desktop development with C++"); für natives Vulkan das Vulkan SDK; für CUDA das CUDA Toolkit; für HIP das HIP SDK [p15, `docs/windows.md`].
```powershell
.\scripts\build-windows.ps1 -DryRun                         # zeigt Plan, Werkzeuge, Features
.\scripts\build-windows.ps1                                  # CPU + wgpu (Vulkan/DX12) + natives Vulkan; dist\fllm-windows-x86_64-msvc.zip
.\target\release\fllm-rs.exe devices --json > devices.json
.\target\release\fllm-rs.exe selftest-gpu --device auto --json selftest.json
.\scripts\build-windows.ps1 -Cuda -CudaArchs "89;120"        # NVIDIA; Fehler beim nvcc-Aufruf mit dem vollen Befehl melden
.\scripts\build-windows.ps1 -Hip                             # AMD
.\install.ps1 -Build                                         # und -Variant cuda
.\scripts\verify-windows.ps1 -ModelUri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf     # alles zusammen
```
Zurückschicken [p15]: `devices.json`, `selftest.json`, bei Build-Fehlern der volle Befehl. Offene Annahmen: nvcc nimmt `-Xcompiler /MD,/EHsc,/bigobj,/Zc:__cplusplus`; `lib.exe @datei`; `Launch-VsDevShell.ps1`-Parameter; `hipcc.exe`/`lib\amdhip64.lib`;
Marlin/CUTLASS unter MSVC (nie versucht); statische CRT mit allen C-Abhängigkeiten (onig, ring). Die Workflows (`.github/`) sind nie gelaufen. Drittanbieter-Aktionen in den Workflows (`jakoch/install-vulkan-sdk-action`, `Jimver/cuda-toolkit` …) wurden nicht ausprobiert.

### 2.8 TPU / Google Colab

```python
!python fllm/scripts/colab/tpu_check.py --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
```
Voraussetzungen [p12, `scripts/colab/tpu_check.md`]: Laufzeittyp TPU v5e-1 (oder v2-8), **danach Laufzeit neu starten** (kein Notebook-Kern darf die TPU belegen), Internet (rustup, `pip install libtpu`). Der Build dauert 10 bis 25 Minuten (`--jobs 2` bei knappem Speicher).
Die Schritte: environment, rust, libtpu, build, devices, selftest (`FLLM_TPU_EXACT=1`), model (Greedy-Text TPU gegen CPU). Mit `scripts/verify-tpu.sh` dasselbe plus Bench und `report.json` (`!bash fllm/scripts/verify-tpu.sh --model-uri …`).
Zurückschicken: `tpu_check_report.json`, `tpu_selftest.json`, `tpu_check_selftest.log`, `tpu_check_devices.log`, `tpu_check_server_tpu.log`, `tpu_check_server_cpu.log`, `tpu_check_build.log`.
Erwartet ist ein erster Lauf **mit Fehlern**: Plugin-Version (PJRT-API 0.116 im Header), Selbsttest-Schranken (einzelne FAIL bei `swiglu`, `geglu`, `attention`, `rms_norm` knapp über der Schranke — dann gehört die Schranke gelockert, nicht der Code), bekannt FAIL: q8-KV (`kv_cache_write q8`, `attention q8`),
DeltaNet-Fälle werden übersprungen. Jede neue Programmform wird einmal übersetzt (0,1–1 s); der Selbsttest ist deshalb langsam. Einstellungen: `FLLM_TPU_LIBRARY`, `FLLM_TPU_EXACT`, `FLLM_TPU_WEIGHTS=bf16|f32|int8`, `FLLM_TPU_NUMERICS=exact|fast`, `FLLM_TPU_DEBUG=1`, `FLLM_TPU_DUMP_DIR=DIR`
(abgelegte `.mlir`-Programme lassen sich mit `scripts/colab/stablehlo_verify.py` ohne TPU parsen). Ein Prozess pro TPU: hängengebliebenes `fllm-rs` mit `pkill fllm-rs` beenden.
Zwei Hardware-Tests ohne Modell: `FLLM_TPU_EXACT=1 cargo test --release -p fllm-engine --features tpu-pjrt -- --ignored tpu_hardware`.

## 3. Auswertung: wann darf eine Zelle der Matrix höher?

Die Regeln gelten je Zelle (Hardware × Format bzw. Operation) in [backends.md](backends.md). Alles andere bleibt, wie es ist; nichts wird aus einem Nachbarn geschlossen (ein bestandenes Q4_0 sagt nichts über Q6_K).

| Stufe | Bedingung | Beleg im Ergebnis |
| --- | --- | --- |
| **L1** | `selftest-gpu` (bzw. `selftest-cpu`, die Hardwaretests des Pakets) meldet für dieses Format bzw. diese Operationsgruppe `ok` auf der echten Hardware — im exakten Lauf; wo es einen schnellen Lauf mit eigener Toleranz gibt, auch dort. `SKIPPED` und `failed` zählen nicht. | `selftest*.json`: `failed = 0` für die Gruppe; `fllm-rs selftest-gpu --list` nennt die Gruppen (`delta linear embedding norm rope activation kv attention sampling host moe fuse`), `--only`/`--quants` schränken ein |
| **L2** | zusätzlich: Greedy-Text mit einem Modell, das die Zelle benutzt, ist im exakten Modus identisch mit der CPU. Weicht nur der schnelle Modus ab, gilt L2 für den exakten Modus, mit Vermerk „schnell: abweichend ab Zeichen N". | `report.json`: `text_vs_cpu` des Falls `exact` = `same`; Server-Log belegt das native Backend |
| **L3** | zusätzlich: `fllm-rs bench` auf der Hardware, mit Vergleichslauf einer Fremdimplementierung auf derselben Hardware, Messwerte und Versionen dokumentiert | `bench-*.json` beider Läufe, `fllm-rs bench-report` |

Besonderheiten:

* **Operationen ohne Selbsttest** bleiben L0, bis ein Modell sie nutzt (dann L2 ohne L1 nur mit Vermerk): Mamba (kein Trait-Hook, [backends.md](backends.md) 5.3), DeltaNet ohne Gruppe `delta`, MoE ohne Gruppe `moe`.
* **Formate, die der Selbsttest nicht kennt**, erreichen L1 nur über das Hardware-Gegenstück des Pakets (Metal: `metal_quant_kernels`; SYCL: `linear_matches_cpu`; Vulkan: `hw_tests`).
* **TPU:** p12 verlangt `failed = 0` für L1 und erwartet doch FAIL bei q8-KV; praktikabel ist L1 je Gruppe, mit q8-KV als dokumentierter Lücke (Entscheidung offen).
* **Intel NPU:** L1 nur, wenn `npu_hardware_selfcheck` besteht *und* der Selbsttest bis auf die 7 CPU-bekannten q8-Attention-FAIL ok ist.
* **XDNA/Hexagon/ANE:** höchstens „Probe gelaufen" (ProbeReport); es gibt keinen Rechenpfad, also keine L1-Zelle.
* **Software-Adapter** (lavapipe, WARP) werden abgelehnt, es sei denn `FLLM_ALLOW_SOFTWARE_GPU=1`; ein Lauf darauf ist ein Test der Logik, nicht der Hardware, und hebt keine Zelle.

Ergebnis → Zelle (Beispiele):

| Ergebnis auf der Hardware | Zellen |
| --- | --- |
| Metal: `metal_quant_kernels` alle `[ok]` für Q4_0…Q6_K, IQ4_NL, MXFP4, NVFP4, Q4Z, F16/BF16 | Zeile „Metal nativ", Spalten L, K, IQ4, MX, NV, Q4Z, F: L1 (Chip nennen) |
| Metal: `metal_ops_kernels` alle `[ok]` | „Metal nativ", Attention (f16/q8_0): L1; MoE und DeltaNet nur, wenn `selftest-gpu` die Gruppen `moe`/`delta` mit `ok` meldet |
| Vulkan: `selftest-exact.json` `failed = 0` | „Vulkan nativ", Formate und Operationen der `ok`-Gruppen: L1 (exakter Lauf); die schnellen Varianten `_i`/`_c` erst mit bestandenem schnellem Lauf |
| HIP: `selftest-gpu --device hip` besteht, Text mit `FLLM_CUDA_EXACT=1` identisch | „ROCm/HIP" der Architektur (gfx-Name notieren): L1/L2; MMA-/Skinny-/Q4Z-/DPP-Pfade nur, wenn die Schalter-Läufe einzeln identisch sind |
| ARM: `check-arm-cpu.sh` Exit 0 | „CPU ARM" der Tiers, die `ok` melden: Q2_K, Q3_K, IQ4_NL, MXFP4, Float, DeltaNet: L1 |
| x86: `check_x86_ext.sh` Exit 0 auf AVX-512-VNNI- bzw. AMX-Hardware | „AVX-512" bzw. „AMX": L1 (CPU-Klasse nennen) |
| TPU: `tpu_selftest.json` ohne FAIL außerhalb der bekannten Lücken, `identical: true` im Modellvergleich | „Google TPU": L1 bzw. L2 (Modell und Gewichtsformat nennen) |

## 4. Fehlerbilder und Schalter (Sammlung aus den Berichten)

| Backend | Schalter / Hilfsmittel | Wirkung |
| --- | --- | --- |
| Vulkan | `FLLM_VULKAN_VALIDATE=1` | Validation-Layer; erster Lauf immer damit |
| Vulkan | `FLLM_VULKAN_VERBOSE=1` | Fähigkeiten des Planers, je Shader einmal Variante, Spezialisierungskonstanten, Gitter |
| Vulkan | `FLLM_VULKAN_EXACT=1` | nur `_s`/`lin_*` (FP32); `set_exact_math(true)` im Code |
| Vulkan | `FLLM_VULKAN_NO_IDOT=1`, `FLLM_VULKAN_NO_COOP=1` | Integer-Dot- bzw. Coopmat-Varianten aus |
| Vulkan | `FLLM_ALLOW_SOFTWARE_GPU=1` | lavapipe/WARP zulassen (nur für Logiktests) |
| Metal | `FLLM_METAL_SERIAL=1` | jede Dispatch serialisiert: schließt Hazard-/Barrierenfehler aus |
| Metal | `FLLM_METAL_FAST_MATH=0`, `FLLM_METAL_EXACT=1` | Fast-Math aus; Float- statt Half-Kacheln |
| Metal | `FLLM_METAL_NO_NOCOPY=1`, `FLLM_METAL_SINGLE_LIB=1`, `FLLM_METAL_MSL=3.0\|3.1`, `FLLM_METAL_RESIDENCY=0` | No-Copy-Mapping aus; eine Bibliothek statt je Datei; MSL-Version; Residency-Set aus |
| Metal | `FLLM_METAL_PREWARM=1`, `FLLM_METAL_DEBUG=1` | alle Bibliotheken beim Start kompilieren; Compilerfehler je Datei; weitere: `FLLM_METAL_DUMP_DIR`, `FLLM_METAL_MEM_LIMIT_MB`, `FLLM_METAL_CB_DISPATCHES` |
| HIP | `FLLM_HIP_NO_MMA`, `FLLM_HIP_NO_SKINNY`, `FLLM_HIP_NO_Q4Z`, `FLLM_HIP_NO_Q4Z_SKINNY` | je Kernelfamilie aus |
| HIP | `FLLM_CUDA_EXACT`, `FLLM_CUDA_NO_TC`, `FLLM_CUDA_NO_ATTN4`, `FLLM_CUDA_NO_MMVQ` | gemeinsame Schalter der CUDA-Quellen |
| SYCL | `FLLM_SYCL_EXACT=1`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `FLLM_SYCL_ALL_BACKENDS=1`, `FLLM_SYCL_AOT`, `FLLM_SYCL_FORMATS` | XMX aus; Gerätewahl; alle SYCL-Backends listen; AOT-Übersetzung (ungeprüft) |
| NPU | `FLLM_NPU=0`, `FLLM_NPU_HOST=1`, `FLLM_NPU_MIN_BATCH`, `FLLM_NPU_MIN_ELEMS`, `FLLM_NPU_CHUNK`, `FLLM_NPU_GROUP`, `FLLM_NPU_BUDGET_MB`, `FLLM_NPU_CACHE_DIR`, `FLLM_NPU_PROPS`, `FLLM_NPU_DEVICE`, `FLLM_NPU_DUMP_IR` | Offload aus; Host-Haken; Planer; Quantisierung; Cache; Eigenschaften; anderes OpenVINO-Gerät zum Üben (z. B. `CPU`); IR ablegen |
| TPU | `FLLM_TPU_EXACT`, `FLLM_TPU_WEIGHTS`, `FLLM_TPU_NUMERICS`, `FLLM_TPU_DEBUG`, `FLLM_TPU_DUMP_DIR`, `FLLM_TPU_LIBRARY` | siehe 2.8 |
| ARM / x86 | `FLLM_CPU_ISA=<tier>`, `FLLM_ARM_SVE=1`, `FLLM_X86_*`, `FLLM_VNNI_GEMM`, `FLLM_AMX_MIN_BATCH` | Tier deckeln bzw. Policy überschreiben; `fllm-rs selftest-cpu --tier <tier>` |
| alle (p15) | `FLLM_AUTO_ORDER`, `FLLM_AUTO_IGPU=0`, `selftest-gpu --only/--quants/--list/--strict` | Reihenfolge von `auto`; iGPUs aus; Selbsttest eingrenzen |

Wenn ein Selbsttest-Check in einem Backend scheitert, hilft die Reihenfolge: (1) mit dem exakten Schalter wiederholen; (2) mit `--only GRUPPE --quants FORMAT` auf eine Zeile eingrenzen; (3) das Gegenstück auf der CPU-Emulation des Kerneltexts laufen lassen
(Vulkan `emu/run_kernels.sh`, Metal `emu*/run_*.sh` — beide auf einem Entwicklerrechner, nicht auf der Testhardware) und prüfen, ob die Emulation dieselbe Konfiguration besteht: besteht sie, sitzt der Fehler im Compiler-/Treiber-/Hardwareverhalten, nicht in der Index-/Layout-Logik.

## 5. Was nach der Runde zurückgeht

1. `verify-out/<backend>/` bzw. die in 2.x genannten Dateien, je Gerät.
2. Je Gerät eine Zeile: Gerät, Treiber-/ROCm-/oneAPI-/macOS-Version, Befehl, Stufe, zuletzt bestandener Schalter.
3. Die Matrix in [backends.md](backends.md) wird aus diesen Dateien angehoben (Abschnitt 3), nicht aus Erinnerung.
