# Intel-NPU über OpenVINO in fLLM

Forschung, Entwurf, Grenzen und Messplan für Feature `npu-openvino` (Paket 10). **Prüfstufe L0**: alles hier ist übersetzt und mit
hardwarefreien Tests geprüft; es lief nie auf einer NPU, nie gegen ein Modell, und nichts davon wurde gemessen. Wo Zahlen stehen,
sind sie entweder zitiert (mit Quelle) oder als *Annahme* gekennzeichnet.

## 1. Kurzfassung

* Die Intel-NPU ist ein Matrixmultiplizierer mit niedriger Leistungsaufnahme (Meteor Lake / Arrow Lake-U: ca. 9,5 TOPS INT8, halb so
  viel FP16; Lunar Lake: 48 TOPS). Sie teilt sich den Arbeitsspeicher mit CPU und iGPU und braucht **statische Shapes**.
* Sie lohnt sich für fLLM **nur für das Prefill** (viele Prompt-Zeilen pro gelesenem Gewicht, rechenbegrenzt). Für **Decode**
  (eine Zeile pro Gewicht, speicherbandbreitenbegrenzt) bringt sie nichts und ist durch die Aufrufkosten eher langsamer als die CPU.
* Gegenüber der **iGPU** (Xe) ist kein Geschwindigkeitsgewinn zu erwarten, sondern Energie- und Entlastungsvorteile (die NPU nahm in einer
  Messung ≤ 7 W auf, die iGPU bis 20 W). Wer auf Intel-Laptops Tempo will, bekommt es eher über die iGPU-Backends (Vulkan, SYCL).
* Gebaut wurde deshalb **kein ganzes Modell auf der NPU**, sondern ein *Matmul-Offload*: `y = x·Wᵀ` für große Projektionen ab einer
  Mindestzeilenzahl, Gewichte als INT4/INT8 (symmetrisch, gruppenweise) oder FP16 im OpenVINO-Graphen; alles andere bleibt auf der CPU.
* Zwei Wege in den Decoder: (a) `--device npu`: ein Hybrid-Backend (CPU-Backend + NPU für `quantized_linear`), vollständig vorhanden, aber
  mit der langsamen Referenz-CPU für alles andere; (b) ein Haken im Host-Decoder (`Linear::forward`), der die schnellen CPU-Kerne behält und nur
  die Prompt-Matmuls an die NPU gibt. (b) ist der sinnvollere Weg; sein Patch liegt bei (Abschnitt 7), ist übersetzt, aber nicht angewendet.

## 2. Hardware

| NPU (Intel-Name) | Plattformcode im Treiber | Prozessoren | INT8-Spitze | Quelle / Status |
|---|---|---|---|---|
| NPU 3720 | `3720` | Core Ultra Serie 1 (Meteor Lake), mutmaßlich auch Core Ultra 200U/200H-Refreshes mit „13 TOPS" | 2 NCE-Tiles × 2048 INT8-MAC je Takt, 1,16 GHz → 9,5 TOPS (Marketing: 11–13) | Chips and Cheese, „Intel Meteor Lake's NPU" (2024) |
| NPU 4 | `4000` | Core Ultra Serie 2 (Lunar Lake) | 48 TOPS (laut Hersteller; die Tile-Zahl ist hier nicht nachgeprüft) | TechPowerUp, Lunar-Lake-Deep-Dive |
| NPU 5 / 5020 / 6 | `5010`, `5020`, `6010` | spätere Generationen | nicht untersucht | Liste der Plattformen in `src/plugins/intel_npu/README.md` (OpenVINO master) |

Die Zielhardware (Core Ultra, Intel NPU der 3720-Klasse) ist **nicht auf der NPU geprüft**. Ob und welche NPU vorhanden ist, zeigt
`fllm-rs devices` (mit `--features npu-openvino`; nur wenn Treiber und OpenVINO-Laufzeit installiert sind) und
`cargo test --features npu-openvino -- --ignored npu_hardware_selfcheck`.

Aufbau der 3720 laut Chips and Cheese: zwei *Neural Compute Engines* (Tiles), jede mit MAC-Array (FP16 mit halber INT8-Rate → ca. 4,7 TFLOPS
FP16-Spitze) und **2 MB softwareverwaltetem SRAM**, daneben Movidius-**SHAVE**-DSPs für alles, was nicht auf das MAC-Array passt (FP32,
Transzendente, Typumwandlung). Die NPU hängt am Scalable Fabric und teilt sich den LPDDR/DDR-Speicher mit allem anderen; der
OpenCL-Kopierdurchsatz lag unter 10 GB/s (iGPU: > 19 GB/s). Gemessener FP16-MatMul-Durchsatz über die NPU-Beschleunigungsbibliothek:
**1,35 TFLOPS** (28 % der Spitze); bei sehr großen Matrizen (7168²) brach er ein, und der iGPU-Wert war bei großen Größen ähnlich.

## 3. Was die NPU rechnet und was wir davon nutzen

Der NPU-Compiler (im Treiber oder im Plugin) übersetzt einen OpenVINO-Graphen in statische Kommandostränge. Die MAC-Arrays (DPU) rechnen
**MatMul/Convolution in INT8 oder FP16**; Elementweises, Softmax, Normalisierung, Typumwandlungen laufen auf den SHAVE-DSPs. Offiziell
unterstützte Datentypen für die Berechnung: FP32 (wird nach FP16 gewandelt), FP16, U8-Quantisierung; die Rechenpräzision der Hardware ist FP16
(OpenVINO-Dokumentation „NPU Device"). Für LLMs verlangt OpenVINO GenAI **symmetrische** 4-Bit-Gewichte (INT4 oder NF4, NF4 erst ab NPU 4),
kanalweise oder in Gruppen von 128, FP16-Aktivierungen.

Wir nutzen genau ein Muster, das NNCF/OpenVINO für komprimierte Gewichte schreibt:

```
Const(i4|i8 [N, G, GS]) → Convert(f16) → Multiply(Const f16 [N, G, 1]) → Reshape [N, K] ─┐
Parameter x f16 [B, K] ─────────────────────────────────────────────────────────────────┴→ MatMul(transpose_b) → Result y f16 [B, N]
```

(kanalweise: ohne Reshape, Skalen `[N, 1]`; FP16-Gewichte: eine einzige Konstante). Aktivierungen und Ergebnis sind f16; die Umwandlung
von und nach f32, Bias und der Tensor-Skalierungsfaktor bleiben auf dem Host.

## 4. Grenzen

* **Statische Shapes.** Jedes kompilierte Modell hat feste Zeilenzahl `B` (Standard 128, `FLLM_NPU_CHUNK`). Ein Prompt wird in volle
  Blöcke zerlegt; ein Rest wird mit Nullen aufgefüllt, wenn er mindestens `FLLM_NPU_MIN_BATCH` (64) Zeilen hat, sonst rechnet ihn die CPU.
* **Kompilierzeit.** Jede Projektion ist ein eigenes Modell (Gewichte als Konstanten). Ein Modell mit 32 Schichten hat über 200 davon.
  Die Kompilierzeit je Modell ist **nicht gemessen**, und ich kenne keine belastbare Zahl für diese Modellform; sie ist der wichtigste Messpunkt (Abschnitt 8). `CACHE_DIR` (Standard `~/.cache/fllm/npu`, abschaltbar mit `FLLM_NPU_CACHE_DIR=`) macht den zweiten Start billiger. Die
  Kompilierung geschieht beim ersten großen Prompt, nicht beim Laden; ein Vorwärmen ist nicht eingebaut.
* **Speicher.** Das kompilierte Modell enthält die Gewichte (komprimiert) im NPU-zugänglichen Speicher; dazu kommt die Host-Kopie. Im Hybrid-Backend
  existieren die Gewichte außerdem doppelt (CPU-Puffer und Registrierung). Eine Budgetgrenze gibt es (`FLLM_NPU_BUDGET_MB`, sonst 75 % von
  `NPU_DEVICE_TOTAL_MEM_SIZE`); Projektionen über dem Budget bleiben auf der CPU.
* **Genauigkeit.** Die NPU rechnet in FP16 (Akkumulation intern nicht dokumentiert). Aktivierungen mit |x| > 65504 (außerhalb von f16) oder NaN/Inf und f16-Überläufe im
  Ergebnis führen zum CPU-Fallback für diesen Aufruf. Q4_0 und Q8_0 werden **exakt** umgerechnet (gleiche Werte wie die CPU-Dequantisierung);
  alle anderen Formate werden aus den entquantisierten Werten **neu quantisiert** (INT4 für ≤ 4,5 bit, sonst INT8; Gruppe 128, `FLLM_NPU_GROUP`).
  Asymmetrische K-Quant-Anteile (Minima) gehen dabei verloren; die asymmetrische u4-Form wäre in OpenVINO möglich, wird aber bewusst nicht
  erzeugt, weil der NPU-Compiler symmetrische Gewichte bevorzugt.
* **Formate.** Nicht umgerechnet werden gerätespezifische Layouts (`Nvfp4Marlin`, `Q4_K_Marlin`, `Fp8E4m3/E5m2` ohne Rust-Dequantisierer, `Other`). Die
  Q4ZP-Formate (GPTQ mit Permutation) werden im Hybrid-Backend nicht registriert.
* **Stateful-KV / Attention.** OpenVINO GenAI bringt LLMs auf die NPU, indem es Prefill und Decode als statische Graphen mit eingebautem KV-Cache
  kompiliert (`MAX_PROMPT_LEN`, `MIN_RESPONSE_LEN`, Prefill in Blöcken `NPUW_LLM_PREFILL_CHUNK_SIZE`). Das passt nicht zu fLLMs geteiltem,
  seitenweisem KV-Cache (`DeviceKvCache`): er müsste in einen statischen NPU-Graphen kopiert werden. Wir lassen Attention deshalb auf der CPU.
* **Versionen.** `ov_element_type_e` wurde in OpenVINO 2025.2 neu nummeriert; die Nummern werden zur Laufzeit ermittelt (Abschnitt 6). Die C-API wird ab 2024.0
  vorausgesetzt (`ov_core_read_model_from_memory_buffer`); mit älteren Laufzeiten geht der Weg über temporäre Dateien. NPU-Treiber:
  OpenVINO empfiehlt 32.0.100.3104 oder neuer (GenAI-Hinweis); Kompatibilität zwischen OpenVINO- und Treiberversion ist nicht garantiert.
* **Ein Gerät.** Das NPU-Plugin unterstützt nach seiner Dokumentation derzeit ein einziges Gerät (`NPU`).

## 5. Erwarteter Gewinn: Prefill gegen Decode

*Alles Folgende ist eine Abschätzung aus Spezifikationen und fremden Messungen, keine Messung von fLLM.*

**Decode.** Je Token wird jedes Gewicht einmal gelesen: 7 Mrd. Parameter in Q4 sind ≈ 3,8 GB. Bei 60–90 GB/s nutzbarer Speicherbandbreite
(Laptop-DDR5/LPDDR5) ist die Obergrenze 15–25 Token/s, egal welcher Rechenkern die Summe bildet. Die NPU kann die Bandbreite nicht besser nutzen
als die CPU (Kopierdurchsatz < 10 GB/s, siehe oben), und jeder Aufruf kostet feste Startzeit; bei 7 Projektionen × 32 Schichten wären das > 200
Aufrufe je Token. Fremde Messungen zeigen dasselbe Bild: in einer OpenVINO-Diskussion (Lunar Lake, Discussion #36484) ist ein kleines Draft-Modell
auf der NPU 3,6× langsamer pro Token als auf der CPU; eine Mobil-Studie (Qualcomm, nicht Intel) beschreibt eine „Phasenteilung": NPUs
sind im rechenbegrenzten Prefill vorn, die CPU im speicherbegrenzten Decode. **Erwartung für fLLM: kein Gewinn, vermutlich Verlust → Decode bleibt auf der CPU.**

**Prefill.** Rechnerisch 2·Parameter FLOP je Token (7 Mrd. → 14 GFLOP). Bei der gemessenen 1,35 TFLOPS (NPU 3720, FP16-MatMul, große Matrizen) wären das
höchstens ≈ 95 Token/s für die Matmuls; bei der Spitze 4,7 TFLOPS ≈ 330 Token/s; NPU 4 hat die vierfache Spitzenleistung. Die CPU
(AVX2/AVX-VNNI-INT8-Kerne von fLLM, `docs/cpu-kernels.md`) liegt auf Laptop-Prozessoren je nach Kernzahl in einer ähnlichen oder niedrigeren
Größenordnung; ob der Unterschied Faktor 1 oder 3 ist, lässt sich nicht aus der Literatur ablesen. Die Standardannahmen des Planers
(`CPU 400 GOPS`, `NPU 1500 GOPS`, `500 µs je Aufruf`) sind **Platzhalter**, die der Messplan (Abschnitt 8) ersetzen soll. Gegenüber der iGPU
(Xe-LPG/Xe2 mit eigenen Matrixeinheiten) ist beim Prefill kein Gewinn zu erwarten: Chips and Cheese fand die iGPU bei großen MatMuls ähnlich schnell
und bei der Stable-Diffusion-Last 62 % schneller als die NPU (FP32-Modell der iGPU gegen INT8-Blob der NPU).

**Fazit.** Ein Gewinn gegenüber der CPU ist für große Prompts (≥ 128 Token) und NPU 4 plausibel, auf der NPU 3720 unsicher; gegenüber der
iGPU gibt es höchstens Energie- und Entlastungsvorteile (die CPU bleibt frei für den Rest). Die Entscheidung über „bauen oder lassen" sollte erst
nach der Messung in Abschnitt 8 fallen. Die Implementierung ist so geschnitten, dass sie ohne Eingriff in die CPU-Pfade zuschaltbar ist
(`FLLM_NPU=0` schaltet jeden Offload ab, der Standardbuild enthält nichts davon).

## 6. Aufbau der Implementierung

Dateien (alle unter `rust/engine/src/backend/npu/`, nur mit Feature `npu-openvino`):

| Datei | Inhalt |
|---|---|
| `openvino/ffi.rs` | `dlopen`/`LoadLibrary` von `libopenvino_c.so` / `openvino_c.dll` (kein Link), Bindungen der benötigten C-Funktionen, Suche der Bibliothek (`FLLM_OPENVINO_C_LIB`, `FLLM_OPENVINO_DIR`, `INTEL_OPENVINO_DIR`, `LD_LIBRARY_PATH`, `PATH`, `VIRTUAL_ENV`, `/opt/intel/openvino*`), Ermittlung der Element-Typ-Nummern |
| `openvino/session.rs` | RAII-Hüllen: Core, Tensor, kompiliertes Modell + Inferenzanfrage; IR aus dem Speicher lesen (Fallback über Dateien), `compile_model` mit Eigenschaften (variadisch) |
| `openvino/ir.rs` | Schreiber für OpenVINO-IR v11 (XML + BIN), nur eigene Testparser, keine Abhängigkeit |
| `openvino/model.rs` | der Matmul-Graph (Abschnitt 3) |
| `openvino/quant.rs` | Umrechnung aus fLLM-Formaten in INT4/INT8/FP16 für die NPU |
| `openvino/plan.rs` | `OffloadPlan`, `should_offload`, `plan` (Auswahl der Projektionen unter einem Speicherbudget) |
| `openvino/engine.rs` | `NpuEngine`: Registrierung, verzögertes Kompilieren, Ausführung in Blöcken, CPU-Fallback, `offload_host`, `selfcheck` |
| `openvino/devices.rs` | Geräteliste (`FULL_DEVICE_NAME`, `DEVICE_ARCHITECTURE`, `NPU_DRIVER_VERSION`, `NPU_DEVICE_TOTAL_MEM_SIZE`) |
| `backend.rs`, `backend/hybrid.rs` | Anbieterwahl (`--device npu[:N]`) und das Hybrid-Backend |
| `device_info.rs` | Geräteinventar aller Anbieter |

**Umrechnung der Formate.**

| fLLM-Format | NPU-Form | Genauigkeit |
|---|---|---|
| Q4_0 | INT4 symmetrisch, Gruppe 32, f16-Skala | exakt (bitgleich zur CPU-Dequantisierung, getestet) |
| Q8_0 | INT8 symmetrisch, Gruppe 32 | exakt (getestet) |
| Q4_1, Q2_K, Q3_K, Q4_K, IQ*, MXFP4, NVFP4, Q4Z* | INT4 symmetrisch, Gruppe `FLLM_NPU_GROUP` (128), RTN | neu quantisiert |
| Q5_0, Q5_1, Q5_K, Q6_K, Q8_1, Q8_K, FP8M | INT8 symmetrisch, Gruppe 128 | neu quantisiert, Fehler ≈ ½ INT8-Schritt |
| F16, BF16, F32 | FP16 | Wertebereich geprüft (Inf → Ablehnung) |
| ungerade Spaltenzahl bei 4-Bit-Formaten | INT8 statt INT4 | wie oben |

**Entscheidung je Aufruf** (`OffloadPlan::should_offload`): Projektion ≥ `FLLM_NPU_MIN_ELEMS` (1 Mi) Gewichte und konvertierbar; mindestens
`FLLM_NPU_MIN_BATCH` Zeilen in einem Block oder Rest; geschätzte NPU-Zeit (Aufrufzeit + Rechenzeit bei `FLLM_NPU_GOPS`) unter der CPU-Zeit
(`FLLM_NPU_CPU_GOPS`). Fehler (Kompilierung, Ausführung, Wertebereich) fallen still auf die CPU zurück (einmal protokolliert); drei Kompilierfehler
in Folge schalten die NPU für den Prozess ab.

**Umgebungsvariablen.**

| Variable | Wirkung |
|---|---|
| `FLLM_NPU=0` | Offload aus |
| `FLLM_NPU_HOST=1` | Host-Decoder-Haken aktiv (Abschnitt 7) |
| `FLLM_NPU_MIN_BATCH`, `FLLM_NPU_CHUNK`, `FLLM_NPU_MIN_ELEMS`, `FLLM_NPU_BUDGET_MB` | Planer, Abschnitt 6 |
| `FLLM_NPU_GROUP` | Gruppengröße neu quantisierter Gewichte (0 = kanalweise); schaltet auch die exakte Q4_0/Q8_0-Umrechnung ab |
| `FLLM_NPU_CPU_GOPS`, `FLLM_NPU_GOPS`, `FLLM_NPU_OVERHEAD_US` | Raten des Kostenmodells (Platzhalter) |
| `FLLM_NPU_CACHE_DIR` | Verzeichnis für `CACHE_DIR` von OpenVINO (leer = aus) |
| `FLLM_NPU_PROPS` | zusätzliche Compile-Eigenschaften `KEY=VAL,KEY=VAL` (höchstens zwei; z. B. `NPU_TILES=2`) |
| `FLLM_NPU_DEVICE` | anderer OpenVINO-Gerätename statt `NPU` (nur zum Üben des Ablaufs, z. B. `CPU`) |
| `FLLM_NPU_DUMP_IR` | Verzeichnis, in das jede kompilierte IR als `.xml`/`.bin` geschrieben wird |
| `FLLM_OPENVINO_C_LIB`, `FLLM_OPENVINO_DIR` | Ort der Laufzeit |

**Was ohne Hardware geprüft wurde** (alle Ausgaben stehen im Bericht):
* `cargo check`/`clippy -D warnings` ohne Feature und mit `npu-openvino`, `cargo check` für `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-gnu`,
  `aarch64-apple-darwin` (mit `npu-openvino`) und mit allen vier NPU-Features gegen Ersatz-Dateien von Paket 11;
* 39 Unit-Tests (IR-XML wohlgeformt und konsistent, Offsets im BIN, Umrechnung Q4_0/Q8_0 exakt, Planer, Hybrid-Backend gegen das CPU-Backend);
* **IR-Parser-Prüfung mit der echten OpenVINO-Laufzeit (pip-Paket `openvino==2026.4.1`, nur CPU-Plugin)**: `read_model` akzeptiert alle fünf
  Beispielgraphen; nach `ConstantFolding` (OpenVINOs eigener Code, kein Gerät) stimmen die entfalteten f16-Gewichte elementweise mit der
  Rust-Dequantisierung überein (größter Abstand = f16-Rundung). Das belegt Nibble-Reihenfolge, Skalen-Layout, Reshape und Gruppierung.
  Der Test `openvino_reader_accepts_generated_ir` (ignoriert, braucht die Bibliothek) lädt dazu die C-API über unseren `dlopen`-Pfad,
  ermittelt die Typ-Nummern (2026.4.1: f16 = 3, f32 = 4), zählt Geräte (`CPU`) und lässt den Leser die Graphen und einen absichtlich kaputten prüfen.
  **Nicht** geprüft: `compile_model` (auch nicht auf dem CPU-Plugin), Inferenz, NPU-Compiler.

## 7. Einbindung in den Decoder

**Was `rust_decoder.rs` kann:** `init_backend` nimmt *ein* vollständiges `Backend`; ein Mechanismus, einzelne Operationen auf ein anderes Gerät
auszulagern, existiert nicht (`docs/gpu-backends.md` beschreibt einen Fallback mit `warn_fallback_once`; im Decoder habe ich keinen Aufrufer davon gefunden; nur
einzelne Pfade — rekurrente Schichten, `host_linear` für gestreamte Gewichte — laufen auf dem Host). Zwei Wege:

1. **`--device npu` (fertig, Hybrid-Backend).** `create_native_backend("npu", N)` liefert `NpuHybridBackend`: das Referenz-CPU-Backend (`backend/cpu.rs`) mit
   `quantized_linear` über die NPU-Engine. Der Rest (Normen, RoPE, Attention, KV-Schreiben, Sampling) ist die Referenzimplementierung des CPU-Backends —
   **einkernig und nicht optimiert**; Decode ist deshalb deutlich langsamer als `--device cpu`. Zeilen unterhalb der Schwelle rechnen die schnellen CPU-Kerne
   (`cpu_quantized::linear`) aus der Registrierungskopie der Gewichte. Einschränkungen: nur dichte Modelle (CPU-Stream hat weder Delta-, noch Gate-Kernel, noch
   `argmax_rows`; `backend_unsupported` weist sie ab), doppelter Gewichtsspeicher, keine MoE-Beschleunigung. `selftest-gpu --device npu` prüft das Backend
   gegen das CPU-Backend (Hinweis: schon das CPU-Backend selbst meldet dabei 7 fehlgeschlagene q8-Attention-Prüfungen, weil die Toleranz enger ist als der q8-Cache-Fehler; der Unit-Test des Hybrid-Backends verlangt dasselbe Urteil wie für das CPU-Backend); mit `FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1` läuft jede Prüfung über die NPU (jede Form kompiliert ein eigenes Modell).
2. **Haken im Host-Decoder (empfohlen, als Patch beigelegt).** `NpuEngine::offload_host(quant, raw, rows, cols, x, batch)` nimmt die gemappten GGUF-Bytes, kopiert nichts und
   liefert die ersten `rows_done` Zeilen; der Rest läuft wie bisher auf den schnellen CPU-Kernen. Eingefügt wird er in `Linear::forward` (`compute.rs`) vor dem
   `fast_cpu`-Zweig und in `Linear::forward_group` (die gruppierten Q8-/K-Kerne teilen eine Aktivierungs-Packung; bei NPU-tauglichen Batches wird
   projektionsweise gerechnet). Aktivierung: `FLLM_NPU_HOST=1`. Der Patch ist mit `--features npu-openvino` und ohne Feature übersetzt und clippy-sauber, aber nicht
   angewendet worden (gemeinsame Datei); siehe Bericht „GEMEINSAME-ZEILEN". Offene Punkte: der Haken sitzt in Layer-Schleifen, die Bias/Skala berücksichtigen
   müssen (im Patch enthalten); MoE-Experten (`forward_experts`) und der Ausgabekopf (`head`) sind nicht erfasst.

## 8. Prüf- und Messplan (für die Testrunde)

**L1 (läuft sie?)** — Voraussetzung: NPU-Treiber (Linux: intel/linux-npu-driver; Windows: Intel-NPU-Treiber), OpenVINO ≥ 2025 mit NPU-Plugin
(`pip install openvino` oder Archiv).
1. `python3 -c "import openvino as ov; print(ov.Core().available_devices)"` muss `NPU` enthalten.
2. `FLLM_OPENVINO_C_LIB=…/libopenvino_c.so… cargo test --release -p fllm-engine --features npu-openvino -- --ignored --nocapture` (zwei ignorierte Tests:
   Leser-Prüfung ohne Gerät, `npu_hardware_selfcheck` mit Gerät: kompiliert eine 192×512-Q4_0-Projektion, vergleicht mit der Host-Rechnung, Schwelle 2 %).
3. `fllm-rs devices` (Eintrag „OpenVINO NPU, platform …").
4. `FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1 fllm-rs selftest-gpu --device npu --json npu.json`.
5. Bei Fehlern: `FLLM_NPU_DUMP_IR=/tmp/ir` und die Dateien mit `benchmark_app -m x.xml -d NPU` und `ov.Core().compile_model(model, "NPU")` prüfen; `OV_NPU_LOG_LEVEL=LOG_INFO`.

**L2 (richtiger Text?)** — Greedy-Text eines kleinen Q4_0-Modells (z. B. 1–3 B) mit `FLLM_NPU=0` und mit `FLLM_NPU_HOST=1` (bzw. `--device npu`)
vergleichen; Abweichungen im Logit-Bereich durch f16-Rechnung sind zu erwarten (Argmax darf nur im Rauschbereich kippen). Zusätzlich je Projektion die Abweichung NPU gegen CPU protokollieren.

**L3 (schneller?)** — Messungen in dieser Reihenfolge; die Ergebnisse ersetzen die drei Platzhalter des Planers:
1. *Mikrobenchmark Matmul*: 4096×4096 und 14336×4096, Q4_0, Batch 16/32/64/128/256/512, auf NPU (`engine.run`), CPU (`cpu_quantized::linear`) und iGPU
   (Vulkan/SYCL-Backend, falls vorhanden): Zeit je Aufruf, GOPS; daraus `FLLM_NPU_GOPS`, `FLLM_NPU_CPU_GOPS`, `FLLM_NPU_OVERHEAD_US`.
2. *Kompilierzeit* je Projektionsform ohne und mit Cache, sowie Gesamtzeit bis zum ersten Token für ein 7B-Modell; Speicherbedarf (RSS und
   `NPU_DEVICE_ALLOC_MEM_SIZE`); größte Zahl gleichzeitig geladener Modelle, bevor der Treiber ablehnt.
3. *Prefill-Durchsatz* (Token/s) für Prompts von 128/512/2048 Token: CPU allein, CPU + NPU, iGPU allein; dazu Energie (RAPL / `powertop` / Windows-Leistungsmesser).
4. *Decode* mit und ohne Offload (muss gleich bleiben; Beweis, dass die Schwelle den Decode nicht berührt).
5. *Genauigkeit*: Perplexität auf einem festen Text und Logit-Abstand (Max/RMS) gegen die CPU, für Q4_0 (exakte Gewichte, nur f16-Rechnung) und Q4_K_M
   (neu quantisiert).
6. *Gruppengröße*: Gruppe 32 (exakt) gegen 128 und kanalweise: Geschwindigkeit und Genauigkeit.
Go/No-Go: Der Offload bleibt nur, wenn (3) bei Prompts ≥ 512 Token mindestens 1,3× gegenüber CPU allein zeigt oder bei gleicher Geschwindigkeit deutlich weniger
Energie braucht; sonst bleibt `FLLM_NPU` standardmäßig aus und das Feature ein Experiment.

## 9. Ungeprüfte Annahmen und Risiken

* Der NPU-Compiler erkennt das Dekompressionsmuster bei **Gruppe 32** und bei **FP16-Eingang/-Ausgang** genauso wie bei Gruppe 128 (NNCF-Standard); Gruppe 32 könnte langsam sein.
* `Reshape` nach `Multiply` auf der NPU: dieses Muster erzeugt NNCF ebenfalls; die Unterstützung durch alle Treiberversionen ist ungeprüft.
* Eigenschaften beim Kompilieren: nur `CACHE_DIR` (und die Nutzerliste); `PERFORMANCE_HINT`, `NPU_TILES`, `NPU_COMPILATION_MODE_PARAMS` (z. B. `optimization-level=2`)
  sind nicht gesetzt, weil ihre Wirkung unbekannt ist.
* Gleichzeitiger Zugriff: pro Projektion gibt es genau eine Inferenzanfrage unter Mutex; der Scheduler ruft Prefill-Projektionen nacheinander auf, eine Parallelisierung
  über Projektionen wurde nicht versucht.
* Die Windows-Pfadbehandlung (`LOAD_WITH_ALTERED_SEARCH_PATH`, TBB-Verzeichnis im `PATH`) ist nur für `x86_64-pc-windows-gnu` übersetzt, nicht für MSVC und nicht ausgeführt.
* Das Hybrid-Backend wurde nie mit einem Modell betrieben; welche Modelle `init_backend` akzeptiert, ergibt sich aus `backend_unsupported` und der Ausstattung des CPU-Streams.
* Die Annahmen über die Leistungszahlen in Abschnitt 5 beruhen auf Spezifikationen und fremden Messungen anderer Software-Stacks.

## 10. Quellen

* OpenVINO, `src/bindings/c/include/openvino/c/*.h` (master 2026.x; Tags 2023.3, 2024.0, 2025.0, 2025.2, 2025.4 für die Nummerierung von `ov_element_type_e` und die Lese-Funktionen).
* OpenVINO, `src/plugins/intel_npu/README.md` (Eigenschaften, Plattformcodes), Dokumentation „NPU Device" und „OpenVINO GenAI on NPU" (docs.openvino.ai/2025).
* OpenVINO, `src/frontends/ir/tests/frontend_test_basic.cpp` (IR-Beispiel mit u4-Konstante), `src/core/dev_api/openvino/core/type/element_iterator.hpp` (Packreihenfolge der 4-Bit-Typen).
* Chips and Cheese, „Intel Meteor Lake's NPU" (April 2024); TechPowerUp, „Intel Lunar Lake Technical Deep Dive".
* OpenVINO GitHub Discussion #36484 (NPU-Draft-Modell auf Lunar Lake); arXiv 2607.05475 (Mobil-NPU-Studie, Phasenteilung Prefill/Decode — Qualcomm-Hardware, nur als Hinweis).
