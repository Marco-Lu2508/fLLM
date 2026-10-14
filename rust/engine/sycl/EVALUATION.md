# Bewertung: Vulkan gegen SYCL (oneAPI) für Intel-GPUs

Gegenstand: Intel Arc (Alchemist/Battlemage, dGPU), Iris Xe (Gen12, iGPU), Core-Ultra-iGPUs (Meteor Lake/Arrow Lake: Xe-LPG ohne XMX; Lunar
Lake/Panther Lake: Xe2/Xe3 mit XMX), Data-Center-GPUs (Max/PVC) am Rand. Stand: Ende dieser Programmierrunde, **alle Backends auf Prüfstufe L0**.

> **Ehrlichkeitsvermerk.** Es gibt keine einzige Messung von uns, weder für Vulkan noch für SYCL. Alles Folgende sind (a) Erwartungen aus
> Hardwaredaten und der Bauart der Kernel, (b) öffentlich berichtete Messungen anderer Projekte (mit Link, ungeprüft, andere Software, andere
> Treiber) und (c) Aufwandsschätzungen. Ein Backend „läuft“ erst ab L1. Der Messplan am Ende sagt, was Marco messen soll, um diese Seite zu
> ersetzen.

## 1. Kurzfassung und Empfehlung

1. **Vulkan (Pakete 1–4) als Standardweg für Intel-GPUs.** Er braucht nichts außer dem Grafiktreiber, läuft mit demselben Code auf
   Linux (Mesa ANV) und Windows (Intel-Treiber), auf iGPU und dGPU und gleichzeitig auf AMD/NVIDIA/Apple; die Pakete 2/3 decken bereits mehr
   Formate und Architekturen ab als SYCL (16 Formate samt Q4Z/NVFP4, Integer-Dot- und Cooperative-Matrix-Varianten, DeltaNet, Mamba, MoE).
2. **SYCL (dieses Paket) als optionales, ausdrücklich experimentelles Feature behalten, aber nicht ausbauen, bevor der Messplan ein
   Ergebnis hat.** Es lohnt sich nur, wenn es auf **Arc-dGPUs** gegenüber Vulkan unter denselben Bedingungen mindestens ca. 20 % schneller
   ist (Decode oder Prompt) und sich das auf Linux **und** Windows hält. Öffentliche Berichte zeigen beide Richtungen (Abschnitt 3.4): auf
   iGPUs liegt Vulkan gleichauf oder vorn, auf einer großen dGPU wird SYCL mit +50 % bei der Token-Erzeugung berichtet. Nur Messung entscheidet.
3. Wird SYCL nicht gebraucht, ist das Entfernen billig: ein Cargo-Feature, ein Ordner `rust/engine/sycl/`, drei Rust-Dateien, eine
   Funktion in `build.rs`; nichts davon berührt CPU oder CUDA.

## 2. Was verglichen wird

| | Vulkan (Pakete 1–4, nativ über `ash`) | SYCL (Paket 9, DPC++/oneAPI, Level Zero) |
|---|---|---|
| Programmiermodell | GLSL-Compute-Shader (SPIR-V), von `glslc` zur Bauzeit übersetzt, Kernel im Repository | C++17 mit SYCL 2020 + Intel-Erweiterungen (`joint_matrix`), von `icpx` übersetzt (SPIR-V, JIT oder AOT) |
| Matrixengine (XMX) | über `VK_KHR_cooperative_matrix` (`mm_coopmat.comp`), abhängig vom Treiber | direkt über `joint_matrix` (`linear_xmx.cpp`, Formen 8x16x16 und 8x8x16), abhängig von der Compilerversion |
| Integer-Dot (DP4A) | `mv_idot.comp`, `mm_idot.comp`, `quant_act_q8.comp` (Q8-Aktivierungen) | **nicht umgesetzt** (FP32-FMA mit Dequantisierung; DP4A ist über `sycl::ext::oneapi::dot_acc` möglich, siehe 6) |
| Formate (Gewichte) | 16 Formate in Paket 2: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K–Q6_K, IQ4_NL, MXFP4, NVFP4, Q4Z/Q4ZP (Stand der Branches beim Schreiben) | 16 GGUF-Typen (F32/F16/BF16, die obigen Legacy- und K-Formate, IQ4_NL/IQ4_XS, MXFP4); FP8, NVFP4, Q4Z werden zu f16 umgewandelt, übrige zu Q5_K |
| Operationen | Attention (Decode, Prompt, Merge), KV-Schreiben, Norm, RoPE, GLU, MoE (Top-k, Linear, Gather/Scatter, Combine), **DeltaNet, Mamba** (Paket 3) | dieselben ohne DeltaNet/Mamba/`host_linear`/Graphen; Attention als skalarer Split-KV-Kernel (kein XMX) |
| Hostanteil | eine Queue, Timeline-Semaphore, ein Command-Buffer pro Schritt geplant (Paket 4 noch offen) | je Operation ein `queue.submit` auf einer In-Order-Queue (ca. 12–20 Starts je Schicht) |
| Plattformen | Intel, AMD, NVIDIA, Apple (MoltenVK), Android | Intel-GPUs; formal auch NVIDIA/AMD über andere SYCL-Plugins (hier ungenutzt, `sub_group` 16 ist Voraussetzung) |
| Umfang (diese Runde) | Pakete 1–3 (Paket 1 allein 5,7 Tsd. Zeilen laut seinem Bericht, dazu die Shader-Sätze der Pakete 2 und 3) | ca. 2,4 Tsd. Zeilen C++ (ohne Test), ca. 1,6 Tsd. Zeilen Rust |

## 3. Erwartete Geschwindigkeit

### 3.1 Rahmenzahlen der Hardware (Herstellerangaben, gerundet)

| Gerät | Bandbreite | FP32 (Vektor) | XMX fp16 | DP4A (INT8 auf Vektor) |
|---|---|---|---|---|
| Arc A770 (Alchemist, 32 Xe-Kerne) | ca. 560 GB/s | ca. 20 TFLOPS | ca. 130 TFLOPS | ca. 4x FP32 |
| Arc B580 (Battlemage, 20 Xe2-Kerne) | ca. 456 GB/s | ca. 14 TFLOPS | ca. 110 TFLOPS | ca. 4x FP32 |
| Iris Xe (Gen12, 96 EU) | ca. 60–85 GB/s (geteilt) | ca. 2–2,5 TFLOPS | keine | ja |
| Core Ultra 7 155H (Xe-LPG, 128 EU) | ca. 90–120 GB/s (geteilt) | ca. 4–5 TFLOPS | keine | ja |
| Lunar Lake / Panther Lake (Xe2/Xe3-iGPU) | ca. 80–130 GB/s (geteilt) | 2–4 TFLOPS | ja | ja |

### 3.2 Decode (Batch 1): Speicherbandbreite entscheidet

Je erzeugtem Token wird fast die ganze Gewichtsdatei einmal gelesen. Obergrenze: `tok/s ≈ Bandbreite x Auslastung / Dateigröße`. Ein gut
gebauter Matvec-Kernel erreicht erfahrungsgemäß 55–70 % der Spitzenbandbreite; beide Wege (Vulkan-Shader und SYCL-Kernel) sind hier gleich
begrenzt, die Unterschiede kommen aus Treiber-Codegenerierung und Startaufwand, nicht aus der Programmiersprache.

| Modell (Beispiel Llama-3.1-8B, Q4_K_M, ca. 4,9 GB) | Erwartung Obergrenze bei 60 % Auslastung |
|---|---|
| A770 | ca. 65–70 tok/s |
| B580 | ca. 55 tok/s |
| Iris Xe | ca. 8–10 tok/s |
| Core Ultra 155H | ca. 10–12 tok/s |

(Rechnung, keine Messung. Q8_0 etwa halb so schnell, Q6_K/Q5_K dazwischen, F16 ein Viertel.) Unterschiede je Formatklasse bei gleicher Bandbreite:

| Formatklasse | Decode in SYCL (Paket 9) | Bemerkung |
|---|---|---|
| Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 (32er-Blöcke) | 1 Work-Item = 1 Block (16 B Quants + 2 B Skala, getrennte Arrays) → 16-Byte-Lasten, wenig ALU je Gewicht | bandbreitenbegrenzt; MXFP4/IQ4_NL mit Tabellenzugriff etwas mehr ALU |
| Q8_0 | 16 Werte je Einheit, 16 B Last | bandbreitenbegrenzt (mehr Bytes je Gewicht) |
| Q4_K, Q5_K | 16 Einheiten je Superblock, Skalen/Mins werden je Einheit aus 12 Bytes entpackt | ca. 10–15 ALU-Operationen je 16 Gewichte; auf iGPUs kann die ALU mitbegrenzen |
| Q6_K, Q2_K, Q3_K | 8 Einheiten je Superblock zu 32 Werten | wie K-Quants, etwas mehr Bit-Arbeit |
| F16/BF16/F32 | Zeilen auf 8 Werte aufgefüllt | rein bandbreitenbegrenzt |

Ein DP4A-Pfad (Aktivierungen als Int8, `dot_acc`) würde die ALU-Last der K-Quants senken und ist für iGPUs ohne XMX der Hauptgewinn,
siehe 6. Vulkan hat ihn in Paket 2 bereits als eigenen Shader.

**Startaufwand.** Ein Decode-Schritt eines 32-Schichten-Modells besteht aus ca. 400–600 Kernel-Starts. Bei 10–20 µs je Start über die
SYCL-Laufzeit (Level Zero) sind das 4–12 ms je Token — bei einem Ziel von 60–100 tok/s (10–16 ms je Token) ein großer Anteil. Abhilfe ist
die Level-Zero-Variante „immediate command lists“ (Standard in neueren Laufzeiten) oder `sycl_ext_oneapi_graph`; Paket 9 benutzt beides
nicht. Vulkan nimmt einen Command-Buffer je Schritt auf; der Hostaufwand liegt dort erwartungsgemäß niedriger. Das ist der wichtigste
Punkt, an dem SYCL in der Praxis gegenüber Vulkan zurückfällt und ein möglicher Grund für den Rückstand in der Fremdmessung (3.4) —
unbewiesen.

### 3.3 Prompt-Verarbeitung (Batch > 8): Rechenleistung entscheidet

Pro Token fallen `2 x Parameter` FLOPs an (8B-Modell: 16 GFLOP).

| Pfad | Gerät | realistischer Anteil der Spitze | grobe Erwartung (8B-Modell) |
|---|---|---|---|
| SYCL `LinTiled` (FP32, Batch 9–15 und alle Geräte ohne XMX) | A770 | 15–25 % von 20 TFLOPS | ca. 190–310 tok/s |
| | Iris Xe | 15–25 % von 2,2 TFLOPS | ca. 20–35 tok/s |
| SYCL `LinXmx` (fp16, Batch ≥ 16) | A770 / B580 | 10–25 % der XMX-Spitze (Dequantisierung und Kachelablage im lokalen Speicher begrenzen) | ca. 800–2000 tok/s |
| Vulkan `mm_coopmat` / `mm_idot` | gleiche Geräte | abhängig von Treiberunterstützung für `VK_KHR_cooperative_matrix` | ähnlich bis niedriger; auf iGPUs ohne XMX ist `mm_idot` (Integer-Dot) der bessere Pfad |

Attention im Prompt ist in beiden Wegen noch skalar (SYCL: `attention.cpp`, FP32; Vulkan: `attn_prefill.comp`). Öffentliche Messungen zeigen,
dass fehlende XMX-Flash-Attention bei langen Prompts der größte Rückstand von llama.cpp-SYCL gegenüber vLLM-XPU ist (2,4–15x im Prompt);
das gilt auch für unseren Kernel. Hier liegt der größte offene Hebel beider Wege.

### 3.4 Öffentlich berichtete Messungen (nicht von uns, andere Software)

* Panther-Lake-iGPU (Arc B390, Mesa ANV, Qwen3.6-35B-A3B Q4_K_M, llama.cpp): Vulkan pp512 633 tok/s, tg128 33,0 tok/s; SYCL (Level Zero) 615 / 30,9;
  Vulkan also +3 % im Prompt, +7 % bei der Erzeugung ([grigio.org, 16.09.2026](https://grigio.org/benchmarking-llama-cpp-backends-on-intel-panther-lake-vulkan-vs-sycl-vs-openvino-vs-cpu/)).
* Windows, Core Ultra 7 155U (iGPU), Gemma-4-E4B Q4_K_M mit MTP-Drafting: Vulkan 20,1 tok/s, SYCL 7,0 tok/s bei der Erzeugung (2,9x), SYCL im Prompt leicht vorn
  ([llama.cpp-Issue 26010](https://github.com/ggml-org/llama.cpp/issues/26010)); ein einzelner Bericht mit spekulativem Decodieren, nicht verallgemeinerbar.
* Arc Pro B70: llama.cpp-SYCL-Prompt weit hinter vLLM-XPU wegen fehlender XMX-Attention, Decode ungefähr gleich (bandbreitenbegrenzt)
  ([PMZFX, engine-comparison](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/engine-comparison.md)).
* Ein Vergleichsbericht nennt für Qwen3.6-27B auf der Arc Pro B70 +52 % Erzeugung für SYCL gegenüber Vulkan
  ([zenn.dev](https://zenn.dev/eyeofjapetus/articles/015908f77349ca?locale=en), Sekundärangabe).
* llama.cpp nutzt auf Vulkan die Cooperative-Matrix-Erweiterung statt XMX direkt; ob sie aktiv ist, hängt von Treiber und gemeldeten Features ab
  ([Diskussion 29907](https://github.com/ggml-org/llama.cpp/discussions/29907)).

Schluss daraus: **kein durchgehender Sieger.** iGPU: Vulkan gleichauf/vorn; große dGPU: SYCL-Einzelberichte vorn bei der Erzeugung; Prompt hängt an
XMX (über Cooperative Matrix oder `joint_matrix`) und an der Attention.

## 4. Treiber- und Installationsaufwand

| | Vulkan | SYCL |
|---|---|---|
| Bauen | `glslc` (im Vulkan-SDK; auf Linux `glslc` aus `shaderc`); die SPIR-V-Dateien entstehen in `build.rs` | **oneAPI-DPC++-Compiler** (`icpx`, Base Toolkit ca. 2 GB Offline-Installer, oder der Open-Source-Compiler aus intel/llvm); Version muss zu Treiber und `libsycl` passen |
| Laufzeit auf Linux | nur Mesa (ANV) oder Intel-Treiber | `libsycl` (Teil der oneAPI-Runtime), Level-Zero-Lader (`libze1`), Compute-Runtime (`intel-level-zero-gpu`/`libze-intel-gpu1`), Kernel-Treiber `xe` oder `i915`; optional AOT mit `ocloc` |
| Laufzeit auf Windows | Intel-Grafiktreiber | Intel-Grafiktreiber (enthält den Level-Zero-Lader) **plus** die oneAPI-Laufzeit-DLLs (`sycl*.dll`, UR-Adapter, SVML/IMF), müssen mit dem Programm ausgeliefert werden (ca. 100–200 MB) |
| Erster Start | Pipeline-Cache auf der Platte (Paket 1) | SPIR-V wird beim ersten Aufruf jedes Kernels vom Treiber übersetzt; viele Instanzen (ca. 150–200 Kernel) → spürbare Aufwärmzeit; AOT löst das, kostet aber eine Zielliste |
| Fehlerbild bei fehlender Installation | „Gerät nicht gefunden“, Rückfall auf wgpu/CPU | Programm läuft ohne `libsycl`/Lader nicht oder meldet „kein GPU-Gerät“ (hier getestet: das Binary meldet das sauber) |
| Container/WSL | Vulkan über `dzn`/Mesa nur eingeschränkt | berichtet: Arc-GPU in WSL2 nicht sichtbar (siehe zenn.dev oben); Docker braucht `--device=/dev/dri` |

## 5. Windows-Tauglichkeit

* **Vulkan**: Gerät und Treiber reichen; kein weiterer Bauaufwand; MSVC-Ziel unkritisch (`ash` lädt `vulkan-1.dll` zur Laufzeit).
* **SYCL**: `icx` + MSVC-Linker sind möglich, `build.rs` enthält einen ungeprüften Windows-Zweig (`/std:c++17`, `llvm-lib`, `sycl*.lib`). Offene Punkte:
  dynamische CRT (`/MD`) muss zu Rusts MSVC-Ziel passen, die Import-Bibliothek heißt je Version `sycl7.lib`/`sycl8.lib`, die DLLs müssen neben
  dem `.exe` liegen oder im `PATH`. `x86_64-pc-windows-gnu` kann SYCL nicht (kein Intel-Compiler für MinGW). Gesamtbewertung: machbar,
  aber ein eigenes Teilprojekt (Paket 15 prüft MSVC).

## 6. Wartungsaufwand und Risiken

* **Zwei Toolchains.** SYCL fügt eine zweite Kernel-Sprache und einen zweiten Compiler hinzu; Vulkan nutzt dieselbe GLSL-Basis für Intel, AMD, NVIDIA und (MoltenVK) Apple.
* **API-Fluktuation.** `joint_matrix` hat sich mehrfach geändert (Layout-Namen, `matrix_combinations`, Formen je Architektur); Code, der mit
  oneAPI 2025.x übersetzt, kann mit 2026.x neu angefasst werden müssen. Paket 9 baut gegen ein Nightly der Open-Source-Linie (clang 24) und
  schreibt die Formenabfrage defensiv (`#ifdef SYCL_EXT_ONEAPI_MATRIX`, `try`/`catch`), bleibt aber ungeprüft gegen das Release-`icpx`.
* **Formenwahl zur Laufzeit.** XMX-Form (8x16x16 mit Sub-Group 16 gegen 8x8x16 mit Sub-Group 8) hängt an der Architektur; falsche Erkennung = Kernel-Fehler beim
  Übersetzen durch den Treiber. Rückfall ist `FLLM_SYCL_EXACT=1`.
* **Nicht abgedeckt in SYCL**: DeltaNet, Mamba, Q4Z/NVFP4/FP8 nativ (Umwandlung zu f16 verdoppelt bis vervierfacht den Speicher der Gewichte), `host_linear`
  (Modelle größer als der VRAM), Graph-Wiedergabe, fused Operationen (QKV, Linear+Residual), XMX-Attention, DP4A.
* **Allokationsgrenze.** Level Zero begrenzt eine Einzelallokation standardmäßig auf ca. 4 GiB (große Vokabular-Matrizen, KV-Pools): Umgebungsvariable nötig.
* **Umfang**: Pakete 1–3 (Vulkan) enthalten bereits die breitere Kernelsammlung; SYCL würde die Pflege jedes neuen Formats/Operators verdoppeln.

## 7. Empfehlung mit Begründung

1. **Intel-Standard = Vulkan**, weil (a) null Installation über den Treiber hinaus, (b) ein Code für alle Hersteller, (c) schon vollständiger, (d) gleiche
   Decode-Obergrenze (Bandbreite), (e) auf iGPUs, dem häufigsten Intel-Gerät, öffentlich gleichauf oder vorn, (f) Windows-Auslieferung ohne
   Laufzeitpaket.
2. **SYCL nur behalten als Forschungspfad**: Er bietet (a) direkten Zugriff auf XMX ohne Abhängigkeit von der Cooperative-Matrix-Unterstützung des Treibers,
   (b) die Möglichkeit, oneMKL/oneDNN-GEMMs und Level-Zero-Funktionen einzubinden, (c) Einzelberichte von Vorsprung auf großen dGPUs. Ob das trägt, zeigt nur die Messung.
3. **Entscheidungsregel nach dem Messplan:** SYCL bleibt und bekommt DP4A, XMX-Attention und Graphen, wenn es auf mindestens einer Arc-dGPU
   (A- oder B-Serie) unter Linux **und** Windows um ≥ 20 % schneller ist als Vulkan (Median von pp512/pp2048 oder tg128 über die Modellliste unten).
   Sonst wird das Feature eingefroren (unverändert, ohne Pflege) oder entfernt.
4. Unabhängig vom Ausgang gilt: Die XMX-Attention (Prompt) ist die größte offene Leistungsreserve beider Wege.

## 8. Messplan (was Marco später messen soll)

Voraussetzung: Pakete 1–4 (Vulkan) und 9 (SYCL) sind auf demselben Rechner gebaut; erst L1 (Selftest), dann L2 (Textgleichheit), dann L3 (Zahlen).

**Rechner:** (1) eine Arc-dGPU (A750/A770/B570/B580, Linux **und** Windows), (2) ein Rechner mit Iris Xe oder Core-Ultra-iGPU, (3) wenn verfügbar Lunar/Panther Lake (Xe2/Xe3-iGPU mit XMX).
Notieren: Treiberversion (`fllm-rs devices` zeigt `driver`), oneAPI-Version, Mesa-Version, Kernel, Energieplan, Taktbegrenzung, Speicher (Kanalzahl/MT/s bei iGPU).

**Modelle:** Llama-3.1-8B-Instruct in Q4_K_M, Q8_0 und Q4_0; ein kleines Modell (Qwen3-1.7B Q8_0); ein MoE-Modell (Qwen3-30B-A3B Q4_K_M, wenn der Speicher reicht);
ein gpt-oss-Modell in MXFP4. KV-Cache f16 und q8_0.

**Schritte je Rechner:**
1. L1: `fllm-rs selftest-gpu --device sycl --json sycl.json` und `--device vulkan` (Paket 4); `cargo test --release -p fllm-engine --features sycl --lib linear_matches_cpu`;
   `tests/run_decode_check.sh` (nur zur Kontrolle der Host-Seite). Fehler zuerst mit `FLLM_SYCL_EXACT=1` eingrenzen (schaltet die fp16-XMX-Kernel ab).
2. L2: je Modell dieselben 5 Prompts mit Greedy-Dekodierung (Temperatur 0) auf CPU, Vulkan, SYCL; die ersten 64 Token müssen übereinstimmen (bei fp16-Pfaden Abweichungen nach den ersten Dutzend Token möglich: Abweichungsposition notieren).
3. L3: `fllm-rs bench` (oder `llama-bench` als Fremdreferenz auf derselben Maschine, `-p 512,2048 -n 128 -fa 1`, Tiefe 0 und 8192) je Backend, je Modell, 5 Wiederholungen, kalt (erste Messung getrennt, enthält Shader-/JIT-Übersetzung) und warm.
   Zusätzlich `llama-bench` mit `-DGGML_VULKAN=ON` und `-DGGML_SYCL=ON` als **Obergrenze der Erwartung** („beste bekannte Engine derselben Hardware“).
4. Zu notieren je Messpunkt: pp512, pp2048, tg128 (tok/s), Zeit bis zum ersten Token, Spitzenspeicher, Leistungsaufnahme (`xpu-smi`/HWiNFO), Anteil der Bandbreite
   (`tg x Dateigröße / Spitzenbandbreite`) für Decode, Anteil der Matrix-Spitze für Prompt.
5. Startaufwand getrennt: Zeit von `fllm-rs serve` bis zum ersten Token kalt/warm (SYCL: JIT; mit `FLLM_SYCL_AOT=<Architektur>` wiederholen), Zahl der Kernel-Starts je Token (Level-Zero-Trace `ZE_ENABLE_TRACING_LAYER` oder VTune/`unitrace`).
6. Windows-Zusatz: Auslieferungsgröße (Programm + DLLs), Start auf einem Rechner ohne oneAPI, Verhalten bei Treiberaktualisierung.

**Auswertung:** Tabelle Backend x Rechner x Modell mit Median und Streuung; die Entscheidungsregel aus Abschnitt 7.3 anwenden. Bei SYCL zusätzlich:
Auswirkung von `FLLM_SYCL_EXACT=1` (fp16-XMX aus) auf pp512, und von `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` auf die Ladezeit.
