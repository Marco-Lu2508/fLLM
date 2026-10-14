# Studie: AMD XDNA (Ryzen AI NPU)

Stand: Paket 11 der Mega-Runde. Prüfstufe der zugehörigen Software: **L0** (kompiliert; auf keiner Ryzen-AI-Maschine gelaufen).
Quellen: Ryzen AI Software 1.8 Dokumentation (ryzenai.docs.amd.com: Model Compilation and Deployment, Application Development, Linux
Installation Instructions), `onnxruntime_c_api.h` v1.22.0, README von FastFlowLM (github.com/ROCm/FastFlowLM). Aussagen, die nicht aus
diesen Quellen stammen, sind als „nach meiner Kenntnis“ oder „nicht geprüft“ markiert.

## 1. Zugang aus Rust

Der von AMD unterstützte Weg ist **ONNX Runtime mit dem VitisAI-Execution-Provider** (VAI EP):

- Modelle sind ONNX-Dateien (Opset 17 empfohlen). Beim ersten Laden einer `InferenceSession` wird das Modell für die NPU kompiliert; das
  Ergebnis liegt im VAI-EP-Cache oder als ORT-EP-Context-Modell. Der Graph wird vom EP automatisch in NPU- und CPU-Teilgraphen zerlegt.
- Die C-API (`OrtGetApiBase` → `OrtApi`) ist die normale ONNX-Runtime-C-API; das Ryzen-AI-`onnxruntime` ist ein Drop-in. Für den EP gibt es
  `SessionOptionsAppendExecutionProvider_VitisAI(options, keys, values, n)` (seit ORT 1.18) und den generischen
  `SessionOptionsAppendExecutionProvider(options, "VitisAI", …)`.
- Provider-Optionen (aus der AMD-Doku): `config_file` (BF16-Compileroptionen), `target` (`X1` = Phoenix/Hawk Point, `X2` = Strix/Krackan),
  `xclbin` (nur INT8-CNN auf PHX/HPT), `cache_dir`, `cache_key`, `enable_cache_file_io_in_mem`, `encryption_key`, `opt_level`.
- Geräteerkennung nach AMD über die PCI-Kennung: Vendor `0x1022`, Device `0x1502` (Phoenix/Hawk Point, XDNA 1), Device `0x17F0` mit Revision
  `0x00/0x10/0x11` (Strix, XDNA 2) und `0x20` (Krackan). Unter Linux steht das in `/sys/bus/pci/devices/*/{vendor,device,revision}`.
- Windows: Ryzen AI Software (MSI-Installer, NPU-Treiber ab 32.0.203.xxx; die Doku nennt je EP-Version Mindest-/Höchsttreiber).
  Linux: seit Ryzen AI 1.8 für Strix und neuer (Ubuntu 24.04, XRT-Pakete + `amdxdna`-Treiberpaket, Python-venv-Installation, danach
  `source /opt/xilinx/xrt/setup.sh`). macOS: nicht vorhanden.
- LLMs laufen bei AMD über `onnxruntime-genai` (OGA) mit speziell vorbereiteten Modellen (NPU-only oder Hybrid mit iGPU) und eigenen
  Custom-Ops (`ryzen_mm.dll`, `onnx_custom_ops.dll`); die Modelle müssen mit AMDs Werkzeugen (Quark) quantisiert und gebaut sein. GGUF wird
  nicht gelesen.

**Im Gerüst** (`xdna.rs`, `xdna/`): `ort.rs` bindet die ONNX-Runtime-C-API zur Laufzeit (`libloading`, nichts wird gelinkt). Die Slot-Nummern
der benutzten `OrtApi`-Einträge (Position im `struct OrtApi`, z. B. `CreateEnv` = 3, `Run` = 9, `SessionOptionsAppendExecutionProvider_VitisAI`
= 276) wurden mit einem Skript aus dem Header v1.22.0 gezählt (318 Einträge). Gesucht wird `$FLLM_XDNA_ORT_LIB`, dann
`$RYZEN_AI_INSTALLATION_PATH/deployment/onnxruntime.dll` bzw. `…/onnxruntime/lib/libonnxruntime.so`, dann der Standardname. `list()` meldet
ein Gerät, wenn die Bibliothek lädt **und** `GetAvailableProviders` den `VitisAIExecutionProvider` nennt (ein normales ORT-Build nennt ihn
nicht); der Name kommt aus der PCI-Erkennung (nur Linux). `probe_matmul(m, k, n)` baut mit `onnx.rs` ein ONNX-Modell mit einem einzigen
`MatMul` (selbst geschriebener Protobuf-Emitter, Feldnummern aus `onnx.proto3` geprüft), erzeugt die Session aus dem Speicher
(`CreateSessionFromArray`), führt sie einmal aus und vergleicht mit einer CPU-Referenz.

## 2. Datenformate

| Pfad | Format | Bemerkung |
|---|---|---|
| CNN, XINT8 / A8W8 / A16W8 | INT8 (QDQ-quantisierte ONNX-Modelle) | Compiler „X2“ (Standard) für STX/KRK, „X1“ für PHX/HPT |
| CNN, NLP-Encoder | BF16 | nur STX/KRK oder neuer; die Deployment-Version des EP kompiliert BF16 nicht zur Laufzeit, die Modelle müssen vorkompiliert mitgeliefert werden |
| LLM (OGA) | AWQ-INT4-Gewichte, BF16/INT-Aktivierungen nach AMDs Rezept | Blockgrößen/Layout bestimmt AMDs Modellbau, nicht dokumentiert |

Die Formate von fLLM (Q4_0/Q4_1/Q5_x/Q8_0, K-Quants, IQ4_NL, MXFP4, NVFP4, Q4Z, FP8M) haben im ONNX/VitisAI-Weg **keine Entsprechung**; ein
Linearlayer müsste als `MatMul` mit BF16-Gewichten oder als QDQ-INT8 neu exportiert werden. Q4Z/Q4ZP (AWQ/GPTQ) ist inhaltlich am
nächsten an AMDs eigenem INT4-Rezept, aber das Gewichtslayout der AMD-Custom-Ops ist nicht öffentlich dokumentiert (nicht geprüft).

## 3. Grenzen

- **Statische Shapes, Kompilierzeit.** Jede Session wird beim ersten Mal kompiliert (Sekunden bis Minuten je Modell; nicht gemessen). Ein
  Modell pro Gewichtsmatrix würde hunderte Sessions bedeuten; ein ganzes Modell als ein ONNX-Graph braucht einen Export mit fester
  Kontextlänge.
- **Kein Operator-für-Operator-Zugriff.** Das Backend-Trait von fLLM ruft einzelne Ops mit Geräte-Puffern auf. Über ORT gehen Daten bei
  jedem `Run` durch ORT-Tensoren (Host-Speicher); es gibt keinen öffentlichen Weg, Geräte-Puffer der NPU zu halten und zwischen Sessions
  weiterzugeben (nicht geprüft).
- **Speicher.** XDNA teilt den LPDDR-Speicher mit CPU und iGPU. Decode (eine Zeile pro Schritt) ist bandbreitenbegrenzt; die NPU hat darauf
  keinen Vorteil gegenüber CPU/iGPU, außer bei Energie.
- **Lizenz.** Ryzen AI Software steht unter AMDs eigener Lizenz; DLLs/Pakete dürfen nicht ins fLLM-Repository. Darum wird alles per
  `dlopen`/`LoadLibrary` gebunden. Die genauen Weitergabebedingungen wurden nicht geprüft.
- **Verfügbarkeit.** Windows 11 (PHX bis KRK) und Linux (ab Strix). Kein macOS. Mindest-Treiberstände siehe AMD-Doku; ein Cache darf nicht
  über EP-/Treiberversionen hinweg wiederverwendet werden.

## 4. Erwarteter Nutzen

- **Prefill:** rechenbegrenzt, hier kann eine NPU mit Dutzenden TOPS helfen – aber nur mit BF16/INT8-Gewichten in AMDs Format, nicht mit den
  GGUF-Blöcken von fLLM. Die Messung des Aufruf-Overheads (Session-`Run` für ein einzelnes `MatMul`) ist genau das, was `probe_matmul` in
  der Testrunde liefern soll.
- **Decode:** bandbreitenbegrenzt; ein NPU-Offload pro Linearlayer mit Host-Roundtrip pro Layer ist sehr wahrscheinlich langsamer als der
  AVX2/VNNI-CPU-Pfad von fLLM (Annahme, nicht gemessen). Der Nutzen wäre Energie, nicht Tempo.
- Unabhängige Referenz: **FastFlowLM** (ROCm/FastFlowLM; Orchestrierung MIT, die NPU-Kernels sind Binärdateien, laut README frei nutzbar auch
  kommerziell mit Namensnennung) führt LLMs auf XDNA 2 mit eigenen Kernels (IRON/AIE-MLIR) aus, hat Windows- und seit 03/2026 Linux-Support
  und einen OpenAI-kompatiblen Server (Port 52625). Die Tempo-/Energieangaben im README sind Herstellerangaben und nicht von mir geprüft.

## 5. Empfehlung (ehrlich)

1. **Kein XDNA-`Backend` auf Basis von ONNX Runtime/VitisAI bauen.** Weder Formate noch Aufrufmodell passen zu fLLM, und der erwartete
   Gewinn bei Decode ist null bis negativ.
2. **Wenn XDNA-Unterstützung gewünscht ist:** (a) fLLM als Frontend vor einen laufenden FastFlowLM-/Lemonade-Server stellen (anderes
   Produktmerkmal als ein Backend, kein eigener Rechenpfad), oder (b) eigene AIE-Kernels mit IRON/mlir-aie für Q4-MatMul schreiben – ein
   eigenes Projekt, kein Gerüst-Thema.
3. **Testrunde (L1):** auf einer Ryzen-AI-Maschine `probe_matmul(1, 4096, 4096)` und `probe_matmul(128, 4096, 4096)` aufrufen (siehe
   Abschnitt 6), mit `XLNX_ONNX_EP_REPORT_FILE=report.json` prüfen, ob der Knoten auf der NPU oder der CPU landete, und die Zeiten notieren.
   Erst danach entscheiden, ob Punkt 2 sich lohnt.

## 6. Prüfanleitung für die Testrunde

Voraussetzung: Ryzen AI Software installiert (Windows: `RYZEN_AI_INSTALLATION_PATH` gesetzt; Linux: venv aktiviert, XRT-`setup.sh`
gesourced, `LD_LIBRARY_PATH` auf `$RYZEN_AI_INSTALLATION_PATH/onnxruntime/lib`).

- Bauen mit `--features npu-xdna`; `fllm-rs devices` muss „AMD XDNA NPU (…)“ als Gerät vom Typ `accel` zeigen. Fehlt es: das Programm
  `create(0)` und die Fehlermeldung nennen jeden versuchten Bibliothekspfad.
- Windows: absolute Bibliothekspfade werden mit `LOAD_WITH_ALTERED_SEARCH_PATH` geladen (`xdna/dl.rs`), damit die DLLs aus dem Ryzen-AI-Ordner
  `deployment` ihre Nachbarn finden; schlägt das Laden dennoch mit Fehler 126 fehl, den Ordner in `PATH` aufnehmen.
- Optionen für den EP über Umgebungsvariablen: `FLLM_XDNA_TARGET` (`X1`/`X2`), `FLLM_XDNA_CACHE_DIR`, `FLLM_XDNA_CONFIG`.
- `backend::npu::xdna::probe_matmul(m, k, n)` liefert `ProbeReport { ort_version, api_version, providers, session_ms, run_ms, max_abs_err }`.
  Erwartet wird `max_abs_err` im Bereich von float32-Rundung, falls der Knoten auf der CPU läuft, und im Bereich von BF16-Rundung (~1e-2
  relativ), falls der EP ihn auf die NPU legt. Ein Aufrufer (z. B. ein Selftest-Zweig) ist noch nicht angeschlossen: `selftest.rs` gehört dem
  Integrator.
- Bekannte Unsicherheiten: ob der EP ein reines float32-`MatMul` überhaupt an die NPU gibt (die Doku nennt INT8-QDQ und BF16-Compile-Flows);
  ob `SessionOptionsAppendExecutionProvider_VitisAI` ohne `config_file` für BF16 genügt; ob alle 18 gebundenen `OrtApi`-Slots in der
  vom Ryzen-AI-Build angebotenen API-Version vorhanden sind (angefragt wird 22 abwärts bis 17; Slot 276 nur ab 18).
