# Studie: Qualcomm Hexagon NPU (HTP) über QNN und Genie

Stand: Paket 11 der Mega-Runde. Prüfstufe der zugehörigen Software: **L0** (kompiliert, Strukturgrößen per Unit-Test gegen die aus den
C-Deklarationen abgeleiteten Werte geprüft; auf keinem Snapdragon-Gerät gelaufen).
Quellen: die Header des QAIRT-SDK 2.37.0.250724 (`QnnInterface.h`, `QnnTypes.h`, `QnnGraph.h`, `QnnDevice.h`, `QnnOpDef.h`,
`HTP/QnnHtpDevice.h`, `HTP/QnnHtpGraph.h`, `HTP/QnnHtpContext.h`, `HTP/QnnHtpMem.h`, `System/QnnSystemInterface.h`, `Genie/GenieCommon.h`,
`Genie/GenieDialog.h`; core API 2.27, Genie API 1.11), gelesen aus dem öffentlich ladbaren „Qualcomm AI Runtime Community“-Paket
(`softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip`, nur die Header
wurden per HTTP-Range-Zugriff gelesen). Aussagen, die nicht aus diesen Headern stammen, sind als „nach meiner Kenntnis“ oder „nicht
geprüft“ markiert.

## 1. Zugang aus Rust

- **Lizenz zuerst:** Die SDK-Header tragen „Confidential and Proprietary – Qualcomm Technologies, Inc.“. Sie werden deshalb **nicht
  kopiert und nicht ins Repository gelegt**; `hexagon/qnn.rs` gibt nur die binäre Schnittstelle (Reihenfolge und Größe der Felder,
  Konstanten) in eigenen Rust-Strukturen wieder. Bibliotheken des SDK werden nicht gelinkt, sondern zur Laufzeit geladen.
- **QNN C-API.** Jede Backend-Bibliothek (`libQnnHtp.so` unter Linux/Android, `QnnHtp.dll` unter Windows on Snapdragon) exportiert
  `QnnInterface_getProviders(const QnnInterface_t*** list, uint32_t* n)`. `QnnInterface_t` = `{ backendId, providerName, apiVersion, Union
  der versionierten Funktionstabellen }`; die v2.x-Tabelle hat in Header-Reihenfolge `propertyHasCapability, backendCreate, …, graphExecute (21),
  tensorCreateGraphTensor (24), …, deviceCreate (40), deviceFree (43), errorGetMessage (48), …`. Das Gerüst deklariert die Tabelle bis
  `errorFreeVerboseMessage` (51 Einträge, Offsets per Unit-Test geprüft) und verlangt core API 2.≥17 (Annahme: ab da existieren alle 51
  Einträge; nicht gegen das Changelog geprüft, aktuelle SDKs liefern 2.27).
- **Ablauf für ein Modell** (so, wie `hexagon/graph.rs` es ausführt): `backendCreate` → `deviceCreate` → `contextCreate` → `graphCreate`
  (mit HTP-Config: Präzision FP16) → `tensorCreateGraphTensor` je Tensor (Eingabe `APP_WRITE`, Gewicht `STATIC` mit Client-Buffer, Ausgabe
  `APP_READ`) → `graphAddNode` (`qti.aisw`/`MatMul`, Parameter `transpose_in0/1` als `BOOL_8`) → `graphFinalize` (hier wird für die HTP
  kompiliert) → `graphExecute`. Kontexte lassen sich als Binärdatei speichern (`contextGetBinary`) und wieder laden
  (`contextCreateFromBinary`): so arbeiten die offline mit den QAIRT-Werkzeugen auf einem x86-PC erzeugten LLM-Modelle.
- **Gerätedaten.** `deviceGetPlatformInfo` liefert je HTP `deviceInfoExtension` mit `arch` (V68, V69, V73, V75, V79, V81), `socModel`,
  `vtcmSize` und Unterstützung für Signed PD/DLBC. `list()` zeigt daraus „Qualcomm Hexagon NPU (HTP v73, SoC model …)“.
- **Genie** (`Genie.dll`/`libGenie.so`, API 1.11) ist die LLM-Schicht über QNN: `GenieDialogConfig_createFromJson` + `GenieDialog_create` +
  `GenieDialog_query(dialog, prompt, sentenceCode, callback, userData)` (Streaming-Callback). Konfiguration: JSON mit Tokenizer, Sampler und
  den vorkompilierten Kontext-Binärdateien des Modells. `hexagon/genie.rs` kapselt genau diese Aufrufe (`Genie::query(json, prompt)`).
- Alternative, die ich nicht gebaut habe: der QNN-Execution-Provider von ONNX Runtime (`QNNExecutionProvider`, Backend-Typ „htp“).
  Er hätte dieselben Grenzen wie der XDNA-Weg (siehe `STUDY_XDNA.md`).

## 2. Datenformate (aus `QnnTypes.h`)

- Tensor-Datentypen: `INT_8/16/32/64`, `UINT_8/16/32/64`, `FLOAT_16/32/64`, `SFIXED_POINT_4/8/16/32`, `UFIXED_POINT_4/8/16/32`, `BOOL_8`.
- Quantisierungsarten: `SCALE_OFFSET`, `AXIS_SCALE_OFFSET` (pro Kanal), `BW_SCALE_OFFSET`, `BW_AXIS_SCALE_OFFSET` (Bitbreite + Achse),
  `BLOCK`, **`BLOCKWISE_EXPANSION`** (Achse, Blöcke pro Achse, Skalen als 8- oder 16-Bit-Speicher, Bitbreite), `VECTOR`. Datenformate
  des Tensors: `DENSE`, `SPARSE`, `CODEBOOK`, `MX`, `HMX_WEIGHT_LAYOUT`.
- Inhaltlich passt **Q4_0** (32er-Blöcke, FP16-Skala, symmetrisch) am besten auf `SFIXED_POINT_4` mit `BLOCKWISE_EXPANSION`; Q4_1/Q5_x/K-Quants/
  IQ4_NL/MXFP4/NVFP4 hätten keine direkte Entsprechung und müssten umgerechnet werden (Genauigkeitsverlust, nicht geprüft). Welche Kombinationen
  die HTP je Architektur tatsächlich als `MatMul`/`FullyConnected` annimmt, steht nicht in den Headern (Op-Definitions-Dokumentation, nicht gelesen).
- HTP-Graph-Optionen (`QnnHtpGraph.h`): Präzision (`QNN_PRECISION_FLOAT16`), VTCM-Größe, Anzahl HVX-Threads, `WEIGHTS_PACKING`,
  `SHARE_IO_BUFFER`, Parallel-Graph-Ausführung. Kontext-Optionen (`QnnHtpContext.h`): `WEIGHT_SHARING_ENABLED`, `REGISTER_MULTI_CONTEXTS`,
  `FILE_READ_MEMORY_BUDGET`, `PREPARE_ONLY`, `SHARE_RESOURCES`. Geteilte Puffer (`QnnHtpMem.h`): `SHARED_BUFFER` (fd + Offset), `WEIGHTS_BUFFER`,
  `SHARED_SPILLFILL_BUFFER`.

## 3. Grenzen

- **Statische Graphen.** Ein HTP-Graph hat feste Shapes; LLMs werden deshalb als mehrere Graphen (z. B. Prefill mit fester Chunk-Länge und
  Decode mit einem Token) in einem Kontext abgelegt, KV-Cache als Ein-/Ausgabe-Tensoren (nach meiner Kenntnis; in Genie versteckt).
- **Kompilierzeit.** `graphFinalize` kompiliert auf dem Gerät: für das Ein-Layer-Beispiel kurz, für ein ganzes Modell nicht praktikabel; für LLMs
  werden die Kontext-Binärdateien offline erzeugt. Die Gewichte stecken in der Binärdatei (Gewichtsformat = HTP-eigenes Layout).
- **Speicher.** Der Adressraum der Hexagon-DSP ist begrenzt (nach meiner Kenntnis 32 Bit, ~4 GiB je Prozessdomäne); große Modelle werden in
  mehrere Kontexte mit gemeinsam genutzten Gewichten (`WEIGHT_SHARING_ENABLED`, `REGISTER_MULTI_CONTEXTS`) zerlegt. Der Host muss Puffer
  über `rpcmem`/Shared Buffers an die DSP geben, sonst wird kopiert.
- **Lizenz.** Das SDK ist proprietär (siehe oben); es darf nicht weiterverteilt werden, der Nutzer installiert es selbst. Die Bedingungen des
  Community-Pakets wurden nicht gelesen.
- **Verfügbarkeit.** Windows on Snapdragon (X Elite/X Plus, Treiber + `QnnHtp.dll` aus dem SDK), Android, einzelne Linux-Boards (aarch64).
  Nicht auf x86-Linux/Windows ohne Qualcomm-Chip (dort gibt es nach meiner Kenntnis nur die Offline-Compiler-Bibliotheken; `list()` findet dann kein
  Gerät) und nicht auf macOS.

## 4. Erwarteter Nutzen

- **Prefill:** rechenbegrenzt; die HTP ist dafür gebaut. Mit offline erzeugten Qualcomm-Modellen (Genie) ist das der Anwendungsfall.
- **Decode:** bandbreitenbegrenzt; Vorteil vor allem bei Energie. Ein fLLM-eigener Decode, der pro Linearlayer ein `graphExecute` mit
  Host-Roundtrip aufruft, ist wegen des Aufrufaufwands (RPC zur DSP; Größenordnung nicht gemessen) sehr wahrscheinlich langsamer als der
  NEON-CPU-Pfad von Paket 13.
- Messwerte gibt es von mir nicht; Qualcomms Herstellerangaben für Genie auf Snapdragon X wurden nicht geprüft.

## 5. Empfehlung (ehrlich)

1. **Kein Hexagon-`Backend` aus fLLMs Einzel-Ops bauen** (Graphen statisch, Kompilierung offline, Gewichte im HTP-Layout, Aufrufaufwand).
2. **Wenn Hexagon gewünscht ist: Genie als „externer Generator“.** `Genie::query` zeigt, dass der Weg mit wenigen FFI-Aufrufen geht; die
   Modelle müssen aber mit Qualcomms Werkzeugen (QAIRT/AI Hub) für das jeweilige SoC erzeugt werden, nicht aus GGUF. Das ist ein anderes
   Produktmerkmal (zweite Engine hinter dem OpenAI-Server), keine Backend-Arbeit.
3. Eine eigene HTP-Pipeline (GGUF → QNN-Graph mit `BLOCKWISE_EXPANSION`-INT4-Gewichten → Kontext-Binärdatei) ist ein großes Einzelprojekt mit
   fraglichem Gewinn gegenüber Genie. Erst nach den Messungen aus Abschnitt 6 entscheiden.

## 6. Prüfanleitung für die Testrunde

Voraussetzung: QAIRT-SDK installiert; `QNN_SDK_ROOT` gesetzt (oder `FLLM_QNN_HTP_LIB` auf `libQnnHtp.so`/`QnnHtp.dll`); unter Android/Linux
die zur SoC-Architektur passenden `libQnnHtpV*Stub.so`/`…Skel.so` über `ADSP_LIBRARY_PATH` auffindbar (nach meiner Kenntnis).

- Windows: absolute Bibliothekspfade werden mit `LOAD_WITH_ALTERED_SEARCH_PATH` geladen (`xdna/dl.rs`); bei Fehler 126 den `lib`-Ordner des SDK in `PATH` aufnehmen.
- Bauen mit `--features npu-qnn`; `fllm-rs devices` muss „Qualcomm Hexagon NPU (HTP v…)“ zeigen. Bei Fehlern zeigt `create(0)` jeden
  versuchten Pfad bzw. den QNN-Fehlercode samt `errorGetMessage`-Text.
- `backend::npu::hexagon::probe_matmul_f16(m, k, n)` baut/kompiliert/führt `y = x · wᵀ` in FP16 aus und gibt `ProbeReport { provider,
  api_version, finalize_ms, run_ms, max_abs_err }`. Erwartet: `max_abs_err` im Bereich von FP16-Rundung (≈1e-2 bei k = 4096 und Eingaben
  in [-1, 1]). Unsicher sind: ob die HTP ein FP16-`MatMul` mit statischem Gewicht und `transpose_in1 = true` annimmt (sonst Fehlercode von
  `graphAddNode`/`graphFinalize`), ob `deviceCreate` mit Null-Konfiguration reicht, ob der Client-Buffer des statischen Tensors bei `tensorCreate`
  kopiert wird (das Gerüst hält ihn bis zum Funktionsende am Leben) und ob für `Qnn_Tensor_t` Version 1 genügt.
- `backend::npu::hexagon::genie::genie()?.query(json, prompt)` für ein vorhandenes Genie-Modell (JSON-Konfiguration nach der Genie-Dokumentation).
