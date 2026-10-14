# Studie: Apple Neural Engine (ANE) über Core ML

Stand: Paket 11 der Mega-Runde. Prüfstufe der zugehörigen Software: **L0** (kompiliert, für `aarch64-apple-darwin` typgeprüft; die reinen
Rust-Teile (MIL-Text, `.mlpackage`-Writer) sind per Unit-Test auf Byte-Ebene geprüft; nie gegen Core ML gelaufen).
Quellen: die `.proto`-Dateien von `apple/coremltools` (BSD-3-Clause: `Model.proto`, `MIL.proto`, `FeatureTypes.proto`) und
`MILBlob/Blob/StorageFormat.hpp`/`BlobDataType.hpp`, `modelpackage/src/ModelPackage.cpp`. Aussagen über Core ML selbst (Verhalten, Grenzen,
Versionsstände) sind, wo sie nicht aus diesen Dateien stammen, „nach meiner Kenntnis“ und nicht an Apples Dokumentation geprüft.

## 1. Zugang aus Rust

- **Es gibt keine öffentliche ANE-API.** Der einzige unterstützte Zugang ist Core ML: Der Entwickler wählt nur eine Präferenz
  (`MLModelConfiguration.computeUnits`: `cpuOnly = 0`, `cpuAndGPU = 1`, `all = 2`, `cpuAndNeuralEngine = 3`), Core ML entscheidet pro
  Operation, ob sie auf der ANE läuft, und fällt bei nicht unterstützten Ops auf CPU/GPU zurück. Es gibt keine Möglichkeit, das Ergebnis
  der Platzierung aus der API zu lesen (Xcode-Instruments oder `powermetrics` zeigen es). Private Frameworks mit direktem ANE-Zugriff existieren
  (nach meiner Kenntnis), sind undokumentiert, nicht App-Store-tauglich und werden hier bewusst nicht benutzt.
- **Rust-Anbindung.** In `Cargo.toml` hängt `npu-ane` an keiner Abhängigkeit (`npu-ane = []`); die `objc2`-Crates gehören zu `metal`. Darum
  spricht `ane/objc.rs` die Objective-C-Runtime direkt an (`objc_getClass`, `sel_registerName`, `objc_msgSend` über Funktionszeiger mit der
  exakten Signatur je Nachricht), öffnet Foundation und CoreML per `dlopen` und kennt nur die rund zwanzig benötigten Nachrichten
  (`NSString`, `NSURL`, `NSArray`, `NSNumber`, `MLModel`, `MLModelConfiguration`, `MLMultiArray`, `MLFeatureValue`,
  `MLDictionaryFeatureProvider`). Ein Umstieg auf `objc2` + `objc2-core-ml` ist mechanisch, sobald die Abhängigkeiten in `Cargo.toml`
  stehen (siehe Bericht).
- **Modell erzeugen.** Ein ML Program (`mlProgram`, Spezifikationsversion ≥ 6; hier 7 = iOS 16/macOS 13) besteht aus
  `Model` ⊃ `MILSpec.Program` ⊃ `Function "main"` ⊃ `Block` ⊃ `Operation`en (`const`, `linear`, …). `ane/mlpackage.rs` schreibt genau das als
  Protobuf (eigener Encoder `xdna/pb.rs`, Feldnummern aus den `.proto`-Dateien übernommen und gegengeprüft), dazu die Gewichte im MIL-Blob-Format
  (Version 2: 64-Byte-Kopf, je Blob 64 Byte Metadaten mit Sentinel `0xDEADBEEF`, 64-Byte-Ausrichtung, Datentyp `Float16 = 1`), dazu
  `Manifest.json` mit `itemInfoEntries` (`model.mlmodel` und `weights`, Autor `com.apple.CoreML`) und `rootModelIdentifier`.
  `ane/mil.rs` druckt dasselbe Programm als MIL-Text (`program(1.0) { func main<ios16>(tensor<fp16, [m, k]> x) { … linear(x = x, weight = weight) … } -> (y); }`)
  zum Lesen und Vergleichen mit `coremltools`; Core ML kann diesen Text nicht selbst kompilieren.
- **Ablauf in `ane/coreml.rs`:** `.mlpackage` in ein Temp-Verzeichnis schreiben → `+[MLModel compileModelAtURL:error:]` → Konfiguration mit
  `computeUnits = cpuAndNeuralEngine` → `+[MLModel modelWithContentsOfURL:configuration:error:]` → `MLMultiArray` (Float16) füllen →
  `-predictionFromFeatures:error:` → Ausgabe über `strides` lesen (die ANE kann gepolsterte Ausgaben liefern) → mit CPU-Referenz vergleichen.
  `detect()` erkennt Apple Silicon (aarch64), liest Chipname und Speicher per `sysctlbyname` und, falls vorhanden (macOS 14), die Kernzahl
  über `MLAllComputeDevices()`/`MLNeuralEngineComputeDevice`.

## 2. Datenformate

- Die ANE rechnet nach meiner Kenntnis in FP16 (neuere Chips zusätzlich INT8-Aktivierungen); Ein-/Ausgaben als FP16-`MLMultiArray`.
- Gewichte: FP16 als `const`-Blob (hier), komprimiert über `constexpr_*`-Ops der neueren Opsets (INT8/INT4 mit Skala/Nullpunkt, blockweise
  Quantisierung, Palettierung/LUT, Sparsity; Opsets iOS 17/18, **nach meiner Kenntnis**, nicht geprüft). Diese Ops werden beim Laden zu dichten
  Gewichten entpackt oder von der ANE direkt gelesen – das entscheidet Core ML.
- fLLMs Formate: Q4_0/Q4_1/Q8_0 ließen sich blockweise auf INT4/INT8 mit Skala abbilden, K-Quants, IQ4_NL, MXFP4, NVFP4, Q4Z, FP8M nicht ohne
  Umrechnung. Es gibt keinen Weg, rohe GGUF-Blöcke an die ANE zu geben.

## 3. Grenzen

- **Statische Shapes.** Flexible Shapes (aufzählbar oder Bereich) sind möglich, kosten aber nach meiner Kenntnis oft die ANE-Platzierung.
  Ein LLM braucht Prefill- und Decode-Programm (oder zustandsbehaftete Modelle, `MLState`, ab iOS 18/macOS 15, für den KV-Cache).
- **Kompilierzeit.** `compileModelAtURL` und die erste Ladung mit ANE-Spezialisierung dauern bei großen Modellen lange (Minuten, nach
  meiner Kenntnis; nicht gemessen). Ein ganzes 7B-Modell müsste beim Laden aus GGUF als ML Program mit allen Gewichten neu geschrieben und
  kompiliert werden.
- **Speicher.** Einheitlicher Speicher; Core ML hält eigene Kopien der Gewichte (Doppelbelegung gegenüber dem mmap des GGUF).
- **Lizenz.** Core ML ist Teil des Systems, keine Weitergabe nötig. Die `.proto`-Dateien von coremltools sind BSD-3-Clause (Hinweis in
  `THIRD_PARTY.md` über den Bericht).
- **Verfügbarkeit.** Nur Apple Silicon (`aarch64-apple-darwin`, macOS 13+ für `cpuAndNeuralEngine`). Linux/Windows: nein. Intel-Macs: keine ANE.

## 4. Erwarteter Nutzen

- **Prefill:** rechenbegrenzt, FP16-Matmul – die ANE ist dafür gebaut; ein hybrider Aufbau (Prefill auf der ANE, Decode auf der Metal-GPU) ist
  grundsätzlich denkbar. Ob das schneller ist als die Metal-GPU der M-Chips (nach meiner Kenntnis oft nicht beim Durchsatz, aber bei Energie)
  ist offen und nicht gemessen.
- **Decode:** bandbreitenbegrenzt; der Speicher wird mit CPU/GPU geteilt. Kein Tempo-Vorteil zu erwarten, nur Energie.

## 5. Empfehlung (ehrlich)

1. **Kein ANE-`Backend`.** Core ML nimmt nur Modelle, keine einzelnen Ops mit Geräte-Puffern; die Platzierung ist nicht steuerbar; fLLMs
   Gewichtsformate passen nicht. Der Metal-Pfad (Pakete 5–7) ist der Apple-Hauptweg.
2. Falls später ein Experiment gewünscht ist: ein **Prefill-Beschleuniger** über ein zur Ladezeit erzeugtes MLProgram (Linearlayer als `linear`
   mit blockquantisierten Gewichten, iOS-18-Opset), mit dem hier vorhandenen Emitter als Ausgangspunkt. Erst die Messung aus Abschnitt 6
   (Kompilier-/Ladezeit, `predict_ms` für M = 1 und M = 128+) zeigt, ob das lohnt.
3. Der Emitter ist auch ohne ANE nützlich: ein erzeugtes `.mlpackage` lässt sich mit `coremltools` oder Xcode öffnen und prüfen.

## 6. Prüfanleitung für die Testrunde

Voraussetzung: Apple-Silicon-Mac, macOS 13 oder neuer, Xcode-Kommandozeilenwerkzeuge.

- Bauen mit `--features npu-ane` auf macOS (Standard-Zielplattform); `fllm-rs devices` muss „Apple Neural Engine (Apple M…)“ zeigen.
- `backend::npu::ane::probe_linear(m, k, n, ComputeUnits::CpuAndNeuralEngine)` gibt `ProbeReport { compile_ms, load_ms, predict_ms, max_abs_err, mil }`.
  Erwartet: `max_abs_err` im Bereich von FP16-Rundung (≈1e-2 bei k = 4096, Eingaben in [-1, 1]). Danach dasselbe mit `CpuOnly` und `All`; der
  Zeitunterschied und `powermetrics --samplers ane_power` zeigen, ob die ANE beteiligt war.
- Erste Fehlerquellen, die ich nicht ausschließen kann: Feldnummern/Feldnamen im Protobuf, die `compileModelAtURL` ablehnt (die Fehlermeldung
  kommt über `NSError.localizedDescription` in den Fehlertext); der Opset-Name `CoreML6`; die Form der `name`-Attribute; ob `Manifest.json` ohne
  weitere Felder reicht. Gegenprobe: dasselbe Programm mit `coremltools` (`ct.convert(..., convert_to="mlprogram")`) erzeugen, mit `protoc
  --decode` zerlegen und mit dem Ausgang von `linear_model()` vergleichen.
