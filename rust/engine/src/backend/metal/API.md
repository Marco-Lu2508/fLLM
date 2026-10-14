# Metal-Kern: API für Paket 6 (Shader) und Paket 7 (Integration)

Stand: **L0** – kompiliert für `aarch64-apple-darwin` (`--features metal`), clippy-sauber, die reinen Teile haben
Unit-Tests auf dem Linux-Host. Nichts davon ist je auf einem Mac gelaufen. Ein Selbsttest für die erste Hardwarerunde
ist eingebaut (Abschnitt 9).

Das Modul `backend::metal` wird nur bei `cfg(all(feature = "metal", target_vendor = "apple"))` kompiliert.
Die vier Kern-Dateien gehören Paket 5; `ops.rs`, `weights.rs`, `shaders.rs` (Paket 6) und `backend.rs`, `stream.rs`
(Paket 7) bauen darauf auf.

| Datei | Inhalt |
| --- | --- |
| `device_info.rs` (+ `device_info/{platform,config,error}.rs`) | Geräteliste, `MetalDevice`, `MetalDeviceInfo`, `MetalConfig` (Umgebungsvariablen), `MetalError`, Chip-Erkennung |
| `buffers.rs` (+ `buffers/plan.rs`) | `MetalBuffer`, `BufferRef`, `MemoryManager`, `MappedRegion`/`MappedRegistry` (No-Copy-GGUF), `AddressTable`, `Arena` |
| `pipelines.rs` (+ `pipelines/{key,source}.rs`) | `PipelineCache`, `Pipeline`, `FunctionConstant`, MSL-Quellaufbereitung |
| `commands.rs` (+ `commands/{hazard,geometry,residency,smoke}.rs`) | `CommandQueue`, `Recorder`, `DispatchBuilder`, `MetalContext`, Hazard-Tracking, Selbsttest |

## 1. Einstieg

```rust
use crate::backend::metal::commands::{Access, MetalContext};

let ctx = MetalContext::new(index)?;            // Gerät, Speicherverwaltung, Pipeline-Cache, Queue, Residency
let mut rec = ctx.recorder();                   // ein Recorder pro Stream
```

`MetalContext` hat die öffentlichen Felder `device`, `config`, `memory` (`Arc<MemoryManager>`), `pipelines`
(`Arc<PipelineCache>`), `queue`, `mapped` (`Arc<MappedRegistry>`), `residency` (`Option<Arc<ResidencyManager>>`).

**Geräteindex.** `device_info::list(first_index)` meldet die Geräte mit globalen Nummern ab `first_index` und merkt sich
`first_index`. `device_info::open(n)` und `MetalContext::new(n)` lesen `n` als globale Nummer, wenn sie in
`[first_index, first_index + Anzahl)` liegt, sonst als Position unter den Metal-Geräten (0 = Standardgerät). Auf
Macs mit nur einer GPU ist beides dasselbe Gerät. `--device auto` reicht allerdings die globale Nummer des *ersten*
Eintrags mit „metal“ in der Beschreibung durch; sind wgpu und natives Metal gleichzeitig gebaut, steht der
wgpu-Adapter davor – das löst der Integrator in `devices.rs` (siehe Bericht).

`MetalDeviceInfo` (aus `ctx.device.info()`): Name, `chip: Option<AppleChip>` (M1..M5, A-Serie, Pro/Max/Ultra – nur
Namensmuster), `apple_family: Option<u8>` (höchste `MTLGPUFamilyAppleN`), `metal3`, `metal4`, `mac_family2`,
`has_unified_memory`, `recommended_max_working_set_bytes`, `max_buffer_length_bytes`, `max_threadgroup_memory_bytes`,
`max_threads_per_threadgroup`, `argument_buffers_tier2`, `simdgroup_matrix`, `simdgroup_reduction`, `bfloat`
(letzteres drei sind Faustregeln aus Apples Feature-Set-Tabellen, nicht gemessen), `msl_version`, `page_size`.

## 2. Umgebungsvariablen (`MetalConfig::from_env()`)

| Variable | Wirkung |
| --- | --- |
| `FLLM_METAL_SERIAL=1` | serieller Encoder (Treiber ordnet jeden Dispatch); **erste Maßnahme bei falschen Ergebnissen**, schließt Barrierefehler aus |
| `FLLM_METAL_CB_DISPATCHES=N` | Command Buffer nach N Dispatches committen (Standard 160, 0 = nur bei `flush`) |
| `FLLM_METAL_MEM_LIMIT_MB=N` | Speicherbudget (Standard `recommendedMaxWorkingSetSize`) |
| `FLLM_METAL_LIB=pfad` | vorkompilierte `.metallib` statt eingebetteter Quellen |
| `FLLM_METAL_SINGLE_LIB=1` | alle Quellen in *eine* Bibliothek (Standard: eine pro Datei, lazy) |
| `FLLM_METAL_FAST_MATH=0\|1` | Compiler-Fast-Math (Standard 1) |
| `FLLM_METAL_NO_NOCOPY=1` | GGUF nie ohne Kopie einbinden |
| `FLLM_METAL_RESIDENCY=0\|1` | Residency-Set (macOS 15) benutzen (Standard 1) |
| `FLLM_METAL_DEBUG=1` | Labels und Debug-Gruppen für GPU-Captures |
| `FLLM_METAL_MSL=3.1` | MSL-Version erzwingen (Standard: neueste, die das OS kann, höchstens 3.2; `4.0` nur so) |

## 3. Speicher (`buffers.rs`)

Alle Puffer sind `MTLStorageModeShared` (Unified Memory: Upload = `memcpy`, Download = Lesen, kein Staging).

```rust
let buf: Arc<MetalBuffer> = ctx.memory.alloc(bytes, BufferUsage::Activations)?;   // Inhalt unbestimmt
let zero = ctx.memory.alloc_zeroed(bytes, BufferUsage::KvCache)?;
let w    = ctx.memory.from_bytes(&host, BufferUsage::Weights)?;                    // Kopie
let r    = BufferRef::whole(buf.clone());                                           // Bindungseinheit
let part = r.slice(offset, len)?;
buf.write(off, &bytes)?; buf.read(off, &mut out)?; buf.fill(off, len, 0)?;
```

* `BufferUsage::Weights` ist *untracked* (kein Hazard-Tracking-Aufwand): Gewichte dürfen nach dem Hochladen von
  keinem Kernel mehr geschrieben werden. `Activations`/`KvCache`/`Staging` sind tracked.
* `write`/`read`/`fill` greifen direkt auf den Speicher zu: nur nach `synchronize()` und vor dem nächsten Dispatch, der
  den Bereich nutzt.
* `MetalBuffer` ist `Send + Sync` und passt als `DeviceBuffer.handle` (`Arc<dyn Any + Send + Sync>`):
  `Arc::downcast::<MetalBuffer>(handle.clone())`. `DeviceBuffer { offset, size }` entspricht `BufferRef { offset, len }`.
* **Ausrichtung.** `Arena` und `AddressTable` richten auf 256 Byte aus. Gebundene Offsets sollten durch 4 teilbar
  sein (Apple-GPUs), 256 für `constant`-Argumente auf Intel/AMD-Macs. Kernels binden Gewichte deshalb als
  `device const …*`, nicht als `constant`.
* Pro `MTLBuffer` reserviert Metal mindestens eine Seite (16 KiB auf Apple Silicon). Kleine Zwischenpuffer deshalb
  aus einer `Arena` (`Arena::new(&ctx.memory, kapazität, usage)`, `alloc(len) -> BufferRef`, `reset()` pro Schritt)
  oder dem vorhandenen `BufferPool` schneiden, nicht einzeln anlegen.
* Überschreiten von `maxBufferLength` oder des Budgets ergibt `MetalError::OutOfMemory { requested, limit }`.

### GGUF ohne Kopie

```rust
unsafe { ctx.register_host_mapping(owner /* Arc<Mmap> o.ä. */, base_ptr, len, größtes_tensor_in_bytes)?; }
// beim Hochladen eines Gewichts:
let weight: BufferRef = ctx.memory.buffer_for_host_slice(&ctx.mapped, raw_bytes)?;   // Fenster oder Kopie
```

`register_host_mapping` zerlegt den Bereich in seitenausgerichtete, überlappende Fenster (≤ `maxBufferLength`), so dass
jedes Objekt bis `max_item_len` ganz in einem Fenster liegt (`buffers/plan.rs`, mit Tests). `buffer_for_host_slice`
findet die Scheibe über ihre Adresse; liegt sie nicht in einem registrierten Bereich (oder ist No-Copy
abgeschaltet), wird sie in einen neuen `Weights`-Puffer kopiert. Voraussetzung für Paket 7: Der Decoder muss die
GGUF-Datei per `mmap` halten und denselben Speicher an `upload_weight` weiterreichen (heute liest der Loader
womöglich in `Vec`s – dann greift die Kopie, ohne Fehler). Der `owner` hält das Mapping am Leben. Das Mapping
wird auch in das Residency-Set aufgenommen, wenn das OS eines hat.

### GPU-Adresstabelle („Argument Buffer“)

`AddressTable::build(&ctx.memory, &[BufferRef])` legt einen Puffer mit den 64-Bit-GPU-Adressen an (für
`upload_pointer_table` der MoE-Experten). Im Kernel: `device const ulong *table`, `(device const T *)table[i]`.
Beim Dispatch `builder.table(index, &table, Access::Read)`: bindet die Tabelle, macht die Ziele mit `useResource`
resident und zählt den Zugriff für die Barrierelogik als „alles“. Braucht Argument-Buffers-Tier 2
(`argument_buffers_tier2`, alle Apple-GPUs).

## 4. Pipelines (`pipelines.rs`)

```rust
let p = ctx.pipelines.get("rms_norm_f32", &[])?;                                   // Arc<Pipeline>
let q = ctx.pipelines.get("gemv_q4_0", &[FunctionConstant::u32(0, 4), FunctionConstant::bool(1, true)])?;
ctx.pipelines.prewarm_libraries()?;               // alle Bibliotheken parallel kompilieren (sonst lazy beim ersten Zugriff)
ctx.pipelines.prewarm(&[("rms_norm_f32", &[]), …])?;
p.max_threads(); p.simd_width(); p.static_threadgroup_memory();
```

* `get` hasht den Schlüssel bei jedem Aufruf: Pipelines **einmal beim Start** auflösen und den `Arc` in der Op halten.
* Function Constants: `[[function_constant(i)]]` in MSL; `FunctionConstant::{bool,i32,u32,f32}(index, wert)`.
  Jede Wertemenge ist eine eigene Pipeline. Falsche Typen/Indizes melden `MetalError::Pipeline` mit dem Metal-Text.

### Layout der Shader-Quellen (für Paket 6 verbindlich)

`build.rs` bettet `rust/engine/metal/*.metal` ein (`common.metal` zuerst, Rest alphabetisch).

* Dateien, deren Name mit `common` oder `_` beginnt, **und Dateien, in denen kein Kernel gefunden wird** (reine
  Hilfsdateien), sind *geteilter Code*: Sie werden jeder anderen Datei vorangestellt (in Quellreihenfolge).
* Jede übrige Datei (mit mindestens einem Kernel) ist eine eigene Bibliothek, lazy kompiliert. Sie darf also nur
  Hilfscode aus geteilten Dateien und ihrem eigenen Text benutzen. Braucht ein Kernel Code aus einer anderen
  Kerneldatei, in eine `common_*.metal` auslagern (oder `FLLM_METAL_SINGLE_LIB=1` setzen).
* Kernel-Namen findet ein kleiner Scanner: `kernel void NAME(`, `[[kernel]] void NAME(` und
  `[[host_name("NAME")]]` bei Template-Instanziierungen. Makro-erzeugte Namen (`host_name("a_" #T)`) sieht er nicht;
  der Cache fragt dann alle Bibliotheken ab (langsamer erster Zugriff, funktioniert).
* Kein `#include "…"` auf Projektdateien (gibt es zur Laufzeit nicht); nur `<metal_stdlib>` & Co.
* Jede Einheit beginnt mit `#define`-Zeilen (immer 0/1 bzw. Zahlen): `FLLM_METAL`, `FLLM_APPLE_GPU_FAMILY` (0 = keine
  Apple-GPU), `FLLM_HAS_SIMDGROUP_MATRIX`, `FLLM_HAS_SIMDGROUP_REDUCTION`, `FLLM_HAS_BFLOAT`, `FLLM_UNIFIED_MEMORY`,
  `FLLM_SIMD_WIDTH_HINT`, `FLLM_MAX_THREADGROUP_MEMORY`. `#line`-Direktiven halten Compilerfehler auf den echten
  Dateinamen und Zeilen.
* Sprachversion: neueste, die das OS kann (macOS 13 → 3.0, 14 → 3.1, 15+ → 3.2), `bfloat` braucht 3.1.
* Vorkompiliert (optional, auf dem Mac): `xcrun -sdk macosx metal -c -I… datei.metal -o datei.air` für jede Datei
  mit vorangestelltem Gemeinsamen (`cat common.metal datei.metal > …`), dann `xcrun -sdk macosx metallib *.air -o
  fllm.metallib` und `FLLM_METAL_LIB=fllm.metallib`. Die `FLLM_*`-Defines fehlen dann; Paket 6 setzt sie ggf. per `-D`.

## 5. Befehle (`commands.rs`)

```rust
let mut rec = ctx.recorder();
rec.begin(&pipeline)?
    .read(0, &x)               // BufferRef, Access::Read
    .write(1, &y)              // Access::Write
    .read_write(2, &acc)       // Access::ReadWrite
    .bytes(3, &params)         // #[repr(C)]-Struct ohne Padding, <= 4096 Byte, setBytes
    .threadgroup_memory(0, 4096)
    .threadgroups([groups, 1, 1], [threads, 1, 1])?;      // oder .threads([gesamt,1,1], [pro_gruppe,1,1])?
rec.synchronize()?;            // Commit + auf die GPU warten; danach dürfen CPU-Zugriffe auf die Puffer erfolgen
```

* **Barrieren.** Der Encoder ist `MTLDispatchTypeConcurrent`. Jede Bindung deklariert, wie der Kernel den Bereich
  benutzt; der `HazardTracker` setzt vor einem Dispatch ein `memoryBarrierWithScope(Buffers)`, wenn er etwas liest,
  was ein früherer Dispatch desselben Fensters schrieb (oder schreibt, was gelesen/geschrieben wurde). Überlappungen
  werden bytegenau geprüft (Teilbereiche desselben Puffers stören sich nicht). **Wer falsch deklariert, bekommt
  Rennen**: ein als `read` gebundener Puffer, den der Kernel schreibt, ist ein Fehler. Notfalls `rec.barrier()?`.
* `FLLM_METAL_SERIAL=1` schaltet das ab (kein Tracking, Treiber serialisiert).
* **Command Buffer.** Nach `FLLM_METAL_CB_DISPATCHES` Dispatches wird committed und in einen neuen Buffer weiter
  kodiert (GPU startet früh). `flush()` committed ohne zu warten. `synchronize()` ist der einzige Ort, an dem auf die
  GPU gewartet wird (außer bei > 40 Buffern in der Luft). Fehler eines Buffers kommen als `MetalError::Gpu` zurück.
* `copy_buffer(&src, &dst, len)` und `fill_zero(&dst)` sind Blits (beenden den Compute-Encoder, ein neuer öffnet bei
  Bedarf); sie funktionieren nur auf tracked Puffern ohne weitere Fences, weil Metal dort über Encoder hinweg ordnet.
* `Recorder` ist `Send` (wandert mit dem Stream), aber nicht `Sync`; alle Methoden nehmen `&mut self`.
* Hilfen in `commands::geometry`: `elementwise_threads(n, simd, max)`, `ceil_div`, `validate_dispatch` (wird intern
  geprüft: Gruppengröße 0, über Pipeline- oder Gerätelimit → `MetalError::InvalidDispatch` statt Abbruch im Treiber).
* Bindungsplätze `0..=30` (Metal erlaubt 31). Konstanten über `bytes` belegen ebenfalls einen Platz.

## 6. Vorschlag für `backend.rs` / `stream.rs` (Paket 7)

| `Backend`/`Stream` | Metal |
| --- | --- |
| `Backend::create_stream` | `ctx.recorder()` in ein `MetalStream` legen |
| `allocate_buffer(size)` | `ctx.memory.alloc(size, Activations)`; `DeviceBuffer::new(id, size, device_id, arc as Arc<dyn Any + Send + Sync>)` |
| `Stream::upload(dst, src)` | `BufferRef.write` (nach `synchronize`, falls der Puffer von der GPU benutzt wird) |
| `Stream::download(dst, src)` | `synchronize()`, dann `BufferRef.read` |
| `Stream::synchronize` | `Recorder::synchronize` |
| `copy_buffer` | `Recorder::copy_buffer` |
| `zero_buffer` | `Recorder::fill_zero` |
| `upload_weight` | `buffer_for_host_slice` (No-Copy) bzw. nach Umpacken (Paket 6) `from_bytes(.., Weights)` |
| `upload_pointer_table` | `AddressTable` |
| `allocate_kv_cache` | `alloc_zeroed(.., KvCache)` (Zugriff nur über `DeviceKvCache`) |
| `BackendError` | `impl From<MetalError> for BackendError` ist vorhanden |

## 7. Garantien und Grenzen

* Metal-Objekte, die Apple als threadsicher dokumentiert (Gerät, Queue, Puffer, Pipelines, Bibliotheken), sind
  `Send + Sync`; Command Buffer und Encoder liegen nur im `Recorder`.
* Jeder Aufruf, der Metal-Objekte erzeugt, läuft in einem `autoreleasepool` (Rust-Threads haben keinen).
* Nicht enthalten: Texturen, Indirect Command Buffer, MTL4 (Metal 4: Command-Allocators, Tensor-API für M5). Die
  neuesten Fähigkeiten (`metal4`, Apple10) werden gemeldet, aber nicht genutzt.
* Mehrere Recorder gleichzeitig auf derselben Queue sind möglich; Abhängigkeiten *zwischen* Recordern regelt nur der
  Tracking-Modus der Puffer (Aktivierungen/KV tracked → Metal ordnet über Buffer hinweg). Explizite Events/Fences
  gibt es nicht.

## 8. Ungeprüfte Annahmen (für die Testrunde)

1. `newBufferWithBytesNoCopy` akzeptiert Fenster aus einem `PROT_READ`-`mmap` und ein Seitenende, das über das Dateiende
   hinaus reicht (Seitenpuffer des Mappings).
2. Untracked + CPU-beschriebene Gewichtspuffer: der erste Kernel sieht die Daten ohne zusätzliche Synchronisation, weil
   `commit` danach folgt.
3. `memoryBarrierWithScope(Buffers)` reicht als Ordnung zwischen Compute-Dispatches im selben Encoder (so nutzt es
   auch MLX).
4. Blits zwischen Compute-Encodern auf tracked Puffern brauchen kein Fence.
5. `useResource` auf den Zielen einer Adresstabelle genügt, damit die GPU sie lesen darf.
6. Die Residency-Set-API (macOS 15) wird über `respondsToSelector` erkannt; auf älteren Systemen entfällt sie.
7. SIMD-Gruppen-Fähigkeiten aus der Familie abgeleitet (Apple7+/Mac2); `Pipeline::simd_width` ist der echte Wert.

## 9. Selbsttest für die erste Hardwarerunde

```
cargo test --release -p fllm-engine --features metal metal_core_smoke -- --nocapture
```

führt `commands::smoke::run` aus (Funktion auch direkt aufrufbar: `smoke::run_default(0)`): Geräteinfo, Puffer und
Budget, Function Constants (zwei Spezialisierungen, Cache-Treffer), Dispatch-Kette mit Barrieren (erwartet ≥ 2
Barrieren im konkurrenten Modus), unabhängige Dispatches (erwartet 0 Barrieren), Threadgroup-Speicher-Reduktion,
Blit-Kopie/-Fill, Adresstabelle und No-Copy-Einbindung eines seitenausgerichteten Heap-Blocks. Die Kernels stehen im
Test selbst (`SMOKE_MSL`), unabhängig von den Shadern des Backends. Jede Zeile nennt Teil und Befund; schlägt eine fehl,
zuerst `FLLM_METAL_SERIAL=1` und `FLLM_METAL_NO_NOCOPY=1` ausprobieren, um die Ursache einzukreisen.
