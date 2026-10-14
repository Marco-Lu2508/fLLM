# Metal-Kernel für MoE, Gated Attention, DeltaNet und Mamba (Paket 6c) — Schnittstelle

Prüfstufe: **L0**. Alle Kernel sind mit `mslcheck.sh` (clang + Shim, grob) syntaxgeprüft und laufen **auf der CPU** in einer Emulation
(`emu_recurrent/`, siehe Abschnitt 9). **Kein Kernel hat je unter Metal übersetzt oder auf einer GPU gerechnet.** Die Planungsdateien
(`src/backend/metal/ops/moe_plan.rs`, `recurrent_plan.rs`) sind Rust-Unit-getestet, aber nur gegen einen lokalen Ersatz für die Typen von `ops.rs`
(Paket 6b); sie kompilieren für `aarch64-apple-darwin`.

Die Kernel sind Portierungen der Vulkan-Shader von Paket 3 (`shaders/vulkan/{moe_,delta_,mamba_,misc_split_q_gate,act_sigmoid_mul}*.comp`, gleiche
Semantik, gleiche Zerlegung in Kernel); Vorbilder der Rechnung sind `cuda/deltanet.cu`, `cpu_delta.rs` und `mamba_forward` / `mamba1_forward` /
`moe_route` im CPU-Decoder. Kein Code von MLX, llama.cpp oder anderen Fremdquellen wurde übernommen (MLX diente nur als Strukturvorbild:
Template-Kernel mit `host_name`-Instanziierung).

## 1. Dateien

| Datei | Inhalt |
|---|---|
| `common_rec.metal` | geteilter Code (Präfix `r_` / `R_`): Aktivierungen, Threadgroup-Reduktionen, Byte-Leser, Formattraits `r_fmt_*` |
| `moe_route.metal` | `moe_topk` |
| `moe_linear.metal` | `moe_linear_<fmt>` (10 Formate) |
| `moe_combine.metal` | `moe_combine` |
| `moe_rows.metal` | `moe_gather_rows`, `moe_scatter_add_rows` |
| `gate_attention.metal` | `gate_split_q`, `gate_sigmoid_mul` |
| `delta_conv.metal`, `delta_rec.metal` | `delta_conv`, `delta_rec_hk{32,64,96,128}` |
| `mamba_conv.metal` | `mamba_conv` |
| `mamba_scan2.metal` | `mamba_scan2_ns{16,32,64,128}`, `mamba_scan2_gen` |
| `mamba_scan1.metal` | `mamba_scan1_ns{8,16,32}`, `mamba_scan1_gen` |
| `mamba_norm.metal` | `mamba_gate_norm`, `mamba_bc_norm` |
| `emu_recurrent/` | CPU-Emulation der Kernel (Abschnitt 9) |
| `../src/backend/metal/ops/moe_plan.rs`, `recurrent_plan.rs` | Dispatch-Planung (`OpPlan`) ohne Metal-Typen |

Layout-Regeln von Paket 5 (`API.md` Abschnitt 4): `common_rec.metal` beginnt mit `common`, ist also geteilter Code und wird jeder anderen Datei
vorangestellt; jede andere Datei ist eine eigene Bibliothek und benutzt nur `common.metal`, `common_rec.metal` und ihren eigenen Text. Template-Kernel
werden mit wörtlichen `template [[host_name("…")]] [[kernel]] decltype(f<T>) f<T>;`-Zeilen instanziiert (der Namensscanner findet sie). Function
Constants werden **nicht** benutzt (`OpPlan` hat kein Feld dafür): jede Variante ist ein eigener Kernelname.

## 2. Konventionen aller Kernel

- **Konstanten**: ein `struct …_params` aus lauter 4-Byte-Wörtern (`uint`, `float`; kein Padding, little endian), mit `setBytes` an **Buffer-Index 0**
  (`moe_plan::CONSTANTS_BUFFER_INDEX`) gebunden. `OpPlan::constants` enthält genau diese Bytes.
- **Puffer** ab Index 1 in der Reihenfolge von `OpPlan::slots` (`BufferSlot::index` ist der Metal-Index). Alle Aktivierungen sind `float` (f32), eine
  Zeile pro Token. Der Zugriff jedes Slots (`Read`/`Write`/`ReadWrite`) ist **bindend**: der Hazard-Tracker von Paket 5 setzt danach seine Barrieren;
  ein falsch deklarierter Slot ist ein Datenrennen. Nach `FLLM_METAL_SERIAL=1` verschwindet diese Fehlerklasse.
- **Dummy-Slots**: ein Eingang, den der Kernel nicht liest (z. B. `bias_table` ohne Bias), muss trotzdem an einen gültigen Puffer (mindestens 16 Byte)
  gebunden werden.
- **Host-Slots** (`SlotSource::Host`): kleine Host-Arrays (Zeilenindizes `int32`, Gewichte `float`), die der Stream für diesen Dispatch in einen
  Puffer legt (Unified Memory: `memcpy`; bei ≤ 4096 Byte geht auch `setBytes`).
- **Threadgroups**: `OpPlan::threadgroups` (Anzahl der Gruppen) und `threads_per_group` (aus dem Kernel-Header). Elementweise Kernel laufen als
  Grid-Stride-Schleifen über `threads_per_grid` (höchstens 65535 Gruppen × 256 Threads), jedes Grid ist also richtig.
- Alle Kernel haben im Quelltext einen maschinenlesbaren Kopf `// FLLM-KERNEL: <Name> | constants: … | buffers: n name(r|w|rw), … | threads: x,y,z`;
  die Rust-Tests vergleichen jeden Plan mit diesem Kopf **und** mit dem `struct …_params` derselben Datei (Namen, Reihenfolge, Typ, Index, Zugriff).
- Threadgroup-Reduktionen laufen im Threadgroup-Speicher (keine Simdgroup-Operationen vorausgesetzt, also auf jeder GPU richtig, aber nicht auf
  Apple-Silizium optimiert).

## 3. Operation → Plan → Kernel

| `Stream`-Operation | Plan (`moe_plan::` / `recurrent_plan::`) | Kernel |
|---|---|---|
| `moe_topk`, `moe_route` | `plan_moe_topk`, `plan_moe_route` | `moe_topk` |
| `moe_linear` | `plan_moe_linear` (`MoeLinear`, `ExpertAccess`) | `moe_linear_<q4_0\|q4_1\|q5_0\|q5_1\|q8_0\|iq4_nl\|mxfp4\|q4_k\|q5_k\|q6_k>` |
| `moe_combine` | `plan_moe_combine` | `moe_combine` |
| `moe_expert_gemm` | `plan_moe_expert_gemm` (= `moe_linear` + `moe_combine`) | wie oben |
| `gather_rows`, `scatter_add_rows` | `plan_gather_rows`, `plan_scatter_add_rows` | `moe_gather_rows`, `moe_scatter_add_rows` |
| `split_q_gate`, `sigmoid_gate_mul`, `supports_attention_gate` | `plan_split_q_gate`, `plan_sigmoid_gate_mul`, `attention_gate_supported` | `gate_split_q`, `gate_sigmoid_mul` |
| `delta_layer`, `supports_delta` | `plan_delta_layer`, `delta_supported` | `delta_conv`, `delta_rec_hk<N>` |
| *(keine Stream-Methode)* Mamba-2-Mixer | `plan_mamba2_mixer` (`plan_mamba_conv`, `plan_mamba2_scan`, `plan_mamba_gate_norm`) | `mamba_conv`, `mamba_scan2_*`, `mamba_gate_norm` |
| *(keine Stream-Methode)* Mamba-1-Mixer | `plan_mamba_conv`, `plan_mamba1_bc_norm`, `plan_mamba1_scan` | `mamba_conv`, `mamba_bc_norm`, `mamba_scan1_*` |

Nicht abgedeckt: die Projektionen (`x_proj`, `dt_proj`, Ein-/Ausgabeprojektionen, Router) sind gewöhnliche `quantized_linear`-Aufrufe der Pakete 6a/7.
`upload_pointer_table` / `upload_bias_table` bauen die Tabellen von Abschnitt 5; `moe_linear` selbst ist ein Matrix-Vektor-Kernel, kein gruppierter GEMM:
für lange Prompts sollte der Stream die Token pro Experte gruppieren (`gather_rows` + `quantized_linear` von 6a + `scatter_add_rows`), wie es der CPU-Decoder tut.

## 4. Kernel im Einzelnen

Die Köpfe der `.metal`-Dateien sind maßgeblich; hier die Kurzfassung (Konstanten in struct-Reihenfolge, Puffer ab Index 1, Threads je Gruppe, Grid in Gruppen).

| Kernel | Konstanten | Puffer | Threads | Grid |
|---|---|---|---|---|
| `moe_topk` | rows, experts, k, normalize_scores, scale(f) | logits(r), ids(w, int32), weights(w) | 128 | [rows,1,1] |
| `moe_linear_<fmt>` | rows, cols, slots, x_div, n_experts, flags, nb | table(r), bias_table(r), ids(r), x(r), out(w), arena(r), bias_arena(r) | 64 | [rows, slots, 1] |
| `moe_combine` | tokens, k, hidden | out(w), y(r), weights(r) | 256 | [min(⌈tokens·hidden/256⌉, 65535),1,1] |
| `moe_gather_rows` | n, row_len, has_weights | dst(w), src(r), idx(r), weights(r) | 256 | wie oben mit n·row_len |
| `moe_scatter_add_rows` | n, row_len, has_weights | dst(rw), src(r), idx(r), weights(r) | 256 | wie oben |
| `gate_split_q` | total, heads, dk, per_head | projected(r), q(w), gate(w) | 256 | ⌈total/256⌉ |
| `gate_sigmoid_mul` | count | x(rw), gate(r) | 256 | ⌈count/256⌉ |
| `delta_conv` | conv_dim, kh, hk, ksz, row0, n, reset, hist_off, qscale(f) | qkv(r), conv(r), state(rw), mixed(w) | 128 | [2·kh + ⌈vh·hv/128⌉, ⌈n/16⌉, 1] je Run |
| `delta_rec_hk<N>` | kh, vh, hv, conv_dim, tiled, row0, n, reset, state_off, beta_scale(f), eps(f) | mixed(r), z(r), beta(r), alpha(r), dt_bias(r), decay(r), norm(r), state(rw), out(w) | 256 | [vh,1,1] je Run |
| `mamba_conv` | channels, ksz, in_stride, in_off, row0, n, reset, hist_off, has_bias | x(r), convw(r), convb(r), state(rw), mixed(w) | 64 | [⌈channels/64⌉, ⌈n/16⌉, 1] je Run |
| `mamba_scan2_*` | inner, ns, heads, groups, head_dim, conv_dim, proj_stride, proj_dt_off, row0, n, reset, state_off, step_min(f) | mixed(r), projected(r), dt_bias(r), decay(r), skip(r), state(rw), out(w) | 64 | [heads, ⌈head_dim/64⌉, 1] je Run |
| `mamba_scan1_*` | inner, ns, rank, split_stride, proj_stride, proj_gate_off, row0, n, reset, state_off | mixed(r), dt(r), split(r), projected(r), decay(r), skip(r), state(rw), out(w) | 64 | [⌈inner/64⌉,1,1] je Run |
| `mamba_gate_norm` | inner, norm_groups, rows, proj_stride, proj_gate_off, eps(f) | y(rw), projected(r), norm(r) | 256 | [norm_groups, rows, 1] |
| `mamba_bc_norm` | rows, rank, state, split_stride, eps(f) | split(rw) | 128 | [3, rows, 1] |

Grenzen: `moe_topk` höchstens 2048 Experten und k ≤ 32; `delta_rec` Schlüsselbreite 32/64/96/128 und Wertbreite ≤ 128; Faltungskernel 2…8;
Mamba-Zustandsgröße ≤ 256 (Register-Varianten für 16/32/64/128 bzw. 8/16/32, sonst `_gen`). Die Planer prüfen das und liefern `UnsupportedOperation`.

### Semantik in Kürze

- `moe_topk`: Softmax über alle Experten, die `k` größten (Gleichstand: kleinerer Index), bei `normalize_scores` durch `max(Summe, 1e-8)` geteilt, mal `scale`
  (`moe_route` = normalisiert, Skala 1). Ausgabe absteigend sortiert.
- `moe_linear`: `out[slot][r] = dot(W[ids[slot]][r], x[slot / x_div]) (+ bias)`; eine Expertennummer außerhalb `0..n_experts` ergibt 0 (wie das CPU-Backend).
- `delta_*`: Faltung über `[q|k|v]`, SiLU, L2-Normierung der q/k-Köpfe (`qscale = hk^-½` für q), Delta-Regel `S = S·exp(decay·softplus(α+dt_bias)); S += k ⊗ ((v − Sᵀk)·σ);
  o = Sᵀq`, gated RMS-Norm `o · rsqrt(mean(o²)+eps) · norm · silu(z)`; Schlüsselkopf von Wertkopf `j`: `tiled ? j % kh : j / (vh/kh)`.
- `mamba_*`: wie `mamba_forward` / `mamba1_forward` (Faltung mit Verlauf, selektiver Scan, gated Group-RMS-Norm; bei Mamba-1 optionale RMS-Norm von dt/B/C).

## 5. Expertengewichte und Tabellen (`moe_linear`)

**Annahme zum Gewichtslayout (Paket 6a kann abweichen):** das Gewicht eines Experten ist der rohe GGUF-Tensor, zeilenweise, `rows × (cols / QK)`
Blöcke genau in der GGUF-Blockstruktur (little endian, ohne Padding, ohne Umpacken):

| Format | QK | Bytes | Aufbau |
|---|---|---|---|
| Q4_0 | 32 | 18 | `half d; uchar qs[16]` — `w = d·(q−8)`, Nibbles 0..15 niedrig, 16..31 hoch |
| Q4_1 | 32 | 20 | `half d, m; uchar qs[16]` — `w = d·q + m` |
| Q5_0 | 32 | 22 | `half d; uchar qh[4], qs[16]` — `w = d·(q−16)`, 5. Bit aus `qh` |
| Q5_1 | 32 | 24 | `half d, m; uchar qh[4], qs[16]` |
| Q8_0 | 32 | 34 | `half d; char qs[32]` |
| IQ4_NL | 32 | 18 | `half d; uchar qs[16]` — `w = d·kvalues_iq4nl[q]` |
| MXFP4 | 32 | 17 | `uchar e; uchar qs[16]` — `w = 2^(e−128)·fp4[q]` (E2M1 verdoppelt) |
| Q4_K | 256 | 144 | `half d, dmin; uchar scales[12], qs[128]` |
| Q5_K | 256 | 176 | `half d, dmin; uchar scales[12], qh[32], qs[128]` |
| Q6_K | 256 | 210 | `uchar ql[128], qh[64]; char scales[16]; half d` |

Die Blöcke werden **bytegenau** gelesen (17- und 18-Byte-Blöcke sind nicht 4-Byte-ausgerichtet). Die Dequantisierung entspricht
`compute::dequantize` / `quant_extra` / `quant_iq` (CPU-Referenz); die Emulation prüft das gegen den echten Rust-Dequantisierer.
Nicht abgedeckt (Planer: `UnsupportedQuant`): Float-Formate, Q2_K, Q3_K, die übrigen IQ-Formate, NVFP4, Q4Z/Q4ZP, FP8, `Q4_K_Marlin`.

Zwei Wege zu den Experten (Pipeline und Kernel sind dieselben; `flags` Bit 1 wählt):

1. **Arena** (`ExpertAccess::Arena`): alle Experten einer Projektion liegen in **einem** Puffer (`arena`, Buffer 6); `table[e]` = **Byte-Offset** von Experte `e`
   darin (`ulong`). Die Bias-Vektoren (je `rows` floats, 4-Byte-ausgerichtet) liegen in `bias_arena` (Buffer 7), `bias_table[e]` = Byte-Offset.
   Hilfen: `moe_plan::expert_bytes`, `arena_offsets(sizes, align)` (Experten bei Vielfachen von 256, Bias bei 4), `pack_u64`. Der Puffer wird als Ganzes gebunden
   (Grenze `maxBufferLength`; 128 Experten 768×2048 Q4_K ≈ 113 MiB).
2. **Adresstabelle** (`ExpertAccess::Address`): `table[e]` / `bias_table[e]` = absolute 64-Bit-GPU-Adresse des Gewichts-/Biaspuffers von Experte `e`
   (`AddressTable::build` aus `buffers.rs`, also `upload_pointer_table`); der Stream muss die Experten per `useResource` resident machen
   (`builder.table(…)`), `arena`/`bias_arena` sind Dummies.

## 6. Zustandslayouts (rekurrente Schichten)

- **DeltaNet** (`DeltaShape`): im Zustandspuffer einer Sequenz beginnt der Zustand der Schicht bei `state_offset` Bytes: zuerst die rekurrenten Matrizen
  `S[Wertkopf][Schlüsselzeile][Wertspalte]` (f32, `rec_bytes()`), danach der Faltungsverlauf `history[Kanal][k]` (`conv_dim × (kernel−1)`, das k-te letzte
  Eingabewort, ältestes zuerst). Das ist das CUDA-Layout; der CPU-Decoder benutzt ein eigenes, die beiden werden nie gemischt. `layer_state_bytes()` ist auf
  256 aufgerundet; `state_offset` muss durch 4 teilbar sein (die Pläne rechnen in Float-Indizes).
- **Mamba** (`Mamba2Shape`, `Mamba1Shape`): zuerst der Verlauf `[Kanal][k]`, dann der Zustand (`[Kopf][Kanal][state]` bei Mamba-2, `[Kanal][state]` bei
  Mamba-1), `layer_state_bytes()` auf 256 aufgerundet. Die Offsets sind Parameter der Pläne.
- `reset = 1`: der alte Zustand wird nicht gelesen (die Emulation füllt ihn mit NaN, um das zu beweisen); ein Run, der nicht der erste einer Sequenz ist, setzt
  `reset = 0` und setzt den Zustand fort.
- Ein Run ist ein Dispatch-Paar je Sequenz (jede Sequenz hat ihren eigenen Zustandspuffer, `SlotSource::RunState(i)`); die Runs sind unabhängig und dürfen parallel laufen.
  Innerhalb eines Runs bearbeiten Threadgroups mit y > 0 (Token 16…) die Faltung ohne den Verlauf zu lesen; nur Gruppe y = 0 schreibt den neuen Verlauf zurück.

## 7. Grundlage für `Stream::supports_delta` (Metal)

`recurrent_plan::delta_supported(&DeltaShape)` ist `true`, wenn alles gilt:

1. `k_dim ∈ {32, 64, 96, 128}` (je ein Kernel `delta_rec_hk<N>`; die Hälfte einer Wertspalte liegt in Registern),
2. `1 ≤ v_dim ≤ 128` (zwei Threads je Wertspalte, 256 Threads je Threadgroup),
3. `k_heads > 0`, `v_heads > 0`, `v_heads` ein Vielfaches von `k_heads`,
4. `2 ≤ conv_kernel ≤ 8` (der Verlauf von höchstens 7 Werten je Kanal liegt in Registern),
5. `conv_dim` und die Kopfzahl passen in die 32-Bit-Indizes und das Grid.

Zusätzlich muss der Stream die Gewichte (`conv`, `dt_bias`, `decay`, `norm`) als f32-Geräte-Puffer führen und den Zustandspuffer jedes Runs auf mindestens
`state_offset + layer_state_bytes()` Bytes dimensionieren (hier nicht prüfbar). `supports_delta` soll genau `delta_supported` zurückgeben; `supports_attention_gate`
ist für die Gate-Kernel immer `true` (`attention_gate_supported`).

## 8. Numerik

Alles in f32. `exp`/`log` der Gates (`forget`, `da`, `softplus`, Softmax) benutzen `precise::`, damit Fast-Math die Genauigkeit der rekurrenten Faktoren nicht
verschlechtert; `exp`-Argumente der Sigmoid-/SiLU-Funktionen sind bei 80 begrenzt, damit nie `inf` entsteht. Die Summationsreihenfolge (Baumreduktion über
Threads) weicht von der CPU ab; die Emulation toleriert relativ 1e-4…3e-4 gegen eine Double-Referenz. Es wurde nichts gegen echte Modelle gemessen (L2 fehlt).

## 9. CPU-Emulation und Tests (`emu_recurrent/`)

`run_emulation.sh <vectors.bin>` übersetzt den **echten Kerneltext** (nach sed-Umschreibung der Attribute) mit einem C++-Compiler: jeder GPU-Thread ist ein
`std::thread`, Threadgroup-Speicher ein statisches Feld, `threadgroup_barrier` eine echte Barriere. Mit `TSAN=1` (ThreadSanitizer) wird eine fehlende Barriere
zum gemeldeten Datenrennen. Verglichen wird mit unabhängigen Double-Referenzen (Portierungen von `mamba_forward`, `mamba1_forward`, `cpu_delta.rs`,
`moe_route`); die Gewichte von `moe_linear` stammen aus dem echten Rust-Dequantisierer (`harness/src/bin/gen_vectors.rs`, zufällige Blöcke aller zehn Formate).
Getestet werden Läufe in einem Stück, in mehreren Runs (Verlauf/Zustand über Run-Grenzen), Token für Token, mit NaN-vorbelegtem Zustand bei `reset`, mit Zustandsoffset,
beide Expertenzugriffe, mit/ohne Bias, `x_div > 1` und ungültigen Expertennummern. `mutation_check.sh` baut absichtlich fehlerhafte Kernel und verlangt,
dass die Emulation jeden entdeckt.

Die Emulation prüft **nicht**: Annahme des MSL-Compilers (Adressräume, Attribute, `[[kernel]]`-Instanziierung), Registerdruck/Threadgroup-Speicher/Limits der GPU,
das Verhalten von Fast-Math auf der GPU, Leistung. Das übernimmt die Testrunde auf einem Mac.
`harness/` enthält das Test-Crate, das die Planungsdateien gegen die echten Typen von `fllm-engine` mit einem Ersatz für `ops.rs` übersetzt und testet
(das Modul `backend::metal` wird nur für Apple-Ziele kompiliert).

## 10. Ungeprüfte Annahmen und bekannte Schwächen (für die Testrunde)

1. Nichts lief unter Metal. Mögliche Übersetzungsfehler, die der Shim nicht sieht: `static constant constexpr const`-Datenmember in Structs, Ganzzahl→Zeiger-Casts
   `(device const uchar *)ulong`, `template [[host_name]] [[kernel]] decltype(…)`-Instanziierungen, `threadgroup`-Felder in Template-Kerneln, `_Pragma("clang loop unroll(full)")`.
2. Das Gewichtslayout (rohe GGUF-Blöcke) ist eine Annahme gegenüber Paket 6a; ändert 6a das Layout, ändern sich nur die `r_fmt_*`-Traits.
3. `moe_linear`: 64 Threads je (Zeile, Slot), byteweises Lesen der Blöcke; nicht auf Bandbreite abgestimmt (Vektorladungen, Simdgroup-Reduktion, mehrere Zeilen je Gruppe fehlen).
4. `delta_rec`: HK=128 hält 64 Floats je Thread in Registern plus 3 KiB Threadgroup-Speicher; ob der Compiler spillt, ist unbekannt. Ein langer Run eines Kopfes ist seriell.
5. Mamba hat keine `Stream`-Methode; die Pläne sind nur durch Unit-Tests und Emulation belegt. `mamba_conv`/`mamba_scan*` legen die Zwischenpuffer (`mixed`) in Scratch.
6. `BufferSlot`/`OpPlan` in `ops.rs` stammen von Paket 6b; die Pläne konstruieren `BufferSlot` an genau einer Stelle (`moe_plan::make_slot`), Feldnamen: `index`, `name`, `access`, `source`.
7. Barrieren zwischen Dispatches (`delta_conv` → `delta_rec`, `moe_linear` → `moe_combine`) hängen an den deklarierten Zugriffen der Slots (Abschnitt 2).
