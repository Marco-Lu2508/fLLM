# CPU-Emulation der Kernel von Paket 6c

Führt den **echten Kerneltext** von `moe_*`, `gate_*`, `delta_*` und `mamba_*` auf der CPU aus (ein `std::thread` je GPU-Thread, echte Barriere,
statischer Threadgroup-Speicher) und vergleicht mit unabhängigen Referenzen. Das ist **kein** Test unter Metal (Stufe L0, siehe `../INTERFACE_RECURRENT.md`
Abschnitt 9); es belegt Algorithmus, Indexrechnung, Blocklayouts und Barrierendisziplin, nicht die Annahme durch den MSL-Compiler oder das Verhalten der GPU.

```
# 1. Testvektoren für moe_linear (zufällige GGUF-Blöcke aller zehn Formate + Gewichte aus dem Rust-Dequantisierer)
cd harness && cp Cargo.toml.example Cargo.toml && cargo run --bin gen_vectors -- /tmp/vectors.bin          # Pfad der fllm-engine in Cargo.toml / src/lib.rs anpassen
# 2. Emulation (g++ >= 12 oder clang >= 15, C++20)
./run_emulation.sh /tmp/vectors.bin                                    # Exit 0 = alle Prüfungen bestanden
TSAN=1 ./run_emulation.sh /tmp/vectors.bin                             # mit ThreadSanitizer: fehlende Barrieren werden zu Datenrennen
# 3. Gegenprobe: absichtlich fehlerhafte Kernel müssen auffallen
TSAN=1 ./mutation_check.sh /tmp/vectors.bin
```

| Datei | Inhalt |
|---|---|
| `msl_emu/emu.h`, `msl_emu/metal_stdlib` | der kleine MSL-Ersatz (Adressraum-Makros, Barriere, Mathe, `as_type`, `uint2`) |
| `emu_main.cpp` | Tests und Referenzen (Double-Portierungen von `mamba_forward`, `mamba1_forward`, `cpu_delta.rs`, `moe_route`) |
| `run_emulation.sh` | schreibt die `.metal`-Dateien per `sed` in C++ um (Attribute weg, `threadgroup`-Felder statisch), baut und startet |
| `mutation_check.sh` | 16 absichtliche Kernelfehler; jeder muss von der Emulation gemeldet werden |
| `harness/` | Test-Crate (`gen_vectors`, Planungstests); `Cargo.toml.example` nach `Cargo.toml` kopieren. Pfade zu `fllm-engine` (Cargo.toml) und zu den Planungsdateien (`src/lib.rs`) sind relativ zum Repository gesetzt und bei Bedarf anzupassen; `cargo run --bin gen_vectors` und `cargo test --lib` |

Die Umschreibung ist absichtlich klein: Template-Instanziierungszeilen (`template [[host_name(…)]] …`) werden gelöscht (sie prüft `mslcheck.sh`), die Kernel
selbst werden unverändert aufgerufen (Grid und Threads setzt der Testtreiber wie ein Metal-Dispatch).
