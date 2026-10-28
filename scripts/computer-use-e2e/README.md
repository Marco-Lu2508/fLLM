# Computer use: end-to-end test on Linux (X11)

This drives the real `computer_*` tools of the coding agent against a real X server, without a model: it posts tool calls to
`POST /api/studio/agent/tool` of a running server (tiny model) exactly as the Code page does.

It proves, on Linux/X11 only: screenshots (a real PNG of the virtual screen), a click on a button of a Tk window, typing into an
entry field, keys, scroll, drag, double click, the refused chords, the rate limit, the step cap, the STOP route and the panic chord
Ctrl+Alt+Shift+Esc pressed on the keyboard of the X server. It does not prove Windows, macOS, Wayland or a model driving it.

```sh
# 1. build the binary (any machine with Rust):  cargo build --release --locked -p fllm-engine --bin fllm-rs
# 2. run the test in a throw-away container:
docker run --rm -v "$PWD/target/release/fllm-rs:/bin/fllm-rs:ro" -v "$PWD/scripts/computer-use-e2e:/e2e:ro" debian:trixie bash -c \
  'apt-get update -qq && apt-get install -y -qq xvfb python3 python3-tk xdotool >/dev/null && bash /e2e/run.sh'
```

The container needs no GPU and no network access after the packages are installed.
