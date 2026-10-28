#!/bin/bash
# Runs inside a Debian container that has xvfb, python3-tk and xdotool, with the fllm-rs binary at /bin/fllm-rs and this folder at
# /e2e (see README.md for the docker command): starts a virtual X screen, the Tk test window and a tiny-model server, then drives
# the computer_* tools against them.
set -u
export DISPLAY=:99
export HOME=/root
Xvfb :99 -screen 0 1280x800x24 -ac +extension XTEST >/tmp/xvfb.log 2>&1 &
sleep 2
python3 /e2e/app.py >/tmp/app.log 2>&1 &
sleep 2
/bin/fllm-rs make-tiny-model /tmp/tiny.gguf >/tmp/tiny.log 2>&1 || { cat /tmp/tiny.log; exit 1; }
/bin/fllm-rs serve --model /tmp/tiny.gguf --device cpu --host 0.0.0.0 --port 8080 >/tmp/serve.log 2>&1 &
sleep 3
python3 /e2e/drive.py
rc=$?
# a copy of the screenshot for whoever runs this
if [ -d /out ]; then cp /tmp/shot1.png /out/ 2>/dev/null; cp /tmp/serve.log /out/ 2>/dev/null; fi
exit $rc
