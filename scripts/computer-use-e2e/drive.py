"""End-to-end check of computer use on Linux/X11: drives the computer_* tools through POST /api/studio/agent/tool against a real
X server (Xvfb) with the Tk window of app.py, and prints every step. Run through run.sh (it starts Xvfb, the window and the server).
Only the Python standard library is used; xdotool is used once, to press the panic chord on the real keyboard of the X server.
"""
import base64
import json
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib

BASE = "http://127.0.0.1:8080"
ok = True


def call(path, body=None, method="POST", base=BASE):
    data = None if method == "GET" else json.dumps(body or {}).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def tool(name, args=None):
    return call("/api/studio/agent/tool", {"name": name, "args": args or {}})[1]


def png_size(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    w, h = struct.unpack(">II", data[16:24])
    raw, pos = b"", 8
    while pos < len(data):
        n, t = struct.unpack(">I4s", data[pos:pos + 8])
        if t == b"IDAT":
            raw += data[pos + 8:pos + 8 + n]
        pos += 12 + n
    zlib.decompress(raw)  # the picture data is valid
    return w, h


def state():
    time.sleep(0.5)
    with open("/tmp/state.json") as f:
        return json.load(f)


def step(title):
    print("\n== " + title)


def show(label, value):
    print("  %s: %s" % (label, value))


def check(cond, text):
    global ok
    print("  [%s] %s" % ("PASS" if cond else "FAIL", text))
    if not cond:
        ok = False


for _ in range(80):
    try:
        code, status = call("/api/studio/agent/computer", method="GET")
        if code == 200:
            break
    except Exception:
        pass
    time.sleep(0.5)
else:
    sys.exit("the server did not start")

step("off by default")
show("status", {k: status[k] for k in ("compiled", "enabled", "available", "platform")})
check(status["enabled"] is False, "computer use is off at the start")
out = tool("computer_screenshot")
show("screenshot while off", out)
check(out.get("ok") is False and "switched off" in out.get("error", ""), "a screenshot is refused while it is off")
code, catalog = call("/api/studio/agent/tools", method="GET")
check(not any(t["function"]["name"].startswith("computer_") for t in catalog["tools"]), "no computer tool in the catalog while it is off")

step("switch on in the runtime settings")
code, rt = call("/api/studio/runtime", {"computer_use": True, "computer_rate": 120, "computer_max_steps": 40})
show("runtime.computer", rt.get("computer"))
check(code == 200 and rt["computer"]["enabled"] is True, "the switch turns on")
code, catalog = call("/api/studio/agent/tools", method="GET")
cu = [t for t in catalog["tools"] if t["function"]["name"].startswith("computer_")]
show("tools offered", [t["function"]["name"] for t in cu])
check(len(cu) == 10 and all(t.get("permission") == "computer" for t in cu), "ten tools in the permission category computer")

step("rate limit and step cap (first, while the last minute is still empty)")
call("/api/studio/runtime", {"computer_rate": 3, "computer_max_steps": 5000})
call("/api/studio/agent/computer/begin")
res = [tool("computer_screenshot") for _ in range(5)]
show("3 per minute, 5 tries", [r.get("ok") for r in res])
check([r.get("ok") for r in res] == [True, True, True, False, False], "exactly three actions are allowed in a minute")
check("rate limit" in res[3]["error"], "the refusal says why: %s" % res[3]["error"][:100])
call("/api/studio/runtime", {"computer_rate": 600, "computer_max_steps": 4})
call("/api/studio/agent/computer/begin")
res = [tool("computer_screenshot") for _ in range(6)]
show("4 per task, 6 tries", [r.get("ok") for r in res])
check([r.get("ok") for r in res] == [True, True, True, True, False, False], "exactly four actions are allowed in one task")
check("step cap" in res[4]["error"], "the refusal says why: %s" % res[4]["error"][:100])
call("/api/studio/agent/computer/begin")
check(tool("computer_screenshot").get("ok") is True, "a new task (begin) starts the count again")
call("/api/studio/runtime", {"computer_rate": 600, "computer_max_steps": 5000})
call("/api/studio/agent/computer/begin")  # also forgets the last screenshot

step("input before a screenshot is refused")
out = tool("computer_click", {"x": 10, "y": 10})
show("click without a screenshot", out)
check(out.get("ok") is False and "screenshot first" in out.get("error", ""), "refused: take a screenshot first")

step("displays and screenshot")
out = tool("computer_displays")
show("displays", out.get("output"))
check(out.get("ok") is True and "1280x800" in out["output"], "the virtual screen is listed (1280x800)")
out = tool("computer_screenshot")
img = out.get("image")
show("screenshot", out.get("output"))
check(out.get("ok") is True and img and img["mime"] == "image/png", "a PNG comes back")
data = base64.b64decode(img["data"])
with open("/tmp/shot1.png", "wb") as f:
    f.write(data)
w, h = png_size(data)
show("png", "%dx%d, %d bytes (saved as /tmp/shot1.png)" % (w, h, len(data)))
check((w, h) == (1280, 800), "the picture is 1280x800")
check(len(data) > 2000, "it is not a blank picture (%d bytes)" % len(data))

with open("/tmp/geom.json") as f:
    geom = json.load(f)
show("screen positions of the test window's widgets", geom)
bx, by, bw, bh = geom["button"]

step("click a button of the Tk window")
before = state()["clicks"]
out = tool("computer_click", {"x": bx + bw // 2, "y": by + bh // 2})
show("click", out.get("output") or out.get("error"))
after = state()
show("application state", after)
check(out.get("ok") is True and after["clicks"] == before + 1, "the button's action ran (clicks %d -> %d)" % (before, after["clicks"]))

step("type into the entry field")
ex, ey, ew, eh = geom["entry"]
tool("computer_click", {"x": ex + ew // 2, "y": ey + eh // 2})
out = tool("computer_type", {"text": "Hello, World 42!"})
show("type", out.get("output") or out.get("error"))
after = state()
check(out.get("ok") is True and after["typed"] == "Hello, World 42!", "the text appeared in the entry field: %r" % after["typed"])

step("keys")
check(tool("computer_key", {"keys": "ctrl+/"}).get("ok") is True, "ctrl+/ (select all in Tk) accepted")
out = tool("computer_key", {"keys": "frobnicate"})
check(out.get("ok") is False, "an unknown key name is explained: %s" % out.get("error"))
out = tool("computer_key", {"keys": "backspace"})
after = state()
check(out.get("ok") is True and after["typed"] == "", "backspace cleared the selection: %r" % after["typed"])
tool("computer_type", {"text": "abc"})
tool("computer_key", {"keys": "enter"})
after = state()
check("Return" in after["keys"] and after["typed"] == "abc", "enter reached the field (keys %s, text %r)" % (after["keys"], after["typed"]))

step("scroll, drag and double click on the canvas")
cx, cy, cw, ch = geom["canvas"]
out = tool("computer_scroll", {"x": cx + 50, "y": cy + 50, "dy": 3})
check(out.get("ok") is True and state()["scroll"] == 3, "scrolled down 3 notches (state %s)" % state()["scroll"])
tool("computer_scroll", {"x": cx + 50, "y": cy + 50, "dy": -1})
check(state()["scroll"] == 2, "scrolled up 1 notch")
out = tool("computer_drag", {"from": {"x": cx + 20, "y": cy + 20}, "to": {"x": cx + 200, "y": cy + 90}})
after = state()
show("drag", after["drag"])
check(out.get("ok") is True and after["drag"] and abs(after["drag"]["from"][0] - 20) <= 2 and abs(after["drag"]["to"][0] - 200) <= 2 and abs(after["drag"]["to"][1] - 90) <= 2, "dragged from (20,20) to (200,90) inside the canvas")
out = tool("computer_click", {"x": cx + 100, "y": cy + 60, "double": True})
check(out.get("ok") is True and state()["doubles"] == 1, "double click recognised (%d)" % state()["doubles"])

step("cursor, move, wait")
tool("computer_move", {"x": 321, "y": 123})
cur = tool("computer_cursor")
show("cursor", cur.get("output"))
check(cur.get("ok") is True and "(321, 123)" in cur["output"], "the pointer is where it was moved")
check(tool("computer_wait", {"ms": 200}).get("ok") is True, "wait works")

step("a zoomed region")
out = tool("computer_screenshot", {"region": {"x": bx, "y": by, "width": 300, "height": 120}})
w, h = png_size(base64.b64decode(out["image"]["data"]))
show("region picture", "%dx%d: %s" % (w, h, out["output"]))
check((w, h) == (300, 120), "the region screenshot is 300x120")
tool("computer_screenshot")  # the full-screen mapping again

step("dangerous chords and bad input are refused")
for keys in ("ctrl+alt+delete", "alt+f4", "ctrl+shift+escape", "ctrl+alt+shift+esc", "ctrl+alt+f2"):
    out = tool("computer_key", {"keys": keys})
    check(out.get("ok") is False and "refused" in out.get("error", ""), "%s refused: %s" % (keys, out.get("error", "")[:70]))
out = tool("computer_click", {"x": 5000, "y": 10})
check(out.get("ok") is False and "outside the last screenshot" in out["error"], "a point outside the screenshot is explained: %s" % out.get("error", "")[:90])
check(tool("computer_type", {"text": "x" * 5000}).get("ok") is False, "too long a text is refused")
check(tool("computer_click", {"x": 1, "y": 1, "button": "fourth"}).get("ok") is False, "an unknown button is refused")

step("the stop route cancels and switches it off")
code, st = call("/api/studio/agent/computer/stop")
check(st["enabled"] is False, "STOP switched computer use off")
out = tool("computer_click", {"x": 10, "y": 10})
check(out.get("ok") is False and "switched off" in out["error"], "actions are refused afterwards: %s" % out.get("error", "")[:60])
code, rt = call("/api/studio/runtime", method="GET")
check(rt["computer"]["enabled"] is False, "the setting says off")

step("another machine cannot reach any of it (a request from the container's own network address)")
import socket
other = "http://%s:8080" % socket.gethostbyname(socket.gethostname())
call("/api/studio/runtime", {"computer_use": True})
show("address used", other)
for method, path, body in (
    ("POST", "/api/studio/agent/tool", {"name": "computer_screenshot", "args": {}}),
    ("GET", "/api/studio/agent/computer", None),
    ("POST", "/api/studio/agent/computer/stop", {}),
    ("POST", "/api/studio/agent/computer/begin", {}),
    ("POST", "/api/studio/runtime", {"computer_use": False}),
    ("GET", "/api/studio/runtime", None),
):
    code, _ = call(path, body, method, base=other)
    check(code == 403, "%s %s from another address: HTTP %d" % (method, path, code))
code, st = call("/api/studio/agent/computer", method="GET")
check(st["enabled"] is True, "the foreign stop and switch-off attempts changed nothing (still on)")
call("/api/studio/agent/computer/stop")

step("the panic chord on the keyboard (Ctrl+Alt+Shift+Esc) switches it off")
call("/api/studio/runtime", {"computer_use": True})
code, st = call("/api/studio/agent/computer", method="GET")
check(st["enabled"] is True, "switched on again")
time.sleep(0.5)
subprocess.run(["xdotool", "keydown", "ctrl", "keydown", "alt", "keydown", "shift", "keydown", "Escape"], check=True)
time.sleep(0.8)
subprocess.run(["xdotool", "keyup", "Escape", "keyup", "shift", "keyup", "alt", "keyup", "ctrl"], check=True)
code, st = call("/api/studio/agent/computer", method="GET")
check(st["enabled"] is False, "the chord pressed on the keyboard switched it off (enabled=%s)" % st["enabled"])

print("\nRESULT: %s" % ("ALL PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
