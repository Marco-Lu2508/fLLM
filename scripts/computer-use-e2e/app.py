"""A tiny Tk window for the computer-use end-to-end test: a button, an entry field and a canvas.

Everything that happens to it is written to /tmp/state.json, and the screen positions of the widgets to /tmp/geom.json, so
that drive.py can click the real widgets through the computer_* tools and then check that the application reacted.
"""
import json
import tkinter as tk

root = tk.Tk()
root.title("cua-test")
root.geometry("640x420+100+80")
state = {"clicks": 0, "doubles": 0, "typed": "", "scroll": 0, "drag": None, "keys": []}


def save():
    with open("/tmp/state.json", "w") as f:
        json.dump(state, f)


def clicked():
    state["clicks"] += 1
    label.config(text="clicked %d" % state["clicks"])
    save()


button = tk.Button(root, text="Click me", width=20, height=3, command=clicked)
button.place(x=40, y=30)
label = tk.Label(root, text="not clicked", font=("TkDefaultFont", 18))
label.place(x=40, y=140)
entry = tk.Entry(root, width=40)
entry.place(x=40, y=200)


def on_key(_e=None):
    state["typed"] = entry.get()
    save()


entry.bind("<KeyRelease>", on_key)
entry.bind("<Return>", lambda e: (state["keys"].append("Return"), save()))

canvas = tk.Canvas(root, width=300, height=120, bg="#ffffcc")
canvas.place(x=40, y=250)
canvas.bind("<Button-4>", lambda e: (state.__setitem__("scroll", state["scroll"] - 1), save()))
canvas.bind("<Button-5>", lambda e: (state.__setitem__("scroll", state["scroll"] + 1), save()))
canvas.bind("<Double-Button-1>", lambda e: (state.__setitem__("doubles", state["doubles"] + 1), save()))
canvas.bind("<ButtonPress-1>", lambda e: state.__setitem__("drag", {"from": [e.x, e.y], "to": [e.x, e.y]}))
canvas.bind("<B1-Motion>", lambda e: state["drag"] and state["drag"].__setitem__("to", [e.x, e.y]))
canvas.bind("<ButtonRelease-1>", lambda e: save())


def write_geometry():
    root.update_idletasks()
    geo = {}
    for name, w in (("button", button), ("entry", entry), ("canvas", canvas)):
        geo[name] = [w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height()]
    with open("/tmp/geom.json", "w") as f:
        json.dump(geo, f)
    save()


root.after(800, write_geometry)
root.mainloop()
