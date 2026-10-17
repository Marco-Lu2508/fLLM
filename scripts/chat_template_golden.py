#!/usr/bin/env python3
"""Generate and check chat-template golden files with the reference renderer (transformers' jinja2 environment).

Sub-commands
  extract SRC_DIR OUT_DIR [NAME ...]
      SRC_DIR holds one directory per model with tokenizer_config.json and optionally chat_template.jinja /
      chat_template.json (the layout `huggingface_hub` produces). For each model writes into OUT_DIR
        <name>.jinja               default template
        <name>@<template>.jinja    named templates (for example tool_use)
        <name>.meta.json           bos_token / eos_token / pad_token / unk_token
      The template source order is the one fLLM's converter uses: chat_template.jinja, chat_template.json,
      tokenizer_config.json (string, or list of {"name","template"}).
  golden DIR CASES.json
      Renders every <name>.jinja (and named variants) in DIR for every case in CASES.json with the same Jinja
      environment as transformers.apply_chat_template (trim_blocks, lstrip_blocks, loopcontrols, tojson with kwargs,
      raise_exception, strftime_now) and writes DIR/<name>.golden.json: {"<template>/<case>": {"ok": text} | {"err": msg}}.
  probes PROBES.json OUT.json
      PROBES.json is {"vars": {...}, "probes": [{"name", "template"}]}; writes the rendered text (or error) per probe.

The Rust side (rust/engine/src/chat_template.rs, tests `corpus_*` and `probes_*`) renders the same inputs and compares.
"""
import json
import os
import sys


def template_sources(model_dir):
    """Return {template_name: text} for one model directory."""
    out = {}
    p = os.path.join(model_dir, "chat_template.jinja")
    if os.path.exists(p):
        out["default"] = open(p, encoding="utf-8").read()
    p = os.path.join(model_dir, "chat_template.json")
    if os.path.exists(p):
        v = json.load(open(p, encoding="utf-8")).get("chat_template")
        add_templates(out, v)
    p = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(p):
        v = json.load(open(p, encoding="utf-8")).get("chat_template")
        add_templates(out, v)
    return out


def add_templates(out, v):
    if isinstance(v, str):
        out.setdefault("default", v)
    elif isinstance(v, list):
        for item in v:
            out.setdefault(item["name"], item["template"])
    elif isinstance(v, dict):
        for k, t in v.items():
            out.setdefault(k, t)


def token_text(v):
    if isinstance(v, dict):
        return v.get("content")
    return v


def extract(src, out, names):
    os.makedirs(out, exist_ok=True)
    for d in sorted(os.listdir(src)):
        name = d.split("__")[-1].lower()
        if names and name not in names:
            continue
        md = os.path.join(src, d)
        srcs = template_sources(md)
        if not srcs:
            print("no template:", d)
            continue
        for tname, text in srcs.items():
            fn = f"{name}.jinja" if tname == "default" else f"{name}@{tname}.jinja"
            with open(os.path.join(out, fn), "w", encoding="utf-8", newline="") as f:
                f.write(text)
        tc = json.load(open(os.path.join(md, "tokenizer_config.json"), encoding="utf-8"))
        meta = {k: token_text(tc.get(k + "_token")) for k in ("bos", "eos", "pad", "unk")}
        with open(os.path.join(out, f"{name}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        print("extracted", name, sorted(srcs))


def reference_env():
    from transformers.utils.chat_template_utils import _compile_jinja_template

    return _compile_jinja_template


def render(compile_fn, text, ctx):
    try:
        return {"ok": compile_fn(text).render(**ctx)}
    except Exception as e:  # noqa: BLE001 - the message is the golden value
        return {"err": f"{type(e).__name__}: {e}"}


def golden(d, cases_path):
    compile_fn = reference_env()
    cases = json.load(open(cases_path, encoding="utf-8"))
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jinja"):
            continue
        stem = fn[: -len(".jinja")]
        name, _, tname = stem.partition("@")
        tname = tname or "default"
        meta_path = os.path.join(d, f"{name}.meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        text = open(os.path.join(d, fn), encoding="utf-8").read()
        gpath = os.path.join(d, f"{name}.golden.json")
        golden_all = json.load(open(gpath, encoding="utf-8")) if os.path.exists(gpath) and tname != "default" else {}
        for case in cases:
            ctx = {
                "messages": case["messages"],
                "tools": case.get("tools"),
                "documents": case.get("documents"),
                "add_generation_prompt": case.get("add_generation_prompt", True),
                "bos_token": meta.get("bos") or "",
                "eos_token": meta.get("eos") or "",
                "pad_token": meta.get("pad") or "",
                "unk_token": meta.get("unk") or "",
            }
            ctx.update(case.get("kwargs", {}))
            golden_all[f"{tname}/{case['name']}"] = render(compile_fn, text, ctx)
        with open(gpath, "w", encoding="utf-8") as f:
            json.dump(golden_all, f, indent=1, ensure_ascii=False, sort_keys=True)
        print("golden", name, tname, len(cases))


def probes(path, out):
    compile_fn = reference_env()
    doc = json.load(open(path, encoding="utf-8"))
    items = doc["probes"]
    res = {}
    for it in items:
        res[it["name"]] = render(compile_fn, it["template"], it.get("vars", doc["vars"]))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, sort_keys=True)
    print("probes", len(items))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract":
        extract(sys.argv[2], sys.argv[3], set(sys.argv[4:]))
    elif cmd == "golden":
        golden(sys.argv[2], sys.argv[3])
    elif cmd == "probes":
        probes(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
