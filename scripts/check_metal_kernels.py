#!/usr/bin/env python3
"""Check that every Metal kernel the Rust planners ask for is defined in rust/engine/metal/*.metal (and vice versa).

The same check exists in Rust (rust/engine/src/backend/metal/backend/kernels.rs: `inventory()` runs the planners, the unit test compares the
result with the embedded sources through the scanner of pipelines/source.rs). This script needs no Rust: it reads the planner sources as text
(string literals that are kernel names, plus the names `plan_quantized_linear` builds from the format tags) and scans the .metal files with the
same rules as the Rust scanner (`kernel void NAME(`, `[[kernel]] void NAME(`, `[[host_name("NAME")]]`; comments, strings and preprocessor lines
are ignored).

    python3 scripts/check_metal_kernels.py                 # exit 1 when a planned kernel is not defined
    python3 scripts/check_metal_kernels.py --list          # print the planned kernel names, one per line
    python3 scripts/check_metal_kernels.py --rust-list F   # additionally compare with a list of names from the Rust inventory
    python3 scripts/check_metal_kernels.py --self-test     # test the scanner itself

Verification level: L0 (a text check; says nothing about whether Apple's compiler accepts the kernels).
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
METAL_DIR = ROOT / "rust" / "engine" / "metal"
PLAN_DIR = ROOT / "rust" / "engine" / "src" / "backend" / "metal"

# Planner files whose string literals name kernels (relative to PLAN_DIR). Their inline unit-test modules are cut off.
PLANNER_FILES = [
    "ops.rs",
    "ops/attention.rs",
    "ops/embedding.rs",
    "ops/moe_plan.rs",
    "ops/recurrent_plan.rs",
    "weights/plan.rs",
]

# Kernel names as they appear in the planners. Everything else in a string literal (field names, messages) is not a kernel.
KERNEL_NAME = re.compile(
    r"fllm_[a-z0-9_]+"
    r"|moe_(?:topk|combine|gather_rows|scatter_add_rows|linear_[a-z0-9_]+)"
    r"|gate_(?:split_q|sigmoid_mul)"
    r"|delta_(?:conv|rec_hk\d+)"
    r"|mamba_[a-z0-9_]+"
)

# Defined on purpose without a planner (scaffold smoke kernel), and C++ template definitions (no [[host_name]]) that are not kernels one can ask for.
KNOWN_UNPLANNED = {"scaffold_fill"}

# Variants of the quantized-linear kernels: fllm_q_mv_<tag>_{r4,r2}, fllm_q_mm_<tag>_{h,f} (weights/plan.rs)
LINEAR_VARIANTS = [("fllm_q_mv_{tag}_r4"), ("fllm_q_mv_{tag}_r2"), ("fllm_q_mm_{tag}_h"), ("fllm_q_mm_{tag}_f")]


def strip_comments(text):
    """Source text without // and /* */ comments (string literals are respected)."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def strip_directives(text):
    """Drops preprocessor lines (with their backslash continuations)."""
    lines, keep, continued = text.split("\n"), [], False
    for line in lines:
        directive = continued or line.lstrip().startswith("#")
        continued = directive and line.rstrip().endswith("\\")
        keep.append("" if directive else line)
    return "\n".join(keep)


def scan_kernel_names(source):
    """Kernel names a Metal source defines, like `scan_kernel_names` of pipelines/source.rs."""
    text = strip_comments(source)
    # host_name("NAME") of template instantiations (directive lines can hold none)
    host = re.findall(r'host_name\s*\(\s*"([^"]+)"\s*\)', strip_directives(text))
    plain = re.sub(r'"(?:[^"\\]|\\.)*"', '""', strip_directives(text))
    kernels = re.findall(r"\bkernel\s*(?:\]\s*)*void\s+([A-Za-z_]\w*)", plain)
    seen, names = set(), []
    for name in kernels + host:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def defined_kernels():
    defined = {}
    for path in sorted(METAL_DIR.glob("*.metal")):
        for name in scan_kernel_names(path.read_text()):
            defined.setdefault(name, path.name)
    return defined


def planner_text(path):
    text = path.read_text()
    # inline unit-test modules (`#[cfg(test)]` + `mod tests {` / `mod testkit {`) are not planners
    return re.split(r"(?m)^#\[cfg\(test\)\]\s*\n\s*(?:pub(?:\([a-z]+\))? )?mod \w+ \{", text, maxsplit=1)[0]


def format_tags():
    """Kernel tags of the quantized formats (weights/format.rs)."""
    text = (PLAN_DIR / "weights" / "format.rs").read_text()
    text = re.split(r"(?m)^#\[cfg\(test\)\]", text, maxsplit=1)[0]
    tags = set(re.findall(r'\b(?:raw|q4z|dense)\(\s*QuantType::\w+\s*,\s*"([a-z0-9_]+)"', text))
    tags |= set(re.findall(r'\btag:\s*"([a-z0-9_]+)"', text))
    return tags


def planned_kernels():
    planned, problems = {}, []
    for rel in PLANNER_FILES:
        path = PLAN_DIR / rel
        if not path.exists():
            problems.append(f"planner file {rel} is missing")
            continue
        for literal in re.findall(r'"([A-Za-z0-9_]+)"', planner_text(path)):
            if KERNEL_NAME.fullmatch(literal):
                planned.setdefault(literal, rel)
    plan_rs = (PLAN_DIR / "weights" / "plan.rs").read_text()
    for pattern in ('"fllm_q_mv_{}_{}"', '"fllm_q_mm_{}_{}"'):
        if pattern not in plan_rs:
            problems.append(f"weights/plan.rs no longer builds names with {pattern}; update LINEAR_VARIANTS in this script")
    tags = format_tags()
    if len(tags) < 20:
        problems.append(f"only {len(tags)} format tags found in weights/format.rs")
    for tag in sorted(tags):
        for variant in LINEAR_VARIANTS:
            planned.setdefault(variant.format(tag=tag), "weights/format.rs (tag)")
    return planned, problems


def self_test():
    source = """
    // kernel void fake_a()
    /* kernel void fake_b() */
    #define K kernel void fake_c
    #define MULTI(x) \\
        kernel void fake_e
    const char *s = "kernel void fake_d";
    kernel void real_one(device float *x [[buffer(0)]]) {}
    [[kernel]] void real_two (device half *x [[buffer(0)]]) {}
    template <typename T> kernel void copy_t(device T *a [[buffer(0)]]) {}
    template [[host_name("copy_f16")]] kernel copy_type copy_t<half>;
    """
    got = scan_kernel_names(source)
    assert got == ["real_one", "real_two", "copy_t", "copy_f16"], got
    assert KERNEL_NAME.fullmatch("fllm_q_mv_q4_0_r4")
    assert KERNEL_NAME.fullmatch("moe_linear_q4_k") and KERNEL_NAME.fullmatch("delta_rec_hk128")
    assert not KERNEL_NAME.fullmatch("gate_stride") and not KERNEL_NAME.fullmatch("moe_route")
    assert not KERNEL_NAME.fullmatch("delta_layer")
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="print the planned kernel names")
    parser.add_argument("--rust-list", metavar="FILE", help="names printed by the Rust inventory (one per line) to compare with")
    parser.add_argument("--self-test", action="store_true", help="test the scanner and exit")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    defined = defined_kernels()
    planned, problems = planned_kernels()
    if args.list:
        print("\n".join(sorted(planned)))
        return 0

    missing = sorted(n for n in planned if n not in defined)
    unplanned = sorted(
        n for n in defined if n not in planned and not n.endswith("_t") and n not in KNOWN_UNPLANNED
    )
    print(f"{len(defined)} kernels defined in {len(list(METAL_DIR.glob('*.metal')))} files, {len(planned)} planned")
    for p in problems:
        print(f"PROBLEM: {p}")
    for n in missing:
        print(f"MISSING: {n} (planned in {planned[n]}) is not defined in any .metal file")
    for n in unplanned:
        print(f"UNPLANNED: {n} (defined in {defined[n]}) is not named by any planner")
    status = 1 if (missing or problems) else 0

    if args.rust_list:
        rust = {line.strip() for line in pathlib.Path(args.rust_list).read_text().splitlines() if line.strip()}
        only_rust = sorted(rust - set(planned))
        only_here = sorted(set(planned) - rust)
        for n in only_rust:
            print(f"DIFF: {n} is in the Rust inventory but not found by this script")
        for n in only_here:
            print(f"DIFF: {n} was found by this script but is not in the Rust inventory")
        status = status or (1 if (only_rust or only_here) else 0)

    print("OK" if status == 0 and not unplanned else ("FAILED" if status else "OK (with unplanned kernels)"))
    return status


if __name__ == "__main__":
    sys.exit(main())
