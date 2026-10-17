#!/usr/bin/env bash
# Compile-only check of the AMD (HIP) kernel sources: every source for every gfx target, one hipcc call per pair, like build.rs does.
# No GPU is needed. Run it where hipcc is installed (the rocm/dev-ubuntu container, see docs/gpu-backends.md), from anywhere:
#
#   scripts/hipcheck.sh                      # all sources, the default target list
#   scripts/hipcheck.sh "gfx1100;gfx942"     # selected targets
#   RESOURCES=1 scripts/hipcheck.sh gfx1100  # also print register / LDS / spill figures of every kernel
#   SOURCES="hip/hip_q4z.cu cuda/mamba1.cu" scripts/hipcheck.sh gfx90a
#   JOBS=3 ...                               # parallel hipcc processes (default 3)
#
# The source list is the one of hip_kernels() in rust/engine/build.rs (the script reads it from there when SOURCES is not set).
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root/rust/engine" || exit 2
archs="${1:-gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201}"
jobs="${JOBS:-3}"
hipcc="${ROCM_PATH:-/opt/rocm}/bin/hipcc"
[ -x "$hipcc" ] || hipcc="$(command -v hipcc || true)"
[ -n "$hipcc" ] || { echo "hipcc not found (set ROCM_PATH)" >&2; exit 2; }

if [ -z "${SOURCES:-}" ]; then
    SOURCES="$(python3 - <<'PY'
import re
t = open("build.rs", encoding="utf-8").read()
b = t[t.index("fn hip_kernels()"):]
out = []
for name in ("let sources", "let hip_sources"):
    m = re.search(name + r"\s*=\s*\[(.*?)\];", b, re.S)
    items = re.findall(r'"([^"]+\.cu)"', m.group(1))
    out += [("hip/" if name.endswith("hip_sources") else "cuda/") + x for x in items]
print(" ".join(out))
PY
)"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
flags="-x hip -c -O3 -std=c++17 -ffast-math -DFLLM_NO_FP4 -Wno-unused-value -Wno-unused-result -fPIC"
[ "${RESOURCES:-0}" = 1 ] && flags="$flags --offload-device-only -Rpass-analysis=kernel-resource-usage"

one() {
    src="$1"; arch="$2"
    log="$tmp/$(echo "$src" | tr / _).$arch.log"
    src_flags="$flags"
    [ "$src" = "cuda/kv_quant.cu" ] && src_flags="${flags/-ffast-math/}"
    if $hipcc $src_flags --offload-arch="$arch" -o "$tmp/$(echo "$src" | tr / _).$arch.o" "$src" >"$log" 2>&1; then
        echo "ok    $arch $src"
    else
        echo "FAIL  $arch $src"; sed 's/^/      /' "$log" | head -40
    fi
    if [ "$RESOURCES" = 1 ]; then
        grep -E "remark: .*(Function Name|VGPRs|AGPRs|ScratchSize|Occupancy|LDS Size)" "$log" | sed 's/^[^ ]* remark: //' | sed 's/^/      /'
        # kernels that spill registers to scratch memory
        awk -v a="$arch" -v s="$src" '/Function Name:/ {f=$(NF-1)} /ScratchSize/ && $(NF-1)+0 > 0 {print "SPILL " a " " s " " f " scratch " $(NF-1) " bytes/lane"}' "$log" |
            sed 's/ \[-Rpass.*//'
    fi
    return 0
}
export -f one
export hipcc flags tmp RESOURCES="${RESOURCES:-0}"

for arch in ${archs//;/ }; do
    for src in $SOURCES; do echo "$src $arch"; done
done | xargs -P "$jobs" -L1 bash -c 'one "$0" "$1"' | tee "$tmp/result.txt"
if grep -q '^FAIL' "$tmp/result.txt"; then echo "hipcheck: failures"; exit 1; fi
grep '^ *SPILL' "$tmp"/result.txt | sort -u | sed 's/^ *//' || true
echo "hipcheck: all $(grep -c '^ok' "$tmp/result.txt") compilations passed (L0: compiled only)"
