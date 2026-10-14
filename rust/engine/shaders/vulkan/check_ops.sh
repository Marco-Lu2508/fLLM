#!/bin/bash
# Compile and validate the non-matmul Vulkan shaders (package 3: attn_ norm_ rope_ act_ kv_ moe_ delta_ mamba_ misc_ embed) exactly like build.rs
# does it (glslc --target-env=vulkan1.2 -O -I <this dir>, variants from variants_ops.list), then run spirv-val on every module and, when
# glslangValidator is installed, compile everything a second time with it. Needs glslc and spirv-val (Vulkan SDK / shaderc / spirv-tools).
# usage: check_ops.sh        (exit status = number of failures)
cd "$(dirname "$0")" || exit 1
OUT=$(mktemp -d /tmp/fllm_ops_spv.XXXXXX)
trap 'rm -rf "$OUT"' EXIT
fail=0; ok=0; second=0
run() { # <output name> <source> [-D...]
  local name=$1 src=$2; shift 2
  if ! glslc --target-env=vulkan1.2 -O -I . "$@" -o "$OUT/$name.spv" "$src" 2> "$OUT/$name.err"; then
    echo "GLSLC FAIL $name ($src $*)"; sed 's/^/    /' "$OUT/$name.err" | head -20; fail=$((fail+1)); return
  fi
  if [ -s "$OUT/$name.err" ]; then echo "glslc warnings $name:"; sed 's/^/    /' "$OUT/$name.err" | head -5; fi
  if ! spirv-val --target-env vulkan1.2 "$OUT/$name.spv" 2> "$OUT/$name.val"; then
    echo "SPIRV-VAL FAIL $name"; sed 's/^/    /' "$OUT/$name.val" | head -10; fail=$((fail+1)); return
  fi
  ok=$((ok+1))
  if command -v glslangValidator > /dev/null; then
    if glslangValidator -V -I. --target-env vulkan1.2 "$@" "$src" -o "$OUT/$name.gv.spv" > "$OUT/$name.gv" 2>&1; then second=$((second+1)); else
      echo "GLSLANGVALIDATOR FAIL $name"; head -5 "$OUT/$name.gv"; fail=$((fail+1)); fi
  fi
}
listed=" "
while read -r line; do
  line=${line%%#*}
  set -- $line
  [ $# -ge 2 ] || continue
  out=$1; src=$2; shift 2
  listed="$listed$src "
  run "$out" "$src" "$@"
done < variants_ops.list
for src in attn_*.comp norm_*.comp rope_*.comp act_*.comp kv_*.comp moe_*.comp delta_*.comp mamba_*.comp misc_*.comp embed.comp; do
  [ -f "$src" ] || continue
  case "$listed" in *" $src "*) continue;; esac
  run "${src%.comp}" "$src"
done
echo "glslc + spirv-val: OK=$ok  second compiler (glslangValidator): OK=$second  FAIL=$fail"
exit $fail
