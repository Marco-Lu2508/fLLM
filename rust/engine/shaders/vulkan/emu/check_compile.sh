#!/bin/bash
# Compiles every line of ../variants_quant.list like build.rs does (glslc --target-env=vulkan1.2 -O -I <dir> -D...) and validates the
# result with spirv-val --target-env vulkan1.2. Optional second compiler: GLSLANG=/path/to/glslang (>= 16, knows GL_EXT_integer_dot_product)
# builds the same lines with the hardware dot-product flavour of the `_i` shaders.
#   usage: check_compile.sh [list-file]
HERE=$(cd "$(dirname "$0")" && pwd)
SH=$(dirname "$HERE")
LIST=${1:-$SH/variants_quant.list}
OUT=${EMU_WORK:-/tmp/fllm-vk-emu}/spv
rm -rf "$OUT"; mkdir -p "$OUT/glslang"
GLSLC=${GLSLC:-glslc}
ok=0; bad=0
while read -r line; do
  line=${line%%#*}
  set -- $line
  [ $# -lt 2 ] && continue
  name=$1; src=$2; shift 2
  if "$GLSLC" --target-env=vulkan1.2 -O -I "$SH" "$@" -o "$OUT/$name.spv" "$SH/$src" 2> "$OUT/$name.err" && spirv-val --target-env vulkan1.2 "$OUT/$name.spv" 2>> "$OUT/$name.err"; then
    ok=$((ok+1))
  else
    bad=$((bad+1)); echo "FAIL (glslc) $name"; head -8 "$OUT/$name.err"
  fi
  if [ -n "$GLSLANG" ]; then
    if "$GLSLANG" -V --target-env vulkan1.2 -I"$SH" "$@" -o "$OUT/glslang/$name.spv" "$SH/$src" > "$OUT/glslang/$name.err" 2>&1 && spirv-val --target-env vulkan1.2 "$OUT/glslang/$name.spv" 2>> "$OUT/glslang/$name.err"; then
      ok=$((ok+1))
    else
      bad=$((bad+1)); echo "FAIL (glslang) $name"; head -8 "$OUT/glslang/$name.err"
    fi
  fi
done < "$LIST"
echo "compiled and validated: ok=$ok failed=$bad"
[ "$bad" = 0 ]
