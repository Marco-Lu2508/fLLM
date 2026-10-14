#!/bin/bash
# Compile and validate EVERY Vulkan shader of the registry like build.rs does (glslc --target-env=vulkan1.2 -O -I <this dir>; variants from every
# variants*.list, every other non-include *.comp once under its file name), then spirv-val each module and, when glslangValidator is installed,
# compile everything a second time with it. Needs glslc and spirv-val. Lines marked '#optional' (cooperative matrices) only warn when they fail.
# usage: check_all.sh        (exit status = number of hard failures)
cd "$(dirname "$0")" || exit 1
OUT=$(mktemp -d /tmp/fllm_all_spv.XXXXXX)
trap 'rm -rf "$OUT"' EXIT
fail=0; ok=0; second=0; optional_skipped=0; total=0
run() { # <output name> <source> <optional 0|1> [-D...]
  local name=$1 src=$2 opt=$3; shift 3
  total=$((total+1))
  if ! glslc --target-env=vulkan1.2 -O -I . "$@" -o "$OUT/$name.spv" "$src" 2> "$OUT/$name.err"; then
    if [ "$opt" = 1 ]; then echo "optional shader $name skipped (glslc rejected $src)"; optional_skipped=$((optional_skipped+1)); return; fi
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
for list in variants*.list; do
  [ -f "$list" ] || continue
  while IFS= read -r raw; do
    opt=0; case "$raw" in *'#optional'*) opt=1;; esac
    line=${raw%%#*}
    set -- $line
    [ $# -ge 2 ] || continue
    out=$1; src=$2; shift 2
    listed="$listed$src "
    run "$out" "$src" "$opt" "$@"
  done < "$list"
done
for src in *.comp; do
  [ -f "$src" ] || continue
  case "$listed" in *" $src "*) continue;; esac
  run "${src%.comp}" "$src" 0
done
echo "modules: $total  glslc + spirv-val OK=$ok  second compiler OK=$second  optional skipped=$optional_skipped  FAIL=$fail"
exit $fail
