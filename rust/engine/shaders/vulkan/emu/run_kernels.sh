#!/bin/bash
# CPU execution of the quantized-linear shaders: the preprocessed GLSL text of mv_*/mm_*/quant_* is transformed to C++ (cpp + transform.sed + shim.h),
# every workgroup runs one std::thread per invocation with std::barrier as barrier(), and the output is compared with an f64 reference built from the
# dequantized weights (row_base / out_stride / bias / scale / x-y workgroup split / act-order permutation included; cooperative matrices are
# emulated redundantly per invocation). This executes the shader TEXT on the CPU; it is not a GPU test (level L0 stays L0).
#   usage: run_kernels.sh               (dumps in $DUMP_DIR, default $EMU_WORK/dump; needs the *_big.bin files, see README.md)
# needs: g++ >= 13 (C++20 <barrier>), cpp, sed, xargs
HERE=$(cd "$(dirname "$0")" && pwd)
SH=$(dirname "$HERE")
W=${EMU_WORK:-/tmp/fllm-vk-emu}
D=${DUMP_DIR:-$W/dump}
JOBS=${JOBS:-6}
E=$W/kernels; S=$HERE
rm -rf "$E"; mkdir -p "$E/T"; cp "$SH"/*.comp "$SH"/*.glsl "$E/T/"
sed -i '/^#version/d;/^#extension/d' "$E"/T/*
FMTS="q4_0:-DFMT_Q4_0=1 q4_1:-DFMT_Q4_1=1 q5_0:-DFMT_Q5_0=1 q5_1:-DFMT_Q5_1=1 q8_0:-DFMT_Q8_0=1 q2_k:-DFMT_Q2_K=1 q3_k:-DFMT_Q3_K=1 q4_k:-DFMT_Q4_K=1 q5_k:-DFMT_Q5_K=1 q6_k:-DFMT_Q6_K=1 iq4_nl:-DFMT_IQ4_NL=1 mxfp4:-DFMT_MXFP4=1 nvfp4:-DFMT_NVFP4=1 q4z32:-DFMT_Q4Z=1,-DQ4Z_GROUP=32 q4z64:-DFMT_Q4Z=1,-DQ4Z_GROUP=64 q4z128:-DFMT_Q4Z=1,-DQ4Z_GROUP=128"
# config: kind,NROWS (spec constant 1),NCOLS (spec constant 2),workgroup size,batch,relative tolerance
CFGS="mv_s,2,1,64,1,2e-5 mv_s,2,3,64,3,2e-5 mv_s,1,8,32,8,2e-5 mv_i,2,1,64,1,0.01 mv_i,2,5,64,5,0.01 mm_s,0,0,256,70,2e-5 mm_i,0,0,256,70,0.01 mm_c32,0,0,128,70,0.004 mm_c16,0,0,64,70,0.004 mm_c64,0,0,256,70,0.004"
PERMCFGS="mv_s_perm,2,3,64,3,2e-5 mv_i_perm,2,5,64,5,0.01 mm_s_perm,0,0,256,70,2e-5 mm_i_perm,0,0,256,70,0.01 mm_c32_perm,0,0,128,70,0.004"
run_cfg() {
  fmt=$1; defs=${2//,/ }; cfg=$3
  IFS=, read kind s1 s2 wg batch rel <<< "$cfg"
  perm=0; kind0=$kind; case $kind in *_perm) perm=1; kind0=${kind%_perm};; esac
  d=$E/${fmt}_${kind}_${s2}_${batch}; mkdir -p "$d"
  case $kind0 in
    mv_s) src=mv_scalar.comp; gd="-DMV";;
    mv_i) src=mv_idot.comp; gd="-DMV -DIDOT";;
    mm_s) src=mm_scalar.comp; gd="-DMM";;
    mm_i) src=mm_idot.comp; gd="-DMM -DIDOT";;
    mm_c32) src=mm_coopmat.comp; gd="-DMM -DSUBGROUP_SIZE=32";;
    mm_c16) src=mm_coopmat.comp; gd="-DMM -DSUBGROUP_SIZE=16";;
    mm_c64) src=mm_coopmat.comp; gd="-DMM -DSUBGROUP_SIZE=64";;
  esac
  [ $perm = 1 ] && gd="$gd -DPERM"
  pre() { cpp -P -undef -nostdinc -I"$E/T" "${@:3}" -x c "$E/T/$2" | sed -f "$S/transform.sed" | { echo "namespace $1 {"; cat; echo "}"; }; }
  pre K $src $defs > "$d/sh_k.inc" 2> "$d/cpp.err" || { echo "CPP-FAIL $fmt $kind"; return 1; }
  case $kind0 in
    *_i) if [ $perm = 1 ]; then pre PRE quant_act_q8.comp -DPERM=1 > "$d/sh_pre.inc"; else pre PRE quant_act_q8.comp > "$d/sh_pre.inc"; fi;;
    *) [ $perm = 1 ] && pre PP quant_permute.comp > "$d/sh_pp.inc";;
  esac
  g++ -std=c++20 -O1 -pthread -w -I"$S" -I"$d" $gd -DSPEC_1=$s1 -DSPEC_2=$s2 -DWG_SIZE=$wg "$S/harness.cpp" -o "$d/run" 2> "$d/g++.err" || { echo "BUILD-FAIL $fmt $kind"; head -8 "$d/g++.err"; return 1; }
  res=$(timeout 300 "$d/run" "$D/${fmt}_big.bin" $batch $rel 2>&1); rc=$?
  echo "$fmt $kind ncols=$s2 batch=$batch nrows=$s1: $(echo "$res" | tail -3 | tr '\n' ' ') rc=$rc"
  return $rc
}
export -f run_cfg; export E S D SH
for f in $FMTS; do
  for c in $CFGS; do echo "${f%%:*} ${f#*:} $c"; done
  case $f in q4z*) for c in $PERMCFGS; do echo "${f%%:*} ${f#*:} $c"; done;; esac
done > "$E/jobs.txt"
xargs -P "$JOBS" -L 1 bash -c 'run_cfg "$0" "$1" "$2"' < "$E/jobs.txt" > "$E/results.txt" 2>&1
sort "$E/results.txt" | grep -v 'rc=0'
total=$(wc -l < "$E/jobs.txt"); ok=$(grep -c 'rc=0' "$E/results.txt")
echo "kernel configurations: total=$total ok=$ok failed=$((total - ok))   (details: $E/results.txt)"
[ "$ok" = "$total" ]
