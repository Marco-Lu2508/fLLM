#!/bin/bash
# Mutation test of the emulation itself: every mutation below breaks one kernel on purpose (wrong index, wrong tie rule, missing barrier, ...);
# the emulation must FAIL for each of them (a mutation the emulation does not notice would be a blind spot). Mutations that remove a barrier are
# only reliably noticed with TSAN=1.
# usage: mutation_check.sh <vectors.bin>        environment as run_emulation.sh; MUT_ONLY=<substring> runs only the mutations whose name contains it
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/.." && pwd)
VECTORS=${1:?usage: mutation_check.sh <vectors.bin>}
W=${EMU_MUT_DIR:-/tmp/p06c-mut}
missed=0
total=0
mutate() { # name file sed-expression
  local name=$1 file=$2 expr=$3
  if [ -n "${MUT_ONLY:-}" ] && [[ "$name" != *$MUT_ONLY* ]]; then return; fi
  rm -rf "$W" && mkdir -p "$W/metal"
  cp "$SRC"/*.metal "$W/metal/"
  sed -i -E "$expr" "$W/metal/$file"
  if cmp -s "$SRC/$file" "$W/metal/$file"; then echo "MUTATION DID NOT APPLY: $name"; missed=$((missed + 1)); total=$((total + 1)); return; fi
  total=$((total + 1))
  if EMU_BUILD="$W/build" EMU_METAL="$W/metal" "$HERE/run_emulation.sh" "$VECTORS" > "$W/out.txt" 2>&1; then
    echo "NOT DETECTED: $name"; missed=$((missed + 1))
  else
    echo "detected    : $name  ($(grep -c '^\[FAIL\]' "$W/out.txt") failed checks$(grep -q 'ThreadSanitizer' "$W/out.txt" && echo ', ThreadSanitizer report'))"
  fi
}
mutate "q6_k scale index"            common_rec.metal 's/b\[192 \+ n \* 8u \+ grp \* 2u \+ jh\]/b[192 + n * 8u + grp * 2u]/'
mutate "q4_k scale high bits"        common_rec.metal 's/\(\(uint\(sc\[g - 4u\]\) >> 6\) << 4\)/((uint(sc[g - 4u]) >> 6) << 3)/'
mutate "q5_0 fifth bit"              common_rec.metal '0,/\(\(qh >> j\) & 1u\) << 4/s//((qh >> j) \& 1u) << 3/'
mutate "mxfp4 nibble order"          common_rec.metal '/struct r_fmt_mxfp4/,/^};/s/const bool hi = part >= 2u;/const bool hi = part < 2u;/'
mutate "moe_topk tie rule"           moe_route.metal  's/\(v == best && e < besti\)/(v == best \&\& e > besti)/'
mutate "moe_topk cross-lane tie rule" moe_route.metal 's/\(ov == s_val\[lane\] && oi < s_idx\[lane\]\)/(ov == s_val[lane] \&\& oi > s_idx[lane])/'
mutate "moe_linear invalid expert"   moe_linear.metal 's/uint\(expert\) < p.n_experts/uint(expert) <= p.n_experts/'
mutate "delta_conv current tap"      delta_conv.metal 's/if \(k == hist_len\) acc \+= wk\[k\] \* cur;/if (k == hist_len) acc += wk[0] * cur;/'
mutate "delta_conv qscale"           delta_conv.metal 's/acc \* inv \* p.qscale/acc * inv/'
mutate "delta_rec forget order"      delta_rec.metal  's/s\[i\] \*= forget;/s[i] *= 1.0f;/'
mutate "mamba_conv history rewrite"  mamba_conv.metal 's/if \(tgp.y == 0u\) \{/if (tgp.y == 99u) {/'
mutate "mamba_scan2 group of head"   mamba_scan2.metal 's/const uint g = h \/ max\(p.heads \/ p.groups, 1u\);/const uint g = h % max(p.groups, 1u);/'
mutate "mamba_scan1 reset"           mamba_scan1.metal 's/\(ch_ok && p.reset == 0u\) \? state\[cbase \+ k\] : 0.0f/ch_ok ? state[cbase + k] : 0.0f/'
mutate "gate_split_q per_head"       gate_attention.metal 's/sq \+ p.dk : base \+ width/sq + p.dk + 1 : base + width/'
mutate "barrier: delta_rec s_qk"     delta_rec.metal  '0,/threadgroup_barrier\(mem_flags::mem_threadgroup\);/s//\/\/ no barrier/'
mutate "barrier: mamba_scan2 end"    mamba_scan2.metal 's/threadgroup_barrier\(mem_flags::mem_threadgroup\);  \/\/ sb \/ sc are rewritten by the next token//'
mutate "barrier: r_tg_sum tail"      common_rec.metal '/const float r = red\[0\];/{n;s/threadgroup_barrier.*/\/\/ none/}'
echo "$((total - missed)) of $total mutations detected"
[ "$missed" = 0 ]
