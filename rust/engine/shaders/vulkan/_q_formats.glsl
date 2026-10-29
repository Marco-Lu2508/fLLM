// FLLM quantized linear: selects the weight format (exactly one FMT_* macro is defined by the variants list).
// Every format file defines QK (weights per block) and
//   void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm)
// g = global block index (row * nb + block in row), cin = chunk (8 weights) inside the block.
// Formats whose values are not small integers (FP8, NF4, FP4) define FLOAT_CHUNK instead and
//   void wchunk_f(uint g, uint cin, out float v[8])   (the 8 weight values); only the f32 shaders (mv_scalar, mm_scalar) accept them.
#if defined(FMT_Q4_0)
#include "_q_fmt_q4_0.glsl"
#elif defined(FMT_Q4_1)
#include "_q_fmt_q4_1.glsl"
#elif defined(FMT_Q5_0)
#include "_q_fmt_q5_0.glsl"
#elif defined(FMT_Q5_1)
#include "_q_fmt_q5_1.glsl"
#elif defined(FMT_Q8_0)
#include "_q_fmt_q8_0.glsl"
#elif defined(FMT_Q2_K)
#include "_q_fmt_q2_k.glsl"
#elif defined(FMT_Q3_K)
#include "_q_fmt_q3_k.glsl"
#elif defined(FMT_Q4_K)
#include "_q_fmt_q4_k.glsl"
#elif defined(FMT_Q5_K)
#include "_q_fmt_q5_k.glsl"
#elif defined(FMT_Q6_K)
#include "_q_fmt_q6_k.glsl"
#elif defined(FMT_IQ4_NL)
#include "_q_fmt_iq4_nl.glsl"
#elif defined(FMT_MXFP4)
#include "_q_fmt_mxfp4.glsl"
#elif defined(FMT_NVFP4)
#include "_q_fmt_nvfp4.glsl"
#elif defined(FMT_Q4Z)
#include "_q_fmt_q4z.glsl"
#elif defined(FMT_FP8B)
#include "_q_fmt_fp8b.glsl"
#elif defined(FMT_NF4) || defined(FMT_FP4)
#include "_q_fmt_nf4.glsl"
#elif defined(FMT_INT8R)
#include "_q_fmt_int8r.glsl"
#elif defined(FMT_MLX)
#include "_q_fmt_mlx.glsl"
#else
#error "define one FMT_* macro (see variants_quant.list)"
#endif