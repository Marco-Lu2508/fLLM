# Expert weights in GGUF files: stacked tensors and per-expert scales

The experts of one MoE layer are stored as three stacked tensors per layer, `blk.N.ffn_gate_exps.weight`, `ffn_up_exps` and `ffn_down_exps`
(or one `ffn_gate_up_exps` with the gate rows followed by the up rows of every expert, as llama.cpp writes Qwen3.5). A stacked tensor has the
GGUF dimensions `[inputs, rows, experts]`: expert `e` is the block of `rows` rows starting at row `e * rows`. The loader slices that block
(`Linear::load_rows`), so every expert is an ordinary `Linear` of its own format.

## What changed

Before, `fllm-rs convert` kept 4-bit experts compact only for dense (non-expert) weights. Per-expert tensors of NVFP4, AWQ and GPTQ
checkpoints were expanded to F16 when they were stacked (four times the memory of the 4-bit weights). Now they stay compact:

| source format | stacked GGUF type | what is new |
|---|---|---|
| ModelOpt NVFP4 (`experts.N.*.weight` U8 + `weight_scale` + `weight_scale_2` + `input_scale`) | type 40 (NVFP4) | `<tensor>.scale` and `<tensor>.input_scale` have one F32 value **per expert** (dimension `[experts]`) |
| AWQ / GPTQ without act-order, compressed-tensors int4 (group 32, 64, 128) | `Q4Z32` / `Q4Z64` / `Q4Z128` | none: rows of all experts follow each other |
| GPTQ with act-order (`desc_act`), FP8 with 128x128 block scales, group sizes other than 32/64/128 | F16 (expanded, as before) | not changed; see "Not done" |

A file with one scale value (a plain, non-stacked NVFP4 tensor, or a stacked file written before this change) reads as before: the engine
takes entry `first / (total_rows / n)` of `n` scale values for the rows it loads (`compute::scale_for_rows`), and the single value of a tensor
with `n = 1`. For a fused `ffn_gate_up_exps` the gate and the up half of an expert share one entry.

The conversion needs one condition per stacked NVFP4 tensor: every expert fits the 64-value GGUF blocks, and either all experts have an
`input_scale` or none has one; otherwise the tensor is expanded as before.

## Where the device uses them

* CPU: every expert is a `Linear` over its slice, the NVFP4 factor of the expert is applied per product.
* CUDA: `upload_weight_scaled` gets the expert's own `scale` and `input_scale`, so each expert has its own device layout and trailer. Q4Z
  experts run through the Marlin GEMM per expert (the host-routed path, `moe_on_device`). The grouped device kernels
  (docs/changes/cuda-moe.md) cover F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q5_K and Q6_K experts; NVFP4 and Q4Z
  stacked experts stay on the per-expert path until a grouped Marlin / FP4 kernel exists.

## Checks

`cargo test -p fllm-engine --lib import_tests` converts tiny Mixtral-style checkpoints (4 experts, top-2) in NVFP4 and compressed-tensors
int4 and runs them on the CPU against the same model dequantized in F32 (`nvfp4_experts_stay_4_bit_with_one_scale_per_expert`,
`int4_experts_stay_q4z`, `block_wise_fp8_experts_convert_to_f16`). The experts of the NVFP4 model have different `weight_scale_2` values, so a
wrong per-expert factor changes the logits. `nvfp4::tests::stacked_experts_keep_their_own_scales_and_rows` and
`packed_quant::tests::stacked_experts_encode_the_rows_of_every_expert_as_q4z` check the row encoders across expert boundaries.

## Not done

* Block-wise FP8 (`weight_scale_inv` per 128x128 tile) has no compact tensor type: the tile scales do not map onto the per-row scales of
  `FP8M`. A new type with a tile-scale table and a GEMM prologue that applies it is needed; the experts are expanded to F16.
* Stacked GPTQ act-order experts (`Q4ZP`) need the permutation table once per tensor; the row slicing of `load_rows` does not handle the
  trailing table of a slice, so they stay expanded.
