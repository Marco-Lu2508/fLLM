#!/usr/bin/env python3
"""Vision-language architectures against transformers.

For every fixture a tiny random checkpoint is built with the official transformers implementation,
converted with `fllm-rs convert` to an F32 GGUF (vision tower included) and run through the CPU
backend of the engine (`examples/verify_multimodal`) for a synthetic image plus a prompt. The
logits of every position are compared with the transformers logits (eager attention, float32): the
relative error is `max |a - b| / (max b - min b)` and must stay below `--tol` (default 2e-3); the
arg-max of every position has to agree unless the reference itself has a near tie.

The Rust side decodes the PNG itself (resize, normalisation, patchify), so the check covers the
whole path from image bytes to logits. Image sizes are chosen so that no resize happens: the
bicubic resampling of PIL and of the Rust `image` crate are not bit-identical.

    cargo build --release -p fllm-engine --bin fllm-rs --example verify_multimodal
    python scripts/verify_multimodal.py --fllm target/release/fllm-rs \
        --example target/release/examples/verify_multimodal
    python scripts/verify_multimodal.py --list

Requires torch, transformers, tokenizers, pillow and safetensors. Nothing is downloaded: the
checkpoints are random.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SPECIALS = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Describe the picture: what do you see here?",
    "user assistant system",
    "def add(a, b):\n    return a + b\n",
    "fLLM runs language models in pure Rust, on CPU and GPU.",
] * 8
SEED = 4321
FPS = 2.0
GENERATED = 8

FIXTURES = {
    "qwen2-vl": "Qwen2-VL: ViT with 2D RoPE, LayerNorm, QuickGELU MLP, patch merger, M-RoPE",
    "qwen2.5-vl": "Qwen2.5-VL: RMSNorm, SwiGLU MLP, window attention with full-attention layers, M-RoPE",
    "qwen3-vl": "Qwen3-VL: learned interpolated position embeddings, DeepStack, interleaved M-RoPE, per-frame video",
}

CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for c in message['content'] %}"
    "{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}"
    "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def make_tokenizer(out: Path, vocab: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS, trainer)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "add_bos_token": False,
                "add_eos_token": False,
                "chat_template": CHAT_TEMPLATE,
            }
        )
    )
    return tok


def synthetic_image(height: int, width: int, shift: int = 0):
    """A smooth, non-trivial RGB test pattern (deterministic, no network)."""
    import numpy as np
    from PIL import Image

    y, x = np.mgrid[0:height, 0:width]
    x = x + shift
    r = 128 + 100 * np.sin(x / 7.0) * np.cos(y / 11.0)
    g = 128 + 100 * np.sin((x + y) / 13.0)
    b = 128 + 100 * np.cos(x / 5.0 - y / 9.0)
    rng = np.random.default_rng(SEED)
    pixels = np.stack([r, g, b], axis=-1) + rng.normal(0, 6, (height, width, 3))
    return Image.fromarray(np.clip(pixels, 0, 255).astype("uint8"))


def qwen_configs(variant: str, vocab: int):
    import transformers as T

    text = dict(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3]},
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    ids = dict(image_token_id=8, video_token_id=9, vision_start_token_id=6, vision_end_token_id=7)
    if variant == "qwen2-vl":
        vision = dict(
            depth=3,
            embed_dim=32,
            hidden_size=64,
            num_heads=2,
            mlp_ratio=2,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            in_channels=3,
            hidden_act="quick_gelu",
        )
        return T.Qwen2VLConfig(text_config=text, vision_config=vision, tie_word_embeddings=True, **ids), T.Qwen2VLForConditionalGeneration
    if variant == "qwen3-vl":
        text3 = dict(text, head_dim=16, rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [4, 2, 2], "mrope_interleaved": True})
        vision3 = dict(
            depth=4,
            hidden_size=32,
            intermediate_size=48,
            num_heads=2,
            out_hidden_size=64,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            in_channels=3,
            num_position_embeddings=16,
            deepstack_visual_indexes=[1, 2],
            hidden_act="gelu_pytorch_tanh",
        )
        return T.Qwen3VLConfig(text_config=text3, vision_config=vision3, tie_word_embeddings=True, **ids), T.Qwen3VLForConditionalGeneration
    vision = dict(
        depth=3,
        hidden_size=32,
        intermediate_size=48,
        num_heads=2,
        out_hidden_size=64,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        window_size=56,
        fullatt_block_indexes=[2],
        hidden_act="silu",
    )
    return T.Qwen2_5_VLConfig(text_config=text, vision_config=vision, tie_word_embeddings=True, **ids), T.Qwen2_5_VLForConditionalGeneration


def build(variant: str, out: Path, vocab: int = 320):
    import torch
    import transformers as T

    out.mkdir(parents=True, exist_ok=True)
    tok = make_tokenizer(out, vocab)
    vocab = tok.get_vocab_size()
    config, cls = qwen_configs(variant, vocab)
    torch.manual_seed(SEED)
    model = cls._from_config(config, dtype=torch.float32, attn_implementation="eager")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or "norm" in name.split(".")[-2] or name.endswith("ln_q.weight"):
                p.copy_(0.8 + 0.4 * torch.rand_like(p))
            elif p.dim() > 1:
                p.normal_(0, 0.12)
            else:
                p.normal_(0, 0.05)
    model.eval()
    model.save_pretrained(out, safe_serialization=True)

    v3 = variant == "qwen3-vl"
    height, width = (96, 128) if v3 else (84, 112)
    mean = [0.5, 0.5, 0.5] if v3 else [0.48145466, 0.4578275, 0.40821073]
    std = [0.5, 0.5, 0.5] if v3 else [0.26862954, 0.26130258, 0.27577711]
    patch = 16 if v3 else 14
    image = synthetic_image(height, width)
    image.save(out / "image.png")
    (out / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "Qwen2VLImageProcessor",
                "do_resize": True,
                "do_rescale": True,
                "rescale_factor": 1 / 255,
                "do_normalize": True,
                "image_mean": mean,
                "image_std": std,
                "patch_size": patch,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "min_pixels": 3136,
                "max_pixels": 1003520,
            }
        )
    )
    processor = T.Qwen2VLImageProcessor(patch_size=patch, image_mean=mean, image_std=std)
    batch = processor(images=[image], return_tensors="pt")
    grid = batch["image_grid_thw"]
    merge = config.vision_config.spatial_merge_size
    count = int(grid.prod(dim=1).sum()) // (merge * merge)

    generator = torch.Generator().manual_seed(SEED + 1)
    before = torch.randint(10, vocab, (5,), generator=generator).tolist()
    after = torch.randint(10, vocab, (4,), generator=generator).tolist()
    ids = before + [6] + [8] * count + [7] + after
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            pixel_values=batch["pixel_values"],
            image_grid_thw=grid,
            mm_token_type_ids=(input_ids == 8).int(),
            use_cache=False,
        ).logits[0].float()
        greedy = model.generate(
            input_ids=input_ids,
            pixel_values=batch["pixel_values"],
            image_grid_thw=grid,
            mm_token_type_ids=(input_ids == 8).int(),
            max_new_tokens=GENERATED,
            do_sample=False,
            pad_token_id=0,
        )[0, len(ids) :].tolist()
    # a four-frame clip: two temporal patches, the video tokens carry the time axis of M-RoPE
    import numpy as np

    frames = [synthetic_image(height, width, shift=9 * i) for i in range(4)]
    for i, frame in enumerate(frames):
        frame.save(out / f"frame{i}.png")
    video_size = {"shortest_edge": 3136, "longest_edge": 1003520}
    (out / "video_preprocessor_config.json").write_text(json.dumps({"size": video_size, "do_sample_frames": False}))
    video_processor = (T.Qwen3VLVideoProcessor if v3 else T.Qwen2VLVideoProcessor)(size=video_size, do_sample_frames=False)
    vbatch = video_processor(videos=[np.stack([np.asarray(f) for f in frames])], return_tensors="pt")
    vgrid = vbatch["video_grid_thw"]
    vcount = int(vgrid.prod(dim=1).sum()) // (merge * merge)
    if v3:
        # one run per temporal patch, each announced by (stand-in) time stamp tokens
        per_frame = vcount // int(vgrid[0, 0])
        video_ids = before
        for g in range(int(vgrid[0, 0])):
            video_ids = video_ids + [20 + g, 21 + g, 6] + [9] * per_frame + [7]
        video_ids = video_ids + after
    else:
        video_ids = before + [6] + [9] * vcount + [7] + after
    video_tensor = torch.tensor([video_ids])
    extra = {"second_per_grid_ts": torch.tensor([config.vision_config.temporal_patch_size / FPS])} if variant == "qwen2.5-vl" else {}
    with torch.no_grad():
        video_logits = model(
            input_ids=video_tensor,
            pixel_values_videos=vbatch["pixel_values_videos"],
            video_grid_thw=vgrid,
            mm_token_type_ids=(video_tensor == 9).int() * 2,
            use_cache=False,
            **extra,
        ).logits[0].float()

    # the same request as a chat: template, expanded image placeholder, greedy continuation
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(out))
    chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the picture"}]}]
    text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    text = text.replace("<|image_pad|>", "<|image_pad|>" * count)
    chat_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chat_tensor = torch.tensor([chat_ids])
    with torch.no_grad():
        chat_greedy = model.generate(
            input_ids=chat_tensor,
            pixel_values=batch["pixel_values"],
            image_grid_thw=grid,
            mm_token_type_ids=(chat_tensor == 8).int(),
            max_new_tokens=GENERATED,
            do_sample=False,
            pad_token_id=0,
        )[0, len(chat_ids) :].tolist()
    golden = dict(
        video_ids=video_ids,
        video_grid=vgrid[0].tolist(),
        video_logits=video_logits.reshape(-1).tolist(),
        video_shape=list(video_logits.shape),
        chat_text=tokenizer.decode(chat_greedy, skip_special_tokens=True),
        arch=variant,
        ids=ids,
        grid=grid[0].tolist(),
        logits=logits.reshape(-1).tolist(),
        shape=list(logits.shape),
        greedy=greedy,
        image_tokens=count,
        pixel_values=batch["pixel_values"].reshape(-1).tolist(),
    )
    (out / "golden.json").write_text(json.dumps(golden))
    return golden


def run(cmd, env=None):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env or os.environ)


def compare(mine, want, shape, argmax_margin=4.0):
    positions, vocab = shape
    span = max(want) - min(want)
    worst = 0.0
    worst_at = (0, 0)
    mismatches = 0
    for p in range(positions):
        a = mine[p * vocab : (p + 1) * vocab]
        b = want[p * vocab : (p + 1) * vocab]
        diffs = [abs(x - y) for x, y in zip(a, b)]
        position_worst = max(diffs)
        if position_worst > worst:
            worst, worst_at = position_worst, (p, diffs.index(position_worst))
        mine_top = max(range(vocab), key=a.__getitem__)
        ref_top = max(range(vocab), key=b.__getitem__)
        if mine_top != ref_top and b[ref_top] - b[mine_top] > argmax_margin * position_worst:
            mismatches += 1
    return worst, worst_at, (worst / span if span else 0.0), mismatches


def served_chat(fllm: str, gguf: Path, image: Path, count: int) -> str:
    """Greedy chat continuation of a prompt with an image through the real server (scheduler, paged cache) on the CPU."""
    import base64
    import socket
    import time
    import urllib.request

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen(
        [fllm, "serve", "--model", str(gguf), "--device", "cpu", "--ctx", "256", "--port", str(port), "--max-seqs", "2", "--cache-type-k", "f32", "--cache-type-v", "f32"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
        body = json.dumps(
            {
                "model": "x",
                "max_tokens": count,
                "temperature": 0,
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Describe the picture"}]}],
            }
        ).encode()
        for _ in range(200):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
                reply = json.load(urllib.request.urlopen(req, timeout=120))
                return reply["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as error:
                raise RuntimeError(error.read().decode()) from error
            except OSError:
                time.sleep(0.3)
        raise RuntimeError("server did not answer")
    finally:
        server.kill()
        server.wait()


def verify(variant: str, fllm: str, example: str, workdir: Path, tol: float, keep: bool, generate: bool = False) -> dict:
    src = workdir / f"{variant}-hf"
    golden = build(variant, src)
    gguf = workdir / f"{variant}.gguf"
    run([fllm, "convert", str(src), str(gguf), "--outtype", "f32", "--overwrite"])
    request = workdir / f"{variant}-request.json"
    request.write_text(json.dumps({"ids": golden["ids"], "image": str(src / "image.png")}))
    out = run([example, str(gguf), str(request)])
    reply = json.loads(out.stdout.strip().splitlines()[-1])
    mine = reply["logits"]
    want = golden["logits"]
    if len(mine) != len(want):
        return {"arch": variant, "ok": False, "error": f"logit count {len(mine)} != {len(want)}"}
    worst, at, relative, mismatches = compare(mine, want, golden["shape"])
    result = {
        "arch": variant,
        "max_abs": worst,
        "worst_position_token": at,
        "relative": relative,
        "argmax_mismatches": mismatches,
        "positions": golden["shape"][0],
        "image_tokens": golden["image_tokens"],
        "grid": golden["grid"],
    }
    merge = 2
    expected_grid = [golden["grid"][0], golden["grid"][1] // merge, golden["grid"][2] // merge]
    if reply.get("grid") != expected_grid:
        result["error"] = f"token grid {reply.get('grid')} != {expected_grid}"
    # negative control: another image must change the logits, otherwise the check proves nothing
    from PIL import Image

    Image.new("RGB", (112, 84), (255, 255, 255)).save(src / "other.png")
    request.write_text(json.dumps({"ids": golden["ids"], "image": str(src / "other.png")}))
    other = json.loads(run([example, str(gguf), str(request)]).stdout.strip().splitlines()[-1])["logits"]
    result["image_sensitivity"] = compare(other, want, golden["shape"])[2]
    if result["image_sensitivity"] < 20 * tol:
        result["error"] = "logits do not depend on the image"
    # negative control: ordinary positions must not reproduce the reference (M-RoPE is exercised)
    request.write_text(json.dumps({"ids": golden["ids"], "image": str(src / "image.png")}))
    plain = json.loads(run([example, str(gguf), str(request)], env=dict(os.environ, VERIFY_PLAIN_ROPE="1")).stdout.strip().splitlines()[-1])["logits"]
    result["mrope_sensitivity"] = compare(plain, want, golden["shape"])[2]
    if result["mrope_sensitivity"] < 20 * tol:
        result["error"] = "logits do not depend on the M-RoPE positions"
    if variant == "qwen3-vl":
        request.write_text(json.dumps({"ids": golden["ids"], "image": str(src / "image.png")}))
        bare = json.loads(run([example, str(gguf), str(request)], env=dict(os.environ, VERIFY_NO_DEEPSTACK="1")).stdout.strip().splitlines()[-1])["logits"]
        result["deepstack_sensitivity"] = compare(bare, want, golden["shape"])[2]
        if result["deepstack_sensitivity"] < 20 * tol:
            result["error"] = "logits do not depend on the DeepStack features"
    # video: four frames through the temporal patches and the time axis of M-RoPE
    request.write_text(json.dumps({"ids": golden["video_ids"], "video": [str(src / f"frame{i}.png") for i in range(4)], "fps": FPS}))
    video = json.loads(run([example, str(gguf), str(request)]).stdout.strip().splitlines()[-1])
    v_worst, _, v_relative, v_mismatches = compare(video["logits"], golden["video_logits"], golden["video_shape"])
    result["video_relative"] = v_relative
    result["video_argmax_mismatches"] = v_mismatches
    vg = golden["video_grid"]
    if video.get("grid") != [vg[0], vg[1] // merge, vg[2] // merge]:
        result["error"] = f"video token grid {video.get('grid')} != {vg}"
    if v_relative > tol or v_mismatches:
        result["error"] = f"video logits differ from the reference (relative {v_relative:.3g})"
    if generate:
        got = served_chat(fllm, gguf, src / "image.png", len(golden["greedy"]))
        result["served_chat_matches"] = got == golden["chat_text"]
        if got != golden["chat_text"]:
            result["error"] = f"served chat {got!r} != reference {golden['chat_text']!r}"
    result["ok"] = relative <= tol and mismatches == 0 and "error" not in result
    if not keep:
        for path in (gguf, request):
            path.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help="fixtures to run (default: all)")
    parser.add_argument("--fllm")
    parser.add_argument("--example")
    parser.add_argument("--tol", type=float, default=2e-3)
    parser.add_argument("--workdir")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--generate", action="store_true", help="also compare a greedy chat continuation served by the real server with transformers")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for name, text in FIXTURES.items():
            print(f"{name:14} {text}")
        return 0
    if not args.fllm or not args.example:
        parser.error("--fllm and --example are required")
    names = args.variants or list(FIXTURES)
    unknown = [n for n in names if n not in FIXTURES]
    if unknown:
        parser.error(f"unknown fixtures {unknown}")
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="verify-vision-"))
    workdir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name in names:
        try:
            result = verify(name, args.fllm, args.example, workdir, args.tol, args.keep, args.generate)
        except subprocess.CalledProcessError as error:
            result = {"arch": name, "ok": False, "error": (error.stderr or error.stdout or str(error))[-2000:]}
        print(json.dumps(result))
        failed += 0 if result["ok"] else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
