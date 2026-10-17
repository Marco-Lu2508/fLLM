"""Check Rust CPU/GPU dequantization and matmul against independent gguf/PyTorch."""
import argparse
import json
import subprocess


def fixture(path):
    import numpy as np
    from gguf import GGMLQuantizationType, GGUFWriter
    writer = GGUFWriter(path, "llama")
    rng = np.random.default_rng(41209)
    layouts = {2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210)}
    for kind, (block, width) in layouts.items():
        raw = rng.integers(0, 256, (3 * 256 // block, width), dtype=np.uint8)
        offset = 208 if kind == 14 else 0
        raw[:, offset:offset + 2] = np.frombuffer(np.float16(0.05).tobytes(), dtype=np.uint8)
        if kind in [3, 7, 12, 13]:
            raw[:, 2:4] = np.frombuffer(np.float16(0.02).tobytes(), dtype=np.uint8)
        writer.add_tensor(f"type{kind}.weight", raw.reshape(3, -1), raw_dtype=GGMLQuantizationType(kind))
    writer.add_tensor("float.weight", rng.normal(0, 1, (3, 256)).astype(np.float32))
    writer.add_tensor("half.weight", rng.normal(0, 1, (3, 256)).astype(np.float16))
    bf16 = (rng.normal(0, 1, (3, 256)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    writer.add_tensor("bf16.weight", bf16, raw_dtype=GGMLQuantizationType.BF16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    import numpy as np
    import torch
    from gguf import GGUFReader, dequantize
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--binary")
    parser.add_argument("--result", help="Saved Rust example JSON, for another machine")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.create:
        fixture(args.fixture)
    if not args.binary and not args.result:
        return
    if args.result:
        with open(args.result) as file:
            results = json.load(file)
    else:
        results = json.loads(subprocess.check_output([args.binary, args.fixture, args.device], text=True))
    reader = GGUFReader(args.fixture)
    inputs = torch.tensor([((j % 97) - 48) / 49 for j in range(512)], dtype=torch.float32).reshape(2, 256)
    for result, tensor in zip(results["results"], reader.tensors, strict=True):
        assert result["name"] == tensor.name
        weights = np.array(dequantize(tensor.data, tensor.tensor_type), copy=True)
        rust_weights = np.array(result["weights"], dtype=np.float32).reshape(3, 256)
        assert np.max(np.abs(weights - rust_weights)) < 1e-5, tensor.name
        expected = torch.nn.functional.linear(inputs, torch.from_numpy(weights)).numpy().reshape(-1)
        actual = np.array(result["output"], dtype=np.float32)
        maximum = float(np.max(np.abs(expected - actual)))
        assert np.allclose(expected, actual, atol=0.003, rtol=1e-5), (tensor.name, maximum)
        print(json.dumps({"device": results["device"], "tensor": tensor.name, "matmul_max_abs_error": maximum}), flush=True)


if __name__ == "__main__":
    main()
