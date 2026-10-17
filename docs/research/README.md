# Research notes

These are research notes, not user documentation. They record studies and evaluations that informed design decisions for optional backends; the
statements in them are estimates or unverified expectations unless a note says otherwise. For supported behaviour see the user documentation in [docs/](../index.md).
Test levels (L0-L3) used in the notes are defined in [test-levels.md](../test-levels.md).

| Note | Contents |
|---|---|
| [npu-study-ane.md](npu-study-ane.md) | Apple Neural Engine through Core ML: access from Rust, data formats, limits, expected benefit, recommendation. |
| [npu-study-hexagon.md](npu-study-hexagon.md) | Qualcomm Hexagon NPU (HTP) through QNN and Genie: interface, tensor formats, limits, recommendation. |
| [npu-study-xdna.md](npu-study-xdna.md) | AMD XDNA (Ryzen AI NPU) through ONNX Runtime and the VitisAI execution provider: access, formats, limits, recommendation. |
| [sycl-evaluation.md](sycl-evaluation.md) | Vulkan versus SYCL (oneAPI) for Intel GPUs: expected speed, installation effort, risks, decision rule and measurement plan. |

The Intel NPU design note (OpenVINO offload) lives next to its code: [rust/engine/src/backend/npu/OPENVINO.md](../../rust/engine/src/backend/npu/OPENVINO.md).
