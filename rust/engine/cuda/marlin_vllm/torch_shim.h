// fLLM: replaces the PyTorch exception header that vLLM's Marlin sources include. The checks of scalar_type.hpp only guard
// compile-time constants, so they compile to nothing here; the host wrapper (marlin_vllm.cu) validates its arguments itself.
#pragma once
#define STD_TORCH_CHECK(cond, ...) ((void)(cond))
