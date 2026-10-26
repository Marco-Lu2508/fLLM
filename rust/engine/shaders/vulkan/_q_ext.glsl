// FLLM quantized linear: extension handling shared by the mv_*/mm_* shaders. Included first (right after #version).
// The integer dot product is used when the compiler knows GL_EXT_integer_dot_product (glslang >= 15.2 / shaderc 2025). Older compilers (the Vulkan SDK
// 1.4.309 used by the Windows release build has glslang 15.0) know GL_EXT_spirv_intrinsics instead: the shaders then emit OpSDot / OpSDotAccSat
// themselves (capabilities DotProduct + DotProductInput4x8BitPacked, extension SPV_KHR_integer_dot_product: exactly what the native path emits), so
// the dot product is one hardware instruction with every compiler. Only a compiler that knows neither gets an exact emulation with plain integer ops.
// FLLM_HW_DOT = 1: native GLSL functions, 2: spirv_instruction. FLLM_NO_HW_DOT forces the emulation, FLLM_NO_SPIRV_DOT only turns variant 2 off.
#extension GL_EXT_control_flow_attributes : enable
#if defined(GL_EXT_integer_dot_product) && !defined(FLLM_NO_HW_DOT)
#extension GL_EXT_integer_dot_product : require
#define FLLM_HW_DOT 1
#elif defined(GL_EXT_spirv_intrinsics) && !defined(FLLM_NO_HW_DOT) && !defined(FLLM_NO_SPIRV_DOT)
#extension GL_EXT_spirv_intrinsics : require
#define FLLM_HW_DOT 2
// OpSDot = 4450, OpSDotAccSat = 4453; capabilities 6018 = DotProductInput4x8BitPacked, 6019 = DotProduct; the literal 0 is PackedVectorFormat4x8Bit
spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6018, 6019], id = 4450)
int fllm_sdot(int a, int b, spirv_literal int packed_format);
spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6018, 6019], id = 4453)
int fllm_sdot_acc(int a, int b, int acc, spirv_literal int packed_format);
#endif
