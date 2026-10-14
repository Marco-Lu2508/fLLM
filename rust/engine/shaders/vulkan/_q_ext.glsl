// FLLM quantized linear: extension handling shared by the mv_*/mm_* shaders. Included first (right after #version).
// The integer dot product is used when the compiler knows GL_EXT_integer_dot_product (glslang >= 15.2 / shaderc 2025);
// older compilers get an exact emulation with plain integer arithmetic, so the `_i` shaders build everywhere.
#extension GL_EXT_control_flow_attributes : enable
#if defined(GL_EXT_integer_dot_product) && !defined(FLLM_NO_HW_DOT)
#extension GL_EXT_integer_dot_product : require
#define FLLM_HW_DOT 1
#endif
