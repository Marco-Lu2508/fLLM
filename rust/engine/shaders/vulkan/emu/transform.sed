s/\[\[unroll\]\]//g
s/^layout(push_constant) uniform \([A-Za-z]*\) {/struct \1 {/
s/^layout(std430, set = 0, binding = [0-9]) \(readonly \|writeonly \)\{0,1\}buffer [A-Za-z]* { \([A-Za-z0-9]*\) \([A-Za-z0-9_]*\)\[\]; };/static \2* \3;/
s/^layout(local_size_x_id = 0.*in;/constexpr uvec3 gl_WorkGroupSize{WG_SIZE, 1, 1};/
s/^layout(local_size_x = \([0-9]*\).*in;/constexpr uvec3 gl_WorkGroupSize{\1, 1, 1};/
s/^layout(constant_id = \([0-9]*\)) const uint \([A-Za-z_]*\) = [0-9]*;/const uint \2 = SPEC_\1;/
s/^shared /static /
s/out float \([a-z]*\)\[8\]/float (\&\1)[8]/
s/out \(uint\|float\) \([a-z0-9]*\)/\1\& \2/g
s/\bvoid main()/void main_()/
/^const int KV/ s/= int\[16\](/= {/
/^const int KV/ s/);$/};/
