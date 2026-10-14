// Scaffold kernel: fills a float buffer with a constant.
kernel void scaffold_fill(device float *data [[buffer(0)]], constant uint &n [[buffer(1)]], constant float &value [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) data[i] = value;
}
