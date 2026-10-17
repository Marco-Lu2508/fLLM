#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Verify that glibc-linked Linux binaries support runtime dlopen() of shared libraries
# (required for libcuda.so.1, libvulkan.so.1, OpenVINO, QNN, ONNX Runtime, and PJRT),
# whereas static musl binaries cannot dlopen glibc-linked host GPU drivers.
set -eu

BIN="${1:-}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

if [ -n "$BIN" ]; then
    if [ ! -f "$BIN" ]; then
        echo "verify-dlopen.sh: binary not found: $BIN" >&2
        exit 1
    fi
    if command -v readelf >/dev/null 2>&1; then
        if ! readelf -d "$BIN" 2>/dev/null | grep -q "NEEDED"; then
            echo "verify-dlopen.sh: $BIN has no dynamic NEEDED entries (static binary cannot dlopen glibc drivers)" >&2
            exit 1
        fi
        if ! readelf -d "$BIN" 2>/dev/null | grep -Eq "libc\.so|ld-linux"; then
            echo "verify-dlopen.sh: $BIN is not linked against glibc" >&2
            exit 1
        fi
        echo "ok - $BIN has dynamic glibc linkage"
    elif command -v ldd >/dev/null 2>&1; then
        if ldd "$BIN" 2>&1 | grep -q "not a dynamic executable"; then
            echo "verify-dlopen.sh: $BIN is a static executable" >&2
            exit 1
        fi
        echo "ok - $BIN is dynamically linked"
    fi
    "$BIN" --version >/dev/null
    echo "ok - $BIN --version succeeded"
fi

# Compile a minimal shared object and a C dlopen probe to prove runtime symbol loading on this host.
CC_BIN="${CC:-cc}"
if command -v "$CC_BIN" >/dev/null 2>&1; then
    cat > "$WORK/libprobe.c" << 'EOF'
int fllm_dlopen_probe_symbol(int x) { return x + 42; }
EOF
    cat > "$WORK/main.c" << 'EOF'
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*probe_fn)(int);

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    dlerror();
    probe_fn fn = (probe_fn)dlsym(h, "fllm_dlopen_probe_symbol");
    const char *err = dlerror();
    if (err != NULL || !fn) {
        fprintf(stderr, "dlsym failed: %s\n", err ? err : "null");
        dlclose(h);
        return 1;
    }
    int res = fn(8);
    dlclose(h);
    if (res != 50) {
        fprintf(stderr, "unexpected probe return: %d\n", res);
        return 1;
    }
    for (int i = 2; i < argc; ++i) {
        void *drv = dlopen(argv[i], RTLD_LAZY | RTLD_LOCAL);
        if (drv) {
            printf("ok - optional driver dlopen succeeded: %s\n", argv[i]);
            dlclose(drv);
        } else {
            printf("info - optional driver not present on host: %s\n", argv[i]);
        }
    }
    return 0;
}
EOF
    "$CC_BIN" -shared -fPIC -O2 "$WORK/libprobe.c" -o "$WORK/libprobe.so"
    "$CC_BIN" -O2 "$WORK/main.c" -o "$WORK/probe" -ldl
    "$WORK/probe" "$WORK/libprobe.so" libcuda.so.1 libvulkan.so.1
    echo "ok - runtime dlopen/dlsym probe passed"
fi
