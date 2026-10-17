#!/usr/bin/env bash
# Smoke test of the "no hardware GPU adapter" behaviour, for runners that only have software renderers (GitHub's windows-2022 has WARP, and
# SwiftShader when the Vulkan runtime is installed; the Linux runners have nothing):
#   - `devices` lists no GPU adapter,
#   - `--device dx12` / `--device vulkan` are refused with a message that names the cause and the way out
#     (FLLM_ALLOW_SOFTWARE_GPU / --device cpu), not a crash and not a silent software run,
#   - `--device auto` has nothing to pick (the decoder then runs on the CPU; that part is the unit test
#     `device_fallback_tests::auto_falls_back_to_the_cpu_without_a_hardware_gpu`, which runs with the other unit tests).
# On a machine that does have a hardware adapter of an API the checks of that API are skipped. Exit status 0 = all checks passed.
#
#   scripts/no-gpu-smoke.sh path/to/fllm-rs[.exe]
set -u
bin=${1:?usage: no-gpu-smoke.sh path/to/fllm-rs}
fail=0
note() { printf '%s\n' "$*"; }

devices=$("$bin" devices --json 2>/dev/null || true)
note "devices --json: ${devices:0:400}"

# a hardware adapter of this API in the inventory (the inventory lists hardware only)
has_hardware() {
  printf '%s' "$devices" | tr -d ' \n\r' | grep -Eqi "\"(backend|api)\":\"$1\""
}

expect_refusal() { # <device> <pattern>
  local device=$1 pattern=$2 out status
  if has_hardware "$device"; then
    note "skip: a hardware $device adapter exists on this machine"
    return
  fi
  out=$("$bin" selftest-gpu --device "$device" --only norm 2>&1)
  status=$?
  note "--- selftest-gpu --device $device (exit $status)"
  note "$out" | tail -5
  if [ "$status" -eq 0 ]; then
    note "FAIL: --device $device succeeded without a hardware adapter"
    fail=1
  elif ! printf '%s' "$out" | grep -Eqi "$pattern"; then
    note "FAIL: the message for --device $device does not match /$pattern/"
    fail=1
  elif printf '%s' "$out" | grep -Eqi "panicked|stack backtrace|RUST_BACKTRACE"; then
    note "FAIL: --device $device panicked"
    fail=1
  fi
}

expect_refusal dx12 "no hardware dx12|no dx12 GPU adapter|not (available|compiled)|no .*device"
expect_refusal vulkan "no hardware|no vulkan|FLLM_ALLOW_SOFTWARE_GPU|not (available|usable)|no .*device"

# the CPU path must keep working
if ! "$bin" selftest-gpu --device cpu --only norm >/dev/null 2>&1; then
  note "FAIL: selftest-gpu --device cpu failed"
  fail=1
fi
[ "$fail" -eq 0 ] && note "no-gpu smoke test: ok"
exit "$fail"
