#!/usr/bin/env bash
# Runs a command and, when it fails, repeats the failing tests and the first error lines as a workflow annotation, so that a
# failure can be read from the check-run annotations without opening the job log.
#   scripts/ci/annotate-failure.sh cargo test --release --locked ...
set -uo pipefail
log=$(mktemp)
"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    summary=$(grep -E '^test .* FAILED$|panicked at|^error(\[|:)|^ *-->|assertion|^ *(left|right):|FAIL|denied|vulnerab|license|unmaintained|RUSTSEC' "$log" \
        | head -60 | cut -c1-260 | awk '{ gsub(/%/, "%25"); gsub(/\r/, ""); printf "%s%%0A", $0 }')
    echo "::error title=Failure details::${summary}"
fi
exit "$status"
