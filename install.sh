#!/bin/sh
# fLLM installer: download the release binary for this machine, verify it and install it to
# ./.local/bin/fllm-rs (the path run.sh expects).
#
#   ./install.sh [VERSION] [OPTIONS]
#
#   VERSION      release tag to install; "latest" (the default) resolves the newest release
#                through the release API of this repository's origin remote (GitHub, Forgejo or Gitea).
#
#   --no-verify  install even when the release publishes no <asset>.sha256 file. A checksum
#                file that does exist is always verified; a mismatch always aborts.
#   --build      skip the download and build this checkout with cargo instead.
#   -h, --help   show this text.
#
# The download URL is derived from `git remote -v` of this checkout, e.g.
#   https://github.com/Marco-Lu2508/fLLM/releases/download/<tag>/<asset>
# and falls back to that URL when git or the remote is missing. Assets (built by the
# release workflow of .github/workflows/release.yml or .github/workflows/release.yml):
#   fllm-linux-x86_64-musl.tar.gz, fllm-linux-aarch64-musl.tar.gz,
#   fllm-macos-aarch64.tar.gz, fllm-macos-x86_64.tar.gz, each with <asset>.sha256
#
# "latest" on github.com: follows the /releases/latest redirect (no API call, no rate limit);
# while the repository has only pre-releases (v0.x) that redirect has no target, so the newest
# release is read from the API (api.github.com, 60 requests per hour without a token).
# Other hosts use the Forgejo/Gitea API (/api/v1). Environment overrides:
#   FLLM_REPO_URL  repository URL instead of the git remote (e.g. a mirror)
#   FLLM_FORGE     github | forgejo, when the host name does not tell
#   FLLM_API_URL   base URL of the GitHub API (default https://api.github.com)
#   FLLM_INSTALL_DIR, RUST_FEATURES, as before
#
# Requirements: curl or wget to download, sha256sum/shasum/openssl for the checksum check,
# tar to unpack. No Python is used, and no checksum value is baked into this script - the
# expected hash always comes from the release's <asset>.sha256 file.
# The fallback build (used when no matching asset exists) compiles the checkout next to this
# script and needs: cargo (Rust toolchain), a C compiler, CMake and Git.
set -eu
# `set -o pipefail` is not POSIX; enable it when the shell provides it (bash, zsh).
( set -o pipefail 2>/dev/null ) && set -o pipefail

DEFAULT_REPO="https://github.com/Marco-Lu2508/fLLM"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INSTALL_DIR=${FLLM_INSTALL_DIR:-"$SCRIPT_DIR/.local/bin"}
VERSION="latest"
VERIFY="1"
BUILD_ONLY="0"

usage() { sed -n '2,/^set -eu$/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

die() { echo "install.sh: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

for arg in "$@"; do
    case "$arg" in
        --no-verify) VERIFY="0" ;;
        --build) BUILD_ONLY="1" ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "install.sh: unknown option $arg (see --help)" >&2; exit 2 ;;
        *) VERSION="$arg" ;;
    esac
done

fetch() { # URL -> stdout
    if have curl; then curl -fsSL "$1"
    elif have wget; then wget -qO- "$1"
    else die "curl or wget is required to download fLLM"
    fi
}

download() { # URL FILE
    if have curl; then curl -fsSL -o "$2" "$1"
    elif have wget; then wget -q -O "$2" "$1"
    else die "curl or wget is required to download fLLM"
    fi
}

sha256_of() { # FILE -> stdout
    if have sha256sum; then out=$(sha256sum "$1"); out=${out%% *}
    elif have shasum; then out=$(shasum -a 256 "$1"); out=${out%% *}
    elif have openssl; then out=$(openssl dgst -sha256 "$1"); out=${out##* }
    else die "no SHA256 tool found (sha256sum, shasum or openssl); re-run with --no-verify"
    fi
    echo "$out"
}

# ------------------------------------------------------------------ platform
case "$(uname -s)" in
    Linux)  OS="linux" ;;
    Darwin) OS="macos" ;;
    CYGWIN*|MINGW*|MSYS*)
        die "Windows detected: use install.ps1, or run this script inside WSL2" ;;
    *) die "unsupported operating system: $(uname -s)" ;;
esac
case "$(uname -m)" in
    x86_64|amd64)  ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *) die "unsupported architecture: $(uname -m)" ;;
esac
case "$OS" in
    linux) ASSET="fllm-linux-$ARCH-musl.tar.gz" ;;
    macos) ASSET="fllm-macos-$ARCH.tar.gz" ;;
esac

# --------------------------------------------------------------- repository URL
# `git remote -v` prints:  origin  https://github.com/Marco-Lu2508/fLLM.git  (fetch)
remote_url() {
    have git || return 1
    git -C "$SCRIPT_DIR" remote -v 2>/dev/null | {
        read -r _name url _rest || return 1
        [ -n "${url:-}" ] || return 1
        printf '%s\n' "$url"
    }
}
to_https() {
    u=${1%.git}
    case "$u" in
        https://*|http://*) printf '%s\n' "$u" ;;
        git@*:*)
            h=${u#git@}; h=${h%%:*}
            p=${u#*:}
            printf 'https://%s/%s\n' "$h" "$p" ;;
        *) return 1 ;;
    esac
}
if [ -n "${FLLM_REPO_URL:-}" ]; then
    REPO_URL=$FLLM_REPO_URL
else
    REPO_URL=$(remote_url | to_https) || REPO_URL="$DEFAULT_REPO"
fi
REPO_URL=${REPO_URL%/}

SCHEME=${REPO_URL%%://*}
rest=${REPO_URL#*://}                 # host/owner/name
HOST=${rest%%/*}
path=${rest#*/}                       # owner/name
name=${path##*/}
head=${path%/*}
[ "$head" != "$path" ] || die "cannot derive owner/repository from $REPO_URL"
SLUG="$head/$name"
BASE="$REPO_URL/releases"

case "${FLLM_FORGE:-}" in
    github|forgejo) FORGE=$FLLM_FORGE ;;
    "") case "$HOST" in
            github.com|www.github.com) FORGE="github" ;;
            *) FORGE="forgejo" ;;
        esac ;;
    *) die "FLLM_FORGE must be 'github' or 'forgejo', not '${FLLM_FORGE}'" ;;
esac

tag_from_json() { # JSON on stdin -> first "tag_name" value
    grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n 1 |
        sed 's/.*"\([^"]*\)"[[:space:]]*$/\1/'
}

# --------------------------------------------------------------- resolve the tag
if [ "$BUILD_ONLY" = "0" ] && [ "$VERSION" = "latest" ]; then
    if [ "$FORGE" = "github" ]; then
        # /releases/latest redirects to /releases/tag/<tag>; it ignores pre-releases (v0.x), then there is no target
        VERSION=""
        if have curl; then
            final=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$REPO_URL/releases/latest" 2>/dev/null) || final=""
            case "$final" in */releases/tag/*) VERSION=${final##*/releases/tag/} ;; esac
        fi
        if [ -z "$VERSION" ]; then
            API="${FLLM_API_URL:-https://api.github.com}/repos/$SLUG/releases?per_page=1"
            json=$(fetch "$API") ||
                die "cannot query $API - is $REPO_URL reachable? (a tag can be given as an argument, --build compiles instead)"
            VERSION=$(printf '%s' "$json" | tag_from_json) || VERSION=""
            [ -n "$VERSION" ] || die "no release found via $API"
        fi
    else
        API="$SCHEME://$HOST/api/v1/repos/$SLUG"
        json=$(fetch "$API/releases/latest") ||
            die "cannot query $API/releases/latest - is $REPO_URL reachable?"
        VERSION=$(printf '%s' "$json" | tag_from_json) || VERSION=""
        [ -n "$VERSION" ] || die "no release found via $API/releases/latest"
    fi
    echo "Newest release: $VERSION"
fi

# --------------------------------------------------------------- build from source
build_from_source() {
    echo "Building fLLM from source in $SCRIPT_DIR" >&2
    echo "(needs cargo, a C compiler, CMake and Git; RUST_FEATURES selects the cargo features)" >&2
    have cargo || die "cargo not found - install a Rust toolchain (https://rustup.rs), or fix the release download reported above"
    [ -f "$SCRIPT_DIR/Cargo.toml" ] ||
        die "$SCRIPT_DIR has no Cargo.toml; the source build needs a checkout of the repository"
    (cd "$SCRIPT_DIR" &&
        cargo build --release --locked -p fllm-engine \
            --features "${RUST_FEATURES:-rust-gpu}" --bin fllm-rs)
    target_dir=${CARGO_TARGET_DIR:-"$SCRIPT_DIR/target"}
    [ -x "$target_dir/release/fllm-rs" ] ||
        die "build finished but $target_dir/release/fllm-rs is missing"
    mkdir -p "$INSTALL_DIR"
    cp "$target_dir/release/fllm-rs" "$INSTALL_DIR/fllm-rs"
    chmod 0755 "$INSTALL_DIR/fllm-rs"
    echo "Installed: $INSTALL_DIR/fllm-rs"
}

if [ "$BUILD_ONLY" = "1" ]; then
    build_from_source
    exit 0
fi

# --------------------------------------------------------------- download + verify
URL="$BASE/download/$VERSION/$ASSET"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Fetching $URL"
if ! download "$URL" "$TMP/$ASSET"; then
    echo "install.sh: no release asset '$ASSET' for tag '$VERSION' (release or asset missing)" >&2
    build_from_source
    exit 0
fi

if download "$URL.sha256" "$TMP/$ASSET.sha256" 2>/dev/null; then
    line=$(cat "$TMP/$ASSET.sha256")
    want=${line%% *}
    [ -n "$want" ] || die "$ASSET.sha256 is empty; refusing to install"
    got=$(sha256_of "$TMP/$ASSET")
    if [ "$want" != "$got" ]; then
        echo "install.sh: CHECKSUM MISMATCH for $ASSET" >&2
        echo "  expected: $want" >&2
        echo "  actual:   $got" >&2
        echo "The download was NOT installed. The asset and its .sha256 file do not belong" >&2
        echo "together; do not use this file." >&2
        exit 1
    fi
    echo "SHA256 OK: $got"
elif [ "$VERIFY" = "1" ]; then
    die "the release for '$VERSION' publishes no $ASSET.sha256, so the download cannot be verified. Re-run with --no-verify to install it anyway"
else
    echo "install.sh: WARNING - --no-verify: installing WITHOUT checksum verification" >&2
fi

tar -xzf "$TMP/$ASSET" -C "$TMP"
[ -f "$TMP/fllm-rs" ] || die "$ASSET does not contain the fllm-rs binary"
mkdir -p "$INSTALL_DIR"
cp "$TMP/fllm-rs" "$INSTALL_DIR/fllm-rs"
chmod 0755 "$INSTALL_DIR/fllm-rs"
echo "Installed $VERSION: $INSTALL_DIR/fllm-rs"
echo "Try: ./run.sh formats"
