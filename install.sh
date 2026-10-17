#!/bin/sh
# fLLM installer: downloads the release binary for this machine, verifies its checksum and installs it.
# Works from a clone and without one:
#
#   curl -fsSL https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.sh | sh -s -- --modify-path --version v0.1.2
#   ./install.sh [OPTIONS]
#
# The binary goes to ${FLLM_INSTALL_DIR:-$HOME/.local/bin}/fllm-rs (the directory is created).
# Options:
#   --version TAG    release tag to install, e.g. v0.1.2 (default: the newest release). A bare TAG argument works too.
#   --prefix DIR     install to DIR/bin instead (FLLM_INSTALL_DIR names the exact directory and wins over the default).
#   --variant V      cpu (default) or cuda. Linux CUDA needs the release asset fllm-linux-x86_64-cuda.tar.gz; when the
#                    release has none the script says so and points to the Docker image. Windows: use install.ps1.
#   --modify-path    append a guarded PATH line to the rc file of your shell (bash, zsh, fish, otherwise ~/.profile).
#                    Without it the script only prints the line to add. Running it again does not add a second line.
#   --uninstall      remove the binary and the PATH line this script added.
#   --build          do not download; build the checkout this script lives in with cargo (needs cargo, a C compiler,
#                    CMake and Git; RUST_FEATURES selects the cargo features, default rust-gpu).
#   --no-verify      install even when the release publishes no checksum. A checksum that does exist is always
#                    verified; a mismatch always aborts.
#   -h, --help       show this text.
#
# What the checksum check protects against: a damaged or truncated download, a CDN or proxy that serves the wrong file,
# and an asset that does not belong to its .sha256 file. The expected hash comes from the release's <asset>.sha256 file and,
# when the release notes carry a "<sha256>  <asset>" list, from there too; both must agree. It is NOT a signature: whoever
# can replace the asset in the release can replace the .sha256 file and the notes as well, and the binaries are unsigned.
# HTTPS to github.com is what authenticates the source.
#
# Assets (built by .github/workflows/release.yml), each with <asset>.sha256:
#   fllm-linux-x86_64-musl.tar.gz, fllm-linux-aarch64-musl.tar.gz, fllm-macos-aarch64.tar.gz, fllm-macos-x86_64.tar.gz
#   (optional, not published yet: fllm-linux-x86_64-cuda.tar.gz)
#
# Environment overrides:
#   FLLM_INSTALL_DIR   install directory (exact)
#   FLLM_REPO_URL      repository URL instead of the default (a mirror, a fork, or a test server)
#   FLLM_FORGE         github | gitea, when the host name does not tell
#   FLLM_API_URL       base URL of the GitHub API (default https://api.github.com)
#   RUST_FEATURES      cargo features of --build
# "latest" on github.com follows the /releases/latest redirect; while the repository has only pre-releases (v0.x) that
# redirect has no target and the newest release is read from the API (60 requests per hour without a token).
#
# Requirements: curl or wget, tar, and sha256sum, shasum or openssl. POSIX sh; no Python.
set -eu
# `set -o pipefail` is not POSIX; enable it when the shell provides it (bash, zsh).
# shellcheck disable=SC3040
( set -o pipefail 2>/dev/null ) && set -o pipefail

DEFAULT_REPO="https://github.com/Marco-Lu2508/fLLM"
MARK_BEGIN="# >>> fllm >>>"
MARK_END="# <<< fllm <<<"

VERSION="latest"
VERIFY="1"
BUILD_ONLY="0"
UNINSTALL="0"
MODIFY_PATH="0"
VARIANT="cpu"
PREFIX=""

usage() {
    cat <<'EOF'
usage: install.sh [--version TAG] [--prefix DIR] [--variant cpu|cuda] [--modify-path] [--uninstall] [--build] [--no-verify]

  curl -fsSL https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.sh | sh
  curl -fsSL https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.sh | sh -s -- --modify-path

Installs fllm-rs to ${FLLM_INSTALL_DIR:-$HOME/.local/bin} (--prefix DIR: DIR/bin).
  --version TAG    release tag (default: newest release)
  --variant V      cpu (default) or cuda (Linux x86_64 asset; Docker alternative is printed when the release has none)
  --modify-path    add the install directory to PATH in your shell rc file (guarded, idempotent)
  --uninstall      remove the binary and the PATH line added by --modify-path
  --build          build this checkout with cargo instead of downloading
  --no-verify      allow a release without a checksum (a checksum that exists is always checked)
The first lines of the script describe the checksum check and the environment overrides.
EOF
}

die() { echo "install.sh: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --no-verify) VERIFY="0" ;;
        --build) BUILD_ONLY="1" ;;
        --uninstall) UNINSTALL="1" ;;
        --modify-path) MODIFY_PATH="1" ;;
        --version) [ $# -ge 2 ] || die "--version needs a tag"; VERSION=$2; shift ;;
        --version=*) VERSION=${1#--version=} ;;
        --prefix) [ $# -ge 2 ] || die "--prefix needs a directory"; PREFIX=$2; shift ;;
        --prefix=*) PREFIX=${1#--prefix=} ;;
        --variant) [ $# -ge 2 ] || die "--variant needs cpu or cuda"; VARIANT=$2; shift ;;
        --variant=*) VARIANT=${1#--variant=} ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "install.sh: unknown option $1 (see --help)" >&2; exit 2 ;;
        *) VERSION=$1 ;;
    esac
    shift
done
case "$VARIANT" in cpu|cuda) ;; *) die "--variant must be cpu or cuda, not '$VARIANT'" ;; esac

[ -n "${HOME:-}" ] || die "HOME is not set"
if [ -n "$PREFIX" ]; then
    INSTALL_DIR="${PREFIX%/}/bin"
elif [ -n "${FLLM_INSTALL_DIR:-}" ]; then
    INSTALL_DIR=${FLLM_INSTALL_DIR%/}
else
    INSTALL_DIR="$HOME/.local/bin"
fi
[ -n "$INSTALL_DIR" ] || INSTALL_DIR="/"
case "$INSTALL_DIR" in /*) ;; *) INSTALL_DIR="$(pwd)/$INSTALL_DIR" ;; esac
case "$VERSION" in [0-9]*) VERSION="v$VERSION" ;; esac

# The directory of this script when it is a real file (a clone); empty when it comes through a pipe.
SCRIPT_DIR=""
case "$0" in
    */*|*.sh) if [ -f "$0" ]; then SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd); fi ;;
esac

# ------------------------------------------------------------------ PATH handling
on_path() { case ":${PATH:-}:" in *":$1:"*) return 0 ;; *) return 1 ;; esac; }

# The rc file for the user's login shell and the line it needs.
shell_kind() {
    case "${SHELL:-}" in
        */fish) echo fish ;;
        */zsh) echo zsh ;;
        */bash) echo bash ;;
        *) echo sh ;;
    esac
}
rc_file_for() { # KIND -> path
    case "$1" in
        fish) echo "${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish" ;;
        zsh) echo "${ZDOTDIR:-$HOME}/.zshrc" ;;
        bash) if [ "$(uname -s)" = "Darwin" ] && [ ! -f "$HOME/.bashrc" ]; then echo "$HOME/.bash_profile"; else echo "$HOME/.bashrc"; fi ;;
        *) echo "$HOME/.profile" ;;
    esac
}
# How the directory is written in the rc file: under $HOME it stays portable.
dir_for_rc() {
    # shellcheck disable=SC2016
    case "$INSTALL_DIR" in
        "$HOME"/*) printf '$HOME/%s\n' "${INSTALL_DIR#"$HOME"/}" ;;
        *) printf '%s\n' "$INSTALL_DIR" ;;
    esac
}
path_line() { # KIND -> the line(s) between the markers
    d=$(dir_for_rc)
    # shellcheck disable=SC2016
    case "$1" in
        fish) printf 'contains -- "%s" $PATH; or set -gx PATH "%s" $PATH\n' "$d" "$d" ;;
        *) printf 'case ":$PATH:" in *":%s:"*) ;; *) export PATH="%s:$PATH" ;; esac\n' "$d" "$d" ;;
    esac
}
unsafe_dir() {
    bs=$(printf '\134')
    case "$INSTALL_DIR" in
        *'"'*|*'`'*|*'$'*) return 0 ;;
        *"$bs"*) return 0 ;;
    esac
    return 1
}

report_path() {
    if on_path "$INSTALL_DIR"; then
        echo "$INSTALL_DIR is already on your PATH."
        return 0
    fi
    kind=$(shell_kind)
    rc=$(rc_file_for "$kind")
    if [ "$MODIFY_PATH" = "1" ]; then
        unsafe_dir && die "the install directory contains characters that cannot be written safely to a shell rc file; add it to PATH by hand"
        if [ -f "$rc" ] && grep -qF "$MARK_BEGIN" "$rc"; then
            echo "PATH line already present in $rc (left unchanged)."
        else
            mkdir -p "$(dirname -- "$rc")"
            lead=""
            if [ -s "$rc" ]; then lead="yes"; fi
            {
                if [ -n "$lead" ]; then echo ""; fi
                echo "$MARK_BEGIN"
                echo "# added by the fLLM install.sh; 'install.sh --uninstall' removes this block"
                path_line "$kind"
                echo "$MARK_END"
            } >> "$rc"
            echo "Added $INSTALL_DIR to PATH in $rc. Open a new terminal, or run:  . \"$rc\""
        fi
    else
        echo ""
        echo "$INSTALL_DIR is not on your PATH. To call 'fllm-rs' from anywhere, add this line to $rc:"
        echo "  $(path_line "$kind")"
        echo "or run this script again with --modify-path to let it do that (guarded, added once)."
    fi
}

remove_path_lines() {
    for rc in "$HOME/.bashrc" "$HOME/.bash_profile" "${ZDOTDIR:-$HOME}/.zshrc" "$HOME/.profile" "${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish"; do
        [ -f "$rc" ] || continue
        grep -qF "$MARK_BEGIN" "$rc" || continue
        tmp="$rc.fllm-tmp.$$"
        sed "/^# >>> fllm >>>\$/,/^# <<< fllm <<<\$/d" "$rc" > "$tmp" && cat "$tmp" > "$rc"
        rm -f "$tmp"
        echo "Removed the fLLM PATH block from $rc"
    done
}

if [ "$UNINSTALL" = "1" ]; then
    if [ -e "$INSTALL_DIR/fllm-rs" ]; then
        rm -f "$INSTALL_DIR/fllm-rs"
        echo "Removed $INSTALL_DIR/fllm-rs"
    else
        echo "No $INSTALL_DIR/fllm-rs to remove."
    fi
    rmdir "$INSTALL_DIR" 2>/dev/null || true
    remove_path_lines
    echo "Models and caches are not touched (see docs/getting-started.md for their location)."
    exit 0
fi

# ------------------------------------------------------------------ download helpers
# download URL FILE: 0 ok, 44 the server answered "not found", 1 anything else
download() {
    if have curl; then
        code=$(curl -sSL -o "$2" -w '%{http_code}' "$1") || return 1
        case "$code" in 2*) return 0 ;; 404|410) return 44 ;; *) return 1 ;; esac
    elif have wget; then
        rc=0
        err=$(wget -q -O "$2" "$1" 2>&1) || rc=$?
        [ "$rc" = "0" ] && return 0
        # GNU wget exits with 8 for any error response; busybox wget exits with 1 and prints the status line
        case "$err" in *404*|*"Not Found"*|*410*) return 44 ;; esac
        [ "$rc" = "8" ] && [ -z "$err" ] && return 44
        return 1
    else
        die "curl or wget is required to download fLLM"
    fi
}
fetch() { # URL -> stdout
    if have curl; then curl -fsSL "$1"
    elif have wget; then wget -qO- "$1"
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

install_binary() { # SOURCE
    mkdir -p "$INSTALL_DIR"
    new="$INSTALL_DIR/.fllm-rs.new.$$"
    cp "$1" "$new"
    chmod 0755 "$new"
    mv -f "$new" "$INSTALL_DIR/fllm-rs"
    if [ "$(uname -s)" = "Darwin" ] && have xattr; then
        # Files fetched by a browser carry com.apple.quarantine and Gatekeeper then refuses an unsigned binary
        # ("cannot be opened because the developer cannot be verified"). curl does not set it; this is for a binary
        # copied here from a browser download. The binary is not signed or notarized.
        if xattr "$INSTALL_DIR/fllm-rs" 2>/dev/null | grep -q com.apple.quarantine; then
            xattr -d com.apple.quarantine "$INSTALL_DIR/fllm-rs" 2>/dev/null || true
            echo "Cleared the macOS quarantine flag (the binary is unsigned; Gatekeeper would otherwise block it)."
        fi
    fi
}

# ------------------------------------------------------------------ build from source
build_from_source() {
    top=""
    if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/Cargo.toml" ]; then top=$SCRIPT_DIR
    elif [ -f "./Cargo.toml" ] && [ -d "./rust/engine" ]; then top=$(pwd)
    fi
    [ -n "$top" ] || die "--build needs a checkout of the repository: git clone $DEFAULT_REPO && cd fLLM && ./install.sh --build"
    echo "Building fLLM from source in $top" >&2
    echo "(needs cargo, a C compiler, CMake and Git; RUST_FEATURES selects the cargo features)" >&2
    have cargo || die "cargo not found - install a Rust toolchain (https://rustup.rs)"
    (cd "$top" &&
        cargo build --release --locked -p fllm-engine \
            --features "${RUST_FEATURES:-rust-gpu}" --bin fllm-rs)
    target_dir=${CARGO_TARGET_DIR:-"$top/target"}
    [ -x "$target_dir/release/fllm-rs" ] ||
        die "build finished but $target_dir/release/fllm-rs is missing"
    install_binary "$target_dir/release/fllm-rs"
    echo "Installed: $INSTALL_DIR/fllm-rs"
}

finish() {
    "$INSTALL_DIR/fllm-rs" --version 2>/dev/null || echo "install.sh: WARNING - the installed binary did not report its version (does it run on this system?)" >&2
    report_path
    echo ""
    echo "Next: fllm-rs devices   (what fLLM sees on this machine)   |   fllm-rs run <huggingface link>"
}

if [ "$BUILD_ONLY" = "1" ]; then
    build_from_source
    finish
    exit 0
fi

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
docker_hint() {
    echo "Docker alternative with CUDA (no install needed, needs the NVIDIA driver and the NVIDIA Container Toolkit):" >&2
    echo "  docker run --rm --gpus all ghcr.io/marco-lu2508/fllm:cuda devices" >&2
    echo "  docker run -it --rm --gpus all -v fllm:/home/fllm/.cache ghcr.io/marco-lu2508/fllm:cuda run unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda" >&2
    echo "  (compose.cuda.yml and docs/docker.md describe the server setup)" >&2
}
if [ "$VARIANT" = "cuda" ]; then
    [ "$OS" = "linux" ] && [ "$ARCH" = "x86_64" ] || die "the cuda variant exists for Linux x86_64 only (Windows: install.ps1 -Variant cuda)"
    ASSET="fllm-linux-x86_64-cuda.tar.gz"
elif [ "$VARIANT" = "gnu" ]; then
    [ "$OS" = "linux" ] || die "the gnu variant exists for Linux only"
    ASSET="fllm-linux-$ARCH-gnu.tar.gz"
elif [ "$OS" = "linux" ] && have nvidia-smi; then
    echo "An NVIDIA GPU driver was found. This default build is the static CPU build; for CUDA use --variant cuda or Docker (docs/docker.md)."
fi

# --------------------------------------------------------------- repository URL
remote_url() { # prints the first remote URL of the checkout this script lives in
    if [ -z "$SCRIPT_DIR" ] || ! have git; then return 1; fi
    git -C "$SCRIPT_DIR" remote -v 2>/dev/null | {
        read -r _name url _rest || return 1
        [ -n "${url:-}" ] || return 1
        printf '%s\n' "$url"
    }
}
to_https() { # URL on stdin -> https URL on stdout
    read -r u || return 1
    u=${u%.git}
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
    REPO_URL=$(remote_url 2>/dev/null | to_https 2>/dev/null) || REPO_URL=""
    [ -n "$REPO_URL" ] || REPO_URL="$DEFAULT_REPO"
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
    github|gitea) FORGE=$FLLM_FORGE ;;
    "") case "$HOST" in
            github.com|www.github.com) FORGE="github" ;;
            *) FORGE="gitea" ;;
        esac ;;
    *) die "FLLM_FORGE must be 'github' or 'gitea', not '${FLLM_FORGE}'" ;;
esac
if [ "$FORGE" = "github" ]; then
    API_REPO="${FLLM_API_URL:-https://api.github.com}/repos/$SLUG"
else
    API_REPO="$SCHEME://$HOST/api/v1/repos/$SLUG"
fi

tag_from_json() { # JSON on stdin -> first "tag_name" value
    grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n 1 |
        sed 's/.*"\([^"]*\)"[[:space:]]*$/\1/'
}

# --------------------------------------------------------------- resolve the tag
if [ "$VERSION" = "latest" ]; then
    if [ "$FORGE" = "github" ]; then
        # /releases/latest redirects to /releases/tag/<tag>; it ignores pre-releases (v0.x), then there is no target
        VERSION=""
        if have curl; then
            final=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$REPO_URL/releases/latest" 2>/dev/null) || final=""
            case "$final" in */releases/tag/*) VERSION=${final##*/releases/tag/} ;; esac
        fi
        if [ -z "$VERSION" ]; then
            json=$(fetch "$API_REPO/releases?per_page=1") ||
                die "cannot query $API_REPO/releases - is $REPO_URL reachable? (a tag can be given with --version)"
            VERSION=$(printf '%s' "$json" | tag_from_json) || VERSION=""
            [ -n "$VERSION" ] || die "no release found via $API_REPO/releases"
        fi
    else
        json=$(fetch "$API_REPO/releases/latest") ||
            die "cannot query $API_REPO/releases/latest - is $REPO_URL reachable?"
        VERSION=$(printf '%s' "$json" | tag_from_json) || VERSION=""
        [ -n "$VERSION" ] || die "no release found via $API_REPO/releases/latest"
    fi
    echo "Newest release: $VERSION"
fi

# --------------------------------------------------------------- download + verify
URL="$BASE/download/$VERSION/$ASSET"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Fetching $URL"
status=0
download "$URL" "$TMP/$ASSET" || status=$?
if [ "$status" = "44" ]; then
    echo "install.sh: the release '$VERSION' has no asset '$ASSET'." >&2
    if [ "$VARIANT" = "cuda" ]; then
        echo "A Linux CUDA binary is not published for every release." >&2
        docker_hint
    elif [ -n "$SCRIPT_DIR" ]; then
        echo "Building from this checkout instead." >&2
        build_from_source
        finish
        exit 0
    else
        echo "Check the tag (--version) and the releases page: $REPO_URL/releases" >&2
        echo "To build from source: git clone $DEFAULT_REPO && cd fLLM && ./install.sh --build" >&2
    fi
    exit 1
elif [ "$status" != "0" ]; then
    die "download of $URL failed (network, proxy or TLS problem)"
fi

# expected hash 1: the <asset>.sha256 file of the release
want_file=""
status=0
download "$URL.sha256" "$TMP/$ASSET.sha256" 2>/dev/null || status=$?
if [ "$status" = "0" ]; then
    line=$(cat "$TMP/$ASSET.sha256")
    want_file=${line%%[[:space:]]*}
    [ -n "$want_file" ] || die "$ASSET.sha256 is empty; refusing to install"
fi
# expected hash 2: a "<sha256>  <asset>" list in the release notes, when there is one (a second place to compare with)
want_notes=""
if json=$(fetch "$API_REPO/releases/tags/$VERSION" 2>/dev/null); then
    want_notes=$(printf '%s' "$json" | grep -o "[0-9a-fA-F]\{64\}[^0-9a-fA-F]\{1,16\}$ASSET" | head -n 1 | cut -c1-64) || want_notes=""
fi
got=$(sha256_of "$TMP/$ASSET")
want_file=$(printf '%s' "$want_file" | tr 'A-F' 'a-f')
want_notes=$(printf '%s' "$want_notes" | tr 'A-F' 'a-f')
if [ -n "$want_file" ] && [ -n "$want_notes" ] && [ "$want_file" != "$want_notes" ]; then
    die "the .sha256 file ($want_file) and the release notes ($want_notes) disagree for $ASSET; refusing to install"
fi
want=${want_file:-$want_notes}
if [ -n "$want" ]; then
    if [ "$want" != "$got" ]; then
        echo "install.sh: CHECKSUM MISMATCH for $ASSET" >&2
        echo "  expected: $want" >&2
        echo "  actual:   $got" >&2
        echo "The download was NOT installed. The asset and its checksum do not belong together;" >&2
        echo "do not use this file." >&2
        exit 1
    fi
    if [ -n "$want_file" ] && [ -n "$want_notes" ]; then
        echo "SHA256 OK: $got (matches the .sha256 file and the release notes)"
    elif [ -n "$want_file" ]; then
        echo "SHA256 OK: $got (matches the release's .sha256 file; the release notes carry no checksum list)"
    else
        echo "SHA256 OK: $got (matches the release notes; the release has no .sha256 file)"
    fi
elif [ "$VERIFY" = "1" ]; then
    die "the release for '$VERSION' publishes no checksum for $ASSET, so the download cannot be verified. Re-run with --no-verify to install it anyway"
else
    echo "install.sh: WARNING - --no-verify: installing WITHOUT checksum verification (sha256 of the file: $got)" >&2
fi

tar -xzf "$TMP/$ASSET" -C "$TMP"
[ -f "$TMP/fllm-rs" ] || die "$ASSET does not contain the fllm-rs binary"
install_binary "$TMP/fllm-rs"
echo "Installed $VERSION: $INSTALL_DIR/fllm-rs"
finish
