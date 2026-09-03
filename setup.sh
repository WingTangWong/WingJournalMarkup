#!/usr/bin/env bash
#
# Wing Journal Markup (WJM) - environment setup.
#
# Fresh checkout:  creates a virtualenv, installs the system Tesseract binary,
#                  installs the package + Python dependencies, and verifies the
#                  result.
# Existing state:  reuses the virtualenv, optionally fast-forwards the repo,
#                  re-syncs dependencies, and re-verifies.
#
# Usage:
#     ./setup.sh                 # full dev setup (runtime + OCR + dev + demo)
#     ./setup.sh --runtime       # runtime + OCR only, no dev/test tooling
#     ./setup.sh --no-ocr        # skip pytesseract + system Tesseract
#     ./setup.sh --no-pull       # don't git pull even if a tree already exists
#     ./setup.sh --check         # also run `pytest -q` at the end
#     ./setup.sh --python python3.12   # force a specific interpreter
#     ./setup.sh --help
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# config / arg parsing
# --------------------------------------------------------------------------- #
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${WJM_VENV:-$REPO_ROOT/.venv}"

PROFILE="dev"          # dev | runtime
WITH_OCR=1
DO_PULL=1
RUN_CHECK=0
FORCE_PYTHON=""

while [ $# -gt 0 ]; do
    case "$1" in
        --runtime)        PROFILE="runtime" ;;
        --dev)            PROFILE="dev" ;;
        --no-ocr)         WITH_OCR=0 ;;
        --no-pull)        DO_PULL=0 ;;
        --check)          RUN_CHECK=1 ;;
        --python)         FORCE_PYTHON="${2:-}"; shift ;;
        --python=*)       FORCE_PYTHON="${1#*=}" ;;
        -h|--help)
            cat <<'EOF'
Wing Journal Markup (WJM) - environment setup.

Fresh checkout:  creates a virtualenv, installs the system Tesseract binary,
                 installs the package + Python dependencies, and verifies it.
Existing state:  reuses the virtualenv, optionally fast-forwards the repo,
                 re-syncs dependencies, and re-verifies.

Usage:
    ./setup.sh                       full dev setup (runtime + OCR + dev + demo)
    ./setup.sh --runtime             runtime + OCR only, no dev/test tooling
    ./setup.sh --no-ocr              skip pytesseract + system Tesseract
    ./setup.sh --no-pull             don't git pull even if a tree already exists
    ./setup.sh --check               also run `pytest -q` at the end
    ./setup.sh --python python3.12   force a specific interpreter
    ./setup.sh --help

Env:
    WJM_VENV   virtualenv location (default: ./.venv)
EOF
            exit 0
            ;;
        *)
            echo "setup.sh: unknown option '$1' (try --help)" >&2
            exit 2
            ;;
    esac
    shift
done

# --------------------------------------------------------------------------- #
# pretty output
# --------------------------------------------------------------------------- #
if [ -t 1 ]; then
    B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; X=$'\033[0m'
else
    B=""; G=""; Y=""; R=""; X=""
fi
step() { printf '\n%s==>%s %s%s\n' "$B$G" "$X$B" "$1" "$X"; }
info() { printf '    %s\n' "$1"; }
warn() { printf '%s!!!%s %s\n' "$Y" "$X" "$1" >&2; }
die()  { printf '%sXXX%s %s\n' "$R" "$X" "$1" >&2; exit 1; }

cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# 1. pick a Python interpreter (>= 3.10, prefer a version with wheels)
# --------------------------------------------------------------------------- #
step "Selecting a Python interpreter"

py_ok() {
    # usable if it exists and reports >= 3.10
    command -v "$1" >/dev/null 2>&1 || return 1
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null
}

PYTHON=""
if [ -n "$FORCE_PYTHON" ]; then
    py_ok "$FORCE_PYTHON" || die "$FORCE_PYTHON is missing or older than Python 3.10"
    PYTHON="$FORCE_PYTHON"
elif [ -x "$VENV_DIR/bin/python" ] && py_ok "$VENV_DIR/bin/python"; then
    # an existing venv already fixes the interpreter; keep it
    PYTHON="$VENV_DIR/bin/python"
    info "reusing interpreter from existing venv"
else
    # prefer 3.12 / 3.11 (broad wheel coverage for numpy + opencv), then others
    for cand in python3.12 python3.11 python3.10 python3.13 python3 python; do
        if py_ok "$cand"; then PYTHON="$cand"; break; fi
    done
    [ -n "$PYTHON" ] || die "no Python >= 3.10 found. Install one and re-run (or pass --python)."
fi

PYVER="$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
info "using $PYTHON  (Python $PYVER)"
case "$PYVER" in
    3.13.*|3.14.*|3.15.*)
        warn "Python $PYVER is very new; numpy/opencv/pillow wheels may be unavailable and pip could try to build from source. Python 3.11 or 3.12 is the safe choice - pass --python python3.12 if the install fails." ;;
esac

# --------------------------------------------------------------------------- #
# 2. system Tesseract binary (for OCR)
# --------------------------------------------------------------------------- #
if [ "$WITH_OCR" -eq 1 ]; then
    step "Ensuring the Tesseract OCR binary is installed"
    if command -v tesseract >/dev/null 2>&1; then
        info "found: $(tesseract --version 2>&1 | head -1)"
    else
        installed=0
        if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
            info "installing via Homebrew..."
            brew install tesseract && installed=1
        elif command -v apt-get >/dev/null 2>&1; then
            info "installing via apt-get (sudo)..."
            sudo apt-get update && sudo apt-get install -y tesseract-ocr && installed=1
        elif command -v dnf >/dev/null 2>&1; then
            info "installing via dnf (sudo)..."
            sudo dnf install -y tesseract && installed=1
        elif command -v pacman >/dev/null 2>&1; then
            info "installing via pacman (sudo)..."
            sudo pacman -S --noconfirm tesseract tesseract-data-eng && installed=1
        fi
        if [ "$installed" -eq 1 ] && command -v tesseract >/dev/null 2>&1; then
            info "installed: $(tesseract --version 2>&1 | head -1)"
        else
            warn "could not install Tesseract automatically. OCR will fall back to the null recognizer."
            warn "install it manually (brew install tesseract / apt-get install tesseract-ocr) and re-run, or pass --no-ocr."
        fi
    fi
else
    step "Skipping Tesseract / OCR (--no-ocr)"
fi

# --------------------------------------------------------------------------- #
# 3. virtualenv - create fresh, or reuse
# --------------------------------------------------------------------------- #
if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
    step "Reusing existing virtualenv"
    info "$VENV_DIR"
    FRESH=0
else
    step "Creating virtualenv"
    info "$VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
    FRESH=1
fi

# from here on, use the venv's tools directly (no `source` needed)
VPY="$VENV_DIR/bin/python"
VPIP="$VPY -m pip"

# --------------------------------------------------------------------------- #
# 4. pull updates if this is an existing checkout
# --------------------------------------------------------------------------- #
if [ "$DO_PULL" -eq 1 ] && [ "$FRESH" -eq 0 ] && git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    step "Pulling repository updates"
    if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
        warn "working tree has local changes - skipping 'git pull'. Commit or stash, then re-run."
    elif ! git -C "$REPO_ROOT" remote get-url origin >/dev/null 2>&1; then
        info "no 'origin' remote configured - nothing to pull."
    else
        branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
        info "git fetch + fast-forward ($branch)..."
        git -C "$REPO_ROOT" fetch --prune origin
        if git -C "$REPO_ROOT" merge-base --is-ancestor HEAD "origin/$branch" 2>/dev/null; then
            git -C "$REPO_ROOT" merge --ff-only "origin/$branch" || warn "fast-forward failed; resolve manually."
        else
            warn "local '$branch' has diverged from origin/$branch - skipping auto-merge."
        fi
    fi
elif [ "$FRESH" -eq 0 ] && [ "$DO_PULL" -eq 0 ]; then
    step "Skipping 'git pull' (--no-pull)"
fi

# --------------------------------------------------------------------------- #
# 5. Python dependencies
# --------------------------------------------------------------------------- #
step "Installing Python dependencies (profile: $PROFILE${WITH_OCR:+, ocr})"

$VPIP install --upgrade pip setuptools wheel

if [ "$PROFILE" = "dev" ]; then
    # dev pulls in runtime + ocr + pytest/ruff + flask
    $VPIP install -r "$REPO_ROOT/requirements-dev.txt"
elif [ "$WITH_OCR" -eq 1 ]; then
    $VPIP install -r "$REPO_ROOT/requirements-ocr.txt"
else
    $VPIP install -r "$REPO_ROOT/requirements.txt"
fi

# the package itself, editable, so `wingjournal` / `wjm` land on PATH
info "installing the wingjournal package (editable)..."
$VPIP install -e "$REPO_ROOT"

# demo front-end deps come in with the dev profile via requirements-dev.txt;
# make sure the standalone file is honoured too
if [ "$PROFILE" = "dev" ] && [ -f "$REPO_ROOT/demo/requirements.txt" ]; then
    $VPIP install -r "$REPO_ROOT/demo/requirements.txt"
fi

# --------------------------------------------------------------------------- #
# 6. enable the commit-message template (best effort, matches CONTRIBUTING.md)
# --------------------------------------------------------------------------- #
if git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 && [ -f "$REPO_ROOT/.gitmessage" ]; then
    git -C "$REPO_ROOT" config commit.template .gitmessage || true
fi

# --------------------------------------------------------------------------- #
# 7. verify
# --------------------------------------------------------------------------- #
step "Verifying the environment"

fail=0

"$VPY" - <<'PY' || fail=1
import importlib, sys

mods = ["numpy", "cv2", "PIL", "wingjournal"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"    ok   {m}")
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL {m}: {exc}")
        sys.exit(1)

import wingjournal
print(f"    wingjournal {wingjournal.__version__}")
PY

if "$VENV_DIR/bin/wingjournal" --help >/dev/null 2>&1; then
    info "ok   'wingjournal' CLI responds"
else
    warn "the 'wingjournal' CLI did not respond to --help"
    fail=1
fi

if [ "$WITH_OCR" -eq 1 ]; then
    if "$VPY" -c 'import pytesseract' 2>/dev/null; then
        if command -v tesseract >/dev/null 2>&1; then
            info "ok   OCR available (pytesseract + tesseract binary)"
        else
            warn "pytesseract installed but no 'tesseract' binary - OCR will use the null fallback"
        fi
    else
        warn "pytesseract not importable"
    fi
fi

if [ "$RUN_CHECK" -eq 1 ]; then
    step "Running the test suite (--check)"
    "$VENV_DIR/bin/pytest" -q || fail=1
fi

# --------------------------------------------------------------------------- #
# done
# --------------------------------------------------------------------------- #
if [ "$fail" -eq 0 ]; then
    step "Setup complete"
else
    step "Setup finished with warnings"
fi

cat <<EOF

  Activate the environment:

      source ${VENV_DIR#$REPO_ROOT/}/bin/activate

  Then, for example:

      wingjournal make-sheet --out writing-sheet.pdf
      wingjournal eval --cases 40
      pytest -q
      python demo/run.py            # -> http://127.0.0.1:5000

EOF

exit "$fail"
