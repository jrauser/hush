#!/usr/bin/env bash
# build_app.sh — build Hush.app, a double-clickable macOS launcher.
#
# The bundle is a thin native launcher that runs this repo's code with this
# repo's .venv; it is not a standalone app. The repo path is baked in, so
# re-run this script if you move the repo or upgrade Python (setup.sh runs
# it for you). The finished app can be dragged to /Applications or the Dock.
#
# Usage:
#   macos/build_app.sh

set -euo pipefail

MACOS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$MACOS_DIR")"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
APP="$REPO_DIR/Hush.app"

if [ "$(uname)" != "Darwin" ]; then
  echo "Error: Hush.app can only be built on macOS."
  exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Error: $VENV_PYTHON not found. Run ./setup.sh first."
  exit 1
fi

if ! command -v clang &>/dev/null; then
  echo "Error: clang not found. Install the Xcode command line tools:"
  echo "  xcode-select --install"
  exit 1
fi

config_var() {
  "$VENV_PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('$1'))"
}

# Must be a framework build: the launcher relies on __PYVENV_LAUNCHER__,
# which only framework builds honor.
if [ "$(config_var PYTHONFRAMEWORK)" = "" ]; then
  echo "Error: the venv's Python is not a framework build (e.g. Homebrew's)."
  exit 1
fi

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "Building $APP..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$MACOS_DIR/Info.plist" "$APP/Contents/Info.plist"

# ── Launcher ──────────────────────────────────────────────────────────────────

clang -Wall -Wextra -Werror -O2 \
  -I"$(config_var INCLUDEPY)" \
  -DHUSH_REPO_DIR="\"$REPO_DIR\"" \
  "$MACOS_DIR/launcher.c" \
  -L"$(config_var LIBPL)" -lpython"$(config_var LDVERSION)" \
  -o "$APP/Contents/MacOS/Hush"

# ── Icon ──────────────────────────────────────────────────────────────────────

ICONSET="$BUILD_DIR/Hush.iconset"
"$VENV_PYTHON" "$MACOS_DIR/make_icon.py" "$ICONSET"
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/Hush.icns"

# ── Sign ──────────────────────────────────────────────────────────────────────
#
# Ad-hoc signature covering the whole bundle (Info.plist and icon included),
# so macOS treats it as one coherent app.

codesign --force --sign - "$APP"

# Make Finder pick up the new icon if an older build was registered.
touch "$APP"

echo "Built $APP — double-click it in Finder to launch Hush."
echo "Output is logged to ~/Library/Logs/Hush.log."
