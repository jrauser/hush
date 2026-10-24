#!/usr/bin/env bash
# setup.sh — one-time setup for hush
#
# Run this once after cloning the repo. It creates the Python virtual
# environment, installs dependencies, and pre-downloads the CLIP model
# weights so the first training run doesn't surprise you with a 350MB
# download.
#
# Usage:
#   ./setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# ── Check prerequisites ───────────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
  echo "Error: 'uv' is not installed."
  echo "Install it from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if ! command -v python3.13 &>/dev/null; then
  echo "Error: Python 3.13 is not installed."
  echo "On macOS: brew install python@3.13"
  echo "On Linux: use your package manager or https://python.org"
  exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  uv venv --python python3.13 .venv
else
  echo "Virtual environment already exists, skipping creation."
fi

# ── Install dependencies ──────────────────────────────────────────────────────

echo "Installing dependencies (this may take a few minutes)..."
uv pip install -e ".[dev]"

# ── Pre-download CLIP weights ─────────────────────────────────────────────────
#
# CLIP downloads ViT-B/32 weights (~350MB) to ~/.cache/clip on first use.
# Doing it here avoids a surprise pause mid-training.

echo "Downloading CLIP model weights (~350MB, cached in ~/.cache/clip)..."
.venv/bin/python - <<'EOF'
import clip
print(f"  Downloading {clip.available_models()[0]}...")
clip.load("ViT-B/32", device="cpu")
print("  Done.")
EOF

# ── Build Hush.app (macOS) ────────────────────────────────────────────────────

if [ "$(uname)" = "Darwin" ]; then
  macos/build_app.sh
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "Setup complete. Next steps:"
echo "  1. Load the Chrome extension: open chrome://extensions, enable"
echo "     Developer mode, click 'Load unpacked', select the extension/ folder."
echo "  2. Launch the app: ./.venv/bin/python -m hush.v2app"
echo "     (or on macOS, double-click Hush.app in Finder)"
echo ""
echo "See README.md for the full getting started guide."
