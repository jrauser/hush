# Hush — TODO

## Mobile annotation via local web UI

Correcting misclassifications currently requires tabbing to the monitor window, pausing, labeling, unpausing, and tabbing back. A local web UI would let you pull out your phone and correct frames without touching the laptop.

**Approach:** Run a small Flask server in a background thread inside `monitor.py`. Serves the current frame grid as an HTML page over the LAN. Phone hits `http://<laptop>.local:<port>` in Safari — Bonjour handles discovery.

- Serve thumbnails from the rolling history
- Tappable frames: tap → pick label → POST back
- Server calls the same `store.add_screenshot()` / `store.set_label()` code path as the keyboard shortcuts
- Auto-refresh or websocket for live updates
- No cloud, no auth, no iOS app — just a web page on the local network

## Background tab capture for audio-only monitoring

Use case: you're working on something else while a sports stream plays in a background Chrome tab. You can hear the audio — your brain handles "is something exciting happening?" — but you want ads muted automatically.

The current `screencapture` approach grabs whatever is on screen, so it captures your work, not the stream. We need the stream's video frames even when the tab isn't visible.

**Approach:** A small Chrome extension using the `chrome.tabCapture` API.

- The extension captures video (and optionally audio) from the stream's tab regardless of foreground state
- Periodically sends frames to a local endpoint (WebSocket or HTTP POST to the monitor)
- Monitor feeds those frames into the existing ResNet classifier — no new model needed
- Audio-based features (volume spikes, crowd noise) could be a useful second signal but aren't required for the initial version
- Extension itself would be ~50 lines of JS

## Upgrade to Python 3.13 + modern dependencies

The version pins in requirements.txt were chosen for the original Intel MacBook. On Apple Silicon + Python 3.13, the upper bounds can be removed:

- `torch<2.5` → unpin (2.5+ dropped Intel Mac wheels, but Apple Silicon is fine)
- `numpy<2` → unpin (only needed because old torch required numpy 1.x)
- `torchvision<0.20` → unpin (was pinned to match old torch range)
- All other deps (Pillow, scikit-learn, joblib, tqdm, flask) work on 3.13 as-is

## Package for distribution via PyPI + uv

Rename the project (hush conflicts with a common shell) and publish to PyPI so users can install with `uv tool install muteads` (or whatever name we pick).

**Prerequisites:**
- Pick a name that's available on PyPI and doesn't collide with existing tools
- Add a `pyproject.toml` with `[project.scripts]` entry points (e.g., `muteads-capture`, `muteads-monitor`, or a single `muteads` CLI with subcommands)
- Migrate from `requirements.txt` to `pyproject.toml` dependencies

**User experience:**
```
brew install uv              # one-time
uv tool install muteads      # installs into isolated env, puts CLI on PATH
muteads monitor --web-port 5150
```

**Considerations:**
- Ships a pretrained "good enough" model, with the correction loop refining it per-user
- Heavy deps (PyTorch ~1GB) mean the initial install is large but pre-built wheels avoid compilation
- Training/labeling tools can ship in the same package — power users retrain, casual users just correct via the web UI
