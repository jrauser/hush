# Hush — TODO

## Mobile annotation via local web UI

Correcting misclassifications currently requires tabbing to the monitor window, pausing, labeling, unpausing, and tabbing back. A local web UI would let you pull out your phone and correct frames without touching the laptop.

**Approach:** Run a small Flask server in a background thread inside `monitor.py`. Serves the current frame grid as an HTML page over the LAN. Phone hits `http://<laptop>.local:<port>` in Safari — Bonjour handles discovery.

- Serve thumbnails from the rolling history
- Tappable frames: tap → pick label → POST back
- Server calls the same `store.add_screenshot()` / `store.set_label()` code path as the keyboard shortcuts
- Auto-refresh or websocket for live updates
- No cloud, no auth, no iOS app — just a web page on the local network
