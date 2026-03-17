"""monitor.py — Real-time ad-detection monitor with correction GUI.

Background thread classifies screenshots at regular intervals. Main thread
shows a rolling history of thumbnails; the user can click a frame and press
A/C/N/X to relabel it, which saves it permanently for the next training run.

Press the Pause button to freeze capture and muting while inspecting frames,
then Resume to continue.
"""

import argparse
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import flask
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk

from label_store import LabelStore

# ── constants ─────────────────────────────────────────────────────────────────

LABEL_NAMES  = ['ad', 'content', 'neither']
LABEL_COLORS = {
    'ad':      '#c0392b',
    'content': '#27ae60',
    'neither': '#7f8c8d',
    'ignore':  '#8e44ad',
}
UNCERTAIN_COLOR  = '#e67e22'
UNCERTAIN_THRESH = 0.80

THUMB_SIZE = 140
PADDING    = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── feature extractor ─────────────────────────────────────────────────────────

def _build_extractor(extractor_name: str, device):
    """Return a callable path → np.ndarray of shape [1, D]."""
    if extractor_name == 'resnet50':
        import torchvision.models as tv_models
        model = tv_models.resnet50(weights='IMAGENET1K_V2')
        net   = nn.Sequential(*list(model.children())[:-1])
        net.eval().to(device)
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        def extract(path: str) -> np.ndarray:
            img = Image.open(path).convert('RGB')
            t   = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                return net(t).squeeze(-1).squeeze(-1).cpu().numpy()  # [1, 2048]
        return extract
    raise ValueError(f'Unknown extractor: {extractor_name!r}')


# ── macOS volume helpers ───────────────────────────────────────────────────────

def _osascript(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(['osascript', '-e', cmd], capture_output=True, text=True)

def _get_muted() -> bool:
    r = _osascript('output muted of (get volume settings)')
    return r.returncode == 0 and r.stdout.strip().lower() == 'true'

def _set_muted(muted: bool) -> None:
    _osascript(f'set volume output muted {str(muted).lower()}')


# ── data ──────────────────────────────────────────────────────────────────────

@dataclass
class FrameEntry:
    path:        str
    captured_at: str
    pred_label:  str          # argmax prediction from classifier
    prob_ad:     float        # P(ad)
    probs:       list         # [P(ad), P(content), P(neither)]
    relabeled:   bool  = False
    user_label:  Optional[str] = None


# ── app ───────────────────────────────────────────────────────────────────────

class MonitorApp:
    def __init__(self, root: tk.Tk, args):
        self.root  = root
        self.args  = args
        self.store = LabelStore(args.db)
        self._test_mode = args.test
        os.makedirs(args.hard_frame_dir, exist_ok=True)

        # Load model
        bundle          = joblib.load(args.model)
        self._clf       = bundle['classifier']
        self._label_map = bundle['label_map']   # {'ad':0, 'content':1, 'neither':2}
        self._ad_class  = self._label_map['ad']

        # Feature extractor
        device        = torch.device('mps') if torch.backends.mps.is_available() \
                        else torch.device('cpu')
        self._extract = _build_extractor(bundle['extractor'], device)

        # Rolling history of temp frames
        self._history      = deque(maxlen=args.history_size)
        self._history_lock = threading.Lock()
        self._temp_dir     = tempfile.mkdtemp(prefix='hush_monitor_')

        # Mute state machine
        self._muted     = False   # are we currently muted?
        self._we_muted  = False   # did *we* issue the mute (vs user)?
        self._override  = False   # user manually changed volume; hands off
        self._ad_streak = 0       # consecutive ad-classified frames
        self._ok_streak = 0       # consecutive non-ad-classified frames

        # Pause state — capture and muting both stop while paused
        self._paused = False

        # UI state
        self._selected   = None   # path of selected FrameEntry, or None
        self._photo_refs = []     # keep ImageTk.PhotoImage objects alive

        # Background → main thread queue
        self._queue   = queue.Queue()
        self._running = True

        self._build_gui()
        self.root.title('hush monitor')
        self.root.geometry('900x420')
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # Key bindings
        for key, lbl in [('a', 'ad'), ('c', 'content'), ('n', 'neither'), ('x', 'ignore')]:
            self.root.bind(key, lambda e, l=lbl: self._relabel_selected(l))
        self.root.bind('<Escape>', lambda e: self._deselect())

        # SIGTERM → clean shutdown on main thread
        signal.signal(signal.SIGTERM, lambda *_: self.root.after(0, self._on_close))

        # Start web server (if requested)
        if args.web_port > 0:
            self._start_web_server(args.web_port)

        # Start background capture thread
        self._bg = threading.Thread(target=self._capture_loop, daemon=True)
        self._bg.start()

        # Start polling loop
        self.root.after(250, self._poll_queue)

    # ── GUI layout ────────────────────────────────────────────────────────────

    def _build_gui(self):
        # ── header bar ──
        hdr = tk.Frame(self.root, bg='#1a252f')
        hdr.pack(fill='x', side='top')

        self._status_lbl = tk.Label(
            hdr, text='LIVE', font=('Helvetica', 22, 'bold'),
            fg='#2ecc71', bg='#1a252f', width=10, anchor='w',
        )
        self._status_lbl.pack(side='left', padx=16, pady=8)

        self._info_lbl = tk.Label(
            hdr, text='', font=('Helvetica', 11),
            fg='#95a5a6', bg='#1a252f', anchor='w',
        )
        self._info_lbl.pack(side='left', fill='x', expand=True, padx=4)

        # Use a Label instead of Button — macOS native buttons ignore bg/fg.
        self._pause_btn = tk.Label(
            hdr, text='⏸  Pause',
            font=('Helvetica', 12), fg='white', bg='#2c3e50',
            padx=12, pady=4, cursor='hand2',
        )
        self._pause_btn.bind('<Button-1>', lambda e: self._toggle_pause())
        self._pause_btn.pack(side='right', padx=16, pady=8)

        # ── legend / help (bottom) ──
        leg = tk.Frame(self.root, bg='#ecf0f1', pady=3)
        leg.pack(fill='x', side='bottom')
        tk.Label(
            leg,
            text='Click to select  ·  A = ad   C = content   N = neither   X = ignore   Esc = deselect',
            font=('Helvetica', 10), bg='#ecf0f1', fg='#7f8c8d',
        ).pack()

        # ── thumbnail grid ──
        self._grid = tk.Frame(self.root, bg='#ecf0f1')
        self._grid.pack(fill='both', expand=True, padx=6, pady=6)

    # ── pause toggle ──────────────────────────────────────────────────────────

    def _toggle_pause(self):
        self._paused = not self._paused
        self._update_header()

    # ── background capture loop ───────────────────────────────────────────────

    def _capture_loop(self):
        seq = 0
        while self._running:
            if self._paused:
                time.sleep(0.25)
                continue

            t0 = time.monotonic()
            ts = datetime.now()

            fname = ts.strftime('%Y%m%d_%H%M%S') + f'_{seq:05d}.jpg'
            path  = os.path.join(self._temp_dir, fname)

            captured = (
                subprocess.run(
                    ['screencapture', '-x', '-t', 'jpg', '-m', path],
                    capture_output=True,
                ).returncode == 0
                and os.path.isfile(path)
                and os.path.getsize(path) > 0
            )

            if captured:
                try:
                    feat   = self._extract(path)           # [1, D]
                    raw    = self._clf.predict_proba(feat)  # [1, k]

                    # Map clf.classes_ columns → full [3] probability vector
                    probs3 = np.zeros(3)
                    for col, cls in enumerate(self._clf.classes_):
                        probs3[cls] = raw[0, col]

                    pred = LABEL_NAMES[int(np.argmax(probs3))]
                    self._queue.put(FrameEntry(
                        path=path,
                        captured_at=ts.isoformat(),
                        pred_label=pred,
                        prob_ad=float(probs3[self._ad_class]),
                        probs=probs3.tolist(),
                    ))
                    seq += 1
                except Exception:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass

            elapsed = time.monotonic() - t0
            print(f'[monitor] frame {seq}: {elapsed:.2f}s', flush=True)
            time.sleep(max(0.0, self.args.interval - elapsed))

    # ── main-thread polling ───────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                self._on_new_frame(self._queue.get_nowait())
        except queue.Empty:
            pass
        if self._running:
            self.root.after(250, self._poll_queue)

    def _on_new_frame(self, entry: FrameEntry):
        # Evict oldest frame if history is full
        if len(self._history) == self._history.maxlen:
            old = self._history[0]
            if not old.relabeled:
                try:
                    os.remove(old.path)
                except OSError:
                    pass

        self._history.append(entry)

        # Update streaks.
        # 'neither' (tabbed away) resets both — the monitor can't know what's
        # happening so it holds the current mute state and starts fresh when
        # known content resumes.
        if entry.pred_label == 'ad':
            self._ad_streak += 1
            self._ok_streak  = 0
        elif entry.pred_label == 'content':
            self._ok_streak  += 1
            self._ad_streak   = 0
        else:  # 'neither'
            self._ad_streak = 0
            self._ok_streak = 0

        self._update_mute_state()
        self._render()
        self._update_header()

    # ── mute state machine ────────────────────────────────────────────────────

    def _update_mute_state(self):
        # Never touch volume while paused
        if self._paused:
            return

        # Check for user override (they manually unmuted while we had it muted)
        if self._we_muted and not self._test_mode and not _get_muted():
            self._override  = True
            self._muted     = False
            self._we_muted  = False
            self._ad_streak = 0    # don't immediately re-mute
            return

        if self._override:
            # Wait for a clean run of content before re-engaging
            if self._ok_streak >= self.args.unmute_delay:
                self._override = False
            return

        if not self._muted and self._ad_streak >= self.args.mute_delay:
            if not self._test_mode:
                _set_muted(True)
            self._muted    = True
            self._we_muted = True

        elif self._muted and self._ok_streak >= self.args.unmute_delay:
            if not self._test_mode:
                _set_muted(False)
            self._muted    = False
            self._we_muted = False

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render(self):
        for w in self._grid.winfo_children():
            w.destroy()
        self._photo_refs.clear()

        self.root.update_idletasks()
        avail_w = max(self._grid.winfo_width(), 600)
        cell_w  = THUMB_SIZE + PADDING * 2 + 8
        n_cols  = max(1, avail_w // cell_w)

        # Newest frame in top-left, oldest bottom-right
        for i, entry in enumerate(reversed(list(self._history))):
            self._render_thumb(entry, row=i // n_cols, col=i % n_cols)

    def _render_thumb(self, entry: FrameEntry, row: int, col: int):
        display_label = entry.user_label if entry.relabeled else entry.pred_label
        uncertain     = max(entry.probs) < UNCERTAIN_THRESH and not entry.relabeled

        border_color = (UNCERTAIN_COLOR if uncertain
                        else LABEL_COLORS.get(display_label, '#7f8c8d'))
        selected = (entry.path == self._selected)
        if selected:
            border_color = '#f1c40f'
        border_width = 4 if selected else 2

        cell = tk.Frame(self._grid, bg=border_color, padx=border_width, pady=border_width)
        cell.grid(row=row, column=col, padx=PADDING // 2, pady=PADDING // 2, sticky='nw')

        inner = tk.Frame(cell, bg='#d5d8dc')
        inner.pack()

        try:
            img = Image.open(entry.path)
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            photo = ImageTk.PhotoImage(img)
            self._photo_refs.append(photo)
            img_lbl = tk.Label(inner, image=photo, bg='#d5d8dc', cursor='hand2')
        except Exception:
            img_lbl = tk.Label(inner, text='[error]', width=12, height=6,
                               bg='#bdc3c7', cursor='hand2')
        img_lbl.pack()

        corrected = '  ✓' if entry.relabeled else ''
        caption   = f'{display_label}  {entry.prob_ad:.0%}{corrected}'
        cap_lbl   = tk.Label(
            inner, text=caption, font=('Helvetica', 9),
            bg=border_color, fg='white', anchor='center',
        )
        cap_lbl.pack(fill='x')

        for w in (cell, inner, img_lbl, cap_lbl):
            w.bind('<Button-1>', lambda e, p=entry.path: self._on_click(p))

    def _on_click(self, path: str):
        self._selected = None if self._selected == path else path
        self.root.focus_set()
        self._render()

    def _deselect(self):
        self._selected = None
        self._render()

    # ── relabeling ────────────────────────────────────────────────────────────

    def _relabel_selected(self, label: str):
        if not self._selected:
            return
        entry = next((e for e in self._history if e.path == self._selected), None)
        if entry is None or entry.relabeled:
            return

        # Move temp file → permanent storage
        fname = os.path.basename(entry.path)
        dest  = os.path.join(self.args.hard_frame_dir, fname)
        shutil.move(entry.path, dest)

        entry.path       = dest
        entry.relabeled  = True
        entry.user_label = label

        # Persist to DB so it appears in the next label.py / train.py cycle
        self.store.add_screenshot(dest, entry.captured_at, source='monitor')
        self.store.set_label(dest, label)

        self._selected = None
        self._render()

    # ── header update ─────────────────────────────────────────────────────────

    def _update_header(self):
        if self._paused:
            self._pause_btn.config(text='▶  Resume', bg='#c0392b')
            self._status_lbl.config(text='PAUSED', fg='#e67e22')
        else:
            self._pause_btn.config(text='⏸  Pause', bg='#2c3e50')
            if self._override:
                self._status_lbl.config(text='OVERRIDE', fg='#e67e22')
            elif self._muted:
                self._status_lbl.config(text='MUTED',    fg='#e74c3c')
            else:
                self._status_lbl.config(text='LIVE',     fg='#2ecc71')

        if self._history:
            last = list(self._history)[-1]
            self._info_lbl.config(
                text=(
                    f'last: {last.pred_label} ({last.prob_ad:.0%} ad)'
                    f'  ·  ad streak: {self._ad_streak}'
                    f'  ·  ok streak: {self._ok_streak}'
                )
            )

    # ── web server ────────────────────────────────────────────────────────────

    def _start_web_server(self, port: int):
        import socket
        app = flask.Flask('hush_monitor')
        # Suppress Flask startup banner and request logs
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        @app.route('/')
        def index():
            return flask.Response(_WEB_HTML, mimetype='text/html')

        @app.route('/thumb/<fname>')
        def thumb(fname):
            # Security: only allow simple filenames (no path traversal)
            if '/' in fname or '..' in fname:
                flask.abort(400)
            for directory in (self._temp_dir, self.args.hard_frame_dir):
                path = os.path.join(directory, fname)
                if os.path.isfile(path):
                    return flask.send_file(path, mimetype='image/jpeg')
            flask.abort(404)

        @app.route('/api/state')
        def api_state():
            if self._paused:
                status = 'PAUSED'
            elif self._override:
                status = 'OVERRIDE'
            elif self._muted:
                status = 'MUTED'
            else:
                status = 'LIVE'

            with self._history_lock:
                frames_snap = list(self._history)

            last = frames_snap[-1] if frames_snap else None
            frames = []
            for e in reversed(frames_snap):  # newest first
                display_label = e.user_label if e.relabeled else e.pred_label
                uncertain = max(e.probs) < UNCERTAIN_THRESH and not e.relabeled
                frames.append({
                    'fname':       os.path.basename(e.path),
                    'pred_label':  e.pred_label,
                    'user_label':  e.user_label,
                    'relabeled':   e.relabeled,
                    'display_label': display_label,
                    'prob_ad':     round(e.prob_ad, 4),
                    'uncertain':   uncertain,
                })

            return flask.jsonify({
                'status':     status,
                'last_label': last.pred_label if last else None,
                'prob_ad':    round(last.prob_ad, 4) if last else None,
                'ad_streak':  self._ad_streak,
                'ok_streak':  self._ok_streak,
                'frames':     frames,
            })

        @app.route('/api/pause', methods=['POST'])
        def api_pause():
            self.root.after(0, self._toggle_pause)
            return flask.jsonify({'ok': True})

        @app.route('/api/relabel', methods=['POST'])
        def api_relabel():
            data = flask.request.get_json(force=True, silent=True) or {}
            fname = data.get('filename', '')
            label = data.get('label', '')
            if label not in ('ad', 'content', 'neither', 'ignore'):
                return flask.jsonify({'ok': False, 'error': 'invalid label'}), 400
            if not fname or '/' in fname or '..' in fname:
                return flask.jsonify({'ok': False, 'error': 'invalid filename'}), 400
            self.root.after(0, lambda: self._relabel_by_fname(fname, label))
            return flask.jsonify({'ok': True})

        t = threading.Thread(
            target=app.run,
            kwargs=dict(host='0.0.0.0', port=port, use_reloader=False, threaded=True),
            daemon=True,
        )
        t.start()

        try:
            h = socket.gethostname()
            hostname = h if h.endswith('.local') else h + '.local'
        except Exception:
            hostname = 'localhost.local'
        print(f'[monitor] web UI: http://localhost:{port}  '
              f'(LAN: http://{hostname}:{port})', flush=True)

    def _relabel_by_fname(self, fname: str, label: str):
        """Relabel a frame by filename — called on the main thread from the web API."""
        with self._history_lock:
            entry = next(
                (e for e in self._history if os.path.basename(e.path) == fname),
                None,
            )
        if entry is None or entry.relabeled:
            return

        dest = os.path.join(self.args.hard_frame_dir, os.path.basename(entry.path))
        shutil.move(entry.path, dest)

        entry.path      = dest
        entry.relabeled = True
        entry.user_label = label

        self.store.add_screenshot(dest, entry.captured_at, source='monitor')
        self.store.set_label(dest, label)

        self._render()

    # ── exit ──────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._running = False
        if self._muted and self._we_muted and not self._test_mode:
            _set_muted(False)
        self._cleanup()
        try:
            self.root.destroy()
        except Exception:
            pass

    def _cleanup(self):
        """Delete temp files for frames the user didn't relabel."""
        for entry in self._history:
            if not entry.relabeled:
                try:
                    os.remove(entry.path)
                except OSError:
                    pass
        try:
            os.rmdir(self._temp_dir)
        except OSError:
            pass


# ── web UI HTML ───────────────────────────────────────────────────────────────

_WEB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hush monitor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a252f; color: #ecf0f1; font-family: -apple-system, sans-serif; }

  #header {
    position: sticky; top: 0; z-index: 10;
    background: #1a252f; padding: 10px 14px;
    display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid #2c3e50;
  }
  #status {
    font-size: 1.3rem; font-weight: bold; min-width: 90px;
  }
  #info { font-size: 0.8rem; color: #95a5a6; flex: 1; }
  #test-badge {
    font-size: 0.7rem; background: #8e44ad; color: white;
    padding: 2px 7px; border-radius: 10px; display: none;
  }
  #pause-btn {
    padding: 7px 14px; border: none; border-radius: 6px;
    font-size: 0.9rem; cursor: pointer;
    background: #2c3e50; color: white; white-space: nowrap;
  }
  #pause-btn.paused { background: #c0392b; }

  #label-bar {
    display: none; position: sticky; top: 52px; z-index: 10;
    background: #2c3e50; padding: 8px 14px;
    gap: 8px; flex-wrap: wrap;
    border-bottom: 1px solid #34495e;
  }
  #label-bar.visible { display: flex; }
  #label-bar span { font-size: 0.8rem; color: #bdc3c7; align-self: center; }
  .lbtn {
    flex: 1; min-width: 60px; padding: 10px 4px;
    border: none; border-radius: 6px; font-size: 0.9rem; font-weight: bold;
    cursor: pointer; color: white;
  }
  .lbtn.ad      { background: #c0392b; }
  .lbtn.content { background: #27ae60; }
  .lbtn.neither { background: #7f8c8d; }
  .lbtn.ignore  { background: #8e44ad; }
  .lbtn.cancel  { background: #34495e; }

  #grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 6px; padding: 8px;
  }

  .cell {
    border-radius: 6px; overflow: hidden; cursor: pointer;
    border: 3px solid transparent; position: relative;
  }
  .cell img { display: block; width: 100%; aspect-ratio: 16/9; object-fit: cover; }
  .caption {
    text-align: center; font-size: 0.7rem; padding: 3px 2px;
    background: inherit; color: white;
  }
  .cell.selected { border-color: #f1c40f !important; }

  .label-ad      { border-color: #c0392b; background: #c0392b; }
  .label-content { border-color: #27ae60; background: #27ae60; }
  .label-neither { border-color: #7f8c8d; background: #7f8c8d; }
  .label-ignore  { border-color: #8e44ad; background: #8e44ad; }
  .uncertain     { border-color: #e67e22 !important; }
</style>
</head>
<body>

<div id="header">
  <div id="status">LIVE</div>
  <div id="info">waiting…</div>
  <div id="test-badge">TEST</div>
  <button id="pause-btn" onclick="togglePause()">⏸ Pause</button>
</div>
<div id="label-bar">
  <span id="sel-name"></span>
  <button class="lbtn ad"      onclick="relabel('ad')">Ad</button>
  <button class="lbtn content" onclick="relabel('content')">Content</button>
  <button class="lbtn neither" onclick="relabel('neither')">Neither</button>
  <button class="lbtn ignore"  onclick="relabel('ignore')">Ignore</button>
  <button class="lbtn cancel"  onclick="deselect()">✕</button>
</div>
<div id="grid"></div>

<script>
const STATUS_COLORS = {LIVE:'#2ecc71', MUTED:'#e74c3c', OVERRIDE:'#e67e22', PAUSED:'#e67e22'};
let selectedFname = null;
let pollTimer = null;

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(poll, 3000);
}
function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
}

async function poll() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) return;
    const d = await r.json();
    render(d);
  } catch(e) {}
}

function render(d) {
  // Header
  const sEl = document.getElementById('status');
  sEl.textContent = d.status;
  sEl.style.color = STATUS_COLORS[d.status] || '#ecf0f1';

  const pbtn = document.getElementById('pause-btn');
  if (d.status === 'PAUSED') {
    pbtn.textContent = '▶ Resume';
    pbtn.classList.add('paused');
  } else {
    pbtn.textContent = '⏸ Pause';
    pbtn.classList.remove('paused');
  }

  if (d.last_label) {
    document.getElementById('info').textContent =
      `last: ${d.last_label} (${Math.round(d.prob_ad*100)}% ad)` +
      `  ·  ad streak: ${d.ad_streak}  ·  ok streak: ${d.ok_streak}`;
  }

  // Grid
  const grid = document.getElementById('grid');
  const prevSelected = selectedFname;
  grid.innerHTML = '';

  for (const f of d.frames) {
    const cell = document.createElement('div');
    cell.className = 'cell label-' + f.display_label;
    if (f.uncertain) cell.classList.add('uncertain');
    if (f.fname === prevSelected) cell.classList.add('selected');

    const img = document.createElement('img');
    img.src = '/thumb/' + encodeURIComponent(f.fname);
    img.loading = 'lazy';
    cell.appendChild(img);

    const cap = document.createElement('div');
    cap.className = 'caption';
    const pct = Math.round(f.prob_ad * 100);
    cap.textContent = f.display_label + '  ' + pct + '%' + (f.relabeled ? ' ✓' : '');
    cell.appendChild(cap);

    cell.addEventListener('click', () => selectFrame(f.fname));
    grid.appendChild(cell);
  }

  updateLabelBar();
}

function selectFrame(fname) {
  if (selectedFname === fname) {
    deselect();
    return;
  }
  selectedFname = fname;
  stopPolling();
  // Highlight
  document.querySelectorAll('.cell').forEach(c => c.classList.remove('selected'));
  // Find by re-querying after render would lose state, so just re-poll once
  poll().then(updateLabelBar);
  updateLabelBar();
}

function deselect() {
  selectedFname = null;
  document.getElementById('label-bar').classList.remove('visible');
  document.getElementById('sel-name').textContent = '';
  startPolling();
}

function updateLabelBar() {
  const bar = document.getElementById('label-bar');
  if (selectedFname) {
    bar.classList.add('visible');
    document.getElementById('sel-name').textContent = selectedFname;
  } else {
    bar.classList.remove('visible');
  }
}

async function togglePause() {
  try {
    await fetch('/api/pause', {method: 'POST'});
  } catch(e) {}
  setTimeout(poll, 300);
}

async function relabel(label) {
  if (!selectedFname) return;
  const fname = selectedFname;
  deselect();
  try {
    await fetch('/api/relabel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: fname, label}),
    });
  } catch(e) {}
  // Give the main thread a moment to process before re-polling
  setTimeout(poll, 500);
}

poll();
startPolling();
</script>
</body>
</html>"""


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='hush monitor — real-time ad muting with correction GUI.',
    )
    parser.add_argument('--model',          default='models/classifier.joblib',
                        help='Trained model bundle (default: models/classifier.joblib)')
    parser.add_argument('--db',             default='hush.db',
                        help='SQLite database (default: hush.db)')
    parser.add_argument('--interval',       type=float, default=3.0,
                        help='Seconds between classifications (default: 3)')
    parser.add_argument('--mute-delay',     type=int,   default=2, dest='mute_delay',
                        help='Consecutive ad frames before muting (default: 2 → 6 s)')
    parser.add_argument('--unmute-delay',   type=int,   default=3, dest='unmute_delay',
                        help='Consecutive non-ad frames before unmuting (default: 3 → 9 s)')
    parser.add_argument('--history-size',   type=int,   default=40, dest='history_size',
                        help='Frames to keep in the rolling display (default: 40 ≈ 2 min)')
    parser.add_argument('--hard-frame-dir', default='hard_frames', dest='hard_frame_dir',
                        help='Where to save relabeled frames (default: hard_frames/)')
    parser.add_argument('--web-port', type=int, default=5150, dest='web_port',
                        help='Port for web annotation UI (default: 5150, 0 = disabled)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: in-memory DB, no audio muting')
    args = parser.parse_args()

    if args.test:
        args.db = ':memory:'
        print('[monitor] TEST MODE — no DB writes, no audio muting', flush=True)

    root = tk.Tk()
    app  = MonitorApp(root, args)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app._on_close()


if __name__ == '__main__':
    main()
