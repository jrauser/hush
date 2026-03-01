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
        self._history  = deque(maxlen=args.history_size)
        self._temp_dir = tempfile.mkdtemp(prefix='hush_monitor_')

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
        if self._we_muted and not _get_muted():
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
            _set_muted(True)
            self._muted    = True
            self._we_muted = True

        elif self._muted and self._ok_streak >= self.args.unmute_delay:
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

    # ── exit ──────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._running = False
        if self._muted and self._we_muted:
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


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='hush monitor — real-time ad muting with correction GUI.',
    )
    parser.add_argument('--model',          default='models/classifier.joblib',
                        help='Trained model bundle (default: models/classifier.joblib)')
    parser.add_argument('--db',             default='hush.db',
                        help='SQLite database (default: hush.db)')
    parser.add_argument('--interval',       type=float, default=5.0,
                        help='Seconds between classifications (default: 5)')
    parser.add_argument('--mute-delay',     type=int,   default=2, dest='mute_delay',
                        help='Consecutive ad frames before muting (default: 2 → 10 s)')
    parser.add_argument('--unmute-delay',   type=int,   default=3, dest='unmute_delay',
                        help='Consecutive non-ad frames before unmuting (default: 3 → 15 s)')
    parser.add_argument('--history-size',   type=int,   default=24, dest='history_size',
                        help='Frames to keep in the rolling display (default: 24 ≈ 2 min)')
    parser.add_argument('--hard-frame-dir', default='hard_frames', dest='hard_frame_dir',
                        help='Where to save relabeled frames (default: hard_frames/)')
    args = parser.parse_args()

    root = tk.Tk()
    app  = MonitorApp(root, args)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app._on_close()


if __name__ == '__main__':
    main()
