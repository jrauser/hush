"""Hush v2 Desktop App — PyQt6 GUI with Live Monitor view.

Entry point: python -m hush.v2app

Launches the FastAPI server in a background thread, connects to the Chrome
extension, and displays a live monitor with frame classification and
mute/unmute control.
"""

import argparse
import logging
import os
import signal
import sys
from collections import deque
from typing import Optional

from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QEasingCurve, QPropertyAnimation,
    pyqtSignal, QObject,
)
from PyQt6.QtGui import QPixmap, QFont, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from hush.frame_pipeline import FramePipeline, FrameResult
from hush.label_store import LabelStore
from hush.server import HushServer, FrameData

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "ad": "#c0392b",
    "content": "#27ae60",
    "neither": "#7f8c8d",
    "ignore": "#8e44ad",
}
UNCERTAIN_COLOR = "#e67e22"
SELECTED_COLOR = "#f1c40f"

THUMB_SIZE = 140
GRID_PADDING = 4
CELL_W = THUMB_SIZE + 10 + GRID_PADDING
CELL_H = THUMB_SIZE + 28 + GRID_PADDING
GRID_MARGIN = 8
HISTORY_SIZE = 40
SLIDE_DURATION_MS = 300
DEFAULT_MONITOR_INTERVAL = 2.0

STATUS_STYLES = {
    "LIVE": ("LIVE", "#2ecc71"),
    "MUTED": ("MUTED", "#e74c3c"),
    "OVERRIDE": ("OVERRIDE", "#e67e22"),
    "PAUSED": ("PAUSED", "#e67e22"),
    "DISCONNECTED": ("NO EXTENSION", "#95a5a6"),
}


# ── Signal bridge (server thread → GUI thread) ───────────────────────────────

class SignalBridge(QObject):
    """Emits Qt signals from non-GUI threads so slots run on the main thread."""
    frame_received = pyqtSignal(object)       # FrameData
    override_received = pyqtSignal(bool)      # muted
    connection_changed = pyqtSignal(bool)     # connected


# ── Thumbnail widget ──────────────────────────────────────────────────────────

class FrameThumb(QFrame):
    """A single thumbnail in the monitor grid."""

    clicked = pyqtSignal(str)  # emits frame path

    def __init__(self, result: FrameResult, selected: bool = False, parent=None):
        super().__init__(parent)
        self.result = result
        self.setFixedSize(THUMB_SIZE + 10, THUMB_SIZE + 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        display_label = result.user_label if result.relabeled else result.pred_label
        uncertain = result.uncertain and not result.relabeled

        if selected:
            border_color = SELECTED_COLOR
        elif uncertain:
            border_color = UNCERTAIN_COLOR
        else:
            border_color = LABEL_COLORS.get(display_label, "#7f8c8d")

        border_width = 3 if selected else 2
        self.setStyleSheet(
            f"FrameThumb {{ border: {border_width}px solid {border_color}; "
            f"border-radius: 4px; background: #d5d8dc; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # Thumbnail image
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setFixedSize(THUMB_SIZE, THUMB_SIZE - 10)

        try:
            pixmap = QPixmap(result.path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    THUMB_SIZE, THUMB_SIZE - 10,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                img_label.setPixmap(pixmap)
            else:
                img_label.setText("[error]")
        except Exception:
            img_label.setText("[error]")

        layout.addWidget(img_label)

        # Caption
        corrected = "  \u2713" if result.relabeled else ""
        caption = f"{display_label}  {result.prob_ad:.0%}{corrected}"
        cap_label = QLabel(caption)
        cap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap_label.setFont(QFont("Helvetica", 9))
        cap_label.setStyleSheet(
            f"background: {border_color}; color: white; "
            f"border-radius: 2px; padding: 1px;"
        )
        layout.addWidget(cap_label)

    def mousePressEvent(self, event):
        self.clicked.emit(self.result.path)


class PlaceholderThumb(QFrame):
    """An empty dashed-border cell for unfilled grid positions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(THUMB_SIZE + 10, THUMB_SIZE + 28)
        self.setStyleSheet(
            "PlaceholderThumb { border: 2px dashed #bdc3c7; "
            "border-radius: 4px; background: transparent; }"
        )


# ── Main Window ───────────────────────────────────────────────────────────────

class HushApp(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Hush Monitor")
        self.resize(960, 480)

        self._store = LabelStore(args.db)

        # Frame history (deque of FrameResult)
        self._history: deque[FrameResult] = deque(maxlen=args.history_size)
        self._selected: Optional[str] = None  # path of selected frame

        # Pipeline (None until model is loaded)
        self._pipeline: Optional[FramePipeline] = None
        self._model_name: str = ""

        # Grid rendering state
        self._grid_children: list[QWidget] = []
        self._animations: list[QPropertyAnimation] = []
        self._prev_n_placeholders: int = -1  # sentinel for first render

        # Track override state for passing to pipeline
        self._pending_user_unmuted = False
        self._pending_user_muted = False

        # Signal bridge for thread-safe GUI updates
        self._bridge = SignalBridge()
        self._bridge.frame_received.connect(self._on_frame_received)
        self._bridge.override_received.connect(self._on_override)
        self._bridge.connection_changed.connect(self._on_connection_changed)

        # Server
        self._server = HushServer(
            port=args.port,
            on_frame=lambda fd: self._bridge.frame_received.emit(fd),
            on_override=lambda m: self._bridge.override_received.emit(m),
            on_connection_change=lambda c: self._bridge.connection_changed.emit(c),
        )

        self._build_ui()
        self._load_model(args.model)
        self._server.start()
        self._update_status()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setStyleSheet("background: #1a252f;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 8, 16, 8)

        self._status_label = QLabel("DISCONNECTED")
        self._status_label.setFont(QFont("Helvetica", 20, QFont.Weight.Bold))
        self._status_label.setStyleSheet("color: #95a5a6;")
        header_layout.addWidget(self._status_label)

        self._info_label = QLabel("")
        self._info_label.setFont(QFont("Helvetica", 11))
        self._info_label.setStyleSheet("color: #95a5a6;")
        header_layout.addWidget(self._info_label, stretch=1)

        self._pause_btn = QPushButton("\u23f8  Pause")
        self._pause_btn.setFont(QFont("Helvetica", 12))
        self._pause_btn.setStyleSheet(
            "QPushButton { background: #2c3e50; color: white; "
            "border: none; border-radius: 6px; padding: 6px 14px; }"
            "QPushButton:hover { background: #34495e; }"
        )
        self._pause_btn.clicked.connect(self._toggle_pause)
        header_layout.addWidget(self._pause_btn)

        main_layout.addWidget(header)

        # Thumbnail grid — manual positioning inside a scroll area so we
        # can animate slide-downs when a new row is created.
        self._scroll_area = QScrollArea()
        self._scroll_area.setStyleSheet(
            "QScrollArea { background: #ecf0f1; border: none; }"
        )
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._grid_surface = QWidget()
        self._grid_surface.setStyleSheet("background: #ecf0f1;")
        self._scroll_area.setWidget(self._grid_surface)
        main_layout.addWidget(self._scroll_area, stretch=1)

        # Legend bar
        legend = QWidget()
        legend.setStyleSheet("background: #ecf0f1;")
        legend_layout = QHBoxLayout(legend)
        legend_layout.setContentsMargins(12, 4, 12, 4)
        legend_label = QLabel(
            "Click to select  \u00b7  A = ad   C = content   "
            "N = neither   X = ignore   Esc = deselect"
        )
        legend_label.setFont(QFont("Helvetica", 10))
        legend_label.setStyleSheet("color: #7f8c8d;")
        legend_layout.addWidget(legend_label)
        main_layout.addWidget(legend)

        # Status bar
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        # Menu bar — model selection
        menu = self.menuBar()
        model_menu = menu.addMenu("Model")
        load_action = QAction("Load Model...", self)
        load_action.triggered.connect(self._choose_model)
        model_menu.addAction(load_action)

    # ── Key bindings ──────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.text().lower()
        label_map = {"a": "ad", "c": "content", "n": "neither", "x": "ignore"}

        if key in label_map and self._selected:
            self._relabel_selected(label_map[key])
        elif event.key() == Qt.Key.Key_Escape:
            self._deselect()
        else:
            super().keyPressEvent(event)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, model_path: str):
        if not os.path.isfile(model_path):
            logger.warning("Model not found: %s", model_path)
            self._statusbar.showMessage(f"Model not found: {model_path}")
            return

        try:
            self._pipeline = FramePipeline(
                model_path=model_path,
                store=self._store,
                hard_frame_dir=self.args.hard_frame_dir,
                mute_delay=self.args.mute_delay,
                unmute_delay=self.args.unmute_delay,
            )
            self._model_name = os.path.basename(model_path)
            self._statusbar.showMessage(f"Model: {self._model_name}")
            logger.info("Loaded model: %s", model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            self._statusbar.showMessage(f"Model error: {e}")

    def _choose_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "models/",
            "Joblib files (*.joblib);;All files (*)",
        )
        if path:
            self._load_model(path)

    # ── Server callbacks (run on main thread via signals) ─────────────────────

    def _on_frame_received(self, frame_data: FrameData):
        if not self._pipeline:
            logger.warning("Frame received but no model loaded — dropping")
            return
        if self._pipeline.fsm._paused:
            return

        result = self._pipeline.process_frame(
            image_bytes=frame_data.image_bytes,
            timestamp=frame_data.timestamp,
            user_unmuted=self._pending_user_unmuted,
            user_muted=self._pending_user_muted,
        )
        self._pending_user_unmuted = False
        self._pending_user_muted = False

        # Evict oldest frame if needed
        if len(self._history) == self._history.maxlen:
            old = self._history[0]
            self._pipeline.delete_temp_file(old)

        self._history.append(result)

        logger.info(
            "Frame #%d: pred=%s  P(ad)=%.1f%%  uncertain=%s  fsm_action=%s",
            len(self._history), result.pred_label, result.prob_ad * 100,
            result.uncertain, result.fsm_action,
        )

        # Act on FSM decision
        if result.fsm_action == "mute":
            self._server.mute()
        elif result.fsm_action == "unmute":
            self._server.unmute()

        self._render_grid()
        self._update_status()

    def _on_override(self, muted: bool):
        if muted:
            self._pending_user_muted = True
        else:
            self._pending_user_unmuted = True

    def _on_connection_changed(self, connected: bool):
        logger.info("Extension %s", "connected" if connected else "disconnected")
        self._update_status()
        if connected:
            if self._pipeline:
                logger.info(
                    "Telling extension to capture every %.1fs", self.args.interval
                )
                self._server.set_mode(
                    "active", interval_sec=self.args.interval
                )
            else:
                logger.warning("Extension connected but no model loaded")
        else:
            # Extension disconnected — clear pending overrides
            self._pending_user_unmuted = False
            self._pending_user_muted = False

    # ── Pause/resume ──────────────────────────────────────────────────────────

    def _toggle_pause(self):
        if not self._pipeline:
            return

        if self._pipeline.fsm._paused:
            self._pipeline.fsm.resume()
            self._server.set_mode("active", interval_sec=self.args.interval)
        else:
            self._pipeline.fsm.pause()
            self._server.set_mode("idle")
        self._update_status()

    # ── Grid rendering ────────────────────────────────────────────────────────

    def _render_grid(self):
        # Stop running animations and remove old widgets
        for anim in self._animations:
            anim.stop()
        self._animations.clear()
        for w in self._grid_children:
            w.setParent(None)
            w.deleteLater()
        self._grid_children.clear()

        grid_width = max(self._scroll_area.viewport().width(), 600)
        n_cols = max(1, (grid_width - GRID_MARGIN * 2) // CELL_W)

        # Layout: newest frames at top, oldest at bottom.  Within each row,
        # time runs left-to-right.  The top row fills right-to-left so new
        # frames slot in without shifting existing ones sideways.  Empty
        # top-row cells get dashed placeholders.  A new placeholder row is
        # created only when the first frame that needs it arrives.
        n = len(self._history)
        if n == 0:
            total_cells = n_cols
        elif n % n_cols == 0:
            total_cells = n
        else:
            total_cells = ((n - 1) // n_cols + 1) * n_cols
        n_placeholders = total_cells - n

        # Detect slide-down: previous render had a full top row (0
        # placeholders) and this render created a new row of them.
        slide_down = (
            self._prev_n_placeholders == 0
            and n_placeholders > 0
            and n > 1
        )
        self._prev_n_placeholders = n_placeholders

        frames = list(reversed(self._history))  # newest first
        n_rows = max(1, total_cells // n_cols)
        self._grid_surface.setMinimumHeight(
            GRID_MARGIN * 2 + n_rows * CELL_H
        )

        for cell in range(total_cells):
            row, col = divmod(cell, n_cols)
            x = GRID_MARGIN + col * CELL_W
            y = GRID_MARGIN + row * CELL_H

            if cell < n_placeholders:
                widget = PlaceholderThumb(parent=self._grid_surface)
                widget.move(x, y)
            else:
                result = frames[cell - n_placeholders]
                widget = FrameThumb(
                    result,
                    selected=(result.path == self._selected),
                    parent=self._grid_surface,
                )
                widget.clicked.connect(self._on_thumb_clicked)

                if slide_down:
                    # Animate from one row up to final position
                    start = QPoint(x, y - CELL_H)
                    widget.move(start)
                    anim = QPropertyAnimation(widget, b"pos")
                    anim.setDuration(SLIDE_DURATION_MS)
                    anim.setStartValue(start)
                    anim.setEndValue(QPoint(x, y))
                    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
                    anim.start()
                    self._animations.append(anim)
                else:
                    widget.move(x, y)

            widget.show()
            self._grid_children.append(widget)

    def _on_thumb_clicked(self, path: str):
        if self._selected == path:
            self._deselect()
        else:
            self._selected = path
            self._render_grid()

    def _deselect(self):
        self._selected = None
        self._render_grid()

    # ── Relabeling ────────────────────────────────────────────────────────────

    def _relabel_selected(self, label: str):
        if not self._selected or not self._pipeline:
            return

        result = next(
            (r for r in self._history if r.path == self._selected), None
        )
        if result is None or result.relabeled:
            return

        self._pipeline.relabel_frame(result, label)
        self._selected = None
        self._render_grid()

    # ── Status display ────────────────────────────────────────────────────────

    def _update_status(self):
        if not self._server.extension.connected:
            state_key = "DISCONNECTED"
        elif self._pipeline and self._pipeline.fsm._paused:
            state_key = "PAUSED"
        elif self._pipeline and self._pipeline.fsm.override:
            state_key = "OVERRIDE"
        elif self._pipeline and self._pipeline.fsm.muted:
            state_key = "MUTED"
        else:
            state_key = "LIVE"

        text, color = STATUS_STYLES[state_key]
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")

        # Pause button text
        if self._pipeline and self._pipeline.fsm._paused:
            self._pause_btn.setText("\u25b6  Resume")
            self._pause_btn.setStyleSheet(
                "QPushButton { background: #c0392b; color: white; "
                "border: none; border-radius: 6px; padding: 6px 14px; }"
                "QPushButton:hover { background: #e74c3c; }"
            )
        else:
            self._pause_btn.setText("\u23f8  Pause")
            self._pause_btn.setStyleSheet(
                "QPushButton { background: #2c3e50; color: white; "
                "border: none; border-radius: 6px; padding: 6px 14px; }"
                "QPushButton:hover { background: #34495e; }"
            )

        # Info label
        if self._pipeline and self._history:
            last = list(self._history)[-1]
            self._info_label.setText(
                f"last: {last.pred_label} ({last.prob_ad:.0%} ad)"
                f"  \u00b7  ad streak: {self._pipeline.fsm.ad_streak}"
                f"  \u00b7  ok streak: {self._pipeline.fsm.ok_streak}"
            )

        # Status bar — connection info + model name
        conn = "Connected" if self._server.extension.connected else "Disconnected"
        ext_mode = self._server.extension.mode
        model = self._model_name or "none"
        self._statusbar.showMessage(
            f"Extension: {conn}  \u00b7  Mode: {ext_mode}  \u00b7  "
            f"Frames: {len(self._history)}  \u00b7  Model: {model}"
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Unmute before exit if we had it muted
        if self._pipeline and self._pipeline.fsm.muted:
            self._server.unmute()

        # Clean up temp files
        if self._pipeline:
            for result in self._history:
                self._pipeline.delete_temp_file(result)
            self._pipeline.cleanup()

        self._server.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hush v2 — ad muting monitor with Chrome extension support.",
    )
    parser.add_argument(
        "--model", default="models/classifier.joblib",
        help="Trained model bundle (default: models/classifier.joblib)",
    )
    parser.add_argument(
        "--db", default="hush.db",
        help="SQLite database (default: hush.db)",
    )
    parser.add_argument(
        "--port", type=int, default=9099,
        help="Server port for extension communication (default: 9099)",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_MONITOR_INTERVAL,
        help="Capture interval in seconds (default: 2)",
    )
    parser.add_argument(
        "--mute-delay", type=int, default=2, dest="mute_delay",
        help="Consecutive ad frames before muting (default: 2)",
    )
    parser.add_argument(
        "--unmute-delay", type=int, default=3, dest="unmute_delay",
        help="Consecutive content frames before unmuting (default: 3)",
    )
    parser.add_argument(
        "--history-size", type=int, default=HISTORY_SIZE, dest="history_size",
        help=f"Frames in rolling display (default: {HISTORY_SIZE})",
    )
    parser.add_argument(
        "--hard-frame-dir", default="hard_frames", dest="hard_frame_dir",
        help="Where to save relabeled frames (default: hard_frames/)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Hush")

    # Let Ctrl+C close the app gracefully. Qt's event loop swallows SIGINT
    # by default; a periodic timer gives Python a chance to run the handler.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    timer = QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    window = HushApp(args)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
