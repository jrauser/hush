"""Hush v2 Desktop App — PyQt6 GUI with Live Monitor and Bulk Capture views.

Entry point: python -m hush.v2app

Launches the FastAPI server in a background thread, connects to the Chrome
extension, and provides two modes:
  - Live Monitoring: real-time frame classification and mute/unmute control
  - Bulk Capture: paginated labeling grid for training data
"""

import argparse
import logging
import math
import os
import signal
import sys
from collections import deque
from typing import Optional

from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QEasingCurve, QPropertyAnimation,
    pyqtSignal, QObject,
)
from PyQt6.QtGui import QPixmap, QFont, QAction, QKeySequence, QShortcut
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
    QComboBox,
    QLineEdit,
    QSpinBox,
    QDialog,
)

from hush.frame_pipeline import FramePipeline, FrameResult
from hush.label_store import LabelStore
from hush.selection import SelectionManager
from hush.server import HushServer, FrameData

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "ad": "#c0392b",
    "content": "#27ae60",
    "neither": "#7f8c8d",
    "ignore": "#8e44ad",
    None: "#bdc3c7",
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
DEFAULT_BULK_INTERVAL = 7

STATUS_STYLES = {
    "LIVE": ("LIVE", "#2ecc71"),
    "MUTED": ("MUTED", "#e74c3c"),
    "OVERRIDE": ("OVERRIDE", "#e67e22"),
    "PAUSED": ("PAUSED", "#e67e22"),
    "DISCONNECTED": ("NO EXTENSION", "#95a5a6"),
    "CAPTURING": ("CAPTURING", "#3498db"),
    "IDLE": ("IDLE", "#95a5a6"),
}

MODE_LIVE = "live"
MODE_BULK = "bulk"

FILTER_OPTIONS = ["All", "Unlabeled", "Ad", "Content", "Neither", "Ignore"]
FILTER_TO_LABEL = {
    "All": None,
    "Unlabeled": "unlabeled",
    "Ad": "ad",
    "Content": "content",
    "Neither": "neither",
    "Ignore": "ignore",
}


# ── Signal bridge (server thread → GUI thread) ───────────────────────────────

class SignalBridge(QObject):
    """Emits Qt signals from non-GUI threads so slots run on the main thread."""
    frame_received = pyqtSignal(object)       # FrameData
    override_received = pyqtSignal(bool)      # muted
    connection_changed = pyqtSignal(bool)     # connected


# ── Thumbnail widget ──────────────────────────────────────────────────────────

class FrameThumb(QFrame):
    """A single thumbnail in the grid. Works for both Live and Bulk modes.

    For Live mode, displays classifier prediction and confidence.
    For Bulk mode, displays the user-assigned label (or 'unlabeled').
    """

    clicked = pyqtSignal(int, bool)  # emits (index, shift_held)

    def __init__(self, result: FrameResult, idx: int,
                 selected: bool = False, parent=None):
        super().__init__(parent)
        self.result = result
        self._idx = idx
        self.setFixedSize(THUMB_SIZE + 10, THUMB_SIZE + 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        display_label = result.user_label if result.relabeled else result.pred_label
        uncertain = result.uncertain and not result.relabeled

        if selected:
            border_color = SELECTED_COLOR
        elif uncertain:
            border_color = UNCERTAIN_COLOR
        else:
            border_color = LABEL_COLORS.get(display_label, "#bdc3c7")

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
                img_label.setText("[missing]")
        except Exception:
            img_label.setText("[error]")

        layout.addWidget(img_label)

        # Caption
        corrected = "  \u2713" if result.relabeled else ""
        if display_label and result.prob_ad is not None:
            caption = f"{display_label}  {result.prob_ad:.0%}{corrected}"
        elif display_label:
            caption = f"{display_label}{corrected}"
        else:
            caption = "unlabeled"
        cap_label = QLabel(caption)
        cap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap_label.setFont(QFont("Helvetica", 9))
        cap_label.setStyleSheet(
            f"background: {border_color}; color: white; "
            f"border-radius: 2px; padding: 1px;"
        )
        layout.addWidget(cap_label)

    def mousePressEvent(self, event):
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self.clicked.emit(self._idx, shift)


class PlaceholderThumb(QFrame):
    """An empty dashed-border cell for unfilled grid positions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(THUMB_SIZE + 10, THUMB_SIZE + 28)
        self.setStyleSheet(
            "PlaceholderThumb { border: 2px dashed #bdc3c7; "
            "border-radius: 4px; background: transparent; }"
        )


# ── Zoom window ──────────────────────────────────────────────────────────────

class ZoomWindow(QDialog):
    """Full-size image viewer with arrow-key navigation and relabeling."""

    relabeled = pyqtSignal(int, str)    # (index_in_items, label)
    deleted = pyqtSignal(int)           # index_in_items

    def __init__(self, items: list[FrameResult], start_idx: int, parent=None):
        super().__init__(parent)
        self._items = items
        self._idx = start_idx
        self.setWindowTitle("Zoom")
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet("background: black;")
        layout.addWidget(self._img_label, stretch=1)

        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setFont(QFont("Helvetica", 11))
        self._info_label.setStyleSheet(
            "padding: 6px; background: #1a252f; color: #ecf0f1;"
        )
        layout.addWidget(self._info_label)

        self._show(self._idx)

    def _show(self, idx: int):
        if idx < 0 or idx >= len(self._items):
            return
        self._idx = idx
        item = self._items[idx]

        pixmap = QPixmap(item.path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self._img_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._img_label.setPixmap(scaled)
        else:
            self._img_label.setText("[cannot load image]")

        label = item.user_label or item.pred_label or "unlabeled"
        fname = os.path.basename(item.path)
        self._info_label.setText(f"{fname}  [{label}]")
        self.setWindowTitle(f"{fname}  [{label}]")

    def keyPressEvent(self, event):
        key = event.text().lower()
        label_map = {"a": "ad", "c": "content", "n": "neither", "x": "ignore"}

        if key in label_map:
            self.relabeled.emit(self._idx, label_map[key])
            self.close()
        elif key == "d":
            self.deleted.emit(self._idx)
            # Advance or close
            if self._idx < len(self._items) - 1:
                self._show(self._idx)  # items list was modified externally
            elif self._idx > 0:
                self._show(self._idx - 1)
            else:
                self.close()
        elif event.key() == Qt.Key.Key_Left:
            self._show(self._idx - 1)
        elif event.key() == Qt.Key.Key_Right:
            self._show(self._idx + 1)
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


# ── Main Window ───────────────────────────────────────────────────────────────

class HushApp(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Hush")
        self.resize(960, 540)

        self._store = LabelStore(args.db)
        self._mode = MODE_LIVE

        # Selection manager (shared between modes)
        self._sel = SelectionManager()

        # ── Live mode state ──────────────────────────────────────────────
        self._history: deque[FrameResult] = deque(maxlen=args.history_size)
        self._pipeline: Optional[FramePipeline] = None
        self._model_name: str = ""
        self._pending_user_unmuted = False
        self._pending_user_muted = False

        # ── Bulk mode state ──────────────────────────────────────────────
        self._bulk_all_rows: list[dict] = []
        self._bulk_filtered_rows: list[dict] = []
        self._bulk_page: int = 0
        self._bulk_capturing: bool = False
        self._bulk_session_count: int = 0
        self._bulk_capture_items: list[FrameResult] = []  # session-local list for stable grid
        self._undo_stack: list[tuple[list[str], dict[str, Optional[str]]]] = []

        # ── Grid rendering state ─────────────────────────────────────────
        self._grid_children: list[QWidget] = []
        self._animations: list[QPropertyAnimation] = []
        self._prev_n_placeholders: int = -1  # sentinel for first render

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

        # Mode button bar — both always visible, active one highlighted
        mode_bar = QWidget()
        mode_bar.setStyleSheet("background: transparent;")
        mode_bar_layout = QHBoxLayout(mode_bar)
        mode_bar_layout.setContentsMargins(0, 0, 0, 0)
        mode_bar_layout.setSpacing(0)

        self._live_btn = QPushButton("Live Monitoring")
        self._live_btn.setFont(QFont("Helvetica", 11))
        self._live_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._live_btn.clicked.connect(self._switch_to_live)
        mode_bar_layout.addWidget(self._live_btn)

        self._bulk_btn = QPushButton("Bulk Capture")
        self._bulk_btn.setFont(QFont("Helvetica", 11))
        self._bulk_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._bulk_btn.clicked.connect(self._switch_to_bulk)
        mode_bar_layout.addWidget(self._bulk_btn)

        header_layout.addWidget(mode_bar)
        self._update_mode_buttons()

        # Pause / Start-Stop button
        self._action_btn = QPushButton("\u23f8  Pause")
        self._action_btn.setFont(QFont("Helvetica", 12))
        self._action_btn.setStyleSheet(
            "QPushButton { background: #2c3e50; color: white; "
            "border: none; border-radius: 6px; padding: 6px 14px; }"
            "QPushButton:hover { background: #34495e; }"
        )
        self._action_btn.clicked.connect(self._on_action_btn)
        header_layout.addWidget(self._action_btn)

        main_layout.addWidget(header)

        # ── Bulk controls bar (hidden in Live mode) ──────────────────────
        self._bulk_controls = QWidget()
        self._bulk_controls.setStyleSheet("background: #d5d8dc;")
        bc_layout = QHBoxLayout(self._bulk_controls)
        bc_layout.setContentsMargins(12, 4, 12, 4)

        bc_layout.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(FILTER_OPTIONS)
        self._filter_combo.setFixedWidth(100)
        self._filter_combo.currentTextChanged.connect(self._apply_bulk_filter)
        bc_layout.addWidget(self._filter_combo)

        bc_layout.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setFixedWidth(180)
        self._search_input.returnPressed.connect(self._apply_bulk_filter)
        bc_layout.addWidget(self._search_input)

        search_btn = QPushButton("Go")
        search_btn.setFixedWidth(40)
        search_btn.clicked.connect(self._apply_bulk_filter)
        bc_layout.addWidget(search_btn)

        clear_btn = QPushButton("\u2715")
        clear_btn.setFixedWidth(28)
        clear_btn.clicked.connect(self._clear_search)
        bc_layout.addWidget(clear_btn)

        bc_layout.addStretch()

        bc_layout.addWidget(QLabel("Interval:"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 60)
        self._interval_spin.setValue(DEFAULT_BULK_INTERVAL)
        self._interval_spin.setSuffix("s")
        self._interval_spin.setFixedWidth(70)
        self._interval_spin.valueChanged.connect(self._on_interval_changed)
        bc_layout.addWidget(self._interval_spin)

        bc_layout.addStretch()

        self._btn_prev = QPushButton("\u2039 Prev")
        self._btn_prev.clicked.connect(self._prev_page)
        bc_layout.addWidget(self._btn_prev)

        self._page_label = QLabel("Page 1/1")
        self._page_label.setFont(QFont("Helvetica", 10))
        bc_layout.addWidget(self._page_label)

        self._btn_next = QPushButton("Next \u203a")
        self._btn_next.clicked.connect(self._next_page)
        bc_layout.addWidget(self._btn_next)

        self._bulk_controls.setVisible(False)
        main_layout.addWidget(self._bulk_controls)

        # ── Thumbnail grid ───────────────────────────────────────────────
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

        # ── Legend bar ───────────────────────────────────────────────────
        legend = QWidget()
        legend.setStyleSheet("background: #ecf0f1;")
        legend_layout = QHBoxLayout(legend)
        legend_layout.setContentsMargins(12, 4, 12, 4)
        self._legend_label = QLabel()
        self._legend_label.setFont(QFont("Helvetica", 10))
        self._legend_label.setStyleSheet("color: #7f8c8d;")
        legend_layout.addWidget(self._legend_label)
        main_layout.addWidget(legend)
        self._update_legend()

        # Status bar
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        # Menu bar
        menu = self.menuBar()
        model_menu = menu.addMenu("Model")
        load_action = QAction("Load Model...", self)
        load_action.triggered.connect(self._choose_model)
        model_menu.addAction(load_action)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo)

    # ── Legend ─────────────────────────────────────────────────────────────────

    def _update_legend(self):
        if self._mode == MODE_LIVE:
            self._legend_label.setText(
                "Click / Shift+click to select  \u00b7  "
                "A = ad   C = content   N = neither   X = ignore  \u00b7  "
                "Z = zoom   Cmd+A = select all   Esc = deselect"
            )
        else:
            self._legend_label.setText(
                "Click / Shift+click to select  \u00b7  "
                "A = ad   C = content   N = neither   X = ignore   D = delete  \u00b7  "
                "Z = zoom   Cmd+A = select all   Esc = deselect  \u00b7  "
                "\u2190/\u2192 = page   Ctrl+Z = undo"
            )

    def _update_mode_buttons(self):
        active_style = (
            "QPushButton { background: #2980b9; color: white; "
            "border: none; border-radius: 6px 0 0 6px; padding: 6px 14px; }"
            "QPushButton:hover { background: #3498db; }"
        )
        inactive_style = (
            "QPushButton { background: #2c3e50; color: #95a5a6; "
            "border: none; border-radius: 0 6px 6px 0; padding: 6px 14px; }"
            "QPushButton:hover { background: #34495e; color: white; }"
        )
        # Swap border-radius depending on which is active
        if self._mode == MODE_LIVE:
            self._live_btn.setStyleSheet(active_style)
            self._bulk_btn.setStyleSheet(inactive_style)
        else:
            self._live_btn.setStyleSheet(
                inactive_style.replace("0 6px 6px 0", "6px 0 0 6px")
            )
            self._bulk_btn.setStyleSheet(
                active_style.replace("6px 0 0 6px", "0 6px 6px 0")
            )

    # ── Key bindings ──────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        # Don't handle keys when a child widget (search box, spinbox) has focus
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QSpinBox)):
            super().keyPressEvent(event)
            return

        key = event.text().lower()
        label_map = {"a": "ad", "c": "content", "n": "neither", "x": "ignore"}

        if key in label_map and self._sel.count() > 0:
            self._relabel_selected(label_map[key])
        elif key == "d" and self._mode == MODE_BULK and self._sel.count() > 0:
            self._delete_selected()
        elif key == "z":
            self._open_zoom_for_last_clicked()
        elif event.key() == Qt.Key.Key_Escape:
            self._sel.clear()
            self._render_grid()
        elif event.key() == Qt.Key.Key_Left and self._mode == MODE_BULK:
            self._prev_page()
        elif event.key() == Qt.Key.Key_Right and self._mode == MODE_BULK:
            self._next_page()
        elif (event.modifiers() & Qt.KeyboardModifier.ControlModifier
              and event.key() == Qt.Key.Key_A):
            self._select_all_visible()
        else:
            super().keyPressEvent(event)

    def _select_all_visible(self):
        keys = self._visible_keys()
        self._sel.select_all(keys)
        self._render_grid()

    # ── Mode switching ────────────────────────────────────────────────────────

    def _switch_to_bulk(self):
        if self._mode == MODE_BULK:
            return
        # Stop bulk capture if somehow running (shouldn't be, but defensive)
        self._bulk_capturing = False
        self._bulk_capture_items.clear()

        # Tear down live state
        self._teardown_live()

        self._mode = MODE_BULK
        self._sel.clear()
        self._update_mode_buttons()
        self._bulk_controls.setVisible(True)
        self._update_legend()

        # Tell extension to idle (bulk starts in stopped state)
        if self._server.extension.connected:
            self._server.set_mode("idle")

        self._reload_bulk_data()
        self._render_grid()
        QTimer.singleShot(0, self._render_grid)
        self._update_status()

    def _switch_to_live(self):
        if self._mode == MODE_LIVE:
            return
        # Stop bulk capture if running
        if self._bulk_capturing:
            self._bulk_capturing = False

        self._mode = MODE_LIVE
        self._sel.clear()
        self._prev_n_placeholders = -1
        self._update_mode_buttons()
        self._bulk_controls.setVisible(False)
        self._update_legend()

        # Fresh FSM + pipeline
        if self._pipeline:
            self._pipeline.cleanup()
        self._history.clear()
        self._load_model(self.args.model)

        # Tell extension to start monitoring
        if self._server.extension.connected and self._pipeline:
            self._server.set_mode("active", interval_sec=self.args.interval)

        self._render_grid()
        # Deferred re-render: the scroll area may not have its final size yet
        # after the bulk controls bar is hidden, so re-render once layout settles.
        QTimer.singleShot(0, self._render_grid)
        self._update_status()

    def _teardown_live(self):
        """Clean up live monitor state (FSM, history, temp files)."""
        if self._pipeline:
            for result in self._history:
                self._pipeline.delete_temp_file(result)
            self._pipeline.cleanup()
            self._pipeline = None
        self._history.clear()
        self._pending_user_unmuted = False
        self._pending_user_muted = False

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
                screenshot_dir=self.args.screenshot_dir,
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

    # ── Action button (Pause/Resume in Live, Start/Stop in Bulk) ─────────────

    def _on_action_btn(self):
        if self._mode == MODE_LIVE:
            self._toggle_pause_live()
        else:
            self._toggle_capture_bulk()

    def _toggle_pause_live(self):
        if not self._pipeline:
            return
        if self._pipeline.fsm._paused:
            self._pipeline.fsm.resume()
            self._server.set_mode("active", interval_sec=self.args.interval)
        else:
            self._pipeline.fsm.pause()
            self._server.set_mode("idle")
        self._update_status()

    def _toggle_capture_bulk(self):
        if self._bulk_capturing:
            self._bulk_capturing = False
            self._server.set_mode("idle")
            # Transition to browse grid backed by full DB
            self._reload_bulk_data()
        else:
            if not self._server.extension.connected:
                logger.warning("Cannot start capture: extension not connected")
                return
            self._bulk_capturing = True
            self._bulk_capture_items.clear()
            self._prev_n_placeholders = -1
            interval = self._interval_spin.value()
            self._server.set_mode("active", interval_sec=float(interval))
        self._render_grid()
        self._update_status()

    def _on_interval_changed(self, value: int):
        """Update capture interval while capturing."""
        if self._mode == MODE_BULK and self._bulk_capturing:
            self._server.set_mode("active", interval_sec=float(value))

    # ── Server callbacks (run on main thread via signals) ─────────────────────

    def _on_frame_received(self, frame_data: FrameData):
        if self._mode == MODE_LIVE:
            self._on_frame_live(frame_data)
        else:
            self._on_frame_bulk(frame_data)

    def _on_frame_live(self, frame_data: FrameData):
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

    def _on_frame_bulk(self, frame_data: FrameData):
        if not self._bulk_capturing:
            return

        # Save to screenshots/
        from datetime import datetime
        fname = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{id(frame_data):08x}.jpg"
        path = os.path.join(self.args.screenshot_dir, fname)
        os.makedirs(self.args.screenshot_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(frame_data.image_bytes)

        timestamp = frame_data.timestamp or datetime.now().isoformat()
        self._store.add_screenshot(path, timestamp, source="extension")
        self._bulk_session_count += 1

        # Append to session-local list (not a full DB reload) so existing
        # thumbnails don't shift. The capturing grid renders from this list.
        self._bulk_capture_items.append(FrameResult(
            path=path,
            captured_at=timestamp,
            pred_label=None,
            prob_ad=None,
            probs=None,
            uncertain=False,
            fsm_action=None,
            relabeled=False,
            user_label=None,
        ))

        self._render_grid()
        self._update_status()

    def _on_override(self, muted: bool):
        if muted:
            self._pending_user_muted = True
        else:
            self._pending_user_unmuted = True

    def _on_connection_changed(self, connected: bool):
        logger.info("Extension %s", "connected" if connected else "disconnected")
        if connected and self._mode == MODE_LIVE and self._pipeline:
            logger.info(
                "Telling extension to capture every %.1fs", self.args.interval
            )
            self._server.set_mode("active", interval_sec=self.args.interval)
        elif not connected:
            self._pending_user_unmuted = False
            self._pending_user_muted = False
            if self._bulk_capturing:
                self._bulk_capturing = False
        self._update_status()

    # ── Bulk data loading ─────────────────────────────────────────────────────

    def _reload_bulk_data(self, clear_selection: bool = True):
        self._bulk_all_rows = self._store.get_all()
        # Auto-purge missing files
        missing = [r["filename"] for r in self._bulk_all_rows
                   if not os.path.isfile(r["filename"])]
        if missing:
            self._store.delete_screenshots(missing)
            missing_set = set(missing)
            self._bulk_all_rows = [
                r for r in self._bulk_all_rows if r["filename"] not in missing_set
            ]
        self._apply_bulk_filter(clear_selection=clear_selection)

    def _apply_bulk_filter(self, _text=None, clear_selection: bool = True):
        fval = self._filter_combo.currentText()
        search = self._search_input.text().strip()

        rows = self._bulk_all_rows
        label_filter = FILTER_TO_LABEL.get(fval)
        if label_filter == "unlabeled":
            rows = [r for r in rows if r["label"] is None]
        elif label_filter is not None:
            rows = [r for r in rows if r["label"] == label_filter]
        if search:
            rows = [r for r in rows if search in r["filename"]]

        self._bulk_filtered_rows = list(reversed(rows))  # newest first
        # Clamp page
        max_page = max(0, self._bulk_total_pages() - 1)
        self._bulk_page = min(self._bulk_page, max_page)
        if clear_selection:
            self._sel.clear()
        self._render_grid()
        self._update_status()

    def _clear_search(self):
        self._search_input.clear()
        self._apply_bulk_filter()

    # ── Bulk pagination ───────────────────────────────────────────────────────

    def _bulk_page_size(self) -> int:
        grid_width = max(self._scroll_area.viewport().width(), 600)
        grid_height = max(self._scroll_area.viewport().height(), 200)
        n_cols = max(1, (grid_width - GRID_MARGIN * 2) // CELL_W)
        n_rows = max(1, (grid_height - GRID_MARGIN * 2) // CELL_H)
        return n_cols * n_rows

    def _bulk_total_pages(self) -> int:
        ps = self._bulk_page_size()
        return max(1, math.ceil(len(self._bulk_filtered_rows) / ps))

    def _bulk_current_page_rows(self) -> list[dict]:
        ps = self._bulk_page_size()
        start = self._bulk_page * ps
        return self._bulk_filtered_rows[start:start + ps]

    def _prev_page(self):
        if self._bulk_page > 0:
            self._bulk_page -= 1
            self._sel.clear()
            self._render_grid()
            self._update_status()

    def _next_page(self):
        if self._bulk_page < self._bulk_total_pages() - 1:
            self._bulk_page += 1
            self._sel.clear()
            self._render_grid()
            self._update_status()

    # ── Helpers for visible items ─────────────────────────────────────────────

    def _visible_keys(self) -> list[str]:
        """Return ordered list of paths for currently visible items."""
        if self._mode == MODE_LIVE:
            return [r.path for r in reversed(self._history)]
        elif self._bulk_capturing:
            return [r.path for r in reversed(self._bulk_capture_items)]
        else:
            return [r["filename"] for r in self._bulk_current_page_rows()]

    def _visible_items(self) -> list[FrameResult]:
        """Return ordered list of FrameResult for currently visible items."""
        if self._mode == MODE_LIVE:
            return list(reversed(self._history))
        elif self._bulk_capturing:
            return list(reversed(self._bulk_capture_items))
        else:
            return [self._row_to_result(r) for r in self._bulk_current_page_rows()]

    def _row_to_result(self, row: dict) -> FrameResult:
        """Convert a DB row to a FrameResult for display in FrameThumb."""
        label = row.get("label")
        return FrameResult(
            path=row["filename"],
            captured_at=row.get("captured_at", ""),
            pred_label=label,
            prob_ad=None,
            probs=None,
            uncertain=False,
            fsm_action=None,
            relabeled=label is not None,
            user_label=label,
        )

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

        if self._mode == MODE_LIVE:
            self._render_live_grid()
        elif self._bulk_capturing:
            self._render_bulk_capturing_grid()
        else:
            self._render_bulk_browse_grid()

    def _render_live_grid(self):
        """Live mode: newest at top, fill right-to-left, slide-down animation.

        The top row fills right-to-left: empty cells get dashed placeholders,
        and each new frame replaces the rightmost placeholder. Nothing moves
        sideways. A new placeholder row (with slide-down animation) is created
        only when the previous row is completely full.
        """
        grid_width = max(self._scroll_area.viewport().width(), 600)
        n_cols = max(1, (grid_width - GRID_MARGIN * 2) // CELL_W)

        n = len(self._history)
        if n == 0:
            total_cells = n_cols
        elif n % n_cols == 0:
            total_cells = n
        else:
            total_cells = ((n - 1) // n_cols + 1) * n_cols
        n_placeholders = total_cells - n

        # Slide-down triggers when the previous render had a completely full
        # top row (0 placeholders) and this render created a new row of them.
        # _prev_n_placeholders starts at -1 (sentinel) so the first render
        # never triggers a slide.
        slide_down = (
            self._prev_n_placeholders == 0
            and n_placeholders > 0
            and n > 1
        )
        self._prev_n_placeholders = n_placeholders

        items = self._visible_items()
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
                item_idx = cell - n_placeholders
                result = items[item_idx]
                widget = FrameThumb(
                    result, idx=item_idx,
                    selected=self._sel.is_selected(result.path),
                    parent=self._grid_surface,
                )
                widget.clicked.connect(self._on_thumb_clicked)

                if slide_down:
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

    def _render_bulk_capturing_grid(self):
        """Bulk Capture while capturing: same fill-right-to-left as live mode.

        Uses _bulk_capture_items (session-local append-only list) instead of
        the full DB so that existing thumbnails never shift when a new frame
        arrives. Items are reversed for display (newest first, closest to
        placeholders) — identical to live mode's reversed(self._history).
        """
        grid_width = max(self._scroll_area.viewport().width(), 600)
        n_cols = max(1, (grid_width - GRID_MARGIN * 2) // CELL_W)

        # Reversed so newest is first (index 0), matching live mode.
        # New frames append to _bulk_capture_items, so they appear at
        # index 0 after reversal and fill the rightmost placeholder.
        items = list(reversed(self._bulk_capture_items))
        n = len(items)

        if n == 0:
            total_cells = n_cols
        elif n % n_cols == 0:
            total_cells = n
        else:
            total_cells = ((n - 1) // n_cols + 1) * n_cols
        n_placeholders = total_cells - n

        slide_down = (
            self._prev_n_placeholders == 0
            and n_placeholders > 0
            and n > 1
        )
        self._prev_n_placeholders = n_placeholders

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
                item_idx = cell - n_placeholders
                result = items[item_idx]
                widget = FrameThumb(
                    result, idx=item_idx,
                    selected=self._sel.is_selected(result.path),
                    parent=self._grid_surface,
                )
                widget.clicked.connect(self._on_thumb_clicked)

                if slide_down:
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

    def _render_bulk_browse_grid(self):
        """Bulk Capture browsing: standard left-to-right, top-to-bottom."""
        grid_width = max(self._scroll_area.viewport().width(), 600)
        n_cols = max(1, (grid_width - GRID_MARGIN * 2) // CELL_W)

        page_rows = self._bulk_current_page_rows()
        items = [self._row_to_result(r) for r in page_rows]

        n_rows = max(1, math.ceil(len(items) / n_cols)) if items else 1
        self._grid_surface.setMinimumHeight(
            GRID_MARGIN * 2 + n_rows * CELL_H
        )

        if not items:
            lbl = QLabel("No images to display.")
            lbl.setFont(QFont("Helvetica", 14))
            lbl.setStyleSheet("color: #7f8c8d;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setParent(self._grid_surface)
            lbl.move(GRID_MARGIN, GRID_MARGIN + 60)
            lbl.resize(grid_width - GRID_MARGIN * 2, 40)
            lbl.show()
            self._grid_children.append(lbl)
            return

        for i, result in enumerate(items):
            row, col = divmod(i, n_cols)
            x = GRID_MARGIN + col * CELL_W
            y = GRID_MARGIN + row * CELL_H

            widget = FrameThumb(
                result, idx=i,
                selected=self._sel.is_selected(result.path),
                parent=self._grid_surface,
            )
            widget.clicked.connect(self._on_thumb_clicked)
            widget.move(x, y)
            widget.show()
            self._grid_children.append(widget)

        # Update page controls
        self._page_label.setText(
            f"Page {self._bulk_page + 1}/{self._bulk_total_pages()}"
        )
        self._btn_prev.setEnabled(self._bulk_page > 0)
        self._btn_next.setEnabled(
            self._bulk_page < self._bulk_total_pages() - 1
        )

    # ── Thumbnail click handling ──────────────────────────────────────────────

    def _on_thumb_clicked(self, idx: int, shift: bool):
        keys = self._visible_keys()
        self._sel.click(idx, keys, shift=shift)
        self._render_grid()

    # ── Relabeling ────────────────────────────────────────────────────────────

    def _relabel_selected(self, label: str):
        if self._sel.count() == 0:
            return

        if self._mode == MODE_LIVE:
            self._relabel_selected_live(label)
        else:
            self._relabel_selected_bulk(label)

    def _relabel_selected_live(self, label: str):
        """Relabel selected frames in Live mode — save to screenshots/ and DB."""
        if not self._pipeline:
            return
        selected = set(self._sel.selected)
        for result in self._history:
            if result.path in selected and not result.relabeled:
                self._pipeline.relabel_frame(result, label)
        self._sel.clear()
        self._render_grid()

    def _relabel_selected_bulk(self, label: str):
        """Relabel selected frames in Bulk mode — update DB labels."""
        selected = set(self._sel.selected)
        filenames = list(selected)

        # Record old labels for undo
        old_labels = {
            r["filename"]: r["label"]
            for r in self._bulk_all_rows
            if r["filename"] in selected
        }
        self._undo_stack.append((filenames, old_labels))

        self._store.set_labels_bulk(filenames, label)

        # Update in-memory rows
        for r in self._bulk_all_rows:
            if r["filename"] in selected:
                r["label"] = label

        self._sel.clear()
        self._apply_bulk_filter()

    def _delete_selected(self):
        """Delete selected frames in Bulk mode — remove from disk and DB."""
        selected = set(self._sel.selected)
        filenames = list(selected)

        for f in filenames:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        self._store.delete_screenshots(filenames)
        self._bulk_all_rows = [
            r for r in self._bulk_all_rows if r["filename"] not in selected
        ]
        self._sel.clear()
        self._apply_bulk_filter()

    def _undo(self):
        """Undo last bulk label action."""
        if self._mode != MODE_BULK or not self._undo_stack:
            return

        filenames, old_labels = self._undo_stack.pop()
        to_clear = [f for f, lbl in old_labels.items() if lbl is None]
        if to_clear:
            self._store.clear_labels_bulk(to_clear)
        for lbl in ("ad", "content", "neither", "ignore"):
            subset = [f for f, old in old_labels.items() if old == lbl]
            if subset:
                self._store.set_labels_bulk(subset, lbl)

        self._reload_bulk_data()

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def _open_zoom_for_last_clicked(self):
        idx = self._sel.last_clicked_idx
        if idx is None:
            return
        items = self._visible_items()
        if idx < 0 or idx >= len(items):
            return
        self._open_zoom(items, idx)

    def _open_zoom(self, items: list[FrameResult], start_idx: int):
        zoom = ZoomWindow(items, start_idx, parent=self)
        zoom.relabeled.connect(
            lambda idx, label: self._on_zoom_relabel(items, idx, label)
        )
        zoom.deleted.connect(
            lambda idx: self._on_zoom_delete(items, idx)
        )
        zoom.exec()

    def _on_zoom_relabel(self, items: list[FrameResult], idx: int, label: str):
        if idx < 0 or idx >= len(items):
            return
        item = items[idx]

        if self._mode == MODE_LIVE:
            if self._pipeline and not item.relabeled:
                self._pipeline.relabel_frame(item, label)
        else:
            self._store.set_label(item.path, label)
            self._reload_bulk_data()

        self._render_grid()

    def _on_zoom_delete(self, items: list[FrameResult], idx: int):
        if self._mode != MODE_BULK:
            return
        if idx < 0 or idx >= len(items):
            return
        item = items[idx]

        try:
            os.remove(item.path)
        except FileNotFoundError:
            pass

        self._store.delete_screenshots([item.path])
        items.pop(idx)

        deleted = {item.path}
        self._bulk_all_rows = [
            r for r in self._bulk_all_rows if r["filename"] not in deleted
        ]
        self._apply_bulk_filter()

    # ── Status display ────────────────────────────────────────────────────────

    def _update_status(self):
        if self._mode == MODE_LIVE:
            self._update_status_live()
        else:
            self._update_status_bulk()

    def _update_status_live(self):
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

        # Action button
        if self._pipeline and self._pipeline.fsm._paused:
            self._action_btn.setText("\u25b6  Resume")
            self._action_btn.setStyleSheet(
                "QPushButton { background: #c0392b; color: white; "
                "border: none; border-radius: 6px; padding: 6px 14px; }"
                "QPushButton:hover { background: #e74c3c; }"
            )
        else:
            self._action_btn.setText("\u23f8  Pause")
            self._action_btn.setStyleSheet(
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
        else:
            self._info_label.setText("")

        # Status bar
        conn = "Connected" if self._server.extension.connected else "Disconnected"
        model = self._model_name or "none"
        self._statusbar.showMessage(
            f"Extension: {conn}  \u00b7  "
            f"Frames: {len(self._history)}  \u00b7  Model: {model}"
        )

    def _update_status_bulk(self):
        if not self._server.extension.connected:
            state_key = "DISCONNECTED"
        elif self._bulk_capturing:
            state_key = "CAPTURING"
        else:
            state_key = "IDLE"

        text, color = STATUS_STYLES[state_key]
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")

        # Action button
        if self._bulk_capturing:
            self._action_btn.setText("\u25a0  Stop Capture")
            self._action_btn.setStyleSheet(
                "QPushButton { background: #c0392b; color: white; "
                "border: none; border-radius: 6px; padding: 6px 14px; }"
                "QPushButton:hover { background: #e74c3c; }"
            )
        else:
            self._action_btn.setText("\u25b6  Start Capture")
            self._action_btn.setStyleSheet(
                "QPushButton { background: #27ae60; color: white; "
                "border: none; border-radius: 6px; padding: 6px 14px; }"
                "QPushButton:hover { background: #2ecc71; }"
            )

        # Info label — label stats
        total = len(self._bulk_all_rows)
        unlabeled = sum(1 for r in self._bulk_all_rows if r["label"] is None)
        n_ad = sum(1 for r in self._bulk_all_rows if r["label"] == "ad")
        n_content = sum(1 for r in self._bulk_all_rows if r["label"] == "content")
        sel_count = self._sel.count()
        sel_text = f"  \u00b7  {sel_count} selected" if sel_count else ""

        self._info_label.setText(
            f"{total} frames  \u00b7  {unlabeled} unlabeled"
            f"  \u00b7  {n_ad} ad  \u00b7  {n_content} content{sel_text}"
        )

        # Status bar
        conn = "Connected" if self._server.extension.connected else "Disconnected"
        showing = len(self._bulk_filtered_rows)
        self._statusbar.showMessage(
            f"Extension: {conn}  \u00b7  "
            f"Showing: {showing}/{total}  \u00b7  "
            f"Session: {self._bulk_session_count} captured"
        )

        # Page controls
        self._page_label.setText(
            f"Page {self._bulk_page + 1}/{self._bulk_total_pages()}"
        )
        self._btn_prev.setEnabled(self._bulk_page > 0)
        self._btn_next.setEnabled(
            self._bulk_page < self._bulk_total_pages() - 1
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
        help="Live monitor capture interval in seconds (default: 2)",
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
        help=f"Frames in live monitor rolling display (default: {HISTORY_SIZE})",
    )
    parser.add_argument(
        "--screenshot-dir", default="screenshots", dest="screenshot_dir",
        help="Where to save screenshots (default: screenshots/)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Hush")

    # Let Ctrl+C close the app gracefully.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    timer = QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    window = HushApp(args)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
