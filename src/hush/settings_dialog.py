"""Settings dialog — persistent app preferences stored in the SQLite settings table."""

import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from hush.label_store import LabelStore

# Keys used in the settings table
KEY_ACTIVE_MODEL = 'active_model_path'
KEY_LIVE_INTERVAL = 'live_interval'
KEY_BULK_INTERVAL = 'bulk_interval'
KEY_MUTE_DELAY = 'mute_delay'
KEY_UNMUTE_DELAY = 'unmute_delay'
KEY_SCREENSHOT_DIR = 'screenshot_dir'


class SettingsDialog(QDialog):
    """Dialog for editing persistent app preferences.

    Reads current values from a ``current`` dict keyed by the KEY_* constants.
    On accept, persists all values to the settings table and emits
    ``settings_changed(dict)`` so the parent can apply changes live.
    """

    settings_changed = pyqtSignal(dict)

    def __init__(self, db_path: str, current: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)
        self.setModal(True)

        self._db_path = db_path
        self._values = dict(current)  # working copy — mutated by browse actions

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setSpacing(12)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        # Active model — file picker (full ModelBrowser is via the Models button)
        model_row = QHBoxLayout()
        self._model_label = QLabel(self._model_display())
        self._model_label.setFont(QFont("Menlo", 14))
        model_row.addWidget(self._model_label, stretch=1)
        model_browse = QPushButton("Browse...")
        model_browse.setFixedWidth(80)
        model_browse.clicked.connect(self._browse_model)
        model_row.addWidget(model_browse)
        form.addRow("Active model:", model_row)

        # Live monitoring interval
        self._live_spin = QDoubleSpinBox()
        self._live_spin.setRange(0.5, 30.0)
        self._live_spin.setSingleStep(0.5)
        self._live_spin.setDecimals(1)
        self._live_spin.setSuffix(" s")
        self._live_spin.setFixedWidth(90)
        self._live_spin.setValue(float(self._values.get(KEY_LIVE_INTERVAL, 2.0)))
        form.addRow("Live monitoring interval:", self._live_spin)

        # Bulk capture interval
        self._bulk_spin = QSpinBox()
        self._bulk_spin.setRange(1, 60)
        self._bulk_spin.setSuffix(" s")
        self._bulk_spin.setFixedWidth(90)
        self._bulk_spin.setValue(int(self._values.get(KEY_BULK_INTERVAL, 7)))
        form.addRow("Bulk capture interval:", self._bulk_spin)

        # Mute delay
        self._mute_spin = QSpinBox()
        self._mute_spin.setRange(1, 20)
        self._mute_spin.setSuffix(" frames")
        self._mute_spin.setFixedWidth(100)
        self._mute_spin.setValue(int(self._values.get(KEY_MUTE_DELAY, 2)))
        form.addRow("Mute delay:", self._mute_spin)

        # Unmute delay
        self._unmute_spin = QSpinBox()
        self._unmute_spin.setRange(1, 20)
        self._unmute_spin.setSuffix(" frames")
        self._unmute_spin.setFixedWidth(100)
        self._unmute_spin.setValue(int(self._values.get(KEY_UNMUTE_DELAY, 3)))
        form.addRow("Unmute delay:", self._unmute_spin)

        # Screenshot directory
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(self._values.get(KEY_SCREENSHOT_DIR, 'screenshots'))
        dir_row.addWidget(self._dir_edit, stretch=1)
        dir_browse = QPushButton("Browse...")
        dir_browse.setFixedWidth(80)
        dir_browse.clicked.connect(self._browse_dir)
        dir_row.addWidget(dir_browse)
        form.addRow("Screenshot directory:", dir_row)

        outer.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _model_display(self) -> str:
        path = self._values.get(KEY_ACTIVE_MODEL, '')
        return os.path.basename(path) if path else '(none)'

    def _browse_model(self):
        from hush.model_browser import ModelBrowser  # noqa: PLC0415
        browser = ModelBrowser(
            self._db_path,
            active_model_path=self._values.get(KEY_ACTIVE_MODEL, ''),
            parent=self,
        )
        browser.model_selected.connect(self._on_model_selected)
        browser.exec()

    def _on_model_selected(self, path: str):
        self._values[KEY_ACTIVE_MODEL] = path
        self._model_label.setText(os.path.basename(path))

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Screenshot Directory",
            self._dir_edit.text() or '.',
        )
        if path:
            self._dir_edit.setText(path)

    # ── Accept ────────────────────────────────────────────────────────────────

    def _on_accept(self):
        result = {
            KEY_ACTIVE_MODEL: self._values.get(KEY_ACTIVE_MODEL, ''),
            KEY_LIVE_INTERVAL: self._live_spin.value(),
            KEY_BULK_INTERVAL: self._bulk_spin.value(),
            KEY_MUTE_DELAY: self._mute_spin.value(),
            KEY_UNMUTE_DELAY: self._unmute_spin.value(),
            KEY_SCREENSHOT_DIR: self._dir_edit.text().strip() or 'screenshots',
        }

        store = LabelStore(self._db_path)
        for key, val in result.items():
            store.set_setting(key, str(val))

        self.settings_changed.emit(result)
        self.accept()
