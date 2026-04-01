"""Training dialog — runs the training pipeline in a background thread with progress."""

import logging
import os

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from hush.label_store import LabelStore
from hush.train import EXTRACTORS, TrainResult, TrainingError, default_model_path, train_model

logger = logging.getLogger(__name__)


class _TrainWorker(QThread):
    """Runs train_model() off the GUI thread, forwarding progress via signals."""

    progress = pyqtSignal(str, int, int)   # stage, current, total
    finished = pyqtSignal(object)          # TrainResult on success
    failed = pyqtSignal(str)               # error message

    def __init__(self, db_path, model_out=None, extractor_name='resnet50', notes=None):
        super().__init__()
        self.db_path = db_path
        self.model_out = model_out
        self.extractor_name = extractor_name
        self.notes = notes

    def run(self):
        try:
            result = train_model(
                db_path=self.db_path,
                model_out=self.model_out,
                extractor_name=self.extractor_name,
                notes=self.notes,
                on_progress=lambda stage, cur, total: self.progress.emit(stage, cur, total),
            )
            self.finished.emit(result)
        except TrainingError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            logger.exception("Training failed unexpectedly")
            self.failed.emit(f"Unexpected error: {exc}")


class TrainDialog(QDialog):
    """Modal dialog for in-app model training.

    Phases:
    1. Confirmation — shows label counts and optional mixing warning.
    2. Progress — two progress bars (extraction, then CV).
    3. Completion — summary of results.

    Emits ``training_complete(TrainResult)`` on success so the parent can
    reload the model or open the model browser.
    """

    training_complete = pyqtSignal(object)  # TrainResult

    def __init__(self, db_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Model")
        self.setMinimumWidth(440)
        self.setModal(True)

        self._db_path = db_path
        self._store = LabelStore(db_path)
        self._worker = None
        self._result = None

        self._build_ui()
        self._show_confirmation()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(12)

        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._layout.addWidget(self._info_label)

        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet(
            "QLabel { background: #fdf2e9; border: 1px solid #e67e22; "
            "border-radius: 4px; padding: 8px; color: #7d4e00; }"
        )
        self._warning_label.setVisible(False)
        self._layout.addWidget(self._warning_label)

        # Options row (model name + extractor) — hidden during training
        self._options_widget = QWidget()
        options_layout = QVBoxLayout(self._options_widget)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(6)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Model name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. classifier_20260331_120000")
        name_row.addWidget(self._name_edit)
        options_layout.addLayout(name_row)

        extractor_row = QHBoxLayout()
        extractor_row.addWidget(QLabel("Feature extractor:"))
        self._extractor_combo = QComboBox()
        for name in EXTRACTORS:
            self._extractor_combo.addItem(name)
        self._extractor_combo.setCurrentText("clip")
        extractor_row.addWidget(self._extractor_combo)
        extractor_row.addStretch()
        options_layout.addLayout(extractor_row)

        self._layout.addWidget(self._options_widget)

        # ── Extraction progress ──────────────────────────────────────────────
        self._extract_label = QLabel("Feature extraction:")
        self._extract_label.setVisible(False)
        self._layout.addWidget(self._extract_label)

        self._extract_bar = QProgressBar()
        self._extract_bar.setVisible(False)
        self._layout.addWidget(self._extract_bar)

        # ── CV progress ──────────────────────────────────────────────────────
        self._cv_label = QLabel("Cross-validation:")
        self._cv_label.setVisible(False)
        self._layout.addWidget(self._cv_label)

        self._cv_bar = QProgressBar()
        self._cv_bar.setVisible(False)
        self._layout.addWidget(self._cv_bar)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Train")
        self._buttons.accepted.connect(self._on_train_clicked)
        self._buttons.rejected.connect(self.reject)
        self._layout.addWidget(self._buttons)

    # ── Phase 1: Confirmation ────────────────────────────────────────────────

    def _show_confirmation(self):
        summary = self._store.get_training_summary()
        total = summary['total']
        lc = summary['label_counts']
        sc = summary['source_counts']

        # Default model name (stem only; path and .joblib added at train time)
        default_stem = os.path.splitext(os.path.basename(default_model_path()))[0]
        self._name_edit.setText(default_stem)

        if total == 0:
            self._info_label.setText("No labeled training data found.\n\n"
                                     "Label some frames first (A/C/N keys).")
            self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return

        lines = [f"Train on {total} labeled frames:"]
        for name in ('ad', 'content', 'neither'):
            count = lc.get(name, 0)
            lines.append(f"  {name}: {count}")
        self._info_label.setText("\n".join(lines))

        # Mixing warning: multiple capture sources
        if len(sc) > 1:
            parts = [f"{count} {source}" for source, count in sorted(sc.items())]
            self._warning_label.setText(
                "Training data includes " + ", ".join(parts) + " captures. "
                "These may look different (e.g. full-screen vs. tab-only). "
                "Check that CV metrics are acceptable after training."
            )
            self._warning_label.setVisible(True)

        # Disable Train if fewer than 2 classes
        present = [n for n in ('ad', 'content', 'neither') if lc.get(n, 0) > 0]
        if len(present) < 2:
            self._info_label.setText(
                self._info_label.text() + "\n\n"
                "Need at least 2 classes to train."
            )
            self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    # ── Phase 2: Training ────────────────────────────────────────────────────

    def _on_train_clicked(self):
        stem = self._name_edit.text().strip() or os.path.splitext(
            os.path.basename(default_model_path()))[0]
        model_out = os.path.join('models', stem + '.joblib')

        self._info_label.setText("Training in progress...")
        self._warning_label.setVisible(False)
        self._options_widget.setVisible(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setEnabled(False)

        # Show both progress sections immediately so the dialog doesn't resize
        self._extract_label.setVisible(True)
        self._extract_bar.setVisible(True)
        self._cv_label.setVisible(True)
        self._cv_bar.setVisible(True)
        self._cv_bar.setMaximum(0)  # indeterminate until CV starts

        extractor_name = self._extractor_combo.currentText()
        self._worker = _TrainWorker(
            self._db_path, model_out=model_out, extractor_name=extractor_name
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, stage: str, current: int, total: int):
        if stage == "extracting":
            if total == 0:
                self._extract_bar.setMaximum(1)
                self._extract_bar.setValue(1)
                self._extract_label.setText(
                    f"Feature extraction: pre-cached for all {self._store.get_training_summary()['total']} frames"
                )
            else:
                self._extract_bar.setMaximum(total)
                self._extract_bar.setValue(current)
                self._extract_label.setText(
                    f"Feature extraction: {current}/{total}"
                )
        elif stage == "training":
            self._cv_bar.setMaximum(total)
            self._cv_bar.setValue(current)
            self._cv_label.setText(f"Cross-validation: fold {current}/{total}")

    # ── Phase 3: Completion ──────────────────────────────────────────────────

    def _on_finished(self, result: TrainResult):
        self._result = result
        self._extract_label.setVisible(False)
        self._extract_bar.setVisible(False)
        self._cv_label.setVisible(False)
        self._cv_bar.setVisible(False)

        self._info_label.setText(
            f"Training complete!\n\n"
            f"Samples: {result.n_samples}\n"
            f"CV Accuracy: {result.cv_accuracy_mean:.3f} "
            f"(+/- {result.cv_accuracy_std:.3f})\n"
            f"CV F1: {result.cv_f1_mean:.3f}\n"
            f"CV AUC: {result.cv_auc_mean:.3f} "
            f"(+/- {result.cv_auc_std:.3f})\n\n"
            f"Model saved: {result.model_path}"
        )

        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setVisible(False)

        # Disconnect old handler and reconnect to close
        self._buttons.accepted.disconnect()
        self._buttons.accepted.connect(self._on_done)

        self.training_complete.emit(result)

    def _on_failed(self, message: str):
        self._extract_label.setVisible(False)
        self._extract_bar.setVisible(False)
        self._cv_label.setVisible(False)
        self._cv_bar.setVisible(False)

        self._info_label.setText(f"Training failed:\n\n{message}")

        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setVisible(False)
        self._buttons.accepted.disconnect()
        self._buttons.accepted.connect(self.reject)

    def _on_done(self):
        self.accept()
