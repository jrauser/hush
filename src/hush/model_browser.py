"""Model browser — list trained models, view diagnostics, select or delete."""

import json
import logging
import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from sklearn.metrics import roc_curve, auc

from hush.label_store import LabelStore
from hush.train import LABEL_NAMES, TrainResult

logger = logging.getLogger(__name__)


class ModelBrowser(QDialog):
    """Window for browsing, inspecting, and managing trained models.

    Emits ``model_selected(str)`` when the user clicks "Use This Model".
    """

    model_selected = pyqtSignal(str)  # model_path

    def __init__(self, db_path: str, active_model_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Browser")
        self.resize(900, 560)

        self._db_path = db_path
        self._store = LabelStore(db_path)
        self._active_model_path = active_model_path
        self._runs: list[dict] = []

        self._build_ui()
        self._load_runs()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: model list ───────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("Trained Models"))
        self._list_widget = QListWidget()
        self._list_widget.setAlternatingRowColors(True)
        self._list_widget.setStyleSheet(
            "QListWidget { background: white; alternate-background-color: #f2f3f4; }"
            "QListWidget::item { padding: 4px 6px; }"
            "QListWidget::item:selected { background: #2980b9; color: white; }"
            "QListWidget::item:selected:alternate { background: #2980b9; color: white; }"
        )
        self._list_widget.currentRowChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._list_widget)

        train_btn = QPushButton("Train New Model...")
        train_btn.clicked.connect(self._on_train_model)
        left_layout.addWidget(train_btn)

        splitter.addWidget(left)

        # ── Right panel: details ─────────────────────────────────────────
        right = QWidget()
        self._right_layout = QVBoxLayout(right)
        self._right_layout.setContentsMargins(4, 0, 0, 0)

        self._detail_label = QLabel("Select a model to view details.")
        self._detail_label.setWordWrap(True)
        self._detail_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._right_layout.addWidget(self._detail_label)

        # Confusion matrix label
        self._cm_label = QLabel()
        self._cm_label.setFont(QFont("Menlo", 12))
        self._cm_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._cm_label.setVisible(False)
        self._right_layout.addWidget(self._cm_label)

        # ROC canvas
        self._figure = Figure(figsize=(4, 3), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setVisible(False)
        self._right_layout.addWidget(self._canvas)

        # Action buttons
        btn_row = QHBoxLayout()
        self._use_btn = QPushButton("Use This Model")
        self._use_btn.setEnabled(False)
        self._use_btn.clicked.connect(self._on_use_model)
        btn_row.addWidget(self._use_btn)

        self._delete_btn = QPushButton("Delete Model")
        self._delete_btn.setEnabled(False)
        self._delete_btn.setStyleSheet("color: #c0392b;")
        self._delete_btn.clicked.connect(self._on_delete_model)
        btn_row.addWidget(self._delete_btn)

        btn_row.addStretch()
        self._right_layout.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load_runs(self):
        all_runs = self._store.get_training_runs()
        # Only show runs whose model file still exists on disk
        self._runs = [r for r in all_runs if os.path.isfile(r['model_path'])]
        self._list_widget.clear()
        for run in self._runs:
            name = os.path.basename(run['model_path'])
            date = run['run_at'][:10]
            auc_val = run['cv_auc_mean']
            n = run['n_ad'] + run['n_not_ad']

            text = f"{name}\n{date}  AUC={auc_val:.3f}  n={n}"
            if run['model_path'] == self._active_model_path:
                text += "  \u2605 active"

            self._list_widget.addItem(QListWidgetItem(text))

    # ── Detail view ──────────────────────────────────────────────────────────

    def _on_selection_changed(self, row: int):
        if row < 0 or row >= len(self._runs):
            self._detail_label.setText("Select a model to view details.")
            self._cm_label.setVisible(False)
            self._canvas.setVisible(False)
            self._use_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            return

        run = self._runs[row]

        # Detail text
        lines = [
            f"Model: {os.path.basename(run['model_path'])}",
            f"Trained: {run['run_at'][:19]}",
            f"Classifier: {run['classifier_type']}   "
            f"Extractor: {run['feature_extractor']}",
            f"Samples: {run['n_ad']} ad + {run['n_not_ad']} not-ad "
            f"= {run['n_ad'] + run['n_not_ad']}",
            f"CV ({run['n_folds']}-fold):  "
            f"Acc={run['cv_accuracy_mean']:.3f}\u00b1{run['cv_accuracy_std']:.3f}   "
            f"F1={run['cv_f1_mean']:.3f}   "
            f"AUC={run['cv_auc_mean']:.3f}\u00b1{run['cv_auc_std']:.3f}",
        ]
        if run.get('notes'):
            lines.append(f"Notes: {run['notes']}")
        self._detail_label.setText("\n".join(lines))

        # Confusion matrix
        self._show_confusion_matrix(run)

        # ROC curve
        self._show_roc_curve(run)

        self._use_btn.setEnabled(True)
        self._delete_btn.setEnabled(True)

    def _show_confusion_matrix(self, run: dict):
        try:
            cm = json.loads(run['confusion_matrix']) if isinstance(
                run['confusion_matrix'], str) else run['confusion_matrix']
        except (json.JSONDecodeError, TypeError):
            self._cm_label.setVisible(False)
            return

        if not isinstance(cm, list) or len(cm) != 3:
            self._cm_label.setVisible(False)
            return

        col_w = 10
        header = " " * 12 + "".join(f"{n:>{col_w}}" for n in LABEL_NAMES)
        rows = []
        for i, name in enumerate(LABEL_NAMES):
            row_str = f"{name:>12}" + "".join(
                f"{cm[i][j]:>{col_w}}" for j in range(3)
            )
            rows.append(row_str)

        text = "Confusion matrix (summed across folds):\n" + header + "\n" + "\n".join(rows)
        self._cm_label.setText(text)
        self._cm_label.setVisible(True)

    def _show_roc_curve(self, run: dict):
        roc_path = run['model_path'].replace('.joblib', '_roc.npz')
        if not os.path.isfile(roc_path):
            self._canvas.setVisible(False)
            return

        try:
            data = np.load(roc_path)
            y_true = data['y_true']
            y_prob = data['y_prob']
        except Exception:
            logger.warning("Failed to load ROC data: %s", roc_path)
            self._canvas.setVisible(False)
            return

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        # Plot per-class ROC curves (one-vs-rest)
        colors = ['#c0392b', '#27ae60', '#7f8c8d']
        present_classes = sorted(set(y_true.astype(int)))
        for cls in present_classes:
            if cls >= y_prob.shape[1]:
                continue
            binary_true = (y_true == cls).astype(int)
            fpr, tpr, _ = roc_curve(binary_true, y_prob[:, cls])
            roc_auc = auc(fpr, tpr)
            color = colors[cls] if cls < len(colors) else None
            label = LABEL_NAMES[cls] if cls < len(LABEL_NAMES) else f"class {cls}"
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{label} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curves (one-vs-rest)", fontsize=12)
        ax.legend(fontsize=10, loc='lower right')
        ax.tick_params(labelsize=10)
        self._figure.tight_layout()
        self._canvas.draw()
        self._canvas.setVisible(True)

    # ── Actions ──────────────────────────────────────────────────────────────

    def _on_use_model(self):
        row = self._list_widget.currentRow()
        if row < 0 or row >= len(self._runs):
            return
        run = self._runs[row]
        model_path = run['model_path']

        self._store.set_setting('active_model_path', model_path)
        self._active_model_path = model_path
        self.model_selected.emit(model_path)
        self.accept()

    def _on_delete_model(self):
        row = self._list_widget.currentRow()
        if row < 0 or row >= len(self._runs):
            return
        run = self._runs[row]
        name = os.path.basename(run['model_path'])

        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete model \"{name}\" and its training record?\n\n"
            "This removes the model file, ROC data, and DB record.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete files
        for path in [run['model_path'],
                     run['model_path'].replace('.joblib', '_roc.npz')]:
            if os.path.isfile(path):
                os.remove(path)
                logger.info("Deleted: %s", path)

        # Delete DB record
        self._store.delete_training_run(run['id'])

        # Clear active model setting if it pointed to this model
        if self._active_model_path == run['model_path']:
            self._store.set_setting('active_model_path', '')
            self._active_model_path = ''

        self._load_runs()

    def _on_train_model(self):
        # Lazy import to avoid module-level circularity with train_dialog
        from hush.train_dialog import TrainDialog  # noqa: PLC0415
        dialog = TrainDialog(self._db_path, parent=self)
        dialog.training_complete.connect(self._on_training_complete_in_browser)
        dialog.exec()

    def _on_training_complete_in_browser(self, result: TrainResult):
        self._load_runs()
        for i, run in enumerate(self._runs):
            if run['model_path'] == result.model_path:
                self._list_widget.setCurrentRow(i)
                break
