import json
import sqlite3
from datetime import datetime


class LabelStore:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS screenshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename    TEXT NOT NULL UNIQUE,
                    captured_at TEXT NOT NULL,
                    label       TEXT,
                    labeled_at  TEXT,
                    source      TEXT NOT NULL DEFAULT 'capture'
                );
                CREATE TABLE IF NOT EXISTS training_runs (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at            TEXT NOT NULL,
                    model_path        TEXT NOT NULL,
                    n_folds           INTEGER NOT NULL,
                    cv_accuracy_mean  REAL NOT NULL,
                    cv_accuracy_std   REAL NOT NULL,
                    cv_f1_mean        REAL NOT NULL,
                    cv_auc_mean       REAL NOT NULL,
                    cv_auc_std        REAL NOT NULL,
                    confusion_matrix  TEXT NOT NULL,
                    n_ad              INTEGER NOT NULL,
                    n_not_ad          INTEGER NOT NULL,
                    classifier_type   TEXT NOT NULL,
                    feature_extractor TEXT NOT NULL,
                    notes             TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_screenshots_label
                    ON screenshots(label);
                CREATE INDEX IF NOT EXISTS idx_screenshots_source
                    ON screenshots(source);
            """)

    def add_screenshot(self, filename, captured_at, source='capture'):
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO screenshots (filename, captured_at, source) VALUES (?, ?, ?)",
                (filename, captured_at, source),
            )
            return cursor.lastrowid

    def get_unlabeled(self):
        rows = self.conn.execute(
            "SELECT * FROM screenshots WHERE label IS NULL ORDER BY captured_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_labeled(self):
        rows = self.conn.execute(
            "SELECT * FROM screenshots WHERE label IS NOT NULL ORDER BY captured_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self):
        rows = self.conn.execute(
            "SELECT * FROM screenshots ORDER BY captured_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_filename(self, filename):
        row = self.conn.execute(
            "SELECT * FROM screenshots WHERE filename = ?", (filename,)
        ).fetchone()
        return dict(row) if row else None

    def set_label(self, filename, label):
        with self.conn:
            self.conn.execute(
                "UPDATE screenshots SET label = ?, labeled_at = ? WHERE filename = ?",
                (label, datetime.now().isoformat(), filename),
            )

    def set_labels_bulk(self, filenames, label):
        now = datetime.now().isoformat()
        with self.conn:
            self.conn.executemany(
                "UPDATE screenshots SET label = ?, labeled_at = ? WHERE filename = ?",
                [(label, now, f) for f in filenames],
            )

    def clear_labels_bulk(self, filenames):
        """Reset filenames to unlabeled (NULL label and labeled_at). Used by undo."""
        with self.conn:
            self.conn.executemany(
                "UPDATE screenshots SET label = NULL, labeled_at = NULL WHERE filename = ?",
                [(f,) for f in filenames],
            )

    def delete_screenshots(self, filenames):
        """Remove DB records for the given filenames. Does not touch the filesystem."""
        with self.conn:
            self.conn.executemany(
                "DELETE FROM screenshots WHERE filename = ?",
                [(f,) for f in filenames],
            )

    def record_training_run(
        self,
        model_path,
        n_folds,
        cv_accuracy_mean,
        cv_accuracy_std,
        cv_f1_mean,
        cv_auc_mean,
        cv_auc_std,
        confusion_matrix,
        n_ad,
        n_not_ad,
        classifier_type,
        feature_extractor,
        notes=None,
    ):
        # confusion_matrix may be passed as a dict; store as JSON string
        cm_json = json.dumps(confusion_matrix) if not isinstance(confusion_matrix, str) else confusion_matrix
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO training_runs
                    (run_at, model_path, n_folds,
                     cv_accuracy_mean, cv_accuracy_std,
                     cv_f1_mean, cv_auc_mean, cv_auc_std,
                     confusion_matrix, n_ad, n_not_ad,
                     classifier_type, feature_extractor, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    model_path,
                    n_folds,
                    cv_accuracy_mean,
                    cv_accuracy_std,
                    cv_f1_mean,
                    cv_auc_mean,
                    cv_auc_std,
                    cm_json,
                    n_ad,
                    n_not_ad,
                    classifier_type,
                    feature_extractor,
                    notes,
                ),
            )
