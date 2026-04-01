import json
import sqlite3
from datetime import datetime

import numpy as np


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
                CREATE TABLE IF NOT EXISTS features (
                    screenshot_id  INTEGER NOT NULL REFERENCES screenshots(id),
                    extractor      TEXT NOT NULL,
                    vector         BLOB NOT NULL,
                    PRIMARY KEY (screenshot_id, extractor)
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                -- Clean up duplicate model_path rows (keep newest) before
                -- adding the unique index.  Runs once; is a no-op thereafter.
                DELETE FROM training_runs
                    WHERE id NOT IN (
                        SELECT MAX(id) FROM training_runs GROUP BY model_path
                    );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_training_runs_model_path
                    ON training_runs(model_path);
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
        """Remove DB records and associated features. Does not touch the filesystem."""
        with self.conn:
            # Delete features for these screenshots before removing the screenshots
            self.conn.execute(
                "DELETE FROM features WHERE screenshot_id IN "
                "(SELECT id FROM screenshots WHERE filename IN ({}))"
                .format(','.join('?' * len(filenames))),
                list(filenames),
            )
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
            # If a previous run wrote to the same model_path, the file was
            # overwritten — the old record is stale.  Remove it first.
            self.conn.execute(
                "DELETE FROM training_runs WHERE model_path = ?",
                (model_path,),
            )
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

    def get_training_runs(self):
        """Return all training runs, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM training_runs ORDER BY run_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_training_run(self, run_id):
        """Delete a training run record by ID."""
        with self.conn:
            self.conn.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))

    def get_training_summary(self):
        """Return label and source counts for trainable labeled data.

        Returns a dict with keys:
          - label_counts: {label_name: count} for ad/content/neither
          - source_counts: {source_name: count}
          - total: total number of trainable frames
        """
        rows = self.conn.execute(
            "SELECT label, source, COUNT(*) as n FROM screenshots "
            "WHERE label IN ('ad', 'content', 'neither') "
            "GROUP BY label, source"
        ).fetchall()
        label_counts = {}
        source_counts = {}
        total = 0
        for r in rows:
            label_counts[r['label']] = label_counts.get(r['label'], 0) + r['n']
            source_counts[r['source']] = source_counts.get(r['source'], 0) + r['n']
            total += r['n']
        return {
            'label_counts': label_counts,
            'source_counts': source_counts,
            'total': total,
        }

    # ── settings ─────────────────────────────────────────────────────────────

    def get_setting(self, key, default=None):
        """Return the value for *key*, or *default* if not set."""
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row['value'] if row else default

    def set_setting(self, key, value):
        """Insert or update a setting."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )

    # ── feature vectors ──────────────────────────────────────────────────────

    def get_features(self, extractor, screenshot_ids):
        """Return {screenshot_id: np.ndarray} for rows that exist."""
        if not screenshot_ids:
            return {}
        placeholders = ','.join('?' * len(screenshot_ids))
        rows = self.conn.execute(
            f"SELECT screenshot_id, vector FROM features "
            f"WHERE extractor = ? AND screenshot_id IN ({placeholders})",
            [extractor] + list(screenshot_ids),
        ).fetchall()
        return {
            row['screenshot_id']: np.frombuffer(row['vector'], dtype=np.float32)
            for row in rows
        }

    def save_feature(self, screenshot_id, extractor, vector):
        """Insert or replace a single feature vector."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO features (screenshot_id, extractor, vector) "
                "VALUES (?, ?, ?)",
                (screenshot_id, extractor, vector.astype(np.float32).tobytes()),
            )

    def save_features_bulk(self, extractor, id_vector_pairs):
        """Insert or replace multiple feature vectors in one transaction."""
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO features (screenshot_id, extractor, vector) "
                "VALUES (?, ?, ?)",
                [
                    (sid, extractor, vec.astype(np.float32).tobytes())
                    for sid, vec in id_vector_pairs
                ],
            )
