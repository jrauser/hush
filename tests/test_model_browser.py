"""Tests for model browser data loading: settings, training runs, delete."""

import json
import os

import sqlite3

import numpy as np
import pytest


class TestSettings:
    def test_get_setting_default(self, store):
        assert store.get_setting('nonexistent') is None
        assert store.get_setting('nonexistent', 'fallback') == 'fallback'

    def test_set_and_get_setting(self, store):
        store.set_setting('active_model_path', 'models/test.joblib')
        assert store.get_setting('active_model_path') == 'models/test.joblib'

    def test_set_setting_overwrites(self, store):
        store.set_setting('key', 'value1')
        store.set_setting('key', 'value2')
        assert store.get_setting('key') == 'value2'

    def test_multiple_settings(self, store):
        store.set_setting('a', '1')
        store.set_setting('b', '2')
        assert store.get_setting('a') == '1'
        assert store.get_setting('b') == '2'


class TestTrainingRuns:
    def _record_run(self, store, model_path="models/test.joblib",
                    run_at_suffix="00", auc=0.95):
        """Helper to record a training run with minimal boilerplate."""
        store.record_training_run(
            model_path=model_path,
            n_folds=5,
            cv_accuracy_mean=0.9,
            cv_accuracy_std=0.01,
            cv_f1_mean=0.88,
            cv_auc_mean=auc,
            cv_auc_std=0.005,
            confusion_matrix=[[10, 1, 0], [0, 20, 1], [1, 0, 5]],
            n_ad=11,
            n_not_ad=26,
            classifier_type="HistGradientBoosting",
            feature_extractor="resnet50",
            notes=f"run {run_at_suffix}",
        )

    def test_get_training_runs_empty(self, store):
        assert store.get_training_runs() == []

    def test_get_training_runs_returns_dicts(self, store):
        self._record_run(store)
        runs = store.get_training_runs()
        assert len(runs) == 1
        assert isinstance(runs[0], dict)
        assert runs[0]['model_path'] == 'models/test.joblib'
        assert runs[0]['cv_auc_mean'] == 0.95

    def test_get_training_runs_newest_first(self, store):
        self._record_run(store, model_path="models/a.joblib", run_at_suffix="01")
        self._record_run(store, model_path="models/b.joblib", run_at_suffix="02")
        runs = store.get_training_runs()
        assert len(runs) == 2
        # Second insert has a later run_at
        assert runs[0]['model_path'] == 'models/b.joblib'
        assert runs[1]['model_path'] == 'models/a.joblib'

    def test_delete_training_run(self, store):
        self._record_run(store, model_path="models/a.joblib")
        self._record_run(store, model_path="models/b.joblib")
        runs = store.get_training_runs()
        assert len(runs) == 2

        store.delete_training_run(runs[0]['id'])
        remaining = store.get_training_runs()
        assert len(remaining) == 1
        assert remaining[0]['model_path'] == 'models/a.joblib'

    def test_delete_nonexistent_run_is_noop(self, store):
        self._record_run(store)
        store.delete_training_run(9999)
        assert len(store.get_training_runs()) == 1

    def test_duplicate_model_path_replaces_old_row(self, store):
        """Recording a run with the same model_path replaces the old record."""
        self._record_run(store, model_path="models/dup.joblib", auc=0.90)
        self._record_run(store, model_path="models/dup.joblib", auc=0.95)
        runs = store.get_training_runs()
        assert len(runs) == 1
        assert runs[0]['cv_auc_mean'] == 0.95

    def test_model_path_unique_constraint(self, store):
        """The UNIQUE index on model_path is enforced."""
        self._record_run(store, model_path="models/unique.joblib")
        # Direct INSERT (bypassing the DELETE-then-INSERT in record_training_run)
        # should fail due to the UNIQUE index.
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                "INSERT INTO training_runs "
                "(run_at, model_path, n_folds, cv_accuracy_mean, cv_accuracy_std, "
                "cv_f1_mean, cv_auc_mean, cv_auc_std, confusion_matrix, "
                "n_ad, n_not_ad, classifier_type, feature_extractor) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("2026-01-01", "models/unique.joblib", 5, 0.9, 0.01,
                 0.88, 0.95, 0.005, "[]", 10, 20, "GBM", "resnet50"),
            )

    def test_confusion_matrix_round_trip(self, store):
        cm = [[10, 1, 0], [2, 20, 1], [0, 1, 5]]
        store.record_training_run(
            model_path="models/cm_test.joblib",
            n_folds=5,
            cv_accuracy_mean=0.9,
            cv_accuracy_std=0.01,
            cv_f1_mean=0.88,
            cv_auc_mean=0.95,
            cv_auc_std=0.005,
            confusion_matrix=cm,
            n_ad=11,
            n_not_ad=26,
            classifier_type="HistGradientBoosting",
            feature_extractor="resnet50",
        )
        runs = store.get_training_runs()
        stored_cm = json.loads(runs[0]['confusion_matrix'])
        assert stored_cm == cm


class TestModelBrowserDataFlow:
    """Tests that exercise the data-loading patterns used by ModelBrowser
    without instantiating the Qt GUI."""

    def test_active_model_persistence(self, store):
        """Setting active_model_path persists and is retrievable."""
        store.set_setting('active_model_path', 'models/my_model.joblib')
        assert store.get_setting('active_model_path') == 'models/my_model.joblib'

        # Overwrite with new selection
        store.set_setting('active_model_path', 'models/new_model.joblib')
        assert store.get_setting('active_model_path') == 'models/new_model.joblib'

    def test_delete_clears_active_setting(self, store, tmp_path):
        """Simulate ModelBrowser's delete-active-model flow."""
        model_path = str(tmp_path / "models" / "test.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'w') as f:
            f.write("fake model")

        store.set_setting('active_model_path', model_path)
        store.record_training_run(
            model_path=model_path,
            n_folds=5,
            cv_accuracy_mean=0.9,
            cv_accuracy_std=0.01,
            cv_f1_mean=0.88,
            cv_auc_mean=0.95,
            cv_auc_std=0.005,
            confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            n_ad=1,
            n_not_ad=2,
            classifier_type="HistGradientBoosting",
            feature_extractor="resnet50",
        )

        runs = store.get_training_runs()
        run = runs[0]

        # Simulate delete: remove file, delete run, clear setting
        os.remove(model_path)
        store.delete_training_run(run['id'])
        if store.get_setting('active_model_path') == model_path:
            store.set_setting('active_model_path', '')

        assert store.get_training_runs() == []
        assert store.get_setting('active_model_path') == ''
        assert not os.path.isfile(model_path)

    def test_roc_data_reconciliation(self, tmp_path):
        """ROC .npz path is derived from model path — verify file matching."""
        model_path = str(tmp_path / "classifier_20260331.joblib")
        roc_path = model_path.replace('.joblib', '_roc.npz')

        # Save dummy ROC data
        np.savez(roc_path,
                 y_true=np.array([0, 1, 0, 1]),
                 y_prob=np.array([[0.9, 0.1, 0.0],
                                  [0.1, 0.8, 0.1],
                                  [0.7, 0.2, 0.1],
                                  [0.2, 0.7, 0.1]]))

        assert os.path.isfile(roc_path)
        data = np.load(roc_path)
        assert data['y_true'].shape == (4,)
        assert data['y_prob'].shape == (4, 3)

    def test_missing_model_file_detected(self, store):
        """Training run exists in DB but file is missing on disk."""
        store.record_training_run(
            model_path="/nonexistent/model.joblib",
            n_folds=5,
            cv_accuracy_mean=0.9,
            cv_accuracy_std=0.01,
            cv_f1_mean=0.88,
            cv_auc_mean=0.95,
            cv_auc_std=0.005,
            confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            n_ad=1,
            n_not_ad=2,
            classifier_type="HistGradientBoosting",
            feature_extractor="resnet50",
        )
        runs = store.get_training_runs()
        assert len(runs) == 1
        assert not os.path.isfile(runs[0]['model_path'])
