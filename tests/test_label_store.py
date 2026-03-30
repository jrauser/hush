import json
import sqlite3

import numpy as np
import pytest


class TestAddAndRetrieve:
    def test_add_returns_rowid(self, store):
        rid = store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        assert isinstance(rid, int)
        assert rid >= 1

    def test_get_all_returns_added(self, store):
        store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        rows = store.get_all()
        assert len(rows) == 1
        assert rows[0]["filename"] == "img/a.jpg"
        assert rows[0]["captured_at"] == "2026-01-01T00:00:00"
        assert rows[0]["label"] is None
        assert rows[0]["source"] == "capture"

    def test_add_with_source(self, store):
        store.add_screenshot("img/b.jpg", "2026-01-01T00:00:01", source="hard_frame")
        row = store.get_by_filename("img/b.jpg")
        assert row["source"] == "hard_frame"

    def test_duplicate_filename_rejected(self, store):
        store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        with pytest.raises(sqlite3.IntegrityError):
            store.add_screenshot("img/a.jpg", "2026-01-01T00:00:01")


class TestLabeling:
    def test_set_label(self, store):
        store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        store.set_label("img/a.jpg", "ad")
        row = store.get_by_filename("img/a.jpg")
        assert row["label"] == "ad"
        assert row["labeled_at"] is not None

    def test_set_labels_bulk(self, store):
        files = [f"img/{i}.jpg" for i in range(5)]
        for f in files:
            store.add_screenshot(f, "2026-01-01T00:00:00")
        store.set_labels_bulk(files[:3], "content")
        labeled = store.get_labeled()
        assert len(labeled) == 3
        assert all(r["label"] == "content" for r in labeled)

    def test_clear_labels_bulk(self, store):
        store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        store.set_label("img/a.jpg", "ad")
        store.clear_labels_bulk(["img/a.jpg"])
        row = store.get_by_filename("img/a.jpg")
        assert row["label"] is None
        assert row["labeled_at"] is None


class TestFilters:
    def test_unlabeled_vs_labeled(self, store):
        store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        store.add_screenshot("img/b.jpg", "2026-01-01T00:00:01")
        store.set_label("img/a.jpg", "ad")

        unlabeled = store.get_unlabeled()
        labeled = store.get_labeled()
        assert len(unlabeled) == 1
        assert unlabeled[0]["filename"] == "img/b.jpg"
        assert len(labeled) == 1
        assert labeled[0]["filename"] == "img/a.jpg"

    def test_get_by_filename_exists(self, store):
        store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        row = store.get_by_filename("img/a.jpg")
        assert row is not None
        assert row["filename"] == "img/a.jpg"

    def test_get_by_filename_missing(self, store):
        assert store.get_by_filename("nonexistent.jpg") is None


class TestDelete:
    def test_delete_screenshots(self, store):
        store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        store.add_screenshot("img/b.jpg", "2026-01-01T00:00:01")
        store.delete_screenshots(["img/a.jpg"])
        rows = store.get_all()
        assert len(rows) == 1
        assert rows[0]["filename"] == "img/b.jpg"

    def test_delete_nonexistent_is_noop(self, store):
        store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        store.delete_screenshots(["img/zzz.jpg"])
        assert len(store.get_all()) == 1

    def test_delete_cascades_to_features(self, store):
        sid = store.add_screenshot("img/a.jpg", "2026-01-01T00:00:00")
        vec = np.ones(2048, dtype=np.float32)
        store.save_feature(sid, "resnet50", vec)
        store.delete_screenshots(["img/a.jpg"])
        assert store.get_features("resnet50", [sid]) == {}


class TestTrainingRun:
    def test_record_and_query(self, store):
        cm = {"tp": 45, "fp": 3, "tn": 210, "fn": 8}
        store.record_training_run(
            model_path="models/test.joblib",
            n_folds=5,
            cv_accuracy_mean=0.95,
            cv_accuracy_std=0.01,
            cv_f1_mean=0.92,
            cv_auc_mean=0.98,
            cv_auc_std=0.005,
            confusion_matrix=cm,
            n_ad=53,
            n_not_ad=213,
            classifier_type="HistGradientBoosting",
            feature_extractor="resnet50",
            notes="test run",
        )
        rows = store.conn.execute("SELECT * FROM training_runs").fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["model_path"] == "models/test.joblib"
        assert row["n_folds"] == 5
        assert json.loads(row["confusion_matrix"]) == cm
        assert row["notes"] == "test run"

    def test_record_with_string_cm(self, store):
        cm_str = '{"tp":1,"fp":0,"tn":10,"fn":0}'
        store.record_training_run(
            model_path="models/test2.joblib",
            n_folds=3,
            cv_accuracy_mean=1.0,
            cv_accuracy_std=0.0,
            cv_f1_mean=1.0,
            cv_auc_mean=1.0,
            cv_auc_std=0.0,
            confusion_matrix=cm_str,
            n_ad=1,
            n_not_ad=10,
            classifier_type="RandomForest",
            feature_extractor="resnet50",
        )
        row = dict(store.conn.execute("SELECT * FROM training_runs").fetchone())
        assert row["confusion_matrix"] == cm_str


class TestFeatures:
    def _add_screenshot(self, store, name="img/a.jpg"):
        return store.add_screenshot(name, "2026-01-01T00:00:00")

    def test_save_and_get_feature(self, store):
        sid = self._add_screenshot(store)
        vec = np.random.rand(2048).astype(np.float32)
        store.save_feature(sid, "resnet50", vec)
        result = store.get_features("resnet50", [sid])
        assert sid in result
        np.testing.assert_array_equal(result[sid], vec)

    def test_get_features_returns_only_existing(self, store):
        sid1 = self._add_screenshot(store, "img/a.jpg")
        sid2 = self._add_screenshot(store, "img/b.jpg")
        sid3 = self._add_screenshot(store, "img/c.jpg")
        vec = np.ones(2048, dtype=np.float32)
        store.save_feature(sid1, "resnet50", vec)
        store.save_feature(sid3, "resnet50", vec)
        result = store.get_features("resnet50", [sid1, sid2, sid3])
        assert set(result.keys()) == {sid1, sid3}

    def test_save_features_bulk(self, store):
        sids = [self._add_screenshot(store, f"img/{i}.jpg") for i in range(5)]
        pairs = [(sid, np.full(2048, sid, dtype=np.float32)) for sid in sids]
        store.save_features_bulk("resnet50", pairs)
        result = store.get_features("resnet50", sids)
        assert len(result) == 5
        for sid in sids:
            np.testing.assert_array_equal(result[sid], np.full(2048, sid, dtype=np.float32))

    def test_upsert_replaces_vector(self, store):
        sid = self._add_screenshot(store)
        vec1 = np.zeros(2048, dtype=np.float32)
        vec2 = np.ones(2048, dtype=np.float32)
        store.save_feature(sid, "resnet50", vec1)
        store.save_feature(sid, "resnet50", vec2)
        result = store.get_features("resnet50", [sid])
        np.testing.assert_array_equal(result[sid], vec2)

    def test_multiple_extractors_coexist(self, store):
        sid = self._add_screenshot(store)
        vec_r = np.ones(2048, dtype=np.float32)
        vec_d = np.full(384, 2.0, dtype=np.float32)
        store.save_feature(sid, "resnet50", vec_r)
        store.save_feature(sid, "dinov2", vec_d)
        r = store.get_features("resnet50", [sid])
        d = store.get_features("dinov2", [sid])
        np.testing.assert_array_equal(r[sid], vec_r)
        np.testing.assert_array_equal(d[sid], vec_d)

    def test_get_features_empty_ids(self, store):
        assert store.get_features("resnet50", []) == {}

    def test_features_table_exists(self, store):
        tables = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='features'"
        ).fetchall()
        assert len(tables) == 1
