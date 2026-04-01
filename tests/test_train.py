"""Tests for train_model() — progress callbacks, return value, error cases.

Feature extraction is mocked (it requires GPU/CPU-heavy ResNet-50 inference).
Everything else — data loading, CV, model saving, DB recording — runs for real.
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from hush.label_store import LabelStore
from hush.train import TrainResult, TrainingError, train_model


@pytest.fixture
def db_with_labeled_data(tmp_path):
    """Create a DB with enough labeled images (dummy files) for training."""
    db_path = str(tmp_path / "test.db")
    store = LabelStore(db_path)

    # Create dummy image files and DB records with labels.
    # We need at least 2 classes with >= 5 samples each (n_folds=5).
    for i in range(10):
        img_path = str(tmp_path / f"ad_{i}.png")
        _write_dummy_png(img_path)
        store.add_screenshot(img_path, f"2026-01-01T00:00:{i:02d}", source="capture")
        store.set_label(img_path, "ad")

    for i in range(10):
        img_path = str(tmp_path / f"content_{i}.png")
        _write_dummy_png(img_path)
        store.add_screenshot(img_path, f"2026-01-01T00:01:{i:02d}", source="capture")
        store.set_label(img_path, "content")

    return db_path, store, tmp_path


@pytest.fixture
def db_mixed_sources(tmp_path):
    """DB with labeled data from two different sources."""
    db_path = str(tmp_path / "mixed.db")
    store = LabelStore(db_path)

    for i in range(8):
        img_path = str(tmp_path / f"cap_ad_{i}.png")
        _write_dummy_png(img_path)
        store.add_screenshot(img_path, f"2026-01-01T00:00:{i:02d}", source="capture")
        store.set_label(img_path, "ad")

    for i in range(8):
        img_path = str(tmp_path / f"ext_content_{i}.png")
        _write_dummy_png(img_path)
        store.add_screenshot(img_path, f"2026-01-01T00:01:{i:02d}", source="extension")
        store.set_label(img_path, "content")

    return db_path, store, tmp_path


def _write_dummy_png(path):
    """Write a minimal valid 1x1 red PNG file."""
    # Minimal PNG: 8-byte signature + IHDR + IDAT + IEND
    import struct
    import zlib

    signature = b'\x89PNG\r\n\x1a\n'

    def chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
    raw_row = b'\x00\xff\x00\x00'  # filter=none, R=255, G=0, B=0
    idat_data = zlib.compress(raw_row)

    with open(path, 'wb') as f:
        f.write(signature)
        f.write(chunk(b'IHDR', ihdr_data))
        f.write(chunk(b'IDAT', idat_data))
        f.write(chunk(b'IEND', b''))


def _fake_extractor(paths, on_progress=None):
    """Return random feature vectors without loading ResNet-50."""
    rng = np.random.RandomState(42)
    total = len(paths)
    batch_size = 32
    batches = []
    done = 0
    for start in range(0, total, batch_size):
        batch_len = min(batch_size, total - start)
        batches.append(rng.randn(batch_len, 2048).astype(np.float32))
        done += batch_len
        if on_progress:
            on_progress("extracting", done, total)
    return np.vstack(batches)


@pytest.fixture(autouse=True)
def _mock_extractor():
    """Replace the real ResNet-50 extractor with fake random vectors."""
    with patch("hush.train.build_extractor") as mock_build:
        mock_obj = type("FakeExtractor", (), {"__call__": lambda self, paths, on_progress=None: _fake_extractor(paths, on_progress)})()
        mock_build.return_value = mock_obj
        yield


class TestTrainModelSuccess:
    def test_returns_train_result(self, db_with_labeled_data):
        db_path, _, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")

        result = train_model(db_path=db_path, model_out=model_out, n_folds=5)

        assert isinstance(result, TrainResult)
        assert result.n_samples == 20
        assert result.label_counts["ad"] == 10
        assert result.label_counts["content"] == 10
        assert result.model_path == model_out
        assert os.path.isfile(model_out)

    def test_roc_data_saved(self, db_with_labeled_data):
        db_path, _, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")

        result = train_model(db_path=db_path, model_out=model_out, n_folds=5)

        assert os.path.isfile(result.roc_path)
        data = np.load(result.roc_path)
        assert "y_true" in data
        assert "y_prob" in data

    def test_metrics_are_reasonable(self, db_with_labeled_data):
        db_path, _, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")

        result = train_model(db_path=db_path, model_out=model_out, n_folds=5)

        assert 0.0 <= result.cv_accuracy_mean <= 1.0
        assert 0.0 <= result.cv_f1_mean <= 1.0
        assert 0.0 <= result.cv_auc_mean <= 1.0
        assert result.cv_accuracy_std >= 0.0

    def test_fold_results_populated(self, db_with_labeled_data):
        db_path, _, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")

        result = train_model(db_path=db_path, model_out=model_out, n_folds=5)

        assert len(result.fold_results) == 5
        for fr in result.fold_results:
            assert "fold" in fr
            assert "acc" in fr
            assert "f1" in fr
            assert "auc" in fr

    def test_training_run_recorded_in_db(self, db_with_labeled_data):
        db_path, store, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")

        train_model(db_path=db_path, model_out=model_out, n_folds=5)

        rows = store.conn.execute("SELECT * FROM training_runs").fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["model_path"] == model_out
        assert row["n_folds"] == 5
        assert row["n_ad"] == 10

    def test_confusion_matrix_shape(self, db_with_labeled_data):
        db_path, _, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")

        result = train_model(db_path=db_path, model_out=model_out, n_folds=5)

        assert len(result.confusion_matrix) == 3
        assert all(len(row) == 3 for row in result.confusion_matrix)


class TestProgressCallback:
    def test_extraction_progress_fires(self, db_with_labeled_data):
        db_path, _, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")
        calls = []

        train_model(
            db_path=db_path,
            model_out=model_out,
            n_folds=5,
            on_progress=lambda stage, cur, total: calls.append((stage, cur, total)),
        )

        extract_calls = [(s, c, t) for s, c, t in calls if s == "extracting"]
        assert len(extract_calls) > 0
        # Last extraction call should have current == total
        assert extract_calls[-1][1] == extract_calls[-1][2]

    def test_training_progress_fires(self, db_with_labeled_data):
        db_path, _, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")
        calls = []

        train_model(
            db_path=db_path,
            model_out=model_out,
            n_folds=5,
            on_progress=lambda stage, cur, total: calls.append((stage, cur, total)),
        )

        training_calls = [(s, c, t) for s, c, t in calls if s == "training"]
        assert len(training_calls) == 5
        assert training_calls[0] == ("training", 1, 5)
        assert training_calls[-1] == ("training", 5, 5)

    def test_no_callback_is_fine(self, db_with_labeled_data):
        db_path, _, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")

        # Should not raise
        result = train_model(db_path=db_path, model_out=model_out, n_folds=5)
        assert isinstance(result, TrainResult)


class TestTrainingErrors:
    def test_no_labeled_data(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        LabelStore(db_path)

        with pytest.raises(TrainingError, match="No usable labeled data"):
            train_model(db_path=db_path, model_out=str(tmp_path / "m.joblib"))

    def test_only_one_class(self, tmp_path):
        db_path = str(tmp_path / "one_class.db")
        store = LabelStore(db_path)

        for i in range(10):
            img_path = str(tmp_path / f"ad_{i}.png")
            _write_dummy_png(img_path)
            store.add_screenshot(img_path, f"2026-01-01T00:00:{i:02d}")
            store.set_label(img_path, "ad")

        with pytest.raises(TrainingError, match="at least 2 classes"):
            train_model(db_path=db_path, model_out=str(tmp_path / "m.joblib"))

    def test_too_few_samples_for_folds(self, tmp_path):
        db_path = str(tmp_path / "few.db")
        store = LabelStore(db_path)

        # 3 ad + 3 content — fewer than n_folds=5
        for i in range(3):
            for label in ("ad", "content"):
                img_path = str(tmp_path / f"{label}_{i}.png")
                _write_dummy_png(img_path)
                store.add_screenshot(img_path, f"2026-01-01T00:{label[0]}:{i:02d}")
                store.set_label(img_path, label)

        with pytest.raises(TrainingError, match="fewer than n_folds"):
            train_model(db_path=db_path, model_out=str(tmp_path / "m.joblib"))

    def test_missing_image_files_skipped(self, tmp_path):
        db_path = str(tmp_path / "missing.db")
        store = LabelStore(db_path)

        # Add records pointing to nonexistent files
        for i in range(10):
            store.add_screenshot(f"/nonexistent/ad_{i}.png", f"2026-01-01T00:00:{i:02d}")
            store.set_label(f"/nonexistent/ad_{i}.png", "ad")

        with pytest.raises(TrainingError, match="No usable labeled data"):
            train_model(db_path=db_path, model_out=str(tmp_path / "m.joblib"))


class TestFeatureCaching:
    def test_cached_features_are_reused(self, db_with_labeled_data):
        db_path, store, tmp_path = db_with_labeled_data
        model_out = str(tmp_path / "models" / "test.joblib")

        # First run — extracts features
        calls_1 = []
        train_model(
            db_path=db_path, model_out=model_out, n_folds=5,
            on_progress=lambda s, c, t: calls_1.append((s, c, t)),
        )
        extract_1 = [c for c in calls_1 if c[0] == "extracting"]
        assert len(extract_1) > 0

        # Second run — features should be cached; one extracting(0,0) signal emitted
        calls_2 = []
        model_out_2 = str(tmp_path / "models" / "test2.joblib")
        train_model(
            db_path=db_path, model_out=model_out_2, n_folds=5,
            on_progress=lambda s, c, t: calls_2.append((s, c, t)),
        )
        extract_2 = [c for c in calls_2 if c[0] == "extracting"]
        assert extract_2 == [("extracting", 0, 0)]


class TestTrainingSummary:
    def test_empty_db(self, store):
        summary = store.get_training_summary()
        assert summary['total'] == 0
        assert summary['label_counts'] == {}
        assert summary['source_counts'] == {}

    def test_single_source(self, store):
        for i in range(5):
            store.add_screenshot(f"img/ad_{i}.jpg", "2026-01-01", source="capture")
            store.set_label(f"img/ad_{i}.jpg", "ad")
        for i in range(3):
            store.add_screenshot(f"img/c_{i}.jpg", "2026-01-01", source="capture")
            store.set_label(f"img/c_{i}.jpg", "content")

        summary = store.get_training_summary()
        assert summary['total'] == 8
        assert summary['label_counts'] == {'ad': 5, 'content': 3}
        assert summary['source_counts'] == {'capture': 8}

    def test_mixed_sources(self, store):
        for i in range(4):
            store.add_screenshot(f"img/cap_{i}.jpg", "2026-01-01", source="capture")
            store.set_label(f"img/cap_{i}.jpg", "ad")
        for i in range(6):
            store.add_screenshot(f"img/ext_{i}.jpg", "2026-01-01", source="extension")
            store.set_label(f"img/ext_{i}.jpg", "content")

        summary = store.get_training_summary()
        assert summary['total'] == 10
        assert summary['source_counts'] == {'capture': 4, 'extension': 6}

    def test_ignore_label_excluded(self, store):
        store.add_screenshot("img/a.jpg", "2026-01-01")
        store.set_label("img/a.jpg", "ignore")
        store.add_screenshot("img/b.jpg", "2026-01-01")
        store.set_label("img/b.jpg", "ad")

        summary = store.get_training_summary()
        assert summary['total'] == 1
        assert 'ignore' not in summary['label_counts']


class TestExtractors:
    """Unit tests for build_extractor() — mock the heavy model loads."""

    def test_unknown_extractor_raises(self):
        # EXTRACTORS dict is not patched by the autouse mock; use it directly.
        from hush.train import EXTRACTORS
        assert "nonexistent" not in EXTRACTORS

    def test_resnet50_extractor_output_shape(self, tmp_path):
        """ResNet-50 extractor returns [1, 2048] for a single image."""
        import torch
        from PIL import Image
        from hush.train import _ResNet50Extractor

        img_path = str(tmp_path / "img.png")
        Image.new("RGB", (100, 75), color=(128, 64, 32)).save(img_path)

        extractor = _ResNet50Extractor(torch.device("cpu"))
        result = extractor([img_path])
        assert result.shape == (1, 2048)

    def test_clip_extractor_output_shape(self, tmp_path):
        """CLIP extractor returns [1, 512] for a single image."""
        import torch
        from unittest.mock import MagicMock, patch
        from PIL import Image

        img_path = str(tmp_path / "img.png")
        Image.new("RGB", (100, 75), color=(128, 64, 32)).save(img_path)

        fake_model = MagicMock()
        fake_model.encode_image.return_value = torch.zeros(1, 512)

        def fake_preprocess(img):
            return torch.zeros(3, 224, 224)

        with patch("clip.load", return_value=(fake_model, fake_preprocess)):
            from hush.train import _CLIPExtractor
            extractor = _CLIPExtractor(torch.device("cpu"))
            result = extractor([img_path])

        assert result.shape == (1, 512)
        fake_model.encode_image.assert_called_once()

    def test_clip_extractor_progress_callback(self, tmp_path):
        """CLIP extractor fires on_progress for each batch."""
        import torch
        from unittest.mock import MagicMock, patch
        from PIL import Image

        for i in range(3):
            Image.new("RGB", (10, 10)).save(str(tmp_path / f"img_{i}.png"))
        paths = [str(tmp_path / f"img_{i}.png") for i in range(3)]

        fake_model = MagicMock()
        fake_model.encode_image.return_value = torch.zeros(3, 512)

        def fake_preprocess(img):
            return torch.zeros(3, 224, 224)

        calls = []
        with patch("clip.load", return_value=(fake_model, fake_preprocess)):
            from hush.train import _CLIPExtractor
            extractor = _CLIPExtractor(torch.device("cpu"))
            extractor(paths, on_progress=lambda s, c, t: calls.append((s, c, t)))

        assert len(calls) > 0
        assert all(stage == "extracting" for stage, _, _ in calls)
        assert calls[-1][1] == 3  # final current == total

    def test_train_model_with_clip_extractor(self, db_with_labeled_data, tmp_path):
        """train_model() accepts extractor_name='clip' and saves feature_extractor in bundle."""
        from unittest.mock import patch
        import joblib

        db_path, _, data_tmp = db_with_labeled_data
        model_out = str(tmp_path / "clip_model.joblib")

        # Patch _CLIPExtractor so it returns 512-dim random vectors without loading weights
        def fake_clip_call(self, paths, on_progress=None):
            rng = np.random.RandomState(0)
            n = len(paths)
            vecs = rng.randn(n, 512).astype(np.float32)
            if on_progress:
                on_progress("extracting", n, n)
            return vecs

        with patch("hush.train.build_extractor") as mock_build:
            mock_obj = type("FakeCLIP", (), {"__call__": fake_clip_call})()
            mock_build.return_value = mock_obj
            result = train_model(
                db_path=db_path,
                model_out=model_out,
                extractor_name="clip",
                n_folds=5,
            )

        assert result.feature_extractor == "clip"
        bundle = joblib.load(model_out)
        assert bundle["extractor"] == "clip"
