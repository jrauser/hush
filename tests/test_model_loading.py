"""Tests for model loading and validation."""

import os
import tempfile

import joblib
import pytest

from hush.frame_pipeline import load_model


class _FakeClassifier:
    """A picklable fake classifier for testing."""
    def __init__(self):
        self.classes_ = [0, 1, 2]

    def predict_proba(self, X):
        return [[0.1, 0.8, 0.1]]


@pytest.fixture
def valid_model_path():
    """Create a minimal valid model bundle on disk."""
    bundle = {
        "classifier": _FakeClassifier(),
        "extractor": "resnet50",
        "label_map": {"ad": 0, "content": 1, "neither": 2},
        "label_names": ["ad", "content", "neither"],
        "trained_at": "2026-03-30T10:00:00",
        "n_training_samples": 100,
    }

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        joblib.dump(bundle, f.name)
        yield f.name

    os.unlink(f.name)


class TestLoadModel:
    def test_load_valid_model(self, valid_model_path):
        bundle = load_model(valid_model_path)
        assert "classifier" in bundle
        assert "label_map" in bundle
        assert bundle["label_map"]["ad"] == 0

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.joblib")

    def test_load_corrupt_file(self, tmp_path):
        bad_file = tmp_path / "bad.joblib"
        bad_file.write_bytes(b"this is not a joblib file at all")
        with pytest.raises(Exception):
            load_model(str(bad_file))

    def test_load_model_missing_keys(self, tmp_path):
        """Valid joblib but missing required keys."""
        incomplete = {"classifier": "fake", "notes": "no label_map"}
        path = tmp_path / "incomplete.joblib"
        joblib.dump(incomplete, str(path))
        with pytest.raises(ValueError, match="missing required keys"):
            load_model(str(path))

    def test_load_model_not_dict(self, tmp_path):
        """Valid joblib containing a non-dict object."""
        path = tmp_path / "notadict.joblib"
        joblib.dump([1, 2, 3], str(path))
        with pytest.raises(ValueError, match="not a valid bundle"):
            load_model(str(path))
