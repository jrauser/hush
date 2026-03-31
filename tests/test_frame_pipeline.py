"""Tests for the frame processing pipeline.

Uses a fake classifier to test the classify → FSM → action chain
without requiring a real model or GPU.
"""

import io
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from hush.frame_pipeline import FramePipeline
from hush.label_store import LabelStore


def _make_jpeg_bytes(width=100, height=75):
    """Create minimal valid JPEG bytes."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_fake_pipeline(
    ad_prob=0.1, content_prob=0.8, neither_prob=0.1,
    mute_delay=2, unmute_delay=3,
):
    """Create a FramePipeline with a mocked classifier and extractor."""
    store = LabelStore(":memory:")
    hard_dir = tempfile.mkdtemp(prefix="hush_test_hard_")

    pipeline = FramePipeline.__new__(FramePipeline)
    pipeline.store = store
    pipeline.screenshot_dir = hard_dir
    pipeline._temp_dir = tempfile.mkdtemp(prefix="hush_test_")

    # Fake classifier
    clf = MagicMock()
    clf.classes_ = np.array([0, 1, 2])
    clf.predict_proba = MagicMock(
        return_value=np.array([[ad_prob, content_prob, neither_prob]])
    )
    pipeline._clf = clf
    pipeline._label_map = {"ad": 0, "content": 1, "neither": 2}
    pipeline._ad_class = 0

    # Fake extractor (just returns a dummy feature vector)
    pipeline._extract = MagicMock(return_value=np.zeros((1, 2048)))

    # Real FSM
    from hush.mute_fsm import MuteFSM
    pipeline.fsm = MuteFSM(mute_delay=mute_delay, unmute_delay=unmute_delay)

    return pipeline


class TestFrameClassification:
    def test_frame_classified_as_content(self):
        pipeline = _make_fake_pipeline(ad_prob=0.05, content_prob=0.90, neither_prob=0.05)
        result = pipeline.process_frame(_make_jpeg_bytes())
        assert result.pred_label == "content"
        assert result.prob_ad == pytest.approx(0.05)
        assert not result.uncertain

    def test_frame_classified_as_ad(self):
        pipeline = _make_fake_pipeline(ad_prob=0.90, content_prob=0.05, neither_prob=0.05)
        result = pipeline.process_frame(_make_jpeg_bytes())
        assert result.pred_label == "ad"
        assert result.prob_ad == pytest.approx(0.90)

    def test_uncertain_frame_flagged(self):
        pipeline = _make_fake_pipeline(ad_prob=0.4, content_prob=0.35, neither_prob=0.25)
        result = pipeline.process_frame(_make_jpeg_bytes())
        assert result.uncertain is True

    def test_certain_frame_not_flagged(self):
        pipeline = _make_fake_pipeline(ad_prob=0.05, content_prob=0.90, neither_prob=0.05)
        result = pipeline.process_frame(_make_jpeg_bytes())
        assert result.uncertain is False


class TestFSMIntegration:
    def test_mute_after_ad_streak(self):
        pipeline = _make_fake_pipeline(ad_prob=0.95, content_prob=0.03, neither_prob=0.02,
                                       mute_delay=2)
        r1 = pipeline.process_frame(_make_jpeg_bytes())
        assert r1.fsm_action is None
        r2 = pipeline.process_frame(_make_jpeg_bytes())
        assert r2.fsm_action == "mute"

    def test_unmute_after_content_streak(self):
        pipeline = _make_fake_pipeline(ad_prob=0.95, content_prob=0.03, neither_prob=0.02,
                                       mute_delay=1, unmute_delay=2)
        # Trigger mute
        pipeline.process_frame(_make_jpeg_bytes())
        assert pipeline.fsm.muted is True

        # Switch classifier to content
        pipeline._clf.predict_proba.return_value = np.array([[0.03, 0.95, 0.02]])
        r1 = pipeline.process_frame(_make_jpeg_bytes())
        assert r1.fsm_action is None
        r2 = pipeline.process_frame(_make_jpeg_bytes())
        assert r2.fsm_action == "unmute"


class TestFrameStorage:
    def test_temp_file_created(self):
        pipeline = _make_fake_pipeline()
        result = pipeline.process_frame(_make_jpeg_bytes())
        assert os.path.isfile(result.path)
        assert result.path.startswith(pipeline._temp_dir)

    def test_temp_file_deleted(self):
        pipeline = _make_fake_pipeline()
        result = pipeline.process_frame(_make_jpeg_bytes())
        path = result.path
        assert os.path.isfile(path)
        pipeline.delete_temp_file(result)
        assert not os.path.isfile(path)

    def test_relabeled_frame_persisted(self):
        pipeline = _make_fake_pipeline()
        result = pipeline.process_frame(_make_jpeg_bytes())
        original_path = result.path

        pipeline.relabel_frame(result, "ad")
        assert result.relabeled is True
        assert result.user_label == "ad"
        assert result.path.startswith(pipeline.screenshot_dir)
        assert os.path.isfile(result.path)
        assert not os.path.isfile(original_path)

        # Should be in the database
        row = pipeline.store.get_by_filename(result.path)
        assert row is not None
        assert row["label"] == "ad"

    def test_relabeled_frame_not_deleted(self):
        pipeline = _make_fake_pipeline()
        result = pipeline.process_frame(_make_jpeg_bytes())
        pipeline.relabel_frame(result, "content")
        path = result.path
        # delete_temp_file should not delete relabeled frames
        pipeline.delete_temp_file(result)
        assert os.path.isfile(path)

    def test_double_relabel_is_noop(self):
        pipeline = _make_fake_pipeline()
        result = pipeline.process_frame(_make_jpeg_bytes())
        pipeline.relabel_frame(result, "ad")
        first_path = result.path
        # Second relabel should be a no-op
        pipeline.relabel_frame(result, "content")
        assert result.user_label == "ad"
        assert result.path == first_path
