"""Frame processing pipeline — classify frames and drive the MuteFSM.

Receives raw JPEG bytes from the server, extracts features, classifies,
updates the FSM, and returns the action and metadata. Decoupled from both
the server and the GUI so it can be tested with a fake classifier.
"""

import io
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from hush.label_store import LabelStore
from hush.mute_fsm import MuteFSM

logger = logging.getLogger(__name__)

LABEL_NAMES = ["ad", "content", "neither"]
UNCERTAIN_THRESH = 0.80

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    path: str                     # temp file path (or permanent if relabeled)
    captured_at: str
    pred_label: str               # argmax prediction
    prob_ad: float                # P(ad)
    probs: list                   # [P(ad), P(content), P(neither)]
    uncertain: bool               # max(probs) < threshold
    fsm_action: Optional[str]     # 'mute', 'unmute', or None
    relabeled: bool = False
    user_label: Optional[str] = None


def build_extractor(extractor_name: str, device):
    """Return a callable (PIL.Image) -> np.ndarray of shape [1, D]."""
    if extractor_name == "resnet50":
        import torchvision.models as tv_models

        model = tv_models.resnet50(weights="IMAGENET1K_V2")
        net = nn.Sequential(*list(model.children())[:-1])
        net.eval().to(device)
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        def extract(img: Image.Image) -> np.ndarray:
            t = tf(img.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                return net(t).squeeze(-1).squeeze(-1).cpu().numpy()
        return extract

    raise ValueError(f"Unknown extractor: {extractor_name!r}")


def load_model(model_path: str) -> dict:
    """Load and validate a model bundle.

    Returns dict with keys: classifier, extractor, label_map, label_names, etc.
    Raises FileNotFoundError or ValueError on problems.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Model file is not a valid bundle: {model_path}")

    required_keys = {"classifier", "label_map"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(
            f"Model bundle missing required keys: {missing}"
        )

    return bundle


class FramePipeline:
    """Processes incoming frames: classify, update FSM, manage temp files."""

    def __init__(
        self,
        model_path: str,
        store: LabelStore,
        hard_frame_dir: str = "hard_frames",
        mute_delay: int = 2,
        unmute_delay: int = 3,
    ):
        self.store = store
        self.hard_frame_dir = hard_frame_dir
        os.makedirs(hard_frame_dir, exist_ok=True)

        # Load model
        bundle = load_model(model_path)
        self._clf = bundle["classifier"]
        self._label_map = bundle["label_map"]
        self._ad_class = self._label_map["ad"]

        # Feature extractor
        device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self._extract = build_extractor(
            bundle.get("extractor", "resnet50"), device
        )

        # FSM
        self.fsm = MuteFSM(mute_delay=mute_delay, unmute_delay=unmute_delay)

        # Temp directory for non-relabeled frames
        self._temp_dir = tempfile.mkdtemp(prefix="hush_v2_")

    def process_frame(
        self,
        image_bytes: bytes,
        timestamp: str = "",
        user_unmuted: bool = False,
        user_muted: bool = False,
    ) -> FrameResult:
        """Classify a frame and update the FSM.

        Args:
            image_bytes: Raw JPEG bytes from the extension.
            timestamp: ISO 8601 timestamp from the extension.
            user_unmuted: True if the user manually unmuted since last frame.
            user_muted: True if the user manually muted since last frame.

        Returns:
            FrameResult with classification and FSM action.
        """
        if not timestamp:
            timestamp = datetime.now().isoformat()

        # Save to temp file
        fname = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{id(image_bytes):08x}.jpg"
        path = os.path.join(self._temp_dir, fname)
        with open(path, "wb") as f:
            f.write(image_bytes)

        # Classify
        img = Image.open(io.BytesIO(image_bytes))
        feat = self._extract(img)
        raw = self._clf.predict_proba(feat)

        probs3 = np.zeros(3)
        for col, cls in enumerate(self._clf.classes_):
            probs3[cls] = raw[0, col]

        pred_label = LABEL_NAMES[int(np.argmax(probs3))]
        prob_ad = float(probs3[self._ad_class])
        uncertain = float(np.max(probs3)) < UNCERTAIN_THRESH

        # FSM decision
        action = self.fsm.update(
            pred_label,
            user_unmuted=user_unmuted,
            user_muted=user_muted,
        )

        return FrameResult(
            path=path,
            captured_at=timestamp,
            pred_label=pred_label,
            prob_ad=prob_ad,
            probs=probs3.tolist(),
            uncertain=uncertain,
            fsm_action=action,
        )

    def relabel_frame(self, result: FrameResult, label: str):
        """Move a frame to permanent storage and record in DB."""
        if result.relabeled:
            return

        fname = os.path.basename(result.path)
        dest = os.path.join(self.hard_frame_dir, fname)
        shutil.move(result.path, dest)

        result.path = dest
        result.relabeled = True
        result.user_label = label

        self.store.add_screenshot(dest, result.captured_at, source="monitor")
        self.store.set_label(dest, label)

    def delete_temp_file(self, result: FrameResult):
        """Delete a temp frame file (called when it ages out of history)."""
        if not result.relabeled:
            try:
                os.remove(result.path)
            except OSError:
                pass

    def cleanup(self):
        """Remove the temp directory. Call on shutdown."""
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except OSError:
            pass
