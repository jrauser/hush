"""Tests for macos/make_icon.py, the Hush.app icon renderer."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="needs the macOS Apple Color Emoji font"
)

MAKE_ICON_PATH = Path(__file__).resolve().parent.parent / "macos" / "make_icon.py"


def _load_make_icon():
    spec = importlib.util.spec_from_file_location("make_icon", MAKE_ICON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_icon = _load_make_icon()


@pytest.fixture(scope="module")
def gui_app():
    from PyQt6.QtGui import QGuiApplication

    return QGuiApplication.instance() or QGuiApplication([])


def test_iconset_entries_cover_all_sizes_at_1x_and_2x():
    entries = dict(make_icon.iconset_entries())
    assert len(entries) == 2 * len(make_icon.ICONSET_POINT_SIZES)
    assert entries["icon_16x16.png"] == 16
    assert entries["icon_16x16@2x.png"] == 32
    assert entries["icon_512x512@2x.png"] == 1024


def test_write_iconset_writes_every_image_at_its_size(gui_app, tmp_path):
    from PyQt6.QtGui import QImage

    iconset = tmp_path / "Hush.iconset"
    make_icon.write_iconset(iconset)

    for filename, pixel_size in make_icon.iconset_entries():
        image = QImage(str(iconset / filename))
        assert not image.isNull(), filename
        assert (image.width(), image.height()) == (pixel_size, pixel_size)
