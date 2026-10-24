"""Tests for hush.app_icon, the emoji app icon."""

import sys

import pytest
from PyQt6.QtGui import QGuiApplication

from hush import app_icon

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="needs the macOS Apple Color Emoji font"
)


@pytest.fixture(scope="module")
def gui_app():
    return QGuiApplication.instance() or QGuiApplication([])


def test_render_icon_draws_opaque_center_and_transparent_corner(gui_app):
    size = 256
    image = app_icon.render_icon(size)
    assert (image.width(), image.height()) == (size, size)
    assert image.pixelColor(size // 2, size // 2).alpha() == 255
    assert image.pixelColor(0, 0).alpha() == 0


def test_app_icon_is_full_runtime_size(gui_app):
    icon = app_icon.app_icon()
    pixels = app_icon.RUNTIME_ICON_PIXELS
    assert not icon.isNull()
    assert icon.availableSizes()[0].width() == pixels
