"""The Hush app icon: the shushing face emoji, drawn from the system emoji font.

Used both at runtime (set as the application icon, which macOS shows in the
Dock without the grey tile it puts around legacy bundle icons) and at build
time by macos/make_icon.py for Hush.app's .icns.
"""

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QFont, QIcon, QImage, QPainter, QPixmap

SHUSHING_FACE = "\U0001f92b"
EMOJI_FONT = "Apple Color Emoji"

# Glyph height as a fraction of the canvas. The emoji artwork has its own
# padding, so this fills the icon about as fully as a native app icon does.
GLYPH_SCALE = 0.9

# Large enough for the biggest Dock magnification on a Retina display.
RUNTIME_ICON_PIXELS = 512


def render_icon(pixel_size: int) -> QImage:
    """Draw the emoji centered on a transparent square canvas."""
    image = QImage(pixel_size, pixel_size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    font = QFont(EMOJI_FONT)
    font.setPixelSize(round(pixel_size * GLYPH_SCALE))

    painter = QPainter(image)
    painter.setFont(font)
    painter.drawText(
        QRectF(0, 0, pixel_size, pixel_size),
        Qt.AlignmentFlag.AlignCenter,
        SHUSHING_FACE,
    )
    painter.end()
    return image


def app_icon() -> QIcon:
    """The icon to pass to QApplication.setWindowIcon()."""
    return QIcon(QPixmap.fromImage(render_icon(RUNTIME_ICON_PIXELS)))
