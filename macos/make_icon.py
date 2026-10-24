"""Write the Hush app icon (see hush.app_icon) as a macOS .iconset.

The emoji artwork comes from the system's Apple Color Emoji font at build
time, so no image is checked into the repo. Convert the result to .icns with
`iconutil -c icns`.

Usage:
    python make_icon.py OUTPUT_DIR.iconset
"""

import sys
from pathlib import Path

from PyQt6.QtGui import QGuiApplication

from hush.app_icon import render_icon

# Point sizes required in a macOS .iconset; each is rendered at 1x and 2x.
ICONSET_POINT_SIZES = (16, 32, 128, 256, 512)


def iconset_entries() -> list[tuple[str, int]]:
    """(filename, pixel size) for every image a macOS .iconset needs."""
    entries = []
    for points in ICONSET_POINT_SIZES:
        entries.append((f"icon_{points}x{points}.png", points))
        entries.append((f"icon_{points}x{points}@2x.png", points * 2))
    return entries


def write_iconset(iconset_dir: Path) -> None:
    iconset_dir.mkdir(parents=True, exist_ok=True)
    for filename, pixel_size in iconset_entries():
        path = iconset_dir / filename
        if not render_icon(pixel_size).save(str(path)):
            raise OSError(f"Could not write {path}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} OUTPUT_DIR.iconset")
    _app = QGuiApplication(sys.argv[:1])  # font rendering needs a GUI app
    write_iconset(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
