"""Unified selection logic for thumbnail grids.

Handles click, shift-click range, select-all, and clear. Used by both
Live Monitoring and Bulk Capture modes in the desktop app. Pure logic —
no GUI imports, no side effects.
"""


class SelectionManager:
    """Tracks a set of selected item keys (typically file paths).

    The caller provides an ordered list of visible keys (the current page or
    rolling history) so that range selection can compute indices. The manager
    never holds a reference to that list — pass it on each call.
    """

    def __init__(self):
        self._selected: set[str] = set()
        self._last_clicked_idx: int | None = None

    @property
    def selected(self) -> set[str]:
        return self._selected

    @property
    def last_clicked_idx(self) -> int | None:
        return self._last_clicked_idx

    def is_selected(self, key: str) -> bool:
        return key in self._selected

    def click(self, idx: int, visible_keys: list[str], shift: bool = False):
        """Handle a click on item at `idx` within `visible_keys`.

        Plain click toggles the item. Shift-click selects the range from
        the last-clicked item to the current one (inclusive).
        """
        if idx < 0 or idx >= len(visible_keys):
            return

        key = visible_keys[idx]

        if shift and self._last_clicked_idx is not None:
            lo = min(self._last_clicked_idx, idx)
            hi = max(self._last_clicked_idx, idx)
            for i in range(lo, hi + 1):
                if i < len(visible_keys):
                    self._selected.add(visible_keys[i])
        else:
            if key in self._selected:
                self._selected.discard(key)
            else:
                self._selected.add(key)

        self._last_clicked_idx = idx

    def select_all(self, visible_keys: list[str]):
        """Select every item in the visible set."""
        self._selected.update(visible_keys)

    def clear(self):
        """Deselect everything and reset last-clicked index."""
        self._selected.clear()
        self._last_clicked_idx = None

    def count(self) -> int:
        return len(self._selected)
