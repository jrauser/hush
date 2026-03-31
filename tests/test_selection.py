"""Tests for the unified selection manager."""

from hush.selection import SelectionManager


KEYS = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]


class TestClick:
    def test_click_selects(self):
        sel = SelectionManager()
        sel.click(0, KEYS)
        assert sel.is_selected("a.jpg")
        assert sel.count() == 1

    def test_click_toggles_off(self):
        sel = SelectionManager()
        sel.click(0, KEYS)
        sel.click(0, KEYS)
        assert not sel.is_selected("a.jpg")
        assert sel.count() == 0

    def test_click_multiple_items(self):
        sel = SelectionManager()
        sel.click(0, KEYS)
        sel.click(2, KEYS)
        assert sel.selected == {"a.jpg", "c.jpg"}

    def test_click_out_of_range_ignored(self):
        sel = SelectionManager()
        sel.click(-1, KEYS)
        sel.click(99, KEYS)
        assert sel.count() == 0


class TestShiftClick:
    def test_shift_click_range_forward(self):
        sel = SelectionManager()
        sel.click(1, KEYS)
        sel.click(3, KEYS, shift=True)
        assert sel.selected == {"b.jpg", "c.jpg", "d.jpg"}

    def test_shift_click_range_backward(self):
        sel = SelectionManager()
        sel.click(3, KEYS)
        sel.click(1, KEYS, shift=True)
        assert sel.selected == {"b.jpg", "c.jpg", "d.jpg"}

    def test_shift_click_without_prior_click_selects_single(self):
        sel = SelectionManager()
        sel.click(2, KEYS, shift=True)
        assert sel.selected == {"c.jpg"}

    def test_shift_click_extends_existing_selection(self):
        sel = SelectionManager()
        sel.click(0, KEYS)  # select a
        sel.click(2, KEYS)  # select c (a stays selected)
        sel.click(4, KEYS, shift=True)  # range from c to e
        assert sel.selected == {"a.jpg", "c.jpg", "d.jpg", "e.jpg"}


class TestSelectAll:
    def test_select_all(self):
        sel = SelectionManager()
        sel.select_all(KEYS)
        assert sel.selected == set(KEYS)
        assert sel.count() == 5

    def test_select_all_idempotent(self):
        sel = SelectionManager()
        sel.select_all(KEYS)
        sel.select_all(KEYS)
        assert sel.count() == 5


class TestClear:
    def test_clear(self):
        sel = SelectionManager()
        sel.click(0, KEYS)
        sel.click(1, KEYS)
        sel.clear()
        assert sel.count() == 0
        assert sel.last_clicked_idx is None

    def test_clear_then_shift_click_is_single(self):
        sel = SelectionManager()
        sel.click(0, KEYS)
        sel.clear()
        sel.click(3, KEYS, shift=True)
        assert sel.selected == {"d.jpg"}


class TestLastClickedIdx:
    def test_tracks_last_click(self):
        sel = SelectionManager()
        sel.click(2, KEYS)
        assert sel.last_clicked_idx == 2

    def test_shift_click_updates_last_clicked(self):
        sel = SelectionManager()
        sel.click(1, KEYS)
        sel.click(3, KEYS, shift=True)
        assert sel.last_clicked_idx == 3

    def test_clear_resets_last_clicked(self):
        sel = SelectionManager()
        sel.click(2, KEYS)
        sel.clear()
        assert sel.last_clicked_idx is None
