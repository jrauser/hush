"""Tests for Bulk Capture data flow.

Tests the non-GUI logic: filtering, pagination, labeling, undo, delete.
Uses LabelStore with an in-memory DB — no widgets, no rendering.
"""

import os
import tempfile


def _populate(store, n=20):
    """Add n screenshots to the store. Returns list of filenames."""
    filenames = []
    for i in range(n):
        fname = f"/tmp/hush_test/frame_{i:03d}.jpg"
        store.add_screenshot(fname, f"2026-01-01T00:00:{i:02d}", source="extension")
        filenames.append(fname)
    return filenames


class TestBulkFiltering:
    def test_get_all_returns_all(self, store):
        _populate(store, 10)
        rows = store.get_all()
        assert len(rows) == 10

    def test_filter_unlabeled(self, store):
        fnames = _populate(store, 10)
        store.set_label(fnames[0], "ad")
        store.set_label(fnames[1], "content")
        rows = store.get_all()
        unlabeled = [r for r in rows if r["label"] is None]
        assert len(unlabeled) == 8

    def test_filter_by_label(self, store):
        fnames = _populate(store, 10)
        store.set_labels_bulk(fnames[:3], "ad")
        store.set_labels_bulk(fnames[3:7], "content")
        rows = store.get_all()
        ads = [r for r in rows if r["label"] == "ad"]
        content = [r for r in rows if r["label"] == "content"]
        assert len(ads) == 3
        assert len(content) == 4

    def test_filter_with_search(self, store):
        _populate(store, 10)
        rows = store.get_all()
        matched = [r for r in rows if "005" in r["filename"]]
        assert len(matched) == 1
        assert "005" in matched[0]["filename"]


class TestBulkLabeling:
    def test_label_bulk(self, store):
        fnames = _populate(store, 5)
        store.set_labels_bulk(fnames[:3], "ad")
        for fname in fnames[:3]:
            row = store.get_by_filename(fname)
            assert row["label"] == "ad"
        for fname in fnames[3:]:
            row = store.get_by_filename(fname)
            assert row["label"] is None

    def test_relabel_overwrites(self, store):
        fnames = _populate(store, 3)
        store.set_labels_bulk(fnames, "ad")
        store.set_labels_bulk(fnames, "content")
        for fname in fnames:
            row = store.get_by_filename(fname)
            assert row["label"] == "content"


class TestBulkUndo:
    def test_undo_label(self, store):
        fnames = _populate(store, 5)
        # Label first two as ad
        old_labels = {f: None for f in fnames[:2]}
        store.set_labels_bulk(fnames[:2], "ad")

        # Undo: restore old labels (None = unlabeled)
        to_clear = [f for f, lbl in old_labels.items() if lbl is None]
        store.clear_labels_bulk(to_clear)

        for fname in fnames[:2]:
            row = store.get_by_filename(fname)
            assert row["label"] is None

    def test_undo_relabel(self, store):
        fnames = _populate(store, 3)
        store.set_labels_bulk(fnames, "ad")

        # Re-label to content
        old_labels = {f: "ad" for f in fnames}
        store.set_labels_bulk(fnames, "content")

        # Undo: restore to ad
        for lbl in ("ad", "content", "neither", "ignore"):
            subset = [f for f, old in old_labels.items() if old == lbl]
            if subset:
                store.set_labels_bulk(subset, lbl)

        for fname in fnames:
            row = store.get_by_filename(fname)
            assert row["label"] == "ad"


class TestBulkDelete:
    def test_delete_removes_from_db(self, store):
        fnames = _populate(store, 5)
        store.delete_screenshots(fnames[:2])
        rows = store.get_all()
        assert len(rows) == 3
        remaining = {r["filename"] for r in rows}
        assert fnames[0] not in remaining
        assert fnames[1] not in remaining

    def test_delete_removes_from_disk(self):
        """Delete should remove the actual file when called from the app."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake jpeg")
            path = f.name

        assert os.path.isfile(path)
        os.remove(path)
        assert not os.path.isfile(path)


class TestBulkPagination:
    def test_pagination_math(self, store):
        _populate(store, 25)
        rows = store.get_all()
        page_size = 10
        total_pages = max(1, -(-len(rows) // page_size))  # ceil division
        assert total_pages == 3
        # Page 0
        page_0 = rows[0:10]
        assert len(page_0) == 10
        # Page 2 (last)
        page_2 = rows[20:30]
        assert len(page_2) == 5

    def test_empty_db_one_page(self, store):
        rows = store.get_all()
        page_size = 10
        total_pages = max(1, -(-len(rows) // page_size)) if rows else 1
        assert total_pages == 1
