import pytest
from hush.label_store import LabelStore


@pytest.fixture
def store():
    """LabelStore backed by an in-memory SQLite database."""
    return LabelStore(":memory:")
