"""Split integrity: perimeter never in fit; hashes stable (PRD §4.5 test_splits.py)."""

import numpy as np

from bapinnsformer.data.splits import (
    leave_one_out,
    split_hash,
    split_interior_perimeter,
    temporal_blocks,
)

BOUNDS = (0.0, 100.0, 0.0, 100.0)


def _grid(n: int = 5):
    xs = np.linspace(5, 95, n)
    return [{"id": f"S{i:02d}_{j:02d}", "x": float(x), "y": float(y)}
            for i, x in enumerate(xs) for j, y in enumerate(xs)]


def test_perimeter_withheld_and_disjoint():
    split = split_interior_perimeter(_grid(), BOUNDS, buffer=20.0)
    fit, val = set(split["fit_ids"]), set(split["val_ids"])
    assert fit and val
    assert fit.isdisjoint(val)
    assert fit | val == {s["id"] for s in _grid()}


def test_perimeter_is_near_edge():
    stations = _grid()
    split = split_interior_perimeter(stations, BOUNDS, buffer=20.0)
    by_id = {s["id"]: s for s in stations}
    for sid in split["val_ids"]:
        s = by_id[sid]
        d = min(s["x"] - 0.0, 100.0 - s["x"], s["y"] - 0.0, 100.0 - s["y"])
        assert d <= 20.0 + 1e-9


def test_hash_stable_and_order_invariant():
    stations = _grid()
    h1 = split_interior_perimeter(stations, BOUNDS)["hash"]
    h2 = split_interior_perimeter(list(reversed(stations)), BOUNDS)["hash"]
    assert h1 == h2
    assert split_hash({"a": [1, 2]}) == split_hash({"a": [1, 2]})
    assert split_hash({"a": [1, 2]}) != split_hash({"a": [1, 3]})


def test_leave_one_out_covers_fit_set():
    ids = ["a", "b", "c"]
    folds = list(leave_one_out(ids))
    assert len(folds) == 3
    for train, held in folds:
        assert len(held) == 1 and set(train) | set(held) == set(ids)


def test_temporal_blocks_partition():
    blocks = temporal_blocks(10, 3)
    assert blocks[0][0] == 0 and blocks[-1][1] == 10
    flat = [i for a, b in blocks for i in range(a, b)]
    assert flat == list(range(10))
