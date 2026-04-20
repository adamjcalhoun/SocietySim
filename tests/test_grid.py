import numpy as np
import pytest
from societysim.grid import Grid


@pytest.fixture
def grid():
    return Grid(rng=np.random.default_rng(42))


def test_sugar_capacity_range(grid):
    assert grid.capacity.min() >= 0
    assert grid.capacity.max() <= 4


def test_two_peaks_exist(grid):
    # Both peaks should be near max capacity
    assert grid.capacity[15, 15] == 4
    assert grid.capacity[35, 35] == 4


def test_growback_does_not_exceed_capacity(grid):
    grid.sugar[:] = 0
    grid.growback()
    assert (grid.sugar <= grid.capacity).all()


def test_growback_increments(grid):
    grid.sugar[:] = 0
    grid.growback()
    # Cells with capacity > 0 should now have sugar = 1
    assert (grid.sugar[grid.capacity > 0] == 1).all()


def test_toroidal_vision_wraps(grid):
    cells = grid.get_cells_in_vision((0, 0), vision=2)
    # Should include wrapped cells
    assert (48, 0) in cells  # row wrap
    assert (0, 48) in cells  # col wrap


def test_vision_count(grid):
    cells = grid.get_cells_in_vision((25, 25), vision=3)
    # 1 (center) + 4 directions × 3 = 13
    assert len(cells) == 13


def test_harvest_zeros_cell(grid):
    grid.sugar[10, 10] = 3.0
    amount = grid.harvest((10, 10))
    assert amount == 3.0
    assert grid.sugar[10, 10] == 0.0


def test_occupancy(grid):
    grid.place_agent(0, (5, 5))
    assert (5, 5) in grid.occupancy
    grid.move_agent(0, (5, 5), (6, 6))
    assert (5, 5) not in grid.occupancy
    assert grid.occupancy[(6, 6)] == 0
