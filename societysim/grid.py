import math
import numpy as np


class Grid:
    """50x50 toroidal Sugarscape grid with two-peak sugar distribution."""

    def __init__(self, width: int = 50, height: int = 50, rng: np.random.Generator = None):
        self.width = width
        self.height = height
        self.rng = rng or np.random.default_rng()
        self.capacity = self._init_sugar_capacity()
        self.sugar = self.capacity.astype(float).copy()
        self.occupancy: dict[tuple, int] = {}  # (r, c) -> agent_id

    def _torus_dist(self, r1: int, c1: int, r2: int, c2: int) -> float:
        dr = min(abs(r1 - r2), self.height - abs(r1 - r2))
        dc = min(abs(c1 - c2), self.width - abs(c1 - c2))
        return math.sqrt(dr ** 2 + dc ** 2)

    def _init_sugar_capacity(self) -> np.ndarray:
        """Two-peak distribution matching E&A Figure II-2.

        Uses equal-width bands so each peak has a proper plateau rather than
        collapsing to a single cell at max capacity (the int-truncation bug).
        Band width = radius / max_cap = 16/4 = 4 cells per level.
        """
        capacity = np.zeros((self.height, self.width), dtype=np.int8)
        peaks = [(15, 15), (35, 35)]
        max_cap = 4
        band = 4.0  # cells per capacity level
        for r in range(self.height):
            for c in range(self.width):
                min_dist = min(self._torus_dist(r, c, pr, pc) for pr, pc in peaks)
                capacity[r, c] = max(0, max_cap - int(min_dist / band))
        return capacity

    def growback(self):
        """Rule G: increment sugar by 1 per tick up to capacity."""
        self.sugar = np.minimum(self.sugar + 1, self.capacity)

    def get_cells_in_vision(self, pos: tuple, vision: int) -> list[tuple]:
        """Cardinal-direction vision (E&A Rule M). Returns pos + reachable cells."""
        r, c = pos
        cells = {pos}
        for d in range(1, vision + 1):
            cells.add(((r + d) % self.height, c))
            cells.add(((r - d) % self.height, c))
            cells.add((r, (c + d) % self.width))
            cells.add((r, (c - d) % self.width))
        return list(cells)

    def available_cells(self, pos: tuple, vision: int) -> list[tuple]:
        """Visible cells that are unoccupied (agent can move there), plus current pos."""
        visible = self.get_cells_in_vision(pos, vision)
        return [c for c in visible if c == pos or c not in self.occupancy]

    def place_agent(self, agent_id: int, pos: tuple):
        self.occupancy[pos] = agent_id

    def move_agent(self, agent_id: int, old_pos: tuple, new_pos: tuple):
        self.occupancy.pop(old_pos, None)
        self.occupancy[new_pos] = agent_id

    def remove_agent(self, pos: tuple):
        self.occupancy.pop(pos, None)

    def harvest(self, pos: tuple) -> float:
        amount = float(self.sugar[pos])
        self.sugar[pos] = 0.0
        return amount

    def adjacent_occupied(self, pos: tuple) -> list[int]:
        """Agent IDs on the four cardinal neighbors (for contact events)."""
        r, c = pos
        neighbors = [
            ((r + 1) % self.height, c),
            ((r - 1) % self.height, c),
            (r, (c + 1) % self.width),
            (r, (c - 1) % self.width),
        ]
        return [self.occupancy[n] for n in neighbors if n in self.occupancy]
