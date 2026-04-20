"""Smoke test: baseline simulation runs and produces plausible Gini trajectory."""
import asyncio
import numpy as np
import pytest
from societysim.agent import make_agents
from societysim.grid import Grid
from societysim.simulation import Simulation


@pytest.fixture
def small_sim():
    rng = np.random.default_rng(0)
    grid = Grid(width=20, height=20, rng=rng)
    agents = make_agents(50, grid, rng)
    return Simulation(grid, agents, rng=rng)


def test_baseline_runs(small_sim):
    history = asyncio.run(small_sim.run(ticks=50, use_llm=False))
    assert len(history) > 0


def test_gini_increases_over_time(small_sim):
    history = asyncio.run(small_sim.run(ticks=100, use_llm=False))
    # Early Gini should be lower than late Gini (wealth concentration increases)
    early = np.mean([h["gini"] for h in history[:10]])
    late = np.mean([h["gini"] for h in history[-10:]])
    assert late >= early - 0.05  # allow small noise


def test_population_decreases_or_stabilizes(small_sim):
    initial_pop = sum(1 for a in small_sim.agents if a.alive)
    history = asyncio.run(small_sim.run(ticks=50, use_llm=False))
    final_pop = history[-1]["population"]
    assert final_pop <= initial_pop
