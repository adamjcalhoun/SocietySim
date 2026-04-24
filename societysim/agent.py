from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AgentState:
    agent_id: int
    pos: tuple          # (row, col)
    sugar: float
    metabolism: int     # sugar burned per tick
    vision: int         # cardinal cells visible

    alive: bool = True

    # Persona (Exp 1c / 1d)
    character_type: str = "neutral"   # neutral | typed | narrative
    character_prompt: str = ""

    # Disease (Exp 2)
    infected: bool = False
    infection_tick: Optional[int] = None
    recovery_tick: Optional[int] = None
    immune_string: list = field(default_factory=lambda: list(np.random.randint(0, 2, 50)))
    believes_infected: bool = False
    disease_belief: str = "standard"
    disclosed_to: set = field(default_factory=set)
    warned_by: set = field(default_factory=set)

    # Lifespan (E&A replacement mechanism)
    age: int = 0
    max_age: int = 0   # set at init; die of old age when age >= max_age

    # Diagnostics
    fallback_count: int = 0
    total_moves: int = 0

    @property
    def turns_to_death(self) -> int:
        if self.metabolism == 0:
            return 9999
        return int(self.sugar // self.metabolism)


def make_agents(n: int, grid, rng: np.random.Generator) -> list[AgentState]:
    """Initialize N agents per E&A spec, placed on non-zero sugar cells."""
    nonzero = list(zip(*np.where(grid.capacity > 0)))
    positions = [tuple(nonzero[i]) for i in rng.choice(len(nonzero), size=n, replace=False)]

    agents = []
    for i, pos in enumerate(positions):
        agent = AgentState(
            agent_id=i,
            pos=pos,
            sugar=float(rng.choice([5, 10, 15, 20, 25])),
            metabolism=int(rng.integers(1, 5)),
            vision=int(rng.integers(1, 7)),
            max_age=int(rng.integers(60, 101)),
        )
        grid.place_agent(i, pos)
        agents.append(agent)
    return agents
