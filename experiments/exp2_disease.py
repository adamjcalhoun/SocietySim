"""
Experiment 2: Disease + Misinformation
Conditions: A=baseline disease, B=LLM 0%, C=LLM 1%, D=LLM 5%, E=LLM 10% misinformation
"""
import asyncio
import uuid
from pathlib import Path

import numpy as np

from societysim.agent import make_agents
from societysim.db import DB
from societysim.grid import Grid
from societysim.simulation import Simulation, MISINFORMATION

N_AGENTS = 250
N_TICKS = 500
N_RUNS = 10
DB_PATH = Path("data/exp2.db")

DISEASE_STRING = list(np.random.default_rng(0).integers(0, 2, 50))

CONDITIONS = {
    "A": {"use_llm": False, "misinfo_pct": 0.0},
    "B": {"use_llm": True,  "misinfo_pct": 0.0},
    "C": {"use_llm": True,  "misinfo_pct": 0.01},
    "D": {"use_llm": True,  "misinfo_pct": 0.05},
    "E": {"use_llm": True,  "misinfo_pct": 0.10},
}


def _seed_disease(agents, rng, initial_infected_pct=0.05):
    n = max(1, int(len(agents) * initial_infected_pct))
    for agent in rng.choice(agents, size=n, replace=False):
        agent.infected = True
        agent.infection_tick = 0
        agent.recovery_tick = 10


def _seed_misinformation(agents, rng, pct: float):
    if pct <= 0:
        return
    n = max(1, int(len(agents) * pct))
    for agent in rng.choice(agents, size=n, replace=False):
        agent.disease_belief = MISINFORMATION


async def run_condition(condition: str, seed: int, llm=None, db: DB = None):
    cfg = CONDITIONS[condition]
    rng = np.random.default_rng(seed)
    grid = Grid(rng=rng)
    agents = make_agents(N_AGENTS, grid, rng)

    _seed_disease(agents, rng)
    _seed_misinformation(agents, rng, cfg["misinfo_pct"])

    run_id = f"exp2_{condition}_{seed}_{uuid.uuid4().hex[:6]}"

    if db:
        backend = type(llm).__name__ if llm else None
        model = getattr(llm, "model", None)
        db.log_run(run_id, "exp2", condition, seed, N_AGENTS, N_TICKS, backend, model)
        db.log_agents(run_id, agents)

    sim = Simulation(
        grid, agents, rng=rng, llm=llm, db=db, run_id=run_id,
        log_agent_ticks=True,
    )
    history = await sim.run(
        ticks=N_TICKS,
        use_llm=cfg["use_llm"],
        disease=DISEASE_STRING,
    )
    return run_id, history


async def main(llm=None):
    db = DB(DB_PATH)
    seeds = list(range(N_RUNS))

    for condition in CONDITIONS:
        print(f"\n=== Condition {condition} ===")
        for seed in seeds:
            run_id, history = await run_condition(condition, seed, llm=llm, db=db)
            final = history[-1]
            print(
                f"  seed={seed} | ticks={len(history)} | "
                f"pop={final['population']} | infected={final['infected_count']} | "
                f"gini={final['gini']:.3f}"
            )

    db.close()
    print("\nDone. Results in", DB_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["baseline", "ollama", "anthropic"], default="baseline")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    llm = None
    if args.backend == "ollama":
        from societysim.llm.ollama_client import OllamaClient
        llm = OllamaClient(model=args.model or "llama3.2:3b")
    elif args.backend == "anthropic":
        from societysim.llm.anthropic_client import AnthropicClient
        llm = AnthropicClient(model=args.model or "claude-haiku-4-5-20251001")

    asyncio.run(main(llm=llm))
