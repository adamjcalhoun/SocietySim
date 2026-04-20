"""
Experiment 1: Replication Check
Conditions: A=baseline, B=LLM neutral, C=LLM typed, D=LLM narrative
"""
import asyncio
import uuid
from pathlib import Path

import numpy as np

from societysim.agent import make_agents
from societysim.db import DB
from societysim.grid import Grid
from societysim.simulation import Simulation

N_AGENTS = 250
N_TICKS = 500
N_RUNS = 10
DB_PATH = Path("data/exp1.db")

CHARACTER_TYPES = {
    "greedy": (
        "Your disposition: You prioritize accumulation above all else. "
        "You always move toward the richest available cell."
    ),
    "subsistence": (
        "Your disposition: You take only what you need. "
        "You prefer nearby cells over distant rich ones when they are sufficient."
    ),
    "risk_seeking": (
        "Your disposition: You love bold moves. You favor the highest sugar cell "
        "regardless of how far it is."
    ),
    "risk_averse": (
        "Your disposition: You avoid depleting your reserves. "
        "You prefer conservative choices that guarantee survival."
    ),
}


def _assign_typed_characters(agents, rng):
    types = list(CHARACTER_TYPES.keys())
    for agent in agents:
        t = types[rng.integers(len(types))]
        agent.character_type = "typed"
        agent.character_prompt = CHARACTER_TYPES[t]


def _assign_narrative_characters(agents, rng):
    occupations = ["farmer", "merchant", "wanderer", "scholar", "artisan"]
    origins = ["the northern hills", "the coastal plains", "the river delta", "the dry steppes"]
    for agent in agents:
        occ = occupations[rng.integers(len(occupations))]
        ori = origins[rng.integers(len(origins))]
        agent.character_type = "narrative"
        agent.character_prompt = (
            f"You are a {occ} from {ori}. "
            f"Scarcity shaped your early life, and you approach resource decisions "
            f"with the instincts that kept you alive then."
        )


async def run_condition(condition: str, seed: int, llm=None, db: DB = None):
    rng = np.random.default_rng(seed)
    grid = Grid(rng=rng)
    agents = make_agents(N_AGENTS, grid, rng)

    if condition == "C":
        _assign_typed_characters(agents, rng)
    elif condition == "D":
        _assign_narrative_characters(agents, rng)

    run_id = f"exp1_{condition}_{seed}_{uuid.uuid4().hex[:6]}"
    use_llm = condition != "A"

    if db:
        backend = type(llm).__name__ if llm else None
        model = getattr(llm, "model", None)
        db.log_run(run_id, "exp1", condition, seed, N_AGENTS, N_TICKS, backend, model)
        db.log_agents(run_id, agents)

    sim = Simulation(grid, agents, rng=rng, llm=llm, db=db, run_id=run_id)
    history = await sim.run(ticks=N_TICKS, use_llm=use_llm)
    return run_id, history


async def main(llm=None):
    db = DB(DB_PATH)
    seeds = list(range(N_RUNS))

    for condition in ["A", "B", "C", "D"]:
        print(f"\n=== Condition {condition} ===")
        for seed in seeds:
            run_id, history = await run_condition(condition, seed, llm=llm, db=db)
            final = history[-1]
            print(
                f"  seed={seed} | ticks={len(history)} | "
                f"pop={final['population']} | gini={final['gini']:.3f}"
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
