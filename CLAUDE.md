# SocietySim — Project Context

## Core question
Does narrative LLM reasoning *extend* or *distort* classic Sugarscape ABM dynamics?
Replication-first philosophy: match baseline Sugarscape outputs before adding complexity.

## Stack
- Python, SQLite (all results logged), async LLM calls
- LLM abstraction layer: swap Ollama (local, dev/testing) vs Anthropic Haiku (full runs)
- Local model: llama3.2:3b via Ollama (GTX 1080 Ti 11GB)
- Anthropic model: claude-haiku-4-5-20251001 (pinned for reproducibility)
- Ollama is a separate app install (ollama.com), not just a pip package

## Sugarscape parameters (E&A standard)
- Grid: 50×50 toroidal, two-peak sugar distribution (peaks at ~(15,15) and (35,35))
- Population: 250 agents
- Turn structure: random sequential — agents shuffle each tick, act one at a time (E&A original)
- Vision: uniform {1–6}, Metabolism: uniform {1–4}, Initial sugar: uniform {5,10,15,20,25}
- Rule G growback: +1/tick up to capacity

## Expected baseline Gini trajectory
- Ticks 0–50: rapid rise from ~0.2 → ~0.45
- Ticks 50–150: continued rise, population drops 250 → ~150–200
- Ticks 150+: steady state ~0.5–0.6
- Key result: Gini ~0.5 emerges endogenously from heterogeneous vision/metabolism alone

## Experiment structure

### Exp 1 — Replication Check (`experiments/exp1_replication.py`)
| Condition | Description | Runs |
|-----------|-------------|------|
| A | Baseline, no LLM | 10 |
| B | LLM, neutral persona | 10 |
| C | LLM, typed personas (greedy / subsistence / risk-seeking / risk-averse) | 10 |
| D | LLM, narrative backstory | 10 |

Metric: Gini trajectory, KS test vs baseline at T=250.

### Exp 2 — Disease + Misinformation (`experiments/exp2_disease.py`)
| Condition | Description | Runs |
|-----------|-------------|------|
| A | Baseline disease, no LLM | 10 |
| B | LLM agents, 0% misinformation | 10 |
| C | LLM agents, 1% misinformation | 10 |
| D | LLM agents, 5% misinformation | 10 |
| E | LLM agents, 10% misinformation | 10 |

Disease: 50-bit immune bitstrings, Hamming-distance transmission, 10-tick recovery.
Misinformation: false belief that proximity builds resistance (adversarially chosen to increase contact rates).
Metrics: epidemic curve, R0 estimate, belief spread curve.

### Exp 3–5 (rough, not yet implemented)
- **Exp 3**: Norm emergence on a commons
- **Exp 4**: Bounded rationality via context window length
- **Exp 5**: Migration with narrative prejudice

## How the LLM is used

### Exp 1 — Movement decisions
In the baseline, each agent uses a hard-coded greedy rule (Rule M): move to the highest-sugar visible cell.
In LLM conditions, this decision is replaced by an LLM call **every tick, per agent**. The LLM is not talking to other agents — it is acting as the *decision-making process* of a single agent.

Each tick, the agent sends the LLM:
- Its current sugar reserves, metabolism rate, and how many ticks until starvation
- A list of cells it can legally move to (within vision range, unoccupied), with the sugar amount at each
- Optionally: a character/persona prompt (Conditions C and D)

The LLM must respond with a single cell coordinate (`ROW,COL`). If the response fails to parse or names an illegal cell, the code falls back to greedy Rule M and logs the failure.

**Key point**: physical constraints are hard-coded and non-negotiable. The LLM cannot choose to skip its turn, move further than its vision, or avoid dying when sugar hits 0. It only controls *which legal cell to move to*.

**Persona types (Exp 1C/1D)**:
- *Neutral*: no persona, just raw state info
- *Typed*: a one-sentence disposition (e.g. "you prioritize accumulation above all else")
- *Narrative*: a 2–3 sentence backstory generated once at agent initialization (occupation, origin, attitude toward scarcity)

### Exp 2 — Contact dialogue
In addition to movement, Exp 2 adds a **contact event** when two agents are on adjacent cells. The LLM plays both sides of a two-turn dialogue:

1. **Agent A** is prompted with its health status and disease belief, and must choose: `WARN: <message>`, `LIE: <message>`, or `SILENT`
2. **Agent B** receives A's message (if not SILENT) and must respond: `BELIEVE`, `SKEPTICAL`, or `IGNORE`

If B responds `BELIEVE`, B adopts A's `disease_belief` string. This is how misinformation spreads socially.

**Disease physics are still hard-coded**: transmission is computed via Hamming distance on immune bitstrings regardless of what the agents say to each other. The LLM dialogue affects *beliefs and future behavior*, not the physics.

## Key implementation notes
- LLM move fallback: parse failure → greedy Rule M; fallback rate is itself a metric
- LLM conditions auto-skip when no backend is configured (never silently run as baseline)
- `sim.run(print_every=50)` controls per-tick progress output
- DB: `data/exp1.db`, `data/exp2.db` — tables: `runs`, `agents`, `tick_stats`, `agent_ticks`, `llm_calls`, `contacts`

## Running

```bash
# Install
pip install -e ".[dev]"

# Baseline only
python experiments/exp1_replication.py --backend baseline

# Local LLM (requires Ollama running + model pulled)
ollama pull llama3.2:3b
python experiments/exp1_replication.py --backend ollama

# Anthropic Haiku (requires ANTHROPIC_API_KEY)
python experiments/exp1_replication.py --backend anthropic
```
