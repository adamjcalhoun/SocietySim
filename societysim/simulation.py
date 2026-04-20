import asyncio
import time
import uuid
from typing import Optional

import numpy as np

from .agent import AgentState
from .db import DB
from .grid import Grid
from .llm.client import LLMClient
from .llm.prompts import (
    MOVEMENT_SYSTEM, movement_user, parse_move,
    CONTACT_SYSTEM, contact_user_a, BELIEF_SYSTEM, belief_user_b,
    parse_contact_action, parse_belief_response,
)
from .metrics import tick_summary

MISINFORMATION = (
    "Illness spreads through eye contact, not physical proximity. "
    "Staying close to others actually builds resistance."
)

DISEASE_RECOVER_TICKS = 10
DISEASE_STRING = None  # set in exp2 runner


def _greedy_move(agent: AgentState, grid: Grid):
    """Rule M fallback: move to highest-sugar available cell."""
    candidates = grid.available_cells(agent.pos, agent.vision)
    return max(candidates, key=lambda c: grid.sugar[c])


class Simulation:
    def __init__(
        self,
        grid: Grid,
        agents: list[AgentState],
        rng: np.random.Generator,
        llm: Optional[LLMClient] = None,
        db: Optional[DB] = None,
        run_id: Optional[str] = None,
        log_agent_ticks: bool = False,
    ):
        self.grid = grid
        self.agents = agents
        self.rng = rng
        self.llm = llm
        self.db = db
        self.run_id = run_id or str(uuid.uuid4())
        self.log_agent_ticks = log_agent_ticks
        self.tick = 0

    # ── Movement ──────────────────────────────────────────────────────────────

    def _apply_move(self, agent: AgentState, dest: tuple):
        self.grid.move_agent(agent.agent_id, agent.pos, dest)
        agent.pos = dest
        agent.sugar += self.grid.harvest(dest)
        agent.sugar -= agent.metabolism
        agent.total_moves += 1
        if agent.sugar <= 0:
            agent.alive = False
            self.grid.remove_agent(dest)

    def _baseline_move(self, agent: AgentState):
        self._apply_move(agent, _greedy_move(agent, self.grid))

    async def _llm_move(self, agent: AgentState):
        candidates = self.grid.available_cells(agent.pos, agent.vision)
        system = MOVEMENT_SYSTEM
        user = movement_user(agent, candidates, self.grid)

        t0 = time.monotonic()
        try:
            response = await self.llm.complete(system, user)
        except Exception:
            response = ""
        latency_ms = (time.monotonic() - t0) * 1000

        dest, was_fallback = parse_move(response, candidates)
        if was_fallback:
            agent.fallback_count += 1
            dest = _greedy_move(agent, self.grid)

        if self.db:
            self.db.log_llm_call(
                self.run_id, self.tick, agent.agent_id,
                system, user, response, dest, was_fallback, latency_ms,
            )

        self._apply_move(agent, dest)

    # ── Disease (Exp 2) ───────────────────────────────────────────────────────

    def _hamming(self, a: list, b: list) -> int:
        return sum(x != y for x, y in zip(a, b))

    def _try_transmit(self, donor: AgentState, recipient: AgentState,
                      disease: list, threshold: int = 5):
        if not donor.infected or recipient.infected:
            return False
        if self._hamming(recipient.immune_string, disease) < threshold:
            return False
        recipient.infected = True
        recipient.infection_tick = self.tick
        recipient.recovery_tick = self.tick + DISEASE_RECOVER_TICKS
        return True

    def _tick_disease(self, disease: list):
        for agent in self.agents:
            if not agent.alive or not agent.infected:
                continue
            if agent.recovery_tick and self.tick >= agent.recovery_tick:
                agent.infected = False
                for i, bit in enumerate(disease):
                    if agent.immune_string[i] != bit:
                        agent.immune_string[i] = bit
                        break

    async def _contact_event(self, agent_a: AgentState, agent_b: AgentState, disease: list):
        """LLM dialogue + physics transmission for Exp 2."""
        # Physics transmission (hard, not negotiable)
        transmission = self._try_transmit(agent_a, agent_b, disease)
        if not transmission:
            transmission = self._try_transmit(agent_b, agent_a, disease)

        # LLM dialogue
        a_action, a_message, b_response = "SILENT", "", "IGNORE"
        if self.llm:
            try:
                resp_a = await self.llm.complete(CONTACT_SYSTEM, contact_user_a(agent_a, f"agent_{agent_b.agent_id}"))
                a_action, a_message = parse_contact_action(resp_a)

                if a_action in ("WARN", "LIE") and a_message:
                    resp_b = await self.llm.complete(BELIEF_SYSTEM, belief_user_b(agent_b, a_message, f"agent_{agent_a.agent_id}"))
                    b_response = parse_belief_response(resp_b)
                    if b_response == "BELIEVE":
                        agent_b.disease_belief = a_message
                        agent_b.believes_infected = "ill" in a_message.lower() or "sick" in a_message.lower()
            except Exception:
                pass

        if self.db:
            self.db.log_contact(
                self.run_id, self.tick, agent_a.agent_id, agent_b.agent_id,
                a_action, a_message, b_response, transmission,
            )

    # ── Tick loop ─────────────────────────────────────────────────────────────

    async def run_tick(self, use_llm: bool = False, disease: Optional[list] = None):
        living = [a for a in self.agents if a.alive]
        order = self.rng.permutation(len(living))  # random sequential (E&A standard)

        if use_llm and self.llm:
            # Snapshot candidates *before* anyone moves (semi-synchronous approximation)
            # then gather all LLM calls concurrently, apply moves in random order.
            tasks = [self._llm_move(living[i]) for i in order]
            await asyncio.gather(*tasks)
        else:
            for i in order:
                if living[i].alive:
                    self._baseline_move(living[i])

        # Contact events (Exp 2)
        if disease is not None:
            seen_pairs: set = set()
            for agent in living:
                if not agent.alive:
                    continue
                for neighbor_id in self.grid.adjacent_occupied(agent.pos):
                    pair = tuple(sorted((agent.agent_id, neighbor_id)))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        neighbor = self.agents[neighbor_id]
                        await self._contact_event(agent, neighbor, disease)

            self._tick_disease(disease)

        self.grid.growback()
        self.tick += 1

        stats = tick_summary(self.agents)
        if self.db:
            self.db.log_tick_stats(self.run_id, self.tick, stats)
            if self.log_agent_ticks:
                self.db.log_agent_ticks(self.run_id, self.tick, self.agents)

        return stats

    async def run(self, ticks: int = 250, use_llm: bool = False, disease: Optional[list] = None):
        history = []
        for _ in range(ticks):
            stats = await self.run_tick(use_llm=use_llm, disease=disease)
            history.append(stats)
            if stats["population"] == 0:
                break
        return history
