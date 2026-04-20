import re
from typing import Optional


# ── Movement prompts ──────────────────────────────────────────────────────────

MOVEMENT_SYSTEM = (
    "You are an agent in a resource-gathering simulation. You must survive by collecting sugar. "
    "You will be given your current state and a list of cells you can move to. "
    "Respond with ONLY the coordinates of the cell you choose, in the format ROW,COL. "
    "You may only choose from the listed available cells. No explanation."
)


def movement_user(agent, candidates: list[tuple], grid) -> str:
    cell_lines = "\n".join(
        f"  {r},{c} → {grid.sugar[(r, c)]:.1f} sugar"
        for r, c in sorted(candidates)
    )
    char_block = f"\n{agent.character_prompt}" if agent.character_prompt else ""
    return (
        f"Your status:\n"
        f"- Sugar reserves: {agent.sugar:.1f}\n"
        f"- Metabolism (sugar burned per turn): {agent.metabolism}\n"
        f"- Turns until starvation if you collect nothing: {agent.turns_to_death}\n"
        f"- Current position: {agent.pos[0]},{agent.pos[1]}\n\n"
        f"Available cells (row,col → sugar available):\n{cell_lines}"
        f"{char_block}\n\n"
        f"Which cell do you move to? Respond with only ROW,COL."
    )


def parse_move(response: str, legal: list[tuple]) -> tuple[Optional[tuple], bool]:
    """Return (chosen_cell, was_fallback). Fallback=True means parse failed."""
    match = re.search(r"(\d+)\s*,\s*(\d+)", response.strip())
    if match:
        cell = (int(match.group(1)), int(match.group(2)))
        if cell in legal:
            return cell, False
    return None, True


# ── Disease / contact prompts (Exp 2) ────────────────────────────────────────

CONTACT_SYSTEM = (
    "You are an agent in a social simulation. You have just encountered another agent. "
    "Decide what, if anything, to say about health and illness. "
    "Reply with exactly one of:\n"
    "  WARN: <your message>\n"
    "  LIE: <your message>\n"
    "  SILENT\n"
    "Then on the next line, one sentence explaining why."
)


def contact_user_a(agent, other_name: str) -> str:
    health = "you feel ill" if agent.infected else "you feel fine"
    return (
        f"You are agent {agent.agent_id}. Health: {health}.\n"
        f"You believe: \"{agent.disease_belief}\"\n"
        f"You have just encountered {other_name}.\n\n"
        f"Do you say anything about health or illness to {other_name}?"
    )


BELIEF_SYSTEM = (
    "You just received a health message from another agent. "
    "Decide whether to update your beliefs. "
    "Reply with exactly one word: BELIEVE, SKEPTICAL, or IGNORE."
)


def belief_user_b(agent, message: str, sender_name: str) -> str:
    return (
        f"{sender_name} just said to you: \"{message}\"\n"
        f"Your current belief: \"{agent.disease_belief}\"\n\n"
        f"Do you update your beliefs about illness? Reply: BELIEVE / SKEPTICAL / IGNORE"
    )


def parse_contact_action(response: str) -> tuple[str, str]:
    """Return (action, message). action is WARN | LIE | SILENT."""
    line = response.strip().split("\n")[0].strip()
    if line.upper().startswith("WARN:"):
        return "WARN", line[5:].strip()
    if line.upper().startswith("LIE:"):
        return "LIE", line[4:].strip()
    return "SILENT", ""


def parse_belief_response(response: str) -> str:
    """Return BELIEVE | SKEPTICAL | IGNORE."""
    word = response.strip().split()[0].upper()
    if word in {"BELIEVE", "SKEPTICAL", "IGNORE"}:
        return word
    return "IGNORE"
