import sqlite3
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    experiment  TEXT,
    condition   TEXT,
    seed        INTEGER,
    n_agents    INTEGER,
    n_ticks     INTEGER,
    llm_backend TEXT,
    llm_model   TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agents (
    run_id          TEXT,
    agent_id        INTEGER,
    vision          INTEGER,
    metabolism      INTEGER,
    initial_sugar   REAL,
    character_type  TEXT,
    character_prompt TEXT,
    PRIMARY KEY (run_id, agent_id)
);

CREATE TABLE IF NOT EXISTS tick_stats (
    run_id          TEXT,
    tick            INTEGER,
    population      INTEGER,
    gini            REAL,
    mean_sugar      REAL,
    median_sugar    REAL,
    max_sugar       REAL,
    infected_count  INTEGER,
    PRIMARY KEY (run_id, tick)
);

CREATE TABLE IF NOT EXISTS agent_ticks (
    run_id      TEXT,
    tick        INTEGER,
    agent_id    INTEGER,
    sugar       REAL,
    pos_row     INTEGER,
    pos_col     INTEGER,
    alive       INTEGER,
    infected    INTEGER,
    disease_belief TEXT,
    PRIMARY KEY (run_id, tick, agent_id)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    run_id          TEXT,
    tick            INTEGER,
    agent_id        INTEGER,
    system_prompt   TEXT,
    user_prompt     TEXT,
    raw_response    TEXT,
    parsed_move     TEXT,
    was_fallback    INTEGER,
    latency_ms      REAL
);

CREATE TABLE IF NOT EXISTS contacts (
    run_id                  TEXT,
    tick                    INTEGER,
    agent_a                 INTEGER,
    agent_b                 INTEGER,
    a_action                TEXT,
    a_message               TEXT,
    b_response              TEXT,
    transmission_occurred   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_tick_stats_run ON tick_stats (run_id);
CREATE INDEX IF NOT EXISTS idx_agent_ticks_run ON agent_ticks (run_id, tick);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls (run_id, tick);
CREATE INDEX IF NOT EXISTS idx_contacts_run ON contacts (run_id, tick);
"""


class DB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def log_run(self, run_id, experiment, condition, seed, n_agents, n_ticks,
                llm_backend=None, llm_model=None):
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
                (run_id, experiment, condition, seed, n_agents, n_ticks, llm_backend, llm_model),
            )

    def log_agents(self, run_id: str, agents: list):
        rows = [
            (run_id, a.agent_id, a.vision, a.metabolism, a.sugar,
             a.character_type, a.character_prompt)
            for a in agents
        ]
        with self._tx() as conn:
            conn.executemany("INSERT OR REPLACE INTO agents VALUES (?,?,?,?,?,?,?)", rows)

    def log_tick_stats(self, run_id: str, tick: int, stats: dict):
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tick_stats VALUES (?,?,?,?,?,?,?,?)",
                (run_id, tick, stats["population"], stats["gini"],
                 stats["mean_sugar"], stats["median_sugar"],
                 stats["max_sugar"], stats["infected_count"]),
            )

    def log_agent_ticks(self, run_id: str, tick: int, agents: list):
        rows = [
            (run_id, tick, a.agent_id, a.sugar, a.pos[0], a.pos[1],
             int(a.alive), int(a.infected), a.disease_belief)
            for a in agents
        ]
        with self._tx() as conn:
            conn.executemany("INSERT OR REPLACE INTO agent_ticks VALUES (?,?,?,?,?,?,?,?,?)", rows)

    def log_llm_call(self, run_id, tick, agent_id, system, user, response,
                     parsed_move, was_fallback, latency_ms):
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, tick, agent_id, system, user, response,
                 str(parsed_move), int(was_fallback), latency_ms),
            )

    def log_contact(self, run_id, tick, agent_a, agent_b, a_action,
                    a_message, b_response, transmission):
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO contacts VALUES (?,?,?,?,?,?,?,?)",
                (run_id, tick, agent_a, agent_b, a_action,
                 a_message, b_response, int(transmission)),
            )

    def close(self):
        self._conn.close()
