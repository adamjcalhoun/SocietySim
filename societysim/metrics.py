import numpy as np
from scipy import stats


def gini(values: list[float]) -> float:
    """Gini coefficient over living agent sugar reserves."""
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() - (n + 1) * arr.sum()) / (n * arr.sum()))


def ks_compare(baseline: list[float], condition: list[float]):
    """KS test between two Gini trajectories. Returns (statistic, p_value)."""
    return stats.ks_2samp(baseline, condition)


def tick_summary(agents: list) -> dict:
    living = [a for a in agents if a.alive]
    sugars = [a.sugar for a in living]
    infected = sum(1 for a in living if a.infected)
    return {
        "population": len(living),
        "gini": gini(sugars) if sugars else 0.0,
        "mean_sugar": float(np.mean(sugars)) if sugars else 0.0,
        "median_sugar": float(np.median(sugars)) if sugars else 0.0,
        "max_sugar": float(np.max(sugars)) if sugars else 0.0,
        "infected_count": infected,
    }
