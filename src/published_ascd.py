"""ASCD selection on the frozen ridge benchmark; separate Stage 3 run.

Implements Algorithm 1 of Stich, Raj & Jaggi (ICML 2017) with their
zero gradient oracle and exact full-gradient initialization. The shared
ridge coordinate step is retained; see research/STAGE3_ASCD_METHOD_SPEC.md.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np

from benchmark import (
    CASES, CHECK, LAM, MAX_STEPS, SEEDS, TARGET, data, objective,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "published_ascd"


def active_set(estimate, radius):
    """Return the smallest safe ASCD active set (strict exclusion)."""
    upper = np.abs(estimate) + radius
    lower = np.maximum(0.0, np.abs(estimate) - radius)
    order = np.argsort(upper, kind="stable")
    selected = np.ones(len(estimate), dtype=bool)
    total = float(np.dot(lower, lower))
    count = len(estimate)

    for j in order:
        # Keep at least one coordinate. Equality does not permit exclusion.
        if count <= 1 or upper[j] ** 2 >= total / count:
            break
        selected[j] = False
        total -= lower[j] ** 2
        count -= 1

    return np.flatnonzero(selected), lower


def run(x, y, seed, verify=False):
    n, d = x.shape
    rng = np.random.default_rng(seed + 9812)
    matrix = x.T @ x + LAM * np.eye(d)
    optimum = np.linalg.solve(matrix, x.T @ y)
    optimal_value = objective(x @ optimum - y, optimum)
    smoothness = np.diag(matrix)
    norms = np.linalg.norm(x, axis=0)

    weights = np.zeros(d)
    residual = -y.copy()
    start = time.perf_counter()
    estimate = x.T @ residual  # Paper-allowed exact initialization.
    radius = np.zeros(d)
    init_work = n * d + d
    work = init_work
    snapshots = []
    chosen_sizes = []

    for step in range(MAX_STEPS + 1):
        if step % CHECK == 0:
            gap = max(0.0, objective(residual, weights) - optimal_value)
            work += n + d
            snapshots.append((step, gap, work, time.perf_counter() - start))

        if step == MAX_STEPS:
            break

        if verify:
            actual = x.T @ residual + LAM * weights
            if not np.all(np.abs(actual - estimate) <= radius + 1e-9):
                raise AssertionError("ASCD gradient interval invalid")

        candidates, lower = active_set(estimate, radius)
        best = np.max(lower[candidates])
        tied = candidates[lower[candidates] == best]
        j = int(rng.choice(tied))
        chosen_sizes.append(len(candidates))

        gradient_j = x[:, j] @ residual + LAM * weights[j]
        delta = -gradient_j / smoothness[j]
        weights[j] += delta
        residual += delta * x[:, j]

        # Paper's zero oracle for all other coordinates. Exact coordinate
        # minimization sets the selected coordinate's new gradient to zero.
        radius += abs(delta) * norms[j] * norms
        estimate[j] = 0.0
        radius[j] = 0.0

        # Transparent workload proxy: baseline update (2n+3), plus sorting
        # d*ceil(log2 d), O(d) bounds/interval work and one draw.
        work += 2 * n + 3 + d * int(np.ceil(np.log2(d))) + 3 * d

        if verify:
            actual = x.T @ residual + LAM * weights
            if not np.all(np.abs(actual - estimate) <= radius + 1e-9):
                raise AssertionError("ASCD gradient interval invalid after update")
            if not np.allclose(residual, x @ weights - y, atol=1e-10):
                raise AssertionError("Residual invariant failed")

    return snapshots, float(np.median(chosen_sizes)), init_work


def audit():
    assert active_set(np.zeros(3), np.zeros(3))[0].size == 3
    assert np.array_equal(
        active_set(np.array([3.0, 1.0, 0.0]), np.zeros(3))[0],
        np.array([0]),
    )
    x, y = data("balanced_correlated", 0)
    run(x[:25, :8], y[:25], 0, verify=True)


def main():
    audit()
    rows = []
    medians = {}
    for case in CASES:
        for seed in SEEDS:
            x, y = data(case, seed)
            snapshots, median_size, init_work = run(x, y, seed)
            medians[(case, seed)] = (median_size, init_work)
            for step, gap, work, seconds in snapshots:
                rows.append({
                    "case": case, "seed": seed, "method": "ascd_zero_oracle",
                    "steps": step, "gap": gap, "proxy_work": work,
                    "seconds": seconds,
                })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for case in CASES:
        subset = [r for r in rows if r["case"] == case]
        final = [r for r in subset if r["steps"] == MAX_STEPS]
        hits = []
        for seed in SEEDS:
            trace = [r for r in subset if r["seed"] == seed]
            hit = next((r for r in trace if r["gap"] <= TARGET), None)
            if hit is not None:
                hits.append(hit)
        summary.append({
            "case": case,
            "method": "ascd_zero_oracle",
            "reached": len(hits),
            "median_final_gap": float(np.median([r["gap"] for r in final])),
            "median_active_set_size": float(np.median([
                medians[(case, seed)][0] for seed in SEEDS
            ])),
            "median_initialization_proxy_work": float(np.median([
                medians[(case, seed)][1] for seed in SEEDS
            ])),
            "median_proxy_work_to_target": (
                float(np.median([r["proxy_work"] for r in hits])) if hits else None
            ),
            "median_seconds_to_target": (
                float(np.median([r["seconds"] for r in hits])) if hits else None
            ),
        })

    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
