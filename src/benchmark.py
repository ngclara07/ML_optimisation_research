"""Cost-aware exploratory benchmark for randomized coordinate descent."""

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEEDS = range(12)
N = 400
D = 48
LAM = 0.03
MAX_STEPS = 2400
CHECK = 120
REFRESH = 60
TARGET = 1e-4

METHODS = (
    "uniform",
    "lipschitz",
    "scheduled_gain",
    "triggered_gain",
)

CASES = (
    "balanced_independent",
    "scaled_independent",
    "balanced_correlated",
    "scaled_correlated",
    "diabetes",
)


def data(case, seed):
    """Return one reproducible regression design and its labels."""
    if case == "diabetes":
        x, y = load_diabetes(return_X_y=True)
        x = x.copy()
        y = y.copy()

        # Fixed standardization; this benchmark makes no prediction claim.
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
        x /= np.sqrt(len(x))
        y = (y - y.mean()) / y.std()
        return x, y

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, D))

    if "correlated" in case:
        latent = rng.normal(size=(N, 8))
        x = 0.72 * latent[:, np.arange(D) % 8] + 0.28 * x

    x /= np.sqrt(N)

    if "scaled" in case:
        x *= np.geomspace(0.2, 5.0, D)

    y = x @ rng.normal(size=D) + 0.1 * rng.normal(size=N)
    return x, y


def objective(residual, weights):
    """Ridge objective given residual = X @ weights - y."""
    return 0.5 * (
        residual @ residual
        + LAM * (weights @ weights)
    )


def audit_coordinate_update():
    """Check one coordinate update against the stated ridge objective."""
    x = np.array([[1.0, 2.0], [3.0, -1.0], [-2.0, 0.5]])
    y = np.array([0.5, -1.0, 2.0])
    weights = np.array([0.2, -0.3])
    residual = x @ weights - y
    coordinate = 1

    matrix = x.T @ x + LAM * np.eye(x.shape[1])
    smoothness = np.diag(matrix)
    expected_l = x[:, coordinate] @ x[:, coordinate] + LAM
    assert np.isclose(smoothness[coordinate], expected_l)

    gradient_j = x[:, coordinate] @ residual + LAM * weights[coordinate]
    epsilon = 1e-6
    direction = np.eye(x.shape[1])[coordinate]
    plus = weights + epsilon * direction
    minus = weights - epsilon * direction
    numerical_gradient = (
        objective(x @ plus - y, plus)
        - objective(x @ minus - y, minus)
    ) / (2 * epsilon)
    assert np.isclose(gradient_j, numerical_gradient, rtol=1e-7, atol=1e-8)

    before = objective(residual, weights)
    delta = -gradient_j / smoothness[coordinate]
    weights[coordinate] += delta
    residual += delta * x[:, coordinate]
    assert np.allclose(residual, x @ weights - y, rtol=1e-12, atol=1e-12)
    after = objective(residual, weights)
    expected_drop = gradient_j**2 / (2 * smoothness[coordinate])
    assert np.isclose(before - after, expected_drop, rtol=1e-10, atol=1e-12)


def run(x, y, method, seed):
    """Run one sampler; return sampled steps, gap, work, time and refreshes."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    n, d = x.shape
    rng = np.random.default_rng(seed + 9812)

    matrix = x.T @ x + LAM * np.eye(d)
    optimum = np.linalg.solve(matrix, x.T @ y)
    optimal_value = objective(x @ optimum - y, optimum)

    # For every j, L_j = ||X_j||^2 + LAM.
    smoothness = np.diag(matrix)
    weights = np.zeros(d)
    residual = -y.copy()
    probabilities = np.full(d, 1 / d)

    if method == "lipschitz":
        probabilities = smoothness / smoothness.sum()

    snapshots = []
    refreshes = 0
    work = 0
    previous_value = None
    previous_decrease = None
    last_refresh = -REFRESH
    start = time.perf_counter()

    for step in range(MAX_STEPS + 1):
        if step % CHECK == 0:
            current_value = objective(residual, weights)
            gap = max(0.0, current_value - optimal_value)

            # Approximate work for an objective checkpoint.
            work += n + d

            snapshots.append(
                (
                    step,
                    gap,
                    work,
                    time.perf_counter() - start,
                    refreshes,
                )
            )

        if step == MAX_STEPS:
            break

        do_refresh = (
            method == "scheduled_gain"
            and step % REFRESH == 0
        )

        if method == "triggered_gain" and step % CHECK == 0:
            # Refresh if the latest observed decrease is less than half
            # the preceding decrease. No optimum is used by this trigger.
            decrease = (
                previous_value - current_value
                if previous_value is not None
                else None
            )

            stalled = (
                decrease is not None
                and previous_decrease is not None
                and decrease < 0.5 * max(previous_decrease, 0.0)
            )

            do_refresh = step == 0 or (
                stalled and step - last_refresh >= REFRESH
            )

            if decrease is not None:
                previous_decrease = decrease

            previous_value = current_value

        if do_refresh:
            gradient = x.T @ residual + LAM * weights
            gain = gradient**2 / smoothness

            probabilities = (
                0.05 / d + 0.95 * gain / gain.sum()
                if gain.sum() > 0
                else np.full(d, 1 / d)
            )

            refreshes += 1
            last_refresh = step
            work += n * d + d

        coordinate = int(rng.choice(d, p=probabilities))

        # Probability-vector sampling costs O(d) here.
        work += d

        gradient_j = (
            x[:, coordinate] @ residual
            + LAM * weights[coordinate]
        )
        delta = -gradient_j / smoothness[coordinate]

        weights[coordinate] += delta
        # Preserve residual = X @ weights - y after changing coordinate j.
        residual += delta * x[:, coordinate]
        work += 2 * n + 3

    return snapshots


def main():
    audit_coordinate_update()
    rows = []

    for case in CASES:
        for seed in SEEDS:
            x, y = data(case, seed)

            for method in METHODS:
                for step, gap, work, elapsed, refreshes in run(
                    x, y, method, seed
                ):
                    rows.append(
                        {
                            "case": case,
                            "seed": seed,
                            "method": method,
                            "steps": step,
                            "gap": gap,
                            "proxy_work": work,
                            "seconds": elapsed,
                            "refreshes": refreshes,
                        }
                    )

    RESULTS.mkdir(parents=True, exist_ok=True)

    with (RESULTS / "trajectories.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = []

    for case in CASES:
        for method in METHODS:
            subset = [
                row for row in rows
                if row["case"] == case
                and row["method"] == method
            ]
            final = [
                row for row in subset
                if row["steps"] == MAX_STEPS
            ]

            hits = []
            for seed in SEEDS:
                trace = [
                    row for row in subset
                    if row["seed"] == seed
                ]
                hit = next(
                    (
                        row for row in trace
                        if row["gap"] <= TARGET
                    ),
                    None,
                )
                if hit is not None:
                    hits.append(hit)

            summary.append(
                {
                    "case": case,
                    "method": method,
                    "reached": len(hits),
                    "median_final_gap": float(
                        np.median(
                            [row["gap"] for row in final]
                        )
                    ),
                    "median_refreshes": float(
                        np.median(
                            [row["refreshes"] for row in final]
                        )
                    ),
                    "median_proxy_work_to_target": (
                        float(
                            np.median(
                                [row["proxy_work"] for row in hits]
                            )
                        )
                        if hits else None
                    ),
                    "median_seconds_to_target": (
                        float(
                            np.median(
                                [row["seconds"] for row in hits]
                            )
                        )
                        if hits else None
                    ),
                }
            )

    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
