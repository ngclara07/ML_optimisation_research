"""Replay baseline runs and diagnose gain-sampling behaviour without edits."""

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark import CHECK, LAM, MAX_STEPS, REFRESH, SEEDS, data, objective


ROOT = Path(__file__).resolve().parents[1]
CASES = ("balanced_independent", "balanced_correlated")
METHODS = ("scheduled_gain", "triggered_gain")
FIELDS = (
    "case", "seed", "method", "steps", "gap", "last_refresh_step",
    "age_since_refresh", "max_probability", "effective_coordinates",
    "top5_overlap", "mass_on_current_top5", "mass_on_old_top5",
)


def replay(case, seed, method):
    """Match the baseline draw and refresh sequence; observe at checkpoints."""
    x, y = data(case, seed)
    n, d = x.shape
    rng = np.random.default_rng(seed + 9812)
    matrix = x.T @ x + LAM * np.eye(d)
    smoothness = np.diag(matrix)
    optimum = np.linalg.solve(matrix, x.T @ y)
    optimal_value = objective(x @ optimum - y, optimum)
    weights = np.zeros(d)
    residual = -y.copy()
    probabilities = np.full(d, 1 / d)
    previous_value = None
    previous_decrease = None
    last_refresh = -REFRESH
    old_gain = None
    rows = []
    checkpoints = []

    for step in range(MAX_STEPS + 1):
        if step % CHECK == 0:
            current_value = objective(residual, weights)
            gap = max(0.0, current_value - optimal_value)
            checkpoints.append((step, gap))

            # Diagnostic gradient is read-only. It is NOT used for a draw or
            # refresh and is not included in baseline time or work.
            if step > 0:
                current_gradient = x.T @ residual + LAM * weights
                current_gain = current_gradient**2 / smoothness
                top = min(5, d)
                old_top = set(np.argsort(old_gain)[-top:])
                current_top = set(np.argsort(current_gain)[-top:])
                rows.append({
                    "case": case,
                    "seed": seed,
                    "method": method,
                    "steps": step,
                    "gap": gap,
                    "last_refresh_step": last_refresh,
                    "age_since_refresh": step - last_refresh,
                    "max_probability": float(probabilities.max()),
                    "effective_coordinates": float(
                        1.0 / np.sum(probabilities**2)
                    ),
                    "top5_overlap": len(old_top & current_top) / top,
                    "mass_on_current_top5": float(
                        probabilities[list(current_top)].sum()
                    ),
                    "mass_on_old_top5": float(
                        probabilities[list(old_top)].sum()
                    ),
                })

        if step == MAX_STEPS:
            break

        do_refresh = method == "scheduled_gain" and step % REFRESH == 0
        if method == "triggered_gain" and step % CHECK == 0:
            decrease = (
                previous_value - current_value
                if previous_value is not None else None
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
            old_gain = gradient**2 / smoothness
            total_gain = old_gain.sum()
            probabilities = (
                0.05 / d + 0.95 * old_gain / total_gain
                if total_gain > 0 else np.full(d, 1 / d)
            )
            last_refresh = step

        coordinate = int(rng.choice(d, p=probabilities))
        gradient_j = (
            x[:, coordinate] @ residual + LAM * weights[coordinate]
        )
        delta = -gradient_j / smoothness[coordinate]
        weights[coordinate] += delta
        residual += delta * x[:, coordinate]

    return rows, checkpoints


def baseline_check(reference, case, seed, method, checkpoints):
    """Require the replay to match every archived objective checkpoint."""
    expected = reference[(case, seed, method)]
    if len(expected) != len(checkpoints):
        raise ValueError("Checkpoint count differs from baseline")
    for (step, gap), (old_step, old_gap) in zip(checkpoints, expected):
        if step != old_step or not np.isclose(
            gap, old_gap, rtol=1e-9, atol=1e-10
        ):
            raise ValueError(
                f"Replay differs at {case}, seed {seed}, {method}, step {step}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-csv", type=Path,
        default=ROOT / "results_baseline_v1" / "trajectories.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "diagnostics" / "sampling.csv",
    )
    args = parser.parse_args()
    if not args.baseline_csv.is_file():
        parser.error(f"Baseline CSV not found: {args.baseline_csv}")

    reference = {}
    with args.baseline_csv.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            key = (item["case"], int(item["seed"]), item["method"])
            reference.setdefault(key, []).append(
                (int(item["steps"]), float(item["gap"]))
            )

    rows = []
    for case in CASES:
        for seed in SEEDS:
            for method in METHODS:
                record, checkpoints = replay(case, seed, method)
                baseline_check(reference, case, seed, method, checkpoints)
                rows.extend(record)

    # No output is written until all 48 replays match the baseline.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Verified {len(CASES) * len(SEEDS) * len(METHODS)} traces")
    print(f"Saved {len(rows)} checkpoint diagnostics to {args.output}")
    for case in CASES:
        for method in METHODS:
            group = [r for r in rows if r["case"] == case and r["method"] == method]
            print({
                "case": case,
                "method": method,
                "median_effective_coordinates": float(np.median(
                    [r["effective_coordinates"] for r in group]
                )),
                "median_top5_overlap": float(np.median(
                    [r["top5_overlap"] for r in group]
                )),
                "median_mass_on_current_top5": float(np.median(
                    [r["mass_on_current_top5"] for r in group]
                )),
            })


if __name__ == "__main__":
    main()
