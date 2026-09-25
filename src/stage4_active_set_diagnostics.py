"""Stage 4 diagnostics for the frozen zero-oracle ASCD implementation.

This script replays the Stage 3 zero-oracle ASCD algorithm without changing
its coordinate selections or updates. It records per-update diagnostics that
help explain why the ASCD active set remains effectively full on the
synthetic ridge-regression cases.

Outputs:
    results/stage4_oracle_analysis/zero_oracle_diagnostics.csv
    results/stage4_oracle_analysis/zero_oracle_summary.json

The script also verifies that its checkpoint objective gaps and proxy-work
values reproduce the frozen Stage 3 ASCD trajectory.
"""

import csv
import json
from pathlib import Path

import numpy as np

from benchmark import (
    CASES,
    CHECK,
    LAM,
    MAX_STEPS,
    SEEDS,
    TARGET,
    data,
    objective,
)

from published_ascd import active_set


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "results" / "stage4_oracle_analysis"

DIAGNOSTICS_CSV = OUTPUT / "zero_oracle_diagnostics.csv"
SUMMARY_JSON = OUTPUT / "zero_oracle_summary.json"

STAGE3_TRAJECTORY = (
    ROOT / "results" / "published_ascd" / "trajectories.csv"
)

METHOD = "ascd_zero_oracle"


def load_stage3_reference():
    """Load the frozen Stage 3 ASCD checkpoint trajectory."""
    if not STAGE3_TRAJECTORY.is_file():
        raise FileNotFoundError(
            f"Missing Stage 3 trajectory: {STAGE3_TRAJECTORY}"
        )

    reference = {}

    with STAGE3_TRAJECTORY.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if row["method"] != METHOD:
                continue

            key = (
                row["case"],
                int(row["seed"]),
                int(row["steps"]),
            )

            reference[key] = {
                "gap": float(row["gap"]),
                "proxy_work": int(float(row["proxy_work"])),
            }

    return reference


def validate_checkpoint(
    reference,
    case,
    seed,
    step,
    gap,
    work,
):
    """Check that Stage 4 replay matches the frozen Stage 3 trajectory."""
    key = (case, seed, step)

    if key not in reference:
        raise AssertionError(
            f"Missing Stage 3 reference checkpoint: {key}"
        )

    expected = reference[key]

    if work != expected["proxy_work"]:
        raise AssertionError(
            f"Proxy-work mismatch at {key}: "
            f"{work} != {expected['proxy_work']}"
        )

    if not np.isclose(
        gap,
        expected["gap"],
        rtol=1e-11,
        atol=1e-12,
    ):
        raise AssertionError(
            f"Gap mismatch at {key}: "
            f"{gap:.16e} != {expected['gap']:.16e}"
        )


def diagnostic_run(
    x,
    y,
    case,
    seed,
    reference,
):
    """Replay one zero-oracle ASCD run and collect diagnostics."""
    n, d = x.shape

    # Identical RNG convention to Stage 3.
    rng = np.random.default_rng(seed + 9812)

    matrix = x.T @ x + LAM * np.eye(d)
    optimum = np.linalg.solve(matrix, x.T @ y)
    optimal_value = objective(
        x @ optimum - y,
        optimum,
    )

    smoothness = np.diag(matrix)
    norms = np.linalg.norm(x, axis=0)

    weights = np.zeros(d)
    residual = -y.copy()

    # Identical exact initialization to Stage 3.
    estimate = x.T @ residual
    radius = np.zeros(d)

    init_work = n * d + d
    work = init_work

    diagnostic_rows = []
    checkpoint_rows = []

    for step in range(MAX_STEPS + 1):

        checkpoint_gap = None

        if step % CHECK == 0:
            gap = max(
                0.0,
                objective(residual, weights) - optimal_value,
            )

            work += n + d

            validate_checkpoint(
                reference,
                case,
                seed,
                step,
                gap,
                work,
            )

            checkpoint_gap = gap

            checkpoint_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "step": step,
                    "gap": gap,
                    "proxy_work": work,
                }
            )

        if step == MAX_STEPS:
            break

        # Evaluation-only full gradient for diagnostics.
        # This is NOT added to the ASCD proxy work.
        actual = (
            x.T @ residual
            + LAM * weights
        )

        # Verify the certified gradient intervals.
        error = np.abs(actual - estimate)
        slack = radius - error

        if np.min(slack) < -1e-9:
            raise AssertionError(
                "Gradient interval violation "
                f"for case={case}, seed={seed}, step={step}"
            )

        candidates, lower = active_set(
            estimate,
            radius,
        )

        upper = np.abs(estimate) + radius

        active_size = len(candidates)
        excluded_count = d - active_size

        best = np.max(lower[candidates])
        tied = candidates[
            lower[candidates] == best
        ]

        # Identical RNG call to Stage 3.
        j = int(rng.choice(tied))

        active_mean_lower_sq = float(
            np.mean(lower[candidates] ** 2)
        )

        abs_actual = np.abs(actual)

        zero_lower_count = int(
            np.count_nonzero(lower == 0.0)
        )

        gradient_j = (
            x[:, j] @ residual
            + LAM * weights[j]
        )

        delta = -gradient_j / smoothness[j]

        diagnostic_rows.append(
            {
                "case": case,
                "seed": seed,
                "step": step,
                "dimension": d,
                "selected_coordinate": j,

                "active_set_size": active_size,
                "active_set_fraction": active_size / d,

                "excluded_count": excluded_count,
                "excluded_fraction": excluded_count / d,

                "median_radius": float(
                    np.median(radius)
                ),
                "max_radius": float(
                    np.max(radius)
                ),

                "median_abs_true_gradient": float(
                    np.median(abs_actual)
                ),
                "max_abs_true_gradient": float(
                    np.max(abs_actual)
                ),

                "median_upper_bound": float(
                    np.median(upper)
                ),
                "max_upper_bound": float(
                    np.max(upper)
                ),

                "median_lower_bound": float(
                    np.median(lower)
                ),
                "max_lower_bound": float(
                    np.max(lower)
                ),

                "zero_lower_count": zero_lower_count,
                "fraction_zero_lower_bound": (
                    zero_lower_count / d
                ),

                "active_mean_lower_sq": (
                    active_mean_lower_sq
                ),

                "selected_true_gradient": float(
                    gradient_j
                ),
                "selected_lower_bound": float(
                    lower[j]
                ),
                "selected_upper_bound": float(
                    upper[j]
                ),

                "selected_delta": float(delta),

                "minimum_interval_slack": float(
                    np.min(slack)
                ),

                "checkpoint_gap": (
                    checkpoint_gap
                    if checkpoint_gap is not None
                    else ""
                ),
            }
        )

        # ------------------------------------------------------------
        # The update below is intentionally identical to Stage 3.
        # ------------------------------------------------------------

        weights[j] += delta
        residual += delta * x[:, j]

        radius += (
            abs(delta)
            * norms[j]
            * norms
        )

        estimate[j] = 0.0
        radius[j] = 0.0

        work += (
            2 * n
            + 3
            + d * int(np.ceil(np.log2(d)))
            + 3 * d
        )

        # Residual invariant check.
        if not np.allclose(
            residual,
            x @ weights - y,
            atol=1e-10,
        ):
            raise AssertionError(
                "Residual invariant failed "
                f"for case={case}, seed={seed}, step={step}"
            )

    return diagnostic_rows, checkpoint_rows


def first_success(checkpoints):
    """Return first saved checkpoint reaching the target, if any."""
    for row in checkpoints:
        if row["gap"] <= TARGET:
            return row

    return None


def summarize_case(
    case,
    rows,
    checkpoints,
):
    """Aggregate Stage 4 diagnostics for one benchmark case."""
    case_rows = [
        row
        for row in rows
        if row["case"] == case
    ]

    case_checkpoints = [
        row
        for row in checkpoints
        if row["case"] == case
    ]

    active_sizes = np.array(
        [
            row["active_set_size"]
            for row in case_rows
        ],
        dtype=float,
    )

    active_fractions = np.array(
        [
            row["active_set_fraction"]
            for row in case_rows
        ],
        dtype=float,
    )

    zero_lower_fractions = np.array(
        [
            row["fraction_zero_lower_bound"]
            for row in case_rows
        ],
        dtype=float,
    )

    median_radii = np.array(
        [
            row["median_radius"]
            for row in case_rows
        ],
        dtype=float,
    )

    median_gradients = np.array(
        [
            row["median_abs_true_gradient"]
            for row in case_rows
        ],
        dtype=float,
    )

    seeds = sorted(
        {
            row["seed"]
            for row in case_rows
        }
    )

    successes = []

    per_seed_median_active = []

    for seed in seeds:
        seed_rows = [
            row
            for row in case_rows
            if row["seed"] == seed
        ]

        per_seed_median_active.append(
            float(
                np.median(
                    [
                        row["active_set_size"]
                        for row in seed_rows
                    ]
                )
            )
        )

        seed_checkpoints = [
            row
            for row in case_checkpoints
            if row["seed"] == seed
        ]

        hit = first_success(seed_checkpoints)

        if hit is not None:
            successes.append(
                {
                    "seed": seed,
                    "step": hit["step"],
                    "proxy_work": hit["proxy_work"],
                }
            )

    d = int(case_rows[0]["dimension"])

    return {
        "case": case,
        "num_seeds": len(seeds),
        "updates_recorded": len(case_rows),

        "successes": len(successes),
        "successful_seeds": [
            item["seed"]
            for item in successes
        ],

        "median_active_set_size_all_updates": float(
            np.median(active_sizes)
        ),

        "median_of_per_seed_median_active_set_size": float(
            np.median(per_seed_median_active)
        ),

        "mean_active_set_fraction": float(
            np.mean(active_fractions)
        ),

        "fraction_updates_full_active_set": float(
            np.mean(active_sizes == d)
        ),

        "fraction_updates_with_any_pruning": float(
            np.mean(active_sizes < d)
        ),

        "fraction_updates_half_or_less": float(
            np.mean(active_sizes <= d / 2)
        ),

        "median_fraction_zero_lower_bound": float(
            np.median(zero_lower_fractions)
        ),

        "mean_fraction_zero_lower_bound": float(
            np.mean(zero_lower_fractions)
        ),

        "median_of_median_radius": float(
            np.median(median_radii)
        ),

        "median_of_median_abs_true_gradient": float(
            np.median(median_gradients)
        ),
    }


def main():
    reference = load_stage3_reference()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []
    all_checkpoints = []

    for case in CASES:
        print(f"Running diagnostics: {case}")

        for seed in SEEDS:
            x, y = data(case, seed)

            rows, checkpoints = diagnostic_run(
                x,
                y,
                case,
                seed,
                reference,
            )

            all_rows.extend(rows)
            all_checkpoints.extend(checkpoints)

    if not all_rows:
        raise RuntimeError(
            "No diagnostic rows were generated"
        )

    with DIAGNOSTICS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=all_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(all_rows)

    summaries = [
        summarize_case(
            case,
            all_rows,
            all_checkpoints,
        )
        for case in CASES
    ]

    payload = {
        "method": METHOD,
        "diagnostic_only": True,

        "trajectory_validation": {
            "reference": str(
                STAGE3_TRAJECTORY.relative_to(ROOT)
            ),
            "matched_all_checkpoints": True,
        },

        "target": TARGET,
        "max_steps": MAX_STEPS,
        "checkpoint_spacing": CHECK,

        "cases": summaries,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Stage 3 trajectory validation: PASSED"
    )

    print()

    for item in summaries:
        print(
            f"{item['case']}: "
            f"success={item['successes']}/{item['num_seeds']}, "
            f"median_active="
            f"{item['median_active_set_size_all_updates']:.1f}, "
            f"full_fraction="
            f"{item['fraction_updates_full_active_set']:.4f}, "
            f"pruned_fraction="
            f"{item['fraction_updates_with_any_pruning']:.4f}, "
            f"median_zero_lower_fraction="
            f"{item['median_fraction_zero_lower_bound']:.4f}"
        )

    print()
    print(
        f"Saved diagnostics: {DIAGNOSTICS_CSV}"
    )

    print(
        f"Saved summary: {SUMMARY_JSON}"
    )


if __name__ == "__main__":
    main()
