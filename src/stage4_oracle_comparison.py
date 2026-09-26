"""Stage 4 Phase B: ASCD oracle-quality comparison.

Configurations:
    g3_zero
    g2_eps_1
    g2_eps_0p5
    g2_eps_0p25
    g2_eps_0p125
    g1_exact

The g3 configuration is required to reproduce the frozen Stage 3
ASCD checkpoint trajectory exactly (within numerical tolerance).

Simulated g2 isolates oracle-quality effects. Its acquisition cost is
intentionally NOT interpreted as the cost of a real embedding oracle.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
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

STAGE3_REFERENCE = (
    ROOT
    / "results"
    / "published_ascd"
    / "trajectories.csv"
)

RUN_CSV = OUTPUT / "oracle_comparison.csv"
TRAJECTORY_CSV = OUTPUT / "oracle_trajectories.csv"
ACTIVE_CSV = OUTPUT / "oracle_active_set_by_step.csv"
SUMMARY_JSON = OUTPUT / "oracle_summary.json"

ACTIVE_FIGURE = OUTPUT / "oracle_active_set_comparison.png"
GAP_FIGURE = OUTPUT / "oracle_gap_trajectories.png"

EPSILONS = (1.0, 0.5, 0.25, 0.125)

CONFIGS = (
    ("g3_zero", "g3", None),
    ("g2_eps_1", "g2", 1.0),
    ("g2_eps_0p5", "g2", 0.5),
    ("g2_eps_0p25", "g2", 0.25),
    ("g2_eps_0p125", "g2", 0.125),
    ("g1_exact", "g1", 0.0),
)

DISPLAY_NAMES = {
    "g3_zero": "g3 zero",
    "g2_eps_1": "g2 eps=1",
    "g2_eps_0p5": "g2 eps=0.5",
    "g2_eps_0p25": "g2 eps=0.25",
    "g2_eps_0p125": "g2 eps=0.125",
    "g1_exact": "g1 exact",
}

CASE_LABELS = {
    "balanced_independent": "Balanced independent",
    "scaled_independent": "Scaled independent",
    "balanced_correlated": "Balanced correlated",
    "scaled_correlated": "Scaled correlated",
    "diabetes": "Diabetes",
}


def load_stage3_reference():
    reference = {}

    with STAGE3_REFERENCE.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "ascd_zero_oracle":
                continue

            key = (
                row["case"],
                int(row["seed"]),
                int(row["steps"]),
            )

            reference[key] = {
                "gap": float(row["gap"]),
                "proxy_work": int(
                    float(row["proxy_work"])
                ),
            }

    return reference


def validate_g3_checkpoint(
    reference,
    case,
    seed,
    step,
    gap,
    work,
):
    key = (case, seed, step)

    expected = reference.get(key)

    if expected is None:
        raise AssertionError(
            f"Missing frozen Stage 3 checkpoint {key}"
        )

    if work != expected["proxy_work"]:
        raise AssertionError(
            f"g3 work mismatch {key}: "
            f"{work} != {expected['proxy_work']}"
        )

    if not np.isclose(
        gap,
        expected["gap"],
        rtol=1e-11,
        atol=1e-12,
    ):
        raise AssertionError(
            f"g3 gap mismatch {key}: "
            f"{gap:.16e} != "
            f"{expected['gap']:.16e}"
        )


def symmetric_noise_matrix(
    d,
    case_index,
    seed,
    epsilon_index,
):
    """Fixed symmetric U[-1,1] perturbations for one run."""
    rng_seed = (
        4_700_001
        + 100_003 * case_index
        + 1_009 * seed
        + 10_007 * epsilon_index
    )

    rng = np.random.default_rng(rng_seed)

    noise = np.zeros((d, d))

    rows, cols = np.triu_indices(d, k=1)

    values = rng.uniform(
        low=-1.0,
        high=1.0,
        size=len(rows),
    )

    noise[rows, cols] = values
    noise[cols, rows] = values

    return noise


def build_oracle(
    kind,
    epsilon,
    gram,
    norms,
    case_index,
    seed,
):
    """Return fixed oracle values and certified errors."""
    d = gram.shape[0]

    norm_outer = np.outer(norms, norms)

    if kind == "g3":
        values = np.zeros_like(gram)
        errors = norm_outer.copy()

        np.fill_diagonal(errors, 0.0)

        return (
            values,
            errors,
            0,
            "costed_zero_oracle",
        )

    if kind == "g1":
        values = gram.copy()
        errors = np.zeros_like(gram)

        # Explicit modelled cost for obtaining X.T @ X.
        n_placeholder = None

        return (
            values,
            errors,
            n_placeholder,
            "gram_setup_modelled",
        )

    if kind != "g2":
        raise ValueError(kind)

    epsilon_index = EPSILONS.index(epsilon)

    noise = symmetric_noise_matrix(
        d,
        case_index,
        seed,
        epsilon_index,
    )

    raw = (
        gram
        + epsilon
        * norm_outer
        * noise
    )

    # Paper's g2 clips the scalar-product estimate to
    # [-||Xi||||Xj||, +||Xi||||Xj||].
    values = np.minimum(
        norm_outer,
        np.maximum(-norm_outer, raw),
    )

    errors = epsilon * norm_outer

    # The selected coordinate is reset exactly, so diagonal
    # oracle values/errors are not needed.
    np.fill_diagonal(values, np.diag(gram))
    np.fill_diagonal(errors, 0.0)

    # This simulation uses exact Gram values to manufacture a
    # certified approximation. It is a quality experiment only.
    return (
        values,
        errors,
        None,
        "quality_only_not_costed",
    )


def verify_interval(
    x,
    residual,
    weights,
    estimate,
    radius,
    case,
    seed,
    config,
    step,
):
    actual = (
        x.T @ residual
        + LAM * weights
    )

    violation = (
        np.abs(actual - estimate)
        - radius
    )

    if np.max(violation) > 1e-8:
        raise AssertionError(
            "Certified interval violation: "
            f"case={case}, seed={seed}, "
            f"config={config}, step={step}, "
            f"max={np.max(violation):.3e}"
        )


def run_one(
    x,
    y,
    case,
    case_index,
    seed,
    config_name,
    oracle_kind,
    epsilon,
    reference,
):
    n, d = x.shape

    # Selection RNG remains identical to Stage 3.
    rng = np.random.default_rng(seed + 9812)

    gram = x.T @ x
    matrix = gram + LAM * np.eye(d)

    optimum = np.linalg.solve(
        matrix,
        x.T @ y,
    )

    optimal_value = objective(
        x @ optimum - y,
        optimum,
    )

    smoothness = np.diag(matrix)
    norms = np.linalg.norm(x, axis=0)

    (
        oracle_values,
        oracle_errors,
        setup_marker,
        cost_status,
    ) = build_oracle(
        oracle_kind,
        epsilon,
        gram,
        norms,
        case_index,
        seed,
    )

    if oracle_kind == "g1":
        # Transparent dense-Gram acquisition proxy.
        oracle_setup_proxy = n * d * d
    elif oracle_kind == "g3":
        oracle_setup_proxy = 0
    else:
        oracle_setup_proxy = None

    weights = np.zeros(d)
    residual = -y.copy()

    # Same full-gradient initialization as Stage 3.
    estimate = x.T @ residual
    radius = np.zeros(d)

    init_work = n * d + d
    core_work = init_work

    trajectories = []

    active_sizes = np.empty(
        MAX_STEPS,
        dtype=np.int16,
    )

    zero_lower_fractions = np.empty(
        MAX_STEPS,
        dtype=float,
    )

    for step in range(MAX_STEPS + 1):

        if step % CHECK == 0:
            gap = max(
                0.0,
                objective(residual, weights)
                - optimal_value,
            )

            core_work += n + d

            if oracle_kind == "g3":
                validate_g3_checkpoint(
                    reference,
                    case,
                    seed,
                    step,
                    gap,
                    core_work,
                )

            verify_interval(
                x,
                residual,
                weights,
                estimate,
                radius,
                case,
                seed,
                config_name,
                step,
            )

            costed_work = (
                None
                if oracle_setup_proxy is None
                else core_work
                + oracle_setup_proxy
            )

            trajectories.append(
                {
                    "case": case,
                    "seed": seed,
                    "config": config_name,
                    "oracle_kind": oracle_kind,
                    "epsilon": (
                        ""
                        if epsilon is None
                        else epsilon
                    ),
                    "steps": step,
                    "gap": gap,
                    "core_proxy_work": core_work,
                    "costed_proxy_work": (
                        ""
                        if costed_work is None
                        else costed_work
                    ),
                }
            )

        if step == MAX_STEPS:
            break

        candidates, lower = active_set(
            estimate,
            radius,
        )

        active_sizes[step] = len(candidates)

        zero_lower_fractions[step] = (
            np.count_nonzero(lower == 0.0)
            / d
        )

        best = np.max(lower[candidates])

        tied = candidates[
            lower[candidates] == best
        ]

        # Exactly one selection RNG draw per update.
        j = int(rng.choice(tied))

        gradient_j = (
            x[:, j] @ residual
            + LAM * weights[j]
        )

        delta = (
            -gradient_j
            / smoothness[j]
        )

        weights[j] += delta
        residual += delta * x[:, j]

        # Algorithm 1 passive-coordinate update.
        estimate += (
            delta
            * oracle_values[j, :]
        )

        radius += (
            abs(delta)
            * oracle_errors[j, :]
        )

        # Exact coordinate minimization sets selected gradient to 0.
        estimate[j] = 0.0
        radius[j] = 0.0

        # Same common/core proxy as frozen Stage 3.
        core_work += (
            2 * n
            + 3
            + d * int(np.ceil(np.log2(d)))
            + 3 * d
        )

    first_hit = next(
        (
            row
            for row in trajectories
            if row["gap"] <= TARGET
        ),
        None,
    )

    success = first_hit is not None

    first_hit_core = (
        None
        if first_hit is None
        else int(first_hit["core_proxy_work"])
    )

    first_hit_costed = None

    if (
        first_hit is not None
        and oracle_setup_proxy is not None
    ):
        first_hit_costed = (
            first_hit_core
            + oracle_setup_proxy
        )

    run_record = {
        "case": case,
        "seed": seed,
        "config": config_name,
        "oracle_kind": oracle_kind,
        "epsilon": (
            ""
            if epsilon is None
            else epsilon
        ),
        "success": success,
        "first_hit_step": (
            ""
            if first_hit is None
            else first_hit["steps"]
        ),
        "first_hit_core_proxy_work": (
            ""
            if first_hit_core is None
            else first_hit_core
        ),
        "oracle_setup_proxy_work": (
            ""
            if oracle_setup_proxy is None
            else oracle_setup_proxy
        ),
        "first_hit_costed_proxy_work": (
            ""
            if first_hit_costed is None
            else first_hit_costed
        ),
        "cost_status": cost_status,
        "final_gap": trajectories[-1]["gap"],
        "dimension": d,
        "median_active_set": float(
            np.median(active_sizes)
        ),
        "mean_active_set_fraction": float(
            np.mean(active_sizes / d)
        ),
        "fraction_full_active_set": float(
            np.mean(active_sizes == d)
        ),
        "fraction_with_pruning": float(
            np.mean(active_sizes < d)
        ),
        "fraction_half_or_less": float(
            np.mean(active_sizes <= d / 2)
        ),
        "median_zero_lower_fraction": float(
            np.median(zero_lower_fractions)
        ),
    }

    return {
        "run_record": run_record,
        "trajectories": trajectories,
        "active_sizes": active_sizes,
        "zero_lower_fractions": zero_lower_fractions,
        "dimension": d,
    }


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            f"No rows generated for {path}"
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(
    run_records,
    step_results,
):
    output = []

    for case in CASES:
        for (
            config_name,
            oracle_kind,
            epsilon,
        ) in CONFIGS:

            subset = [
                row
                for row in run_records
                if row["case"] == case
                and row["config"] == config_name
            ]

            successes = [
                row
                for row in subset
                if row["success"]
            ]

            arrays = step_results[
                (case, config_name)
            ]

            active = np.concatenate(
                [
                    item["active_sizes"]
                    for item in arrays
                ]
            )

            zero_lower = np.concatenate(
                [
                    item[
                        "zero_lower_fractions"
                    ]
                    for item in arrays
                ]
            )

            core_work = [
                int(
                    row[
                        "first_hit_core_proxy_work"
                    ]
                )
                for row in successes
            ]

            costed_work = [
                int(
                    row[
                        "first_hit_costed_proxy_work"
                    ]
                )
                for row in successes
                if row[
                    "first_hit_costed_proxy_work"
                ] != ""
            ]

            output.append(
                {
                    "case": case,
                    "config": config_name,
                    "oracle_kind": oracle_kind,
                    "epsilon": epsilon,
                    "successes": len(successes),
                    "successful_seeds": [
                        row["seed"]
                        for row in successes
                    ],
                    "median_final_gap": float(
                        np.median(
                            [
                                row["final_gap"]
                                for row in subset
                            ]
                        )
                    ),
                    "median_core_proxy_to_target": (
                        None
                        if not core_work
                        else float(
                            np.median(core_work)
                        )
                    ),
                    "median_costed_proxy_to_target": (
                        None
                        if len(costed_work)
                        != len(successes)
                        or not costed_work
                        else float(
                            np.median(costed_work)
                        )
                    ),
                    "active_set_q10": float(
                        np.quantile(active, 0.10)
                    ),
                    "active_set_q25": float(
                        np.quantile(active, 0.25)
                    ),
                    "active_set_median": float(
                        np.median(active)
                    ),
                    "active_set_q75": float(
                        np.quantile(active, 0.75)
                    ),
                    "active_set_q90": float(
                        np.quantile(active, 0.90)
                    ),
                    "fraction_full_active_set": float(
                        np.mean(
                            active
                            == arrays[0]["dimension"]
                        )
                    ),
                    "fraction_with_pruning": float(
                        np.mean(
                            active
                            < arrays[0]["dimension"]
                        )
                    ),
                    "fraction_half_or_less": float(
                        np.mean(
                            active
                            <= arrays[0]["dimension"] / 2
                        )
                    ),
                    "median_zero_lower_fraction": float(
                        np.median(zero_lower)
                    ),
                    "mean_zero_lower_fraction": float(
                        np.mean(zero_lower)
                    ),
                }
            )

    return output


def build_step_summary(step_results):
    rows = []

    for case in CASES:
        for (
            config_name,
            _,
            _,
        ) in CONFIGS:

            runs = step_results[
                (case, config_name)
            ]

            d = runs[0]["dimension"]

            active_matrix = np.vstack(
                [
                    item["active_sizes"]
                    for item in runs
                ]
            )

            zero_matrix = np.vstack(
                [
                    item["zero_lower_fractions"]
                    for item in runs
                ]
            )

            for step in range(MAX_STEPS):
                values = active_matrix[:, step]

                rows.append(
                    {
                        "case": case,
                        "config": config_name,
                        "step": step,
                        "dimension": d,
                        "median_active_set": float(
                            np.median(values)
                        ),
                        "q25_active_set": float(
                            np.quantile(values, 0.25)
                        ),
                        "q75_active_set": float(
                            np.quantile(values, 0.75)
                        ),
                        "median_active_fraction": float(
                            np.median(values / d)
                        ),
                        "median_zero_lower_fraction": float(
                            np.median(
                                zero_matrix[:, step]
                            )
                        ),
                    }
                )

    return rows


def build_paired_vs_g3(run_records):
    paired = {}

    for case in CASES:
        g3 = {
            row["seed"]: row
            for row in run_records
            if row["case"] == case
            and row["config"] == "g3_zero"
        }

        case_result = {}

        for (
            config_name,
            _,
            _,
        ) in CONFIGS:

            if config_name == "g3_zero":
                continue

            other = {
                row["seed"]: row
                for row in run_records
                if row["case"] == case
                and row["config"] == config_name
            }

            g3_success = {
                seed
                for seed, row in g3.items()
                if row["success"]
            }

            other_success = {
                seed
                for seed, row in other.items()
                if row["success"]
            }

            shared = sorted(
                g3_success & other_success
            )

            core_differences = []
            core_ratios = []

            for seed in shared:
                base = int(
                    g3[seed][
                        "first_hit_core_proxy_work"
                    ]
                )

                candidate = int(
                    other[seed][
                        "first_hit_core_proxy_work"
                    ]
                )

                core_differences.append(
                    candidate - base
                )

                core_ratios.append(
                    candidate / base
                )

            costed_differences = []
            costed_ratios = []

            for seed in shared:
                left = g3[seed][
                    "first_hit_costed_proxy_work"
                ]

                right = other[seed][
                    "first_hit_costed_proxy_work"
                ]

                if left == "" or right == "":
                    continue

                left = int(left)
                right = int(right)

                costed_differences.append(
                    right - left
                )

                costed_ratios.append(
                    right / left
                )

            case_result[config_name] = {
                "g3_successes": sorted(
                    g3_success
                ),
                "config_successes": sorted(
                    other_success
                ),
                "shared_successes": shared,
                "g3_only_successes": sorted(
                    g3_success - other_success
                ),
                "config_only_successes": sorted(
                    other_success - g3_success
                ),
                "median_paired_core_difference": (
                    None
                    if not core_differences
                    else float(
                        np.median(
                            core_differences
                        )
                    )
                ),
                "median_paired_core_ratio": (
                    None
                    if not core_ratios
                    else float(
                        np.median(
                            core_ratios
                        )
                    )
                ),
                "median_paired_costed_difference": (
                    None
                    if len(costed_differences)
                    != len(shared)
                    or not costed_differences
                    else float(
                        np.median(
                            costed_differences
                        )
                    )
                ),
                "median_paired_costed_ratio": (
                    None
                    if len(costed_ratios)
                    != len(shared)
                    or not costed_ratios
                    else float(
                        np.median(
                            costed_ratios
                        )
                    )
                ),
            }

        paired[case] = case_result

    return paired


def plot_gap_trajectories(trajectory_rows):
    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    for row in trajectory_rows:
        key = (
            row["case"],
            row["config"],
        )

        grouped[key][
            int(row["steps"])
        ].append(float(row["gap"]))

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.0, 6.5),
        sharex=True,
    )

    axes = axes.ravel()

    for index, case in enumerate(CASES):
        ax = axes[index]

        for config_name, _, _ in CONFIGS:
            points = grouped[
                (case, config_name)
            ]

            steps = sorted(points)

            medians = [
                max(
                    1e-14,
                    float(
                        np.median(
                            points[step]
                        )
                    ),
                )
                for step in steps
            ]

            ax.plot(
                steps,
                medians,
                linewidth=1.25,
                label=DISPLAY_NAMES[
                    config_name
                ],
            )

        ax.axhline(
            TARGET,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )

        ax.set_yscale("log")
        ax.set_title(
            CASE_LABELS[case],
            fontsize=10,
        )
        ax.grid(alpha=0.25)
        ax.set_xlabel("Coordinate update")

        if index % 3 == 0:
            ax.set_ylabel("Median objective gap")

    axes[-1].axis("off")

    axes[0].legend(
        fontsize=7,
        ncol=2,
    )

    fig.suptitle(
        "Stage 4 oracle-quality comparison"
    )

    fig.tight_layout()

    fig.savefig(
        GAP_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_active_sets(step_rows):
    grouped = defaultdict(list)

    for row in step_rows:
        grouped[
            (
                row["case"],
                row["config"],
            )
        ].append(row)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.0, 6.5),
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    for index, case in enumerate(CASES):
        ax = axes[index]

        for config_name, _, _ in CONFIGS:
            rows = grouped[
                (case, config_name)
            ]

            rows.sort(
                key=lambda row: row["step"]
            )

            ax.plot(
                [
                    row["step"]
                    for row in rows
                ],
                [
                    row[
                        "median_active_fraction"
                    ]
                    for row in rows
                ],
                linewidth=1.15,
                label=DISPLAY_NAMES[
                    config_name
                ],
            )

        ax.set_ylim(-0.02, 1.02)
        ax.set_title(
            CASE_LABELS[case],
            fontsize=10,
        )
        ax.grid(alpha=0.25)
        ax.set_xlabel("Coordinate update")

        if index % 3 == 0:
            ax.set_ylabel(
                "Median active-set fraction"
            )

    axes[-1].axis("off")

    axes[0].legend(
        fontsize=7,
        ncol=2,
    )

    fig.suptitle(
        "Active-set fraction by oracle quality"
    )

    fig.tight_layout()

    fig.savefig(
        ACTIVE_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = load_stage3_reference()

    run_records = []
    trajectory_rows = []

    step_results = defaultdict(list)

    for case_index, case in enumerate(CASES):

        print(f"\nCASE: {case}")

        for (
            config_name,
            oracle_kind,
            epsilon,
        ) in CONFIGS:

            print(
                f"  running {config_name}"
            )

            for seed in SEEDS:
                x, y = data(case, seed)

                result = run_one(
                    x,
                    y,
                    case,
                    case_index,
                    seed,
                    config_name,
                    oracle_kind,
                    epsilon,
                    reference,
                )

                run_records.append(
                    result["run_record"]
                )

                trajectory_rows.extend(
                    result["trajectories"]
                )

                step_results[
                    (case, config_name)
                ].append(
                    {
                        "active_sizes": (
                            result[
                                "active_sizes"
                            ]
                        ),
                        "zero_lower_fractions": (
                            result[
                                "zero_lower_fractions"
                            ]
                        ),
                        "dimension": (
                            result["dimension"]
                        ),
                    }
                )

    step_rows = build_step_summary(
        step_results
    )

    aggregate = aggregate_results(
        run_records,
        step_results,
    )

    paired = build_paired_vs_g3(
        run_records
    )

    write_csv(
        RUN_CSV,
        run_records,
    )

    write_csv(
        TRAJECTORY_CSV,
        trajectory_rows,
    )

    write_csv(
        ACTIVE_CSV,
        step_rows,
    )

    payload = {
        "phase": "Stage 4 Phase B",
        "benchmark_frozen": True,
        "g3_replay_validation": "PASSED",
        "g2_simulation": {
            "quality_only": True,
            "costed_acquisition": False,
            "epsilon_grid": list(EPSILONS),
            "note": (
                "Fixed symmetric per-run scalar-product "
                "approximations satisfying the certified "
                "epsilon bound. This is not a costed "
                "embedding implementation."
            ),
        },
        "g1_cost_model": {
            "gram_setup_proxy": "n*d*d",
            "note": (
                "Dense Gram acquisition is charged "
                "separately from the common Stage 3 "
                "core proxy."
            ),
        },
        "aggregate": aggregate,
        "paired_vs_g3": paired,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_gap_trajectories(
        trajectory_rows
    )

    plot_active_sets(
        step_rows
    )

    print()
    print(
        "G3 FROZEN-TRAJECTORY VALIDATION: PASSED"
    )

    print()

    for case in CASES:
        print(case)

        for item in aggregate:
            if item["case"] != case:
                continue

            print(
                "  "
                f"{item['config']}: "
                f"success={item['successes']}/12, "
                f"median_active="
                f"{item['active_set_median']:.1f}, "
                f"pruned_fraction="
                f"{item['fraction_with_pruning']:.4f}, "
                f"zero_lower="
                f"{item['median_zero_lower_fraction']:.4f}, "
                f"core_work="
                f"{item['median_core_proxy_to_target']}"
            )

    print()
    print(f"Saved: {RUN_CSV}")
    print(f"Saved: {TRAJECTORY_CSV}")
    print(f"Saved: {ACTIVE_CSV}")
    print(f"Saved: {SUMMARY_JSON}")
    print(f"Saved: {ACTIVE_FIGURE}")
    print(f"Saved: {GAP_FIGURE}")


if __name__ == "__main__":
    main()
