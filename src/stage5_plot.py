"""Stage 5 publication-oriented figures.

This script visualizes the complete Stage 5 confirmatory result set.

Inputs
------
results/stage5_confirmatory/
    trajectories.csv
    run_results.csv
    summary.json
    timing_summary.json
    paired_results.csv

Outputs
-------
results/stage5_confirmatory/
    heldout_gap_trajectories.png
    real_gap_trajectories.png
    active_set_generalization.png
    runtime_comparison.png

The script does not rerun optimization and does not alter numerical
results.

Run from repository root:

    python -m py_compile src/stage5_plot.py
    python src/stage5_plot.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    ROOT
    / "results"
    / "stage5_confirmatory"
)

TRAJECTORIES_PATH = (
    RESULT_ROOT
    / "trajectories.csv"
)

SUMMARY_PATH = (
    RESULT_ROOT
    / "summary.json"
)

TIMING_SUMMARY_PATH = (
    RESULT_ROOT
    / "timing_summary.json"
)

PAIRED_RESULTS_PATH = (
    RESULT_ROOT
    / "paired_results.csv"
)

HELDOUT_GAP_PATH = (
    RESULT_ROOT
    / "heldout_gap_trajectories.png"
)

REAL_GAP_PATH = (
    RESULT_ROOT
    / "real_gap_trajectories.png"
)

ACTIVE_SET_PATH = (
    RESULT_ROOT
    / "active_set_generalization.png"
)

RUNTIME_PATH = (
    RESULT_ROOT
    / "runtime_comparison.png"
)


# =============================================================================
# Frozen display configuration
# =============================================================================

METHODS = (
    "uniform",
    "lipschitz",
    "ascd_g3_zero",
    "ascd_g1_exact",
)

G3_METHOD = "ascd_g3_zero"
G1_METHOD = "ascd_g1_exact"

SYNTHETIC_CASES = (
    "heldout_balanced_moderate",
    "heldout_balanced_strong",
    "heldout_scaled_moderate",
    "heldout_scaled_strong",
)

REAL_CASES = (
    "california_housing",
    "concrete_compressive_strength",
    "airfoil_self_noise",
    "residential_building",
)

CASE_ORDER = (
    *SYNTHETIC_CASES,
    *REAL_CASES,
)

CASE_LABELS = {
    "heldout_balanced_moderate": (
        "Balanced, "
        r"$\rho=0.6$"
    ),

    "heldout_balanced_strong": (
        "Balanced, "
        r"$\rho=0.9$"
    ),

    "heldout_scaled_moderate": (
        "Scaled, "
        r"$\rho=0.6$"
    ),

    "heldout_scaled_strong": (
        "Scaled, "
        r"$\rho=0.9$"
    ),

    "california_housing": (
        "California Housing"
    ),

    "concrete_compressive_strength": (
        "Concrete Strength"
    ),

    "airfoil_self_noise": (
        "Airfoil Self-Noise"
    ),

    "residential_building": (
        "Residential Building"
    ),
}

SHORT_CASE_LABELS = {
    "heldout_balanced_moderate": "Bal. 0.6",
    "heldout_balanced_strong": "Bal. 0.9",
    "heldout_scaled_moderate": "Scaled 0.6",
    "heldout_scaled_strong": "Scaled 0.9",
    "california_housing": "California",
    "concrete_compressive_strength": "Concrete",
    "airfoil_self_noise": "Airfoil",
    "residential_building": "Residential",
}

METHOD_LABELS = {
    "uniform": "Uniform",
    "lipschitz": "Lipschitz",
    "ascd_g3_zero": r"ASCD $g^3$",
    "ascd_g1_exact": r"ASCD $g^1$",
}

METHOD_COLORS = {
    "uniform": "#4C78A8",
    "lipschitz": "#F58518",
    "ascd_g3_zero": "#54A24B",
    "ascd_g1_exact": "#B279A2",
}

METHOD_LINESTYLES = {
    "uniform": "-",
    "lipschitz": "--",
    "ascd_g3_zero": "-.",
    "ascd_g1_exact": "-",
}

PRIMARY_TARGET = 1e-6
SECONDARY_TARGET = 1e-4

LOG_FLOOR = 1e-14

FIG_DPI = 300


# =============================================================================
# Matplotlib style
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.titlesize": 11.5,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)


# =============================================================================
# General utilities
# =============================================================================

def read_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file:\n  {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file:\n  {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def assert_inputs_exist() -> None:
    required = (
        TRAJECTORIES_PATH,
        SUMMARY_PATH,
        TIMING_SUMMARY_PATH,
        PAIRED_RESULTS_PATH,
    )

    missing = [
        path
        for path in required
        if not path.is_file()
    ]

    if missing:
        formatted = "\n".join(
            f"  {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Stage 5 analysis outputs are incomplete.\n\n"
            f"Missing:\n{formatted}\n\n"
            "Run first:\n"
            "    python src/stage5_analyze.py"
        )


# =============================================================================
# Trajectory loading and aggregation
# =============================================================================

def load_trajectories() -> list[dict[str, Any]]:
    raw_rows = read_csv(
        TRAJECTORIES_PATH
    )

    rows: list[
        dict[str, Any]
    ] = []

    for raw in raw_rows:
        rows.append(
            {
                "problem_id": (
                    raw["problem_id"]
                ),

                "kind": (
                    raw["kind"]
                ),

                "case": (
                    raw["case"]
                ),

                "problem_seed": (
                    None
                    if raw[
                        "problem_seed"
                    ].strip() == ""
                    else int(
                        raw[
                            "problem_seed"
                        ]
                    )
                ),

                "algorithm_seed": int(
                    raw[
                        "algorithm_seed"
                    ]
                ),

                "method": (
                    raw["method"]
                ),

                "coordinate_epochs": float(
                    raw[
                        "coordinate_epochs"
                    ]
                ),

                "relative_gap": float(
                    raw[
                        "relative_gap"
                    ]
                ),
            }
        )

    return rows


def aggregate_trajectory(
    rows: list[dict[str, Any]],
    *,
    case: str,
    method: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    selected = [
        row
        for row in rows
        if (
            row["case"] == case
            and row["method"] == method
        )
    ]

    if not selected:
        raise RuntimeError(
            f"No trajectory rows for "
            f"{case}/{method}."
        )

    by_epoch: dict[
        float,
        list[float],
    ] = defaultdict(list)

    for row in selected:
        by_epoch[
            row[
                "coordinate_epochs"
            ]
        ].append(
            row[
                "relative_gap"
            ]
        )

    epochs = np.asarray(
        sorted(
            by_epoch
        ),
        dtype=np.float64,
    )

    medians: list[float] = []
    q25: list[float] = []
    q75: list[float] = []

    for epoch in epochs:
        values = np.asarray(
            by_epoch[
                float(
                    epoch
                )
            ],
            dtype=np.float64,
        )

        if len(values) != 12:
            raise AssertionError(
                f"{case}/{method}/epoch={epoch}: "
                f"expected 12 values, found "
                f"{len(values)}."
            )

        values = np.maximum(
            values,
            LOG_FLOOR,
        )

        medians.append(
            float(
                np.median(
                    values
                )
            )
        )

        q25.append(
            float(
                np.percentile(
                    values,
                    25,
                )
            )
        )

        q75.append(
            float(
                np.percentile(
                    values,
                    75,
                )
            )
        )

    return (
        epochs,
        np.asarray(
            medians
        ),
        np.asarray(
            q25
        ),
        np.asarray(
            q75
        ),
    )


# =============================================================================
# Gap trajectory figures
# =============================================================================

def plot_gap_grid(
    *,
    trajectories: list[dict[str, Any]],
    cases: tuple[str, ...],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            10.4,
            6.9,
        ),
        sharex=True,
        sharey=True,
    )

    flat_axes = (
        axes.ravel()
    )

    for axis, case in zip(
        flat_axes,
        cases,
    ):
        for method in METHODS:
            (
                epochs,
                median,
                q25,
                q75,
            ) = aggregate_trajectory(
                trajectories,
                case=case,
                method=method,
            )

            color = (
                METHOD_COLORS[
                    method
                ]
            )

            axis.plot(
                epochs,
                median,
                label=(
                    METHOD_LABELS[
                        method
                    ]
                ),
                color=color,
                linestyle=(
                    METHOD_LINESTYLES[
                        method
                    ]
                ),
                linewidth=1.8,
            )

            axis.fill_between(
                epochs,
                q25,
                q75,
                color=color,
                alpha=0.10,
                linewidth=0.0,
            )

        axis.axhline(
            SECONDARY_TARGET,
            color="#777777",
            linewidth=0.9,
            linestyle=":",
        )

        axis.axhline(
            PRIMARY_TARGET,
            color="#222222",
            linewidth=1.0,
            linestyle=":",
        )

        axis.set_yscale(
            "log"
        )

        axis.set_ylim(
            LOG_FLOOR,
            2.0,
        )

        axis.set_xlim(
            0,
            200,
        )

        axis.set_title(
            CASE_LABELS[
                case
            ]
        )

    for axis in axes[
        1,
        :
    ]:
        axis.set_xlabel(
            "Coordinate epochs"
        )

    for axis in axes[
        :,
        0
    ]:
        axis.set_ylabel(
            "Normalized objective gap"
        )

    handles, labels = (
        flat_axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.965,
        ),
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        title,
        y=0.995,
    )

    fig.text(
        0.5,
        0.015,
        (
            "Solid/dashed curves show medians; "
            "shading shows the interquartile range. "
            r"Horizontal dotted lines mark $10^{-4}$ and $10^{-6}$."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.tight_layout(
        rect=(
            0.0,
            0.045,
            1.0,
            0.93,
        )
    )

    fig.savefig(
        output_path,
        dpi=FIG_DPI,
    )

    plt.close(
        fig
    )


def plot_gap_trajectories(
    trajectories: list[dict[str, Any]],
) -> None:
    plot_gap_grid(
        trajectories=trajectories,
        cases=SYNTHETIC_CASES,
        output_path=(
            HELDOUT_GAP_PATH
        ),
        title=(
            "Stage 5 held-out synthetic "
            "optimization trajectories"
        ),
    )

    plot_gap_grid(
        trajectories=trajectories,
        cases=REAL_CASES,
        output_path=(
            REAL_GAP_PATH
        ),
        title=(
            "Stage 5 real-data "
            "optimization trajectories"
        ),
    )


# =============================================================================
# Active-set mechanism figure
# =============================================================================

def require_metric(
    summary: dict[str, Any],
    *,
    case: str,
    method: str,
    metric: str,
) -> float:
    value = (
        summary[
            "by_case"
        ][
            case
        ][
            "methods"
        ][
            method
        ][
            metric
        ]
    )

    if value is None:
        raise RuntimeError(
            f"{case}/{method}: "
            f"missing {metric}."
        )

    return float(
        value
    )


def plot_active_set_generalization(
    summary: dict[str, Any],
) -> None:
    x = np.arange(
        len(
            CASE_ORDER
        ),
        dtype=np.float64,
    )

    width = 0.36

    g3_active = np.asarray(
        [
            require_metric(
                summary,
                case=case,
                method=G3_METHOD,
                metric=(
                    "median_active_set_fraction"
                ),
            )
            for case in CASE_ORDER
        ],
        dtype=np.float64,
    )

    g1_active = np.asarray(
        [
            require_metric(
                summary,
                case=case,
                method=G1_METHOD,
                metric=(
                    "median_active_set_fraction"
                ),
            )
            for case in CASE_ORDER
        ],
        dtype=np.float64,
    )

    g3_zero = np.asarray(
        [
            require_metric(
                summary,
                case=case,
                method=G3_METHOD,
                metric=(
                    "median_zero_lower_fraction"
                ),
            )
            for case in CASE_ORDER
        ],
        dtype=np.float64,
    )

    g1_zero = np.asarray(
        [
            require_metric(
                summary,
                case=case,
                method=G1_METHOD,
                metric=(
                    "median_zero_lower_fraction"
                ),
            )
            for case in CASE_ORDER
        ],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            11.0,
            6.8,
        ),
        sharex=True,
    )

    axes[0].bar(
        x - width / 2,
        g3_active,
        width=width,
        label=r"ASCD $g^3$",
        color=(
            METHOD_COLORS[
                G3_METHOD
            ]
        ),
        alpha=0.90,
    )

    axes[0].bar(
        x + width / 2,
        g1_active,
        width=width,
        label=r"ASCD $g^1$",
        color=(
            METHOD_COLORS[
                G1_METHOD
            ]
        ),
        alpha=0.90,
    )

    axes[0].set_ylabel(
        "Median active-set fraction"
    )

    axes[0].set_ylim(
        0.0,
        1.05,
    )

    axes[0].legend(
        frameon=False,
        ncol=2,
    )

    axes[0].set_title(
        "Active-set size"
    )

    axes[1].bar(
        x - width / 2,
        g3_zero,
        width=width,
        label=r"ASCD $g^3$",
        color=(
            METHOD_COLORS[
                G3_METHOD
            ]
        ),
        alpha=0.90,
    )

    axes[1].bar(
        x + width / 2,
        g1_zero,
        width=width,
        label=r"ASCD $g^1$",
        color=(
            METHOD_COLORS[
                G1_METHOD
            ]
        ),
        alpha=0.90,
    )

    axes[1].set_ylabel(
        "Median zero-lower-bound fraction"
    )

    axes[1].set_ylim(
        0.0,
        1.05,
    )

    axes[1].set_title(
        "Certified lower-bound collapse"
    )

    axes[1].set_xticks(
        x
    )

    axes[1].set_xticklabels(
        [
            SHORT_CASE_LABELS[
                case
            ]
            for case in CASE_ORDER
        ],
        rotation=25,
        ha="right",
    )

    fig.suptitle(
        "Stage 5 ASCD mechanism generalization",
        y=0.995,
    )

    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.96,
        )
    )

    fig.savefig(
        ACTIVE_SET_PATH,
        dpi=FIG_DPI,
    )

    plt.close(
        fig
    )


# =============================================================================
# Paired runtime comparison
# =============================================================================

def extract_runtime_ratio(
    summary: dict[str, Any],
    case: str,
) -> tuple[
    float,
    float,
    float,
    int,
] | None:
    """Return paired g1/g3 runtime summary when shared successes exist.

    A problem family with no shared primary success has no meaningful
    time-to-target ratio and is therefore omitted from the runtime-ratio
    figure rather than treated as zero or infinity.
    """

    payload = (
        summary[
            "g1_vs_g3_paired"
        ][
            case
        ][
            "harmonized_runtime"
        ][
            "paired_ratio_g1_over_g3"
        ]
    )

    if payload is None:
        return None

    return (
        float(
            payload[
                "median"
            ]
        ),
        float(
            payload[
                "ci95_lower"
            ]
        ),
        float(
            payload[
                "ci95_upper"
            ]
        ),
        int(
            payload["n"]
        ),
    )


def plot_runtime_comparison(
    summary: dict[str, Any],
) -> None:
    plotted_cases: list[str] = []

    medians: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    shared_counts: list[int] = []

    omitted_cases: list[str] = []

    for case in CASE_ORDER:
        result = extract_runtime_ratio(
            summary,
            case,
        )

        if result is None:
            omitted_cases.append(
                case
            )
            continue

        (
            median,
            lower,
            upper,
            n,
        ) = result

        plotted_cases.append(
            case
        )

        medians.append(
            median
        )

        lower_errors.append(
            median - lower
        )

        upper_errors.append(
            upper - median
        )

        shared_counts.append(
            n
        )

    if not plotted_cases:
        raise RuntimeError(
            "No problem family has a defined paired "
            "g1/g3 runtime ratio."
        )

    y = np.arange(
        len(
            plotted_cases
        )
    )

    medians_array = np.asarray(
        medians,
        dtype=np.float64,
    )

    errors = np.asarray(
        [
            lower_errors,
            upper_errors,
        ],
        dtype=np.float64,
    )

    fig, axis = plt.subplots(
        figsize=(
            9.2,
            5.4,
        )
    )

    axis.errorbar(
        medians_array,
        y,
        xerr=errors,
        fmt="o",
        color="#4C78A8",
        ecolor="#4C78A8",
        elinewidth=1.5,
        capsize=3.5,
        markersize=6,
    )

    axis.axvline(
        1.0,
        color="#222222",
        linestyle="--",
        linewidth=1.1,
    )

    axis.set_yticks(
        y
    )

    axis.set_yticklabels(
        [
            SHORT_CASE_LABELS[
                case
            ]
            for case
            in plotted_cases
        ]
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        r"Paired median runtime ratio "
        r"$g^1/g^3$"
    )

    axis.set_title(
        "Stage 5 harmonized runtime comparison"
    )

    all_lower = (
        medians_array
        - errors[0]
    )

    all_upper = (
        medians_array
        + errors[1]
    )

    minimum = float(
        np.min(
            all_lower
        )
    )

    maximum = float(
        np.max(
            all_upper
        )
    )

    span = max(
        maximum - minimum,
        0.1,
    )

    axis.set_xlim(
        max(
            0.0,
            minimum
            - 0.12 * span,
        ),
        maximum
        + 0.22 * span,
    )

    for index, (
        median,
        n,
    ) in enumerate(
        zip(
            medians,
            shared_counts,
        )
    ):
        axis.annotate(
            f"n={n}",
            xy=(
                median,
                index,
            ),
            xytext=(
                7,
                0,
            ),
            textcoords=(
                "offset points"
            ),
            va="center",
            fontsize=8,
            color="#555555",
        )

    omission_note = ""

    if omitted_cases:
        omitted_labels = ", ".join(
            SHORT_CASE_LABELS[
                case
            ]
            for case
            in omitted_cases
        )

        omission_note = (
            " Families without shared primary successes "
            f"({omitted_labels}) are omitted."
        )

    fig.text(
        0.5,
        0.015,
        (
            "Points are paired medians over shared primary successes; "
            "error bars are 95% bootstrap intervals. "
            "The dashed vertical line marks equal measured runtime."
            + omission_note
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.tight_layout(
        rect=(
            0.0,
            0.055,
            1.0,
            1.0,
        )
    )

    fig.savefig(
        RUNTIME_PATH,
        dpi=FIG_DPI,
    )

    plt.close(
        fig
    )


# =============================================================================
# Output audit
# =============================================================================

def audit_output(
    path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    size = (
        path.stat().st_size
    )

    if size <= 0:
        raise RuntimeError(
            f"Generated empty figure: {path}"
        )

    print(
        f"  {path.name}: "
        f"{size:,} bytes"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    assert_inputs_exist()

    summary = read_json(
        SUMMARY_PATH
    )

    # Ensure expected case/method structure exists.
    observed_cases = set(
        summary[
            "by_case"
        ]
    )

    if observed_cases != set(
        CASE_ORDER
    ):
        raise RuntimeError(
            "summary.json case set does not match "
            "the Stage 5 plotting protocol."
        )

    for case in CASE_ORDER:
        observed_methods = set(
            summary[
                "by_case"
            ][
                case
            ][
                "methods"
            ]
        )

        if observed_methods != set(
            METHODS
        ):
            raise RuntimeError(
                f"{case}: method set mismatch."
            )

    trajectories = (
        load_trajectories()
    )

    print(
        "Generating Stage 5 figures..."
    )

    plot_gap_trajectories(
        trajectories
    )

    plot_active_set_generalization(
        summary
    )

    plot_runtime_comparison(
        summary
    )

    print()
    print(
        "Stage 5 figure generation: PASSED"
    )

    print(
        "\nGenerated figures:"
    )

    for path in (
        HELDOUT_GAP_PATH,
        REAL_GAP_PATH,
        ACTIVE_SET_PATH,
        RUNTIME_PATH,
    ):
        audit_output(
            path
        )

    print()
    print(
        "No optimization methods were rerun."
    )

    print(
        "No Stage 5 numerical result was modified."
    )


if __name__ == "__main__":
    main()
