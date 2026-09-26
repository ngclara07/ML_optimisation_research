"""Plot completed Stage 4 Phase A zero-oracle diagnostics."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "results"
    / "stage4_oracle_analysis"
    / "zero_oracle_diagnostics.csv"
)
OUTPUT = ROOT / "results" / "stage4_oracle_analysis"

ACTIVE_FIGURE = OUTPUT / "active_set_trajectories.png"
RADIUS_FIGURE = OUTPUT / "radius_tightness.png"

CASES = (
    "balanced_independent",
    "scaled_independent",
    "balanced_correlated",
    "scaled_correlated",
    "diabetes",
)

LABELS = {
    "balanced_independent": "Balanced independent",
    "scaled_independent": "Scaled independent",
    "balanced_correlated": "Balanced correlated",
    "scaled_correlated": "Scaled correlated",
    "diabetes": "Diabetes",
}


def load_rows():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    rows = []

    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "case": row["case"],
                    "seed": int(row["seed"]),
                    "step": int(row["step"]),
                    "active_set_fraction": float(
                        row["active_set_fraction"]
                    ),
                    "fraction_zero_lower_bound": float(
                        row["fraction_zero_lower_bound"]
                    ),
                    "median_radius": float(row["median_radius"]),
                    "median_abs_true_gradient": float(
                        row["median_abs_true_gradient"]
                    ),
                }
            )

    return rows


def aggregate(rows, field):
    """Median across problem seeds at each update."""
    grouped = defaultdict(lambda: defaultdict(list))

    for row in rows:
        grouped[row["case"]][row["step"]].append(
            row[field]
        )

    output = {}

    for case in CASES:
        steps = sorted(grouped[case])
        medians = [
            float(np.median(grouped[case][step]))
            for step in steps
        ]

        output[case] = (
            np.asarray(steps),
            np.asarray(medians),
        )

    return output


def plot_active_set(rows):
    active = aggregate(rows, "active_set_fraction")
    zero_lower = aggregate(
        rows,
        "fraction_zero_lower_bound",
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.4),
        sharex=True,
    )

    for case in CASES:
        steps, values = active[case]
        axes[0].plot(
            steps,
            values,
            linewidth=1.4,
            label=LABELS[case],
        )

        steps, values = zero_lower[case]
        axes[1].plot(
            steps,
            values,
            linewidth=1.4,
            label=LABELS[case],
        )

    axes[0].set_ylabel("Median active-set fraction")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)

    axes[1].set_ylabel(
        "Median zero-lower-bound fraction"
    )
    axes[1].set_xlabel("Coordinate update")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)

    axes[0].legend(
        fontsize=8,
        ncol=2,
        loc="lower right",
    )

    fig.suptitle(
        "Zero-oracle ASCD: active-set and lower-bound behaviour"
    )

    fig.tight_layout()
    fig.savefig(
        ACTIVE_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_radius_tightness(rows):
    radius = aggregate(rows, "median_radius")
    gradient = aggregate(
        rows,
        "median_abs_true_gradient",
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.4),
        sharex=True,
    )

    axes = axes.ravel()

    floor = 1e-16

    for index, case in enumerate(CASES):
        ax = axes[index]

        steps, radius_values = radius[case]
        _, gradient_values = gradient[case]

        ax.plot(
            steps,
            np.maximum(radius_values, floor),
            label="median radius",
            linewidth=1.4,
        )

        ax.plot(
            steps,
            np.maximum(gradient_values, floor),
            label="median |true gradient|",
            linewidth=1.4,
        )

        ax.set_yscale("log")
        ax.set_title(LABELS[case], fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Update")

        if index % 3 == 0:
            ax.set_ylabel("Magnitude")

    axes[-1].axis("off")

    axes[0].legend(fontsize=8)

    fig.suptitle(
        "Zero-oracle ASCD: certified radius versus true-gradient scale"
    )

    fig.tight_layout()
    fig.savefig(
        RADIUS_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    rows = load_rows()

    expected = 5 * 12 * 2400

    if len(rows) != expected:
        raise AssertionError(
            f"Expected {expected} diagnostic rows, "
            f"found {len(rows)}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    plot_active_set(rows)
    plot_radius_tightness(rows)

    print(f"Rows validated: {len(rows)}")
    print(f"Saved: {ACTIVE_FIGURE}")
    print(f"Saved: {RADIUS_FIGURE}")


if __name__ == "__main__":
    main()
