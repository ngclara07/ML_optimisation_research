"""Plot median objective gaps from the benchmark's saved trajectories."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CASES = (
    "balanced_independent",
    "balanced_correlated",
)

METHODS = (
    "uniform",
    "lipschitz",
    "scheduled_gain",
    "triggered_gain",
)


def main():
    input_path = RESULTS / "trajectories.csv"

    if not input_path.is_file():
        raise SystemExit(
            f"Missing {input_path}. "
            "Run 'python src/benchmark.py' first."
        )

    with input_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 3.6),
        constrained_layout=True,
    )

    for axis, case in zip(axes, CASES):
        for method in METHODS:
            group = [
                row for row in rows
                if row["case"] == case
                and row["method"] == method
            ]
            steps = sorted(
                {int(row["steps"]) for row in group}
            )
            median_gaps = [
                np.median(
                    [
                        float(row["gap"])
                        for row in group
                        if int(row["steps"]) == step
                    ]
                )
                for step in steps
            ]

            axis.semilogy(
                steps,
                np.maximum(median_gaps, 1e-12),
                label=method.replace("_", " "),
            )

        axis.axhline(
            1e-4,
            color="gray",
            linestyle=":",
            label="target",
        )
        axis.set(
            title=case.replace("_", " ").title(),
            xlabel="Coordinate updates",
            ylabel="Median objective gap",
        )
        axis.grid(alpha=0.2)

    axes[1].legend(fontsize=8)

    output_path = RESULTS / "median_trajectories.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
