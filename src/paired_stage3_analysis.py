"""Paired Stage 3 comparison of uniform baseline and ASCD.

Compares the same problem seeds on the scaled-correlated case.
A hit is the first saved checkpoint with objective gap <= 1e-4.

Outputs:
  results/published_ascd/paired_scaled_correlated.csv
  results/published_ascd/paired_scaled_correlated_summary.txt
"""

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BASELINE = ROOT / "results" / "trajectories.csv"
ASCD = ROOT / "results" / "published_ascd" / "trajectories.csv"

OUTPUT_CSV = (
    ROOT
    / "results"
    / "published_ascd"
    / "paired_scaled_correlated.csv"
)

OUTPUT_TXT = (
    ROOT
    / "results"
    / "published_ascd"
    / "paired_scaled_correlated_summary.txt"
)

CASE = "scaled_correlated"
TARGET = 1e-4


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def traces_by_seed(rows, method):
    traces = {}

    for row in rows:
        if row["case"] != CASE or row["method"] != method:
            continue

        seed = int(row["seed"])
        traces.setdefault(seed, []).append(
            {
                "steps": int(row["steps"]),
                "gap": float(row["gap"]),
                "proxy_work": int(float(row["proxy_work"])),
            }
        )

    for seed in traces:
        traces[seed].sort(key=lambda r: r["steps"])

    return traces


def first_hit(trace):
    return next(
        (row for row in trace if row["gap"] <= TARGET),
        None,
    )


def final_row(trace):
    return max(trace, key=lambda r: r["steps"])


def main():
    if not BASELINE.is_file():
        raise SystemExit(f"Missing baseline trajectory: {BASELINE}")

    if not ASCD.is_file():
        raise SystemExit(f"Missing ASCD trajectory: {ASCD}")

    baseline_rows = read_rows(BASELINE)
    ascd_rows = read_rows(ASCD)

    uniform = traces_by_seed(baseline_rows, "uniform")
    ascd = traces_by_seed(ascd_rows, "ascd_zero_oracle")

    seeds = sorted(set(uniform) & set(ascd))

    if seeds != list(range(12)):
        raise AssertionError(
            f"Expected paired seeds 0..11, found {seeds}"
        )

    paired_rows = []

    uniform_successes = []
    ascd_successes = []
    shared_successes = []

    for seed in seeds:
        u_hit = first_hit(uniform[seed])
        a_hit = first_hit(ascd[seed])

        if u_hit is not None:
            uniform_successes.append(seed)

        if a_hit is not None:
            ascd_successes.append(seed)

        if u_hit is not None and a_hit is not None:
            shared_successes.append(seed)

        u_final = final_row(uniform[seed])
        a_final = final_row(ascd[seed])

        paired_rows.append(
            {
                "seed": seed,
                "uniform_success": u_hit is not None,
                "ascd_success": a_hit is not None,
                "uniform_hit_steps": (
                    u_hit["steps"] if u_hit is not None else ""
                ),
                "ascd_hit_steps": (
                    a_hit["steps"] if a_hit is not None else ""
                ),
                "uniform_hit_work": (
                    u_hit["proxy_work"] if u_hit is not None else ""
                ),
                "ascd_hit_work": (
                    a_hit["proxy_work"] if a_hit is not None else ""
                ),
                "paired_work_difference": (
                    a_hit["proxy_work"] - u_hit["proxy_work"]
                    if u_hit is not None and a_hit is not None
                    else ""
                ),
                "ascd_to_uniform_work_ratio": (
                    a_hit["proxy_work"] / u_hit["proxy_work"]
                    if u_hit is not None and a_hit is not None
                    else ""
                ),
                "uniform_final_gap": u_final["gap"],
                "ascd_final_gap": a_final["gap"],
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=paired_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(paired_rows)

    shared = [
        row
        for row in paired_rows
        if row["seed"] in shared_successes
    ]

    uniform_shared_work = [
        int(row["uniform_hit_work"]) for row in shared
    ]
    ascd_shared_work = [
        int(row["ascd_hit_work"]) for row in shared
    ]
    differences = [
        int(row["paired_work_difference"]) for row in shared
    ]
    ratios = [
        float(row["ascd_to_uniform_work_ratio"]) for row in shared
    ]

    lines = []

    lines.append(
        f"Case: {CASE}; target: {TARGET:g}; "
        f"paired problem seeds: {len(seeds)}"
    )
    lines.append(
        f"Uniform successes: {len(uniform_successes)}/12 "
        f"{uniform_successes}"
    )
    lines.append(
        f"ASCD successes: {len(ascd_successes)}/12 "
        f"{ascd_successes}"
    )
    lines.append(
        f"Shared successes: {len(shared_successes)}/12 "
        f"{shared_successes}"
    )

    lines.append("")
    lines.append("Shared-success first-hit comparison:")

    for row in shared:
        lines.append(
            "seed {seed}: "
            "uniform step={uniform_hit_steps}, "
            "work={uniform_hit_work}; "
            "ASCD step={ascd_hit_steps}, "
            "work={ascd_hit_work}; "
            "difference={paired_work_difference}; "
            "ratio={ascd_to_uniform_work_ratio:.4f}".format(
                **row
            )
        )

    if shared:
        lines.append("")
        lines.append(
            "Median uniform work on shared successes: "
            f"{statistics.median(uniform_shared_work):,.0f}"
        )
        lines.append(
            "Median ASCD work on shared successes: "
            f"{statistics.median(ascd_shared_work):,.0f}"
        )
        lines.append(
            "Median paired ASCD-minus-uniform work: "
            f"{statistics.median(differences):,.0f}"
        )
        lines.append(
            "Median paired ASCD/uniform work ratio: "
            f"{statistics.median(ratios):.4f}"
        )
        lines.append(
            "ASCD has greater proxy work on every shared success: "
            f"{all(diff > 0 for diff in differences)}"
        )

    text = "\n".join(lines) + "\n"

    OUTPUT_TXT.write_text(text, encoding="utf-8")

    print(text)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
