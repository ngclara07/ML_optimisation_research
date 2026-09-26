"""Stage 6 final numerical/report audit.

This script verifies major numerical claims in the final research report
against the canonical artifacts generated during Stages 1--5.

It DOES NOT rerun optimization.

Primary inputs
--------------
Baseline:
    results_baseline_v1/trajectories.csv

Stage 3:
    results/published_ascd/summary.json
    results/published_ascd/trajectories.csv

Stage 4:
    results/stage4_oracle_analysis/oracle_summary.json

Stage 5:
    results/stage5_confirmatory/dataset_manifest.json
    results/stage5_confirmatory/summary.json
    results/stage5_confirmatory/timing_summary.json
    results/stage5_confirmatory/run_results.csv
    results/stage5_confirmatory/trajectories.csv

Report:
    report/main.tex

Output
------
    results/stage6_finalization/report_audit.json

Exit code
---------
0 : all mandatory checks passed
1 : one or more mandatory checks failed

Run
---
From repository root:

    python -m py_compile src/stage6_report_audit.py
    python src/stage6_report_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# =============================================================================
# Repository paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "stage6_finalization"
)

OUTPUT_PATH = (
    OUTPUT_ROOT
    / "report_audit.json"
)

REPORT_TEX = (
    ROOT
    / "report"
    / "main.tex"
)

BASELINE_TRAJECTORIES = (
    ROOT
    / "results_baseline_v1"
    / "trajectories.csv"
)

STAGE3_SUMMARY = (
    ROOT
    / "results"
    / "published_ascd"
    / "summary.json"
)

STAGE3_TRAJECTORIES = (
    ROOT
    / "results"
    / "published_ascd"
    / "trajectories.csv"
)

STAGE4_SUMMARY = (
    ROOT
    / "results"
    / "stage4_oracle_analysis"
    / "oracle_summary.json"
)

STAGE5_MANIFEST = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "dataset_manifest.json"
)

STAGE5_SUMMARY = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "summary.json"
)

STAGE5_TIMING = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "timing_summary.json"
)

STAGE5_RUN_RESULTS = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "run_results.csv"
)

STAGE5_TRAJECTORIES = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "trajectories.csv"
)


# =============================================================================
# Frozen research constants
# =============================================================================

BASELINE_TARGET = 1e-4

BASELINE_CASES = (
    "balanced_independent",
    "scaled_independent",
    "balanced_correlated",
    "scaled_correlated",
    "diabetes",
)

BASELINE_METHODS = (
    "uniform",
    "lipschitz",
    "scheduled_gain",
    "triggered_gain",
)

BASELINE_EXPECTED_SUCCESS = {
    "balanced_independent": {
        "uniform": 12,
        "lipschitz": 12,
        "scheduled_gain": 12,
        "triggered_gain": 12,
    },
    "scaled_independent": {
        "uniform": 12,
        "lipschitz": 0,
        "scheduled_gain": 12,
        "triggered_gain": 12,
    },
    "balanced_correlated": {
        "uniform": 12,
        "lipschitz": 12,
        "scheduled_gain": 0,
        "triggered_gain": 0,
    },
    "scaled_correlated": {
        "uniform": 6,
        "lipschitz": 0,
        "scheduled_gain": 0,
        "triggered_gain": 0,
    },
    "diabetes": {
        "uniform": 12,
        "lipschitz": 12,
        "scheduled_gain": 9,
        "triggered_gain": 5,
    },
}

STAGE3_EXPECTED = {
    "balanced_independent": {
        "success": 12,
        "work": 762936.0,
        "active": 48.0,
    },
    "scaled_independent": {
        "success": 12,
        "work": 762936.0,
        "active": 48.0,
    },
    "balanced_correlated": {
        "success": 12,
        "work": 2249416.0,
        "active": 48.0,
    },
    "scaled_correlated": {
        "success": 9,
        "work": 2844008.0,
        "active": 48.0,
    },
    "diabetes": {
        "success": 12,
        "work": 984864.0,
        "active": 10.0,
    },
}

STAGE4_CASES = BASELINE_CASES

STAGE4_CONFIGS = (
    "g3_zero",
    "g2_eps_1",
    "g2_eps_0p5",
    "g2_eps_0p25",
    "g2_eps_0p125",
    "g1_exact",
)

STAGE4_EXPECTED_SUCCESS = {
    "balanced_independent": {
        "g3_zero": 12,
        "g2_eps_1": 12,
        "g2_eps_0p5": 0,
        "g2_eps_0p25": 0,
        "g2_eps_0p125": 0,
        "g1_exact": 12,
    },
    "scaled_independent": {
        "g3_zero": 12,
        "g2_eps_1": 12,
        "g2_eps_0p5": 0,
        "g2_eps_0p25": 0,
        "g2_eps_0p125": 0,
        "g1_exact": 12,
    },
    "balanced_correlated": {
        "g3_zero": 12,
        "g2_eps_1": 12,
        "g2_eps_0p5": 0,
        "g2_eps_0p25": 0,
        "g2_eps_0p125": 0,
        "g1_exact": 12,
    },
    "scaled_correlated": {
        "g3_zero": 9,
        "g2_eps_1": 11,
        "g2_eps_0p5": 0,
        "g2_eps_0p25": 0,
        "g2_eps_0p125": 0,
        "g1_exact": 12,
    },
    "diabetes": {
        "g3_zero": 12,
        "g2_eps_1": 12,
        "g2_eps_0p5": 0,
        "g2_eps_0p25": 0,
        "g2_eps_0p125": 0,
        "g1_exact": 12,
    },
}

STAGE4_EXPECTED_COSTED_RATIOS = {
    "balanced_independent": 1.62345465,
    "scaled_independent": 1.62345465,
    "balanced_correlated": 0.74887704,
    "scaled_correlated": 0.85364598,
    "diabetes": 0.16747397,
}

STAGE5_CASES = (
    "heldout_balanced_moderate",
    "heldout_balanced_strong",
    "heldout_scaled_moderate",
    "heldout_scaled_strong",
    "california_housing",
    "concrete_compressive_strength",
    "airfoil_self_noise",
    "residential_building",
)

STAGE5_METHODS = (
    "uniform",
    "lipschitz",
    "ascd_g3_zero",
    "ascd_g1_exact",
)

STAGE5_EXPECTED_SUCCESS = {
    "heldout_balanced_moderate": {
        "uniform": 12,
        "lipschitz": 12,
        "ascd_g3_zero": 12,
        "ascd_g1_exact": 12,
    },
    "heldout_balanced_strong": {
        "uniform": 12,
        "lipschitz": 12,
        "ascd_g3_zero": 12,
        "ascd_g1_exact": 12,
    },
    "heldout_scaled_moderate": {
        "uniform": 12,
        "lipschitz": 0,
        "ascd_g3_zero": 12,
        "ascd_g1_exact": 12,
    },
    "heldout_scaled_strong": {
        "uniform": 12,
        "lipschitz": 0,
        "ascd_g3_zero": 12,
        "ascd_g1_exact": 9,
    },
    "california_housing": {
        "uniform": 12,
        "lipschitz": 12,
        "ascd_g3_zero": 12,
        "ascd_g1_exact": 12,
    },
    "concrete_compressive_strength": {
        "uniform": 12,
        "lipschitz": 12,
        "ascd_g3_zero": 12,
        "ascd_g1_exact": 12,
    },
    "airfoil_self_noise": {
        "uniform": 12,
        "lipschitz": 12,
        "ascd_g3_zero": 12,
        "ascd_g1_exact": 12,
    },
    "residential_building": {
        "uniform": 0,
        "lipschitz": 0,
        "ascd_g3_zero": 0,
        "ascd_g1_exact": 0,
    },
}

STAGE5_EXPECTED_DIMENSIONS = {
    "heldout_balanced_moderate": (800, 96),
    "heldout_balanced_strong": (800, 96),
    "heldout_scaled_moderate": (800, 96),
    "heldout_scaled_strong": (800, 96),
    "california_housing": (16512, 8),
    "concrete_compressive_strength": (824, 8),
    "airfoil_self_noise": (1202, 5),
    "residential_building": (297, 107),
}

STAGE5_EXPECTED_RUNTIME = {
    "heldout_balanced_moderate": (
        0.8179,
        0.6954,
        0.9811,
    ),
    "heldout_balanced_strong": (
        0.8389,
        0.7417,
        0.9123,
    ),
    "heldout_scaled_moderate": (
        3.2393,
        2.6883,
        4.2613,
    ),
    "heldout_scaled_strong": (
        3.1274,
        2.1715,
        3.3162,
    ),
    "california_housing": (
        0.2080,
        0.1859,
        0.2802,
    ),
    "concrete_compressive_strength": (
        0.3779,
        0.3655,
        0.4520,
    ),
    "airfoil_self_noise": (
        0.5132,
        0.3836,
        0.6628,
    ),
}

EXPECTED_STAGE5_QUALIFYING_CASES = {
    "heldout_balanced_moderate",
    "heldout_balanced_strong",
    "california_housing",
    "concrete_compressive_strength",
    "airfoil_self_noise",
}


# =============================================================================
# Basic file helpers
# =============================================================================

def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def relative(path: Path) -> str:
    try:
        return str(
            path.relative_to(ROOT)
        ).replace("\\", "/")
    except ValueError:
        return str(path)


def read_json(
    path: Path,
) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            path
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
            path
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


def git_value(
    *arguments: str,
) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                *arguments,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

        return result.stdout.strip()

    except Exception:
        return None


def json_safe(
    value: Any,
) -> Any:
    """Recursively convert audit values to JSON-serializable objects.

    In particular, Python sets are converted to deterministically
    sorted lists. This preserves the content of audit diagnostics
    while allowing the final report_audit.json artifact to be written.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return relative(
            value
        )

    if isinstance(
        value,
        set,
    ):
        return [
            json_safe(
                item
            )
            for item
            in sorted(
                value,
                key=lambda item: str(
                    item
                ),
            )
        ]

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(
                item
            )
            for item
            in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(
                item
            )
            for key, item
            in value.items()
        }

    # Defensive fallback for unusual diagnostic objects.
    return str(
        value
    )


def numeric(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
        result = float(value)

        return (
            result
            if math.isfinite(result)
            else None
        )

    text = str(value).strip()

    if text == "":
        return None

    try:
        result = float(text)
    except ValueError:
        return None

    return (
        result
        if math.isfinite(result)
        else None
    )


# =============================================================================
# Recursive JSON helpers
# =============================================================================

def recursive_dicts(
    value: Any,
) -> Iterable[dict[str, Any]]:
    if isinstance(
        value,
        dict,
    ):
        yield value

        for child in value.values():
            yield from recursive_dicts(
                child
            )

    elif isinstance(
        value,
        list,
    ):
        for child in value:
            yield from recursive_dicts(
                child
            )


def find_rows(
    value: Any,
    required_keys: set[str],
) -> list[dict[str, Any]]:
    return [
        item
        for item in recursive_dicts(
            value
        )
        if required_keys.issubset(
            set(item)
        )
    ]


# =============================================================================
# Audit result recorder
# =============================================================================

class Audit:
    def __init__(
        self,
    ) -> None:
        self.checks: list[
            dict[str, Any]
        ] = []

    def _record(
        self,
        *,
        check_id: str,
        passed: bool,
        expected: Any,
        observed: Any,
        source: Path | str,
        note: str = "",
    ) -> None:
        self.checks.append(
            {
                "id": check_id,
                "status": (
                    "PASSED"
                    if passed
                    else "FAILED"
                ),
                "expected": expected,
                "observed": observed,
                "source": (
                    relative(source)
                    if isinstance(
                        source,
                        Path,
                    )
                    else source
                ),
                "note": note,
            }
        )

    def true(
        self,
        check_id: str,
        condition: bool,
        *,
        source: Path | str,
        observed: Any = None,
        note: str = "",
    ) -> None:
        self._record(
            check_id=check_id,
            passed=bool(
                condition
            ),
            expected=True,
            observed=(
                condition
                if observed is None
                else observed
            ),
            source=source,
            note=note,
        )

    def equal(
        self,
        check_id: str,
        observed: Any,
        expected: Any,
        *,
        source: Path | str,
        note: str = "",
    ) -> None:
        self._record(
            check_id=check_id,
            passed=(
                observed
                == expected
            ),
            expected=expected,
            observed=observed,
            source=source,
            note=note,
        )

    def close(
        self,
        check_id: str,
        observed: float | None,
        expected: float,
        *,
        source: Path | str,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        note: str = "",
    ) -> None:
        passed = (
            observed is not None
            and math.isclose(
                observed,
                expected,
                abs_tol=atol,
                rel_tol=rtol,
            )
        )

        self._record(
            check_id=check_id,
            passed=passed,
            expected=expected,
            observed=observed,
            source=source,
            note=note,
        )

    def contains(
        self,
        check_id: str,
        text: str,
        needle: str,
        *,
        source: Path | str,
        note: str = "",
    ) -> None:
        self._record(
            check_id=check_id,
            passed=(
                needle in text
            ),
            expected=(
                f"contains {needle!r}"
            ),
            observed=(
                needle in text
            ),
            source=source,
            note=note,
        )

    def absent(
        self,
        check_id: str,
        text: str,
        needle: str,
        *,
        source: Path | str,
        note: str = "",
    ) -> None:
        self._record(
            check_id=check_id,
            passed=(
                needle not in text
            ),
            expected=(
                f"does not contain {needle!r}"
            ),
            observed=(
                needle in text
            ),
            source=source,
            note=note,
        )

    @property
    def failed(
        self,
    ) -> list[dict[str, Any]]:
        return [
            check
            for check
            in self.checks
            if check[
                "status"
            ] == "FAILED"
        ]


# =============================================================================
# Trajectory-derived historical results
# =============================================================================

def first_work_to_target(
    rows: list[dict[str, str]],
    *,
    target: float,
) -> dict[
    tuple[str, int, str],
    float,
]:
    """Return first proxy work at target for each case/seed/method."""

    grouped: dict[
        tuple[str, int, str],
        list[
            tuple[
                int,
                float,
                float,
            ]
        ],
    ] = {}

    for row in rows:
        case = (
            row.get(
                "case",
                "",
            )
            .strip()
        )

        method = (
            row.get(
                "method",
                "",
            )
            .strip()
        )

        seed_text = (
            row.get(
                "seed",
                ""
            )
            .strip()
        )

        step_text = (
            row.get(
                "steps",
                ""
            )
            .strip()
        )

        gap_text = (
            row.get(
                "gap",
                ""
            )
            .strip()
        )

        work_text = (
            row.get(
                "proxy_work",
                ""
            )
            .strip()
        )

        if not (
            case
            and method
            and seed_text
            and step_text
            and gap_text
            and work_text
        ):
            continue

        key = (
            case,
            int(seed_text),
            method,
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            (
                int(step_text),
                float(gap_text),
                float(work_text),
            )
        )

    result: dict[
        tuple[str, int, str],
        float,
    ] = {}

    for key, values in grouped.items():
        values.sort(
            key=lambda item: item[0]
        )

        for (
            _step,
            gap,
            work,
        ) in values:
            if gap <= target:
                result[
                    key
                ] = work
                break

    return result


# =============================================================================
# Main audit phases
# =============================================================================

def audit_required_files(
    audit: Audit,
) -> None:
    required = (
        REPORT_TEX,
        BASELINE_TRAJECTORIES,
        STAGE3_SUMMARY,
        STAGE3_TRAJECTORIES,
        STAGE4_SUMMARY,
        STAGE5_MANIFEST,
        STAGE5_SUMMARY,
        STAGE5_TIMING,
        STAGE5_RUN_RESULTS,
        STAGE5_TRAJECTORIES,
    )

    for path in required:
        audit.true(
            "file_exists:"
            + relative(path),
            path.is_file(),
            source=path,
            observed=path.is_file(),
        )


def audit_baseline(
    audit: Audit,
) -> None:
    rows = read_csv(
        BASELINE_TRAJECTORIES
    )

    work = first_work_to_target(
        rows,
        target=BASELINE_TARGET,
    )

    for case in BASELINE_CASES:
        for method in BASELINE_METHODS:
            observed = sum(
                1
                for seed
                in range(12)
                if (
                    case,
                    seed,
                    method,
                ) in work
            )

            audit.equal(
                (
                    f"baseline_success:"
                    f"{case}:"
                    f"{method}"
                ),
                observed,
                BASELINE_EXPECTED_SUCCESS[
                    case
                ][
                    method
                ],
                source=(
                    BASELINE_TRAJECTORIES
                ),
            )


def audit_stage3(
    audit: Audit,
) -> None:
    summary = read_json(
        STAGE3_SUMMARY
    )

    summary_rows = find_rows(
        summary,
        {
            "case",
            "reached",
            "median_active_set_size",
            "median_proxy_work_to_target",
        },
    )

    by_case = {
        row[
            "case"
        ]: row
        for row in summary_rows
        if row.get(
            "case"
        ) in STAGE3_EXPECTED
    }

    for case, expected in (
        STAGE3_EXPECTED.items()
    ):
        row = by_case.get(
            case
        )

        audit.true(
            f"stage3_summary_row:{case}",
            row is not None,
            source=STAGE3_SUMMARY,
            observed=(
                None
                if row is None
                else row.get(
                    "case"
                )
            ),
        )

        if row is None:
            continue

        audit.equal(
            f"stage3_success:{case}",
            int(
                row[
                    "reached"
                ]
            ),
            expected[
                "success"
            ],
            source=STAGE3_SUMMARY,
        )

        audit.close(
            f"stage3_work:{case}",
            numeric(
                row[
                    "median_proxy_work_to_target"
                ]
            ),
            expected[
                "work"
            ],
            source=STAGE3_SUMMARY,
            atol=0.5,
            rtol=0.0,
        )

        audit.close(
            f"stage3_active_set:{case}",
            numeric(
                row[
                    "median_active_set_size"
                ]
            ),
            expected[
                "active"
            ],
            source=STAGE3_SUMMARY,
            atol=1e-12,
            rtol=0.0,
        )

    # -------------------------------------------------------------------------
    # Reconstruct the Stage 3 paired scaled-correlated comparison directly
    # from frozen trajectories rather than relying on prose.
    # -------------------------------------------------------------------------

    baseline_rows = read_csv(
        BASELINE_TRAJECTORIES
    )

    stage3_rows = read_csv(
        STAGE3_TRAJECTORIES
    )

    baseline_work = (
        first_work_to_target(
            baseline_rows,
            target=BASELINE_TARGET,
        )
    )

    stage3_work = (
        first_work_to_target(
            stage3_rows,
            target=BASELINE_TARGET,
        )
    )

    uniform_by_seed: dict[
        int,
        float,
    ] = {}

    ascd_by_seed: dict[
        int,
        float,
    ] = {}

    for seed in range(12):
        key = (
            "scaled_correlated",
            seed,
            "uniform",
        )

        if key in baseline_work:
            uniform_by_seed[
                seed
            ] = baseline_work[
                key
            ]

    # Determine the actual ASCD method name from the trajectory file.
    ascd_methods = sorted(
        {
            row[
                "method"
            ]
            for row
            in stage3_rows
            if row.get(
                "case"
            ) == "scaled_correlated"
        }
    )

    audit.true(
        "stage3_scaled_correlated_method_detected",
        len(
            ascd_methods
        ) == 1,
        source=STAGE3_TRAJECTORIES,
        observed=ascd_methods,
    )

    if ascd_methods:
        method_name = (
            ascd_methods[0]
        )

        for seed in range(12):
            key = (
                "scaled_correlated",
                seed,
                method_name,
            )

            if key in stage3_work:
                ascd_by_seed[
                    seed
                ] = stage3_work[
                    key
                ]

    shared = sorted(
        set(
            uniform_by_seed
        )
        & set(
            ascd_by_seed
        )
    )

    audit.equal(
        "stage3_shared_success_count",
        len(
            shared
        ),
        5,
        source=STAGE3_TRAJECTORIES,
    )

    audit.equal(
        "stage3_shared_success_seeds",
        shared,
        [
            0,
            4,
            5,
            6,
            10,
        ],
        source=STAGE3_TRAJECTORIES,
    )

    if shared:
        differences = [
            ascd_by_seed[
                seed
            ]
            - uniform_by_seed[
                seed
            ]
            for seed
            in shared
        ]

        ratios = [
            ascd_by_seed[
                seed
            ]
            / uniform_by_seed[
                seed
            ]
            for seed
            in shared
        ]

        median_difference = float(
            statistics.median(
                differences
            )
        )

        median_ratio = float(
            statistics.median(
                ratios
            )
        )

        audit.true(
            "stage3_ascd_work_greater_on_every_shared_success",
            all(
                difference > 0
                for difference
                in differences
            ),
            source=STAGE3_TRAJECTORIES,
            observed=differences,
        )

        audit.close(
            "stage3_median_paired_work_difference",
            median_difference,
            997336.0,
            source=STAGE3_TRAJECTORIES,
            atol=0.5,
            rtol=0.0,
        )

        audit.close(
            "stage3_median_paired_work_ratio",
            median_ratio,
            1.5401,
            source=STAGE3_TRAJECTORIES,
            atol=5e-5,
            rtol=0.0,
        )


def audit_stage4(
    audit: Audit,
) -> None:
    payload = read_json(
        STAGE4_SUMMARY
    )

    aggregate_rows = find_rows(
        payload,
        {
            "case",
            "config",
            "successes",
            "active_set_median",
            "median_zero_lower_fraction",
        },
    )

    lookup = {
        (
            row[
                "case"
            ],
            row[
                "config"
            ],
        ): row
        for row
        in aggregate_rows
        if (
            row.get(
                "case"
            )
            in STAGE4_CASES
            and row.get(
                "config"
            )
            in STAGE4_CONFIGS
        )
    }

    for case in STAGE4_CASES:
        for config in STAGE4_CONFIGS:
            key = (
                case,
                config,
            )

            row = lookup.get(
                key
            )

            audit.true(
                (
                    f"stage4_row:"
                    f"{case}:"
                    f"{config}"
                ),
                row is not None,
                source=STAGE4_SUMMARY,
                observed=(
                    None
                    if row is None
                    else key
                ),
            )

            if row is None:
                continue

            audit.equal(
                (
                    f"stage4_success:"
                    f"{case}:"
                    f"{config}"
                ),
                int(
                    row[
                        "successes"
                    ]
                ),
                STAGE4_EXPECTED_SUCCESS[
                    case
                ][
                    config
                ],
                source=STAGE4_SUMMARY,
            )

            d = (
                10
                if case
                == "diabetes"
                else 48
            )

            if config == "g3_zero":
                audit.close(
                    (
                        f"stage4_g3_active:"
                        f"{case}"
                    ),
                    numeric(
                        row[
                            "active_set_median"
                        ]
                    ),
                    float(d),
                    source=STAGE4_SUMMARY,
                    atol=1e-12,
                    rtol=0.0,
                )

                audit.close(
                    (
                        f"stage4_g3_zero_lower:"
                        f"{case}"
                    ),
                    numeric(
                        row[
                            "median_zero_lower_fraction"
                        ]
                    ),
                    1.0,
                    source=STAGE4_SUMMARY,
                    atol=1e-12,
                    rtol=0.0,
                )

                full_fraction = numeric(
                    row.get(
                        "fraction_full_active_set"
                    )
                )

                audit.true(
                    (
                        f"stage4_g3_effectively_full:"
                        f"{case}"
                    ),
                    (
                        full_fraction
                        is not None
                        and full_fraction
                        >= 0.998
                    ),
                    source=STAGE4_SUMMARY,
                    observed=full_fraction,
                    note=(
                        "Threshold is an audit of the "
                        "report's approximately-full statement."
                    ),
                )

            elif config == "g1_exact":
                audit.close(
                    (
                        f"stage4_g1_active:"
                        f"{case}"
                    ),
                    numeric(
                        row[
                            "active_set_median"
                        ]
                    ),
                    1.0,
                    source=STAGE4_SUMMARY,
                    atol=1e-12,
                    rtol=0.0,
                )

            elif config in {
                "g2_eps_0p5",
                "g2_eps_0p25",
                "g2_eps_0p125",
            }:
                expected_active = (
                    9.0
                    if case
                    == "diabetes"
                    else 47.0
                )

                audit.close(
                    (
                        f"stage4_intermediate_active:"
                        f"{case}:"
                        f"{config}"
                    ),
                    numeric(
                        row[
                            "active_set_median"
                        ]
                    ),
                    expected_active,
                    source=STAGE4_SUMMARY,
                    atol=1e-12,
                    rtol=0.0,
                )

    paired = payload.get(
        "paired_vs_g3",
        {},
    )

    for case, expected_ratio in (
        STAGE4_EXPECTED_COSTED_RATIOS.items()
    ):
        case_payload = paired.get(
            case,
            {},
        )

        g1_payload = (
            case_payload.get(
                "g1_exact",
                {}
            )
        )

        observed = numeric(
            g1_payload.get(
                "median_paired_costed_ratio"
            )
        )

        audit.close(
            (
                f"stage4_costed_ratio:"
                f"{case}"
            ),
            observed,
            expected_ratio,
            source=STAGE4_SUMMARY,
            atol=5e-8,
            rtol=1e-7,
        )


def audit_stage5_manifest(
    audit: Audit,
    manifest: dict[str, Any],
) -> None:
    protocol = manifest[
        "protocol"
    ]

    synthetic = protocol[
        "synthetic"
    ]

    audit.equal(
        "stage5_split_seed",
        protocol[
            "split_seed"
        ],
        20260926,
        source=STAGE5_MANIFEST,
    )

    audit.close(
        "stage5_train_fraction",
        numeric(
            protocol[
                "train_fraction"
            ]
        ),
        0.80,
        source=STAGE5_MANIFEST,
        atol=1e-12,
        rtol=0.0,
    )

    audit.close(
        "stage5_test_fraction",
        numeric(
            protocol[
                "test_fraction"
            ]
        ),
        0.20,
        source=STAGE5_MANIFEST,
        atol=1e-12,
        rtol=0.0,
    )

    audit.equal(
        "stage5_synthetic_n",
        synthetic[
            "n"
        ],
        800,
        source=STAGE5_MANIFEST,
    )

    audit.equal(
        "stage5_synthetic_d",
        synthetic[
            "d"
        ],
        96,
        source=STAGE5_MANIFEST,
    )

    audit.equal(
        "stage5_synthetic_seeds",
        synthetic[
            "seeds"
        ],
        list(
            range(
                100,
                112,
            )
        ),
        source=STAGE5_MANIFEST,
    )

    audit.close(
        "stage5_noise_std",
        numeric(
            synthetic[
                "noise_std"
            ]
        ),
        0.15,
        source=STAGE5_MANIFEST,
        atol=1e-12,
        rtol=0.0,
    )

    real_records = manifest[
        "real_datasets"
    ]

    synthetic_records = manifest[
        "synthetic_datasets"
    ]

    audit.equal(
        "stage5_real_dataset_count",
        len(
            real_records
        ),
        4,
        source=STAGE5_MANIFEST,
    )

    audit.equal(
        "stage5_synthetic_instance_count",
        len(
            synthetic_records
        ),
        48,
        source=STAGE5_MANIFEST,
    )

    residential = next(
        (
            row
            for row
            in real_records
            if row.get(
                "slug"
            )
            == "residential_building"
        ),
        None,
    )

    audit.true(
        "stage5_residential_manifest_row",
        residential
        is not None,
        source=STAGE5_MANIFEST,
        observed=(
            None
            if residential is None
            else residential.get(
                "slug"
            )
        ),
    )

    if residential is not None:
        audit.equal(
            "stage5_residential_target",
            residential[
                "selected_target"
            ],
            "V-9",
            source=STAGE5_MANIFEST,
        )

        audit.equal(
            "stage5_residential_predictor_count",
            residential[
                "prepared_feature_count"
            ],
            107,
            source=STAGE5_MANIFEST,
        )

        names = residential[
            "kept_feature_names"
        ]

        audit.equal(
            "stage5_residential_feature_name_count",
            len(
                names
            ),
            107,
            source=STAGE5_MANIFEST,
        )

        audit.equal(
            "stage5_residential_unique_feature_names",
            len(
                set(
                    names
                )
            ),
            107,
            source=STAGE5_MANIFEST,
        )

        audit.true(
            "stage5_residential_no_v9_leakage",
            "V-9"
            not in names,
            source=STAGE5_MANIFEST,
            observed=(
                "V-9"
                in names
            ),
        )

        audit.true(
            "stage5_residential_no_v10_leakage",
            "V-10"
            not in names,
            source=STAGE5_MANIFEST,
            observed=(
                "V-10"
                in names
            ),
        )

        for name in (
            "V-11_lag1",
            "V-29_lag1",
            "V-11_lag5",
            "V-29_lag5",
        ):
            audit.true(
                (
                    "stage5_residential_lag_name:"
                    + name
                ),
                name in names,
                source=STAGE5_MANIFEST,
                observed=(
                    name in names
                ),
            )


def audit_stage5(
    audit: Audit,
) -> None:
    summary = read_json(
        STAGE5_SUMMARY
    )

    manifest = read_json(
        STAGE5_MANIFEST
    )

    timing = read_json(
        STAGE5_TIMING
    )

    audit_stage5_manifest(
        audit,
        manifest,
    )

    protocol = summary[
        "analysis_protocol"
    ]

    input_audit = summary[
        "input_audit"
    ]

    audit.close(
        "stage5_primary_target",
        numeric(
            protocol[
                "primary_normalized_gap"
            ]
        ),
        1e-6,
        source=STAGE5_SUMMARY,
        atol=1e-15,
        rtol=0.0,
    )

    audit.close(
        "stage5_secondary_target",
        numeric(
            protocol[
                "secondary_normalized_gap"
            ]
        ),
        1e-4,
        source=STAGE5_SUMMARY,
        atol=1e-15,
        rtol=0.0,
    )

    audit.equal(
        "stage5_bootstrap_resamples",
        protocol[
            "bootstrap_resamples"
        ],
        10000,
        source=STAGE5_SUMMARY,
    )

    audit.equal(
        "stage5_run_rows",
        input_audit[
            "run_rows"
        ],
        384,
        source=STAGE5_SUMMARY,
    )

    audit.equal(
        "stage5_trajectory_rows",
        input_audit[
            "trajectory_rows"
        ],
        15744,
        source=STAGE5_SUMMARY,
    )

    audit.equal(
        "stage5_checkpoints_per_run",
        input_audit[
            "expected_checkpoints_per_run"
        ],
        41,
        source=STAGE5_SUMMARY,
    )

    by_case = summary[
        "by_case"
    ]

    for case in STAGE5_CASES:
        payload = by_case[
            case
        ]

        observed_dimensions = (
            payload[
                "n"
            ],
            payload[
                "d"
            ],
        )

        audit.equal(
            (
                f"stage5_dimensions:"
                f"{case}"
            ),
            observed_dimensions,
            STAGE5_EXPECTED_DIMENSIONS[
                case
            ],
            source=STAGE5_SUMMARY,
        )

        methods = payload[
            "methods"
        ]

        for method in STAGE5_METHODS:
            observed = methods[
                method
            ][
                "success_primary_count"
            ]

            audit.equal(
                (
                    f"stage5_primary_success:"
                    f"{case}:"
                    f"{method}"
                ),
                observed,
                STAGE5_EXPECTED_SUCCESS[
                    case
                ][
                    method
                ],
                source=STAGE5_SUMMARY,
            )

        g3 = methods[
            "ascd_g3_zero"
        ]

        g1 = methods[
            "ascd_g1_exact"
        ]

        audit.close(
            (
                f"stage5_g3_active_fraction:"
                f"{case}"
            ),
            numeric(
                g3[
                    "median_active_set_fraction"
                ]
            ),
            1.0,
            source=STAGE5_SUMMARY,
            atol=1e-12,
            rtol=0.0,
        )

        audit.close(
            (
                f"stage5_g3_zero_lower:"
                f"{case}"
            ),
            numeric(
                g3[
                    "median_zero_lower_fraction"
                ]
            ),
            1.0,
            source=STAGE5_SUMMARY,
            atol=1e-12,
            rtol=0.0,
        )

        audit.close(
            (
                f"stage5_g1_active_size:"
                f"{case}"
            ),
            numeric(
                g1[
                    "median_active_set_size"
                ]
            ),
            1.0,
            source=STAGE5_SUMMARY,
            atol=1e-12,
            rtol=0.0,
        )

        expected_fraction = (
            1.0
            / payload[
                "d"
            ]
        )

        audit.close(
            (
                f"stage5_g1_active_fraction:"
                f"{case}"
            ),
            numeric(
                g1[
                    "median_active_set_fraction"
                ]
            ),
            expected_fraction,
            source=STAGE5_SUMMARY,
            atol=1e-12,
            rtol=0.0,
        )

    residential_g1 = (
        by_case[
            "residential_building"
        ][
            "methods"
        ][
            "ascd_g1_exact"
        ]
    )

    audit.equal(
        "stage5_residential_g1_secondary_successes",
        residential_g1[
            "success_secondary_count"
        ],
        12,
        source=STAGE5_SUMMARY,
    )

    paired = summary[
        "g1_vs_g3_paired"
    ]

    for (
        case,
        (
            expected_median,
            expected_lower,
            expected_upper,
        ),
    ) in (
        STAGE5_EXPECTED_RUNTIME.items()
    ):
        ratio = (
            paired[
                case
            ][
                "harmonized_runtime"
            ][
                "paired_ratio_g1_over_g3"
            ]
        )

        audit.true(
            (
                f"stage5_runtime_ratio_defined:"
                f"{case}"
            ),
            ratio is not None,
            source=STAGE5_SUMMARY,
            observed=ratio,
        )

        if ratio is None:
            continue

        audit.close(
            (
                f"stage5_runtime_ratio_median:"
                f"{case}"
            ),
            numeric(
                ratio[
                    "median"
                ]
            ),
            expected_median,
            source=STAGE5_SUMMARY,
            atol=5e-5,
            rtol=0.0,
        )

        audit.close(
            (
                f"stage5_runtime_ratio_ci_lower:"
                f"{case}"
            ),
            numeric(
                ratio[
                    "ci95_lower"
                ]
            ),
            expected_lower,
            source=STAGE5_SUMMARY,
            atol=5e-5,
            rtol=0.0,
        )

        audit.close(
            (
                f"stage5_runtime_ratio_ci_upper:"
                f"{case}"
            ),
            numeric(
                ratio[
                    "ci95_upper"
                ]
            ),
            expected_upper,
            source=STAGE5_SUMMARY,
            atol=5e-5,
            rtol=0.0,
        )

        audit.equal(
            (
                f"stage5_runtime_bootstrap_count:"
                f"{case}"
            ),
            ratio[
                "bootstrap_resamples"
            ],
            10000,
            source=STAGE5_SUMMARY,
        )

    residential_ratio = (
        paired[
            "residential_building"
        ][
            "harmonized_runtime"
        ][
            "paired_ratio_g1_over_g3"
        ]
    )

    audit.equal(
        "stage5_residential_runtime_ratio_undefined",
        residential_ratio,
        None,
        source=STAGE5_SUMMARY,
    )

    decision = summary[
        "confirmatory_decision_rules"
    ]

    h4 = decision[
        "H4_stage4_mechanism_generalizes"
    ]

    audit.equal(
        "stage5_h4_active_set_count",
        h4[
            "active_set_count"
        ],
        8,
        source=STAGE5_SUMMARY,
    )

    audit.equal(
        "stage5_h4_zero_lower_count",
        h4[
            "zero_lower_count"
        ],
        8,
        source=STAGE5_SUMMARY,
    )

    audit.equal(
        "stage5_h4_criterion_met",
        h4[
            "criterion_met"
        ],
        True,
        source=STAGE5_SUMMARY,
    )

    pays = decision[
        "predeclared_exact_information_can_sometimes_pay_rule"
    ]

    audit.equal(
        "stage5_exact_information_sometimes_pays",
        pays[
            "criterion_met"
        ],
        True,
        source=STAGE5_SUMMARY,
    )

    audit.equal(
        "stage5_exact_information_qualifying_cases",
        sorted(
            pays[
                "qualifying_cases"
            ]
        ),
        sorted(
            EXPECTED_STAGE5_QUALIFYING_CASES
        ),
        source=STAGE5_SUMMARY,
    )

    h3 = decision[
        "H3_information_cost_is_regime_dependent"
    ]

    audit.equal(
        "stage5_runtime_mixed_direction",
        h3[
            "mixed_runtime_direction_observed"
        ],
        True,
        source=STAGE5_SUMMARY,
    )

    # -------------------------------------------------------------------------
    # Timing protocol
    # -------------------------------------------------------------------------

    timing_protocol = timing[
        "timing_protocol"
    ]

    audit.equal(
        "stage5_timing_warmups",
        timing_protocol[
            "warmup_executions_per_method_problem"
        ],
        1,
        source=STAGE5_TIMING,
    )

    audit.equal(
        "stage5_timing_repetitions",
        timing_protocol[
            "timed_repetitions_per_method_problem"
        ],
        3,
        source=STAGE5_TIMING,
    )

    for key in (
        "method_specific_setup_included",
        "coordinate_selection_included",
        "coordinate_updates_included",
        "g1_gram_acquisition_included",
        "checkpoint_evaluation_excluded_from_primary_runtime",
        "dense_reference_optimum_excluded_from_method_runtime",
    ):
        audit.equal(
            (
                "stage5_timing_protocol:"
                + key
            ),
            timing_protocol[
                key
            ],
            True,
            source=STAGE5_TIMING,
        )

    audit.equal(
        "stage5_requested_threads",
        timing[
            "threadpool_policy"
        ][
            "requested_threads"
        ],
        1,
        source=STAGE5_TIMING,
    )


def normalize_tex(
    text: str,
) -> str:
    text = re.sub(
        r"%.*?$",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def audit_report_text(
    audit: Audit,
) -> None:
    """Audit important numerical and interpretive claims in main.tex.

    The numerical checks below intentionally complement, rather than replace,
    the artifact-level checks performed earlier in this script.

    Interpretation checks accept a small set of semantically equivalent
    phrasings so that harmless grammatical edits such as singular/plural
    "reproduction(s)" do not cause a false audit failure.
    """

    raw_text = REPORT_TEX.read_text(
        encoding="utf-8"
    )

    text = normalize_tex(
        raw_text
    )

    lower_text = (
        text.lower()
    )

    # -------------------------------------------------------------------------
    # Important numerical values that should occur in the final report.
    # -------------------------------------------------------------------------

    numerical_tokens = (
        # Stage 3 paired ratio.
        "1.5401",

        # Stage 4 costed g1/g3 ratios.
        "1.6235",
        "0.7489",
        "0.8536",
        "0.1675",

        # Stage 5 paired runtime ratios.
        "0.8179",
        "0.8389",
        "3.2393",
        "3.1274",
        "0.2080",
        "0.3779",
        "0.5132",
    )

    for token in numerical_tokens:
        audit.contains(
            (
                "report_contains_numeric:"
                + token
            ),
            text,
            token,
            source=REPORT_TEX,
            note=(
                "The numerical value is also independently "
                "validated against its canonical Stage 3--5 "
                "artifact elsewhere in this audit."
            ),
        )

    # -------------------------------------------------------------------------
    # Important interpretation / limitation language.
    #
    # These checks deliberately accept several semantically equivalent
    # grammatical forms. The purpose is to verify the scientific limitation,
    # not enforce one exact English sentence.
    # -------------------------------------------------------------------------

    interpretation_checks: dict[
        str,
        tuple[str, ...],
    ] = {
        "not_a_reproduction": (
            "not a reproduction",
            "not reproductions",
            "not a reproduction of",
            "not reproductions of",
        ),

        "bookkeeping": (
            "bookkeeping",
        ),

        "regime_dependent": (
            "regime-dependent",
            "regime dependent",
        ),

        "not_supported": (
            "not supported",
        ),

        "ai_assistance": (
            "ai assistance",
            "ai-assisted",
            "ai assisted",
        ),
    }

    for (
        check_name,
        alternatives,
    ) in interpretation_checks.items():

        matched_alternatives = [
            phrase
            for phrase
            in alternatives
            if phrase.lower()
            in lower_text
        ]

        matched = bool(
            matched_alternatives
        )

        audit.true(
            (
                "report_interpretation_phrase:"
                + check_name
            ),
            matched,
            source=REPORT_TEX,
            observed={
                "matched": (
                    matched
                ),
                "matched_alternatives": (
                    matched_alternatives
                ),
                "accepted_alternatives": list(
                    alternatives
                ),
            },
            note=(
                "This is a semantic guardrail rather than "
                "an exact-sentence requirement."
            ),
        )

    # -------------------------------------------------------------------------
    # Strong conclusions that the completed evidence does NOT support.
    #
    # These checks remain outside the interpretation loop so that each
    # prohibited assertion is tested exactly once.
    # -------------------------------------------------------------------------

    prohibited_assertions = (
        "exact ascd is generally faster",
        "adaptive coordinate selection is generally superior",
        "exact information always pays for itself",
        "ascd is universally superior",
    )

    for phrase in prohibited_assertions:
        audit.absent(
            (
                "report_prohibited_claim:"
                + phrase
            ),
            lower_text,
            phrase,
            source=REPORT_TEX,
            note=(
                "The Stage 5 confirmatory result is "
                "regime-dependent and does not support "
                "this universal/general claim."
            ),
        )


# =============================================================================
# Output
# =============================================================================

def write_output(
    audit: Audit,
) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    failed = audit.failed

    payload = {
        "stage": 6,
        "audit": (
            "final numerical and report claim audit"
        ),
        "generated_at_utc": (
            utc_now()
        ),
        "git": {
            "commit": git_value(
                "rev-parse",
                "HEAD",
            ),
            "short_commit": git_value(
                "rev-parse",
                "--short=7",
                "HEAD",
            ),
            "branch": git_value(
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ),
        },
        "checks_total": len(
            audit.checks
        ),
        "checks_passed": (
            len(
                audit.checks
            )
            - len(
                failed
            )
        ),
        "checks_failed": len(
            failed
        ),
        "overall_status": (
            "PASSED"
            if not failed
            else "FAILED"
        ),
        "checks": audit.checks,
    }

    temporary = (
        OUTPUT_PATH
        .with_suffix(
            ".json.tmp"
        )
    )

    safe_payload = json_safe(
        payload
    )

    temporary.write_text(
        json.dumps(
            safe_payload,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # Parse the written artifact before replacing canonical output.
    json.loads(
        temporary.read_text(
            encoding="utf-8"
        )
    )

    temporary.replace(
        OUTPUT_PATH
    )

    return safe_payload


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit major numerical claims in the "
            "final optimization research report."
        )
    )

    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "Continue all audit phases even if an "
            "unexpected exception occurs in one phase. "
            "Normal use does not require this flag."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        "=" * 78
    )
    print(
        "STAGE 6 FINAL NUMERICAL / REPORT AUDIT"
    )
    print(
        "=" * 78
    )

    audit = Audit()

    phases = (
        (
            "required files",
            audit_required_files,
        ),
        (
            "baseline",
            audit_baseline,
        ),
        (
            "Stage 3",
            audit_stage3,
        ),
        (
            "Stage 4",
            audit_stage4,
        ),
        (
            "Stage 5",
            audit_stage5,
        ),
        (
            "report text",
            audit_report_text,
        ),
    )

    for phase_name, phase_function in phases:
        print(
            f"\n[{phase_name}]"
        )

        try:
            phase_function(
                audit
            )

            print(
                "  completed"
            )

        except Exception as exc:
            audit._record(
                check_id=(
                    "phase_exception:"
                    + phase_name
                ),
                passed=False,
                expected=(
                    "phase completes "
                    "without exception"
                ),
                observed=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                source=(
                    "stage6_report_audit.py"
                ),
                note=(
                    "An audit phase raised an exception."
                ),
            )

            print(
                f"  ERROR: {exc}"
            )

            if not args.keep_going:
                # Still write the partial audit artifact.
                payload = write_output(
                    audit
                )

                print()
                print(
                    f"Audit artifact written to:\n"
                    f"  {OUTPUT_PATH}"
                )

                print(
                    f"\nOverall status: "
                    f"{payload['overall_status']}"
                )

                raise SystemExit(
                    1
                ) from exc

    payload = write_output(
        audit
    )

    print()
    print(
        "-" * 78
    )

    print(
        f"Checks total : "
        f"{payload['checks_total']}"
    )

    print(
        f"Checks passed: "
        f"{payload['checks_passed']}"
    )

    print(
        f"Checks failed: "
        f"{payload['checks_failed']}"
    )

    print(
        f"Overall      : "
        f"{payload['overall_status']}"
    )

    print()
    print(
        f"Output:\n"
        f"  {OUTPUT_PATH}"
    )

    if audit.failed:
        print()
        print(
            "FAILED CHECKS:"
        )

        for check in audit.failed:
            print(
                "  - "
                f"{check['id']}: "
                f"expected={check['expected']!r}, "
                f"observed={check['observed']!r}"
            )

        raise SystemExit(
            1
        )

    print()
    print(
        "STAGE 6 REPORT AUDIT: PASSED"
    )


if __name__ == "__main__":
    main()
