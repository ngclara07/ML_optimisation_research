"""Stage 5 confirmatory analysis.

This script analyzes the complete output of:

    src/stage5_confirmatory_benchmark.py

It implements the predeclared Stage 5 analysis:

- primary and secondary success counts;
- coordinate epochs to target;
- harmonized wall-clock time to target;
- proxy-work summaries;
- ASCD active-set diagnostics;
- zero-lower-bound diagnostics;
- exact g1 versus zero-oracle g3 paired comparisons;
- method-only success sets;
- 10,000-resample nonparametric bootstrap confidence intervals;
- real-data test-RMSE diagnostics;
- confirmatory Stage 5 decision rules.

The script does not rerun optimization.

Inputs
------
results/stage5_confirmatory/
    run_results.csv
    trajectories.csv
    environment.json
    dataset_manifest.json

Outputs
-------
results/stage5_confirmatory/
    paired_results.csv
    summary.json
    timing_summary.json

The plotting stage is intentionally separate:

    src/stage5_plot.py

Run from repository root:

    python src/stage5_analyze.py --audit-only

then:

    python src/stage5_analyze.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# =============================================================================
# Repository paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    ROOT
    / "results"
    / "stage5_confirmatory"
)

RUN_RESULTS_PATH = (
    RESULT_ROOT
    / "run_results.csv"
)

TRAJECTORIES_PATH = (
    RESULT_ROOT
    / "trajectories.csv"
)

ENVIRONMENT_PATH = (
    RESULT_ROOT
    / "environment.json"
)

MANIFEST_PATH = (
    RESULT_ROOT
    / "dataset_manifest.json"
)

PAIRED_RESULTS_PATH = (
    RESULT_ROOT
    / "paired_results.csv"
)

SUMMARY_PATH = (
    RESULT_ROOT
    / "summary.json"
)

TIMING_SUMMARY_PATH = (
    RESULT_ROOT
    / "timing_summary.json"
)


# =============================================================================
# Frozen Stage 5 analysis constants
# =============================================================================

EXPECTED_RUN_ROWS = 384
EXPECTED_TRAJECTORY_ROWS = 15744
EXPECTED_CHECKPOINTS_PER_RUN = 41

EXPECTED_REPLICATES_PER_CASE_METHOD = 12

PRIMARY_TARGET = 1e-6
SECONDARY_TARGET = 1e-4

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_BASE_SEED = 20260926

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


# =============================================================================
# Required input columns
# =============================================================================

REQUIRED_RUN_COLUMNS = {
    "problem_id",
    "kind",
    "case",
    "problem_seed",
    "algorithm_seed",
    "method",
    "n",
    "d",
    "max_steps",
    "checkpoint_steps",
    "max_coordinate_epochs",
    "primary_target",
    "secondary_target",
    "success_primary",
    "success_secondary",
    "steps_to_primary",
    "epochs_to_primary",
    "steps_to_secondary",
    "epochs_to_secondary",
    "initial_objective",
    "reference_objective",
    "initial_absolute_gap",
    "final_objective",
    "final_absolute_gap",
    "final_relative_gap",
    "median_setup_seconds",
    "median_oracle_setup_seconds",
    "median_update_seconds",
    "median_total_method_seconds",
    "median_evaluation_seconds_excluded",
    "median_seconds_to_primary",
    "median_seconds_to_secondary",
    "repeat1_total_method_seconds",
    "repeat2_total_method_seconds",
    "repeat3_total_method_seconds",
    "repeat1_seconds_to_primary",
    "repeat2_seconds_to_primary",
    "repeat3_seconds_to_primary",
    "proxy_common_setup",
    "proxy_method_setup",
    "proxy_oracle_setup",
    "proxy_per_update",
    "proxy_core_to_primary",
    "proxy_costed_to_primary",
    "proxy_core_final",
    "proxy_costed_final",
    "evaluation_proxy_work_excluded",
    "median_active_set_size",
    "median_active_set_fraction",
    "full_active_fraction",
    "pruned_fraction",
    "half_or_less_active_fraction",
    "median_zero_lower_fraction",
    "reference_test_rmse_standardized",
    "final_test_rmse_standardized",
    "test_rmse_difference_standardized",
    "reference_test_rmse_original",
    "final_test_rmse_original",
    "test_rmse_difference_original",
    "final_weights_sha256",
}

REQUIRED_TRAJECTORY_COLUMNS = {
    "problem_id",
    "kind",
    "case",
    "problem_seed",
    "algorithm_seed",
    "method",
    "n",
    "d",
    "steps",
    "coordinate_epochs",
    "objective",
    "absolute_gap",
    "relative_gap",
    "proxy_core",
    "proxy_costed",
}


# =============================================================================
# General helpers
# =============================================================================

def utc_now_string() -> str:
    return (
        datetime
        .now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    json.loads(
        temporary.read_text(
            encoding="utf-8"
        )
    )

    temporary.replace(
        path
    )


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    fieldnames = list(
        rows[0].keys()
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            if list(
                row.keys()
            ) != fieldnames:
                raise RuntimeError(
                    "CSV row schema mismatch."
                )

            writer.writerow(
                {
                    key: (
                        ""
                        if value is None
                        else value
                    )
                    for key, value
                    in row.items()
                }
            )

    temporary.replace(
        path
    )


def read_csv(
    path: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                f"{path} has no header."
            )

        rows = list(
            reader
        )

        return (
            list(
                reader.fieldnames
            ),
            rows,
        )


def require_columns(
    *,
    observed: Iterable[str],
    required: set[str],
    source_name: str,
) -> None:
    observed_set = set(
        observed
    )

    missing = (
        required
        - observed_set
    )

    if missing:
        raise RuntimeError(
            f"{source_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def as_float(
    value: str | None,
    *,
    field: str,
    nullable: bool = False,
) -> float | None:
    if value is None:
        if nullable:
            return None

        raise ValueError(
            f"{field}: missing value."
        )

    stripped = value.strip()

    if stripped == "":
        if nullable:
            return None

        raise ValueError(
            f"{field}: empty value."
        )

    result = float(
        stripped
    )

    if not math.isfinite(
        result
    ):
        raise ValueError(
            f"{field}: non-finite value {result}."
        )

    return result


def as_int(
    value: str | None,
    *,
    field: str,
    nullable: bool = False,
) -> int | None:
    if value is None:
        if nullable:
            return None

        raise ValueError(
            f"{field}: missing value."
        )

    stripped = value.strip()

    if stripped == "":
        if nullable:
            return None

        raise ValueError(
            f"{field}: empty value."
        )

    return int(
        stripped
    )


def median_or_none(
    values: Iterable[
        float | None
    ],
) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return None

    return float(
        np.median(
            np.asarray(
                clean,
                dtype=np.float64,
            )
        )
    )


def mean_or_none(
    values: Iterable[
        float | None
    ],
) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return None

    return float(
        np.mean(
            np.asarray(
                clean,
                dtype=np.float64,
            )
        )
    )


def minimum_or_none(
    values: Iterable[
        float | None
    ],
) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return None

    return float(
        np.min(
            np.asarray(
                clean,
                dtype=np.float64,
            )
        )
    )


def maximum_or_none(
    values: Iterable[
        float | None
    ],
) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return None

    return float(
        np.max(
            np.asarray(
                clean,
                dtype=np.float64,
            )
        )
    )


# =============================================================================
# Deterministic bootstrap
# =============================================================================

def bootstrap_seed(
    label: str,
) -> int:
    digest = hashlib.sha256(
        label.encode(
            "utf-8"
        )
    ).digest()

    derived = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )

    return (
        derived
        ^ BOOTSTRAP_BASE_SEED
    ) % (
        2**63 - 1
    )


def bootstrap_median_ci(
    values: list[float],
    *,
    label: str,
) -> dict[str, Any] | None:
    if not values:
        return None

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if np.any(
        ~np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{label}: non-finite bootstrap input."
        )

    n = len(
        array
    )

    rng = np.random.default_rng(
        bootstrap_seed(
            label
        )
    )

    indices = rng.integers(
        low=0,
        high=n,
        size=(
            BOOTSTRAP_RESAMPLES,
            n,
        ),
    )

    resampled = array[
        indices
    ]

    medians = np.median(
        resampled,
        axis=1,
    )

    lower, upper = np.percentile(
        medians,
        [
            2.5,
            97.5,
        ],
    )

    return {
        "n": n,
        "median": float(
            np.median(
                array
            )
        ),
        "ci95_lower": float(
            lower
        ),
        "ci95_upper": float(
            upper
        ),
        "bootstrap_resamples": (
            BOOTSTRAP_RESAMPLES
        ),
        "bootstrap_seed": (
            bootstrap_seed(
                label
            )
        ),
    }


# =============================================================================
# Run-result parsing
# =============================================================================

def parse_run_row(
    raw: dict[str, str],
) -> dict[str, Any]:
    problem_seed = as_int(
        raw["problem_seed"],
        field="problem_seed",
        nullable=True,
    )

    algorithm_seed = as_int(
        raw["algorithm_seed"],
        field="algorithm_seed",
    )

    success_primary = as_int(
        raw["success_primary"],
        field="success_primary",
    )

    success_secondary = as_int(
        raw["success_secondary"],
        field="success_secondary",
    )

    if success_primary not in {
        0,
        1,
    }:
        raise ValueError(
            "success_primary is not binary."
        )

    if success_secondary not in {
        0,
        1,
    }:
        raise ValueError(
            "success_secondary is not binary."
        )

    row = {
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
            problem_seed
        ),

        "algorithm_seed": (
            algorithm_seed
        ),

        "method": (
            raw["method"]
        ),

        "n": as_int(
            raw["n"],
            field="n",
        ),

        "d": as_int(
            raw["d"],
            field="d",
        ),

        "max_steps": as_int(
            raw["max_steps"],
            field="max_steps",
        ),

        "checkpoint_steps": (
            as_int(
                raw[
                    "checkpoint_steps"
                ],
                field="checkpoint_steps",
            )
        ),

        "max_coordinate_epochs": (
            as_float(
                raw[
                    "max_coordinate_epochs"
                ],
                field="max_coordinate_epochs",
            )
        ),

        "primary_target": (
            as_float(
                raw["primary_target"],
                field="primary_target",
            )
        ),

        "secondary_target": (
            as_float(
                raw["secondary_target"],
                field="secondary_target",
            )
        ),

        "success_primary": (
            bool(
                success_primary
            )
        ),

        "success_secondary": (
            bool(
                success_secondary
            )
        ),

        "steps_to_primary": (
            as_int(
                raw[
                    "steps_to_primary"
                ],
                field="steps_to_primary",
                nullable=True,
            )
        ),

        "epochs_to_primary": (
            as_float(
                raw[
                    "epochs_to_primary"
                ],
                field="epochs_to_primary",
                nullable=True,
            )
        ),

        "steps_to_secondary": (
            as_int(
                raw[
                    "steps_to_secondary"
                ],
                field="steps_to_secondary",
                nullable=True,
            )
        ),

        "epochs_to_secondary": (
            as_float(
                raw[
                    "epochs_to_secondary"
                ],
                field="epochs_to_secondary",
                nullable=True,
            )
        ),

        "initial_objective": (
            as_float(
                raw[
                    "initial_objective"
                ],
                field="initial_objective",
            )
        ),

        "reference_objective": (
            as_float(
                raw[
                    "reference_objective"
                ],
                field="reference_objective",
            )
        ),

        "initial_absolute_gap": (
            as_float(
                raw[
                    "initial_absolute_gap"
                ],
                field="initial_absolute_gap",
            )
        ),

        "final_objective": (
            as_float(
                raw[
                    "final_objective"
                ],
                field="final_objective",
            )
        ),

        "final_absolute_gap": (
            as_float(
                raw[
                    "final_absolute_gap"
                ],
                field="final_absolute_gap",
            )
        ),

        "final_relative_gap": (
            as_float(
                raw[
                    "final_relative_gap"
                ],
                field="final_relative_gap",
            )
        ),

        "median_setup_seconds": (
            as_float(
                raw[
                    "median_setup_seconds"
                ],
                field="median_setup_seconds",
            )
        ),

        "median_oracle_setup_seconds": (
            as_float(
                raw[
                    "median_oracle_setup_seconds"
                ],
                field="median_oracle_setup_seconds",
            )
        ),

        "median_update_seconds": (
            as_float(
                raw[
                    "median_update_seconds"
                ],
                field="median_update_seconds",
            )
        ),

        "median_total_method_seconds": (
            as_float(
                raw[
                    "median_total_method_seconds"
                ],
                field="median_total_method_seconds",
            )
        ),

        "median_evaluation_seconds_excluded": (
            as_float(
                raw[
                    "median_evaluation_seconds_excluded"
                ],
                field=(
                    "median_evaluation_seconds_excluded"
                ),
            )
        ),

        "median_seconds_to_primary": (
            as_float(
                raw[
                    "median_seconds_to_primary"
                ],
                field="median_seconds_to_primary",
                nullable=True,
            )
        ),

        "median_seconds_to_secondary": (
            as_float(
                raw[
                    "median_seconds_to_secondary"
                ],
                field="median_seconds_to_secondary",
                nullable=True,
            )
        ),

        "proxy_common_setup": (
            as_float(
                raw[
                    "proxy_common_setup"
                ],
                field="proxy_common_setup",
            )
        ),

        "proxy_method_setup": (
            as_float(
                raw[
                    "proxy_method_setup"
                ],
                field="proxy_method_setup",
            )
        ),

        "proxy_oracle_setup": (
            as_float(
                raw[
                    "proxy_oracle_setup"
                ],
                field="proxy_oracle_setup",
            )
        ),

        "proxy_per_update": (
            as_float(
                raw[
                    "proxy_per_update"
                ],
                field="proxy_per_update",
            )
        ),

        "proxy_core_to_primary": (
            as_float(
                raw[
                    "proxy_core_to_primary"
                ],
                field="proxy_core_to_primary",
                nullable=True,
            )
        ),

        "proxy_costed_to_primary": (
            as_float(
                raw[
                    "proxy_costed_to_primary"
                ],
                field="proxy_costed_to_primary",
                nullable=True,
            )
        ),

        "proxy_core_final": (
            as_float(
                raw[
                    "proxy_core_final"
                ],
                field="proxy_core_final",
            )
        ),

        "proxy_costed_final": (
            as_float(
                raw[
                    "proxy_costed_final"
                ],
                field="proxy_costed_final",
            )
        ),

        "evaluation_proxy_work_excluded": (
            as_float(
                raw[
                    "evaluation_proxy_work_excluded"
                ],
                field=(
                    "evaluation_proxy_work_excluded"
                ),
            )
        ),

        "median_active_set_size": (
            as_float(
                raw[
                    "median_active_set_size"
                ],
                field="median_active_set_size",
                nullable=True,
            )
        ),

        "median_active_set_fraction": (
            as_float(
                raw[
                    "median_active_set_fraction"
                ],
                field="median_active_set_fraction",
                nullable=True,
            )
        ),

        "full_active_fraction": (
            as_float(
                raw[
                    "full_active_fraction"
                ],
                field="full_active_fraction",
                nullable=True,
            )
        ),

        "pruned_fraction": (
            as_float(
                raw[
                    "pruned_fraction"
                ],
                field="pruned_fraction",
                nullable=True,
            )
        ),

        "half_or_less_active_fraction": (
            as_float(
                raw[
                    "half_or_less_active_fraction"
                ],
                field="half_or_less_active_fraction",
                nullable=True,
            )
        ),

        "median_zero_lower_fraction": (
            as_float(
                raw[
                    "median_zero_lower_fraction"
                ],
                field="median_zero_lower_fraction",
                nullable=True,
            )
        ),

        "reference_test_rmse_standardized": (
            as_float(
                raw[
                    "reference_test_rmse_standardized"
                ],
                field=(
                    "reference_test_rmse_standardized"
                ),
                nullable=True,
            )
        ),

        "final_test_rmse_standardized": (
            as_float(
                raw[
                    "final_test_rmse_standardized"
                ],
                field=(
                    "final_test_rmse_standardized"
                ),
                nullable=True,
            )
        ),

        "test_rmse_difference_standardized": (
            as_float(
                raw[
                    "test_rmse_difference_standardized"
                ],
                field=(
                    "test_rmse_difference_standardized"
                ),
                nullable=True,
            )
        ),

        "reference_test_rmse_original": (
            as_float(
                raw[
                    "reference_test_rmse_original"
                ],
                field=(
                    "reference_test_rmse_original"
                ),
                nullable=True,
            )
        ),

        "final_test_rmse_original": (
            as_float(
                raw[
                    "final_test_rmse_original"
                ],
                field=(
                    "final_test_rmse_original"
                ),
                nullable=True,
            )
        ),

        "test_rmse_difference_original": (
            as_float(
                raw[
                    "test_rmse_difference_original"
                ],
                field=(
                    "test_rmse_difference_original"
                ),
                nullable=True,
            )
        ),

        "final_weights_sha256": (
            raw[
                "final_weights_sha256"
            ]
        ),
    }

    return row


# =============================================================================
# Pair/unit identifiers
# =============================================================================

def pair_key(
    row: dict[str, Any],
) -> tuple[
    str,
    int | None,
    int,
]:
    return (
        row["case"],
        row["problem_seed"],
        row["algorithm_seed"],
    )


def pair_id(
    row: dict[str, Any],
) -> str:
    if (
        row["kind"]
        == "synthetic"
    ):
        return (
            f"problem_seed:"
            f"{row['problem_seed']}"
        )

    return (
        f"algorithm_seed:"
        f"{row['algorithm_seed']}"
    )


# =============================================================================
# Structural audit
# =============================================================================

def audit_inputs() -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    dict[str, Any],
    dict[str, Any],
]:
    print(
        "Auditing Stage 5 benchmark outputs..."
    )

    run_header, raw_runs = (
        read_csv(
            RUN_RESULTS_PATH
        )
    )

    trajectory_header, trajectories = (
        read_csv(
            TRAJECTORIES_PATH
        )
    )

    require_columns(
        observed=run_header,
        required=REQUIRED_RUN_COLUMNS,
        source_name=(
            "run_results.csv"
        ),
    )

    require_columns(
        observed=trajectory_header,
        required=(
            REQUIRED_TRAJECTORY_COLUMNS
        ),
        source_name=(
            "trajectories.csv"
        ),
    )

    if len(
        raw_runs
    ) != EXPECTED_RUN_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_RUN_ROWS} run rows; "
            f"found {len(raw_runs)}."
        )

    if len(
        trajectories
    ) != EXPECTED_TRAJECTORY_ROWS:
        raise AssertionError(
            "Expected "
            f"{EXPECTED_TRAJECTORY_ROWS} trajectory rows; "
            f"found {len(trajectories)}."
        )

    environment = (
        read_json(
            ENVIRONMENT_PATH
        )
    )

    manifest = (
        read_json(
            MANIFEST_PATH
        )
    )

    if environment.get(
        "completed_run_rows"
    ) != EXPECTED_RUN_ROWS:
        raise AssertionError(
            "environment.json run count mismatch."
        )

    if environment.get(
        "completed_trajectory_rows"
    ) != EXPECTED_TRAJECTORY_ROWS:
        raise AssertionError(
            "environment.json trajectory count mismatch."
        )

    runs = [
        parse_run_row(
            raw
        )
        for raw
        in raw_runs
    ]

    observed_methods = {
        row["method"]
        for row in runs
    }

    if (
        observed_methods
        != set(
            METHODS
        )
    ):
        raise AssertionError(
            "Method set mismatch: "
            f"{sorted(observed_methods)}"
        )

    observed_cases = {
        row["case"]
        for row in runs
    }

    if (
        observed_cases
        != set(
            CASE_ORDER
        )
    ):
        raise AssertionError(
            "Case set mismatch: "
            f"{sorted(observed_cases)}"
        )

    # -------------------------------------------------------------------------
    # Every case/method combination must contain exactly 12 units.
    # -------------------------------------------------------------------------

    counts: dict[
        tuple[str, str],
        int,
    ] = defaultdict(int)

    for row in runs:
        counts[
            (
                row["case"],
                row["method"],
            )
        ] += 1

    for case in CASE_ORDER:
        for method in METHODS:
            observed = counts[
                (
                    case,
                    method,
                )
            ]

            if (
                observed
                != EXPECTED_REPLICATES_PER_CASE_METHOD
            ):
                raise AssertionError(
                    f"{case}/{method}: expected "
                    f"{EXPECTED_REPLICATES_PER_CASE_METHOD} rows; "
                    f"found {observed}."
                )

    # -------------------------------------------------------------------------
    # Target definitions must be frozen.
    # -------------------------------------------------------------------------

    for row in runs:
        if not np.isclose(
            row[
                "primary_target"
            ],
            PRIMARY_TARGET,
            rtol=0.0,
            atol=1e-15,
        ):
            raise AssertionError(
                "Primary target mismatch."
            )

        if not np.isclose(
            row[
                "secondary_target"
            ],
            SECONDARY_TARGET,
            rtol=0.0,
            atol=1e-15,
        ):
            raise AssertionError(
                "Secondary target mismatch."
            )

        if (
            row["success_primary"]
            != (
                row[
                    "epochs_to_primary"
                ]
                is not None
            )
        ):
            raise AssertionError(
                f"{row['problem_id']}/{row['method']}: "
                "primary-success flag inconsistent "
                "with epochs_to_primary."
            )

        if (
            row["success_primary"]
            != (
                row[
                    "median_seconds_to_primary"
                ]
                is not None
            )
        ):
            raise AssertionError(
                f"{row['problem_id']}/{row['method']}: "
                "primary-success flag inconsistent "
                "with time-to-primary."
            )

        if (
            row["success_secondary"]
            != (
                row[
                    "epochs_to_secondary"
                ]
                is not None
            )
        ):
            raise AssertionError(
                f"{row['problem_id']}/{row['method']}: "
                "secondary-success flag inconsistent."
            )

        if (
            row["success_primary"]
            and not row[
                "success_secondary"
            ]
        ):
            raise AssertionError(
                "Primary success without secondary success."
            )

    # -------------------------------------------------------------------------
    # ASCD diagnostics should exist only for ASCD methods.
    # -------------------------------------------------------------------------

    for row in runs:
        diagnostics = (
            row[
                "median_active_set_fraction"
            ],
            row[
                "full_active_fraction"
            ],
            row[
                "median_zero_lower_fraction"
            ],
        )

        if row["method"] in {
            G3_METHOD,
            G1_METHOD,
        }:
            if any(
                value is None
                for value in diagnostics
            ):
                raise AssertionError(
                    f"{row['case']}/{row['method']}: "
                    "missing ASCD diagnostic."
                )

        else:
            if any(
                value is not None
                for value in diagnostics
            ):
                raise AssertionError(
                    f"{row['case']}/{row['method']}: "
                    "unexpected ASCD diagnostic."
                )

    # -------------------------------------------------------------------------
    # Trajectory validation.
    # -------------------------------------------------------------------------

    run_lookup = {
        (
            row["problem_id"],
            row["algorithm_seed"],
            row["method"],
        ): row
        for row in runs
    }

    if len(
        run_lookup
    ) != EXPECTED_RUN_ROWS:
        raise AssertionError(
            "Run identity is not unique."
        )

    trajectory_groups: dict[
        tuple[str, int, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for raw in trajectories:
        key = (
            raw["problem_id"],
            int(
                raw[
                    "algorithm_seed"
                ]
            ),
            raw["method"],
        )

        trajectory_groups[
            key
        ].append(
            raw
        )

    if len(
        trajectory_groups
    ) != EXPECTED_RUN_ROWS:
        raise AssertionError(
            "Trajectory run-group count mismatch."
        )

    for key, group in (
        trajectory_groups.items()
    ):
        if (
            len(group)
            != EXPECTED_CHECKPOINTS_PER_RUN
        ):
            raise AssertionError(
                f"{key}: expected "
                f"{EXPECTED_CHECKPOINTS_PER_RUN} checkpoints; "
                f"found {len(group)}."
            )

        run = run_lookup.get(
            key
        )

        if run is None:
            raise AssertionError(
                f"Trajectory has no corresponding run: {key}"
            )

        steps = [
            int(
                item["steps"]
            )
            for item
            in group
        ]

        expected_steps = list(
            range(
                0,
                run["max_steps"] + 1,
                run[
                    "checkpoint_steps"
                ],
            )
        )

        if steps != expected_steps:
            raise AssertionError(
                f"{key}: checkpoint steps do not match protocol."
            )

        relative_gaps = np.asarray(
            [
                float(
                    item[
                        "relative_gap"
                    ]
                )
                for item
                in group
            ],
            dtype=np.float64,
        )

        if np.any(
            ~np.isfinite(
                relative_gaps
            )
        ):
            raise AssertionError(
                f"{key}: non-finite trajectory gap."
            )

        if np.any(
            relative_gaps < 0.0
        ):
            raise AssertionError(
                f"{key}: negative normalized gap."
            )

        if not np.isclose(
            relative_gaps[0],
            1.0,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise AssertionError(
                f"{key}: initial normalized gap is not 1."
            )

        # Coordinate minimization should not increase the objective.
        differences = np.diff(
            relative_gaps
        )

        if np.any(
            differences > 1e-9
        ):
            raise AssertionError(
                f"{key}: normalized objective gap increased "
                "beyond numerical tolerance."
            )

        if not np.isclose(
            relative_gaps[-1],
            run[
                "final_relative_gap"
            ],
            rtol=1e-9,
            atol=1e-12,
        ):
            raise AssertionError(
                f"{key}: final trajectory gap does not "
                "match run_results.csv."
            )

    print(
        "Stage 5 benchmark-output audit: PASSED"
    )

    print(
        f"  run rows        : {len(runs)}"
    )

    print(
        f"  trajectory rows : {len(trajectories)}"
    )

    print(
        f"  run groups       : {len(trajectory_groups)}"
    )

    return (
        runs,
        trajectories,
        environment,
        manifest,
    )


# =============================================================================
# Case/method summaries
# =============================================================================

def success_unit_ids(
    rows: list[dict[str, Any]],
) -> list[str]:
    return sorted(
        pair_id(
            row
        )
        for row in rows
        if row[
            "success_primary"
        ]
    )


def summarize_method(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(
        rows
    ) != EXPECTED_REPLICATES_PER_CASE_METHOD:
        raise AssertionError(
            "Unexpected replicate count in method summary."
        )

    success_primary = [
        row
        for row in rows
        if row[
            "success_primary"
        ]
    ]

    success_secondary = [
        row
        for row in rows
        if row[
            "success_secondary"
        ]
    ]

    result: dict[str, Any] = {
        "runs": len(
            rows
        ),

        "success_primary_count": (
            len(
                success_primary
            )
        ),

        "success_secondary_count": (
            len(
                success_secondary
            )
        ),

        "primary_success_units": (
            success_unit_ids(
                rows
            )
        ),

        "median_epochs_to_primary_among_successes": (
            median_or_none(
                row[
                    "epochs_to_primary"
                ]
                for row
                in success_primary
            )
        ),

        "median_seconds_to_primary_among_successes": (
            median_or_none(
                row[
                    "median_seconds_to_primary"
                ]
                for row
                in success_primary
            )
        ),

        "median_proxy_core_to_primary_among_successes": (
            median_or_none(
                row[
                    "proxy_core_to_primary"
                ]
                for row
                in success_primary
            )
        ),

        "median_proxy_costed_to_primary_among_successes": (
            median_or_none(
                row[
                    "proxy_costed_to_primary"
                ]
                for row
                in success_primary
            )
        ),

        "median_final_relative_gap": (
            median_or_none(
                row[
                    "final_relative_gap"
                ]
                for row
                in rows
            )
        ),

        "median_total_method_seconds": (
            median_or_none(
                row[
                    "median_total_method_seconds"
                ]
                for row
                in rows
            )
        ),

        "median_setup_seconds": (
            median_or_none(
                row[
                    "median_setup_seconds"
                ]
                for row
                in rows
            )
        ),

        "median_oracle_setup_seconds": (
            median_or_none(
                row[
                    "median_oracle_setup_seconds"
                ]
                for row
                in rows
            )
        ),

        "median_update_seconds": (
            median_or_none(
                row[
                    "median_update_seconds"
                ]
                for row
                in rows
            )
        ),
    }

    method = rows[0][
        "method"
    ]

    if method in {
        G3_METHOD,
        G1_METHOD,
    }:
        result.update(
            {
                "median_active_set_size": (
                    median_or_none(
                        row[
                            "median_active_set_size"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_active_set_fraction": (
                    median_or_none(
                        row[
                            "median_active_set_fraction"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_full_active_fraction": (
                    median_or_none(
                        row[
                            "full_active_fraction"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_pruned_fraction": (
                    median_or_none(
                        row[
                            "pruned_fraction"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_half_or_less_active_fraction": (
                    median_or_none(
                        row[
                            "half_or_less_active_fraction"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_zero_lower_fraction": (
                    median_or_none(
                        row[
                            "median_zero_lower_fraction"
                        ]
                        for row
                        in rows
                    )
                ),
            }
        )

    if rows[0][
        "kind"
    ] == "real":
        reference_std_values = [
            row[
                "reference_test_rmse_standardized"
            ]
            for row in rows
        ]

        reference_original_values = [
            row[
                "reference_test_rmse_original"
            ]
            for row in rows
        ]

        result.update(
            {
                "reference_test_rmse_standardized": (
                    median_or_none(
                        reference_std_values
                    )
                ),

                "median_final_test_rmse_standardized": (
                    median_or_none(
                        row[
                            "final_test_rmse_standardized"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_test_rmse_difference_standardized": (
                    median_or_none(
                        row[
                            "test_rmse_difference_standardized"
                        ]
                        for row
                        in rows
                    )
                ),

                "reference_test_rmse_original": (
                    median_or_none(
                        reference_original_values
                    )
                ),

                "median_final_test_rmse_original": (
                    median_or_none(
                        row[
                            "final_test_rmse_original"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_test_rmse_difference_original": (
                    median_or_none(
                        row[
                            "test_rmse_difference_original"
                        ]
                        for row
                        in rows
                    )
                ),
            }
        )

    return result


def build_case_summaries(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in runs:
        grouped[
            (
                row["case"],
                row["method"],
            )
        ].append(
            row
        )

    summaries: dict[
        str,
        Any,
    ] = {}

    for case in CASE_ORDER:
        case_rows = [
            row
            for row in runs
            if row[
                "case"
            ] == case
        ]

        kinds = {
            row["kind"]
            for row
            in case_rows
        }

        dimensions = {
            (
                row["n"],
                row["d"],
            )
            for row
            in case_rows
        }

        if len(
            kinds
        ) != 1:
            raise AssertionError(
                f"{case}: multiple kind values."
            )

        if len(
            dimensions
        ) != 1:
            raise AssertionError(
                f"{case}: dimensions differ across runs."
            )

        n, d = next(
            iter(
                dimensions
            )
        )

        summaries[
            case
        ] = {
            "kind": next(
                iter(
                    kinds
                )
            ),

            "n": n,
            "d": d,

            "methods": {
                method: summarize_method(
                    grouped[
                        (
                            case,
                            method,
                        )
                    ]
                )
                for method
                in METHODS
            },
        }

    return summaries


# =============================================================================
# Paired g1 versus g3 analysis
# =============================================================================

def safe_ratio(
    numerator: float,
    denominator: float,
    *,
    label: str,
) -> float:
    if denominator <= 0.0:
        raise ValueError(
            f"{label}: denominator must be positive."
        )

    return (
        numerator
        / denominator
    )


def pair_g1_g3(
    runs: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    g1_lookup = {
        pair_key(
            row
        ): row
        for row in runs
        if row[
            "method"
        ] == G1_METHOD
    }

    g3_lookup = {
        pair_key(
            row
        ): row
        for row in runs
        if row[
            "method"
        ] == G3_METHOD
    }

    if set(
        g1_lookup
    ) != set(
        g3_lookup
    ):
        raise AssertionError(
            "g1/g3 pairing units differ."
        )

    paired_rows: list[
        dict[str, Any]
    ] = []

    case_pair_rows: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for key in sorted(
        g1_lookup,
        key=lambda item: (
            CASE_ORDER.index(
                item[0]
            ),
            -1
            if item[1] is None
            else item[1],
            item[2],
        ),
    ):
        g1 = g1_lookup[
            key
        ]

        g3 = g3_lookup[
            key
        ]

        if (
            g1["case"]
            != g3["case"]
        ):
            raise AssertionError(
                "g1/g3 case mismatch."
            )

        shared_success = (
            g1[
                "success_primary"
            ]
            and g3[
                "success_primary"
            ]
        )

        epochs_diff = None
        epochs_ratio = None

        runtime_diff = None
        runtime_ratio = None

        proxy_core_diff = None
        proxy_core_ratio = None

        proxy_costed_diff = None
        proxy_costed_ratio = None

        if shared_success:
            assert (
                g1[
                    "epochs_to_primary"
                ]
                is not None
            )

            assert (
                g3[
                    "epochs_to_primary"
                ]
                is not None
            )

            assert (
                g1[
                    "median_seconds_to_primary"
                ]
                is not None
            )

            assert (
                g3[
                    "median_seconds_to_primary"
                ]
                is not None
            )

            assert (
                g1[
                    "proxy_core_to_primary"
                ]
                is not None
            )

            assert (
                g3[
                    "proxy_core_to_primary"
                ]
                is not None
            )

            assert (
                g1[
                    "proxy_costed_to_primary"
                ]
                is not None
            )

            assert (
                g3[
                    "proxy_costed_to_primary"
                ]
                is not None
            )

            epochs_diff = (
                g1[
                    "epochs_to_primary"
                ]
                - g3[
                    "epochs_to_primary"
                ]
            )

            epochs_ratio = safe_ratio(
                g1[
                    "epochs_to_primary"
                ],
                g3[
                    "epochs_to_primary"
                ],
                label=(
                    f"{key} epoch ratio"
                ),
            )

            runtime_diff = (
                g1[
                    "median_seconds_to_primary"
                ]
                - g3[
                    "median_seconds_to_primary"
                ]
            )

            runtime_ratio = safe_ratio(
                g1[
                    "median_seconds_to_primary"
                ],
                g3[
                    "median_seconds_to_primary"
                ],
                label=(
                    f"{key} runtime ratio"
                ),
            )

            proxy_core_diff = (
                g1[
                    "proxy_core_to_primary"
                ]
                - g3[
                    "proxy_core_to_primary"
                ]
            )

            proxy_core_ratio = safe_ratio(
                g1[
                    "proxy_core_to_primary"
                ],
                g3[
                    "proxy_core_to_primary"
                ],
                label=(
                    f"{key} core proxy ratio"
                ),
            )

            proxy_costed_diff = (
                g1[
                    "proxy_costed_to_primary"
                ]
                - g3[
                    "proxy_costed_to_primary"
                ]
            )

            proxy_costed_ratio = safe_ratio(
                g1[
                    "proxy_costed_to_primary"
                ],
                g3[
                    "proxy_costed_to_primary"
                ],
                label=(
                    f"{key} costed proxy ratio"
                ),
            )

        paired = {
            "kind": (
                g1["kind"]
            ),

            "case": (
                g1["case"]
            ),

            "pair_id": (
                pair_id(
                    g1
                )
            ),

            "problem_seed": (
                g1[
                    "problem_seed"
                ]
            ),

            "algorithm_seed": (
                g1[
                    "algorithm_seed"
                ]
            ),

            "g3_success_primary": int(
                g3[
                    "success_primary"
                ]
            ),

            "g1_success_primary": int(
                g1[
                    "success_primary"
                ]
            ),

            "shared_primary_success": int(
                shared_success
            ),

            "g3_epochs_to_primary": (
                g3[
                    "epochs_to_primary"
                ]
            ),

            "g1_epochs_to_primary": (
                g1[
                    "epochs_to_primary"
                ]
            ),

            "epochs_difference_g1_minus_g3": (
                epochs_diff
            ),

            "epochs_ratio_g1_over_g3": (
                epochs_ratio
            ),

            "g3_seconds_to_primary": (
                g3[
                    "median_seconds_to_primary"
                ]
            ),

            "g1_seconds_to_primary": (
                g1[
                    "median_seconds_to_primary"
                ]
            ),

            "runtime_difference_g1_minus_g3": (
                runtime_diff
            ),

            "runtime_ratio_g1_over_g3": (
                runtime_ratio
            ),

            "g3_proxy_core_to_primary": (
                g3[
                    "proxy_core_to_primary"
                ]
            ),

            "g1_proxy_core_to_primary": (
                g1[
                    "proxy_core_to_primary"
                ]
            ),

            "proxy_core_difference_g1_minus_g3": (
                proxy_core_diff
            ),

            "proxy_core_ratio_g1_over_g3": (
                proxy_core_ratio
            ),

            "g3_proxy_costed_to_primary": (
                g3[
                    "proxy_costed_to_primary"
                ]
            ),

            "g1_proxy_costed_to_primary": (
                g1[
                    "proxy_costed_to_primary"
                ]
            ),

            "proxy_costed_difference_g1_minus_g3": (
                proxy_costed_diff
            ),

            "proxy_costed_ratio_g1_over_g3": (
                proxy_costed_ratio
            ),

            "g3_median_active_set_fraction": (
                g3[
                    "median_active_set_fraction"
                ]
            ),

            "g1_median_active_set_fraction": (
                g1[
                    "median_active_set_fraction"
                ]
            ),

            "g3_median_zero_lower_fraction": (
                g3[
                    "median_zero_lower_fraction"
                ]
            ),

            "g1_median_zero_lower_fraction": (
                g1[
                    "median_zero_lower_fraction"
                ]
            ),

            "g3_final_relative_gap": (
                g3[
                    "final_relative_gap"
                ]
            ),

            "g1_final_relative_gap": (
                g1[
                    "final_relative_gap"
                ]
            ),
        }

        paired_rows.append(
            paired
        )

        case_pair_rows[
            g1["case"]
        ].append(
            paired
        )

    if len(
        paired_rows
    ) != (
        len(
            CASE_ORDER
        )
        * EXPECTED_REPLICATES_PER_CASE_METHOD
    ):
        raise AssertionError(
            "Unexpected g1/g3 paired row count."
        )

    case_summaries: dict[
        str,
        Any,
    ] = {}

    for case in CASE_ORDER:
        rows = case_pair_rows[
            case
        ]

        if len(
            rows
        ) != EXPECTED_REPLICATES_PER_CASE_METHOD:
            raise AssertionError(
                f"{case}: paired replicate count mismatch."
            )

        g1_success = {
            row[
                "pair_id"
            ]
            for row in rows
            if row[
                "g1_success_primary"
            ]
        }

        g3_success = {
            row[
                "pair_id"
            ]
            for row in rows
            if row[
                "g3_success_primary"
            ]
        }

        shared = (
            g1_success
            & g3_success
        )

        g1_only = (
            g1_success
            - g3_success
        )

        g3_only = (
            g3_success
            - g1_success
        )

        all_units = {
            row[
                "pair_id"
            ]
            for row in rows
        }

        neither = (
            all_units
            - g1_success
            - g3_success
        )

        shared_rows = [
            row
            for row in rows
            if row[
                "shared_primary_success"
            ]
        ]

        epochs_differences = [
            float(
                row[
                    "epochs_difference_g1_minus_g3"
                ]
            )
            for row
            in shared_rows
        ]

        epochs_ratios = [
            float(
                row[
                    "epochs_ratio_g1_over_g3"
                ]
            )
            for row
            in shared_rows
        ]

        runtime_differences = [
            float(
                row[
                    "runtime_difference_g1_minus_g3"
                ]
            )
            for row
            in shared_rows
        ]

        runtime_ratios = [
            float(
                row[
                    "runtime_ratio_g1_over_g3"
                ]
            )
            for row
            in shared_rows
        ]

        core_differences = [
            float(
                row[
                    "proxy_core_difference_g1_minus_g3"
                ]
            )
            for row
            in shared_rows
        ]

        core_ratios = [
            float(
                row[
                    "proxy_core_ratio_g1_over_g3"
                ]
            )
            for row
            in shared_rows
        ]

        costed_differences = [
            float(
                row[
                    "proxy_costed_difference_g1_minus_g3"
                ]
            )
            for row
            in shared_rows
        ]

        costed_ratios = [
            float(
                row[
                    "proxy_costed_ratio_g1_over_g3"
                ]
            )
            for row
            in shared_rows
        ]

        case_summaries[
            case
        ] = {
            "runs": len(
                rows
            ),

            "g3_success_count": (
                len(
                    g3_success
                )
            ),

            "g1_success_count": (
                len(
                    g1_success
                )
            ),

            "shared_success_count": (
                len(
                    shared
                )
            ),

            "g1_only_success_units": (
                sorted(
                    g1_only
                )
            ),

            "g3_only_success_units": (
                sorted(
                    g3_only
                )
            ),

            "shared_success_units": (
                sorted(
                    shared
                )
            ),

            "neither_success_units": (
                sorted(
                    neither
                )
            ),

            "coordinate_epochs": {
                "paired_difference_g1_minus_g3": (
                    bootstrap_median_ci(
                        epochs_differences,
                        label=(
                            f"{case}:"
                            "epochs_difference"
                        ),
                    )
                ),

                "paired_ratio_g1_over_g3": (
                    bootstrap_median_ci(
                        epochs_ratios,
                        label=(
                            f"{case}:"
                            "epochs_ratio"
                        ),
                    )
                ),
            },

            "harmonized_runtime": {
                "paired_difference_g1_minus_g3": (
                    bootstrap_median_ci(
                        runtime_differences,
                        label=(
                            f"{case}:"
                            "runtime_difference"
                        ),
                    )
                ),

                "paired_ratio_g1_over_g3": (
                    bootstrap_median_ci(
                        runtime_ratios,
                        label=(
                            f"{case}:"
                            "runtime_ratio"
                        ),
                    )
                ),
            },

            "core_proxy": {
                "paired_difference_g1_minus_g3": (
                    bootstrap_median_ci(
                        core_differences,
                        label=(
                            f"{case}:"
                            "core_proxy_difference"
                        ),
                    )
                ),

                "paired_ratio_g1_over_g3": (
                    bootstrap_median_ci(
                        core_ratios,
                        label=(
                            f"{case}:"
                            "core_proxy_ratio"
                        ),
                    )
                ),
            },

            "costed_proxy": {
                "paired_difference_g1_minus_g3": (
                    bootstrap_median_ci(
                        costed_differences,
                        label=(
                            f"{case}:"
                            "costed_proxy_difference"
                        ),
                    )
                ),

                "paired_ratio_g1_over_g3": (
                    bootstrap_median_ci(
                        costed_ratios,
                        label=(
                            f"{case}:"
                            "costed_proxy_ratio"
                        ),
                    )
                ),
            },
        }

    return (
        paired_rows,
        case_summaries,
    )


# =============================================================================
# Confirmatory decision rules
# =============================================================================

def build_decision_rule_summary(
    case_summaries: dict[str, Any],
    paired_summaries: dict[str, Any],
) -> dict[str, Any]:
    active_set_cases: list[
        str
    ] = []

    zero_lower_cases: list[
        str
    ] = []

    g1_lower_epoch_cases: list[
        str
    ] = []

    runtime_below_one_cases: list[
        str
    ] = []

    runtime_above_one_cases: list[
        str
    ] = []

    exact_information_pays_cases: list[
        str
    ] = []

    for case in CASE_ORDER:
        methods = (
            case_summaries[
                case
            ][
                "methods"
            ]
        )

        g3 = methods[
            G3_METHOD
        ]

        g1 = methods[
            G1_METHOD
        ]

        g3_active = (
            g3[
                "median_active_set_fraction"
            ]
        )

        g1_active = (
            g1[
                "median_active_set_fraction"
            ]
        )

        if (
            g3_active is not None
            and g1_active is not None
            and g3_active
            > g1_active
        ):
            active_set_cases.append(
                case
            )

        g3_zero = (
            g3[
                "median_zero_lower_fraction"
            ]
        )

        g1_zero = (
            g1[
                "median_zero_lower_fraction"
            ]
        )

        if (
            g3_zero is not None
            and g1_zero is not None
            and g3_zero
            > g1_zero
        ):
            zero_lower_cases.append(
                case
            )

        paired = (
            paired_summaries[
                case
            ]
        )

        epochs_ratio_summary = (
            paired[
                "coordinate_epochs"
            ][
                "paired_ratio_g1_over_g3"
            ]
        )

        if (
            epochs_ratio_summary
            is not None
            and epochs_ratio_summary[
                "median"
            ] < 1.0
        ):
            g1_lower_epoch_cases.append(
                case
            )

        runtime_ratio_summary = (
            paired[
                "harmonized_runtime"
            ][
                "paired_ratio_g1_over_g3"
            ]
        )

        if (
            runtime_ratio_summary
            is not None
        ):
            runtime_ratio = (
                runtime_ratio_summary[
                    "median"
                ]
            )

            if runtime_ratio < 1.0:
                runtime_below_one_cases.append(
                    case
                )

            elif runtime_ratio > 1.0:
                runtime_above_one_cases.append(
                    case
                )

        g1_success = (
            paired[
                "g1_success_count"
            ]
        )

        g3_success = (
            paired[
                "g3_success_count"
            ]
        )

        if (
            runtime_ratio_summary
            is not None
            and g1_success
            >= g3_success
            and runtime_ratio_summary[
                "median"
            ] < 1.0
        ):
            exact_information_pays_cases.append(
                case
            )

    majority_threshold = (
        len(
            CASE_ORDER
        )
        // 2
        + 1
    )

    mechanism_supported = (
        len(
            active_set_cases
        )
        >= majority_threshold
        and len(
            zero_lower_cases
        )
        >= majority_threshold
    )

    exact_pruning_supported = (
        len(
            active_set_cases
        )
        >= majority_threshold
    )

    exact_faster_in_updates_somewhere = (
        len(
            g1_lower_epoch_cases
        )
        >= 1
    )

    sometimes_pays = (
        len(
            exact_information_pays_cases
        )
        >= 1
    )

    mixed_runtime_direction = (
        bool(
            runtime_below_one_cases
        )
        and bool(
            runtime_above_one_cases
        )
    )

    return {
        "problem_family_count": (
            len(
                CASE_ORDER
            )
        ),

        "majority_threshold": (
            majority_threshold
        ),

        "H1_zero_oracle_pruning_failure": {
            "criterion": (
                "g3 median active-set fraction exceeds "
                "g1 on a majority of held-out problem families"
            ),

            "cases": (
                active_set_cases
            ),

            "count": (
                len(
                    active_set_cases
                )
            ),

            "criterion_met": (
                exact_pruning_supported
            ),
        },

        "H2_exact_information_pruning_and_updates": {
            "smaller_active_set_cases": (
                active_set_cases
            ),

            "lower_paired_median_epochs_cases": (
                g1_lower_epoch_cases
            ),

            "smaller_active_set_majority": (
                exact_pruning_supported
            ),

            "fewer_coordinate_epochs_on_at_least_one_family": (
                exact_faster_in_updates_somewhere
            ),
        },

        "H3_information_cost_is_regime_dependent": {
            "paired_median_runtime_ratio_below_one_cases": (
                runtime_below_one_cases
            ),

            "paired_median_runtime_ratio_above_one_cases": (
                runtime_above_one_cases
            ),

            "mixed_runtime_direction_observed": (
                mixed_runtime_direction
            ),

            "note": (
                "This is a descriptive check. "
                "The predeclared hypothesis permits mixed "
                "or one-sided outcomes."
            ),
        },

        "H4_stage4_mechanism_generalizes": {
            "g3_larger_active_set_cases": (
                active_set_cases
            ),

            "g3_more_zero_lower_bound_cases": (
                zero_lower_cases
            ),

            "active_set_count": (
                len(
                    active_set_cases
                )
            ),

            "zero_lower_count": (
                len(
                    zero_lower_cases
                )
            ),

            "criterion_met": (
                mechanism_supported
            ),
        },

        "predeclared_exact_information_can_sometimes_pay_rule": {
            "criterion": (
                "at least one held-out problem family has "
                "g1 success count no worse than g3 and "
                "paired median harmonized runtime ratio g1/g3 < 1"
            ),

            "qualifying_cases": (
                exact_information_pays_cases
            ),

            "criterion_met": (
                sometimes_pays
            ),

            "paired_uncertainty_reported": (
                True
            ),

            "individual_pairs_reported_in": (
                "paired_results.csv"
            ),
        },
    }


# =============================================================================
# Timing summary
# =============================================================================

def build_timing_summary(
    runs: list[dict[str, Any]],
    paired_summaries: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in runs:
        grouped[
            (
                row["case"],
                row["method"],
            )
        ].append(
            row
        )

    by_case: dict[
        str,
        Any,
    ] = {}

    for case in CASE_ORDER:
        method_payload: dict[
            str,
            Any,
        ] = {}

        for method in METHODS:
            rows = grouped[
                (
                    case,
                    method,
                )
            ]

            success_rows = [
                row
                for row in rows
                if row[
                    "success_primary"
                ]
            ]

            method_payload[
                method
            ] = {
                "runs": len(
                    rows
                ),

                "primary_successes": (
                    len(
                        success_rows
                    )
                ),

                "median_setup_seconds": (
                    median_or_none(
                        row[
                            "median_setup_seconds"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_oracle_setup_seconds": (
                    median_or_none(
                        row[
                            "median_oracle_setup_seconds"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_update_seconds": (
                    median_or_none(
                        row[
                            "median_update_seconds"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_total_method_seconds": (
                    median_or_none(
                        row[
                            "median_total_method_seconds"
                        ]
                        for row
                        in rows
                    )
                ),

                "median_seconds_to_primary_among_successes": (
                    median_or_none(
                        row[
                            "median_seconds_to_primary"
                        ]
                        for row
                        in success_rows
                    )
                ),

                "minimum_total_method_seconds": (
                    minimum_or_none(
                        row[
                            "median_total_method_seconds"
                        ]
                        for row
                        in rows
                    )
                ),

                "maximum_total_method_seconds": (
                    maximum_or_none(
                        row[
                            "median_total_method_seconds"
                        ]
                        for row
                        in rows
                    )
                ),
            }

        by_case[
            case
        ] = {
            "methods": (
                method_payload
            ),

            "g1_vs_g3_paired_runtime": (
                paired_summaries[
                    case
                ][
                    "harmonized_runtime"
                ]
            ),
        }

    return {
        "stage": 5,

        "generated_at_utc": (
            utc_now_string()
        ),

        "timing_protocol": (
            environment.get(
                "timing_protocol"
            )
        ),

        "threadpool_policy": (
            environment.get(
                "threadpool_policy"
            )
        ),

        "timer": (
            environment.get(
                "timer"
            )
        ),

        "by_case": (
            by_case
        ),
    }


# =============================================================================
# Final summary
# =============================================================================

def build_summary(
    *,
    runs: list[dict[str, Any]],
    case_summaries: dict[str, Any],
    paired_summaries: dict[str, Any],
    decision_rules: dict[str, Any],
    environment: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": 5,

        "analysis": (
            "confirmatory robustness and generalization"
        ),

        "generated_at_utc": (
            utc_now_string()
        ),

        "analysis_protocol": {
            "primary_normalized_gap": (
                PRIMARY_TARGET
            ),

            "secondary_normalized_gap": (
                SECONDARY_TARGET
            ),

            "bootstrap_resamples": (
                BOOTSTRAP_RESAMPLES
            ),

            "bootstrap_base_seed": (
                BOOTSTRAP_BASE_SEED
            ),

            "bootstrap_statistic": (
                "median paired difference or "
                "median paired ratio"
            ),

            "bootstrap_interval": (
                "percentile 95%"
            ),

            "pairing": {
                "synthetic": (
                    "problem seed, with the "
                    "predeclared paired algorithm seed"
                ),

                "real": (
                    "algorithm seed on the fixed "
                    "train/test split"
                ),
            },

            "conditional_median_rule": (
                "time/epochs/proxy to target medians "
                "are reported only among successful runs "
                "and always alongside success counts"
            ),
        },

        "input_audit": {
            "run_rows": len(
                runs
            ),

            "expected_run_rows": (
                EXPECTED_RUN_ROWS
            ),

            "trajectory_rows": (
                EXPECTED_TRAJECTORY_ROWS
            ),

            "expected_checkpoints_per_run": (
                EXPECTED_CHECKPOINTS_PER_RUN
            ),

            "dataset_manifest_sha256": (
                sha256_file(
                    MANIFEST_PATH
                )
            ),

            "run_results_sha256": (
                sha256_file(
                    RUN_RESULTS_PATH
                )
            ),

            "trajectories_sha256": (
                sha256_file(
                    TRAJECTORIES_PATH
                )
            ),

            "environment_sha256": (
                sha256_file(
                    ENVIRONMENT_PATH
                )
            ),

            "benchmark_completed_run_rows": (
                environment.get(
                    "completed_run_rows"
                )
            ),

            "benchmark_completed_trajectory_rows": (
                environment.get(
                    "completed_trajectory_rows"
                )
            ),

            "dataset_preparation_generated_at_utc": (
                manifest.get(
                    "generated_at_utc"
                )
            ),
        },

        "case_order": list(
            CASE_ORDER
        ),

        "methods": list(
            METHODS
        ),

        "by_case": (
            case_summaries
        ),

        "g1_vs_g3_paired": (
            paired_summaries
        ),

        "confirmatory_decision_rules": (
            decision_rules
        ),

        "interpretation_constraints": {
            "runtime_is_hardware_and_implementation_dependent":
            True,

            "proxy_work_is_not_exact_flops":
            True,

            "test_rmse_is_secondary":
            True,

            "all_methods_optimize_same_training_objective":
            True,

            "conditional_success_medians_must_not_be_read_without_success_counts":
            True,

            "no_general_superiority_claim_from_subset_of_successes":
            True,
        },

        "next_planned_step": (
            "Stage 5 visualization and report integration"
        ),
    }


# =============================================================================
# Console summary
# =============================================================================

def print_compact_results(
    case_summaries: dict[str, Any],
    paired_summaries: dict[str, Any],
    decision_rules: dict[str, Any],
) -> None:
    print()
    print(
        "=" * 78
    )
    print(
        "STAGE 5 CONFIRMATORY ANALYSIS: COMPLETED"
    )
    print(
        "=" * 78
    )

    print()
    print(
        "Primary success counts "
        f"(relative gap <= {PRIMARY_TARGET:g}; out of 12)"
    )

    header = (
        f"{'Case':34s} "
        f"{'Uniform':>8s} "
        f"{'Lipschitz':>10s} "
        f"{'g3':>6s} "
        f"{'g1':>6s}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for case in CASE_ORDER:
        methods = (
            case_summaries[
                case
            ][
                "methods"
            ]
        )

        values = [
            methods[
                method
            ][
                "success_primary_count"
            ]
            for method
            in METHODS
        ]

        print(
            f"{case:34s} "
            f"{values[0]:>8d} "
            f"{values[1]:>10d} "
            f"{values[2]:>6d} "
            f"{values[3]:>6d}"
        )

    print()
    print(
        "g1 versus g3 paired primary-target summaries"
    )

    for case in CASE_ORDER:
        paired = (
            paired_summaries[
                case
            ]
        )

        runtime = (
            paired[
                "harmonized_runtime"
            ][
                "paired_ratio_g1_over_g3"
            ]
        )

        epochs = (
            paired[
                "coordinate_epochs"
            ][
                "paired_ratio_g1_over_g3"
            ]
        )

        runtime_text = (
            "NA"
            if runtime is None
            else (
                f"{runtime['median']:.4f} "
                f"[{runtime['ci95_lower']:.4f}, "
                f"{runtime['ci95_upper']:.4f}]"
            )
        )

        epochs_text = (
            "NA"
            if epochs is None
            else (
                f"{epochs['median']:.4f}"
            )
        )

        print(
            f"  {case}: "
            f"g3={paired['g3_success_count']}/12, "
            f"g1={paired['g1_success_count']}/12, "
            f"shared={paired['shared_success_count']}, "
            f"epoch-ratio={epochs_text}, "
            f"runtime-ratio={runtime_text}"
        )

    mechanism = (
        decision_rules[
            "H4_stage4_mechanism_generalizes"
        ]
    )

    pays = (
        decision_rules[
            "predeclared_exact_information_can_sometimes_pay_rule"
        ]
    )

    print()
    print(
        "Predeclared decision-rule summary:"
    )

    print(
        "  Stage 4 pruning mechanism "
        f"criterion met: "
        f"{mechanism['criterion_met']}"
    )

    print(
        "  Exact information can sometimes "
        f"pay criterion met: "
        f"{pays['criterion_met']}"
    )

    print(
        "  Qualifying runtime cases: "
        f"{pays['qualifying_cases']}"
    )

    print()
    print(
        "Canonical analysis outputs:"
    )

    print(
        f"  {PAIRED_RESULTS_PATH}"
    )

    print(
        f"  {SUMMARY_PATH}"
    )

    print(
        f"  {TIMING_SUMMARY_PATH}"
    )

    print()
    print(
        "Next step: generate Stage 5 figures from these "
        "complete-set summaries; do not alter the "
        "predeclared experiment in response to individual cases."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the complete Stage 5 "
            "confirmatory benchmark."
        )
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Validate benchmark outputs and trajectory "
            "structure without producing analysis outputs."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of existing Stage 5 "
            "analysis outputs."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    (
        runs,
        _trajectories,
        environment,
        manifest,
    ) = audit_inputs()

    if args.audit_only:
        print()
        print(
            "AUDIT-ONLY MODE COMPLETE."
        )

        print(
            "No Stage 5 analysis outputs were generated."
        )

        return

    canonical_outputs = (
        PAIRED_RESULTS_PATH,
        SUMMARY_PATH,
        TIMING_SUMMARY_PATH,
    )

    existing = [
        path
        for path in canonical_outputs
        if path.exists()
    ]

    if (
        existing
        and not args.overwrite
    ):
        formatted = "\n".join(
            f"  {path}"
            for path
            in existing
        )

        raise RuntimeError(
            "Stage 5 canonical analysis output already exists.\n\n"
            f"{formatted}\n\n"
            "The script will not silently overwrite it.\n"
            "For a documented technical rerun, use --overwrite."
        )

    case_summaries = (
        build_case_summaries(
            runs
        )
    )

    (
        paired_rows,
        paired_summaries,
    ) = pair_g1_g3(
        runs
    )

    decision_rules = (
        build_decision_rule_summary(
            case_summaries,
            paired_summaries,
        )
    )

    timing_summary = (
        build_timing_summary(
            runs,
            paired_summaries,
            environment,
        )
    )

    summary = (
        build_summary(
            runs=runs,
            case_summaries=(
                case_summaries
            ),
            paired_summaries=(
                paired_summaries
            ),
            decision_rules=(
                decision_rules
            ),
            environment=(
                environment
            ),
            manifest=(
                manifest
            ),
        )
    )

    # Write only after all analysis and audits succeed.
    write_csv_atomic(
        PAIRED_RESULTS_PATH,
        paired_rows,
    )

    write_json_atomic(
        SUMMARY_PATH,
        summary,
    )

    write_json_atomic(
        TIMING_SUMMARY_PATH,
        timing_summary,
    )

    # Independent parse check.
    read_json(
        SUMMARY_PATH
    )

    read_json(
        TIMING_SUMMARY_PATH
    )

    print_compact_results(
        case_summaries,
        paired_summaries,
        decision_rules,
    )


if __name__ == "__main__":
    main()
