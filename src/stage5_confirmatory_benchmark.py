"""Stage 5 confirmatory ridge-regression benchmark.

This script implements the predeclared Stage 5 confirmatory comparison:

    1. uniform coordinate sampling
    2. fixed Lipschitz sampling
    3. ASCD g3 zero oracle
    4. ASCD g1 exact inner-product oracle

Primary goals
-------------
- held-out synthetic validation;
- additional real-data validation;
- harmonized method timing;
- normalized objective-gap success criteria;
- ASCD active-set diagnostics;
- exact-vs-zero oracle comparison;
- secondary proxy-work accounting.

IMPORTANT
---------
The script does not modify the prepared datasets.

It verifies the Stage 5 dataset manifest and prepared-file hashes before
running the benchmark.

Timing protocol
---------------
For each method/problem pair:

    - one untimed warm-up execution;
    - three timed repetitions;
    - method-specific setup is included;
    - coordinate-selection/update time is included;
    - checkpoint objective evaluation is timed separately and excluded
      from primary method runtime.

The warm-up run supplies:
    - objective trajectories;
    - ASCD active-set diagnostics;
    - residual/invariant audits;
    - final prediction diagnostics.

The three timed repetitions are used for wall-clock summaries.

Reference-optimum computation is outside method timing.

Run from repository root:

    python src/stage5_confirmatory_benchmark.py --audit-only

then, for the canonical experiment:

    python src/stage5_confirmatory_benchmark.py

Outputs
-------
results/stage5_confirmatory/
    run_results.csv
    trajectories.csv
    environment.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from threadpoolctl import (
        threadpool_info,
        threadpool_limits,
    )
except ImportError as exc:
    raise RuntimeError(
        "threadpoolctl is required for the Stage 5 timing protocol.\n"
        "It is normally installed with scikit-learn.\n\n"
        "Install with:\n"
        "    python -m pip install threadpoolctl"
    ) from exc


# =============================================================================
# Repository paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    ROOT
    / "results"
    / "stage5_confirmatory"
)

PREPARED_ROOT = (
    RESULT_ROOT
    / "prepared_data"
)

MANIFEST_PATH = (
    RESULT_ROOT
    / "dataset_manifest.json"
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


# =============================================================================
# Frozen Stage 5 benchmark constants
# =============================================================================

LAM = 0.03

PRIMARY_TARGET = 1e-6
SECONDARY_TARGET = 1e-4

MAX_EPOCHS = 200
CHECKPOINT_EPOCHS = 5

TIMED_REPEATS = 3

METHODS = (
    "uniform",
    "lipschitz",
    "ascd_g3_zero",
    "ascd_g1_exact",
)

REAL_DATASET_ORDER = (
    "california_housing",
    "concrete_compressive_strength",
    "airfoil_self_noise",
    "residential_building",
)

SYNTHETIC_CASE_ORDER = (
    "heldout_balanced_moderate",
    "heldout_balanced_strong",
    "heldout_scaled_moderate",
    "heldout_scaled_strong",
)

SYNTHETIC_PROBLEM_SEEDS = tuple(
    range(100, 112)
)

ALGORITHM_SEEDS = tuple(
    range(9100, 9112)
)

INTERVAL_ATOL = 1e-8
INTERVAL_RTOL = 1e-10

RESIDUAL_ATOL = 1e-9
RESIDUAL_RTOL = 1e-10

REPEAT_ATOL = 1e-11
REPEAT_RTOL = 1e-10


# =============================================================================
# Problem container
# =============================================================================

@dataclass
class Problem:
    problem_id: str
    kind: str
    case: str

    problem_seed: int | None

    X: np.ndarray
    y: np.ndarray

    X_test: np.ndarray | None
    y_test: np.ndarray | None

    target_std: float | None

    max_steps: int
    checkpoint_steps: int

    reference_weights: np.ndarray
    reference_objective: float

    initial_objective: float
    initial_gap: float

    reference_test_rmse_standardized: float | None
    reference_test_rmse_original: float | None


# =============================================================================
# Utility functions
# =============================================================================

def utc_now_string() -> str:
    return (
        datetime
        .now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def package_version(
    name: str,
) -> str | None:
    try:
        return importlib.metadata.version(
            name
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sha256_array(
    array: np.ndarray,
) -> str:
    arr = np.ascontiguousarray(
        array
    )

    digest = hashlib.sha256()

    digest.update(
        str(arr.dtype).encode(
            "utf-8"
        )
    )

    digest.update(
        str(arr.shape).encode(
            "utf-8"
        )
    )

    digest.update(
        arr.tobytes(
            order="C"
        )
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


def csv_safe(
    value: Any,
) -> Any:
    if value is None:
        return ""

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    fieldnames = list(
        rows[0].keys()
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
                    key: csv_safe(value)
                    for key, value
                    in row.items()
                }
            )

    temporary.replace(
        path
    )


def git_state() -> dict[str, Any]:
    def run_git(
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
                timeout=10,
                check=True,
            )

            return (
                result.stdout.strip()
            )

        except Exception:
            return None

    return {
        "head": run_git(
            "rev-parse",
            "HEAD",
        ),
        "branch": run_git(
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ),
        "status_porcelain": run_git(
            "status",
            "--porcelain",
        ),
    }


# =============================================================================
# Ridge objective and reference solution
# =============================================================================

def objective(
    residual: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        0.5
        * (
            residual @ residual
            + LAM
            * (
                weights @ weights
            )
        )
    )


def rmse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    error = (
        prediction
        - target
    )

    return float(
        np.sqrt(
            np.mean(
                error * error
            )
        )
    )


def build_problem(
    *,
    problem_id: str,
    kind: str,
    case: str,
    problem_seed: int | None,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    target_std: float | None = None,
    max_steps: int | None = None,
    checkpoint_steps: int | None = None,
) -> Problem:

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    if X.ndim != 2:
        raise ValueError(
            f"{problem_id}: X must be 2-D."
        )

    if y.ndim != 1:
        raise ValueError(
            f"{problem_id}: y must be 1-D."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{problem_id}: X/y row mismatch."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            f"{problem_id}: X contains non-finite values."
        )

    if not np.all(
        np.isfinite(y)
    ):
        raise ValueError(
            f"{problem_id}: y contains non-finite values."
        )

    n, d = X.shape

    if max_steps is None:
        max_steps = (
            MAX_EPOCHS
            * d
        )

    if checkpoint_steps is None:
        checkpoint_steps = (
            CHECKPOINT_EPOCHS
            * d
        )

    if max_steps % checkpoint_steps != 0:
        raise ValueError(
            f"{problem_id}: max_steps must be divisible "
            "by checkpoint_steps."
        )

    # Reference solve is evaluation-only and outside method timing.
    hessian = (
        X.T @ X
        + LAM
        * np.eye(
            d,
            dtype=np.float64,
        )
    )

    rhs = (
        X.T @ y
    )

    reference_weights = (
        np.linalg.solve(
            hessian,
            rhs,
        )
    )

    reference_residual = (
        X @ reference_weights
        - y
    )

    reference_objective = objective(
        reference_residual,
        reference_weights,
    )

    zero_weights = np.zeros(
        d,
        dtype=np.float64,
    )

    initial_residual = (
        -y.copy()
    )

    initial_objective = objective(
        initial_residual,
        zero_weights,
    )

    initial_gap = max(
        0.0,
        initial_objective
        - reference_objective,
    )

    if not np.isfinite(
        initial_gap
    ):
        raise RuntimeError(
            f"{problem_id}: invalid initial objective gap."
        )

    if initial_gap <= 0.0:
        raise RuntimeError(
            f"{problem_id}: initial gap is non-positive."
        )

    reference_rmse_std = None
    reference_rmse_original = None

    if X_test is not None:
        if y_test is None:
            raise ValueError(
                f"{problem_id}: X_test provided without y_test."
            )

        X_test = np.asarray(
            X_test,
            dtype=np.float64,
        )

        y_test = np.asarray(
            y_test,
            dtype=np.float64,
        ).reshape(-1)

        prediction = (
            X_test
            @ reference_weights
        )

        reference_rmse_std = rmse(
            prediction,
            y_test,
        )

        if target_std is not None:
            reference_rmse_original = (
                reference_rmse_std
                * float(
                    target_std
                )
            )

    return Problem(
        problem_id=problem_id,
        kind=kind,
        case=case,
        problem_seed=problem_seed,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        target_std=target_std,
        max_steps=max_steps,
        checkpoint_steps=checkpoint_steps,
        reference_weights=reference_weights,
        reference_objective=reference_objective,
        initial_objective=initial_objective,
        initial_gap=initial_gap,
        reference_test_rmse_standardized=(
            reference_rmse_std
        ),
        reference_test_rmse_original=(
            reference_rmse_original
        ),
    )


# =============================================================================
# ASCD active-set rule
# =============================================================================

def active_set(
    estimate: np.ndarray,
    radius: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Return the smallest safe ASCD active set.

    Equality does not permit exclusion.
    """

    upper = (
        np.abs(
            estimate
        )
        + radius
    )

    lower = np.maximum(
        0.0,
        np.abs(
            estimate
        )
        - radius,
    )

    order = np.argsort(
        upper,
        kind="stable",
    )

    selected = np.ones(
        len(estimate),
        dtype=bool,
    )

    total = float(
        np.dot(
            lower,
            lower,
        )
    )

    count = len(
        estimate
    )

    for j in order:
        if (
            count <= 1
            or upper[j] ** 2
            >= total / count
        ):
            break

        selected[j] = False

        total -= (
            lower[j] ** 2
        )

        count -= 1

    return (
        np.flatnonzero(
            selected
        ),
        lower,
    )


# =============================================================================
# Proxy accounting
# =============================================================================

def proxy_components(
    n: int,
    d: int,
    method: str,
) -> dict[str, int]:

    common_setup = (
        n * d
        + d
    )

    method_setup = 0

    oracle_setup = 0

    update = (
        2 * n
        + 3
    )

    if method == "lipschitz":
        method_setup = (
            2 * d
        )

    elif method in {
        "ascd_g3_zero",
        "ascd_g1_exact",
    }:
        # Exact full-gradient initialization.
        method_setup = (
            n * d
            + d
        )

        update += (
            d
            * int(
                math.ceil(
                    math.log2(
                        max(
                            d,
                            2,
                        )
                    )
                )
            )
            + 3 * d
        )

        if (
            method
            == "ascd_g1_exact"
        ):
            # Dense Gram acquisition model.
            oracle_setup = (
                n
                * d
                * d
            )

    elif method != "uniform":
        raise ValueError(
            f"Unknown method: {method}"
        )

    return {
        "common_setup": (
            common_setup
        ),
        "method_setup": (
            method_setup
        ),
        "oracle_setup": (
            oracle_setup
        ),
        "per_update": (
            update
        ),
        "evaluation_per_checkpoint": (
            n + d
        ),
    }


# =============================================================================
# Manifest and prepared-data validation
# =============================================================================

def validate_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing Stage 5 manifest:\n"
            f"  {MANIFEST_PATH}"
        )

    manifest = read_json(
        MANIFEST_PATH
    )

    if manifest.get(
        "stage"
    ) != 5:
        raise RuntimeError(
            "dataset_manifest.json is not a Stage 5 manifest."
        )

    validation = manifest.get(
        "validation",
        {},
    )

    required_flags = (
        "finite_values_checked",
        "real_expected_schemas_checked",
        "residential_sale_price_verified_from_official_workbook_description",
        "residential_other_output_excluded_from_predictors",
        "residential_predictor_names_unique",
        "residential_lag_structure_validated",
    )

    for flag in required_flags:
        if validation.get(
            flag
        ) is not True:
            raise RuntimeError(
                "Stage 5 manifest validation flag "
                f"{flag!r} is not true."
            )

    if validation.get(
        "optimization_methods_run"
    ) is not False:
        raise RuntimeError(
            "The dataset manifest does not identify "
            "the preparation phase as optimization-free."
        )

    real_records = manifest.get(
        "real_datasets",
        [],
    )

    synthetic_records = manifest.get(
        "synthetic_datasets",
        [],
    )

    if len(
        real_records
    ) != 4:
        raise RuntimeError(
            "Expected 4 prepared real datasets."
        )

    if len(
        synthetic_records
    ) != 48:
        raise RuntimeError(
            "Expected 48 held-out synthetic datasets."
        )

    print(
        "Verifying prepared-data hashes..."
    )

    records = (
        real_records
        + synthetic_records
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        relative = record.get(
            "prepared_file"
        )

        expected_hash = record.get(
            "prepared_file_sha256"
        )

        if not relative:
            raise RuntimeError(
                "Manifest record missing prepared_file."
            )

        if not expected_hash:
            raise RuntimeError(
                "Manifest record missing prepared_file_sha256."
            )

        path = (
            ROOT
            / Path(relative)
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        observed_hash = (
            sha256_file(
                path
            )
        )

        if (
            observed_hash
            != expected_hash
        ):
            raise RuntimeError(
                "Prepared-data hash mismatch:\n"
                f"  {path}\n"
                f"Expected: {expected_hash}\n"
                f"Observed: {observed_hash}"
            )

        if (
            index % 10 == 0
            or index == len(records)
        ):
            print(
                f"  verified {index}/{len(records)} files"
            )

    print(
        "Prepared-data hash validation: PASSED"
    )

    return manifest


# =============================================================================
# Prepared problem loading
# =============================================================================

def load_real_problem(
    record: dict[str, Any],
) -> Problem:

    path = (
        ROOT
        / record[
            "prepared_file"
        ]
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as payload:

        X_train = np.asarray(
            payload["X_train"],
            dtype=np.float64,
        )

        y_train = np.asarray(
            payload["y_train"],
            dtype=np.float64,
        )

        X_test = np.asarray(
            payload["X_test"],
            dtype=np.float64,
        )

        y_test = np.asarray(
            payload["y_test"],
            dtype=np.float64,
        )

        target_std = float(
            payload[
                "target_std"
            ]
        )

        feature_names = [
            str(item)
            for item
            in payload[
                "feature_names"
            ].tolist()
        ]

        target_name = str(
            payload[
                "target_name"
            ].item()
        )

    if len(
        feature_names
    ) != X_train.shape[1]:
        raise RuntimeError(
            f"{record['slug']}: feature-name count mismatch."
        )

    if len(
        set(feature_names)
    ) != len(
        feature_names
    ):
        raise RuntimeError(
            f"{record['slug']}: feature names are not unique."
        )

    if (
        record["slug"]
        == "residential_building"
    ):
        if target_name != "V-9":
            raise RuntimeError(
                "Residential target is not V-9."
            )

        if (
            "V-9" in feature_names
            or "V-10" in feature_names
        ):
            raise RuntimeError(
                "Residential output leakage detected."
            )

        if not {
            "V-11_lag1",
            "V-29_lag1",
            "V-11_lag5",
            "V-29_lag5",
        }.issubset(
            set(
                feature_names
            )
        ):
            raise RuntimeError(
                "Residential lag-aware feature schema invalid."
            )

    return build_problem(
        problem_id=record[
            "slug"
        ],
        kind="real",
        case=record[
            "slug"
        ],
        problem_seed=None,
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        target_std=target_std,
    )


def load_synthetic_problem(
    record: dict[str, Any],
) -> Problem:

    path = (
        ROOT
        / record[
            "prepared_file"
        ]
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as payload:

        X = np.asarray(
            payload["X"],
            dtype=np.float64,
        )

        y = np.asarray(
            payload["y"],
            dtype=np.float64,
        )

        stored_seed = int(
            payload["seed"]
        )

    seed = int(
        record["seed"]
    )

    if stored_seed != seed:
        raise RuntimeError(
            f"{record['case']}: seed mismatch."
        )

    return build_problem(
        problem_id=(
            f"{record['case']}_seed{seed}"
        ),
        kind="synthetic",
        case=record["case"],
        problem_seed=seed,
        X=X,
        y=y,
    )


# =============================================================================
# One method execution
# =============================================================================

def execute_method(
    problem: Problem,
    method: str,
    algorithm_seed: int,
    *,
    collect_trace: bool,
    verify: bool,
) -> dict[str, Any]:

    if method not in METHODS:
        raise ValueError(
            f"Unknown method: {method}"
        )

    X = problem.X
    y = problem.y

    n, d = X.shape

    rng = np.random.default_rng(
        algorithm_seed
    )

    proxies = proxy_components(
        n,
        d,
        method,
    )

    proxy_core = (
        proxies[
            "common_setup"
        ]
        + proxies[
            "method_setup"
        ]
    )

    proxy_costed = (
        proxy_core
        + proxies[
            "oracle_setup"
        ]
    )

    # -------------------------------------------------------------------------
    # Harmonized timed method setup
    # -------------------------------------------------------------------------

    setup_start = (
        time.perf_counter()
    )

    weights = np.zeros(
        d,
        dtype=np.float64,
    )

    residual = (
        -y.copy()
    )

    smoothness = (
        np.einsum(
            "ij,ij->j",
            X,
            X,
            optimize=True,
        )
        + LAM
    )

    if np.any(
        smoothness <= 0.0
    ):
        raise RuntimeError(
            f"{problem.problem_id}: "
            "non-positive coordinate smoothness."
        )

    probabilities = None

    estimate = None
    radius = None
    norms = None
    exact_hessian = None

    oracle_setup_seconds = 0.0

    if method == "uniform":
        pass

    elif method == "lipschitz":
        probabilities = (
            smoothness
            / np.sum(
                smoothness
            )
        )

    elif method == "ascd_g3_zero":
        norms = np.sqrt(
            np.maximum(
                smoothness
                - LAM,
                0.0,
            )
        )

        estimate = (
            X.T @ residual
        )

        radius = np.zeros(
            d,
            dtype=np.float64,
        )

    elif method == "ascd_g1_exact":
        oracle_start = (
            time.perf_counter()
        )

        exact_hessian = (
            X.T @ X
        )

        diagonal = (
            np.diag_indices(
                d
            )
        )

        exact_hessian[
            diagonal
        ] += LAM

        oracle_setup_seconds = (
            time.perf_counter()
            - oracle_start
        )

        estimate = (
            X.T @ residual
        )

        radius = np.zeros(
            d,
            dtype=np.float64,
        )

        if verify:
            if not np.allclose(
                np.diag(
                    exact_hessian
                ),
                smoothness,
                rtol=1e-11,
                atol=1e-12,
            ):
                raise AssertionError(
                    "g1 Hessian diagonal does not "
                    "match coordinate smoothness."
                )

    setup_seconds = (
        time.perf_counter()
        - setup_start
    )

    update_seconds = 0.0
    evaluation_seconds = 0.0

    method_elapsed = (
        setup_seconds
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    active_sizes: list[int] = []
    zero_lower_fractions: list[float] = []

    full_active_count = 0
    half_or_less_count = 0

    trace_rows: list[
        dict[str, Any]
    ] = []

    first_primary_step = None
    first_primary_seconds = None
    first_primary_proxy_core = None
    first_primary_proxy_costed = None

    first_secondary_step = None
    first_secondary_seconds = None
    first_secondary_proxy_core = None
    first_secondary_proxy_costed = None

    previous_objective = None

    # -------------------------------------------------------------------------
    # Checkpoint-block loop
    # -------------------------------------------------------------------------

    step = 0

    while True:
        # Evaluation-only checkpoint.
        evaluation_start = (
            time.perf_counter()
        )

        current_objective = objective(
            residual,
            weights,
        )

        absolute_gap = max(
            0.0,
            current_objective
            - problem.reference_objective,
        )

        relative_gap = (
            absolute_gap
            / max(
                problem.initial_gap,
                1e-30,
            )
        )

        if verify:
            reconstructed = (
                X @ weights
                - y
            )

            if not np.allclose(
                residual,
                reconstructed,
                rtol=RESIDUAL_RTOL,
                atol=RESIDUAL_ATOL,
            ):
                raise AssertionError(
                    f"{method}: residual invariant failed "
                    f"at step {step}."
                )

            if previous_objective is not None:
                tolerance = (
                    1e-10
                    * max(
                        1.0,
                        abs(
                            previous_objective
                        ),
                    )
                    + 1e-12
                )

                if (
                    current_objective
                    > previous_objective
                    + tolerance
                ):
                    raise AssertionError(
                        f"{method}: objective increased "
                        f"at checkpoint step {step}."
                    )

            if method in {
                "ascd_g3_zero",
                "ascd_g1_exact",
            }:
                assert estimate is not None
                assert radius is not None

                actual_gradient = (
                    X.T @ residual
                    + LAM
                    * weights
                )

                discrepancy = np.abs(
                    actual_gradient
                    - estimate
                )

                tolerance = (
                    INTERVAL_ATOL
                    + INTERVAL_RTOL
                    * np.abs(
                        actual_gradient
                    )
                )

                if np.any(
                    discrepancy
                    > radius
                    + tolerance
                ):
                    worst = float(
                        np.max(
                            discrepancy
                            - radius
                            - tolerance
                        )
                    )

                    raise AssertionError(
                        f"{method}: ASCD interval containment "
                        f"failed at step {step}; "
                        f"worst excess={worst:.3e}."
                    )

            previous_objective = (
                current_objective
            )

        evaluation_seconds += (
            time.perf_counter()
            - evaluation_start
        )

        if (
            first_primary_step
            is None
            and relative_gap
            <= PRIMARY_TARGET
        ):
            first_primary_step = (
                step
            )

            first_primary_seconds = (
                method_elapsed
            )

            first_primary_proxy_core = (
                proxy_core
            )

            first_primary_proxy_costed = (
                proxy_costed
            )

        if (
            first_secondary_step
            is None
            and relative_gap
            <= SECONDARY_TARGET
        ):
            first_secondary_step = (
                step
            )

            first_secondary_seconds = (
                method_elapsed
            )

            first_secondary_proxy_core = (
                proxy_core
            )

            first_secondary_proxy_costed = (
                proxy_costed
            )

        if collect_trace:
            trace_rows.append(
                {
                    "steps": step,
                    "coordinate_epochs": (
                        step / d
                    ),
                    "objective": (
                        current_objective
                    ),
                    "absolute_gap": (
                        absolute_gap
                    ),
                    "relative_gap": (
                        relative_gap
                    ),
                    "proxy_core": (
                        proxy_core
                    ),
                    "proxy_costed": (
                        proxy_costed
                    ),
                }
            )

        if (
            step
            == problem.max_steps
        ):
            break

        block_steps = min(
            problem.checkpoint_steps,
            problem.max_steps
            - step,
        )

        block_start = (
            time.perf_counter()
        )

        for _ in range(
            block_steps
        ):
            lower = None
            candidates = None

            if method == "uniform":
                coordinate = int(
                    rng.integers(
                        d
                    )
                )

            elif method == "lipschitz":
                assert probabilities is not None

                coordinate = int(
                    rng.choice(
                        d,
                        p=probabilities,
                    )
                )

            else:
                assert estimate is not None
                assert radius is not None

                candidates, lower = active_set(
                    estimate,
                    radius,
                )

                best = np.max(
                    lower[
                        candidates
                    ]
                )

                tied = candidates[
                    lower[
                        candidates
                    ]
                    == best
                ]

                coordinate = int(
                    rng.choice(
                        tied
                    )
                )

                if collect_trace:
                    active_size = int(
                        len(
                            candidates
                        )
                    )

                    active_sizes.append(
                        active_size
                    )

                    if active_size == d:
                        full_active_count += 1

                    if (
                        active_size
                        <= d / 2
                    ):
                        half_or_less_count += 1

                    zero_lower_fraction = float(
                        np.mean(
                            lower
                            == 0.0
                        )
                    )

                    zero_lower_fractions.append(
                        zero_lower_fraction
                    )

            gradient = float(
                X[
                    :,
                    coordinate,
                ]
                @ residual
                + LAM
                * weights[
                    coordinate
                ]
            )

            delta = (
                -gradient
                / smoothness[
                    coordinate
                ]
            )

            weights[
                coordinate
            ] += delta

            residual += (
                delta
                * X[
                    :,
                    coordinate
                ]
            )

            if method == "ascd_g3_zero":
                assert estimate is not None
                assert radius is not None
                assert norms is not None

                radius += (
                    abs(
                        delta
                    )
                    * norms[
                        coordinate
                    ]
                    * norms
                )

                estimate[
                    coordinate
                ] = 0.0

                radius[
                    coordinate
                ] = 0.0

            elif method == "ascd_g1_exact":
                assert estimate is not None
                assert radius is not None
                assert exact_hessian is not None

                estimate += (
                    delta
                    * exact_hessian[
                        :,
                        coordinate,
                    ]
                )

                # Exact coordinate minimization sets the selected
                # coordinate gradient to zero. Reset explicitly to
                # suppress accumulated floating-point drift.
                estimate[
                    coordinate
                ] = 0.0

                radius[
                    coordinate
                ] = 0.0

        block_elapsed = (
            time.perf_counter()
            - block_start
        )

        update_seconds += (
            block_elapsed
        )

        method_elapsed += (
            block_elapsed
        )

        proxy_increment = (
            block_steps
            * proxies[
                "per_update"
            ]
        )

        proxy_core += (
            proxy_increment
        )

        proxy_costed += (
            proxy_increment
        )

        step += (
            block_steps
        )

    # -------------------------------------------------------------------------
    # Final run diagnostics
    # -------------------------------------------------------------------------

    final_objective = (
        current_objective
    )

    final_absolute_gap = (
        absolute_gap
    )

    final_relative_gap = (
        relative_gap
    )

    total_method_seconds = (
        setup_seconds
        + update_seconds
    )

    checkpoint_count = (
        problem.max_steps
        // problem.checkpoint_steps
        + 1
    )

    evaluation_proxy_work = (
        checkpoint_count
        * proxies[
            "evaluation_per_checkpoint"
        ]
    )

    median_active_set_size = None
    median_active_set_fraction = None
    full_active_fraction = None
    pruned_fraction = None
    half_or_less_fraction = None
    median_zero_lower_fraction = None

    if active_sizes:
        median_active_set_size = float(
            np.median(
                active_sizes
            )
        )

        median_active_set_fraction = (
            median_active_set_size
            / d
        )

        update_count = len(
            active_sizes
        )

        full_active_fraction = (
            full_active_count
            / update_count
        )

        pruned_fraction = (
            1.0
            - full_active_fraction
        )

        half_or_less_fraction = (
            half_or_less_count
            / update_count
        )

        median_zero_lower_fraction = float(
            np.median(
                zero_lower_fractions
            )
        )

    return {
        "weights": weights.copy(),

        "trace_rows": trace_rows,

        "first_primary_step": (
            first_primary_step
        ),

        "first_primary_seconds": (
            first_primary_seconds
        ),

        "first_primary_proxy_core": (
            first_primary_proxy_core
        ),

        "first_primary_proxy_costed": (
            first_primary_proxy_costed
        ),

        "first_secondary_step": (
            first_secondary_step
        ),

        "first_secondary_seconds": (
            first_secondary_seconds
        ),

        "first_secondary_proxy_core": (
            first_secondary_proxy_core
        ),

        "first_secondary_proxy_costed": (
            first_secondary_proxy_costed
        ),

        "final_objective": (
            final_objective
        ),

        "final_absolute_gap": (
            final_absolute_gap
        ),

        "final_relative_gap": (
            final_relative_gap
        ),

        "setup_seconds": (
            setup_seconds
        ),

        "oracle_setup_seconds": (
            oracle_setup_seconds
        ),

        "update_seconds": (
            update_seconds
        ),

        "total_method_seconds": (
            total_method_seconds
        ),

        "evaluation_seconds": (
            evaluation_seconds
        ),

        "proxy_common_setup": (
            proxies[
                "common_setup"
            ]
        ),

        "proxy_method_setup": (
            proxies[
                "method_setup"
            ]
        ),

        "proxy_oracle_setup": (
            proxies[
                "oracle_setup"
            ]
        ),

        "proxy_per_update": (
            proxies[
                "per_update"
            ]
        ),

        "proxy_core_final": (
            proxy_core
        ),

        "proxy_costed_final": (
            proxy_costed
        ),

        "evaluation_proxy_work": (
            evaluation_proxy_work
        ),

        "median_active_set_size": (
            median_active_set_size
        ),

        "median_active_set_fraction": (
            median_active_set_fraction
        ),

        "full_active_fraction": (
            full_active_fraction
        ),

        "pruned_fraction": (
            pruned_fraction
        ),

        "half_or_less_active_fraction": (
            half_or_less_fraction
        ),

        "median_zero_lower_fraction": (
            median_zero_lower_fraction
        ),
    }


# =============================================================================
# Repeat consistency
# =============================================================================

def assert_repeat_consistency(
    warmup: dict[str, Any],
    timed: dict[str, Any],
    *,
    method: str,
    problem_id: str,
) -> None:

    if (
        warmup[
            "first_primary_step"
        ]
        != timed[
            "first_primary_step"
        ]
    ):
        raise AssertionError(
            f"{problem_id}/{method}: "
            "primary target step changed between "
            "warm-up and timed repetition."
        )

    if (
        warmup[
            "first_secondary_step"
        ]
        != timed[
            "first_secondary_step"
        ]
    ):
        raise AssertionError(
            f"{problem_id}/{method}: "
            "secondary target step changed between "
            "warm-up and timed repetition."
        )

    if not np.allclose(
        warmup["weights"],
        timed["weights"],
        rtol=REPEAT_RTOL,
        atol=REPEAT_ATOL,
    ):
        difference = float(
            np.max(
                np.abs(
                    warmup["weights"]
                    - timed["weights"]
                )
            )
        )

        raise AssertionError(
            f"{problem_id}/{method}: "
            "final weights changed between repeated runs; "
            f"max difference={difference:.3e}."
        )

    if not np.isclose(
        warmup[
            "final_relative_gap"
        ],
        timed[
            "final_relative_gap"
        ],
        rtol=REPEAT_RTOL,
        atol=REPEAT_ATOL,
    ):
        raise AssertionError(
            f"{problem_id}/{method}: "
            "final normalized gap changed between repetitions."
        )


# =============================================================================
# Method/problem benchmark unit
# =============================================================================

def benchmark_pair(
    problem: Problem,
    method: str,
    algorithm_seed: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:

    # -------------------------------------------------------------------------
    # Warm-up + diagnostics
    # -------------------------------------------------------------------------

    warmup = execute_method(
        problem,
        method,
        algorithm_seed,
        collect_trace=True,
        verify=True,
    )

    # -------------------------------------------------------------------------
    # Three timing repetitions
    # -------------------------------------------------------------------------

    timed_runs: list[
        dict[str, Any]
    ] = []

    for _ in range(
        TIMED_REPEATS
    ):
        timed = execute_method(
            problem,
            method,
            algorithm_seed,
            collect_trace=False,
            verify=False,
        )

        assert_repeat_consistency(
            warmup,
            timed,
            method=method,
            problem_id=(
                problem.problem_id
            ),
        )

        timed_runs.append(
            timed
        )

    def median_field(
        field: str,
    ) -> float:
        return float(
            np.median(
                [
                    run[field]
                    for run
                    in timed_runs
                ]
            )
        )

    target_times = [
        run[
            "first_primary_seconds"
        ]
        for run
        in timed_runs
        if run[
            "first_primary_seconds"
        ]
        is not None
    ]

    secondary_times = [
        run[
            "first_secondary_seconds"
        ]
        for run
        in timed_runs
        if run[
            "first_secondary_seconds"
        ]
        is not None
    ]

    median_primary_seconds = (
        float(
            np.median(
                target_times
            )
        )
        if target_times
        else None
    )

    median_secondary_seconds = (
        float(
            np.median(
                secondary_times
            )
        )
        if secondary_times
        else None
    )

    # -------------------------------------------------------------------------
    # Real-data predictive diagnostics
    # -------------------------------------------------------------------------

    final_test_rmse_std = None
    final_test_rmse_original = None

    reference_test_rmse_std = (
        problem
        .reference_test_rmse_standardized
    )

    reference_test_rmse_original = (
        problem
        .reference_test_rmse_original
    )

    test_rmse_difference_std = None
    test_rmse_difference_original = None

    if (
        problem.X_test
        is not None
    ):
        prediction = (
            problem.X_test
            @ warmup[
                "weights"
            ]
        )

        assert (
            problem.y_test
            is not None
        )

        final_test_rmse_std = rmse(
            prediction,
            problem.y_test,
        )

        if (
            problem.target_std
            is not None
        ):
            final_test_rmse_original = (
                final_test_rmse_std
                * problem.target_std
            )

        if (
            reference_test_rmse_std
            is not None
        ):
            test_rmse_difference_std = (
                final_test_rmse_std
                - reference_test_rmse_std
            )

        if (
            reference_test_rmse_original
            is not None
            and final_test_rmse_original
            is not None
        ):
            test_rmse_difference_original = (
                final_test_rmse_original
                - reference_test_rmse_original
            )

    n, d = (
        problem.X.shape
    )

    primary_step = (
        warmup[
            "first_primary_step"
        ]
    )

    secondary_step = (
        warmup[
            "first_secondary_step"
        ]
    )

    row = {
        "problem_id": (
            problem.problem_id
        ),

        "kind": (
            problem.kind
        ),

        "case": (
            problem.case
        ),

        "problem_seed": (
            problem.problem_seed
        ),

        "algorithm_seed": (
            algorithm_seed
        ),

        "method": (
            method
        ),

        "n": n,
        "d": d,

        "max_steps": (
            problem.max_steps
        ),

        "checkpoint_steps": (
            problem.checkpoint_steps
        ),

        "max_coordinate_epochs": (
            problem.max_steps / d
        ),

        "primary_target": (
            PRIMARY_TARGET
        ),

        "secondary_target": (
            SECONDARY_TARGET
        ),

        "success_primary": (
            int(
                primary_step
                is not None
            )
        ),

        "success_secondary": (
            int(
                secondary_step
                is not None
            )
        ),

        "steps_to_primary": (
            primary_step
        ),

        "epochs_to_primary": (
            (
                primary_step
                / d
            )
            if primary_step
            is not None
            else None
        ),

        "steps_to_secondary": (
            secondary_step
        ),

        "epochs_to_secondary": (
            (
                secondary_step
                / d
            )
            if secondary_step
            is not None
            else None
        ),

        "initial_objective": (
            problem.initial_objective
        ),

        "reference_objective": (
            problem.reference_objective
        ),

        "initial_absolute_gap": (
            problem.initial_gap
        ),

        "final_objective": (
            warmup[
                "final_objective"
            ]
        ),

        "final_absolute_gap": (
            warmup[
                "final_absolute_gap"
            ]
        ),

        "final_relative_gap": (
            warmup[
                "final_relative_gap"
            ]
        ),

        "median_setup_seconds": (
            median_field(
                "setup_seconds"
            )
        ),

        "median_oracle_setup_seconds": (
            median_field(
                "oracle_setup_seconds"
            )
        ),

        "median_update_seconds": (
            median_field(
                "update_seconds"
            )
        ),

        "median_total_method_seconds": (
            median_field(
                "total_method_seconds"
            )
        ),

        "median_evaluation_seconds_excluded": (
            median_field(
                "evaluation_seconds"
            )
        ),

        "median_seconds_to_primary": (
            median_primary_seconds
        ),

        "median_seconds_to_secondary": (
            median_secondary_seconds
        ),

        "repeat1_total_method_seconds": (
            timed_runs[0][
                "total_method_seconds"
            ]
        ),

        "repeat2_total_method_seconds": (
            timed_runs[1][
                "total_method_seconds"
            ]
        ),

        "repeat3_total_method_seconds": (
            timed_runs[2][
                "total_method_seconds"
            ]
        ),

        "repeat1_seconds_to_primary": (
            timed_runs[0][
                "first_primary_seconds"
            ]
        ),

        "repeat2_seconds_to_primary": (
            timed_runs[1][
                "first_primary_seconds"
            ]
        ),

        "repeat3_seconds_to_primary": (
            timed_runs[2][
                "first_primary_seconds"
            ]
        ),

        "proxy_common_setup": (
            warmup[
                "proxy_common_setup"
            ]
        ),

        "proxy_method_setup": (
            warmup[
                "proxy_method_setup"
            ]
        ),

        "proxy_oracle_setup": (
            warmup[
                "proxy_oracle_setup"
            ]
        ),

        "proxy_per_update": (
            warmup[
                "proxy_per_update"
            ]
        ),

        "proxy_core_to_primary": (
            warmup[
                "first_primary_proxy_core"
            ]
        ),

        "proxy_costed_to_primary": (
            warmup[
                "first_primary_proxy_costed"
            ]
        ),

        "proxy_core_final": (
            warmup[
                "proxy_core_final"
            ]
        ),

        "proxy_costed_final": (
            warmup[
                "proxy_costed_final"
            ]
        ),

        "evaluation_proxy_work_excluded": (
            warmup[
                "evaluation_proxy_work"
            ]
        ),

        "median_active_set_size": (
            warmup[
                "median_active_set_size"
            ]
        ),

        "median_active_set_fraction": (
            warmup[
                "median_active_set_fraction"
            ]
        ),

        "full_active_fraction": (
            warmup[
                "full_active_fraction"
            ]
        ),

        "pruned_fraction": (
            warmup[
                "pruned_fraction"
            ]
        ),

        "half_or_less_active_fraction": (
            warmup[
                "half_or_less_active_fraction"
            ]
        ),

        "median_zero_lower_fraction": (
            warmup[
                "median_zero_lower_fraction"
            ]
        ),

        "reference_test_rmse_standardized": (
            reference_test_rmse_std
        ),

        "final_test_rmse_standardized": (
            final_test_rmse_std
        ),

        "test_rmse_difference_standardized": (
            test_rmse_difference_std
        ),

        "reference_test_rmse_original": (
            reference_test_rmse_original
        ),

        "final_test_rmse_original": (
            final_test_rmse_original
        ),

        "test_rmse_difference_original": (
            test_rmse_difference_original
        ),

        "final_weights_sha256": (
            sha256_array(
                warmup[
                    "weights"
                ]
            )
        ),
    }

    trajectory_rows: list[
        dict[str, Any]
    ] = []

    for trace in warmup[
        "trace_rows"
    ]:
        trajectory_rows.append(
            {
                "problem_id": (
                    problem.problem_id
                ),
                "kind": (
                    problem.kind
                ),
                "case": (
                    problem.case
                ),
                "problem_seed": (
                    problem.problem_seed
                ),
                "algorithm_seed": (
                    algorithm_seed
                ),
                "method": (
                    method
                ),
                "n": n,
                "d": d,
                **trace,
            }
        )

    return (
        row,
        trajectory_rows,
    )


# =============================================================================
# Internal implementation audit
# =============================================================================

def self_audit() -> None:
    print(
        "Running Stage 5 implementation audit..."
    )

    # Active-set edge checks.
    candidates, _ = active_set(
        np.zeros(3),
        np.zeros(3),
    )

    if candidates.size != 3:
        raise AssertionError(
            "Zero-bound active-set audit failed."
        )

    candidates, _ = active_set(
        np.array(
            [
                3.0,
                1.0,
                0.0,
            ]
        ),
        np.zeros(3),
    )

    if not np.array_equal(
        candidates,
        np.array(
            [0]
        ),
    ):
        raise AssertionError(
            "Exact active-set audit failed."
        )

    # Small problem unrelated to Stage 5 outcome datasets.
    rng = np.random.default_rng(
        20260926
    )

    X = rng.normal(
        size=(
            40,
            6,
        )
    )

    y = (
        X
        @ rng.normal(
            size=6
        )
        + 0.1
        * rng.normal(
            size=40
        )
    )

    problem = build_problem(
        problem_id="implementation_audit",
        kind="audit",
        case="implementation_audit",
        problem_seed=None,
        X=X,
        y=y,
        max_steps=60,
        checkpoint_steps=6,
    )

    for method in METHODS:
        result = execute_method(
            problem,
            method,
            9100,
            collect_trace=True,
            verify=True,
        )

        if not np.all(
            np.isfinite(
                result[
                    "weights"
                ]
            )
        ):
            raise AssertionError(
                f"{method}: audit produced non-finite weights."
            )

    print(
        "Stage 5 implementation audit: PASSED"
    )


# =============================================================================
# Benchmark problem enumeration
# =============================================================================

def sorted_synthetic_records(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:

    case_rank = {
        case: index
        for index, case
        in enumerate(
            SYNTHETIC_CASE_ORDER
        )
    }

    records = list(
        manifest[
            "synthetic_datasets"
        ]
    )

    records.sort(
        key=lambda item: (
            case_rank[
                item["case"]
            ],
            int(
                item["seed"]
            ),
        )
    )

    observed_cases = {
        record["case"]
        for record
        in records
    }

    if observed_cases != set(
        SYNTHETIC_CASE_ORDER
    ):
        raise RuntimeError(
            "Synthetic case set does not match Stage 5 protocol."
        )

    return records


def sorted_real_records(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:

    by_slug = {
        record["slug"]:
        record
        for record
        in manifest[
            "real_datasets"
        ]
    }

    if set(
        by_slug
    ) != set(
        REAL_DATASET_ORDER
    ):
        raise RuntimeError(
            "Real dataset set does not match Stage 5 protocol."
        )

    return [
        by_slug[
            slug
        ]
        for slug
        in REAL_DATASET_ORDER
    ]


# =============================================================================
# Environment record
# =============================================================================

def build_environment_record(
    *,
    manifest: dict[str, Any],
    started_at: str,
    completed_at: str,
    run_count: int,
    trajectory_count: int,
    threadpools: list[dict[str, Any]],
) -> dict[str, Any]:

    timer_info = (
        time.get_clock_info(
            "perf_counter"
        )
    )

    return {
        "stage": 5,

        "experiment": (
            "confirmatory benchmark"
        ),

        "started_at_utc": (
            started_at
        ),

        "completed_at_utc": (
            completed_at
        ),

        "dataset_manifest": (
            str(
                MANIFEST_PATH
                .relative_to(ROOT)
            )
        ),

        "dataset_manifest_sha256": (
            sha256_file(
                MANIFEST_PATH
            )
        ),

        "dataset_manifest_generated_at_utc": (
            manifest.get(
                "generated_at_utc"
            )
        ),

        "git": (
            git_state()
        ),

        "platform": {
            "python": (
                sys.version
            ),

            "python_executable": (
                sys.executable
            ),

            "operating_system": (
                platform.platform()
            ),

            "machine": (
                platform.machine()
            ),

            "processor": (
                platform.processor()
            ),

            "logical_cpu_count": (
                os.cpu_count()
            ),
        },

        "packages": {
            "numpy": (
                np.__version__
            ),

            "threadpoolctl": (
                package_version(
                    "threadpoolctl"
                )
            ),

            "scikit_learn": (
                package_version(
                    "scikit-learn"
                )
            ),
        },

        "threadpool_policy": {
            "requested_threads": 1,
            "threadpool_info_while_limited": (
                threadpools
            ),
        },

        "timer": {
            "implementation": (
                timer_info.implementation
            ),
            "monotonic": (
                timer_info.monotonic
            ),
            "adjustable": (
                timer_info.adjustable
            ),
            "resolution_seconds": (
                timer_info.resolution
            ),
        },

        "objective": {
            "lambda": LAM,
        },

        "convergence": {
            "primary_relative_gap": (
                PRIMARY_TARGET
            ),

            "secondary_relative_gap": (
                SECONDARY_TARGET
            ),

            "normalized_gap_definition": (
                "(f(w)-f*) / "
                "max(f(w0)-f*, 1e-30)"
            ),

            "maximum_coordinate_epochs": (
                MAX_EPOCHS
            ),

            "checkpoint_coordinate_epochs": (
                CHECKPOINT_EPOCHS
            ),
        },

        "timing_protocol": {
            "warmup_executions_per_method_problem":
            1,

            "timed_repetitions_per_method_problem":
            TIMED_REPEATS,

            "method_specific_setup_included":
            True,

            "coordinate_selection_included":
            True,

            "coordinate_updates_included":
            True,

            "g1_gram_acquisition_included":
            True,

            "checkpoint_evaluation_excluded_from_primary_runtime":
            True,

            "dense_reference_optimum_excluded_from_method_runtime":
            True,
        },

        "proxy_model": {
            "primary_role": (
                "secondary accounting measure; "
                "not exact FLOPs"
            ),

            "common_setup": (
                "n*d + d"
            ),

            "lipschitz_additional_setup": (
                "2*d"
            ),

            "ascd_full_gradient_setup": (
                "n*d + d"
            ),

            "base_coordinate_update": (
                "2*n + 3"
            ),

            "ascd_selection_and_maintenance_per_update": (
                "d*ceil(log2(d)) + 3*d"
            ),

            "g1_oracle_acquisition": (
                "n*d*d"
            ),

            "checkpoint_evaluation": (
                "n + d, reported separately and excluded "
                "from method proxy"
            ),
        },

        "methods": list(
            METHODS
        ),

        "algorithm_seeds": list(
            ALGORITHM_SEEDS
        ),

        "synthetic_problem_seeds": list(
            SYNTHETIC_PROBLEM_SEEDS
        ),

        "completed_run_rows": (
            run_count
        ),

        "completed_trajectory_rows": (
            trajectory_count
        ),

        "outputs": {
            "run_results": str(
                RUN_RESULTS_PATH
                .relative_to(ROOT)
            ),

            "trajectories": str(
                TRAJECTORIES_PATH
                .relative_to(ROOT)
            ),

            "environment": str(
                ENVIRONMENT_PATH
                .relative_to(ROOT)
            ),
        },
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Stage 5 confirmatory "
            "coordinate-descent benchmark."
        )
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Run implementation audits only. "
            "Do not load Stage 5 datasets or produce benchmark outputs."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of existing canonical Stage 5 "
            "benchmark outputs. Use only for a documented technical rerun."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with threadpool_limits(
        limits=1
    ):
        self_audit()

    if args.audit_only:
        print(
            "\nAUDIT-ONLY MODE COMPLETE."
        )
        print(
            "No Stage 5 confirmatory outcomes were generated."
        )
        return

    canonical_outputs = (
        RUN_RESULTS_PATH,
        TRAJECTORIES_PATH,
        ENVIRONMENT_PATH,
    )

    existing = [
        path
        for path
        in canonical_outputs
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
            "Canonical Stage 5 benchmark output already exists.\n\n"
            f"{formatted}\n\n"
            "The script will not silently overwrite confirmatory results.\n"
            "If a technical rerun is genuinely required, archive the "
            "existing outputs and use --overwrite."
        )

    print()
    print(
        "=" * 76
    )
    print(
        "STAGE 5 CONFIRMATORY BENCHMARK"
    )
    print(
        "=" * 76
    )

    print(
        "\nLoading and validating frozen Stage 5 inputs..."
    )

    manifest = validate_manifest()

    git = git_state()

    if git.get(
        "status_porcelain"
    ):
        print()
        print(
            "WARNING: Git working tree is not clean."
        )
        print(
            "For the publication-oriented canonical run, "
            "committing the frozen protocol/scripts first is preferable."
        )

    run_rows: list[
        dict[str, Any]
    ] = []

    trajectory_rows: list[
        dict[str, Any]
    ] = []

    started_at = (
        utc_now_string()
    )

    total_pairs = (
        4
        * 12
        * len(
            METHODS
        )
        + 4
        * 12
        * len(
            METHODS
        )
    )

    pair_index = 0

    with threadpool_limits(
        limits=1
    ):
        limited_threadpool_info = (
            threadpool_info()
        )

        # =====================================================================
        # Held-out synthetic validation
        # =====================================================================

        print()
        print(
            "[SYNTHETIC CONFIRMATORY RUNS]"
        )

        synthetic_records = (
            sorted_synthetic_records(
                manifest
            )
        )

        for record in (
            synthetic_records
        ):
            problem = (
                load_synthetic_problem(
                    record
                )
            )

            problem_seed = int(
                record["seed"]
            )

            if problem_seed not in (
                SYNTHETIC_PROBLEM_SEEDS
            ):
                raise RuntimeError(
                    "Unexpected synthetic problem seed."
                )

            seed_index = (
                problem_seed
                - SYNTHETIC_PROBLEM_SEEDS[0]
            )

            algorithm_seed = (
                ALGORITHM_SEEDS[
                    seed_index
                ]
            )

            for method in METHODS:
                pair_index += 1

                print(
                    f"[{pair_index:03d}/{total_pairs}] "
                    f"{problem.problem_id} | "
                    f"alg_seed={algorithm_seed} | "
                    f"{method}",
                    flush=True,
                )

                row, traces = (
                    benchmark_pair(
                        problem,
                        method,
                        algorithm_seed,
                    )
                )

                run_rows.append(
                    row
                )

                trajectory_rows.extend(
                    traces
                )

                print(
                    "    "
                    f"primary={bool(row['success_primary'])}, "
                    f"final_rel_gap="
                    f"{row['final_relative_gap']:.3e}, "
                    f"median_time="
                    f"{row['median_total_method_seconds']:.6f}s",
                    flush=True,
                )

        # =====================================================================
        # Real-data validation
        # =====================================================================

        print()
        print(
            "[REAL-DATA CONFIRMATORY RUNS]"
        )

        real_records = (
            sorted_real_records(
                manifest
            )
        )

        for record in (
            real_records
        ):
            problem = (
                load_real_problem(
                    record
                )
            )

            for algorithm_seed in (
                ALGORITHM_SEEDS
            ):
                for method in METHODS:
                    pair_index += 1

                    print(
                        f"[{pair_index:03d}/{total_pairs}] "
                        f"{problem.problem_id} | "
                        f"alg_seed={algorithm_seed} | "
                        f"{method}",
                        flush=True,
                    )

                    row, traces = (
                        benchmark_pair(
                            problem,
                            method,
                            algorithm_seed,
                        )
                    )

                    run_rows.append(
                        row
                    )

                    trajectory_rows.extend(
                        traces
                    )

                    print(
                        "    "
                        f"primary={bool(row['success_primary'])}, "
                        f"final_rel_gap="
                        f"{row['final_relative_gap']:.3e}, "
                        f"median_time="
                        f"{row['median_total_method_seconds']:.6f}s",
                        flush=True,
                    )

    # =========================================================================
    # Final structural audits
    # =========================================================================

    expected_run_rows = (
        384
    )

    expected_checkpoints_per_run = (
        MAX_EPOCHS
        // CHECKPOINT_EPOCHS
        + 1
    )

    expected_trajectory_rows = (
        expected_run_rows
        * expected_checkpoints_per_run
    )

    if (
        pair_index
        != expected_run_rows
    ):
        raise AssertionError(
            f"Expected {expected_run_rows} method/problem pairs; "
            f"executed {pair_index}."
        )

    if (
        len(run_rows)
        != expected_run_rows
    ):
        raise AssertionError(
            f"Expected {expected_run_rows} run-result rows; "
            f"found {len(run_rows)}."
        )

    if (
        len(
            trajectory_rows
        )
        != expected_trajectory_rows
    ):
        raise AssertionError(
            f"Expected {expected_trajectory_rows} trajectory rows; "
            f"found {len(trajectory_rows)}."
        )

    # Each method/problem trace must have exactly 41 checkpoints.
    trace_counts: dict[
        tuple[Any, ...],
        int,
    ] = {}

    for row in (
        trajectory_rows
    ):
        key = (
            row["problem_id"],
            row["algorithm_seed"],
            row["method"],
        )

        trace_counts[
            key
        ] = (
            trace_counts.get(
                key,
                0,
            )
            + 1
        )

    bad_trace_counts = {
        key: value
        for key, value
        in trace_counts.items()
        if value
        != expected_checkpoints_per_run
    }

    if bad_trace_counts:
        raise AssertionError(
            "Unexpected checkpoint count for one or more runs: "
            f"{bad_trace_counts}"
        )

    # =========================================================================
    # Write canonical outputs only after all audits pass
    # =========================================================================

    completed_at = (
        utc_now_string()
    )

    write_csv_atomic(
        RUN_RESULTS_PATH,
        run_rows,
    )

    write_csv_atomic(
        TRAJECTORIES_PATH,
        trajectory_rows,
    )

    environment = (
        build_environment_record(
            manifest=manifest,
            started_at=started_at,
            completed_at=completed_at,
            run_count=len(
                run_rows
            ),
            trajectory_count=len(
                trajectory_rows
            ),
            threadpools=(
                limited_threadpool_info
            ),
        )
    )

    write_json_atomic(
        ENVIRONMENT_PATH,
        environment,
    )

    print()
    print(
        "=" * 76
    )
    print(
        "STAGE 5 CONFIRMATORY BENCHMARK: COMPLETED"
    )
    print(
        "=" * 76
    )

    print(
        f"\nMethod/problem rows: "
        f"{len(run_rows)}"
    )

    print(
        f"Trajectory rows: "
        f"{len(trajectory_rows)}"
    )

    print(
        f"Checkpoints per run: "
        f"{expected_checkpoints_per_run}"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  {RUN_RESULTS_PATH}"
    )

    print(
        f"  {TRAJECTORIES_PATH}"
    )

    print(
        f"  {ENVIRONMENT_PATH}"
    )

    print()
    print(
        "Do not interpret individual printed outcomes yet."
    )
    print(
        "Next step: run the predeclared Stage 5 analysis "
        "over the complete result set."
    )


if __name__ == "__main__":
    main()
