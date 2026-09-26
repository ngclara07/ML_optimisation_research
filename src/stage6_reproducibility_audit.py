"""Stage 6 reproducibility and provenance audit.

This script records and verifies the reproducibility state of the completed
Stages 1--5 research package and the Stage 6 release state.

It DOES NOT rerun optimization.

Primary checks
--------------
- required research artifacts exist;
- Stage 5 milestone tag still points to frozen commit 5ec3b22;
- Stage 5 summary hashes match the canonical Stage 5 artifacts;
- Stage 5 prepared-data files, when present, match their manifest hashes;
- package and numerical-library versions are recorded;
- canonical research files receive SHA-256 hashes;
- report PDF metadata/page count is recorded where possible;
- requirements-file encoding is inspected;
- Git branch/status/tags are recorded;
- final checksum manifest is verified in final mode;
- finalization summary is verified in final mode.

Output
------
Working mode writes:

    results/stage6_finalization/reproducibility_manifest.json

Modes
-----

Working mode, recording the current reproducibility state:

    python src/stage6_reproducibility_audit.py

Working non-mutating verification:

    python src/stage6_reproducibility_audit.py --verify-only

Strict final-release verification:

    python src/stage6_reproducibility_audit.py --final --verify-only

The strict final mode requires:

- branch main;
- clean Git working tree;
- README.md;
- requirements-lock.txt;
- UTF-8 dependency files;
- passing report_audit.json;
- checksums.sha256;
- finalization_summary.json;
- verified checksum entries;
- PRE_RELEASE_READY finalization summary;
- unchanged Stage 5 milestone tag.

Important
---------
Final mode is intentionally non-mutating. Therefore --final requires
--verify-only. This prevents the final audit itself from rewriting
reproducibility_manifest.json and making a clean release tree dirty.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
import warnings

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# Repository paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "stage6_finalization"
)

MANIFEST_PATH = (
    OUTPUT_ROOT
    / "reproducibility_manifest.json"
)

REPORT_AUDIT_PATH = (
    OUTPUT_ROOT
    / "report_audit.json"
)

CHECKSUMS_PATH = (
    OUTPUT_ROOT
    / "checksums.sha256"
)

FINALIZATION_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "finalization_summary.json"
)

STAGE5_SUMMARY_PATH = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "summary.json"
)

STAGE5_MANIFEST_PATH = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "dataset_manifest.json"
)

STAGE5_RUN_RESULTS_PATH = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "run_results.csv"
)

STAGE5_TRAJECTORIES_PATH = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "trajectories.csv"
)

STAGE5_ENVIRONMENT_PATH = (
    ROOT
    / "results"
    / "stage5_confirmatory"
    / "environment.json"
)

REPORT_TEX_PATH = (
    ROOT
    / "report"
    / "main.tex"
)

REPORT_PDF_PATH = (
    ROOT
    / "report"
    / "main.pdf"
)

REQUIREMENTS_PATH = (
    ROOT
    / "requirements.txt"
)

REQUIREMENTS_LOCK_PATH = (
    ROOT
    / "requirements-lock.txt"
)

README_PATH = (
    ROOT
    / "README.md"
)


# =============================================================================
# Frozen milestones / release identifiers
# =============================================================================

STAGE5_TAG = (
    "stage5-showcase-2026-09-26"
)

STAGE5_FROZEN_SHORT_COMMIT = (
    "5ec3b22"
)

FINAL_RELEASE_TAG = (
    "final-research-2026-09-26"
)


# =============================================================================
# Files required during normal Stage 6 execution
# =============================================================================

CORE_REQUIRED_FILES = (
    # -------------------------------------------------------------------------
    # Specifications
    # -------------------------------------------------------------------------
    "research/STAGE3_ASCD_METHOD_SPEC.md",
    "research/STAGE4_ORACLE_ANALYSIS_SPEC.md",
    "research/STAGE5_CONFIRMATORY_VALIDATION_SPEC.md",
    "research/STAGE6_FINALIZATION_SPEC.md",

    # -------------------------------------------------------------------------
    # Core benchmark / analysis sources
    # -------------------------------------------------------------------------
    "src/benchmark.py",
    "src/published_ascd.py",
    "src/paired_stage3_analysis.py",
    "src/stage4_active_set_diagnostics.py",
    "src/stage4_plot_diagnostics.py",
    "src/stage4_oracle_comparison.py",
    "src/stage5_prepare_data.py",
    "src/stage5_confirmatory_benchmark.py",
    "src/stage5_analyze.py",
    "src/stage5_plot.py",
    "src/stage6_report_audit.py",
    "src/stage6_reproducibility_audit.py",

    # -------------------------------------------------------------------------
    # Stage 3
    # -------------------------------------------------------------------------
    "results/published_ascd/summary.json",
    "results/published_ascd/trajectories.csv",
    "results/published_ascd/paired_scaled_correlated.csv",
    "results/published_ascd/paired_scaled_correlated_summary.txt",

    # -------------------------------------------------------------------------
    # Stage 4
    # -------------------------------------------------------------------------
    "results/stage4_oracle_analysis/zero_oracle_diagnostics.csv",
    "results/stage4_oracle_analysis/zero_oracle_summary.json",
    "results/stage4_oracle_analysis/oracle_comparison.csv",
    "results/stage4_oracle_analysis/oracle_trajectories.csv",
    "results/stage4_oracle_analysis/oracle_active_set_by_step.csv",
    "results/stage4_oracle_analysis/oracle_summary.json",
    "results/stage4_oracle_analysis/active_set_trajectories.png",
    "results/stage4_oracle_analysis/radius_tightness.png",
    "results/stage4_oracle_analysis/oracle_active_set_comparison.png",
    "results/stage4_oracle_analysis/oracle_gap_trajectories.png",

    # -------------------------------------------------------------------------
    # Stage 5
    # -------------------------------------------------------------------------
    "results/stage5_confirmatory/dataset_manifest.json",
    "results/stage5_confirmatory/environment.json",
    "results/stage5_confirmatory/run_results.csv",
    "results/stage5_confirmatory/trajectories.csv",
    "results/stage5_confirmatory/paired_results.csv",
    "results/stage5_confirmatory/summary.json",
    "results/stage5_confirmatory/timing_summary.json",
    "results/stage5_confirmatory/heldout_gap_trajectories.png",
    "results/stage5_confirmatory/real_gap_trajectories.png",
    "results/stage5_confirmatory/active_set_generalization.png",
    "results/stage5_confirmatory/runtime_comparison.png",

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    "report/main.tex",
    "report/main.pdf",

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------
    "requirements.txt",
)


FINAL_REQUIRED_FILES = (
    "README.md",
    "requirements-lock.txt",
    "results/stage6_finalization/report_audit.json",
    "results/stage6_finalization/reproducibility_manifest.json",
    "results/stage6_finalization/checksums.sha256",
    "results/stage6_finalization/finalization_summary.json",
)


# =============================================================================
# Package versions to record
# =============================================================================

PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "ucimlrepo",
    "openpyxl",
    "threadpoolctl",
)


# =============================================================================
# General helpers
# =============================================================================

def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def relative(
    path: Path,
) -> str:
    try:
        return str(
            path.relative_to(ROOT)
        ).replace(
            "\\",
            "/",
        )

    except ValueError:
        return str(
            path
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
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
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


def run_command(
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int = 20,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        return {
            "command": (
                command
            ),
            "returncode": (
                result.returncode
            ),
            "stdout": (
                result.stdout.strip()
            ),
            "stderr": (
                result.stderr.strip()
            ),
        }

    except Exception as exc:
        return {
            "command": (
                command
            ),
            "returncode": None,
            "stdout": "",
            "stderr": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        }


def git(
    *arguments: str,
) -> str | None:
    result = run_command(
        [
            "git",
            *arguments,
        ]
    )

    if (
        result[
            "returncode"
        ]
        != 0
    ):
        return None

    return result[
        "stdout"
    ]


# =============================================================================
# Audit check recorder
# =============================================================================

class CheckRecorder:
    def __init__(
        self,
    ) -> None:
        self.checks: list[
            dict[str, Any]
        ] = []

        self.warnings: list[
            dict[str, Any]
        ] = []

    def check(
        self,
        check_id: str,
        condition: bool,
        *,
        expected: Any,
        observed: Any,
        source: str | Path,
        note: str = "",
    ) -> None:
        self.checks.append(
            {
                "id": (
                    check_id
                ),
                "status": (
                    "PASSED"
                    if condition
                    else "FAILED"
                ),
                "expected": (
                    expected
                ),
                "observed": (
                    observed
                ),
                "source": (
                    relative(
                        source
                    )
                    if isinstance(
                        source,
                        Path,
                    )
                    else source
                ),
                "note": (
                    note
                ),
            }
        )

    def warn(
        self,
        warning_id: str,
        *,
        observed: Any,
        note: str,
    ) -> None:
        self.warnings.append(
            {
                "id": (
                    warning_id
                ),
                "observed": (
                    observed
                ),
                "note": (
                    note
                ),
            }
        )

    @property
    def failed(
        self,
    ) -> list[dict[str, Any]]:
        return [
            row
            for row
            in self.checks
            if row[
                "status"
            ] == "FAILED"
        ]


# =============================================================================
# Text-encoding audit
# =============================================================================

def detect_text_encoding(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "encoding": None,
            "utf8_valid": False,
            "has_utf8_bom": False,
            "has_utf16_bom": False,
        }

    raw = path.read_bytes()

    has_utf8_bom = (
        raw.startswith(
            b"\xef\xbb\xbf"
        )
    )

    has_utf16_bom = (
        raw.startswith(
            b"\xff\xfe"
        )
        or raw.startswith(
            b"\xfe\xff"
        )
    )

    encoding = None
    utf8_valid = False

    try:
        raw.decode(
            "utf-8"
        )

        encoding = (
            "utf-8-sig"
            if has_utf8_bom
            else "utf-8"
        )

        utf8_valid = True

    except UnicodeDecodeError:
        try:
            raw.decode(
                "utf-16"
            )

            encoding = (
                "utf-16"
            )

        except UnicodeDecodeError:
            encoding = (
                "unknown"
            )

    return {
        "exists": True,
        "encoding": (
            encoding
        ),
        "utf8_valid": (
            utf8_valid
        ),
        "has_utf8_bom": (
            has_utf8_bom
        ),
        "has_utf16_bom": (
            has_utf16_bom
        ),
        "size_bytes": (
            len(
                raw
            )
        ),
    }


# =============================================================================
# NumPy / threadpool information
# =============================================================================

def numpy_information() -> dict[str, Any]:
    try:
        import numpy as np

    except ImportError:
        return {
            "available": False,
        }

    stream = io.StringIO()

    # NumPy may emit:
    #
    #   UserWarning: Install `pyyaml` for better output
    #
    # from np.__config__.show(). This warning concerns formatting of
    # diagnostic output only and has no effect on the numerical environment.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Install `pyyaml` for better output"
            ),
            category=UserWarning,
        )

        with contextlib.redirect_stdout(
            stream
        ):
            try:
                np.__config__.show()

            except Exception as exc:
                print(
                    f"NumPy config unavailable: "
                    f"{exc}"
                )

    return {
        "available": True,
        "version": (
            np.__version__
        ),
        "configuration": (
            stream
            .getvalue()
            .strip()
        ),
    }


def threadpool_information() -> dict[str, Any]:
    try:
        from threadpoolctl import (
            threadpool_info,
        )

        return {
            "available": True,
            "threadpools": (
                threadpool_info()
            ),
        }

    except Exception as exc:
        return {
            "available": False,
            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        }


# =============================================================================
# PDF page counting
# =============================================================================

def pdf_page_count(
    path: Path,
) -> dict[str, Any]:
    """Return the PDF page count using optional readers or safe fallbacks.

    The Python PDF libraries are intentionally loaded dynamically so that
    pypdf and PyPDF2 remain optional dependencies. This avoids requiring
    either package solely for Stage 6 provenance auditing.
    """

    if not path.is_file():
        return {
            "pages": None,
            "method": None,
        }

    # -------------------------------------------------------------------------
    # First preference: optional Python PDF readers
    # -------------------------------------------------------------------------

    for module_name in (
        "pypdf",
        "PyPDF2",
    ):
        try:
            module = (
                importlib.import_module(
                    module_name
                )
            )

            pdf_reader = getattr(
                module,
                "PdfReader",
            )

            reader = pdf_reader(
                str(
                    path
                )
            )

            return {
                "pages": (
                    len(
                        reader.pages
                    )
                ),
                "method": (
                    module_name
                ),
            }

        except Exception:
            # The package may be absent, incompatible, or unable to read
            # the particular PDF. Continue to the next supported method.
            pass

    # -------------------------------------------------------------------------
    # Second preference: pdfinfo executable
    # -------------------------------------------------------------------------

    result = run_command(
        [
            "pdfinfo",
            str(
                path
            ),
        ]
    )

    if (
        result[
            "returncode"
        ]
        == 0
    ):
        match = re.search(
            r"^Pages:\s+(\d+)",
            result[
                "stdout"
            ],
            flags=re.MULTILINE,
        )

        if match:
            return {
                "pages": (
                    int(
                        match.group(
                            1
                        )
                    )
                ),
                "method": (
                    "pdfinfo"
                ),
            }

    # -------------------------------------------------------------------------
    # Last-resort structural approximation
    # -------------------------------------------------------------------------

    try:
        data = path.read_bytes()

    except OSError:
        return {
            "pages": None,
            "method": None,
        }

    matches = re.findall(
        rb"/Type\s*/Page\b",
        data,
    )

    if matches:
        return {
            "pages": (
                len(
                    matches
                )
            ),
            "method": (
                "raw_pdf_page_object_count"
            ),
        }

    return {
        "pages": None,
        "method": None,
    }


# =============================================================================
# Hash collection
# =============================================================================

def collect_file_hashes(
    paths: list[Path],
) -> dict[str, Any]:
    result: dict[
        str,
        Any,
    ] = {}

    for path in sorted(
        paths,
        key=lambda item: (
            relative(
                item
            )
        ),
    ):
        key = relative(
            path
        )

        if not path.is_file():
            result[
                key
            ] = {
                "exists": False,
            }

            continue

        result[
            key
        ] = {
            "exists": True,
            "size_bytes": (
                path.stat().st_size
            ),
            "sha256": (
                sha256_file(
                    path
                )
            ),
        }

    return result


# =============================================================================
# Stage 5 hash-chain audit
# =============================================================================

def verify_stage5_hash_chain(
    recorder: CheckRecorder,
) -> dict[str, Any]:
    summary = read_json(
        STAGE5_SUMMARY_PATH
    )

    audit = summary[
        "input_audit"
    ]

    expected = {
        "dataset_manifest": (
            audit[
                "dataset_manifest_sha256"
            ],
            STAGE5_MANIFEST_PATH,
        ),

        "run_results": (
            audit[
                "run_results_sha256"
            ],
            STAGE5_RUN_RESULTS_PATH,
        ),

        "trajectories": (
            audit[
                "trajectories_sha256"
            ],
            STAGE5_TRAJECTORIES_PATH,
        ),

        "environment": (
            audit[
                "environment_sha256"
            ],
            STAGE5_ENVIRONMENT_PATH,
        ),
    }

    output: dict[
        str,
        Any,
    ] = {}

    for (
        name,
        (
            expected_hash,
            path,
        ),
    ) in expected.items():

        exists = (
            path.is_file()
        )

        observed_hash = (
            sha256_file(
                path
            )
            if exists
            else None
        )

        matches = (
            exists
            and observed_hash
            == expected_hash
        )

        recorder.check(
            (
                "stage5_hash_chain:"
                + name
            ),
            matches,
            expected=(
                expected_hash
            ),
            observed=(
                observed_hash
            ),
            source=(
                path
            ),
        )

        output[
            name
        ] = {
            "path": (
                relative(
                    path
                )
            ),
            "expected_sha256": (
                expected_hash
            ),
            "observed_sha256": (
                observed_hash
            ),
            "matches": (
                matches
            ),
        }

    return output


# =============================================================================
# Prepared-data optional verification
# =============================================================================

def verify_prepared_data(
    recorder: CheckRecorder,
) -> dict[str, Any]:
    manifest = read_json(
        STAGE5_MANIFEST_PATH
    )

    records = (
        list(
            manifest[
                "real_datasets"
            ]
        )
        + list(
            manifest[
                "synthetic_datasets"
            ]
        )
    )

    present = 0
    absent = 0
    verified = 0

    mismatched: list[
        str
    ] = []

    for record in records:
        relative_path = (
            record.get(
                "prepared_file"
            )
        )

        expected_hash = (
            record.get(
                "prepared_file_sha256"
            )
        )

        if not relative_path:
            continue

        path = (
            ROOT
            / relative_path
        )

        if not path.is_file():
            absent += 1
            continue

        present += 1

        observed = (
            sha256_file(
                path
            )
        )

        if (
            expected_hash
            == observed
        ):
            verified += 1

        else:
            mismatched.append(
                relative_path
            )

    recorder.check(
        "stage5_prepared_data_no_hash_mismatches",
        (
            len(
                mismatched
            )
            == 0
        ),
        expected=[],
        observed=(
            mismatched
        ),
        source=(
            STAGE5_MANIFEST_PATH
        ),
        note=(
            "Prepared .npz files are regenerable and may be "
            "absent because they are intentionally Git-ignored. "
            "Any file that is present must match its manifest hash."
        ),
    )

    return {
        "records_in_manifest": (
            len(
                records
            )
        ),
        "present_locally": (
            present
        ),
        "absent_regenerable": (
            absent
        ),
        "verified_hash_matches": (
            verified
        ),
        "hash_mismatches": (
            mismatched
        ),
    }


# =============================================================================
# Git provenance
# =============================================================================

def collect_git_state(
    recorder: CheckRecorder,
    *,
    final_mode: bool,
) -> dict[str, Any]:
    head = git(
        "rev-parse",
        "HEAD",
    )

    short_head = git(
        "rev-parse",
        "--short=7",
        "HEAD",
    )

    branch = git(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )

    status = (
        git(
            "status",
            "--porcelain",
        )
        or ""
    )

    tags = (
        git(
            "tag",
            "--list",
        )
        or ""
    ).splitlines()

    stage5_tag_commit = git(
        "rev-parse",
        "--short=7",
        (
            STAGE5_TAG
            + "^{commit}"
        ),
    )

    final_tag_commit = git(
        "rev-parse",
        "--short=7",
        (
            FINAL_RELEASE_TAG
            + "^{commit}"
        ),
    )

    recorder.check(
        "git_branch_main",
        (
            branch
            == "main"
        ),
        expected=(
            "main"
        ),
        observed=(
            branch
        ),
        source=(
            "git"
        ),
    )

    recorder.check(
        "stage5_tag_exists",
        (
            STAGE5_TAG
            in tags
        ),
        expected=True,
        observed=(
            STAGE5_TAG
            in tags
        ),
        source=(
            "git"
        ),
    )

    recorder.check(
        "stage5_tag_still_frozen",
        (
            stage5_tag_commit
            == STAGE5_FROZEN_SHORT_COMMIT
        ),
        expected=(
            STAGE5_FROZEN_SHORT_COMMIT
        ),
        observed=(
            stage5_tag_commit
        ),
        source=(
            "git"
        ),
        note=(
            "The Stage 5 milestone tag must never move."
        ),
    )

    clean = (
        status.strip()
        == ""
    )

    if final_mode:
        recorder.check(
            "git_working_tree_clean",
            clean,
            expected=True,
            observed=(
                clean
            ),
            source=(
                "git"
            ),
        )

    elif not clean:
        recorder.warn(
            "git_working_tree_not_clean",
            observed=(
                status.splitlines()
            ),
            note=(
                "Expected during active Stage 6 development. "
                "The strict --final --verify-only audit "
                "will require a clean working tree."
            ),
        )

    return {
        "head": (
            head
        ),
        "short_head": (
            short_head
        ),
        "branch": (
            branch
        ),
        "status_porcelain": (
            status.splitlines()
            if status
            else []
        ),
        "clean": (
            clean
        ),
        "tags": (
            tags
        ),
        "stage5_tag": (
            STAGE5_TAG
        ),
        "stage5_tag_commit": (
            stage5_tag_commit
        ),
        "planned_final_release_tag": (
            FINAL_RELEASE_TAG
        ),
        "final_release_tag_commit_if_present": (
            final_tag_commit
        ),
    }


# =============================================================================
# Required-file audit
# =============================================================================

def audit_required_files(
    recorder: CheckRecorder,
    *,
    final_mode: bool,
) -> list[Path]:
    required_names = list(
        CORE_REQUIRED_FILES
    )

    if final_mode:
        required_names.extend(
            FINAL_REQUIRED_FILES
        )

    paths: list[
        Path
    ] = []

    for name in required_names:
        path = (
            ROOT
            / name
        )

        paths.append(
            path
        )

        recorder.check(
            (
                "required_file:"
                + name
            ),
            path.is_file(),
            expected=True,
            observed=(
                path.is_file()
            ),
            source=(
                path
            ),
        )

    return paths


# =============================================================================
# Report-audit linkage
# =============================================================================

def audit_report_audit(
    recorder: CheckRecorder,
) -> dict[str, Any] | None:
    if not REPORT_AUDIT_PATH.is_file():
        recorder.check(
            "stage6_report_audit_exists",
            False,
            expected=True,
            observed=False,
            source=(
                REPORT_AUDIT_PATH
            ),
        )

        return None

    payload = read_json(
        REPORT_AUDIT_PATH
    )

    status = payload.get(
        "overall_status"
    )

    failed = payload.get(
        "checks_failed"
    )

    recorder.check(
        "stage6_report_audit_passed",
        (
            status
            == "PASSED"
            and failed
            == 0
        ),
        expected={
            "overall_status": "PASSED",
            "checks_failed": 0,
        },
        observed={
            "overall_status": (
                status
            ),
            "checks_failed": (
                failed
            ),
        },
        source=(
            REPORT_AUDIT_PATH
        ),
    )

    return {
        "path": (
            relative(
                REPORT_AUDIT_PATH
            )
        ),
        "sha256": (
            sha256_file(
                REPORT_AUDIT_PATH
            )
        ),
        "overall_status": (
            status
        ),
        "checks_total": (
            payload.get(
                "checks_total"
            )
        ),
        "checks_passed": (
            payload.get(
                "checks_passed"
            )
        ),
        "checks_failed": (
            failed
        ),
    }


# =============================================================================
# Final checksum verification
# =============================================================================

def verify_checksum_manifest(
    recorder: CheckRecorder,
) -> dict[str, Any]:
    if not CHECKSUMS_PATH.is_file():
        recorder.check(
            "final_checksum_manifest_exists",
            False,
            expected=True,
            observed=False,
            source=(
                CHECKSUMS_PATH
            ),
        )

        return {
            "exists": False,
            "entries": 0,
            "verified": False,
            "failures": [
                "checksums.sha256 is missing"
            ],
        }

    lines = [
        line
        for line
        in CHECKSUMS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    failures: list[
        dict[str, Any]
    ] = []

    observed_paths: set[
        str
    ] = set()

    checksum_pattern = re.compile(
        r"^[0-9a-fA-F]{64}$"
    )

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        try:
            expected_hash, relative_path = (
                line.split(
                    "  ",
                    1,
                )
            )

        except ValueError:
            failures.append(
                {
                    "line": (
                        line_number
                    ),
                    "reason": (
                        "malformed entry"
                    ),
                    "text": (
                        line
                    ),
                }
            )
            continue

        if not checksum_pattern.fullmatch(
            expected_hash
        ):
            failures.append(
                {
                    "line": (
                        line_number
                    ),
                    "reason": (
                        "invalid SHA-256 digest"
                    ),
                    "path": (
                        relative_path
                    ),
                }
            )
            continue

        normalized = (
            relative_path
            .replace(
                "\\",
                "/",
            )
        )

        if normalized in observed_paths:
            failures.append(
                {
                    "line": (
                        line_number
                    ),
                    "reason": (
                        "duplicate path"
                    ),
                    "path": (
                        normalized
                    ),
                }
            )
            continue

        observed_paths.add(
            normalized
        )

        # Prevent a self-referential checksum manifest.
        if normalized == relative(
            CHECKSUMS_PATH
        ):
            failures.append(
                {
                    "line": (
                        line_number
                    ),
                    "reason": (
                        "checksum manifest must "
                        "not hash itself"
                    ),
                    "path": (
                        normalized
                    ),
                }
            )
            continue

        path = (
            ROOT
            / normalized
        )

        if not path.is_file():
            failures.append(
                {
                    "line": (
                        line_number
                    ),
                    "reason": (
                        "missing file"
                    ),
                    "path": (
                        normalized
                    ),
                }
            )
            continue

        observed_hash = (
            sha256_file(
                path
            )
        )

        if (
            observed_hash.lower()
            != expected_hash.lower()
        ):
            failures.append(
                {
                    "line": (
                        line_number
                    ),
                    "reason": (
                        "hash mismatch"
                    ),
                    "path": (
                        normalized
                    ),
                    "expected": (
                        expected_hash.lower()
                    ),
                    "observed": (
                        observed_hash.lower()
                    ),
                }
            )

    verified = (
        len(
            failures
        )
        == 0
    )

    recorder.check(
        "final_checksum_manifest_verified",
        verified,
        expected=True,
        observed={
            "entries": (
                len(
                    lines
                )
            ),
            "failures": (
                failures
            ),
        },
        source=(
            CHECKSUMS_PATH
        ),
    )

    return {
        "exists": True,
        "entries": (
            len(
                lines
            )
        ),
        "verified": (
            verified
        ),
        "failures": (
            failures
        ),
        "sha256": (
            sha256_file(
                CHECKSUMS_PATH
            )
        ),
    }


# =============================================================================
# Finalization-summary verification
# =============================================================================

def audit_finalization_summary(
    recorder: CheckRecorder,
) -> dict[str, Any] | None:
    if not FINALIZATION_SUMMARY_PATH.is_file():
        recorder.check(
            "finalization_summary_exists",
            False,
            expected=True,
            observed=False,
            source=(
                FINALIZATION_SUMMARY_PATH
            ),
        )

        return None

    payload = read_json(
        FINALIZATION_SUMMARY_PATH
    )

    overall_status = (
        payload.get(
            "overall_status"
        )
    )

    pre_release_ready = (
        payload.get(
            "pre_release_ready"
        )
    )

    recorder.check(
        "finalization_summary_pre_release_ready",
        (
            overall_status
            == "PRE_RELEASE_READY"
            and pre_release_ready
            is True
        ),
        expected={
            "overall_status": (
                "PRE_RELEASE_READY"
            ),
            "pre_release_ready": True,
        },
        observed={
            "overall_status": (
                overall_status
            ),
            "pre_release_ready": (
                pre_release_ready
            ),
        },
        source=(
            FINALIZATION_SUMMARY_PATH
        ),
    )

    checksum_verified = (
        payload
        .get(
            "checksum_manifest",
            {},
        )
        .get(
            "verified"
        )
    )

    recorder.check(
        "finalization_summary_checksum_verified",
        (
            checksum_verified
            is True
        ),
        expected=True,
        observed=(
            checksum_verified
        ),
        source=(
            FINALIZATION_SUMMARY_PATH
        ),
    )

    visual_review = (
        payload
        .get(
            "report",
            {},
        )
        .get(
            "author_page_by_page_visual_review_confirmed"
        )
    )

    recorder.check(
        "finalization_summary_pdf_review_confirmed",
        (
            visual_review
            is True
        ),
        expected=True,
        observed=(
            visual_review
        ),
        source=(
            FINALIZATION_SUMMARY_PATH
        ),
    )

    return {
        "path": (
            relative(
                FINALIZATION_SUMMARY_PATH
            )
        ),
        "sha256": (
            sha256_file(
                FINALIZATION_SUMMARY_PATH
            )
        ),
        "overall_status": (
            overall_status
        ),
        "pre_release_ready": (
            pre_release_ready
        ),
        "checksum_verified": (
            checksum_verified
        ),
        "pdf_review_confirmed": (
            visual_review
        ),
    }


# =============================================================================
# Environment
# =============================================================================

def collect_environment() -> dict[str, Any]:
    packages = {
        name: package_version(
            name
        )
        for name
        in PACKAGE_NAMES
    }

    return {
        "python": {
            "version": (
                sys.version
            ),
            "version_info": {
                "major": (
                    sys.version_info.major
                ),
                "minor": (
                    sys.version_info.minor
                ),
                "micro": (
                    sys.version_info.micro
                ),
            },
            "executable": (
                sys.executable
            ),
        },

        "platform": {
            "platform": (
                platform.platform()
            ),
            "system": (
                platform.system()
            ),
            "release": (
                platform.release()
            ),
            "version": (
                platform.version()
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

        "packages": (
            packages
        ),

        "numpy": (
            numpy_information()
        ),

        "threadpool": (
            threadpool_information()
        ),
    }


# =============================================================================
# Requirements audit
# =============================================================================

def audit_requirements(
    recorder: CheckRecorder,
    *,
    final_mode: bool,
) -> dict[str, Any]:
    direct = detect_text_encoding(
        REQUIREMENTS_PATH
    )

    lock = detect_text_encoding(
        REQUIREMENTS_LOCK_PATH
    )

    recorder.check(
        "requirements_exists",
        (
            direct[
                "exists"
            ]
        ),
        expected=True,
        observed=(
            direct[
                "exists"
            ]
        ),
        source=(
            REQUIREMENTS_PATH
        ),
    )

    if final_mode:
        recorder.check(
            "requirements_utf8",
            (
                direct[
                    "utf8_valid"
                ]
            ),
            expected=True,
            observed=(
                direct
            ),
            source=(
                REQUIREMENTS_PATH
            ),
        )

        recorder.check(
            "requirements_lock_exists",
            (
                lock[
                    "exists"
                ]
            ),
            expected=True,
            observed=(
                lock[
                    "exists"
                ]
            ),
            source=(
                REQUIREMENTS_LOCK_PATH
            ),
        )

        recorder.check(
            "requirements_lock_utf8",
            (
                lock[
                    "utf8_valid"
                ]
            ),
            expected=True,
            observed=(
                lock
            ),
            source=(
                REQUIREMENTS_LOCK_PATH
            ),
        )

    else:
        if (
            direct[
                "exists"
            ]
            and not direct[
                "utf8_valid"
            ]
        ):
            recorder.warn(
                "requirements_not_yet_utf8",
                observed=(
                    direct
                ),
                note=(
                    "Stage 6 dependency cleanup has not yet "
                    "been completed. Normalize this file before "
                    "the final --final --verify-only audit."
                ),
            )

        if not lock[
            "exists"
        ]:
            recorder.warn(
                "requirements_lock_not_yet_created",
                observed=False,
                note=(
                    "Create requirements-lock.txt before "
                    "the final release audit."
                ),
            )

    return {
        "requirements": (
            direct
        ),
        "requirements_lock": (
            lock
        ),
    }


# =============================================================================
# Command-line interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and record reproducibility provenance "
            "for the completed optimization research project."
        )
    )

    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Enable strict final-release checks. "
            "This mode requires --verify-only so that the "
            "audit does not modify the clean release tree."
        ),
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Run all applicable checks but do not rewrite "
            "reproducibility_manifest.json."
        ),
    )

    args = parser.parse_args()

    if (
        args.final
        and not args.verify_only
    ):
        parser.error(
            "--final requires --verify-only. "
            "The strict final audit must be non-mutating."
        )

    return args


# =============================================================================
# Manifest writing
# =============================================================================

def write_manifest(
    payload: dict[str, Any],
) -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        MANIFEST_PATH
        .with_suffix(
            ".json.tmp"
        )
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

    # Independently parse the generated JSON before replacing the
    # canonical artifact.
    json.loads(
        temporary.read_text(
            encoding="utf-8"
        )
    )

    temporary.replace(
        MANIFEST_PATH
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # In verification-only mode we intentionally avoid creating directories
    # or files. A release verification must not alter the repository.
    if not args.verify_only:
        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "=" * 78
    )
    print(
        "STAGE 6 REPRODUCIBILITY / PROVENANCE AUDIT"
    )
    print(
        "=" * 78
    )

    if args.final:
        mode_label = (
            "FINAL RELEASE / VERIFY ONLY"
        )

    elif args.verify_only:
        mode_label = (
            "STAGE 6 WORKING / VERIFY ONLY"
        )

    else:
        mode_label = (
            "STAGE 6 WORKING"
        )

    print(
        f"\nMode: "
        f"{mode_label}"
    )

    recorder = CheckRecorder()

    # -------------------------------------------------------------------------
    # 1. Required files
    # -------------------------------------------------------------------------

    print(
        "\n[1/8] Required artifact audit"
    )

    required_paths = (
        audit_required_files(
            recorder,
            final_mode=(
                args.final
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 2. Git
    # -------------------------------------------------------------------------

    print(
        "\n[2/8] Git provenance audit"
    )

    git_state = (
        collect_git_state(
            recorder,
            final_mode=(
                args.final
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 3. Stage 6 report audit
    # -------------------------------------------------------------------------

    print(
        "\n[3/8] Stage 6 report-audit linkage"
    )

    report_audit = (
        audit_report_audit(
            recorder
        )
    )

    # -------------------------------------------------------------------------
    # 4. Stage 5 frozen hash chain
    # -------------------------------------------------------------------------

    print(
        "\n[4/8] Stage 5 canonical hash chain"
    )

    stage5_hash_chain = (
        verify_stage5_hash_chain(
            recorder
        )
    )

    # -------------------------------------------------------------------------
    # 5. Stage 5 prepared-data provenance
    # -------------------------------------------------------------------------

    print(
        "\n[5/8] Stage 5 prepared-data provenance"
    )

    prepared_data = (
        verify_prepared_data(
            recorder
        )
    )

    # -------------------------------------------------------------------------
    # 6. Dependencies
    # -------------------------------------------------------------------------

    print(
        "\n[6/8] Dependency-file audit"
    )

    requirements = (
        audit_requirements(
            recorder,
            final_mode=(
                args.final
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 7. Environment
    # -------------------------------------------------------------------------

    print(
        "\n[7/8] Environment capture"
    )

    environment = (
        collect_environment()
    )

    # -------------------------------------------------------------------------
    # 8. Canonical files / final release checks
    # -------------------------------------------------------------------------

    print(
        "\n[8/8] Canonical file hashing"
    )

    checksum_audit: dict[
        str,
        Any,
    ] | None = None

    finalization_summary: dict[
        str,
        Any,
    ] | None = None

    if args.final:
        checksum_audit = (
            verify_checksum_manifest(
                recorder
            )
        )

        finalization_summary = (
            audit_finalization_summary(
                recorder
            )
        )

    hash_paths = list(
        required_paths
    )

    # The report audit is included in working mode when it already exists.
    if REPORT_AUDIT_PATH.is_file():
        hash_paths.append(
            REPORT_AUDIT_PATH
        )

    # Include README and lock in working mode as soon as they exist.
    for optional_path in (
        README_PATH,
        REQUIREMENTS_LOCK_PATH,
    ):
        if optional_path.is_file():
            hash_paths.append(
                optional_path
            )

    # Final artifacts are included in the live hash inventory when present.
    for final_path in (
        CHECKSUMS_PATH,
        FINALIZATION_SUMMARY_PATH,
    ):
        if final_path.is_file():
            hash_paths.append(
                final_path
            )

    # Remove duplicates deterministically.
    unique_hash_paths = {
        path.resolve(): (
            path
        )
        for path in hash_paths
    }

    file_hashes = (
        collect_file_hashes(
            list(
                unique_hash_paths.values()
            )
        )
    )

    pdf_info = (
        pdf_page_count(
            REPORT_PDF_PATH
        )
    )

    if (
        pdf_info[
            "pages"
        ]
        is None
    ):
        recorder.warn(
            "pdf_page_count_unavailable",
            observed=(
                pdf_info
            ),
            note=(
                "Could not determine report page count "
                "using installed PDF readers, pdfinfo, or "
                "raw PDF page-object counting."
            ),
        )

    failures = (
        recorder.failed
    )

    payload = {
        "stage": 6,

        "audit": (
            "reproducibility and provenance"
        ),

        "mode": (
            "final"
            if args.final
            else "working"
        ),

        "verify_only": (
            args.verify_only
        ),

        "generated_at_utc": (
            utc_now()
        ),

        "overall_status": (
            "PASSED"
            if not failures
            else "FAILED"
        ),

        "checks_total": (
            len(
                recorder.checks
            )
        ),

        "checks_passed": (
            len(
                recorder.checks
            )
            - len(
                failures
            )
        ),

        "checks_failed": (
            len(
                failures
            )
        ),

        "warnings_count": (
            len(
                recorder.warnings
            )
        ),

        "git": (
            git_state
        ),

        "environment": (
            environment
        ),

        "requirements": (
            requirements
        ),

        "report": {
            "tex_path": (
                relative(
                    REPORT_TEX_PATH
                )
            ),

            "pdf_path": (
                relative(
                    REPORT_PDF_PATH
                )
            ),

            "pdf_pages": (
                pdf_info[
                    "pages"
                ]
            ),

            "pdf_page_count_method": (
                pdf_info[
                    "method"
                ]
            ),

            "main_tex_sha256": (
                sha256_file(
                    REPORT_TEX_PATH
                )
                if REPORT_TEX_PATH.is_file()
                else None
            ),

            "main_pdf_sha256": (
                sha256_file(
                    REPORT_PDF_PATH
                )
                if REPORT_PDF_PATH.is_file()
                else None
            ),
        },

        "stage6_report_audit": (
            report_audit
        ),

        "stage5_hash_chain": (
            stage5_hash_chain
        ),

        "stage5_prepared_data": (
            prepared_data
        ),

        "final_checksum_audit": (
            checksum_audit
        ),

        "finalization_summary": (
            finalization_summary
        ),

        "canonical_file_hashes": (
            file_hashes
        ),

        "checks": (
            recorder.checks
        ),

        "warnings": (
            recorder.warnings
        ),

        "write_policy": {
            "manifest_path": (
                relative(
                    MANIFEST_PATH
                )
            ),
            "manifest_rewritten": (
                not args.verify_only
            ),
            "note": (
                "Strict final verification is non-mutating."
            ),
        },
    }

    # -------------------------------------------------------------------------
    # Write only in recording mode.
    # -------------------------------------------------------------------------

    if not args.verify_only:
        write_manifest(
            payload
        )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

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
        f"Warnings     : "
        f"{payload['warnings_count']}"
    )

    print(
        f"Overall      : "
        f"{payload['overall_status']}"
    )

    print()
    print(
        "Output:"
    )

    if args.verify_only:
        print(
            "  verification-only mode; "
            "reproducibility_manifest.json was NOT rewritten"
        )

    else:
        print(
            f"  {MANIFEST_PATH}"
        )

    if recorder.warnings:
        print()
        print(
            "WARNINGS:"
        )

        for warning in (
            recorder.warnings
        ):
            print(
                "  - "
                f"{warning['id']}: "
                f"{warning['note']}"
            )

    if failures:
        print()
        print(
            "FAILED CHECKS:"
        )

        for failure in failures:
            print(
                "  - "
                f"{failure['id']}: "
                f"expected="
                f"{failure['expected']!r}, "
                f"observed="
                f"{failure['observed']!r}"
            )

        raise SystemExit(
            1
        )

    print()
    print(
        "STAGE 6 REPRODUCIBILITY AUDIT: PASSED"
    )


if __name__ == "__main__":
    main()
