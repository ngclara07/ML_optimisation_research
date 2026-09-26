# Stage 6: Final Research Synthesis, Reproducibility, and Release Audit

Date: 26 September 2026

Status: **predeclared finalization protocol**

Stage 5 frozen commit:

`5ec3b22 Complete Stage 5 confirmatory validation`

Stage 5 milestone tag:

`stage5-showcase-2026-09-26`

Primary branch:

`main`

---

# 1. Purpose

Stage 6 is the final mandatory stage of the independent research project.

Its purpose is not to introduce another optimization method, tune an existing
method, add new datasets, change the experimental protocol, or search for more
favorable empirical results.

The purpose of Stage 6 is to convert the completed experimental research from
Stages 1--5 into a final, auditable, reproducible, publication-oriented research
package.

The central Stage 6 question is:

> Can every important scientific claim in the final report be traced to a
> canonical experiment artifact, reproduced from documented code and
> environment information, and released in a repository state that another
> researcher can inspect without requiring access to the development chat
> history?

Stage 6 therefore focuses on:

1. numerical claim verification;
2. experiment provenance;
3. environment reproducibility;
4. documentation;
5. report quality assurance;
6. release integrity;
7. final repository freezing.

No new confirmatory scientific result is required for successful completion of
Stage 6.

---

# 2. Starting point

Stages 1--5 are treated as completed experimental work.

The principal completed research layers are:

- baseline coordinate-descent experiments;
- published ASCD zero-oracle implementation and comparison;
- Stage 3 paired scaled-correlated analysis;
- Stage 4 zero-oracle mechanism diagnostics;
- Stage 4 oracle-quality comparison;
- Stage 5 held-out confirmatory validation;
- Stage 5 harmonized runtime comparison;
- Stage 5 real-data validation;
- integration of the Stage 5 evidence into the research report.

The Stage 5 milestone is frozen at:

`5ec3b22`

with annotated tag:

`stage5-showcase-2026-09-26`.

Stage 6 must not rewrite the empirical history represented by that tag.

If a Stage 6 audit discovers an actual technical or reporting error, the error
must be documented explicitly and corrected transparently. It must not be
silently changed.

---

# 3. Final research conclusion being protected

The final report currently supports the following research-level conclusion:

> Oracle information quality strongly determines whether the ASCD active-set
> mechanism can prune coordinates, but whether the additional information pays
> for itself computationally depends on the problem regime and the cost of
> obtaining that information.

This conclusion is narrower than either of the following claims:

- adaptive coordinate selection is generally superior;
- exact-oracle ASCD is generally faster.

Neither stronger claim is supported by the completed experiments.

Stage 6 must preserve this distinction.

---

# 4. Scope of Stage 6

Stage 6 contains five finalization phases.

## Phase A: numerical report audit

Programmatically verify the major numerical statements appearing in the report
against the canonical result artifacts.

## Phase B: reproducibility and environment audit

Record the exact repository, software, platform, numerical-library, dataset,
and result-artifact state required to reproduce or inspect the work.

## Phase C: documentation and dependency freeze

Produce complete repository-level reproduction instructions and normalize the
dependency specification.

## Phase D: final report quality assurance

Perform final LaTeX, layout, reference, figure, table, terminology, and claim
audits.

## Phase E: release manifest and repository freeze

Generate final checksums and provenance records, commit the completed research
package, and create the final research tag.

---

# 5. Explicit non-goals

The following are outside the mandatory Stage 6 scope:

- adding another optimization algorithm;
- changing the ridge objective;
- changing lambda;
- changing Stage 5 success thresholds;
- extending the Stage 5 coordinate-epoch budget;
- changing Stage 5 seeds;
- adding or removing Stage 5 datasets based on observed outcomes;
- changing Stage 5 timing results;
- changing the ASCD oracle definitions;
- introducing another g2 precision grid;
- post-hoc tuning of synthetic correlation or scaling parameters;
- creating a new heuristic sampling method;
- rerunning experiments merely because an observed result is inconvenient;
- conducting a new publication experiment without a new protocol.

Such work would constitute a separate Stage 7 methodological extension.

---

# 6. Canonical research artifacts

Stage 6 treats the following outputs as canonical experimental evidence.

## Stage 3

Primary directory:

`results/published_ascd/`

Important artifacts include:

- `summary.json`;
- `trajectories.csv`;
- `paired_scaled_correlated.csv`;
- `paired_scaled_correlated_summary.txt`.

Associated specification:

`research/STAGE3_ASCD_METHOD_SPEC.md`

---

## Stage 4

Primary directory:

`results/stage4_oracle_analysis/`

Important artifacts include:

- `zero_oracle_diagnostics.csv`;
- `zero_oracle_summary.json`;
- `oracle_comparison.csv`;
- `oracle_trajectories.csv`;
- `oracle_active_set_by_step.csv`;
- `oracle_summary.json`;
- `active_set_trajectories.png`;
- `radius_tightness.png`;
- `oracle_active_set_comparison.png`;
- `oracle_gap_trajectories.png`.

Associated specification:

`research/STAGE4_ORACLE_ANALYSIS_SPEC.md`

---

## Stage 5

Primary directory:

`results/stage5_confirmatory/`

Important canonical artifacts include:

- `dataset_manifest.json`;
- `environment.json`;
- `run_results.csv`;
- `trajectories.csv`;
- `paired_results.csv`;
- `summary.json`;
- `timing_summary.json`;
- `heldout_gap_trajectories.png`;
- `real_gap_trajectories.png`;
- `active_set_generalization.png`;
- `runtime_comparison.png`.

Associated specification:

`research/STAGE5_CONFIRMATORY_VALIDATION_SPEC.md`

Stage 5 regenerable binary prepared datasets and downloaded caches are not
required to be committed when the preparation script and manifest provide a
reproducible route to reconstruction.

---

# 7. Canonical report

The research report source is:

`report/main.tex`

The compiled report is:

`report/main.pdf`

The Stage 6 report audit must treat `main.tex` as the authoritative source for
reported claims and `main.pdf` as the final publication artifact.

A successful Stage 6 report audit must verify that the final PDF is generated
from the committed `main.tex`.

---

# 8. Stage 6 implementation files

The planned Stage 6 scripts are:

`src/stage6_report_audit.py`

and

`src/stage6_reproducibility_audit.py`.

The planned Stage 6 output directory is:

`results/stage6_finalization/`

with the following expected artifacts:

```text
results/stage6_finalization/
    report_audit.json
    reproducibility_manifest.json
    checksums.sha256
    finalization_summary.json
```
