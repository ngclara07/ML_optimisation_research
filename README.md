# When Does Adaptive Coordinate Sampling Pay for Itself?

## Ridge-Regression Baselines, ASCD Oracle Quality, and Confirmatory Validation

Independent research project in optimization and machine learning.

Author: **Ng Jing Yi, Clara**

Research period: **September 2026**

---

## Overview

This repository studies coordinate-selection strategies for ridge-regression
optimization.

The central research question is:

> When does additional information about coordinate quality improve
> optimization enough to justify the cost of obtaining, maintaining, and
> using that information?

The work begins with simple coordinate-sampling baselines, proceeds to a
published Approximate Steepest Coordinate Descent (ASCD) selection mechanism,
then analyzes the effect of oracle quality on ASCD active-set pruning, and
finally performs a held-out confirmatory experiment with harmonized runtime
accounting.

The project is an optimization study. It does not claim a new coordinate
descent algorithm, universal superiority of adaptive sampling, or predictive
machine-learning superiority.

The final research conclusion is:

> Oracle information quality strongly determines whether the ASCD active-set
> mechanism can prune coordinates, but whether that information pays for
> itself computationally depends on the problem regime and the cost of
> obtaining that information.

---

## Research progression

The completed research is organized into six stages.

### Stage 1 — Frozen baseline benchmark

The initial benchmark compares:

- uniform coordinate sampling;
- fixed Lipschitz sampling;
- scheduled gradient-gain sampling;
- triggered gradient-gain sampling.

The ridge objective is

\[
f(w)
=
\frac{1}{2}\|Xw-y\|_2^2
+
\frac{\lambda}{2}\|w\|_2^2,
\qquad
\lambda=0.03.
\]

Each coordinate update uses exact coordinate minimization.

The original benchmark contains:

- balanced independent synthetic data;
- scaled independent synthetic data;
- balanced correlated synthetic data;
- scaled correlated synthetic data;
- the scikit-learn diabetes regression dataset.

Each method is evaluated over 12 seeded problem instances and at most
2,400 coordinate updates.

The baseline results show that the two gradient-gain heuristics do not provide
a consistent advantage over fixed sampling.

---

### Stage 2 — Sampling diagnosis

Stage 2 investigates one possible explanation for the poor behavior of the
gradient-gain heuristics on correlated designs.

The diagnostic measures include:

- maximum sampling probability;
- effective number of sampled coordinates;
- top-five gain-coordinate overlap;
- probability mass assigned to current high-gain coordinates.

The diagnostic does not support the hypothesis that correlated-case failure is
explained simply by greater probability concentration.

Stage 2 is exploratory and does not modify any sampling rule.

---

### Stage 3 — Published ASCD comparison

Stage 3 implements the selection mechanism of:

Sebastian U. Stich, Anant Raj, and Martin Jaggi,
**Approximate Steepest Coordinate Descent**, ICML 2017.

The implementation uses the paper's zero oracle \(g^3\) with exact
full-gradient initialization while retaining this project's exact
ridge-coordinate update.

On the original scaled-correlated development case:

- uniform sampling reaches the target in 6/12 runs;
- zero-oracle ASCD reaches it in 9/12 runs.

However, on the five problem seeds where both methods succeed, ASCD records
greater proxy work on every shared-success seed.

The median paired ASCD/uniform proxy-work ratio is approximately:

`1.5401`

The median zero-oracle ASCD active set remains the full coordinate dimension.

Stage 3 therefore does not demonstrate a general computational advantage for
ASCD under the project's accounting model.

Specification:

`research/STAGE3_ASCD_METHOD_SPEC.md`

Primary outputs:

`results/published_ascd/`

---

### Stage 4 — Oracle-quality and active-set analysis

Stage 4 investigates why zero-oracle ASCD performs almost no active-set
pruning.

The analysis compares:

- exact oracle `g1`;
- simulated approximate oracle `g2`;
- zero oracle `g3`.

The tested simulated `g2` precision levels are:

`1.0, 0.5, 0.25, 0.125`

The principal Stage 4 mechanism result is that the zero oracle accumulates
certified uncertainty large enough to drive most lower bounds to zero.

When the lower bounds collapse, the strict safe-exclusion rule has little
ability to remove coordinates, leaving the active set effectively full.

The exact `g1` oracle changes the regime completely:

- median active-set size becomes one coordinate;
- the target is reached in all Stage 4 runs.

The tested intermediate simulated `g2` settings do not provide a useful
middle regime.

After charging a modelled dense-Gram setup cost, exact information is cheaper
than zero-oracle ASCD on some problem families and more expensive on others.

This motivates the regime-dependent cost question tested in Stage 5.

Specification:

`research/STAGE4_ORACLE_ANALYSIS_SPEC.md`

Primary outputs:

`results/stage4_oracle_analysis/`

---

### Stage 5 — Confirmatory robustness and runtime validation

Stage 5 tests whether the Stage 4 mechanism persists outside the development
benchmark.

The confirmatory comparison retains only:

- uniform sampling;
- fixed Lipschitz sampling;
- zero-oracle ASCD `g3`;
- exact-oracle ASCD `g1`.

#### Held-out synthetic data

The held-out synthetic generator uses:

- `n = 800`;
- `d = 96`;
- Toeplitz covariance;
- correlation `rho` in `{0.6, 0.9}`;
- balanced and geometrically scaled variants;
- problem seeds `100..111`;
- Gaussian observation noise with standard deviation `0.15`.

This produces four held-out synthetic problem families and 48 synthetic
problem instances.

#### Real datasets

Stage 5 adds:

- California Housing;
- Concrete Compressive Strength;
- Airfoil Self-Noise;
- Residential Building.

The real datasets use a fixed 80/20 train/test split with preprocessing
statistics estimated from the training split only.

For Residential Building:

- `V-9` is the sale-price target;
- `V-10` is the second output and is excluded;
- both outputs are excluded from the predictor matrix;
- 107 predictors remain before zero-variance filtering;
- repeated economic variables are assigned lag-aware identifiers such as
  `V-11_lag1` through `V-29_lag5`.

#### Stage 5 convergence protocol

The normalized objective gap is

\[
R_t
=
\frac{f(w_t)-f^\star}
{\max\{f(w_0)-f^\star,10^{-30}\}}.
\]

Primary target:

`R_t <= 1e-6`

Secondary target:

`R_t <= 1e-4`

Maximum budget:

`200 coordinate epochs`

Checkpoint interval:

`5 coordinate epochs`

#### Timing protocol

Each method/problem pair receives:

- one untimed warm-up;
- three timed repetitions;
- method-specific setup inside the timer;
- coordinate selection inside the timer;
- coordinate updates inside the timer;
- ASCD maintenance inside the timer;
- exact Gram acquisition inside the `g1` timer.

Checkpoint evaluation and computation of the dense reference optimum are
excluded from the primary method runtime.

Numerical-library thread count is constrained to one where supported.

#### Main Stage 5 result

The Stage 4 active-set mechanism generalizes across all eight held-out problem
families.

For every Stage 5 family:

- median `g3` active-set fraction is `1.0`;
- median `g3` zero-lower-bound fraction is `1.0`;
- median `g1` active-set size is one coordinate.

However, the computational consequence is regime-dependent.

The paired median `g1/g3` runtime ratio is below one on:

- balanced synthetic, `rho = 0.6`;
- balanced synthetic, `rho = 0.9`;
- California Housing;
- Concrete Strength;
- Airfoil Self-Noise.

It is above one on:

- scaled synthetic, `rho = 0.6`;
- scaled synthetic, `rho = 0.9`.

On the strongly scaled correlated family, `g1` reaches the primary target in
9/12 runs while `g3` reaches it in 12/12.

On Residential Building, no method reaches the primary `1e-6` target within
the declared budget.

Therefore the confirmatory evidence supports a mechanism-level conclusion
about oracle quality and pruning, but not a universal exact-oracle runtime
advantage.

Specification:

`research/STAGE5_CONFIRMATORY_VALIDATION_SPEC.md`

Milestone tag:

`stage5-showcase-2026-09-26`

Frozen Stage 5 commit:

`5ec3b22`

Primary outputs:

`results/stage5_confirmatory/`

---

### Stage 6 — Final research synthesis and reproducibility audit

Stage 6 does not introduce another optimization experiment.

It performs:

- numerical report auditing;
- experiment-provenance auditing;
- dependency freezing;
- repository documentation;
- final LaTeX/PDF quality assurance;
- canonical file hashing;
- final release preparation.

Specification:

`research/STAGE6_FINALIZATION_SPEC.md`

Audit scripts:

`src/stage6_report_audit.py`

`src/stage6_reproducibility_audit.py`

Stage 6 outputs are stored in:

`results/stage6_finalization/`

The planned final release tag is:

`final-research-2026-09-26`

---

# Repository structure

```text
kaust_optimisation_research/
│
├── README.md
├── requirements.txt
├── requirements-lock.txt
│
├── research/
│   ├── STAGE3_ASCD_METHOD_SPEC.md
│   ├── STAGE4_ORACLE_ANALYSIS_SPEC.md
│   ├── STAGE5_CONFIRMATORY_VALIDATION_SPEC.md
│   └── STAGE6_FINALIZATION_SPEC.md
│
├── src/
│   ├── benchmark.py
│   ├── diagnose_sampling.py
│   ├── plot.py
│   ├── published_ascd.py
│   ├── paired_stage3_analysis.py
│   ├── stage4_active_set_diagnostics.py
│   ├── stage4_plot_diagnostics.py
│   ├── stage4_oracle_comparison.py
│   ├── stage5_prepare_data.py
│   ├── stage5_confirmatory_benchmark.py
│   ├── stage5_analyze.py
│   ├── stage5_plot.py
│   ├── stage6_report_audit.py
│   └── stage6_reproducibility_audit.py
│
├── results/
│   ├── diagnostics/
│   ├── published_ascd/
│   ├── stage4_oracle_analysis/
│   ├── stage5_confirmatory/
│   └── stage6_finalization/
│
├── results_baseline_v1/
│
├── data/
│   └── stage5_cache/          # local/regenerable; Git-ignored
│
└── report/
    ├── main.tex
    └── main.pdf
```
