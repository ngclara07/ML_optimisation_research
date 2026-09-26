# Stage 5: confirmatory robustness and generalization study

Date: 26 September 2026.

Status: **predeclared confirmatory protocol; no Stage 5 outcome has yet
been inspected.**

Stage 5 is intended to test whether the conclusions developed during
Stages 3 and 4 persist outside the development benchmark.

No Stage 5 thresholds, datasets, primary methods, or primary endpoints
should be changed after outcome inspection without explicitly labelling
the resulting analysis as exploratory.

---

## 1. Starting point

Stages 1--4 developed and audited a ridge-regression coordinate-descent
benchmark.

The main Stage 4 findings were:

1. zero-oracle ASCD retained an effectively full active set on the
   development synthetic problems;
2. its certified lower bounds collapsed to zero on almost every update;
3. exact inner-product information reduced the active set to one
   coordinate;
4. exact information improved optimization substantially;
5. after a modelled dense-Gram acquisition charge, exact information
   was cost-effective on some development regimes but not all;
6. the tested simulated intermediate g2 configurations did not provide
   a useful precision/cost middle regime.

These findings were developed on a small frozen benchmark and therefore
require independent robustness testing before stronger conclusions are
made.

Stage 5 does not modify the frozen Stage 3 or Stage 4 outputs.

---

## 2. Confirmatory research question

The primary Stage 5 question is:

> Do the Stage 4 conclusions about oracle uncertainty, active-set pruning,
> and the cost of exact information persist on held-out synthetic
> structures and additional real regression datasets under harmonized
> runtime accounting?

The analysis is confirmatory with respect to the qualitative Stage 4
mechanism.

It is not a reproduction of the ASCD paper and does not predeclare a
general superiority claim.

---

## 3. Primary hypotheses

### H1: zero-oracle pruning failure

On held-out problems with nontrivial dimension, zero-oracle ASCD will
retain a substantially larger active set than exact-oracle ASCD.

Primary mechanism measurements:

- median active-set fraction;
- fraction of updates with full active set;
- fraction of coordinates with zero lower bound.

### H2: exact-information pruning

Exact g1 information will produce materially smaller active sets than
g3 and will reduce coordinate updates required to reach the normalized
objective target on at least some held-out problems.

### H3: information cost is regime-dependent

After including the declared exact-oracle setup charge and harmonized
method runtime, the benefit of exact information will not necessarily
have the same sign on every problem.

This hypothesis deliberately allows both positive and negative
cost-effectiveness outcomes.

### H4: Stage 4 mechanism generalizes

When g3 retains nearly full active sets, this should coincide with a
high fraction of zero lower bounds.

The analysis will report the association rather than treat it as a
formal causal proof.

---

## 4. Confirmatory methods

Only the following methods are part of the primary Stage 5 comparison:

1. uniform coordinate sampling;
2. fixed Lipschitz sampling;
3. zero-oracle ASCD g3;
4. exact-inner-product ASCD g1.

The simulated Stage 4 g2 precision grid is excluded from the primary
confirmatory experiment.

Any later g2 experiment belongs to an explicitly exploratory extension.

All four methods use the same exact ridge coordinate-minimization update.

---

## 5. Objective

The training objective remains

f(w) = 1/2 ||Xw-y||^2 + lambda/2 ||w||^2

with

lambda = 0.03.

This deliberately preserves the optimization objective used in the
earlier stages.

The dense ridge solution is used only to define the evaluation reference
f* and is calculated outside method timing.

---

## 6. Primary convergence metric

Because Stage 5 contains datasets with substantially different sample
sizes and target scales, the primary metric is the normalized objective
gap

R_t =
    (f(w_t) - f*) /
    max(f(w_0) - f*, 1e-30).

Primary success threshold:

R_t <= 1e-6.

Secondary success threshold:

R_t <= 1e-4.

Raw absolute objective gaps will also be retained in the output for
comparison with earlier stages, but they are not the primary Stage 5
success criterion.

---

## 7. Coordinate-update budget

The maximum optimization budget is dimension normalized:

maximum updates = 200 * d.

Saved checkpoints occur every

5 * d

coordinate updates, in addition to step zero and the final step.

The primary update-to-target quantity is therefore also reported in
coordinate epochs:

updates / d.

This avoids assigning the same absolute update budget to problems with
very different coordinate dimensions.

---

# Held-out synthetic validation

## 8. Synthetic design

The Stage 5 synthetic generator is intentionally different from the
group-correlated development generator.

For every synthetic case:

- n = 800;
- d = 96;
- latent coefficients are generated before the design response;
- feature dependence follows a Toeplitz covariance matrix

  Sigma_ij = rho^|i-j|;

- Gaussian observation noise is added using a fixed predeclared noise
  scale;
- 12 new problem seeds are used;
- these problem seeds must not reuse the Stage 1--4 development seeds.

The four held-out synthetic regimes are:

| Case | rho | feature scaling |
| --- | ---: | --- |
| heldout_balanced_moderate | 0.60 | all scales equal |
| heldout_balanced_strong | 0.90 | all scales equal |
| heldout_scaled_moderate | 0.60 | geometric scales |
| heldout_scaled_strong | 0.90 | geometric scales |

For scaled cases, feature multipliers are fixed before running as

logspace(-1.5, 1.5, d).

The same generated problem instance is supplied to every method for a
given problem seed.

No Stage 5 synthetic parameter is to be retuned after examining method
outcomes.

---

# Real-data validation

## 9. Confirmatory real datasets

Stage 5 adds four real regression datasets that were not used to develop
the Stage 3--4 interpretation.

### California Housing

Loader:

sklearn.datasets.fetch_california_housing

Primary properties:

- 20,640 observations;
- 8 numeric predictive features;
- real-valued target.

### Concrete Compressive Strength

UCI repository identifier:

165

Primary properties:

- 1,030 observations;
- 8 quantitative input variables;
- one quantitative strength target;
- no missing values reported by UCI.

### Airfoil Self-Noise

UCI repository identifier:

291

Primary properties:

- 1,503 observations;
- 5 predictive variables;
- one continuous sound-pressure target;
- no missing values reported by UCI.

### Residential Building

UCI repository identifier:

437

Primary properties:

- 372 observations;
- high-dimensional numeric design;
- two reported output variables;
- no missing values reported by UCI.

The Stage 5 target is the sale-price output.

The data-preparation script must verify the returned target columns and
halt if the sale-price target cannot be identified unambiguously.

---

## 10. Real-data preprocessing

A single fixed train/test split is created for each real dataset using

split seed = 20260926.

Training fraction:

80%.

Test fraction:

20%.

Preprocessing parameters are estimated from the training split only.

For every feature:

- subtract the training mean;
- divide by the training standard deviation;
- if a training feature has zero variance, remove that feature and record
  its name.

The response is centered and divided by its training standard deviation.

The same transformed training design and target are supplied to every
optimization method.

The test set is transformed using training statistics only.

Dataset loading and preprocessing occur before method timing.

---

## 11. Predictive quantities

Stage 5 remains primarily an optimization study.

Test-set prediction is therefore secondary rather than a method-selection
criterion.

For each real dataset report:

- test RMSE at the dense ridge reference optimum;
- test RMSE at each method's final iterate;
- difference between final-iterate RMSE and reference-optimum RMSE.

These measurements check that optimization conclusions are not being
misrepresented as evidence of improved predictive modelling.

No method is declared superior based on test RMSE alone because all
methods optimize the same training objective.

---

# Runtime and cost protocol

## 12. Harmonized timing boundary

For each method, timing begins after:

- dataset generation/loading;
- deterministic train/test preprocessing;
- dense reference optimum computation.

Timing begins before any method-specific setup.

Therefore the timed region includes, as applicable:

- sampling-distribution setup;
- full-gradient initialization;
- ASCD bound initialization;
- exact Gram acquisition for g1;
- coordinate selection;
- coordinate updates;
- method-internal maintenance.

Evaluation-only checkpoint calculations are timed separately and excluded
from the primary method runtime.

This boundary is identical across methods.

---

## 13. Timing repetitions

For every method/problem pair:

- perform one untimed warm-up execution;
- perform three timed repetitions;
- use identical problem data;
- use the same predeclared algorithm seed for stochastic methods within
  each paired comparison.

The primary time measurement for a run is the median of the three timed
repetitions.

Threaded numerical kernels should be restricted to one thread during
timing where possible.

The environment record must include:

- Python version;
- NumPy version;
- scikit-learn version;
- BLAS/threadpool information;
- operating system;
- processor description.

---

## 14. Seeds

Synthetic problem seeds:

100 through 111 inclusive.

Algorithmic selection seeds:

9100 through 9111 inclusive.

For real datasets, the data split is fixed and the 12 algorithmic seeds
provide stochastic repetitions for uniform, Lipschitz, and g3.

Exact g1 is deterministic except for exact lower-bound ties; its seed is
still recorded for interface consistency.

---

## 15. Primary Stage 5 endpoints

For every dataset/method combination report:

- success count at relative gap 1e-6;
- success count at relative gap 1e-4;
- median coordinate epochs to the primary target among successes;
- median harmonized time to target among successes;
- median final normalized gap;
- median active-set fraction for ASCD methods;
- fraction of updates with full active set;
- median zero-lower-bound fraction;
- method-specific setup time;
- total timed runtime;
- costed proxy work, retained as a secondary accounting measure.

For real datasets additionally report test RMSE quantities defined above.

---

## 16. Paired analysis

Primary comparisons are paired by problem seed or algorithm seed.

For g1 versus g3 report, where both reach the target:

- paired difference in coordinate epochs to target;
- paired ratio of coordinate epochs;
- paired runtime difference;
- paired runtime ratio;
- paired costed-proxy difference;
- paired costed-proxy ratio.

Also report method-only success sets rather than discarding failures.

Conditional medians must never be interpreted without their corresponding
success counts.

---

## 17. Uncertainty summaries

For paired continuous outcomes report:

- median paired difference;
- median paired ratio;
- nonparametric bootstrap 95% confidence interval using 10,000 bootstrap
  resamples over the paired run units.

For success proportions report the raw numerator and denominator.

Confidence intervals are descriptive uncertainty summaries; Stage 5 is
not powered around a formal null-hypothesis significance test.

---

## 18. Confirmatory decision rules

Stage 5 supports generalization of the Stage 4 pruning mechanism if:

1. g3 has a larger median active-set fraction than g1 on the majority of
   held-out datasets; and
2. g3 shows substantially more zero lower bounds than g1 on the majority
   of held-out datasets.

Stage 5 supports the narrower claim that exact information can sometimes
pay for itself if at least one held-out real or synthetic regime shows:

- g1 success no worse than g3; and
- paired median harmonized runtime-to-target below g3; and
- the corresponding uncertainty interval and individual paired outcomes
  are reported rather than hidden.

Stage 5 does not predeclare that g1 must dominate on every dataset.

A mixed result is an acceptable confirmatory outcome and must be reported
as such.

---

## 19. Failure and stopping rules

The protocol must stop and be reviewed before interpretation if:

- a method violates the residual invariant;
- an ASCD certified interval fails to contain the true gradient beyond
  numerical tolerance;
- g3 fails to reproduce its implementation-level audit tests;
- dataset preprocessing produces NaN or infinite values;
- fewer than all declared real datasets can be loaded and the missing
  dataset is silently replaced;
- the timing implementation uses different setup boundaries between
  methods.

A dataset may be removed only for a documented technical reason, not
because of its experimental outcome.

---

## 20. Planned implementation

Primary Stage 5 files:

src/stage5_prepare_data.py
src/stage5_confirmatory_benchmark.py
src/stage5_analyze.py
src/stage5_plot.py

Planned outputs:

results/stage5_confirmatory/
    dataset_manifest.json
    environment.json
    run_results.csv
    trajectories.csv
    paired_results.csv
    summary.json
    timing_summary.json
    heldout_gap_trajectories.png
    real_gap_trajectories.png
    active_set_generalization.png
    runtime_comparison.png

---

## 21. Interpretation constraints

Stage 5 is intended to test robustness of conclusions developed on the
earlier benchmark.

It is not a reproduction of the ASCD paper.

The ASCD theoretical one-step comparison must not be invoked as a direct
guarantee for these experiments because the project retains its existing
coordinate-specific smoothness and exact coordinate-minimization rule.

The study must distinguish:

- algorithmic convergence;
- active-set mechanism;
- proxy cost;
- measured wall-clock runtime;
- predictive test performance.

No result may be described as general superiority solely from a subset of
successful runs.

---

## 22. Publication-oriented role of Stage 5

If the principal Stage 4 mechanism survives the held-out synthetic and
real-data study, Stage 5 supplies evidence that the finding is not merely
an artefact of the development benchmark.

If it does not survive, that negative result becomes part of the final
research conclusion.

The purpose of Stage 5 is therefore validation rather than narrative
confirmation.

---

## 23. Provenance

Stage 5 experimental planning and implementation are AI-assisted.

The protocol is frozen before Stage 5 outcomes are inspected.

Any later change made after inspecting results must be recorded as a
protocol deviation or exploratory analysis.

Claims of independent verification should reflect checks personally
performed by the author.
