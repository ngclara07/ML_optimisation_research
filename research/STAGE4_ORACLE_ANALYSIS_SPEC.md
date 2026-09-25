# Stage 4: ASCD oracle precision and active-set mechanism analysis

Date: 26 September 2026.

Status: predeclared analysis plan before Stage 4 experimental runs.

## Starting point

Stage 3 implemented Approximate Steepest Coordinate Descent (ASCD) on
the frozen ridge-regression benchmark using the paper's zero gradient
oracle and exact full-gradient initialization.

On all four synthetic cases, whose coordinate dimension is 48, the
median ASCD active-set size was 48. On the scaled-correlated case,
ASCD reached the target in 9/12 runs versus 6/12 for uniform sampling,
but on all five seeds where both methods succeeded ASCD recorded greater
proxy work under the Stage 3 accounting convention.

Stage 4 does not alter or reinterpret the frozen Stage 3 results.

## Research question

Why does zero-oracle ASCD retain an effectively full active set on the
synthetic ridge problems, and can a tighter published gradient oracle
reduce active-set size sufficiently to compensate for its additional
information-acquisition cost?

## Primary source

Sebastian U. Stich, Anant Raj, and Martin Jaggi,
"Approximate Steepest Coordinate Descent", ICML 2017,
especially Algorithm 1, Section 4, and Section 7.

For least squares the paper lists:

1. exact inner-product oracle g1 with delta_ij = 0;
2. approximate inner-product oracle g2 with
   delta_ij = epsilon ||X_i|| ||X_j||;
3. zero oracle g3 = 0 with
   delta_ij = ||X_i|| ||X_j||.

The Stage 3 method corresponds to g3.

## Frozen experimental conditions

Unless explicitly identified as a diagnostic-only experiment, Stage 4
retains:

- the same five benchmark cases;
- the same 12 problem seeds;
- lambda = 0.03;
- w = 0 initialization;
- target objective gap 1e-4;
- 120-update checkpoint spacing;
- maximum 2,400 coordinate updates;
- the same exact ridge coordinate minimization rule.

The existing Stage 3 tag remains the immutable comparison point.

## Phase A: zero-oracle mechanism diagnosis

Replay the existing zero-oracle ASCD algorithm while recording per-update
diagnostics without changing its coordinate selections or updates.

For each update record at least:

- case and problem seed;
- update number;
- selected coordinate;
- active-set size |I|;
- active-set fraction |I|/d;
- median and maximum certified radius;
- median and maximum true |gradient|;
- median upper and lower bounds;
- fraction of coordinates with lower bound equal to zero;
- number/fraction of coordinates excluded;
- objective gap at available checkpoints.

The primary mechanism question is whether accumulated zero-oracle radii
cause lower bounds to collapse toward zero and/or upper bounds to remain
too large for the strict ASCD exclusion condition.

That explanation must be supported by the recorded diagnostics rather
than assumed from the aggregate active-set median.

## Phase B: published oracle-quality comparison

Compare the following oracle configurations while retaining the same
benchmark and coordinate-update rule:

- g3: existing zero oracle;
- g1: exact inner-product oracle;
- simulated g2 at predeclared epsilon values.

Initial simulated g2 precision grid:

epsilon in {1.0, 0.5, 0.25, 0.125}.

These epsilon values are our predeclared experimental choices; they are
not claimed to be the paper's exact experimental grid.

For simulated g2, construct an oracle value satisfying

|S(i,j) - X_i^T X_j|
    <= epsilon ||X_i|| ||X_j||

and use the corresponding certified error radius.

The simulation is intended to isolate the effect of oracle precision.
It must not be described as a measured low-dimensional-embedding cost.

## Primary Stage 4 measurements

For every oracle configuration report:

- target successes out of 12;
- first saved checkpoint reaching the target;
- median active-set size;
- active-set-size distribution over updates;
- fraction of updates with |I| < d;
- fraction of updates with |I| <= d/2;
- objective-gap trajectories;
- recorded setup cost;
- recorded per-update selection/oracle cost;
- total proxy work to target among successful runs.

Paired seed comparisons should be reported whenever success sets overlap.

## Cost-accounting rule

Stage 4 must separate:

1. optimization/update work;
2. active-set computation;
3. oracle setup/acquisition;
4. oracle maintenance;
5. evaluation-only work.

Exact Gram-matrix availability must not be treated as free merely because
the benchmark computes related quantities for evaluation.

A simulated g2 experiment measures oracle-quality effects only unless a
real acquisition mechanism and its cost are explicitly implemented.

## Interpretation constraints

Stage 4 is a comparison on the frozen benchmark, not a reproduction of
the paper's figures.

The paper's one-step theorem should not be applied directly as a guarantee
for this benchmark because the benchmark retains coordinate-specific L_j
values and its existing exact coordinate-minimization update.

No new-method or superiority claim is predeclared.

## Planned outputs

results/stage4_oracle_analysis/
    zero_oracle_diagnostics.csv
    zero_oracle_summary.json
    oracle_comparison.csv
    oracle_summary.json
    active_set_trajectories.png
    radius_tightness.png

## Decision rule before a costed approximate oracle

Proceed to an actual costed approximate-inner-product implementation only
if the Phase B precision sweep shows that tighter certified errors
materially reduce active-set size or updates to target on the frozen
benchmark.

If oracle precision does not materially change pruning, Stage 4 should
report that negative result rather than introducing additional heuristics.

## Provenance

Stage 4 planning and implementation are AI-assisted. Conclusions should
distinguish primary-source statements, directly measured experimental
results, and hypotheses/inferences.
