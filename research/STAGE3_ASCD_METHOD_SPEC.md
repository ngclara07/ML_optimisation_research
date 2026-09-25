# Stage 3 ASCD: method specification and completed initial comparison

Date: 26 September 2026 (Singapore).

Status: **initial implementation, automated audit, manual rule/radius review,
and paired scaled-correlated comparison completed; harmonized cost accounting,
additional automated edge-case checks, and stronger oracle comparisons remain
pending**.

This memo was prepared with AI assistance.

## Primary source

Sebastian U. Stich, Anant Raj, and Martin Jaggi, *Approximate Steepest
Coordinate Descent*, ICML 2017, Algorithm 1 and Sections 3, 4, and 7:

https://proceedings.mlr.press/v70/stich17a/stich17a.pdf

The paper includes ridge-regression experiments but uses larger, different
synthetic datasets and different oracle configurations. Our 400-by-48 and
diabetes benchmarks provide **cross-method comparisons under our settings**,
not a reproduction of the paper's reported figures.

## Model and faithful selection rule

Our existing objective is

`f(w) = 1/2 ||Xw-y||^2 + lambda/2 ||w||^2`.

The baseline uses `lambda = 0.03`, residual `z = Xw-y`, coordinate derivative

`g_j = X_j.T z + lambda*w_j`,

and

`L_j = ||X_j||^2 + lambda`.

The paper's ASCD maintains, for each coordinate `j`, an estimate `gtilde_j`
and a certified radius

`r_j >= |g_j - gtilde_j|`.

Its safe upper/lower bounds on `|g_j|` are

`u_j = |gtilde_j| + r_j`

and

`ell_j = max(0, |gtilde_j| - r_j)`.

The active set is the **smallest** nonempty subset `I` for which every excluded
coordinate `j` satisfies

`u_j^2 < mean(ell_i^2 for i in I)`.

With equality, the coordinate is retained rather than excluded. The selected
coordinate is drawn uniformly among indices in `I` whose lower bound `ell_j`
is maximal. An implementation must follow this selection rule and maintain
valid radii; replacing it with an argmax of estimated gradients would define a
different algorithm.

## Implemented paper-listed oracle for our ridge benchmark

The implementation uses the paper's zero oracle (`g^3_ij = 0` for `i != j`)
with certified error

`delta_ij = ||X_i|| * ||X_j||`.

For ridge regression, when coordinate `i` changes by `h`, the exact gradient
change for another coordinate `j != i` is

`h * X_j.T X_i`.

Cauchy--Schwarz gives the bound

`|h| * ||X_j|| * ||X_i||`.

Thus the code leaves `gtilde_j` unchanged and increases `r_j` by

`abs(h) * ||X_j|| * ||X_i||`

for `j != i`.

The ridge regularizer has no cross-coordinate gradient change. Exact
minimization sets the selected coordinate's new gradient to zero within
numerical precision, so its estimate and radius are reset to zero. A
small-problem audit checks

`|true_g_j - gtilde_j| <= r_j + tolerance`

after every update.

To avoid starting from infinite radii, the implementation initializes

`gtilde = X.T @ (-y)`

and

`r = 0`.

*Full-gradient initialization is an allowed configuration discussed in the
paper*; Algorithm 1 defaults instead to zero estimates with infinite error
bounds. The implemented full gradient is included in ASCD's work proxy and
timer. Feature norms can be derived from the same coordinate smoothness
quantities as the benchmark, although some shared matrix construction occurs
before the method timer.

## Manual rule and radius review

The active-set implementation was traced manually on the exact-bound example

`gtilde = (3, 1, 0)`

and

`r = (0, 0, 0)`.

The initial squared-lower-bound mean is

`(3^2 + 1^2 + 0^2) / 3 = 10/3`.

The zero coordinate has squared upper bound zero, so it can be excluded. The
remaining mean is

`(3^2 + 1^2) / 2 = 5`.

The coordinate with gradient magnitude one then has squared upper bound one,
so it can also be excluded, leaving only the coordinate with gradient
magnitude three. This agrees with the implementation.

The radius update was also traced algebraically. If coordinate `i` changes by
`h`, then for `j != i`

`g_j(new) - g_j = h * X_j.T X_i`.

Cauchy--Schwarz yields

`|g_j(new) - g_j| <= |h| ||X_j|| ||X_i||`,

which is the radius increment implemented by the zero oracle. For the selected
coordinate,

`h = -g_i/L_i`

and therefore

`g_i(new) = g_i + h L_i = 0`

up to numerical error. This justifies resetting that coordinate's estimate and
radius to zero.

The current automated audit does not yet include a dedicated equality-boundary
test or an explicit monotonic-objective assertion. Those remain useful
additional checks.

## Comparison rules

The run uses the same cases, 12 seeds, initial `w=0`, objective, target
`1e-4`, 120-update checkpoint spacing, and maximum of 2,400 updates as the
baseline.

Stage 3 outputs are stored in `results/published_ascd/` and record success
rate, work to target among successful runs, full trajectories, wall time,
median active-set size, and estimated setup and selection costs.

Draws differ from the baseline because ASCD selects from an active set, but
paired problem seeds remain identical.

The paper's theoretical comparison uses a shared coordinate Lipschitz
constant and an update method shared by comparison algorithms. Our benchmark
uses coordinate-dependent `L_j` values and exact coordinate minimization.

Accordingly, **do not invoke the paper's one-step dominance theorem as a
guarantee for this benchmark**. Interpret the results as a practical
implementation of its selection mechanism under our update rule.

A strict paper-figure reproduction would additionally require the paper's
data generation, step sizes, oracle precision, and plotting protocol.

## Implemented checks and their limits

1. `src/published_ascd.py` tests an all-zero-bound active set and a
   one-coordinate active set. Its rule uses strict exclusion; equality is
   retained by the implementation, although there is not yet a separate
   automated equality-edge test.

2. On a small dense ridge case, after every update the audit recomputes
   `X.T @ (Xw-y) + lambda*w`, checks that every true gradient lies in its
   estimated interval, and checks the residual invariant.

3. The coordinate step minimizes a quadratic along that coordinate and
   therefore predicts nonincreasing objective mathematically. The current
   audit does **not** independently assert monotonic objective values at every
   step.

4. The method ran on all five frozen cases and wrote outputs to a separate
   folder. This validates one implemented comparison, not the paper's reported
   figures or a new contribution.

5. `src/paired_stage3_analysis.py` now performs a paired seed-level comparison
   of the baseline uniform method and ASCD on the scaled-correlated case. Its
   outputs are:

   - `results/published_ascd/paired_scaled_correlated.csv`
   - `results/published_ascd/paired_scaled_correlated_summary.txt`

## Work accounting in the current code

For `n` rows and `d` coordinates, the initial full gradient is charged

`n*d + d`

proxy units.

Each saved checkpoint adds

`n + d`

for objective evaluation.

Each coordinate update adds

`2*n + 3 + d*ceil(log2(d)) + 3*d`,

representing coordinate-update operations, modelled sorting, and
bounds/interval upkeep.

These are bookkeeping units, **not** exact FLOPs.

The ridge optimum used to evaluate gaps is calculated before the timer.
Initial data generation, some matrix formation, and other common setup also
fall outside the timed method section. In contrast, the ASCD full-gradient
initialization lies inside that timer; baseline timing has a different setup
boundary.

Therefore the recorded proxy-work differences, including the paired
comparison below, should be treated as descriptive under the current
accounting conventions rather than exact controlled operation-count
comparisons. Wall-time comparisons should likewise not be treated as
controlled until setup and implementation boundaries are harmonized.

## Initial implementation check and observed results

`src/published_ascd.py` implements this chosen configuration without changing
`src/benchmark.py`.

Its initial audit checks an all-zero-bound active set, a one-coordinate active
set, the residual invariant, and containment of every true coordinate
gradient inside the maintained interval after each update on a small dense
ridge example.

Stage 3 uses all five frozen problem cases; the generated outputs are in
`results/published_ascd/`.

| Case | Uniform success | Uniform work among successes | ASCD success | ASCD work among successes | ASCD median active set | ASCD initial work |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced independent | 12/12 | 513,288 | 12/12 | 762,936 | 48 | 19,248 |
| Scaled independent | 12/12 | 513,288 | 12/12 | 762,936 | 48 | 19,248 |
| Balanced correlated | 12/12 | 1,641,536 | 12/12 | 2,249,416 | 48 | 19,248 |
| Scaled correlated | 6/12 | 1,897,956 | 9/12 | 2,844,008 | 48 | 19,248 |
| Diabetes | 12/12 | 919,234 | 12/12 | 984,864 | 10 | 4,430 |

These results are **a comparison on Clara's benchmark**, not a replication of
the paper's published graphs.

The reported median active set is the median of per-run median sizes; it is
not the full distribution over steps. On all four 48-coordinate synthetic
cases that median equals 48, indicating little pruning in this configuration.
The zero oracle's wide certified bounds are a possible explanation that still
requires direct per-step measurement.

ASCD reaches the scaled-correlated target in 9/12 runs versus 6/12 for
uniform sampling. Its successful-run median proxy work is greater than the
uniform value in all five aggregate cases. Because the scaled-correlated
successful-run medians use different seed sets, a paired analysis is required
for a like-for-like seed comparison.

That paired comparison has now been completed.

## Paired scaled-correlated comparison

The paired analysis uses the same 12 problem seeds and defines work to target
as the proxy work at the first saved checkpoint satisfying

`f(w) - f* <= 1e-4`.

The success sets are:

- Uniform: 6/12 seeds `[0, 3, 4, 5, 6, 10]`
- ASCD: 9/12 seeds `[0, 1, 4, 5, 6, 7, 8, 9, 10]`
- Shared successes: 5/12 seeds `[0, 4, 5, 6, 10]`

Seed 3 succeeds only under uniform sampling. Seeds 1, 7, 8, and 9 succeed
only under ASCD.

For the five shared successes:

| Seed | Uniform first-hit step | Uniform work | ASCD first-hit step | ASCD work | ASCD - Uniform | ASCD / Uniform |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2160 | 1,846,672 | 2280 | 2,844,008 | 997,336 | 1.5401 |
| 4 | 2160 | 1,846,672 | 2400 | 2,992,656 | 1,145,984 | 1.6206 |
| 5 | 2400 | 2,051,808 | 2400 | 2,992,656 | 940,848 | 1.4585 |
| 6 | 2280 | 1,949,240 | 2280 | 2,844,008 | 894,768 | 1.4590 |
| 10 | 1920 | 1,641,536 | 2280 | 2,844,008 | 1,202,472 | 1.7325 |

Summary over the five shared successes:

- Median uniform work: **1,846,672**
- Median ASCD work: **2,844,008**
- Median paired ASCD-minus-uniform work: **997,336**
- Median ASCD/uniform recorded-work ratio: **1.5401**
- ASCD records greater proxy work on **all 5 of 5** shared successes.

This result separates update-budget success from recorded cost. ASCD reaches
the target on more scaled-correlated problem instances within 2,400 updates,
but on every instance where both methods succeed, its recorded proxy work is
higher under the present accounting conventions.

The median ratio of 1.5401 can be described as ASCD recording approximately
54% more proxy work than uniform on the median shared-success comparison.
However, this should remain explicitly qualified: the two implementations do
not yet have fully harmonized setup and accounting boundaries, so the ratio is
not an exact FLOP or controlled wall-clock ratio.

## Interpretation after paired analysis

The Stage 3 evidence currently supports the following narrow conclusions:

1. The zero-oracle ASCD configuration changes which scaled-correlated
   instances reach the target and increases the success count from 6/12 under
   uniform sampling to 9/12 under ASCD within the fixed coordinate-update
   budget.

2. The increased success count is not merely the same success set with
   different timing. One uniform-only success and four ASCD-only successes are
   observed.

3. On the five shared-success seeds, ASCD records greater proxy work on every
   seed. The median recorded-work ratio is 1.5401.

4. The median active set equals the full coordinate dimension on all four
   synthetic cases, so the implemented zero oracle provides little observed
   pruning under these settings.

5. These observations do not establish theoretical or general computational
   superiority for either method. The paper's one-step theorem does not
   directly apply to this benchmark because the benchmark uses
   coordinate-specific `L_j` values and its existing exact
   coordinate-minimization update rule.

A more informative paper-listed configuration may require a tighter
certified oracle, such as the paper's approximate inner-product oracle,
together with explicit accounting for its information-acquisition cost.

No theoretical superiority or new contribution is claimed.

## Next verification before extending the method

1. Ensure that the primary-source review of Algorithm 1 and the least-squares
   oracle is personally verified by Clara before describing the derivation as
   independently authored or independently checked.

2. **Paired scaled-correlated comparison completed.** Uniform succeeds on
   seeds `[0, 3, 4, 5, 6, 10]`; ASCD succeeds on
   `[0, 1, 4, 5, 6, 7, 8, 9, 10]`; the shared-success seeds are
   `[0, 4, 5, 6, 10]`. ASCD records greater proxy work on all five shared
   successes, with median shared-success work 1,846,672 for uniform versus
   2,844,008 for ASCD and a median recorded-work ratio of 1.5401.

3. Reconcile setup, objective-check, initialization, and selection accounting
   before making a controlled cost-difference claim. A stronger exact
   paper-figure reproduction would require that figure's dataset,
   initialization, oracle, step, and plotting settings.

4. Add an explicit equality-boundary unit test for the active-set rule and an
   objective-monotonicity assertion to the small-problem audit.

5. Examine the per-update distribution of active-set sizes rather than only
   the median of per-run medians, particularly on the synthetic cases where
   the reported median is 48.

6. Predeclare a narrow research question and evaluation split before
   implementing a tighter *published* oracle or proposing a new rule.

## Provenance

The implementation, paired-analysis workflow, and parts of the interpretation
and report text were developed with AI assistance.

The code, raw trajectories, paired CSV, paired summary, and report outputs
should remain archived together for reproducibility.

Any statement that Clara independently verified the source paper, derivation,
algorithm, cost convention, or outputs should reflect checks she personally
performed rather than AI-assisted review alone.
