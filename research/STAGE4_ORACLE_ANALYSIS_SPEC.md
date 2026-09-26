# Stage 4: ASCD oracle precision and active-set mechanism analysis

Date: 26 September 2026.

Status: **Phase A zero-oracle mechanism diagnosis and Phase B published
oracle-quality comparison completed and audited. Under the predeclared
decision rule, the tested simulated approximate-oracle configurations do
not justify proceeding directly to a costed approximate-inner-product
implementation.**

This Stage 4 analysis was developed with AI assistance. The frozen Stage 3
results remain unchanged.

---

## 1. Starting point

Stage 3 implemented Approximate Steepest Coordinate Descent (ASCD) on the
frozen ridge-regression benchmark using the paper's zero gradient oracle
and exact full-gradient initialization.

On all four synthetic cases, whose coordinate dimension is 48, the
reported median ASCD active-set size was 48. On the scaled-correlated case,
ASCD reached the target in 9/12 runs versus 6/12 for uniform sampling.

A paired Stage 3 comparison showed that five seeds succeeded under both
uniform sampling and zero-oracle ASCD:

`[0, 4, 5, 6, 10]`.

On all five shared successes, zero-oracle ASCD recorded greater proxy work
under the Stage 3 accounting convention. Median recorded work on those
shared successes was:

- uniform: `1,846,672`;
- ASCD: `2,844,008`;
- median paired ASCD-minus-uniform difference: `997,336`;
- median ASCD/uniform recorded-work ratio: `1.5401`.

Those results established that the zero-oracle ASCD configuration changed
which scaled-correlated problem instances reached the target but did not
demonstrate a recorded cost advantage under the Stage 3 proxy convention.

Stage 4 does not alter, overwrite, or reinterpret the frozen Stage 3
results. The Stage 3 Git tag remains the immutable comparison point:

`stage3-showcase-2026-09-26`.

---

## 2. Research question

Stage 4 asks:

> Why does zero-oracle ASCD retain an effectively full active set on the
> synthetic ridge problems, and can tighter published gradient-oracle
> information reduce active-set size sufficiently to compensate for its
> additional information-acquisition cost?

Stage 4 was divided into two phases:

1. **Phase A:** diagnose the mechanism responsible for the almost-full
   active sets observed under the zero oracle;
2. **Phase B:** compare published ASCD oracle-quality configurations under
   the same frozen benchmark and coordinate-update rule.

Both phases are now complete.

---

## 3. Primary source

The primary source is:

Sebastian U. Stich, Anant Raj, and Martin Jaggi,
*Approximate Steepest Coordinate Descent*, ICML 2017,
especially Algorithm 1, Section 4, and Section 7.

For least-squares problems, the paper describes a hierarchy of gradient
oracles.

### Exact inner-product oracle: `g1`

The exact oracle uses the true cross-coordinate inner product and has

`delta_ij = 0`.

### Approximate inner-product oracle: `g2`

The approximate oracle supplies an approximation with a certified error
bound of the form

`delta_ij = epsilon ||X_i|| ||X_j||`.

### Zero oracle: `g3`

The zero oracle uses

`g3_ij = 0`

for cross-coordinate gradient changes, with certified error

`delta_ij = ||X_i|| ||X_j||`.

The Stage 3 ASCD implementation corresponds to `g3`.

The Stage 4 simulated `g2` configurations are oracle-quality experiments.
They are not claimed to reproduce a practical embedding implementation or
the exact synthetic protocol used in the paper.

---

## 4. Frozen experimental conditions

Unless explicitly identified as diagnostic-only, Stage 4 retains the same
benchmark conditions as Stage 3:

- the same five benchmark cases;
- the same 12 problem seeds;
- `lambda = 0.03`;
- initial point `w = 0`;
- target objective gap `1e-4`;
- 120-update checkpoint spacing;
- maximum of 2,400 coordinate updates;
- the same exact ridge coordinate-minimization rule;
- the same problem-generation code;
- the same ASCD active-set construction.

The five cases are:

1. balanced independent;
2. scaled independent;
3. balanced correlated;
4. scaled correlated;
5. diabetes.

The four synthetic cases use `d = 48`. The diabetes case uses `d = 10`.

The Stage 3 results remain the baseline reference and are not replaced by
Stage 4 results.

---

# Phase A: zero-oracle mechanism diagnosis

## 5. Phase A objective

Phase A replayed the existing zero-oracle ASCD algorithm while recording
per-update diagnostics **without changing coordinate selections or
optimization updates**.

The predeclared mechanism hypothesis was:

> Accumulated zero-oracle uncertainty causes the certified lower gradient
> bounds to collapse toward zero while the upper bounds remain too loose
> for the strict ASCD exclusion condition to remove coordinates.

This was treated as a hypothesis before the diagnostic run rather than as
an assumed explanation.

---

## 6. Phase A diagnostic implementation

The implementation is:

`src/stage4_active_set_diagnostics.py`.

It preserves:

- the same random-number generator convention;
- the same seed offset;
- the same `active_set()` implementation;
- the same coordinate-selection draw;
- the same exact coordinate update;
- the same residual update;
- the same zero-oracle radius update;
- the same Stage 3 checkpoint proxy-work calculation.

Additional full-gradient calculations are used only for diagnostics and
are not charged to the Stage 3 proxy.

Per-update measurements include:

- case;
- problem seed;
- update number;
- selected coordinate;
- active-set size;
- active-set fraction;
- excluded-coordinate count and fraction;
- median and maximum certified radius;
- median and maximum true absolute gradient;
- median and maximum upper bound;
- median and maximum lower bound;
- number and fraction of coordinates with zero lower bound;
- mean squared active-set lower bound;
- selected-coordinate gradient and bounds;
- selected-coordinate step;
- minimum certified-interval slack;
- objective gap at available checkpoints.

---

## 7. Frozen Stage 3 replay validation

The Phase A implementation compares every saved checkpoint against

`results/published_ascd/trajectories.csv`.

Both objective gap and recorded proxy work are validated.

The diagnostic run reported:

`Stage 3 trajectory validation: PASSED`.

Therefore the Phase A measurements describe the frozen Stage 3
zero-oracle trajectory rather than a modified optimization method.

---

## 8. Phase A results

The principal aggregate results were:

| Case | Success | Median active set | Fraction full active set | Fraction with pruning | Median zero-lower-bound fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced independent | 12/12 | 48 | 0.9994 | 0.0006 | 1.0000 |
| Scaled independent | 12/12 | 48 | 0.9994 | 0.0006 | 1.0000 |
| Balanced correlated | 12/12 | 48 | 0.9994 | 0.0006 | 1.0000 |
| Scaled correlated | 9/12 | 48 | 0.9994 | 0.0006 | 1.0000 |
| Diabetes | 12/12 | 10 | 0.9996 | 0.0004 | 1.0000 |

The earlier Stage 3 statement that the median active-set size equals the
full dimension therefore understates the strength of the effect. The full
coordinate set is active on almost every zero-oracle update.

For the synthetic cases, pruning occurs on only about 0.06% of recorded
updates. For diabetes, pruning occurs on about 0.04% of recorded updates.

---

## 9. Lower-bound collapse

The ASCD lower bound is

`ell_j = max(0, |gtilde_j| - r_j)`.

The median zero-lower-bound fraction is `1.0000` on every benchmark case.

Thus, at a typical recorded update, all coordinates have zero certified
lower bound.

The safe exclusion rule requires

`u_j^2 < mean(ell_i^2 for i in I)`.

When every lower bound is zero, the right-hand side is zero. Since
`u_j^2 >= 0`, strict exclusion is impossible.

The diagnostic plots also show that the certified radii generally remain
above the scale of the true coordinate gradients, especially on the
correlated cases.

---

## 10. Phase A mechanism conclusion

Phase A strongly supports the proposed mechanism:

1. passive-coordinate uncertainty accumulates under the zero oracle;
2. certified radii become at least as large as many stored gradient
   estimates;
3. most or all lower bounds collapse to zero;
4. the active-set exclusion threshold collapses with them;
5. safe coordinate exclusion becomes almost impossible;
6. the active set remains essentially full.

This is an empirical mechanism diagnosis, not a formal causal proof.

Phase B was designed as a direct intervention: reduce oracle uncertainty
and observe whether active-set behavior changes.

---

## 11. Phase A outputs

Completed outputs:

```text
results/stage4_oracle_analysis/
    zero_oracle_diagnostics.csv
    zero_oracle_summary.json
    active_set_trajectories.png
    radius_tightness.png
```
