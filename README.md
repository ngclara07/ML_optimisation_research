# Stage 2 sampling diagnosis

Date: 26 September 2026. Exploratory diagnostic of the frozen baseline, developed with AI assistance. It is not a method change or an independent confirmation of causality.

## Question specified before this diagnostic

Could gain-based sampling fail on correlated problems because its probabilities become concentrated on coordinates whose gains change before the next refresh? If concentration and changing gain rankings are similar on the successful balanced-independent case, this explanation is insufficient on its own.

## Procedure

`python src/diagnose_sampling.py` replays the two gain methods for 12 seeds on balanced-independent and balanced-correlated data. It checks all 21 objective checkpoints of each of the 48 replays against `results_baseline_v1/trajectories.csv` before saving 960 observations to `results/diagnostics/sampling.csv`. Diagnostic gradients are computed for measurement only and are never used to select a coordinate or trigger a refresh. They are not included in the original runtime or work proxy.

At each nonzero 120-update checkpoint, before any refresh at that step, the file records the maximum probability, effective number of sampled coordinates `1 / sum(p_j**2)`, the fraction of the old top-five gain coordinates still in the current top five, and sampling mass on each top-five set. These are descriptive checkpoint diagnostics, not a convergence proof. In particular, top-five ranks near numerical convergence may be noisy.

## Preliminary checkpoint medians

For comparability, these medians exclude checkpoints where the gap is already at or below `1e-4`.

| Case | Method | Checkpoints above target | Effective coordinates | Max probability | Top-five overlap | Mass on current top five |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Balanced independent | Scheduled gain | 38 | 12.68 | 0.155 | 0.00 | 0.0526 |
| Balanced independent | Triggered gain | 104 | 10.20 | 0.191 | 0.00 | 0.0146 |
| Balanced correlated | Scheduled gain | 240 | 17.30 | 0.107 | 0.00 | 0.00789 |
| Balanced correlated | Triggered gain | 240 | 13.32 | 0.137 | 0.00 | 0.00581 |

The independent case typically reaches the target earlier, so the counts of above-target checkpoints differ greatly and medians should not be interpreted as paired causal comparisons. At these checkpoints the correlated case has a *larger* effective coordinate count and a *smaller* maximum probability than the independent case for each method; thus the proposed **extra concentration under correlation** is not supported by this measure. Top-five overlap is low in both regimes. Outdated gain rankings may matter, but this diagnosis does not show that they uniquely explain the correlated-case failure. The low current-top-five mass also occurs in the successful independent case. No sampling rule was changed in this stage.

## Follow-on analysis

Inspect paired checkpoints at the same update number before drawing stronger conclusions, measure realized objective decrease per selected coordinate and per total work, and compare with a correctly reproduced published adaptive method. Any new sampling design requires a separate development setting and held-out evaluation. Clara should record her own assessment after examining the CSV.
