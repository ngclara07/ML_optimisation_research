# Baseline run notes

Run date: 26 September 2026 (Singapore). Clara ran the baseline and then reran the updated benchmark with the coordinate-update audit. The files checked for this note are the supplied `summary.json`, `trajectories.csv`, `median_trajectories.png`, earlier `run.log`, and `run_audited.log`. The checked numerical results belong to the audited rerun. The initial code and draft interpretations involved AI assistance; Clara's own observations and research choices should be recorded separately.

## Experimental setup

The benchmark solves ridge regression with `f(w) = 0.5 * ||X @ w - y||^2 + 0.5 * LAM * ||w||^2` and `LAM = 0.03`. Four methods share the same coordinate update: uniform, fixed Lipschitz, scheduled gradient gain, and triggered gradient gain. The five cases comprise balanced independent, scaled independent, balanced correlated, scaled correlated, and the packaged diabetes dataset. Each case and method has 12 seeded runs, at most 2,400 coordinate updates, and checkpoints every 120 updates. A saved run reaches the target if its objective gap at a checkpoint is at most `1e-4`. An exact ridge solve supplies the reference objective *for evaluation only*.

## Coordinate-update checks

The updated `benchmark.py` calls `audit_coordinate_update()` before the timed runs. On a small deterministic example it checks:

- `smoothness[j] = (X.T @ X + LAM * I)[j, j] = ||X[:, j]||^2 + LAM`, matching the stated `L_j`;
- `g_j = X[:, j] @ residual + LAM * w[j]`, against a finite-difference derivative of the objective;
- `delta = -g_j / L_j` and the predicted objective decrease `g_j**2 / (2 * L_j)`;
- after `w[j] += delta` and `residual += delta * X[:, j]`, an independent calculation still gives `residual = X @ w - y`.

The audit passed when the updated script was run. It checks this coordinate calculation on one deterministic example; it does not verify every sampling rule, establish why one method wins, or demonstrate originality. Clara's personal review of the code remains to be recorded below.

## File and rerun consistency

- The attached `summary.json` contains 20 records: five cases times four methods. `run_audited.log` has the matching 20 printed records.
- The attached `trajectories.csv` contains 5,040 data rows: five cases times four methods times 12 seeds times 21 checkpoints (steps 0 through 2,400). Its success counts agree with the summary.
- The earlier `run.log` also contains 20 records. Its success counts, median final gaps, refresh counts, and proxy-work-to-target values agree with the audited rerun; its elapsed times differ. The current `summary.json` belongs to the later run saved in `run_audited.log`.
- The figure shows median trajectories only for balanced independent and balanced correlated data. The other three cases are present in the CSV and JSON.

## Target successes in the audited rerun

| Case | Uniform | Lipschitz | Scheduled gain | Triggered gain |
| --- | ---: | ---: | ---: | ---: |
| Balanced independent | 12/12 | 12/12 | 12/12 | 12/12 |
| Scaled independent | 12/12 | 0/12 | 12/12 | 12/12 |
| Balanced correlated | 12/12 | 12/12 | 0/12 | 0/12 |
| Scaled correlated | 6/12 | 0/12 | 0/12 | 0/12 |
| Diabetes | 12/12 | 12/12 | 9/12 | 5/12 |

On balanced independent data, all four methods reached the target in all runs. On balanced correlated data, uniform and Lipschitz reached it in all 12 runs; scheduled and triggered gain reached it in none within the 2,400-update limit. Their respective median final objective gaps were about `6.61e-6`, `7.35e-6`, `5.48e-4`, and `1.42e-2`. The plotted median curves visually agree with this ordering. The scaled and diabetes results are reported in the table but are not shown in that figure. Neither gain heuristic has a consistent advantage across the five cases.

## Interpretation and measurement limits

`median_proxy_work_to_target` and `median_seconds_to_target` are medians **over successful runs only**; a null value means no run reached the target. On scaled correlated data, uniform succeeded in 6/12 runs, so its median work to target describes those six successes. A median plotted curve cannot establish that all individual runs reached a target. Work is an approximate proxy, not measured FLOPs or energy. Time depends on hardware and runtime conditions, and the two log files represent separate executions. A saved checkpoint may occur after the actual first target crossing. A displayed zero objective gap reflects floating-point precision. The reference solve and initial matrix construction happen before each method's timer and are not included in its reported per-run seconds.

These observations do not isolate the cause of the correlated-case difference. Probability concentration, stale gain estimates, feature geometry, refresh frequency, and cost accounting are possible mechanisms to investigate; none has yet been measured here as an explanation.

## Record of subsequent stages

The findings below extend these **notes**, while the baseline PDF and its original results remain a report of Stage 1. Keep `results_baseline_v1/` as an archive and check locally that the copy contains the intended summary, figure and trajectories; the working archive available here contains its trajectories file. The current `results/summary.json` and `results/trajectories.csv` are the baseline working outputs, whereas Stage 2 and Stage 3 have separate output directories.

## Stage 2 follow-up: separate diagnostic

The baseline outputs and baseline report remain frozen. A separate replay in `src/diagnose_sampling.py` was designed to examine the **hypothesis**, posed before running it, that gain sampling under correlated features fails because probabilities become unusually concentrated and the gain ranking becomes stale. It saved 960 checkpoint measurements to `results/diagnostics/sampling.csv` and verified all 21 objective checkpoints of each of 48 replays against the archived baseline trajectories. Its extra gradient calculations are diagnostic measurements, not inputs to the original optimizer or charges in its work proxy. See `results/diagnostics/README.md` for definitions, measurements, and limits.

| Case | Method | Above-target checkpoints | Median effective coordinates | Median max probability | Median top-five overlap | Median mass on current top five |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Balanced independent | Scheduled gain | 38 | 12.68 | 0.155 | 0.00 | 0.0526 |
| Balanced independent | Triggered gain | 104 | 10.20 | 0.191 | 0.00 | 0.0146 |
| Balanced correlated | Scheduled gain | 240 | 17.30 | 0.107 | 0.00 | 0.00789 |
| Balanced correlated | Triggered gain | 240 | 13.32 | 0.137 | 0.00 | 0.00581 |

These medians exclude checkpoints that have already reached the target. The independent case typically reaches the target sooner, so these are not matched observations at equal steps. The initial diagnostic does **not** support greater probability concentration in the correlated case: the median effective coordinate count is higher and maximum probability lower there for both gain methods. Top-five gain overlap is low in both cases, including the successful independent case. These observations weaken the simple concentration-and-staleness explanation as a *correlation-specific* cause. They do not establish an alternative cause.

## Stage 3 follow-up: published ASCD selection

`src/published_ascd.py` implements the active-set rule of Stich, Raj and Jaggi, *Approximate Steepest Coordinate Descent* (ICML 2017), with the paper's zero cross-coordinate gradient oracle and an exact initial gradient. It ran on our five ridge cases, with the same seeds, target, update budget and checkpoints. The result files are `results/published_ascd/summary.json` and `trajectories.csv`; details and audit scope are in `research/STAGE3_ASCD_METHOD_SPEC.md`. This is **a comparison using a published method on our benchmark**, not a replication of its published experimental figures.

| Case | Uniform success | Uniform median work among successes | ASCD success | ASCD median work among successes | ASCD median active set |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced independent | 12/12 | 513,288 | 12/12 | 762,936 | 48 |
| Scaled independent | 12/12 | 513,288 | 12/12 | 762,936 | 48 |
| Balanced correlated | 12/12 | 1,641,536 | 12/12 | 2,249,416 | 48 |
| Scaled correlated | 6/12 | 1,897,956 | 9/12 | 2,844,008 | 48 |
| Diabetes | 12/12 | 919,234 | 12/12 | 984,864 | 10 |

ASCD reached the target in three more scaled-correlated runs than uniform sampling, but its successful-run median proxy work was larger in every case. The scaled-correlated medians condition on **different success sets** and are not paired work differences. For synthetic cases, the median active set included all 48 coordinates; this configuration appears to prune little at the recorded aggregate level. Wider oracle error bounds are one possible reason, not a tested causal explanation. ASCD charges an initial full gradient and modelled selection costs; baseline and ASCD timers have different setup boundaries. Neither the proxy nor runtime establishes a controlled advantage here. No new algorithm or novelty claim follows from Stage 3.

## Clara's personal checks and next research decision

- [ ] Derive one coordinate update and explain in my own words why the residual remains `X @ w - y`.
- [ ] Read Algorithm 1 and the least-squares oracle section of the paper; trace `active_set()` and the gradient bound on one example.
- [ ] Inspect per-seed trajectories, especially scaled correlated; compare work for **shared successful seeds**, and record failures.
- [ ] Before a new method, define a focused question and predeclare development versus held-out settings. One candidate is whether a **published, tighter certified gradient oracle** shrinks ASCD's active set enough to justify its acquisition and selection costs.

Personal interpretation after examining these files: **Pending Clara's own review.** A broader independent research report should be written after the next question, fair comparisons, and held-out results have been validated.
