# Baseline run notes

Run date: 26 September 2026 (Singapore). This note records outputs supplied from Clara's local benchmark run. The chart includes the two balanced cases; the CSV and JSON include all five cases. The commands completed and the files were checked for internal consistency. This is a draft analysis to review and rewrite in Clara's own words.

## Fixed comparison

The benchmark compares uniform, fixed Lipschitz, scheduled gradient gain, and triggered gradient gain on four synthetic cases and the packaged diabetes dataset. Each method has 12 runs per case, a maximum of 2,400 coordinate updates, and a target training-objective gap of 1e-4. Target success counts below are read from `summary.json`.

| Case | Uniform | Lipschitz | Scheduled gain | Triggered gain |
| --- | ---: | ---: | ---: | ---: |
| Balanced independent | 12/12 | 12/12 | 12/12 | 12/12 |
| Scaled independent | 12/12 | 0/12 | 12/12 | 12/12 |
| Balanced correlated | 12/12 | 12/12 | 0/12 | 0/12 |
| Scaled correlated | 6/12 | 0/12 | 0/12 | 0/12 |
| Diabetes | 12/12 | 12/12 | 9/12 | 5/12 |

## Observations grounded in the outputs

- All methods met the target on balanced independent data, but that alone does not show that they did so at the same computational cost.
- On scaled independent data, fixed Lipschitz sampling did not meet the target in any run. The other three methods did in every run.
- On balanced correlated data, both fixed rules met the target in every run, whereas neither gradient-gain rule did. On scaled correlated data, uniform was the only method with any successful runs.
- On diabetes, the fixed rules succeeded in all runs; scheduled and triggered gain succeeded in 9 and 5 runs, respectively. Thus the current adaptive heuristics do not show a general advantage.
- The chart displays *median* trajectories for two balanced cases. A median curve crossing the target does not establish that every run crossed it; consult the success counts and individual trajectories.

## Interpret carefully

The median time and proxy work to target are calculated among successful runs only; null means no run met the target. In particular, the scaled-correlated uniform and diabetes triggered-gain cost figures describe only successes, so comparing them alone with a 12/12 method would be misleading. The proxy is an approximate work accounting method, not wall time, energy, or an exact FLOP count. A reported final gap of zero reflects floating point precision. Differences between methods may involve feature geometry, the gain rule, refresh frequency, and sampling overhead; this run does not isolate causes. The chart does not include scaled cases or diabetes.

## Clara's research notes to complete after checking the figure and code

1. My own description of the most surprising outcome: [write here]
2. One plausible, testable explanation and what measurement could disprove it: [write here]
3. Code, formulas, and cost assumptions I personally checked: [write here]
4. A published adaptive sampling baseline to reproduce, with full citation: [write here]
5. A held-out experiment and success criterion chosen before running it: [write here]

## Next research milestone

Reproduce one suitable published adaptive coordinate method, verify its implementation, and compare it under the same stopping criterion with refresh and selection costs included. Choose new test settings before adjusting a candidate method. Update the report only after these checks and decisions are completed.
