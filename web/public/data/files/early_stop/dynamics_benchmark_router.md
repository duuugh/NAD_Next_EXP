# Dynamics Benchmark Router Report

- Input summary JSON: `/home/jovyan/work/NAD_Next/result/dynamics_v2_local_batch_summary.json`
- Router type: rule-based (no training / no additional model)
- Best mode candidates: `rho_tail_only`, `neg_A_accel_only`, `rho_tail_plus_neg_A_accel`

## Per-benchmark Decisions

| Benchmark | Best Mode | ΔAUROC | ΔSelAcc | ΔStop@100 | Strength | Conservative | Aggressive | Source |
|---|---|---:|---:|---:|---|---|---|---|
| DS-R1/aime24 | rho_tail_plus_neg_A_accel | N/A | N/A | N/A | strong | rho_tail_plus_neg_A_accel | rho_tail_plus_neg_A_accel | prior_validated_rule |
| DS-R1/aime25 | neg_A_accel_only | +0.137600 | +0.000000 | +0.000000 | strong | neg_A_accel_only | neg_A_accel_only | auto_from_summary |
| DS-R1/gpqa | rho_tail_plus_neg_A_accel | +0.004277 | +0.003030 | +0.000000 | borderline | disable | rho_tail_plus_neg_A_accel | auto_from_summary |
| DS-R1/hmmt25 | rho_tail_plus_neg_A_accel | +0.128250 | +0.000000 | +0.000000 | strong | rho_tail_plus_neg_A_accel | rho_tail_plus_neg_A_accel | auto_from_summary |
| DS-R1/lcb_v5 | neg_A_accel_only | +0.017056 | +0.002994 | +0.000000 | borderline | disable | neg_A_accel_only | auto_from_summary |

## Policies

### Conservative

| Benchmark | Route |
|---|---|
| DS-R1/aime24 | rho_tail_plus_neg_A_accel |
| DS-R1/aime25 | neg_A_accel_only |
| DS-R1/brumo25 | disable |
| DS-R1/gpqa | disable |
| DS-R1/hmmt25 | rho_tail_plus_neg_A_accel |
| DS-R1/lcb_v5 | disable |

### Aggressive

| Benchmark | Route |
|---|---|
| DS-R1/aime24 | rho_tail_plus_neg_A_accel |
| DS-R1/aime25 | neg_A_accel_only |
| DS-R1/brumo25 | disable |
| DS-R1/gpqa | rho_tail_plus_neg_A_accel |
| DS-R1/hmmt25 | rho_tail_plus_neg_A_accel |
| DS-R1/lcb_v5 | neg_A_accel_only |

## Manual Injection Notes

- `DS-R1/aime24` is injected as `rho_tail_plus_neg_A_accel` by prior validated rule (not auto-decided from this batch summary).

## Required Final Answers

- Safe strong benchmarks now: DS-R1/aime25, DS-R1/hmmt25
- Borderline-only benchmarks now: DS-R1/gpqa, DS-R1/lcb_v5
- Weak benchmarks now: (none)
- Conservative vs aggressive: conservative disables borderline, aggressive keeps borderline best_mode enabled.
- Main leaderboard recommendation: conservative policy.
- Should final recommended strategy be the provided 5-benchmark JSON?
  - Yes. Router output matches that strategy on the 5 core benchmarks.
