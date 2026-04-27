# NAD_Next: Best-of-N and Early-Stop Experiments on Top of NAD


## 🏆 Best results in this repo

| Task | Best solution | Core idea |
|---|---|---|
| Best-of-N | `nad_mixed_v2_aime_top2_gap1e3_logprob` | Only flip when top2 gap ≤ 1e-3, use logprob as tie-break, change only AIME |
| Early Stop | `early_stop_dynamics_router_conservative_submit` | Benchmark-selective router: enable dynamics plugin only on stable benchmarks |

This repository is my experiment workspace built around `NAD_Next`, focused on two inference-time problems:

1. **Best-of-N selection**: given 64 candidate answers for each problem, choose the final answer without re-generating.
2. **Early stop / dynamic budget allocation**: decide when a sample already looks good enough and further generation budget is unlikely to pay off.

Instead of treating these as two unrelated tasks, I use the same workspace to study a common question:

> Can neuron-activation structure, token-level confidence, and lightweight routing rules improve inference-time decision making in a stable way?

The repo contains the framework code, experiment scripts, work logs, and the actual result artifacts under `result/`.

---

## Why this repo exists

`NAD_Next` was originally a framework for activation-cache analysis, selector evaluation, and visualization. On top of that framework, I used it as a research sandbox for:

- building **submission-safe best-of-n selectors**
- testing **small local tie-break rules** instead of large end-to-end rewrites
- probing whether **activation-derived signals** can help decision making
- developing **early-stop backbones** from confidence statistics
- adding **benchmark-selective dynamics plugins** only where they help

So this repo is both:

- an **analysis framework**, and
- a **research log with runnable artifacts**.

---

## Research summary

### 1) Best-of-N: from mixed baseline to tiny, controlled edits

The core best-of-n task here is:

- each problem has **64 candidate responses**
- the system must **rank / score / select** one final answer
- no new generation is introduced at selection time

The exploration path was roughly:

1. **Mixed baseline**
   - A dataset-specific mixture of selectors and fallback rules.
   - This became the stable reference point for later work.

2. **Unified hybrid scoring**
   - Example: `0.6 * knn_norm + 0.4 * selfcert_norm`.
   - Main lesson: simple global mixing of structure and confidence did **not** beat the mixed baseline.

3. **Paper-inspired E_M-regularized Best-of-N**
   - Implemented in `plugins/em_regularized_bon_selector.py`.
   - Explored `M = 2 / 4 / 8` variants as lightweight stochastic regularization.
   - This later became useful as a **specialist component** for some benchmark families.

4. **Small-step local correction on AIME**
   - Instead of redesigning the whole selector, I restricted changes to **near-tie cases**.
   - The strongest backbone in the logs became:
     - `result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json`
   - Design principle: only flip when top candidates are already extremely close.

5. **Activation-based veto experiments**
   - Activation signals produced interesting diagnostics.
   - But once connected to submission-time decisions, they were not stable enough.
   - Example: `mixed_v3` improved local `DS-R1/aime24 + aime25` from `45/60` to `46/60`, yet the leaderboard feedback was worse, so this direction was rejected as a mainline rule.

### 2) Best-of-N: what actually mattered

A recurring lesson in the work logs is that **small, benchmark-aware changes beat large unified rewrites**.

What survived repeated iterations:

- treat `mixed` / `mixed_v2` as the backbone
- intervene only on **small-gap top-2 or top-k cases**
- keep the output **submission-safe** by preserving full score dictionaries
- prefer **local corrections** over globally changing the whole ranking rule

What did not hold up well:

- making activation-derived signals the main selector
- assuming a signal that helps on a few local cases will help the full leaderboard

### 3) Rule distillation: extracting practical heuristics

One later stage turned local audits into compact rules instead of large models.

A representative artifact is:

- `result/rule_distillation_summary.json`

From that audit:

- `token_only_conservative_v1` was the best distilled rule on the primary labeled scope
- primary accuracy moved from `0.75` to `0.7833`
- it applied `6` flips with `2` correct flips and `0` wrong flips
- the gain stayed local and did **not** automatically generalize to secondary caches

This is a good example of the general style of this repo: prefer interpretable, conservative decision rules first, then check whether they survive broader evaluation.

---

## Early-stop exploration

The second main thread in this repo is early stop.

The question is not “which answer is best among 64,” but rather:

> At what point does the current partial trajectory already look strong enough that more budget is probably unnecessary?

### Backbone evolution

The strongest backbone in the current artifacts is confidence-based rather than pure dynamics-based:

- `confidence_only`
- `confidence_plus_dynamics_conservative`
- `dynamics_only_conservative`

See:

- `result/early_stop_mean_confidence_variants_eval.md`

Key comparison from that report:

- `confidence_plus_dynamics_conservative` vs `confidence_only`
  - `AUC-AUROC`: `+0.019857`
  - `AUC-SelAcc`: `+0.039444`
  - `Stop@100`: `+0.038889`

This means the best pattern was **not** replacing confidence with dynamics, but **using confidence as the backbone and only adding conservative dynamics where it helps**.

### Benchmark-selective dynamics routing

The dynamics plugin is explicitly local rather than global.

See:

- `result/dynamics_benchmark_router.md`
- `result/dynamics_v2_local_batch_summary.md`
- `result/early_stop_mean_confidence_plus_dyn_conservative_report.json`

The conservative router only enables changes on:

- `DS-R1/aime24`
- `DS-R1/aime25`
- `DS-R1/hmmt25`

and leaves the rest unchanged.

That is an important design choice in this repo:

- **do not globally turn on a plugin just because it helps somewhere**
- route by benchmark when the evidence is local

### Trimmed backbone sweep

I also explored trimmed confidence-style backbones.

See:

- `result/early_stop_mean_confidence_trimmed_alpha_sweep_eval.md`

Among `alpha = 0.12 / 0.14 / 0.18`, the selected variant was:

- `alpha = 0.18`

with the corresponding final ready-to-submit artifacts such as:

- `result/early_stop_submission_ready_alpha018.json`
- `result/early_stop_submission_ready_alpha018_validate.json`

### Validation artifacts

There are several submission-ready or validation-ready early-stop outputs, for example:

- `result/early_stop_v6_1_balanced.json`
- `result/early_stop_v6_1_stable.json`
- `result/early_stop_submission_ready_alpha014.json`
- `result/early_stop_submission_ready_alpha018.json`

and the validation reports confirm the expected early-stop task structure over:

- `12` caches
- `970` problems
- `62080` samples

---

## Main takeaways

If I had to compress the work into a few conclusions, they would be:

1. **Best-of-N improved more from conservative local corrections than from unified global redesigns.**
2. **Activation signals are useful for analysis, but not yet reliable enough as a primary submission rule.**
3. **Confidence is the strongest early-stop backbone in this workspace.**
4. **Dynamics helps most when routed selectively, not when enabled everywhere.**
5. **Many promising local gains disappear on full leaderboard evaluation, so stability matters more than cleverness.**

---

## Repository map

### Core code

- `nad/`: main Python package
- `plugins/`: custom selector implementations
- `scripts/`: experiment and submission-building scripts
- `tools/`: utilities
- `minimal_visualization_next/`: local visualization app

### Experiment records

- `WORKLOG.md`: earlier compact work summary
- `WORKLOG2.0.md`: more detailed second-stage log
- `result/`: experiment outputs, notes, summaries, reports, model artifacts

### Particularly useful artifacts

- **Best-of-N backbone**:
  - `result/best_of_n_nad_mixed_v1_complete.json`
  - `result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json`
- **Best-of-N ablations / rejected directions**:
  - `result/best_of_n_hybrid_v1_wrapped.json`
  - `result/best_of_n_nad_mixed_v3_aime_top2_gap1e3_logprob_tailveto_submit.json`
- **Best-of-N specialists / EM variants**:
  - `result/best_of_n_em_regularized_m2_seed42_keepall.json`
  - `result/best_of_n_em_regularized_m4_seed42_keepall.json`
  - `result/best_of_n_specialists_v2_aime_mixedv2_science_em_m4_coding_em_m2.json`
- **Best-of-N distilled rules**:
  - `result/rule_distillation_summary.json`
  - `result/best_of_n_nad_mixed_v7_token_only_conservative_v1.json`
- **Early-stop backbone and routing**:
  - `result/early_stop_mean_confidence_variants_eval.md`
  - `result/dynamics_benchmark_router.md`
  - `result/early_stop_mean_confidence_plus_dyn_conservative_report.json`
- **Early-stop final candidates**:
  - `result/early_stop_submission_ready_alpha014.json`
  - `result/early_stop_submission_ready_alpha018.json`

---

## Reproducing or extending the work

Run commands from the repository root.

### Environment

```bash
bash cookbook/00_setup/install.sh
bash cookbook/00_setup/verify.sh
```

### General NAD analysis

```bash
python3 -m nad.cli analyze \
  --cache-root MUI_HUB/cache/... \
  --selectors all \
  --out result.json

python3 -m nad.cli accuracy \
  --selection result.json \
  --cache-root MUI_HUB/cache/... \
  --out accuracy.json
```

### Ranking historical selector runs

```bash
python3 scripts/rank_selectors.py \
  --results-dir ./result/all_model_TIMESTAMP \
  --csv --json
```

### Local browser tools

```bash
bash cookbook/01_cache_browser/cache_browser.sh --background
```

---

## Notes

- `MUI_HUB` is a local symlink to cache storage and is **not** bundled as GitHub-friendly data.
- Many files in `result/` are generated artifacts, intermediate submissions, or evaluation notes.
- This repo intentionally keeps **negative results** and **rejected ideas**, because they explain why later rules became much more conservative.

---

## If you are reading this on GitHub

The best place to start is:

1. `WORKLOG.md`
2. `WORKLOG2.0.md`
3. `result/rule_distillation_summary.json`
4. `result/early_stop_mean_confidence_variants_eval.md`
5. `result/dynamics_benchmark_router.md`

These five files together capture most of the research story of this repository.
