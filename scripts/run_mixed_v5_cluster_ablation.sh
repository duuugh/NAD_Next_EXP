#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jovyan/work/NAD_Next"
BASE_INPUT="${1:-$ROOT_DIR/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json}"
OUT_DIR="${2:-$ROOT_DIR/result}"

mkdir -p "$OUT_DIR"

run_case () {
  local name="$1"
  shift
  echo "[run] $name"
  python3 "$ROOT_DIR/scripts/build_mixed_v5_cluster_router.py" \
    --input "$BASE_INPUT" \
    --method-name "$name" \
    --output "$OUT_DIR/best_of_n_${name}_submit.json" \
    --notes-output "$OUT_DIR/best_of_n_${name}_submit_notes.json" \
    "$@"
}

# A1: AIME-only minimal upgrade, base_max + eps*base_mass
run_case "nad_mixed_v5_aime_cluster_mass_only" \
  --target-cache-keys "DS-R1/aime24,DS-R1/aime25,Qwen3-4B/aime24,Qwen3-4B/aime25" \
  --tau-max-gap 0.001 --tau-mass-gap 0.002 \
  --cluster-topr 3 \
  --disable-aux-rerank --disable-count-rerank

# A2: Trigger relaxed
run_case "nad_mixed_v5_aime_cluster_mass_only_tau_relaxed" \
  --target-cache-keys "DS-R1/aime24,DS-R1/aime25,Qwen3-4B/aime24,Qwen3-4B/aime25" \
  --tau-max-gap 0.002 --tau-mass-gap 0.005 \
  --cluster-topr 3 \
  --disable-aux-rerank --disable-count-rerank

# A3: Trigger strict
run_case "nad_mixed_v5_aime_cluster_mass_only_tau_strict" \
  --target-cache-keys "DS-R1/aime24,DS-R1/aime25,Qwen3-4B/aime24,Qwen3-4B/aime25" \
  --tau-max-gap 0.0005 --tau-mass-gap 0.001 \
  --cluster-topr 3 \
  --disable-aux-rerank --disable-count-rerank

# A4: Add aux in triggered bucket only
run_case "nad_mixed_v5_aime_cluster_mass_aux" \
  --target-cache-keys "DS-R1/aime24,DS-R1/aime25,Qwen3-4B/aime24,Qwen3-4B/aime25" \
  --tau-max-gap 0.001 --tau-mass-gap 0.002 \
  --cluster-topr 3 \
  --metric tok_logprob \
  --disable-count-rerank

# A5: Expand to Qwen compressed branches
run_case "nad_mixed_v5_aime_plus_qwen_comp_cluster" \
  --target-cache-keys "DS-R1/aime24,DS-R1/aime25,Qwen3-4B/aime24,Qwen3-4B/aime25,Qwen3-4B/gpqa,Qwen3-4B/brumo25,Qwen3-4B/hmmt25" \
  --enable-qwen-compressed \
  --tau-max-gap 0.001 --tau-mass-gap 0.002 \
  --cluster-topr 3 \
  --metric tok_logprob \
  --disable-count-rerank

echo "done. outputs in: $OUT_DIR"
