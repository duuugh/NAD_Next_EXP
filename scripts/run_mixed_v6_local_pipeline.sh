#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_INPUT="${1:-$ROOT_DIR/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json}"
TABLE_CSV="${2:-$ROOT_DIR/result/mixed_v6_local_training_table.csv}"
TABLE_META="${3:-$ROOT_DIR/result/mixed_v6_local_training_table_meta.json}"
ABLAT_CSV="${4:-$ROOT_DIR/result/mixed_v6_bucket_ablation.csv}"
ABLAT_JSON="${5:-$ROOT_DIR/result/mixed_v6_bucket_ablation.json}"
MODEL_PKL="${6:-$ROOT_DIR/result/mixed_v6_local_head.pkl}"
METRICS_JSON="${7:-$ROOT_DIR/result/mixed_v6_local_head_metrics.json}"
OUT_JSON="${8:-$ROOT_DIR/result/best_of_n_nad_mixed_v6_local_head_submit.json}"
OUT_NOTES="${9:-$ROOT_DIR/result/best_of_n_nad_mixed_v6_local_head_submit_notes.json}"

python3 "$ROOT_DIR/scripts/build_mixed_v6_local_training_table.py" \
  --input "$BASE_INPUT" \
  --output "$TABLE_CSV" \
  --meta-output "$TABLE_META" \
  --aime-only \
  --min-gap 0 \
  --max-gap 0.005 \
  --bucket-edges 5e-4,1e-3,2e-3,5e-3

python3 "$ROOT_DIR/scripts/eval_mixed_v6_bucket_ablation.py" \
  --input "$TABLE_CSV" \
  --output-csv "$ABLAT_CSV" \
  --output-json "$ABLAT_JSON" \
  --bucket-edges 5e-4,1e-3,2e-3,5e-3 \
  --logprob-delta-threshold 0 \
  --selfcert-delta-threshold 0

python3 "$ROOT_DIR/scripts/train_mixed_v6_local_head.py" \
  --input "$TABLE_CSV" \
  --model-output "$MODEL_PKL" \
  --metrics-output "$METRICS_JSON" \
  --target-max-flip-rate 0.08

python3 "$ROOT_DIR/scripts/apply_mixed_v6_local_head.py" \
  --input "$BASE_INPUT" \
  --model "$MODEL_PKL" \
  --output "$OUT_JSON" \
  --notes-output "$OUT_NOTES" \
  --aime-only \
  --min-gap 0 \
  --max-gap 0.005

echo "Done:"
echo "  table   -> $TABLE_CSV"
echo "  ablation-> $ABLAT_CSV"
echo "  model   -> $MODEL_PKL"
echo "  submit  -> $OUT_JSON"
