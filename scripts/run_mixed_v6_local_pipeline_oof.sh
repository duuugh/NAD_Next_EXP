#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_INPUT="${1:-$ROOT_DIR/result/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json}"
TABLE_CSV="${2:-$ROOT_DIR/result/mixed_v6_local_training_table.csv}"
TABLE_META="${3:-$ROOT_DIR/result/mixed_v6_local_training_table_meta.json}"
ABLAT_CSV="${4:-$ROOT_DIR/result/mixed_v6_bucket_ablation.csv}"
ABLAT_JSON="${5:-$ROOT_DIR/result/mixed_v6_bucket_ablation.json}"

OOF_PRED="${6:-$ROOT_DIR/result/mixed_v6_local_oof_predictions.csv}"
OOF_SUMMARY="${7:-$ROOT_DIR/result/mixed_v6_local_oof_summary.json}"
FULL_MODEL="${8:-$ROOT_DIR/result/mixed_v6_local_head_full.pkl}"
SWEEP_G_CSV="${9:-$ROOT_DIR/result/mixed_v6_threshold_sweep_global.csv}"
SWEEP_G_JSON="${10:-$ROOT_DIR/result/mixed_v6_threshold_sweep_global.json}"
SWEEP_B_CSV="${11:-$ROOT_DIR/result/mixed_v6_threshold_sweep_per_bucket.csv}"
SWEEP_B_JSON="${12:-$ROOT_DIR/result/mixed_v6_threshold_sweep_per_bucket.json}"
SEL_JSON="${13:-$ROOT_DIR/result/mixed_v6_selected_thresholds.json}"

CMP_CSV="${14:-$ROOT_DIR/result/mixed_v6_rule_vs_head_comparison.csv}"
CMP_JSON="${15:-$ROOT_DIR/result/mixed_v6_rule_vs_head_comparison.json}"
AUDIT_CSV="${16:-$ROOT_DIR/result/mixed_v6_flip_audit.csv}"

OUT_G_JSON="${17:-$ROOT_DIR/result/best_of_n_nad_mixed_v6_local_head_oof_global_submit.json}"
OUT_G_NOTES="${18:-$ROOT_DIR/result/best_of_n_nad_mixed_v6_local_head_oof_global_submit_notes.json}"
OUT_B_JSON="${19:-$ROOT_DIR/result/best_of_n_nad_mixed_v6_local_head_oof_bucket_submit.json}"
OUT_B_NOTES="${20:-$ROOT_DIR/result/best_of_n_nad_mixed_v6_local_head_oof_bucket_submit_notes.json}"

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

python3 "$ROOT_DIR/scripts/train_mixed_v6_local_head_oof.py" \
  --input "$TABLE_CSV" \
  --n-splits 5 \
  --seed 42 \
  --lambda-fp 2.0 \
  --oof-pred-output "$OOF_PRED" \
  --oof-summary-output "$OOF_SUMMARY" \
  --full-model-output "$FULL_MODEL" \
  --sweep-global-csv "$SWEEP_G_CSV" \
  --sweep-global-json "$SWEEP_G_JSON" \
  --sweep-bucket-csv "$SWEEP_B_CSV" \
  --sweep-bucket-json "$SWEEP_B_JSON" \
  --selected-thresholds-output "$SEL_JSON"

python3 "$ROOT_DIR/scripts/eval_mixed_v6_rule_vs_head.py" \
  --oof-predictions "$OOF_PRED" \
  --selected-thresholds "$SEL_JSON" \
  --lambda-fp 2.0 \
  --output-csv "$CMP_CSV" \
  --output-json "$CMP_JSON" \
  --audit-csv "$AUDIT_CSV"

python3 "$ROOT_DIR/scripts/apply_mixed_v6_local_head_oof_selected.py" \
  --input "$BASE_INPUT" \
  --model "$FULL_MODEL" \
  --selected-thresholds "$SEL_JSON" \
  --mode global \
  --output "$OUT_G_JSON" \
  --notes-output "$OUT_G_NOTES" \
  --aime-only \
  --min-gap 0 \
  --max-gap 0.005

python3 "$ROOT_DIR/scripts/apply_mixed_v6_local_head_oof_selected.py" \
  --input "$BASE_INPUT" \
  --model "$FULL_MODEL" \
  --selected-thresholds "$SEL_JSON" \
  --mode per_bucket \
  --output "$OUT_B_JSON" \
  --notes-output "$OUT_B_NOTES" \
  --aime-only \
  --min-gap 0 \
  --max-gap 0.005

echo "Done OOF pipeline. Key outputs:"
echo "  OOF preds         -> $OOF_PRED"
echo "  Thresholds        -> $SEL_JSON"
echo "  Rule vs head      -> $CMP_CSV"
echo "  Flip audit        -> $AUDIT_CSV"
echo "  Apply global      -> $OUT_G_JSON"
echo "  Apply per-bucket  -> $OUT_B_JSON"
