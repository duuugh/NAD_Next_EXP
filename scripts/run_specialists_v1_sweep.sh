#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jovyan/work/NAD_Next"
RESULT_DIR="$ROOT_DIR/result"
BUILDER="$ROOT_DIR/scripts/build_best_of_n_specialists_submission.py"

AIME_SRC="$RESULT_DIR/best_of_n_nad_mixed_v2_aime_top2_gap1e3_logprob_submit.json"
MIXEDV1_SRC="$RESULT_DIR/best_of_n_nad_mixed_v1_complete.json"
EM2_SRC="$RESULT_DIR/best_of_n_em_regularized_m2_seed42_keepall.json"
EM4_SRC="$RESULT_DIR/best_of_n_em_regularized_m4_seed42_keepall.json"
EM8_SRC="$RESULT_DIR/best_of_n_em_regularized_m8_seed42_keepall.json"

AIME_KEYS=("DS-R1/aime24" "DS-R1/aime25" "Qwen3-4B/aime24" "Qwen3-4B/aime25")
SCIENCE_KEYS=("DS-R1/gpqa" "DS-R1/brumo25" "DS-R1/hmmt25" "Qwen3-4B/gpqa" "Qwen3-4B/brumo25" "Qwen3-4B/hmmt25")
CODING_KEYS=("DS-R1/lcb_v5" "Qwen3-4B/lcb_v5")

mkdir -p "$RESULT_DIR"

python3 - <<'PY'
import json
from pathlib import Path
root=Path('/home/jovyan/work/NAD_Next/result')
patterns=[
'best_of_n_nad_mixed_v1_complete*.json',
'best_of_n_nad_mixed_v2*.json',
'best_of_n_em_regularized_m2*.json',
'best_of_n_em_regularized_m4*.json',
'best_of_n_em_regularized_m8*.json',
]
items=[]
seen=set()
for p in patterns:
    for fp in sorted(root.glob(p)):
        if fp.name.endswith('_notes.json'):
            continue
        if fp in seen:
            continue
        seen.add(fp)
        obj=json.loads(fp.read_text())
        scores=obj.get('scores',{}) if isinstance(obj,dict) else {}
        looks_full=False
        for _, pm in scores.items():
            for _, sid_scores in pm.items():
                looks_full = len(sid_scores) > 1
                break
            break
        items.append({
            'path': str(fp),
            'method_name': obj.get('method_name') if isinstance(obj,dict) else None,
            'task': obj.get('task') if isinstance(obj,dict) else None,
            'cache_keys': sorted(scores.keys()) if isinstance(scores,dict) else [],
            'cache_key_count': len(scores) if isinstance(scores,dict) else 0,
            'looks_submission_safe_full_scores': looks_full,
            'notes_path': str(fp.with_name(fp.stem + '_notes.json')) if fp.with_name(fp.stem + '_notes.json').exists() else None,
        })
out={'inventory_generated_from_patterns':patterns,'count':len(items),'items':items}
(root/'specialists_inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('inventory:', len(items), 'files')
PY

have_file () {
  local path="$1"
  [[ -f "$path" ]]
}

build_main_case () {
  local method_name="$1"
  local base_src="$2"
  local output="$RESULT_DIR/best_of_n_${method_name}.json"
  local notes="$RESULT_DIR/best_of_n_${method_name}_notes.json"

  python3 "$BUILDER" \
    --method-name "$method_name" \
    --source "aime=$AIME_SRC" \
    --source "base=$base_src" \
    --default-source base \
    --base-source base \
    --map "DS-R1/aime24=aime" \
    --map "DS-R1/aime25=aime" \
    --map "Qwen3-4B/aime24=aime" \
    --map "Qwen3-4B/aime25=aime" \
    --output "$output" \
    --notes-output "$notes"

  GENERATED+=("$output" "$notes")
}

build_variant_science_coding () {
  local method_name="specialists_v2_aime_mixedv2_science_em_m4_coding_em_m2"
  local output="$RESULT_DIR/best_of_n_${method_name}.json"
  local notes="$RESULT_DIR/best_of_n_${method_name}_notes.json"

  python3 "$BUILDER" \
    --method-name "$method_name" \
    --source "aime=$AIME_SRC" \
    --source "fallback=$MIXEDV1_SRC" \
    --source "science=$EM4_SRC" \
    --source "coding=$EM2_SRC" \
    --default-source fallback \
    --base-source fallback \
    --map "DS-R1/aime24=aime" \
    --map "DS-R1/aime25=aime" \
    --map "Qwen3-4B/aime24=aime" \
    --map "Qwen3-4B/aime25=aime" \
    --map "DS-R1/gpqa=science" \
    --map "DS-R1/brumo25=science" \
    --map "DS-R1/hmmt25=science" \
    --map "Qwen3-4B/gpqa=science" \
    --map "Qwen3-4B/brumo25=science" \
    --map "Qwen3-4B/hmmt25=science" \
    --map "DS-R1/lcb_v5=coding" \
    --map "Qwen3-4B/lcb_v5=coding" \
    --output "$output" \
    --notes-output "$notes"

  GENERATED+=("$output" "$notes")
}

build_variant_model_side () {
  local method_name="specialists_v2_aime_mixedv2_ds_em_m4_qwen_em_m2"
  local output="$RESULT_DIR/best_of_n_${method_name}.json"
  local notes="$RESULT_DIR/best_of_n_${method_name}_notes.json"

  python3 "$BUILDER" \
    --method-name "$method_name" \
    --source "aime=$AIME_SRC" \
    --source "fallback=$MIXEDV1_SRC" \
    --source "ds=$EM4_SRC" \
    --source "qwen=$EM2_SRC" \
    --default-source fallback \
    --base-source fallback \
    --map "DS-R1/aime24=aime" \
    --map "DS-R1/aime25=aime" \
    --map "Qwen3-4B/aime24=aime" \
    --map "Qwen3-4B/aime25=aime" \
    --map "DS-R1/gpqa=ds" \
    --map "DS-R1/brumo25=ds" \
    --map "DS-R1/hmmt25=ds" \
    --map "DS-R1/lcb_v5=ds" \
    --map "Qwen3-4B/gpqa=qwen" \
    --map "Qwen3-4B/brumo25=qwen" \
    --map "Qwen3-4B/hmmt25=qwen" \
    --map "Qwen3-4B/lcb_v5=qwen" \
    --output "$output" \
    --notes-output "$notes"

  GENERATED+=("$output" "$notes")
}

GENERATED=()

if ! have_file "$AIME_SRC"; then
  echo "[error] missing AIME source: $AIME_SRC" >&2
  exit 1
fi

if have_file "$MIXEDV1_SRC"; then
  build_main_case "specialists_v1_aime_mixedv2_else_mixedv1" "$MIXEDV1_SRC"
else
  echo "[warn] skip mixedv1 main case: missing $MIXEDV1_SRC"
fi

if have_file "$EM2_SRC"; then
  build_main_case "specialists_v1_aime_mixedv2_else_em_m2" "$EM2_SRC"
else
  echo "[warn] skip em_m2 main case: missing $EM2_SRC"
fi

if have_file "$EM4_SRC"; then
  build_main_case "specialists_v1_aime_mixedv2_else_em_m4" "$EM4_SRC"
else
  echo "[warn] skip em_m4 main case: missing $EM4_SRC"
fi

if have_file "$EM8_SRC"; then
  build_main_case "specialists_v1_aime_mixedv2_else_em_m8" "$EM8_SRC"
else
  echo "[warn] skip em_m8 main case: missing $EM8_SRC"
fi

if have_file "$MIXEDV1_SRC" && have_file "$EM2_SRC" && have_file "$EM4_SRC"; then
  build_variant_science_coding
  build_variant_model_side
else
  echo "[info] optional variants skipped due to missing source files"
fi

echo "[done] generated files:"
for f in "${GENERATED[@]}"; do
  echo " - $f"
done
