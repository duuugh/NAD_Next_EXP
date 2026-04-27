#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an early-stop submission JSON against expected cache/problem/sample coverage and value schema."
        )
    )
    parser.add_argument("--input", required=True, help="Submission JSON to validate.")
    parser.add_argument(
        "--reference",
        default="/home/jovyan/work/NAD_Next/result/best_of_n_nad_mixed_v1_complete.json",
        help="Reference JSON defining required cache/problem/sample ids.",
    )
    parser.add_argument(
        "--expected-task",
        default="early_stop",
        help="Expected top-level `task` value. Use empty string to skip task check.",
    )
    parser.add_argument(
        "--check-range",
        action="store_true",
        help="Require each value to be in [0,1].",
    )
    parser.add_argument(
        "--check-monotonic",
        action="store_true",
        help="Require each 10-point list to be non-decreasing.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write machine-readable validation report.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=200,
        help="Maximum number of detailed errors to collect before truncation.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected top-level object in {path}")
    if "scores" not in obj or not isinstance(obj["scores"], dict):
        raise ValueError(f"Expected dict field `scores` in {path}")
    return obj


def err(errors: List[str], message: str, max_errors: int) -> None:
    if len(errors) < max_errors:
        errors.append(message)


def scalar_or_last(value: Any) -> float:
    if isinstance(value, list):
        if not value:
            raise ValueError("Expected non-empty list")
        return float(value[-1])
    return float(value)


def is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    f = float(value)
    return not (math.isnan(f) or math.isinf(f))


def compare_sets(label: str, actual: set[str], expected: set[str], errors: List[str], max_errors: int) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        err(errors, f"{label}: missing {len(missing)} ids, e.g. {missing[:5]}", max_errors)
    if extra:
        err(errors, f"{label}: extra {len(extra)} ids, e.g. {extra[:5]}", max_errors)


def build_expected_structure(reference_scores: Dict[str, Any]) -> Dict[str, Dict[str, set[str]]]:
    expected: Dict[str, Dict[str, set[str]]] = {}
    for cache_key, problem_map in reference_scores.items():
        if not isinstance(problem_map, dict):
            raise ValueError(f"Reference scores[{cache_key}] must be an object")
        out_problem: Dict[str, set[str]] = {}
        for problem_id, sample_map in problem_map.items():
            if not isinstance(sample_map, dict):
                raise ValueError(f"Reference scores[{cache_key}][{problem_id}] must be an object")
            sample_ids = set(str(sample_id) for sample_id in sample_map.keys())
            if not sample_ids:
                raise ValueError(f"Reference has empty sample set at {cache_key}/{problem_id}")
            # Probe one value to ensure parseable style (scalar or list) for sanity only.
            _ = scalar_or_last(next(iter(sample_map.values())))
            out_problem[str(problem_id)] = sample_ids
        expected[str(cache_key)] = out_problem
    return expected


def validate(args: argparse.Namespace) -> Tuple[bool, Dict[str, Any], List[str]]:
    sub_obj = load_json(Path(args.input))
    ref_obj = load_json(Path(args.reference))

    errors: List[str] = []
    summary: Dict[str, Any] = {
        "input": str(args.input),
        "reference": str(args.reference),
        "expected_task": args.expected_task,
        "cache_count": 0,
        "problem_count": 0,
        "sample_count": 0,
    }

    if args.expected_task and str(sub_obj.get("task", "")) != args.expected_task:
        err(errors, f"task mismatch: expected `{args.expected_task}`, got `{sub_obj.get('task')}`", args.max_errors)

    expected = build_expected_structure(ref_obj["scores"])
    actual_scores = sub_obj["scores"]

    compare_sets(
        "cache_keys",
        set(str(cache_key) for cache_key in actual_scores.keys()),
        set(expected.keys()),
        errors,
        args.max_errors,
    )

    for cache_key, expected_problems in expected.items():
        if cache_key not in actual_scores:
            continue
        actual_problem_map = actual_scores[cache_key]
        if not isinstance(actual_problem_map, dict):
            err(errors, f"scores[{cache_key}] must be an object", args.max_errors)
            continue

        compare_sets(
            f"{cache_key}/problem_ids",
            set(str(problem_id) for problem_id in actual_problem_map.keys()),
            set(expected_problems.keys()),
            errors,
            args.max_errors,
        )

        for problem_id, expected_samples in expected_problems.items():
            if problem_id not in actual_problem_map:
                continue
            sample_map = actual_problem_map[problem_id]
            if not isinstance(sample_map, dict):
                err(errors, f"scores[{cache_key}][{problem_id}] must be an object", args.max_errors)
                continue

            compare_sets(
                f"{cache_key}/{problem_id}/sample_ids",
                set(str(sample_id) for sample_id in sample_map.keys()),
                expected_samples,
                errors,
                args.max_errors,
            )

            for sample_id in expected_samples:
                if sample_id not in sample_map:
                    continue
                seq = sample_map[sample_id]
                if not isinstance(seq, list):
                    err(errors, f"{cache_key}/{problem_id}/{sample_id}: expected list, got {type(seq).__name__}", args.max_errors)
                    continue
                if len(seq) != 10:
                    err(errors, f"{cache_key}/{problem_id}/{sample_id}: expected 10 values, got {len(seq)}", args.max_errors)
                    continue

                numeric_seq: List[float] = []
                bad_value = False
                for idx, value in enumerate(seq):
                    if not is_finite_number(value):
                        err(
                            errors,
                            f"{cache_key}/{problem_id}/{sample_id}[{idx}]: non-finite/non-numeric value {value}",
                            args.max_errors,
                        )
                        bad_value = True
                        break
                    numeric_seq.append(float(value))

                if bad_value:
                    continue

                if args.check_range:
                    for idx, value in enumerate(numeric_seq):
                        if value < 0.0 or value > 1.0:
                            err(
                                errors,
                                f"{cache_key}/{problem_id}/{sample_id}[{idx}]: out of [0,1], got {value}",
                                args.max_errors,
                            )
                            break

                if args.check_monotonic:
                    for idx in range(1, len(numeric_seq)):
                        if numeric_seq[idx] + 1e-12 < numeric_seq[idx - 1]:
                            err(
                                errors,
                                (
                                    f"{cache_key}/{problem_id}/{sample_id}: not non-decreasing at "
                                    f"index {idx - 1}->{idx} ({numeric_seq[idx - 1]} -> {numeric_seq[idx]})"
                                ),
                                args.max_errors,
                            )
                            break

                summary["sample_count"] += 1

            summary["problem_count"] += 1

        summary["cache_count"] += 1

    ok = len(errors) == 0
    summary["ok"] = ok
    summary["error_count"] = len(errors)
    summary["truncated"] = len(errors) >= args.max_errors
    return ok, summary, errors


def main() -> None:
    args = parse_args()
    ok, summary, errors = validate(args)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"summary": summary, "errors": errors}, ensure_ascii=False, indent=2))
        print(f"wrote {report_path}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print("errors:")
        for item in errors[: min(20, len(errors))]:
            print(f"- {item}")
        if len(errors) > 20:
            print(f"- ... and {len(errors) - 20} more")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
