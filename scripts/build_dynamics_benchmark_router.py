#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


MODES = (
    'rho_tail_only',
    'neg_A_accel_only',
    'rho_tail_plus_neg_A_accel',
)

DEFAULT_POLICY_UNIVERSE = (
    'DS-R1/aime24',
    'DS-R1/aime25',
    'DS-R1/hmmt25',
    'DS-R1/gpqa',
    'DS-R1/lcb_v5',
    'DS-R1/brumo25',
)

PRIOR_INJECTIONS = {
    'DS-R1/aime24': 'rho_tail_plus_neg_A_accel',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build rule-based benchmark router from dynamics v2 local batch summary.')
    parser.add_argument(
        '--input-summary',
        default='/home/jovyan/work/NAD_Next/result/dynamics_v2_local_batch_summary.json',
        help='Input dynamics v2 local batch summary JSON.',
    )
    parser.add_argument(
        '--output-json',
        default='/home/jovyan/work/NAD_Next/result/dynamics_benchmark_router.json',
        help='Main router JSON output path.',
    )
    parser.add_argument(
        '--output-markdown',
        default='/home/jovyan/work/NAD_Next/result/dynamics_benchmark_router.md',
        help='Router markdown report output path.',
    )
    parser.add_argument(
        '--output-conservative-policy',
        default='/home/jovyan/work/NAD_Next/result/dynamics_policy_conservative.json',
        help='Conservative policy JSON path.',
    )
    parser.add_argument(
        '--output-aggressive-policy',
        default='/home/jovyan/work/NAD_Next/result/dynamics_policy_aggressive.json',
        help='Aggressive policy JSON path.',
    )
    parser.add_argument(
        '--policy-universe',
        default=','.join(DEFAULT_POLICY_UNIVERSE),
        help='Comma-separated benchmark list to guarantee in final policies. Missing benchmarks default to disable.',
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'input summary not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def normalize_benchmark_name(name: str) -> str:
    text = str(name).strip()
    if '/' in text:
        return text
    return f'DS-R1/{text}'


def parse_policy_universe(text: str) -> List[str]:
    entries = [normalize_benchmark_name(x) for x in str(text).split(',') if x.strip()]
    return list(dict.fromkeys(entries))


def get_per_benchmark(summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    per = summary.get('per_benchmark')
    if isinstance(per, dict):
        return {normalize_benchmark_name(k): v for k, v in per.items()}
    raise ValueError('summary JSON missing dict field `per_benchmark`')


def parse_delta(mode_delta: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (
        float(mode_delta.get('delta AUC-AUROC', 0.0)),
        float(mode_delta.get('delta AUC-SelAcc', 0.0)),
        float(mode_delta.get('delta Stop@100', 0.0)),
    )


def pick_best_mode(benchmark_data: Mapping[str, Any]) -> str:
    delta_map = benchmark_data.get('delta_vs_baseline', {})
    if not isinstance(delta_map, Mapping):
        delta_map = {}

    available_modes = [mode for mode in MODES if mode in delta_map]
    if not available_modes:
        raise ValueError('delta_vs_baseline does not contain any supported mode')

    decision = benchmark_data.get('decision', {})
    recommended_mode = None
    if isinstance(decision, Mapping):
        recommended_mode = decision.get('recommended_mode')
    if isinstance(recommended_mode, str) and recommended_mode in available_modes:
        return recommended_mode

    return max(available_modes, key=lambda m: parse_delta(delta_map.get(m, {})))


def classify_strength(delta_auroc: float, delta_selacc: float, delta_stop: float) -> str:
    strong = (
        (delta_auroc >= 0.08 and delta_selacc >= -0.01)
        or (delta_auroc >= 0.03 and delta_selacc >= 0.0 and delta_stop >= 0.0)
    )
    if strong:
        return 'strong'

    borderline = (
        (delta_auroc >= 0.01 and delta_selacc >= 0.0)
        or (delta_auroc >= 0.0 and delta_selacc >= 0.003)
        or (delta_auroc >= 0.02 and delta_selacc > -0.01 and delta_stop >= 0.0)
    )
    if borderline:
        return 'borderline'

    return 'weak'


def route_by_strength(strength: str, best_mode: str) -> Tuple[str, str]:
    if strength == 'strong':
        return best_mode, best_mode
    if strength == 'borderline':
        return 'disable', best_mode
    return 'disable', 'disable'


def build_router_result(summary: Mapping[str, Any], policy_universe: Iterable[str]) -> Dict[str, Any]:
    per_benchmark_summary = get_per_benchmark(summary)
    per_benchmark_result: Dict[str, Dict[str, Any]] = {}
    conservative_policy: Dict[str, str] = {}
    aggressive_policy: Dict[str, str] = {}

    for benchmark, payload in per_benchmark_summary.items():
        best_mode = pick_best_mode(payload)
        delta_map = payload.get('delta_vs_baseline', {})
        if not isinstance(delta_map, Mapping):
            delta_map = {}
        delta_auroc, delta_selacc, delta_stop = parse_delta(delta_map.get(best_mode, {}))
        strength = classify_strength(delta_auroc, delta_selacc, delta_stop)
        conservative_route, aggressive_route = route_by_strength(strength, best_mode)

        per_benchmark_result[benchmark] = {
            'best_mode': best_mode,
            'delta_AUROC': delta_auroc,
            'delta_SelAcc': delta_selacc,
            'delta_Stop': delta_stop,
            'strength': strength,
            'conservative_route': conservative_route,
            'aggressive_route': aggressive_route,
            'source': 'auto_from_summary',
        }
        conservative_policy[benchmark] = conservative_route
        aggressive_policy[benchmark] = aggressive_route

    full_universe = list(dict.fromkeys([*policy_universe, *per_benchmark_result.keys(), *PRIOR_INJECTIONS.keys()]))
    for benchmark in full_universe:
        conservative_policy.setdefault(benchmark, 'disable')
        aggressive_policy.setdefault(benchmark, 'disable')

    manual_injections: Dict[str, Dict[str, str]] = {}
    for benchmark, mode in PRIOR_INJECTIONS.items():
        conservative_policy[benchmark] = mode
        aggressive_policy[benchmark] = mode
        manual_injections[benchmark] = {
            'route': mode,
            'reason': 'prior_validated_rule',
        }
        if benchmark not in per_benchmark_result:
            per_benchmark_result[benchmark] = {
                'best_mode': mode,
                'delta_AUROC': None,
                'delta_SelAcc': None,
                'delta_Stop': None,
                'strength': 'strong',
                'conservative_route': mode,
                'aggressive_route': mode,
                'source': 'prior_validated_rule',
            }

    return {
        'per_benchmark': dict(sorted(per_benchmark_result.items())),
        'policies': {
            'conservative': dict(sorted(conservative_policy.items())),
            'aggressive': dict(sorted(aggressive_policy.items())),
        },
        'manual_injections': dict(sorted(manual_injections.items())),
    }


def format_float(value: Any) -> str:
    if value is None:
        return 'N/A'
    return f'{float(value):+.6f}'


def build_markdown_report(result_payload: Mapping[str, Any], input_summary: str) -> str:
    per_benchmark = result_payload['per_benchmark']
    conservative = result_payload['policies']['conservative']
    aggressive = result_payload['policies']['aggressive']
    manual_injections = result_payload.get('manual_injections', {})

    auto_benchmarks = [
        bench for bench, info in per_benchmark.items()
        if str(info.get('source')) == 'auto_from_summary'
    ]
    strong = [bench for bench in auto_benchmarks if per_benchmark[bench]['strength'] == 'strong']
    borderline = [bench for bench in auto_benchmarks if per_benchmark[bench]['strength'] == 'borderline']
    weak = [bench for bench in auto_benchmarks if per_benchmark[bench]['strength'] == 'weak']

    expected_final = {
        'DS-R1/aime24': 'rho_tail_plus_neg_A_accel',
        'DS-R1/aime25': 'neg_A_accel_only',
        'DS-R1/hmmt25': 'rho_tail_plus_neg_A_accel',
        'DS-R1/gpqa': 'disable',
        'DS-R1/lcb_v5': 'disable',
    }
    conservative_core = {k: conservative.get(k, 'disable') for k in expected_final}
    is_expected_match = conservative_core == expected_final

    lines: List[str] = []
    lines.append('# Dynamics Benchmark Router Report')
    lines.append('')
    lines.append(f'- Input summary JSON: `{input_summary}`')
    lines.append('- Router type: rule-based (no training / no additional model)')
    lines.append('- Best mode candidates: `rho_tail_only`, `neg_A_accel_only`, `rho_tail_plus_neg_A_accel`')
    lines.append('')
    lines.append('## Per-benchmark Decisions')
    lines.append('')
    lines.append('| Benchmark | Best Mode | ΔAUROC | ΔSelAcc | ΔStop@100 | Strength | Conservative | Aggressive | Source |')
    lines.append('|---|---|---:|---:|---:|---|---|---|---|')
    for benchmark in sorted(per_benchmark):
        info = per_benchmark[benchmark]
        lines.append(
            f"| {benchmark} | {info['best_mode']} | {format_float(info['delta_AUROC'])} | "
            f"{format_float(info['delta_SelAcc'])} | {format_float(info['delta_Stop'])} | "
            f"{info['strength']} | {info['conservative_route']} | {info['aggressive_route']} | {info['source']} |"
        )

    lines.append('')
    lines.append('## Policies')
    lines.append('')
    lines.append('### Conservative')
    lines.append('')
    lines.append('| Benchmark | Route |')
    lines.append('|---|---|')
    for benchmark in sorted(conservative):
        lines.append(f'| {benchmark} | {conservative[benchmark]} |')

    lines.append('')
    lines.append('### Aggressive')
    lines.append('')
    lines.append('| Benchmark | Route |')
    lines.append('|---|---|')
    for benchmark in sorted(aggressive):
        lines.append(f'| {benchmark} | {aggressive[benchmark]} |')

    lines.append('')
    lines.append('## Manual Injection Notes')
    lines.append('')
    if manual_injections:
        for benchmark, info in sorted(manual_injections.items()):
            lines.append(
                f'- `{benchmark}` is injected as `{info["route"]}` by prior validated rule '
                f'(not auto-decided from this batch summary).'
            )
    else:
        lines.append('- No manual injection is applied.')

    lines.append('')
    lines.append('## Required Final Answers')
    lines.append('')
    lines.append(f'- Safe strong benchmarks now: {", ".join(strong) if strong else "(none)"}')
    lines.append(f'- Borderline-only benchmarks now: {", ".join(borderline) if borderline else "(none)"}')
    lines.append(f'- Weak benchmarks now: {", ".join(weak) if weak else "(none)"}')
    lines.append('- Conservative vs aggressive: conservative disables borderline, aggressive keeps borderline best_mode enabled.')
    lines.append('- Main leaderboard recommendation: conservative policy.')
    lines.append('- Should final recommended strategy be the provided 5-benchmark JSON?')
    if is_expected_match:
        lines.append('  - Yes. Router output matches that strategy on the 5 core benchmarks.')
    else:
        lines.append('  - No. Router output differs from the provided 5-benchmark strategy on core benchmarks.')
        lines.append(f'  - Router conservative core policy: `{json.dumps(conservative_core, ensure_ascii=False, sort_keys=True)}`')
        lines.append(f'  - Provided target strategy: `{json.dumps(expected_final, ensure_ascii=False, sort_keys=True)}`')
        lines.append('  - Difference reason: strict thresholding rules on this batch summary produce different classification/routes.')

    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    input_summary = Path(args.input_summary)
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_conservative = Path(args.output_conservative_policy)
    output_aggressive = Path(args.output_aggressive_policy)

    summary = load_json(input_summary)
    policy_universe = parse_policy_universe(args.policy_universe)
    router_result = build_router_result(summary, policy_universe=policy_universe)

    final_payload = {
        'input_summary': str(input_summary),
        'per_benchmark': router_result['per_benchmark'],
        'policies': router_result['policies'],
        'manual_injections': router_result['manual_injections'],
    }
    markdown = build_markdown_report(final_payload, input_summary=str(input_summary))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_conservative.parent.mkdir(parents=True, exist_ok=True)
    output_aggressive.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    output_markdown.write_text(markdown, encoding='utf-8')
    output_conservative.write_text(
        json.dumps(final_payload['policies']['conservative'], ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    output_aggressive.write_text(
        json.dumps(final_payload['policies']['aggressive'], ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(f'wrote {output_json}')
    print(f'wrote {output_markdown}')
    print(f'wrote {output_conservative}')
    print(f'wrote {output_aggressive}')


if __name__ == '__main__':
    main()
