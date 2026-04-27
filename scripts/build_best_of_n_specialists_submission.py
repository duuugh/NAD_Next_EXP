#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def parse_kv(text: str, sep: str = '=') -> tuple[str, str]:
    if sep not in text:
        raise ValueError(f"Expected '<key>{sep}<value>', got: {text}")
    key, value = text.split(sep, 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"Invalid mapping: {text}")
    return key, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a cache-level specialists best_of_n submission by routing each cache_key to a source submission file.'
    )
    parser.add_argument('--method-name', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--notes-output', required=True)

    parser.add_argument(
        '--source', action='append', default=[],
        help="Source submission alias mapping, format: alias=/path/to/submission.json (repeatable).",
    )
    parser.add_argument(
        '--map', action='append', default=[],
        help='Cache routing mapping, format: cache_key=source_alias (repeatable).',
    )

    parser.add_argument('--mapping-json', default=None, help='Optional JSON file with explicit cache_key -> source_alias mapping.')
    parser.add_argument('--default-source', required=True, help='Fallback source alias for cache_keys not explicitly mapped.')
    parser.add_argument('--base-source', default=None, help='Source alias used as canonical full cache-key set (default: same as --default-source).')
    return parser.parse_args()


def load_submission(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f'Invalid submission json object: {path}')
    if 'scores' not in obj or not isinstance(obj['scores'], dict):
        raise ValueError(f'Missing dict field `scores`: {path}')
    return obj


def main() -> None:
    args = parse_args()

    sources: Dict[str, Path] = {}
    for raw in args.source:
        alias, path = parse_kv(raw)
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f'Source not found for alias `{alias}`: {source_path}')
        sources[alias] = source_path

    if not sources:
        raise ValueError('At least one --source alias=path must be provided.')

    if args.default_source not in sources:
        raise ValueError(f'--default-source `{args.default_source}` not found in --source aliases: {sorted(sources)}')

    base_source = args.base_source or args.default_source
    if base_source not in sources:
        raise ValueError(f'--base-source `{base_source}` not found in --source aliases: {sorted(sources)}')

    explicit_mapping: Dict[str, str] = {}
    for raw in args.map:
        cache_key, alias = parse_kv(raw)
        explicit_mapping[cache_key] = alias

    if args.mapping_json:
        mp = json.loads(Path(args.mapping_json).read_text())
        if not isinstance(mp, dict):
            raise ValueError('--mapping-json must contain an object of cache_key -> source_alias')
        for cache_key, alias in mp.items():
            explicit_mapping[str(cache_key)] = str(alias)

    for cache_key, alias in explicit_mapping.items():
        if alias not in sources:
            raise ValueError(f'cache `{cache_key}` maps to unknown alias `{alias}`. Known aliases: {sorted(sources)}')

    loaded = {alias: load_submission(path) for alias, path in sources.items()}

    tasks = sorted({str(obj.get('task', '')) for obj in loaded.values()})
    if len(tasks) != 1:
        raise ValueError(f'Source submissions have inconsistent tasks: {tasks}')
    task = tasks[0]

    base_scores = loaded[base_source]['scores']
    target_cache_keys = sorted(base_scores.keys())

    out_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    cache_source_resolved: Dict[str, Dict[str, Any]] = {}

    for cache_key in target_cache_keys:
        source_alias = explicit_mapping.get(cache_key, args.default_source)
        source_obj = loaded[source_alias]
        source_scores = source_obj['scores']
        if cache_key not in source_scores:
            raise KeyError(
                f'cache_key `{cache_key}` missing from source alias `{source_alias}` ({sources[source_alias]}). '
                f'Available cache_keys: {sorted(source_scores.keys())}'
            )
        out_scores[cache_key] = source_scores[cache_key]
        cache_source_resolved[cache_key] = {
            'source_alias': source_alias,
            'source_path': str(sources[source_alias]),
            'problem_count': len(source_scores[cache_key]),
            'source_method_name': source_obj.get('method_name'),
        }

    out = {
        'task': task,
        'method_name': args.method_name,
        'scores': out_scores,
    }

    notes = {
        'task': task,
        'method_name': args.method_name,
        'output': str(args.output),
        'sources': {alias: str(path) for alias, path in sources.items()},
        'default_source': args.default_source,
        'base_source': base_source,
        'explicit_mapping': explicit_mapping,
        'resolved_cache_mapping': cache_source_resolved,
        'cache_key_count': len(target_cache_keys),
        'cache_keys': target_cache_keys,
    }

    output_path = Path(args.output)
    notes_path = Path(args.notes_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2))

    print(f'wrote {output_path}')
    print(f'wrote {notes_path}')
    print(f'cache_key_count: {len(target_cache_keys)}')


if __name__ == '__main__':
    main()
