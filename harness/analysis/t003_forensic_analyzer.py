#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = DEFAULT_ROOT / 'harness' / 'reports' / 'v06_confirmatory_deepseek.json'
DEFAULT_OUTPUT = DEFAULT_ROOT / 'harness' / 'analysis' / 't003_forensic_report.md'

REPRESENTATION_ARMS = [
    'selection_representation',
    'selection_representation_A',
    'selection_representation_B',
]

BASELINE_ARM = 'selection_only'
TASK_ID = 'T003'
EXPECTED_REPETITIONS = list(range(1, 11))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError('Report root must be a JSON object')

    return data


def flatten_runs(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = report.get('runs')

    if isinstance(runs, list):
        return [r for r in runs if isinstance(r, dict)]

    flattened: List[Dict[str, Any]] = []

    for task in report.get('tasks', []):
        if not isinstance(task, dict):
            continue

        task_id = task.get('task_id')
        arms = task.get('runs') or task.get('arms')

        if isinstance(arms, dict):
            for arm_id, entries in arms.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            row = dict(entry)
                            row.setdefault('task_id', task_id)
                            row.setdefault('arm_id', arm_id)
                            flattened.append(row)

    return flattened


def metric(row: Dict[str, Any], name: str) -> int:
    try:
        return int(row['metrics'][name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Run {row.get('run_id', '<unknown>')} missing numeric metrics.{name}"
        ) from exc


def rep(row: Dict[str, Any]) -> int:
    try:
        return int(row['repetition'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Run {row.get('run_id', '<unknown>')} has invalid repetition"
        ) from exc


def arm(row: Dict[str, Any]) -> str:
    value = row.get('arm_id')

    if not isinstance(value, str):
        raise ValueError(
            f"Run {row.get('run_id', '<unknown>')} missing arm_id"
        )

    return value


def index_t003(
    rows: List[Dict[str, Any]]
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for row in rows:
        if row.get('task_id') != TASK_ID:
            continue

        key = (arm(row), rep(row))

        if key in index:
            raise ValueError(
                f'Duplicate T003 run for arm={key[0]} repetition={key[1]}'
            )

        index[key] = row

    return index


def classify(
    delta_reasoning: int,
    delta_total: int,
    epsilon: float = 0.0
) -> str:

    reasoning_zero = abs(delta_reasoning) <= epsilon
    total_zero = abs(delta_total) <= epsilon

    if delta_reasoning < -epsilon and delta_total < -epsilon:
        return 'BENEFICIAL'

    if reasoning_zero:
        return 'NEUTRAL_REASONING'

    if delta_reasoning > epsilon and delta_total > epsilon:
        return 'HARMFUL'

    if delta_reasoning < -epsilon and total_zero:
        return 'REASONING_REDUCED_TOTAL_NEUTRAL'

    if delta_reasoning > epsilon and total_zero:
        return 'REASONING_INCREASED_TOTAL_NEUTRAL'

    if delta_reasoning < -epsilon and delta_total > epsilon:
        return 'TRADEOFF_REASONING_DOWN_TOTAL_UP'

    if delta_reasoning > epsilon and delta_total < -epsilon:
        return 'TRADEOFF_REASONING_UP_TOTAL_DOWN'

    return 'MIXED_OR_ZERO_TOTAL'


def pair_rows(
    index: Dict[Tuple[str, int], Dict[str, Any]],
    treatment_arm: str
) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    for repetition in EXPECTED_REPETITIONS:
        baseline = index.get((BASELINE_ARM, repetition))
        treatment = index.get((treatment_arm, repetition))

        if baseline is None or treatment is None:
            continue

        delta_reasoning = (
            metric(treatment, 'reasoning_tokens')
            - metric(baseline, 'reasoning_tokens')
        )

        delta_total = (
            metric(treatment, 'total_tokens')
            - metric(baseline, 'total_tokens')
        )

        rows.append({
            'repetition': repetition,
            'delta_input': (
                metric(treatment, 'input_tokens')
                - metric(baseline, 'input_tokens')
            ),
            'delta_reasoning': delta_reasoning,
            'delta_output': (
                metric(treatment, 'output_tokens')
                - metric(baseline, 'output_tokens')
            ),
            'delta_total': delta_total,
            'delta_latency': (
                float(treatment['metrics']['latency_ms'])
                - float(baseline['metrics']['latency_ms'])
            ),
            'classification': classify(
                delta_reasoning,
                delta_total
            ),
            'baseline': baseline,
            'treatment': treatment,
        })

    return rows


def fmt_num(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:+.2f}'

    return f'{value:+d}'


def raw_text(row: Dict[str, Any]) -> str:
    value = row.get('raw_text')

    if value is None:
        return '[RAW_TEXT_MISSING]'

    return str(value)


def output_text(row: Dict[str, Any]) -> str:
    value = row.get('output')

    if value is None:
        return '[OUTPUT_MISSING]'

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2
        )
    except TypeError:
        return str(value)


def build_report(
    report: Dict[str, Any],
    index: Dict[Tuple[str, int], Dict[str, Any]]
) -> str:

    all_t003 = [
        run
        for (arm_id, _), run in index.items()
    ]

    lines: List[str] = []

    lines.append('# PromptForge v0.6 — T003 Forensic Autopsy')
    lines.append('')

    lines.append('## Scope and integrity')
    lines.append('')

    lines.append(
        f'- Source: `{DEFAULT_REPORT}`'
    )
    lines.append(
        '- Analysis mode: **static local read-only**'
    )
    lines.append(
        '- API calls: **0**'
    )
    lines.append(
        '- Runner modified/executed: **NO**'
    )
    lines.append(
        '- Baseline modified: **NO**'
    )
    lines.append(
        f'- Task isolated: **{TASK_ID}**'
    )
    lines.append(
        f'- Baseline for all representation comparisons: **{BASELINE_ARM}**'
    )
    lines.append(
        '- Reasoning neutrality epsilon: **0 tokens** '
        '(integral token counters; zero is neutral)'
    )
    lines.append('')

    lines.append(
        'The classifications below are purely mathematical. '
        'They do not infer internal model cognition.'
    )
    lines.append('')

    lines.append('## 1. T003 integrity check')
    lines.append('')

    expected_arms = [BASELINE_ARM] + REPRESENTATION_ARMS

    for arm_id in expected_arms:
        repetitions = sorted(
            rep(run)
            for (current_arm, _), run in index.items()
            if current_arm == arm_id
        )

        missing = sorted(
            set(EXPECTED_REPETITIONS) - set(repetitions)
        )

        extra = sorted(
            set(repetitions) - set(EXPECTED_REPETITIONS)
        )

        status = (
            'PASS'
            if repetitions == EXPECTED_REPETITIONS
            else 'FAIL'
        )

        lines.append(
            f'- `{arm_id}`: {status}; '
            f'repetitions={repetitions}; '
            f'missing={missing or "none"}; '
            f'extra={extra or "none"}'
        )

    lines.append(
        f'- T003 run count across isolated arms: **{len(all_t003)}**'
    )
    lines.append('')

    pair_cache: Dict[str, List[Dict[str, Any]]] = {}

    for treatment_arm in REPRESENTATION_ARMS:
        pair_cache[treatment_arm] = pair_rows(
            index,
            treatment_arm
        )

    lines.append('## 2. Extremes and per-repetition regimes')
    lines.append('')

    lines.append(
        'All deltas are treatment minus `selection_only`, '
        'aligned by identical repetition.'
    )
    lines.append('')

    for treatment_arm in REPRESENTATION_ARMS:

        rows = pair_cache[treatment_arm]

        lines.append(
            f'### `{treatment_arm}` vs `{BASELINE_ARM}`'
        )
        lines.append('')

        lines.append(
            '| Rep | Δinput | Δreasoning | Δoutput | Δtotal | '
            'Δlatency ms | Regime |'
        )
        lines.append(
            '|---:|---:|---:|---:|---:|---:|---|'
        )

        for row in rows:
            lines.append(
                f"| {row['repetition']} | "
                f"{fmt_num(row['delta_input'])} | "
                f"{fmt_num(row['delta_reasoning'])} | "
                f"{fmt_num(row['delta_output'])} | "
                f"{fmt_num(row['delta_total'])} | "
                f"{row['delta_latency']:+.2f} | "
                f"{row['classification']} |"
            )

        lines.append('')

        low = sorted(
            rows,
            key=lambda row: (
                row['delta_reasoning'],
                row['repetition']
            )
        )[:3]

        high = sorted(
            rows,
            key=lambda row: (
                -row['delta_reasoning'],
                row['repetition']
            )
        )[:3]

        lines.append(
            '**Three strongest reasoning reductions in this arm:**'
        )
        lines.append('')

        lines.append(
            '| Rank | Rep | Δreasoning | Δtotal | Regime |'
        )
        lines.append(
            '|---:|---:|---:|---:|---|'
        )

        for idx, row in enumerate(low, 1):
            lines.append(
                f"| {idx} | {row['repetition']} | "
                f"{row['delta_reasoning']:+d} | "
                f"{row['delta_total']:+d} | "
                f"{row['classification']} |"
            )

        lines.append('')

        lines.append(
            '**Three strongest reasoning increases in this arm:**'
        )
        lines.append('')

        lines.append(
            '| Rank | Rep | Δreasoning | Δtotal | Regime |'
        )
        lines.append(
            '|---:|---:|---:|---:|---|'
        )

        for idx, row in enumerate(high, 1):
            lines.append(
                f"| {idx} | {row['repetition']} | "
                f"{row['delta_reasoning']:+d} | "
                f"{row['delta_total']:+d} | "
                f"{row['classification']} |"
            )

        lines.append('')

    lines.append('## 3. Rep-by-rep co-occurrence matrix')
    lines.append('')

    lines.append(
        'The same repetition number is aligned across all three '
        'representation arms.'
    )
    lines.append('')

    lines.append(
        '| Rep | SR ΔR/ΔT | SR-A ΔR/ΔT | SR-B ΔR/ΔT | '
        'SR regime | A regime | B regime |'
    )
    lines.append(
        '|---:|---:|---:|---:|---|---|---|'
    )

    for repetition in EXPECTED_REPETITIONS:

        values: List[str] = []
        regimes: List[str] = []

        for treatment_arm in REPRESENTATION_ARMS:
            row = next(
                (
                    item
                    for item in pair_cache[treatment_arm]
                    if item['repetition'] == repetition
                ),
                None
            )

            if row is None:
                values.append('MISSING')
                regimes.append('MISSING')
            else:
                values.append(
                    f"{row['delta_reasoning']:+d}/"
                    f"{row['delta_total']:+d}"
                )
                regimes.append(
                    row['classification']
                )

        lines.append(
            f'| {repetition} | '
            f'{values[0]} | {values[1]} | {values[2]} | '
            f'{regimes[0]} | {regimes[1]} | {regimes[2]} |'
        )

    lines.append('')

    lines.append('### Sign recurrence across representations')
    lines.append('')

    def sign(value: int) -> str:
        if value < 0:
            return '-'

        if value > 0:
            return '+'

        return '0'

    sign_pairs: Dict[Tuple[str, str], int] = {}

    for first_index, first_arm in enumerate(REPRESENTATION_ARMS):
        for second_arm in REPRESENTATION_ARMS[first_index + 1:]:

            matches = 0

            for repetition in EXPECTED_REPETITIONS:

                first = next(
                    row
                    for row in pair_cache[first_arm]
                    if row['repetition'] == repetition
                )

                second = next(
                    row
                    for row in pair_cache[second_arm]
                    if row['repetition'] == repetition
                )

                if (
                    sign(first['delta_reasoning'])
                    == sign(second['delta_reasoning'])
                ):
                    matches += 1

            sign_pairs[(first_arm, second_arm)] = matches

    lines.append(
        '| Pair | Same Δreasoning sign | / 10 |'
    )
    lines.append(
        '|---|---:|---:|'
    )

    for (first_arm, second_arm), matches in sign_pairs.items():
        lines.append(
            f'| `{first_arm}` vs `{second_arm}` | '
            f'{matches} | {matches}/10 |'
        )

    lines.append('')

    lines.append('## 4. Cross-arm regime counts')
    lines.append('')

    lines.append(
        '| Arm | Beneficial | Neutral reasoning | '
        'Harmful | Other / mixed |'
    )
    lines.append(
        '|---|---:|---:|---:|---:|'
    )

    for treatment_arm in REPRESENTATION_ARMS:

        counts = {
            'BENEFICIAL': 0,
            'NEUTRAL_REASONING': 0,
            'HARMFUL': 0,
        }

        for row in pair_cache[treatment_arm]:
            regime = row['classification']

            if regime in counts:
                counts[regime] += 1

        other = (
            len(pair_cache[treatment_arm])
            - sum(counts.values())
        )

        lines.append(
            f"| `{treatment_arm}` | "
            f"{counts['BENEFICIAL']} | "
            f"{counts['NEUTRAL_REASONING']} | "
            f"{counts['HARMFUL']} | "
            f"{other} |"
        )

    lines.append('')

    pooled: List[Dict[str, Any]] = []

    for treatment_arm, rows in pair_cache.items():
        for row in rows:
            pooled.append({
                **row,
                'treatment_arm': treatment_arm
            })

    pooled_low = sorted(
        pooled,
        key=lambda row: (
            row['delta_reasoning'],
            row['treatment_arm'],
            row['repetition']
        )
    )[:3]

    pooled_high = sorted(
        pooled,
        key=lambda row: (
            -row['delta_reasoning'],
            row['treatment_arm'],
            row['repetition']
        )
    )[:3]

    def raw_block(label: str, row: Dict[str, Any]) -> None:

        treatment = row['treatment']

        lines.append(
            f"### {label} — "
            f"`{row['treatment_arm']}` rep {row['repetition']}"
        )
        lines.append('')

        lines.append(
            f"- run_id: `{treatment.get('run_id', 'n/a')}`"
        )

        lines.append(
            f"- response_id: "
            f"`{treatment.get('execution', {}).get('response_id', 'n/a')}`"
        )

        lines.append(
            f"- input_tokens: **{metric(treatment, 'input_tokens')}**"
        )

        lines.append(
            f"- reasoning_tokens: "
            f"**{metric(treatment, 'reasoning_tokens')}**"
        )

        lines.append(
            f"- total_tokens: **{metric(treatment, 'total_tokens')}**"
        )

        lines.append(
            f"- delta_reasoning vs selection_only: "
            f"**{row['delta_reasoning']:+d}**"
        )

        lines.append(
            f"- delta_total vs selection_only: "
            f"**{row['delta_total']:+d}**"
        )

        lines.append(
            f"- regime: **{row['classification']}**"
        )

        lines.append('')

        lines.append('**raw_text**')
        lines.append('')
        lines.append('```text')
        lines.append(raw_text(treatment))
        lines.append('```')
        lines.append('')

        lines.append('**parsed output**')
        lines.append('')
        lines.append('```json')
        lines.append(output_text(treatment))
        lines.append('```')
        lines.append('')

    lines.append(
        '## 5. Extreme Low — three largest reasoning reductions'
    )
    lines.append('')

    for idx, row in enumerate(pooled_low, 1):
        raw_block(
            f'Extreme Low #{idx}',
            row
        )

    lines.append(
        '## 6. Extreme High — three largest reasoning increases'
    )
    lines.append('')

    for idx, row in enumerate(pooled_high, 1):
        raw_block(
            f'Extreme High #{idx}',
            row
        )

    lines.append('## 7. Data-only closure')
    lines.append('')

    lines.append(
        '- This report does not attribute causality to internal model cognition.'
    )

    lines.append(
        '- The only comparison baseline for representation arms is '
        '`selection_only`, aligned by repetition.'
    )

    lines.append(
        '- Any apparent recurrence across the same repetition number is '
        'reported only as an observed sign pattern.'
    )

    lines.append(
        '- Raw model text is reproduced exactly as stored in the v0.6 '
        'report where present.'
    )

    lines.append('')

    return '\n'.join(lines) + '\n'


def main() -> int:

    parser = argparse.ArgumentParser(
        description='Static forensic analyzer for PromptForge v0.6 T003'
    )

    parser.add_argument(
        '--report',
        type=Path,
        default=DEFAULT_REPORT
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    if not args.report.is_file():
        raise FileNotFoundError(
            f'Report not found: {args.report}'
        )

    report = load_json(args.report)
    rows = flatten_runs(report)
    index = index_t003(rows)

    required = [
        (arm_id, repetition)
        for arm_id in [BASELINE_ARM] + REPRESENTATION_ARMS
        for repetition in EXPECTED_REPETITIONS
    ]

    missing = [
        key
        for key in required
        if key not in index
    ]

    if missing:
        raise ValueError(
            f'Missing required T003 runs: {missing}'
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    args.output.write_text(
        build_report(report, index),
        encoding='utf-8'
    )

    print('============================================================')
    print('PROMPTFORGE T003 FORENSIC ANALYZER')
    print('============================================================')
    print('status=COMPLETE')
    print(f'task={TASK_ID}')
    print(f'baseline={BASELINE_ARM}')
    print(f'paired_arms={len(REPRESENTATION_ARMS)}')
    print(
        f'paired_rows_per_arm={len(EXPECTED_REPETITIONS)}'
    )
    print(f'output={args.output}')
    print('api_calls=0')
    print('runner_modified=NO')
    print('baseline_modified=NO')
    print('============================================================')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
