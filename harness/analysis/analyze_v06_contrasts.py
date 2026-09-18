import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'harness' / 'reports' / 'v06_confirmatory_deepseek.json'
OUTPUT = ROOT / 'harness' / 'analysis' / 'v06_contrasts_analysis.md'

PAIRS = [
    ('selection_only', 'selection_representation'),
    ('selection_only', 'selection_representation_A'),
    ('selection_only', 'selection_representation_B'),
    ('noop', 'representation_A'),
    ('noop', 'representation_B'),
]

METRICS = [
    'input_tokens',
    'reasoning_tokens',
    'output_tokens',
    'total_tokens',
    'latency_ms',
]


def load_report():
    if not REPORT.exists():
        raise FileNotFoundError('Missing report: {}'.format(REPORT))

    with REPORT.open('r', encoding='utf-8') as fh:
        data = json.load(fh)

    if data.get('schema_version') != '0.6':
        raise ValueError(
            'Unexpected schema_version: {}'.format(
                data.get('schema_version')
            )
        )

    return data


def group_runs(data):
    grouped = defaultdict(dict)

    for run in data.get('runs', []):
        key = (run['task_id'], int(run['repetition']))
        grouped[key][run['arm_id']] = run

    return grouped


def calc_stats(values):
    if not values:
        return {
            'n': 0,
            'mean': None,
            'median': None,
            'min': None,
            'max': None,
        }

    return {
        'n': len(values),
        'mean': mean(values),
        'median': median(values),
        'min': min(values),
        'max': max(values),
    }


def paired(grouped, control_arm, treatment_arm):
    rows = []

    for (task_id, repetition), arms in sorted(grouped.items()):
        if control_arm not in arms or treatment_arm not in arms:
            continue

        control = arms[control_arm]
        treatment = arms[treatment_arm]

        row = {
            'task_id': task_id,
            'repetition': repetition,
            'control_arm': control_arm,
            'treatment_arm': treatment_arm,
        }

        for metric in METRICS:
            row[metric + '_delta'] = (
                treatment['metrics'][metric]
                - control['metrics'][metric]
            )

        row['control_quality'] = bool(
            control['verification']['passed']
        )
        row['treatment_quality'] = bool(
            treatment['verification']['passed']
        )
        row['quality_preserved'] = (
            row['control_quality']
            and row['treatment_quality']
        )

        rows.append(row)

    return rows


def summarize(rows):
    result = {'n': len(rows)}

    for metric in METRICS:
        key = metric + '_delta'
        result[key] = calc_stats([r[key] for r in rows])

    result['quality_preserved_rate'] = (
        sum(r['quality_preserved'] for r in rows) / len(rows)
        if rows else 0.0
    )

    result['control_quality_rate'] = (
        sum(r['control_quality'] for r in rows) / len(rows)
        if rows else 0.0
    )

    result['treatment_quality_rate'] = (
        sum(r['treatment_quality'] for r in rows) / len(rows)
        if rows else 0.0
    )

    return result


def representation_tradeoff(rows):
    values = []

    for row in rows:
        input_delta = row['input_tokens_delta']
        reasoning_delta = row['reasoning_tokens_delta']
        output_delta = row['output_tokens_delta']
        total_delta = row['total_tokens_delta']

        input_overhead = max(0, input_delta)
        reasoning_reduction = max(0, -reasoning_delta)
        output_reduction = max(0, -output_delta)

        reasoning_efficiency = None
        output_efficiency = None

        if input_overhead > 0:
            reasoning_efficiency = (
                reasoning_reduction / input_overhead
            )
            output_efficiency = (
                output_reduction / input_overhead
            )

        values.append({
            'input_overhead': input_overhead,
            'reasoning_reduction': reasoning_reduction,
            'output_reduction': output_reduction,
            'total_delta': total_delta,
            'reasoning_efficiency': reasoning_efficiency,
            'output_efficiency': output_efficiency,
            'quality_preserved': row['quality_preserved'],
        })

    return {
        'n': len(values),
        'input_overhead': calc_stats([
            v['input_overhead'] for v in values
        ]),
        'reasoning_reduction': calc_stats([
            v['reasoning_reduction'] for v in values
        ]),
        'output_reduction': calc_stats([
            v['output_reduction'] for v in values
        ]),
        'total_delta': calc_stats([
            v['total_delta'] for v in values
        ]),
        'reasoning_efficiency': calc_stats([
            v['reasoning_efficiency']
            for v in values
            if v['reasoning_efficiency'] is not None
        ]),
        'output_efficiency': calc_stats([
            v['output_efficiency']
            for v in values
            if v['output_efficiency'] is not None
        ]),
        'reasoning_reduction_rate': (
            sum(v['reasoning_reduction'] > 0 for v in values)
            / len(values)
            if values else 0.0
        ),
        'output_reduction_rate': (
            sum(v['output_reduction'] > 0 for v in values)
            / len(values)
            if values else 0.0
        ),
        'total_improvement_rate': (
            sum(v['total_delta'] < 0 for v in values)
            / len(values)
            if values else 0.0
        ),
        'quality_preserved_rate': (
            sum(v['quality_preserved'] for v in values)
            / len(values)
            if values else 0.0
        ),
    }


def original_compensation(rows):
    ratios = []

    for row in rows:
        delta_input = row['input_tokens_delta']
        delta_output = row['output_tokens_delta']

        if delta_input < 0:
            input_removed = -delta_input
            compensation = max(0, delta_output)
            ratios.append(compensation / input_removed)

    return calc_stats(ratios)


def fmt(value, digits=2):
    if value is None:
        return 'n/a'
    return ('{0:.' + str(digits) + 'f}').format(value)


def fmt_signed(value):
    if value is None:
        return 'n/a'
    return '{:+.2f}'.format(value)


def render_delta_table(lines, summary):
    lines.append(
        '| Metric | Mean | Median | Min | Max |'
    )
    lines.append(
        '|---|---:|---:|---:|---:|'
    )

    for metric in METRICS:
        s = summary[metric + '_delta']
        lines.append(
            '| {} | {} | {} | {} | {} |'.format(
                metric,
                fmt_signed(s['mean']),
                fmt_signed(s['median']),
                fmt_signed(s['min']),
                fmt_signed(s['max']),
            )
        )

    lines.append('')
    lines.append(
        '- paired n: **{}**'.format(summary['n'])
    )
    lines.append(
        '- quality preserved: **{}**'.format(
            fmt(summary['quality_preserved_rate'], 3)
        )
    )


def render_tradeoff(lines, tradeoff):
    lines.append(
        '| Metric | Mean | Median | Min | Max |'
    )
    lines.append(
        '|---|---:|---:|---:|---:|'
    )

    for key, label in [
        ('input_overhead', 'input overhead'),
        ('reasoning_reduction', 'reasoning reduction'),
        ('output_reduction', 'output reduction'),
        ('total_delta', 'total delta'),
        ('reasoning_efficiency', 'reasoning reduction / input overhead'),
        ('output_efficiency', 'output reduction / input overhead'),
    ]:
        s = tradeoff[key]
        lines.append(
            '| {} | {} | {} | {} | {} |'.format(
                label,
                fmt(s['mean']),
                fmt(s['median']),
                fmt(s['min']),
                fmt(s['max']),
            )
        )

    lines.append('')
    lines.append(
        '- reasoning reduction rate: **{}**'.format(
            fmt(tradeoff['reasoning_reduction_rate'], 3)
        )
    )
    lines.append(
        '- output reduction rate: **{}**'.format(
            fmt(tradeoff['output_reduction_rate'], 3)
        )
    )
    lines.append(
        '- total-token improvement rate: **{}**'.format(
            fmt(tradeoff['total_improvement_rate'], 3)
        )
    )
    lines.append(
        '- quality preserved: **{}**'.format(
            fmt(tradeoff['quality_preserved_rate'], 3)
        )
    )


def main():
    data = load_report()
    grouped = group_runs(data)

    task_ids = sorted({
        task_id for task_id, _ in grouped.keys()
    })

    repetitions = sorted({
        repetition for _, repetition in grouped.keys()
    })

    pair_rows = {}

    for control, treatment in PAIRS:
        pair_rows[(control, treatment)] = paired(
            grouped,
            control,
            treatment,
        )

    lines = [
        '# PromptForge v0.6 - Advanced Local Contrasts',
        '',
        '## Integrity',
        '',
        '- Source: `{}`'.format(REPORT),
        '- API calls: **0**',
        '- Runner modified: **NO**',
        '- Executor modified: **NO**',
        '- Total source runs: **{}**'.format(
            len(data.get('runs', []))
        ),
        '- Tasks: **{}**'.format(len(task_ids)),
        '- Repetitions: **{}**'.format(len(repetitions)),
        '',
        '## Global contrasts',
        '',
    ]

    for control, treatment in PAIRS:
        rows = pair_rows[(control, treatment)]
        summary = summarize(rows)

        lines.append(
            '### `{}` vs `{}`'.format(
                treatment,
                control,
            )
        )
        lines.append('')

        render_delta_table(lines, summary)
        lines.append('')

        if control == 'selection_only':
            tradeoff = representation_tradeoff(rows)

            lines.append(
                '**Representation-on-selection tradeoff**'
            )
            lines.append('')
            lines.append(
                'Here positive input delta is representation overhead; '
                'negative reasoning/output delta is reduction.'
            )
            lines.append('')

            render_tradeoff(lines, tradeoff)
            lines.append('')

        if control == 'noop':
            comp = original_compensation(rows)

            lines.append(
                '- original compensation-ratio mean: **{}**'.format(
                    fmt(comp['mean'])
                )
            )
            lines.append(
                '- compensation-ratio applicable n: **{}**'.format(
                    comp['n']
                )
            )
            lines.append('')

    lines.append('## Per-task contrasts')
    lines.append('')

    for task_id in task_ids:
        lines.append('## {}'.format(task_id))
        lines.append('')

        for control, treatment in PAIRS:
            rows = [
                row for row in pair_rows[(control, treatment)]
                if row['task_id'] == task_id
            ]

            summary = summarize(rows)

            lines.append(
                '### `{}` vs `{}`'.format(
                    treatment,
                    control,
                )
            )
            lines.append('')

            render_delta_table(lines, summary)
            lines.append('')

            if control == 'selection_only':
                tradeoff = representation_tradeoff(rows)

                lines.append(
                    '**Representation-on-selection tradeoff**'
                )
                lines.append('')
                render_tradeoff(lines, tradeoff)
                lines.append('')

    lines.append('## Audit: paired rows')
    lines.append('')

    for control, treatment in PAIRS:
        lines.append(
            '### `{}` vs `{}`'.format(
                treatment,
                control,
            )
        )
        lines.append('')

        lines.append(
            '| Task | Rep | Δ Input | Δ Reasoning | '
            'Δ Output | Δ Total | Δ Latency | Quality |'
        )
        lines.append(
            '|---|---:|---:|---:|---:|---:|---:|---|'
        )

        for row in pair_rows[(control, treatment)]:
            lines.append(
                '| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                    row['task_id'],
                    row['repetition'],
                    fmt_signed(row['input_tokens_delta']),
                    fmt_signed(row['reasoning_tokens_delta']),
                    fmt_signed(row['output_tokens_delta']),
                    fmt_signed(row['total_tokens_delta']),
                    fmt_signed(row['latency_ms_delta']),
                    'PASS' if row['quality_preserved'] else 'FAIL',
                )
            )

        lines.append('')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )

    print('')
    print('============================================================')
    print('PROMPTFORGE v0.6 - LOCAL CONTRAST ANALYSIS')
    print('============================================================')
    print('source : {}'.format(REPORT))
    print('output : {}'.format(OUTPUT))
    print('runs   : {}'.format(len(data.get('runs', []))))
    print('tasks  : {}'.format(len(task_ids)))
    print('reps   : {}'.format(len(repetitions)))
    print('API    : 0')
    print('status : COMPLETE')
    print('============================================================')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())