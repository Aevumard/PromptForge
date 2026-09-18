from statistics import mean, median


def stats(values):
    if not values:
        return {
            'n': 0,
            'mean': None,
            'median': None,
            'min': None,
            'max': None
        }

    return {
        'n': len(values),
        'mean': mean(values),
        'median': median(values),
        'min': min(values),
        'max': max(values)
    }


def summarize(rows):
    usable = [
        r for r in rows
        if r['execution']['provider_status'] == 'MODEL_OK'
    ]

    return {
        'n': len(rows),
        'model_ok_rate': sum(
            r['execution']['provider_status'] == 'MODEL_OK'
            for r in rows
        ) / len(rows) if rows else 0,
        'verification_pass_rate': sum(
            r['verification']['passed']
            for r in rows
        ) / len(rows) if rows else 0,
        'input_tokens': stats([
            r['metrics']['input_tokens']
            for r in usable
        ]),
        'output_tokens': stats([
            r['metrics']['output_tokens']
            for r in usable
        ]),
        'reasoning_tokens': stats([
            r['metrics']['reasoning_tokens']
            for r in usable
        ]),
        'total_tokens': stats([
            r['metrics']['total_tokens']
            for r in usable
        ]),
        'latency_ms': stats([
            r['metrics']['latency_ms']
            for r in usable
        ]),
        'successful_runs': len([
            r for r in rows
            if r['verification']['passed']
        ])
    }


def _compensation_ratio(delta_input, delta_output):
    if delta_input >= 0:
        return None
    input_removed = -delta_input
    downstream_compensation = max(0, delta_output)
    return downstream_compensation / input_removed


def paired(control, treatment):
    rows = []

    by_rep_control = {
        r['repetition']: r
        for r in control
    }

    by_rep_treatment = {
        r['repetition']: r
        for r in treatment
    }

    common = sorted(
        set(by_rep_control) & set(by_rep_treatment)
    )

    for rep in common:
        a = by_rep_control[rep]
        b = by_rep_treatment[rep]

        delta_input = (
            b['metrics']['input_tokens']
            - a['metrics']['input_tokens']
        )
        delta_output = (
            b['metrics']['output_tokens']
            - a['metrics']['output_tokens']
        )

        rows.append({
            'repetition': rep,
            'control_status': a['status'],
            'treatment_status': b['status'],
            'input_tokens_delta': delta_input,
            'output_tokens_delta': delta_output,
            'reasoning_tokens_delta': (
                b['metrics']['reasoning_tokens']
                - a['metrics']['reasoning_tokens']
            ),
            'total_tokens_delta': (
                b['metrics']['total_tokens']
                - a['metrics']['total_tokens']
            ),
            'latency_delta_ms': (
                b['metrics']['latency_ms']
                - a['metrics']['latency_ms']
            ),
            'quality_pass': bool(b['verification']['passed']),
            'verification_preserved': (
                a['verification']['passed']
                and b['verification']['passed']
            ),
            'downstream_compensation': (
                max(0, delta_output)
                if delta_input < 0 else None
            ),
            'input_removed': (
                -delta_input
                if delta_input < 0 else None
            ),
            'compensation_ratio': _compensation_ratio(
                delta_input,
                delta_output
            )
        })

    ratios = [
        r['compensation_ratio']
        for r in rows
        if r['compensation_ratio'] is not None
    ]

    return {
        'n': len(rows),
        'input_tokens_delta': stats([
            r['input_tokens_delta'] for r in rows
        ]),
        'output_tokens_delta': stats([
            r['output_tokens_delta'] for r in rows
        ]),
        'reasoning_tokens_delta': stats([
            r['reasoning_tokens_delta'] for r in rows
        ]),
        'total_tokens_delta': stats([
            r['total_tokens_delta'] for r in rows
        ]),
        'latency_delta_ms': stats([
            r['latency_delta_ms'] for r in rows
        ]),
        'downstream_compensation': stats([
            r['downstream_compensation']
            for r in rows
            if r['downstream_compensation'] is not None
        ]),
        'compensation_ratio': stats(ratios),
        'verification_preserved_rate': (
            sum(r['verification_preserved'] for r in rows) / len(rows)
            if rows else 0
        ),
        'quality_pass_rate': (
            sum(r['quality_pass'] for r in rows) / len(rows)
            if rows else 0
        ),
        'rows': rows
    }
