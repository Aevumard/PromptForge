from copy import deepcopy


def noop(ctx):
    src = ctx['data']
    return {
        'value': deepcopy(src),
        'included': list(src.keys()),
        'excluded': []
    }


def selection_only(ctx):
    src = ctx['data']
    keep = ctx['required']
    value = {k: deepcopy(src[k]) for k in keep if k in src}
    excluded = [k for k in src if k not in value]
    return {
        'value': value,
        'included': list(value.keys()),
        'excluded': excluded
    }


def representation_only(ctx):
    src = ctx['data']
    value = {k: src[k] for k in src}
    return {
        'value': value,
        'included': list(src.keys()),
        'excluded': []
    }


def selection_representation(ctx):
    return selection_only(ctx)


# v0.5.x compatibility surface is intentionally frozen.
TRANSFORMS = {
    'noop': noop,
    'selection_only': selection_only,
    'representation_only': representation_only,
    'selection_representation': selection_representation
}


def _state_from_context(ctx):
    value = deepcopy(ctx['data'])
    return {
        'value': value,
        'included': list(value.keys()),
        'excluded': [],
        'transform_sequence': []
    }


def _selection_stage(state, ctx):
    source = state['value']
    required = ctx['required']
    selected = {k: deepcopy(source[k]) for k in required if k in source}
    excluded = [k for k in source if k not in selected]
    state['value'] = selected
    state['included'] = list(selected.keys())
    state['excluded'] = excluded
    return state


def _representation_a(state, _ctx):
    # Lossless pair representation: [{"key": ..., "value": ...}, ...].
    # This is intentionally different from the object representation.
    source = state['value']
    state['value'] = {
        'fields': [
            {'key': key, 'value': deepcopy(value)}
            for key, value in source.items()
        ]
    }
    return state


def _representation_b(state, _ctx):
    # Lossless tuple representation: [["key", value], ...].
    # This provides a second, materially different structural encoding.
    source = state['value']
    state['value'] = {
        'pairs': [
            [key, deepcopy(value)]
            for key, value in source.items()
        ]
    }
    return state


STAGES = {
    'noop': lambda state, ctx: state,
    'selection': _selection_stage,
    'representation_A': _representation_a,
    'representation_B': _representation_b,
    'representation': _representation_a,
}


V06_ARM_SEQUENCES = {
    'noop': ['noop'],
    'selection_only': ['selection'],
    'representation_only': ['representation'],
    'representation_A': ['representation_A'],
    'representation_B': ['representation_B'],
    'selection_representation': ['selection', 'representation'],
    'selection_representation_A': ['selection', 'representation_A'],
    'selection_representation_B': ['selection', 'representation_B'],
}


def apply_sequence(ctx, sequence):
    if not sequence:
        raise ValueError('transform sequence must not be empty')

    state = _state_from_context(ctx)

    for stage_id in sequence:
        try:
            stage = STAGES[stage_id]
        except KeyError as exc:
            raise ValueError('unknown transform stage: {}'.format(stage_id)) from exc
        state = stage(state, ctx)
        state['transform_sequence'].append(stage_id)

    return state


def compile_v06_arm(ctx, arm_id):
    try:
        sequence = V06_ARM_SEQUENCES[arm_id]
    except KeyError as exc:
        raise ValueError('unknown v0.6 arm: {}'.format(arm_id)) from exc

    compiled = apply_sequence(ctx, sequence)
    compiled['arm_id'] = arm_id
    compiled['transform_id'] = '|'.join(sequence)
    return compiled


def decode_representation_a(value):
    if not isinstance(value, dict) or not isinstance(value.get('fields'), list):
        raise ValueError('invalid representation_A payload')
    return {
        item['key']: item['value']
        for item in value['fields']
    }


def decode_representation_b(value):
    if not isinstance(value, dict) or not isinstance(value.get('pairs'), list):
        raise ValueError('invalid representation_B payload')
    return {
        pair[0]: pair[1]
        for pair in value['pairs']
    }
