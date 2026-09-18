import json
import time
from copy import deepcopy


def serialized(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2
    )


def execute(task, compiled):
    start = time.perf_counter()

    source = compiled['value']
    expected = task['expected']

    output = {}

    for key in expected:
        if key in source:
            output[key] = deepcopy(source[key])

    source_serialized = serialized(source)
    output_serialized = serialized(output)

    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        'output': output,
        'input_chars': len(source_serialized),
        'output_chars': len(output_serialized),
        'tool_calls': 0,
        'retrieval_calls': 0,
        'latency_ms': latency_ms
    }