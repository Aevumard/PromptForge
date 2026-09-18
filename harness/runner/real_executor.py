import json

from harness.providers.deepseek import DeepSeekProvider


PROVIDER = DeepSeekProvider()


def build_prompt(task, compiled):
    return (
        task['instruction']
        + '\n\nCONTEXT:\n'
        + json.dumps(
            compiled['value'],
            ensure_ascii=False,
            indent=2
        )
        + '\n\nRETURN ONLY THIS JSON OBJECT:\n'
        + json.dumps(
            task['schema'],
            ensure_ascii=False,
            indent=2
        )
    )


def parse_json(text):
    if not text:
        return None, 'empty_model_output'

    text = text.strip()

    try:
        return json.loads(text), None
    except Exception:
        start = text.find('{')
        end = text.rfind('}')

        if start >= 0 and end > start:
            try:
                return json.loads(
                    text[start:end + 1]
                ), None
            except Exception as exc:
                return None, 'json_parse_error: {}'.format(exc)

        return None, 'json_parse_error: no_object'


def execute(task, compiled):
    prompt = build_prompt(task, compiled)
    result = PROVIDER.generate(prompt)

    base = {
        'provider_status': result['status'],
        'error': result.get('error'),
        'response_id': result.get('response_id'),
        'model': result.get('model'),
        'response_status': result.get('response_status'),
        'input_tokens': result.get('input_tokens', 0),
        'output_tokens': result.get('output_tokens', 0),
        'reasoning_tokens': result.get('reasoning_tokens', 0),
        'total_tokens': result.get('total_tokens', 0),
        'latency_ms': result.get('latency_ms', 0),
        'raw_text': result.get('text', ''),
        'output': None
    }

    if result['status'] != 'MODEL_OK':
        return base

    parsed, error = parse_json(result.get('text', ''))

    base['provider_status'] = (
        'MODEL_OK'
        if error is None
        else 'OUTPUT_ERROR'
    )

    base['error'] = error
    base['output'] = parsed

    return base