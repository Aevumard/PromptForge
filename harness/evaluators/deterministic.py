def evaluate(expected, actual):
    errors = []

    for key, value in expected.items():
        if actual.get(key) != value:
            errors.append(key)

    passed = len(errors) == 0

    return passed, {
        'evaluator': 'deterministic_exact_fields',
        'passed': passed,
        'missing_or_incorrect': errors
    }