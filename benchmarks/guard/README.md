# V27.3 — PromptForge action guard

V27.3 wraps a candidate agent prediction with the existing PromptForge
epistemic/action boundary.

The guard is intentionally conservative:

1. it leaves classification fields untouched;
2. it exposes the case's available evidence to the action gate;
3. contradictory evidence makes the proposed operational action inadmissible;
4. blocked actions are converted to `human_review`;
5. the guard preserves an explicit audit record.

This is **not** an LLM adapter. It is a safety-layer experiment showing how a
prediction that looks acceptable at the classifier level can be stopped before
an operational side effect.

The next model-in-the-loop benchmark can feed real agent predictions into
`guard_predictions()` and compare:

```
agent
  -> raw prediction
  -> PromptForge guard
  -> final action
```
