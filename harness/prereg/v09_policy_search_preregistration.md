# PromptForge V0.9 — Policy Search Discovery

STATUS: FROZEN

OBJECTIVE:
Search selectively for task-specific structural representations that reduce effective execution cost relative to noop_compact.

TASKS:
T001
T002
T003
T004

MODEL:
DeepSeek

CONDITIONS:
noop_compact
pairs_compact
pairs_bare_compact
flat_compact

REPETITIONS:
15 per cell

TOTAL RUNS:
240

PRIMARY METRIC:
delta_total = treatment total_tokens - noop_compact total_tokens

SECONDARY:
delta_reasoning
delta_input
latency
provider status
deterministic quality

BENEFIT_SIGNAL:
mean delta_total < 0

HARM_SIGNAL:
mean delta_total > 0

UNCERTAIN:
zero or non-estimable

PROTECTED:
transforms.py
real_executor.py
deepseek_adapter.py
v084_zoom_runner.py

NO AUTOMATIC RETRIES.
ATOMIC REPORT PERSISTENCE.
RESUME ENABLED.