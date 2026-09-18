
import json
from pathlib import Path

ROOT = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1")
schedule = json.loads((
    ROOT / "harness" / "runner" / "v083_schedule.json"
).read_text(encoding="utf-8"))

assert len(schedule["runs"]) == 120
assert [int(x["global_execution_position"]) for x in schedule["runs"]] == list(range(120))

expected = {
    "noop_pretty",
    "noop_compact",
    "representation_A_pretty",
    "representation_A_compact",
    "representation_B_pretty",
    "representation_B_compact",
}

assert set(x["condition_id"] for x in schedule["runs"]) == expected

print("V083_TESTS=PASS")
print("V083_TEST_COUNT=3")
print("V083_API_CALLS=0")
