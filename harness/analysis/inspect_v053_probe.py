import json
from pathlib import Path

PROBE = Path(r"D:\PROMPTFORGE\promptforge-harness-v0.1\harness\analysis\v053_report_probe.json")

def shorten(value, limit=500):
    text = repr(value)
    if len(text) > limit:
        return text[:limit] + "...<TRUNCATED>"
    return text

def main():
    data = json.loads(PROBE.read_text(encoding="utf-8"))

    candidates = data.get("sample_candidates", [])

    print("PROBE_FILE={}".format(PROBE))
    print("CANDIDATES_IN_PROBE={}".format(len(candidates)))
    print("")

    for index, candidate in enumerate(candidates[:10]):
        print("=" * 70)
        print("CANDIDATE {}".format(index + 1))
        print("=" * 70)

        print("TASK        :", repr(candidate.get("task")))
        print("ARM         :", repr(candidate.get("arm")))
        print("REP         :", repr(candidate.get("rep")))
        print("PATH        :")
        for item in candidate.get("path", []):
            print("   ", repr(item))

        print("KEYS        :")
        for key in candidate.get("keys", []):
            print("   ", repr(key))

        print("METRICS     :")
        metrics = candidate.get("metrics", {})
        for key, value in metrics.items():
            print("   {} = {}".format(key, repr(value)))

        print("RAW CANDIDATE:")
        print(shorten(candidate, 2500))
        print("")

    print("=" * 70)
    print("IDENTIFICATION SUMMARY")
    print("=" * 70)

    task_values = sorted(set(str(c.get("task")) for c in candidates))
    arm_values = sorted(set(str(c.get("arm")) for c in candidates))
    rep_values = sorted(set(str(c.get("rep")) for c in candidates))

    print("TASK VALUES:")
    for value in task_values:
        print("  ", value)

    print("ARM VALUES:")
    for value in arm_values:
        print("  ", value)

    print("REP VALUES:")
    for value in rep_values:
        print("  ", value)

    print("")
    print("RAW PROBE TOP-LEVEL KEYS:")
    if isinstance(data, dict):
        for key in data.keys():
            value = data[key]
            print("  {} -> {}".format(key, type(value).__name__))

if __name__ == "__main__":
    main()