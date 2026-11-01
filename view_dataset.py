"""Shows data/train.jsonl in a readable way and checks the format.

Run:
    python view_dataset.py        -> all examples + check
    python view_dataset.py 3      -> only example 3
    python view_dataset.py list   -> short list
"""
import json
import sys

from validate import CHECK_PREFIX, infer_kind, validate

sys.stdout.reconfigure(encoding="utf-8")

with open("data/train.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

convs = []
for r in rows:
    msgs = r["messages"][1:]  # skip the system message
    convs.append([(msgs[i]["content"], msgs[i + 1]["content"]) for i in range(0, len(msgs), 2)])

arg = sys.argv[1] if len(sys.argv) > 1 else None
for i, turns in enumerate(convs, start=1):
    kind = infer_kind(turns)
    first_user, first_answer = turns[0]
    is_check = first_user.startswith(CHECK_PREFIX)
    name = {"follow": "FOLLOW-UP", "off": "OFF-TOPIC"}.get(kind, "CHECK" if is_check else "EXPLAIN")
    if arg == "list":
        if kind == "single" and is_check:
            topic = first_user.split("\n")[1].replace("Topic:", "").strip()
            verdict = first_answer.split("\n")[0].replace("Verdict:", "").strip()
            print(f"#{i:<3} {name:<10} {topic:<24} -> {verdict}")
        else:
            extra = f"  (+{len(turns) - 1} follow-up)" if len(turns) > 1 else ""
            print(f"#{i:<3} {name:<10} {first_user.splitlines()[0]}{extra}")
    elif arg is None or arg == str(i):
        print("=" * 70)
        print(f"#{i}  [{name}]")
        for t, (user, answer) in enumerate(turns, start=1):
            print("-" * 70)
            print(f"STUDENT (turn {t}):\n{user}\n")
            print(f"COACH:\n{answer}\n")

problems = validate([{"kind": infer_kind(t), "turns": t} for t in convs])
print("=" * 70)
print(f"Total examples: {len(convs)}")
if problems:
    print("PROBLEMS:")
    for p in problems:
        print("  -", p)
else:
    print("Format check: no problems found")
