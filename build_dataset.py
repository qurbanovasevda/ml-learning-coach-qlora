"""Merges examples_part*.py into data/train.jsonl.
If any format problem is found, the file is NOT written and the problems are printed.
Run:  python build_dataset.py
"""
import glob
import importlib
import json
import os
import random
import sys
from collections import Counter

from system_prompt import SYSTEM
from validate import normalize, validate

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

examples = []
for path in sorted(glob.glob("examples_part*.py")):
    module = importlib.import_module(path[:-3])
    examples += module.EXAMPLES
    print(f"{path}: {len(module.EXAMPLES)} examples")

problems = validate(examples)
if problems:
    print("\nFORMAT PROBLEMS (train.jsonl was NOT written):")
    for p in problems:
        print("  -", p)
    sys.exit(1)

random.shuffle(examples)
os.makedirs("data", exist_ok=True)
kinds = Counter()
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        kind, turns = normalize(ex)
        kinds[kind] += 1
        messages = [{"role": "system", "content": SYSTEM}]
        for user, assistant in turns:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

print(f"Total: {len(examples)} examples -> data/train.jsonl (format check passed)")
print(f"  single-turn: {kinds['single']} | multi-turn: {kinds['follow']} | off-topic: {kinds['off']}")
