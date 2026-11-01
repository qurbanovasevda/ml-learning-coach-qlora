"""Tests the inference engine from the terminal: streaming, prefix cache and speed.

Run:   python test_engine.py            (about 1-2 minutes on a slow CPU)
       python test_engine.py --force    (skip the free-RAM check at your own risk)
"""
import argparse
import sys
import time

from engine import CoachEngine, ModelLoadError

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=40)
parser.add_argument("--force", action="store_true")
parser.add_argument("--threads", type=int, default=0)
args = parser.parse_args()

try:
    engine = CoachEngine(force=args.force, threads=args.threads)
except ModelLoadError as e:
    sys.exit(f"\nCould not start: {e}")
print("Engine ready:", engine.info, flush=True)

PROMPT = [{"role": "user", "content": "What is overfitting?"}]


def run(label, **opts):
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    text = ""
    for piece in engine.stream(PROMPT, max_new_tokens=args.tokens, **opts):
        if not text:
            print(f"[first piece after {time.time() - t0:.1f} s]", flush=True)
        text += piece
        print(piece, end="", flush=True)
    print("\n", engine.last_stats, flush=True)
    return text


engine.prefix_cache_enabled = True
a = run("WITH prefix cache")
engine.prefix_cache_enabled = False
b = run("WITHOUT prefix cache")
print("\nSame text with and without cache:", a == b)
engine.prefix_cache_enabled = True
run("adapter DISABLED (plain base model)", adapter_enabled=False)
