"""Tests multi-turn conversation: memory, conversation cache and speed.

Run:   python test_conversation.py --quick  (about 1-2 minutes on a slow CPU)
       python test_conversation.py          (also runs the slow no-cache comparison)
Save the output as evidence:  python test_conversation.py | Tee-Object results/conversation_test.txt
"""
import argparse
import sys

from engine import CoachEngine, ModelLoadError

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=40)
parser.add_argument("--force", action="store_true")
parser.add_argument("--quick", action="store_true", help="skip the slow no-cache comparison")
parser.add_argument("--threads", type=int, default=0)
args = parser.parse_args()

try:
    engine = CoachEngine(force=args.force, threads=args.threads)
except ModelLoadError as e:
    sys.exit(f"\nCould not start: {e}")
print("Engine ready:", engine.info)


def ask(label, messages):
    print(f"\n=== {label} ===", flush=True)
    text = ""
    for piece in engine.stream(messages, max_new_tokens=args.tokens):
        text += piece
        print(piece, end="", flush=True)
    print("\n", engine.last_stats, flush=True)
    return text.strip()


msgs = [{"role": "user", "content": "What is overfitting?"}]
a1 = ask("TURN 1", msgs)

msgs += [{"role": "assistant", "content": a1},
         {"role": "user", "content": "Can you give me a different example?"}]
engine.prefix_cache_enabled = True
a2 = ask("TURN 2 (conversation cache ON)", msgs)

if not args.quick:
    engine.prefix_cache_enabled = False
    a2_nocache = ask("TURN 2 (cache OFF, for comparison)", msgs)
    print("\nTurn 2 identical with and without cache:", a2 == a2_nocache)

engine.prefix_cache_enabled = True
ask("REGENERATE turn 2 (same messages again)", msgs)
