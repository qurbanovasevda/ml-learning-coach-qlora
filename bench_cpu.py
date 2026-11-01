"""Quick CPU benchmark (about 30 s, does NOT load the model).
Measures one Qwen-sized matrix multiplication in fp32, bf16 and int8, and estimates tokens/s.

Run:  python bench_cpu.py            (add --threads 2 to test with fewer threads)
"""
import argparse
import os
import platform
import sys
import time

import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=0)
args = parser.parse_args()
threads = args.threads or max(1, (os.cpu_count() or 4) // 2)
torch.set_num_threads(threads)

print("CPU:", platform.processor() or platform.machine())
print("Logical cores:", os.cpu_count(), "| threads used:", threads)
try:
    print("PyTorch CPU capability:", torch.backends.cpu.get_cpu_capability())
except Exception:
    pass
try:
    print("Quantized engines:", torch.backends.quantized.supported_engines)
except Exception:
    pass

# One MLP matrix of Qwen2.5-1.5B: 1536 -> 8960. The whole model is about 112 such matrices
# per generated token (all layers + the output head).
IN, OUT, FACTOR = 1536, 8960, 112


def bench(fn, n):
    for _ in range(3):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000  # ms per call


results = {}
lin32 = nn.Linear(IN, OUT).eval()
for label, rows, n in [("decode (1 token)", 1, 30), ("prefill (300 tokens)", 300, 4)]:
    x32 = torch.randn(rows, IN)
    with torch.inference_mode():
        results[("fp32", label)] = bench(lambda: lin32(x32), n)

        lin16 = nn.Linear(IN, OUT).eval().to(torch.bfloat16)
        x16 = x32.to(torch.bfloat16)
        results[("bf16", label)] = bench(lambda: lin16(x16), n)

        try:
            q = torch.ao.quantization.quantize_dynamic(
                nn.Sequential(nn.Linear(IN, OUT).eval()), {nn.Linear}, dtype=torch.qint8)
            results[("int8", label)] = bench(lambda: q(x32), n)
        except Exception as e:
            print("int8 dynamic quantization failed:", type(e).__name__, str(e)[:120])

print()
print(f"{'precision':<10}{'1 token (ms)':>14}{'est. tokens/s':>15}{'300-token prefill (ms)':>25}{'est. prefill (s)':>18}")
for p in ["fp32", "bf16", "int8"]:
    d = results.get((p, "decode (1 token)"))
    f = results.get((p, "prefill (300 tokens)"))
    if d is None:
        continue
    print(f"{p:<10}{d:>14.2f}{1000 / (d * FACTOR):>15.2f}{f:>25.1f}{f * FACTOR / 1000:>18.1f}")

print("\nHow to read this: the best row is the fastest precision on your CPU.")
print("Memory per precision for the whole model: fp32 ~6.2 GB, bf16 ~3.1 GB, int8 ~1.8 GB.")
