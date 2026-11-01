"""Local check: load the base model, attach OUR adapter, generate, and measure speed.

Run (inside the project folder, with the 'adapter' folder next to this file):
    python test_load.py                 # safe defaults
    python test_load.py --dtype bf16    # force the low-memory mode (about 3 GB)
    python test_load.py --compare       # also run once with the adapter disabled
The first run downloads the base model (~3 GB). After that it works fully offline.
"""
import argparse
import os
import subprocess
import sys
import time

import torch

try:
    import psutil
except ImportError:
    sys.exit("psutil is missing. Run:  pip install psutil")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from system_prompt import SYSTEM

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=["auto", "fp32", "bf16"], default="auto")
parser.add_argument("--tokens", type=int, default=40, help="max new tokens (keep it small for a first test)")
parser.add_argument("--compare", action="store_true", help="also generate with the adapter disabled")
parser.add_argument("--force", action="store_true", help="run even if free RAM is below the estimate (a watchdog protects the computer)")
parser.add_argument("--threads", type=int, default=0, help="CPU threads (default: half of the cores)")
args = parser.parse_args()

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "adapter"
PROMPT = "What is overfitting?"

if not os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
    sys.exit(f"Adapter not found in '{ADAPTER_DIR}'. Put the adapter folder (adapter_config.json, "
             "adapter_model.safetensors, ...) next to this script.")

# Leave some CPU free so the computer stays usable
threads = args.threads or max(1, (os.cpu_count() or 4) // 2)
torch.set_num_threads(threads)

free_gb = psutil.virtual_memory().available / 1024**3
total_gb = psutil.virtual_memory().total / 1024**3
print(f"torch {torch.__version__} | CPU threads used: {threads} | RAM: {free_gb:.1f} GB free of {total_gb:.1f} GB")

# Measured need: fp32 ~ 6.8 GB, bf16 ~ 3.6 GB (weights + Python/PyTorch runtime)
NEED = {"fp32": 7.5, "bf16": 3.8}
mode = args.dtype
if mode == "auto":
    mode = "fp32" if free_gb >= 9 else "bf16"
print(f"Chosen precision: {mode} (needs about {NEED[mode]} GB free)")

if free_gb < NEED[mode] and not args.force:
    sys.exit(f"Free RAM ({free_gb:.1f} GB) is below the estimate ({NEED[mode]} GB). Close the browser and other "
             "programs and run again, or try:  python test_load.py --dtype bf16 --force --threads 2")


def start_watchdog(pid, min_free_gb=0.4):
    """Separate process: if free RAM drops below min_free_gb, it stops this script
    so Windows does not freeze. It exits by itself when this script ends."""
    code = (
        "import psutil, time\n"
        f"p = psutil.Process({pid})\n"
        "while p.is_running():\n"
        f"    if psutil.virtual_memory().available / 1024**3 < {min_free_gb}:\n"
        "        p.kill()\n"
        f"        print('WATCHDOG: free RAM fell below {min_free_gb} GB, the test was stopped to keep the computer responsive.')\n"
        "        break\n"
        "    time.sleep(0.5)\n"
    )
    subprocess.Popen([sys.executable, "-c", code])


if free_gb < NEED[mode]:
    print(f"WARNING: running with less free RAM than estimated. Watchdog is active. Save your work first.")
start_watchdog(os.getpid())

dtype = torch.float32 if mode == "fp32" else torch.bfloat16


def load_base():
    """The argument name differs between transformers versions."""
    try:
        return AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype, low_cpu_mem_usage=True)


t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(BASE)
print("Loading base model ...", flush=True)
base = load_base()
rss = psutil.Process().memory_info().rss / 1024**3
print(f"Base model loaded in {time.time() - t0:.0f} s | process memory: {rss:.1f} GB", flush=True)

t0 = time.time()
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
print(f"Adapter loaded in {time.time() - t0:.1f} s | adapters: {list(model.peft_config)}", flush=True)


def generate(prompt):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
    new = out[0][inputs["input_ids"].shape[1]:]
    secs = time.time() - t
    return tokenizer.decode(new, skip_special_tokens=True).strip(), len(new), secs


print(f"\n=== WITH our adapter (prompt: {PROMPT!r}, up to {args.tokens} tokens) ===", flush=True)
text, n, secs = generate(PROMPT)
print(text)
print(f"[{n} tokens in {secs:.0f} s = {n / secs:.2f} tokens/s]")

if args.compare:
    print("\n=== adapter DISABLED (plain base model) ===", flush=True)
    with model.disable_adapter():
        text, n, secs = generate(PROMPT)
    print(text)
    print(f"[{n} tokens in {secs:.0f} s = {n / secs:.2f} tokens/s]")

print(f"\nPeak-ish process memory: {psutil.Process().memory_info().rss / 1024**3:.1f} GB")
