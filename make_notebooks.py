"""Generates the three Colab notebooks into notebooks/. Run:  python make_notebooks.py"""
import nbformat as nbf

META = {
    "accelerator": "GPU",
    "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}


class NB:
    def __init__(self):
        self.cells = []

    def md(self, t):
        self.cells.append(nbf.v4.new_markdown_cell(t))

    def code(self, t):
        self.cells.append(nbf.v4.new_code_cell(t))

    def save(self, path):
        nb = nbf.v4.new_notebook()
        nb["cells"] = self.cells
        nb["metadata"] = META
        nbf.validate(nb)
        nbf.write(nb, path)


MOUNT = r"""import os
from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive/ml-coach-en"
os.makedirs(DRIVE_DIR, exist_ok=True)
print(os.listdir(DRIVE_DIR))"""

FETCH = r"""import shutil
from google.colab import files

def fetch(name):
    if os.path.exists(name):
        return
    p = f"{DRIVE_DIR}/{name}"
    if os.path.exists(p):
        shutil.copy(p, name)
        return
    print(f"{name} not found, please upload it:")
    up = files.upload()
    k = list(up)[0]
    if k != name:
        shutil.move(k, name)
    shutil.copy(name, p)
"""

# ---------------------------------------------------------------- 01 baseline
n = NB()
n.md("""# ML Learning Coach (English): Part 1, Baseline
Saves how the **untouched base model** (`Qwen/Qwen2.5-1.5B-Instruct`) answers our test prompts, BEFORE fine-tuning.
Runtime > Change runtime type > **T4 GPU**. Run the cells from top to bottom.""")
n.md("### 1) GPU and libraries")
n.code("!nvidia-smi")
n.code("!pip install -q -U transformers peft bitsandbytes accelerate datasets")
n.md("### 2) Connect Drive")
n.code(MOUNT)
n.md("### 3) Get the files\nUses Drive if the files are there, otherwise asks you to upload `train.jsonl` and `test_prompts.json`.")
n.code(FETCH + r"""
for name in ["train.jsonl", "test_prompts.json"]:
    fetch(name)
print("Files ready")""")
n.md("### 4) Read the data\nThe system prompt is taken from `train.jsonl`, so training and testing use exactly the same one.")
n.code(r"""import json

with open("train.jsonl", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f if l.strip()]
SYSTEM = rows[0]["messages"][0]["content"]
with open("test_prompts.json", encoding="utf-8") as f:
    tests = json.load(f)
print("Training examples:", len(rows), "| test prompts:", len(tests))""")
n.md("### 5) Load the base model in 4-bit\nTraining will use the same 4-bit (NF4) loading, so the baseline is measured the same way.")
n.code(r"""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model.eval()
print("GPU memory (GB):", round(torch.cuda.memory_allocated() / 1024**3, 2))""")
n.md("### 6) Answer the test prompts\n`do_sample=False` makes the output repeatable, so the before/after comparison is fair.")
n.code(r"""def generate(messages, max_new_tokens=300):
    msgs = [{"role": "system", "content": SYSTEM}] + messages
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


results = []
for t in tests:
    answer = generate(t["messages"])
    results.append({"id": t["id"], "type": t["type"],
                    "last_user": t["messages"][-1]["content"], "answer": answer})
    print("=" * 70)
    print(f"[{t['id']}]  ({t['type']})")
    print("STUDENT:", t["messages"][-1]["content"])
    print("BASE MODEL:\n" + answer)""")
n.md("### 7) Save the results\nKeep `baseline_outputs.json`: it is the \"before\" evidence for the README.")
n.code(r"""with open("baseline_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
shutil.copy("baseline_outputs.json", f"{DRIVE_DIR}/baseline_outputs.json")
files.download("baseline_outputs.json")""")
n.save("notebooks/01_baseline.ipynb")

# ---------------------------------------------------------------- 02 training
n = NB()
n.md("""# ML Learning Coach (English): Part 2, QLoRA training
Loads `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit, adds a LoRA adapter and trains it on `train.jsonl`.
Outputs: **our own adapter**, a loss curve and `training_info.json` (proof of a real training run).
Runtime > Change runtime type > **T4 GPU**. Run the cells from top to bottom, and make sure the LAST cell finishes.""")
n.md("### 1) GPU and libraries")
n.code("!nvidia-smi")
n.code("!pip install -q -U transformers peft bitsandbytes accelerate datasets")
n.md("### 2) Connect Drive\nThe adapter is written to Drive so it survives a closed session.")
n.code(MOUNT)
n.md("### 3) Get the training file")
n.code(FETCH + r"""
fetch("train.jsonl")
print("train.jsonl ready:", os.path.getsize("train.jsonl") // 1024, "KB")""")
n.md("### 4) Read the data and split off a validation set\n12 examples are kept aside. The model never trains on them, so we can watch for overfitting.")
n.code(r"""import json, random, hashlib
from datasets import Dataset

raw = open("train.jsonl", "rb").read()
DATA_SHA = hashlib.sha256(raw).hexdigest()[:12]
rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]

random.seed(42)
random.shuffle(rows)
VAL_N = 12
val_rows, train_rows = rows[:VAL_N], rows[VAL_N:]
print(f"Total: {len(rows)} | train: {len(train_rows)} | validation: {len(val_rows)} | dataset SHA: {DATA_SHA}")""")
n.md("""### 5) Tokenization and masking
We want the model to learn **only from the coach answers**. System prompt and student messages get `labels = -100`, so no loss is computed on them.
In multi-turn conversations, every coach answer is learned.""")
n.code(r"""from transformers import AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MARKER, END = "<|im_start|>assistant\n", "<|im_end|>"
MAX_LEN = 2048


def encode(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    ids, labels, pos = [], [], 0
    while True:
        s = text.find(MARKER, pos)
        if s == -1:                                   # trailing part (masked)
            piece = tokenizer(text[pos:], add_special_tokens=False)["input_ids"]
            ids += piece; labels += [-100] * len(piece)
            break
        prompt = tokenizer(text[pos:s + len(MARKER)], add_special_tokens=False)["input_ids"]
        ids += prompt; labels += [-100] * len(prompt)  # system/user (masked)
        e = text.find(END, s + len(MARKER))
        answer = tokenizer(text[s + len(MARKER):e + len(END)], add_special_tokens=False)["input_ids"]
        ids += answer; labels += answer                # coach answer (learned)
        pos = e + len(END)
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def build(rs):
    ds = Dataset.from_list([encode(r["messages"]) for r in rs])
    return ds.filter(lambda x: len(x["input_ids"]) <= MAX_LEN)


train_ds, val_ds = build(train_rows), build(val_rows)
lens = [len(x) for x in train_ds["input_ids"]]
print(f"Train: {len(train_ds)} | validation: {len(val_ds)} | longest: {max(lens)} tokens | mean: {sum(lens)//len(lens)} tokens")

# Masking check: shows what the model actually LEARNS from
ex = train_ds[0]
learned = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
print("\n--- What the model learns from (should be coach answers only) ---")
print(tokenizer.decode(learned))""")
n.md("""### 6) Load the model in 4-bit and add the LoRA adapter
- **4-bit (NF4):** the base weights stay frozen and compressed, which saves memory.
- **LoRA (r=16):** only small adapter matrices are trained, about 1% of the model.""")
n.code(r"""import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)   # also enables gradient checkpointing

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()""")
n.md("### 7) Training\n3 epochs, effective batch size 8 (2 x 4 accumulation), learning rate 2e-4, cosine schedule. Validation loss is measured after every epoch.")
n.code(r"""import inspect, time
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, set_seed

set_seed(42)
params = inspect.signature(TrainingArguments.__init__).parameters
eval_kw = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"

args = TrainingArguments(
    output_dir="/content/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=2,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    seed=42,
    **{eval_kw: "epoch"},
)
collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collator)

t0 = time.time()
trainer.train()
TRAIN_SECONDS = round(time.time() - t0)
print(f"\nTraining finished in {TRAIN_SECONDS} seconds")""")
n.md("### 8) Loss curve")
n.code(r"""import matplotlib.pyplot as plt

log = trainer.state.log_history
train_pts = [(l["step"], l["loss"]) for l in log if "loss" in l]
eval_pts = [(l["step"], l["eval_loss"]) for l in log if "eval_loss" in l]

plt.figure(figsize=(7, 4))
plt.plot(*zip(*train_pts), label="train loss")
if eval_pts:
    plt.plot(*zip(*eval_pts), "o-", label="validation loss")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3)
plt.title("QLoRA training: loss")
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print("First train loss:", round(train_pts[0][1], 3))
print("Last train loss: ", round(train_pts[-1][1], 3))
for s, v in eval_pts:
    print(f"validation loss (step {s}):", round(v, 3))""")
n.md("""### 9) Save the adapter and the training evidence (IMPORTANT: this cell must finish)
- `adapter/`: our trained LoRA weights (on Drive, plus `adapter.zip` for your computer)
- `training_info.json`: date, hyperparameters, dataset SHA, losses. It proves the run is ours.
- `training_log.json` and `loss_curve.png`: evidence for the README.""")
n.code(r"""import datetime
import transformers, peft

ADAPTER_DIR = f"{DRIVE_DIR}/adapter"
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

info = {
    "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "base_model": BASE,
    "gpu": torch.cuda.get_device_name(0),
    "dataset_sha256_prefix": DATA_SHA,
    "n_train": len(train_ds), "n_val": len(val_ds),
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": sorted(lora.target_modules)},
    "hyperparameters": {"epochs": 3, "batch_size": 2, "grad_accum": 4, "learning_rate": 2e-4,
                        "scheduler": "cosine", "warmup_ratio": 0.05, "optimizer": "paged_adamw_8bit",
                        "fp16": True, "max_len": MAX_LEN, "seed": 42},
    "train_seconds": TRAIN_SECONDS,
    "first_train_loss": train_pts[0][1], "last_train_loss": train_pts[-1][1],
    "eval_losses": eval_pts,
    "versions": {"transformers": transformers.__version__, "peft": peft.__version__, "torch": torch.__version__},
}
json.dump(info, open("training_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(log, open("training_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

for name in ["training_info.json", "training_log.json", "loss_curve.png"]:
    shutil.copy(name, f"{DRIVE_DIR}/{name}")

shutil.make_archive("/content/adapter", "zip", ADAPTER_DIR)
print("Adapter files:", os.listdir(ADAPTER_DIR))
print("Written to Drive:", ADAPTER_DIR)

for name in ["/content/adapter.zip", "training_info.json", "training_log.json", "loss_curve.png"]:
    files.download(name)""")
n.save("notebooks/02_train_qlora.ipynb")

# ---------------------------------------------------------------- 03 evaluation
n = NB()
n.md("""# ML Learning Coach (English): Part 3, Reload the adapter and compare before/after
Loads the adapter **from disk** with `PeftModel.from_pretrained`, runs the same test prompts and compares with the baseline.
Outputs: `finetuned_outputs.json` and `before_after.md` (for the README).
Runtime > Change runtime type > **T4 GPU**.""")
n.md("### 1) Libraries")
n.code("!pip install -q -U transformers peft bitsandbytes accelerate")
n.md("### 2) Connect Drive and check the adapter")
n.code(MOUNT + r"""
ADAPTER_DIR = f"{DRIVE_DIR}/adapter"
print("Adapter folder:", os.listdir(ADAPTER_DIR))
assert "adapter_config.json" in os.listdir(ADAPTER_DIR), 'Adapter not found. Run 02_train_qlora first (and its LAST cell).'"""
)
n.md("### 3) Get the files")
n.code(FETCH + r"""
for name in ["train.jsonl", "test_prompts.json", "baseline_outputs.json"]:
    fetch(name)
print("Files ready")""")
n.md("### 4) Read the data")
n.code(r"""import json

SYSTEM = json.loads(open("train.jsonl", encoding="utf-8").readline())["messages"][0]["content"]
tests = json.load(open("test_prompts.json", encoding="utf-8"))
before = {b["id"]: b["answer"] for b in json.load(open("baseline_outputs.json", encoding="utf-8"))}
print("Test prompts:", len(tests), "| baseline answers:", len(before))""")
n.md("""### 5) Load the base model, then attach OUR adapter
This is the required check: the model is rebuilt from the **adapter on disk**, independent of the training session.""")
n.code(r"""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")

model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
print("Adapter loaded:", list(model.peft_config))
print("GPU memory (GB):", round(torch.cuda.memory_allocated() / 1024**3, 2))""")
n.md("### 6) Run the same prompts with the fine-tuned model")
n.code(r"""def generate(messages, max_new_tokens=300):
    msgs = [{"role": "system", "content": SYSTEM}] + messages
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


after = []
for t in tests:
    ans = generate(t["messages"])
    after.append({"id": t["id"], "type": t["type"], "last_user": t["messages"][-1]["content"], "answer": ans})
    print("=" * 70)
    print(f"[{t['id']}]  ({t['type']})")
    print("STUDENT:", t["messages"][-1]["content"])
    print("\n--- BEFORE (base model) ---\n" + before[t["id"]])
    print("\n--- AFTER (fine-tuned) ---\n" + ans)""")
n.md("""### 7) Simple measurements: format, verdict correctness, memory
Not a full quality measure, but objective evidence: does the model follow the format? Does it give the right verdict on wrong claims? Does it give a NEW example when asked?""")
n.code(r"""MODE1 = ["Simple explanation:", "Example:", "Check yourself:"]
MODE2 = ["Verdict:", "What is right:", "What to improve:", "Think about it:"]
VERDICTS = ["Correct", "Partly correct", "Wrong"]
EXPECTED = {"check_overfitting": "Wrong", "check_knn": "Wrong", "check_regularization": "Correct",
            "check_random_forest": "Wrong", "check_split": "Correct"}


def labels_ok(a, labels):
    lines = a.split("\n")
    return all(any(l.startswith(x) for l in lines) for x in labels)


def verdict_of(a):
    first = a.split("\n")[0]
    return first.replace("Verdict:", "").strip() if first.startswith("Verdict:") else None


def format_ok(tid, a):
    if tid.startswith("off"):
        return (not a.startswith(("Simple explanation:", "Verdict:"))) and len(a.split()) <= 60 and a.count("?") == 1
    if tid.startswith("check") or tid == "follow_answer":
        return labels_ok(a, MODE2) and verdict_of(a) in VERDICTS
    return labels_ok(a, MODE1)


def example_line(a):
    for l in a.split("\n"):
        if l.startswith("Example:"):
            return l
    return None


after_d = {a["id"]: a["answer"] for a in after}
ids = [t["id"] for t in tests]

fmt_before = sum(format_ok(i, before[i]) for i in ids)
fmt_after = sum(format_ok(i, after_d[i]) for i in ids)
ver_before = sum(verdict_of(before[i]) == EXPECTED[i] for i in EXPECTED)
ver_after = sum(verdict_of(after_d[i]) == EXPECTED[i] for i in EXPECTED)

prev = next(t for t in tests if t["id"] == "follow_example")["messages"][1]["content"]
def new_example(a):
    e = example_line(a)
    return bool(e) and e != example_line(prev)

print(f"Answers in the right format:     before {fmt_before}/{len(ids)}  ->  after {fmt_after}/{len(ids)}")
print(f"Correct verdict (check mode):    before {ver_before}/{len(EXPECTED)}  ->  after {ver_after}/{len(EXPECTED)}")
print(f"Gave a NEW example (memory):     before {new_example(before['follow_example'])}  ->  after {new_example(after_d['follow_example'])}")
print()
for i in EXPECTED:
    print(f"{i:<22} expected: {EXPECTED[i]:<8} before: {verdict_of(before[i])!s:<15} after: {verdict_of(after_d[i])}")""")
n.md("### 8) Save the results\n`before_after.md` can be pasted into the README. `finetuned_outputs.json` is evidence.")
n.code(r"""with open("finetuned_outputs.json", "w", encoding="utf-8") as f:
    json.dump(after, f, ensure_ascii=False, indent=2)

lines = ["# Before / After comparison", "",
         f"- Answers in the right format: **{fmt_before}/{len(ids)} -> {fmt_after}/{len(ids)}**",
         f"- Correct verdict (check mode): **{ver_before}/{len(EXPECTED)} -> {ver_after}/{len(EXPECTED)}**", ""]
for t in tests:
    i = t["id"]
    lines += [f"## {i}  ({t['type']})", "", "**Student:**", "", "```", t["messages"][-1]["content"], "```", "",
              "**Before (base model):**", "", "```", before[i], "```", "",
              "**After (fine-tuned):**", "", "```", after_d[i], "```", ""]
open("before_after.md", "w", encoding="utf-8").write("\n".join(lines))

for name in ["finetuned_outputs.json", "before_after.md"]:
    shutil.copy(name, f"{DRIVE_DIR}/{name}")
    files.download(name)
print("Saved")""")
n.save("notebooks/03_evaluate_adapter.ipynb")
print("3 notebooks written")
