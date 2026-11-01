"""Inference engine: base model + OUR LoRA adapter, loaded locally, with token streaming.

- Base model:   Qwen/Qwen2.5-1.5B-Instruct   (AutoModelForCausalLM.from_pretrained)
- Adapter:      ./adapter                    (PeftModel.from_pretrained(base, adapter_dir))
- Streaming:    TextIteratorStreamer + generate() running in a background thread
- No network calls and no external LLM APIs are used. After the first download of the base
  model, everything works offline.
"""
import copy
import os
import threading
import time

import psutil
import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

from system_prompt import SYSTEM

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = "adapter"
NEED_RAM_GB = {"bf16": 3.8, "fp32": 7.5}   # measured: weights + Python/PyTorch runtime


class _StopOnEvent(StoppingCriteria):
    """Lets us cancel a running generation (for example when the user leaves the page)."""

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(),
                          dtype=torch.bool, device=input_ids.device)


def _from_pretrained(cls, name, **kwargs):
    """Offline first: use the local Hugging Face cache without contacting the Hub.
    Only if the files are not cached yet (first run) it downloads them."""
    try:
        return cls.from_pretrained(name, local_files_only=True, **kwargs)
    except OSError:
        return cls.from_pretrained(name, **kwargs)


class ModelLoadError(Exception):
    """A problem the user can understand and fix (missing adapter, not enough memory, ...)."""


class CoachEngine:
    def __init__(self, adapter_dir=DEFAULT_ADAPTER_DIR, dtype="auto", threads=0, force=False):
        self.adapter_dir = adapter_dir
        self.prefix_cache_enabled = True
        self.last_stats = {}
        self._lock = threading.Lock()
        self._prefix_ids = None
        self._prefix_cache = None
        self._conv_ids = None       # tokens processed in the previous generation
        self._conv_cache = None     # and their attention cache
        self._cached_n = 0
        self._cache_error = ""
        self.info = {}
        self._load(dtype, threads, force)

    # ------------------------------------------------------------------ loading
    def _check_adapter(self):
        cfg = os.path.join(self.adapter_dir, "adapter_config.json")
        weights = os.path.join(self.adapter_dir, "adapter_model.safetensors")
        if not os.path.exists(cfg) or not os.path.exists(weights):
            raise ModelLoadError(
                f"Adapter not found in '{self.adapter_dir}'. The folder must contain "
                "adapter_config.json and adapter_model.safetensors (our trained LoRA weights). "
                "See the README, step 'Get the adapter'.")

    def _pick_device(self, dtype, force):
        if torch.cuda.is_available():
            return "cuda", torch.float16, "fp16"
        free = psutil.virtual_memory().available / 1024**3
        mode = "bf16" if dtype == "auto" else dtype
        if free < NEED_RAM_GB[mode] and not force:
            raise ModelLoadError(
                f"Not enough free memory: {free:.1f} GB free, about {NEED_RAM_GB[mode]} GB needed. "
                "Close the browser tabs and other heavy programs, then reload this page.")
        return "cpu", (torch.float32 if mode == "fp32" else torch.bfloat16), mode

    def _load(self, dtype, threads, force):
        self._check_adapter()
        self.device, torch_dtype, dtype_name = self._pick_device(dtype, force)
        if self.device == "cpu":
            torch.set_num_threads(threads or max(1, (os.cpu_count() or 4) // 2))
        t0 = time.time()
        try:
            self.tokenizer = _from_pretrained(AutoTokenizer, BASE_MODEL)
            try:
                base = _from_pretrained(
                    AutoModelForCausalLM, BASE_MODEL, dtype=torch_dtype, low_cpu_mem_usage=True)
            except TypeError:  # older transformers versions call it torch_dtype
                base = _from_pretrained(
                    AutoModelForCausalLM, BASE_MODEL, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
            base.to(self.device)
            self.model = PeftModel.from_pretrained(base, self.adapter_dir)   # OUR adapter
            self.model.eval()
        except MemoryError as e:
            raise ModelLoadError("Out of memory while loading the model. Close other programs "
                                 "and try again.") from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise ModelLoadError("Out of memory while loading the model. Close other programs "
                                     "and try again.") from e
            raise ModelLoadError(f"The model could not be loaded: {e}") from e
        except OSError as e:
            raise ModelLoadError(
                "The base model could not be found. The first run needs internet access to download "
                f"{BASE_MODEL} (about 3 GB). Afterwards it works offline. Details: {e}") from e
        self.load_seconds = time.time() - t0
        self._build_prefix_cache()
        self.info = {
            "device": self.device, "dtype": dtype_name, "load_seconds": round(self.load_seconds, 1),
            "adapters": list(self.model.peft_config), "prefix_cache": self._prefix_cache is not None,
            "process_memory_gb": round(psutil.Process().memory_info().rss / 1024**3, 1),
        }

    def _build_prefix_cache(self):
        """The system prompt is the same for every request, so its attention cache is computed once.
        On a slow CPU this saves most of the waiting time. If anything fails we simply run without it."""
        try:
            text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}], tokenize=False)
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            ids = ids.to(self.device)
            with torch.inference_mode():
                out = self.model(input_ids=ids, use_cache=True)
            self._prefix_ids, self._prefix_cache = ids, out.past_key_values
        except Exception:
            self._prefix_ids, self._prefix_cache = None, None

    # ------------------------------------------------------------------ generation
    def _cache_for(self, input_ids, adapter_enabled):
        """Returns a copy of the attention cache that covers the longest prefix this prompt shares
        with (a) the previous conversation turn or (b) the system prompt. None if nothing is reusable.
        If something goes wrong the reason is stored in self._cache_error (shown in last_stats)."""
        self._cached_n = 0
        self._cache_error = ""
        if not (self.prefix_cache_enabled and adapter_enabled):
            return None
        new = input_ids[0]
        best = None
        for name, ids, cache in (("conversation", self._conv_ids, self._conv_cache),
                                 ("system prompt", self._prefix_ids, self._prefix_cache)):
            if ids is None or cache is None:
                continue
            try:
                m = min(ids.shape[1], new.shape[0])
                same = int((ids[0, :m] == new[:m]).to(torch.int64).cumprod(0).sum().item())
                cache_len = int(cache.get_seq_length())
                n = min(same, new.shape[0] - 1, cache_len)    # keep at least 1 new token
            except Exception as e:
                self._cache_error += f"[{name}: lookup failed: {e!r}] "
                continue
            if n > 0 and (best is None or n > best[0]):
                best = (n, cache, cache_len)
        if best is None:
            return None
        n, cache, cache_len = best
        try:
            reusable = copy.deepcopy(cache)
            if n < cache_len:               # only cut when needed (conversation turns)
                reusable.crop(-(cache_len - n))   # negative value = number of tokens to remove
        except Exception as e:
            self._cache_error += f"[copy/crop failed: {e!r}] "
            return None
        self._cached_n = n
        return reusable

    def _remember(self, result):
        """Keeps the cache of the finished generation so the next turn only processes new text."""
        try:
            out = result[0]
            cache = getattr(out, "past_key_values", None)
            if cache is None:
                return
            self._conv_ids = out.sequences.detach()
            self._conv_cache = cache
        except Exception:
            self._conv_ids, self._conv_cache = None, None

    def stream(self, messages, max_new_tokens=300, adapter_enabled=True):
        """Generator that yields text pieces while the model is still generating.
        messages: [{"role": "user"|"assistant", "content": ...}, ...] without the system prompt.
        If the consumer stops iterating (for example the page was left), generation is cancelled."""
        if not self._lock.acquire(timeout=90):
            raise RuntimeError("The model is still busy with the previous answer. Please try again in a moment.")
        stop = threading.Event()
        started = False
        try:
            msgs = [{"role": "system", "content": SYSTEM}] + list(messages)
            inputs = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                          do_sample=False, temperature=None, top_p=None, top_k=None,
                          stopping_criteria=StoppingCriteriaList([_StopOnEvent(stop)]))
            cache = self._cache_for(inputs["input_ids"], adapter_enabled)
            if cache is not None:
                kwargs["past_key_values"] = cache
            failure, result = [], []

            def run():
                try:
                    with torch.inference_mode():
                        if adapter_enabled:
                            result.append(self.model.generate(**kwargs, return_dict_in_generate=True))
                        else:
                            with self.model.disable_adapter():
                                result.append(self.model.generate(**kwargs, return_dict_in_generate=True))
                    if adapter_enabled and self.prefix_cache_enabled:
                        self._remember(result)
                except Exception as e:                      # make sure the consumer never hangs
                    failure.append(e)
                    streamer.on_finalized_text("", stream_end=True)
                finally:
                    self._lock.release()                    # released only when the model is really free

            t0 = time.time()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            started = True
            first, parts = None, []
            for piece in streamer:
                if first is None and piece:
                    first = time.time() - t0
                parts.append(piece)
                yield piece
            thread.join()
            if failure:
                raise RuntimeError(f"Generation failed: {failure[0]}")
            total = time.time() - t0
            n_new = len(self.tokenizer("".join(parts), add_special_tokens=False)["input_ids"])
            self.last_stats = {
                "first_token_s": round(first or total, 1), "total_s": round(total, 1),
                "new_tokens": n_new, "tokens_per_s": round(n_new / max(total, 1e-9), 2),
                "cached_tokens": self._cached_n if cache is not None else 0,
                "adapter_enabled": adapter_enabled,
                "cache_error": self._cache_error or None,
            }
        finally:
            stop.set()                                      # cancels the worker if we left early
            if not started:
                self._lock.release()

    def generate(self, messages, **kwargs):
        """Non-streaming helper (used by tests)."""
        return "".join(self.stream(messages, **kwargs)).strip()
