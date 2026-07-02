#!/usr/bin/env python3
# Must be set before torch is imported so the CUDA allocator picks it up.
# (Harmless / ignored on Intel XPU and CPU backends.)
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

"""
FormulaGen — Excel Formula Generation  |  Intel AI PC (Arc 140V GPU, 24 GB shared)
===================================================================================
LoRA fine-tuning of Qwen2.5 models (1.5B / 3B) for Excel formula generation,
NO quantization — bf16 base + LoRA adapters.

This version is BACKEND-AGNOSTIC: it auto-detects Intel Arc (XPU) → NVIDIA
(CUDA) → CPU and uses NO quantization (bitsandbytes is CUDA-only and is not
needed here — the Arc 140V has 24 GB of shared memory, plenty for fp16/bf16).

SETUP
-----
0. *** Install the Intel XPU build of PyTorch *** (the plain `pip install torch`
   gives a CPU-only build that CANNOT see the Arc GPU):
       pip uninstall -y torch
       pip install torch --index-url https://download.pytorch.org/whl/xpu
   Verify:  python -c "import torch; print(torch.xpu.is_available())"  → True

1. Install the remaining dependencies (note: NO bitsandbytes — CUDA only):
       pip install "transformers>=4.52.0" "datasets>=2.18.0" "accelerate>=0.27.0" \
                   "peft>=0.10.0" "trl>=0.8.6" \
                   sentencepiece protobuf

2. Place data files in ./data/:
       data/train.json          (~68 MB)
       data/valid.json          (~7 MB)
       data/test.json           (~13 MB)

3. Run:
       python formulagen_local.py

OUTPUT
------
    outputs/<exp_id>/    — model checkpoints
    results/<exp_id>/    — metrics.json + predictions.jsonl

═══════════════════════════════════════════════════════════════════
NO QUANTIZATION ON THE INTEL ARC (24 GB)
═══════════════════════════════════════════════════════════════════
The Arc 140V exposes ~24 GB of shared memory — there is no reason to
quantize, and bitsandbytes (4-bit/8-bit) is CUDA-only anyway. Everything
here runs in bf16 (native on Arc) with NO quantization.

LoRA bf16 memory (base frozen, only tiny adapters train, grad-checkpointing):
    1.5B → ~3 GB weights + ~6 GB activations(bs=8)  ≈  9 GB   ← easy
    3B   → ~6 GB weights + ~9 GB activations(bs=8)  ≈ 15 GB   ← fits with headroom

Full FT bf16 memory (weights + grads + optimizer):
    1B   ≈ 6 GB     1.5B ≈ 9 GB    ← both fit; ≥3B → use LoRA instead

═══════════════════════════════════════════════════════════════════
PATH TO HIGH EXACT MATCH  (target: 90%)
═══════════════════════════════════════════════════════════════════
1. Train on the FULL 52k dataset  ← the single biggest factor
   (MAX_TRAIN_EXAMPLES = None — already the default below)
2. LoRA, NO quantization, r=64 / alpha=128, bf16 base
3. Evaluate on the full 10k test set (MAX_TEST_SAMPLES = None)

Realistic expectations:
    5k examples, 1B model        → ~40–55% EM
    full 52k, 1B/1.5B model      → ~65–75% EM
    full 52k, Qwen2.5-3B (max)   → the best shot at ~80–90% EM

>>> Project constraint: models must be ≤3B, so Qwen2.5-3B-Instruct is the
    ceiling. If 3B still falls short of 90%, the remaining levers are: more
    epochs, higher LoRA rank (r=128), and cleaning/augmenting the training
    data — not a bigger model.
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import gc
import importlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# Patch: transformers ≥4.52 upcasts logits to fp32 inside ForCausalLMLoss,
# which causes a large VRAM spike on 6 GB GPUs. Keep logits in native dtype.
try:
    import transformers.loss.loss_utils as _lu
    import torch.nn.functional as _F

    def _causal_lm_loss_native_dtype(
        logits, labels, vocab_size,
        num_items_in_batch=None, ignore_index=-100, **kw
    ):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = _F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=ignore_index,
            reduction="sum" if num_items_in_batch is not None else "mean",
        )
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        return loss

    _lu.ForCausalLMLoss = _causal_lm_loss_native_dtype
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR    = "./data"
OUTPUTS_DIR = "./outputs"
RESULTS_DIR = "./results"

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# DEVICE DETECTION  (Intel Arc XPU  →  NVIDIA CUDA  →  CPU)
# ═══════════════════════════════════════════════════════════════════════════

def _detect_device() -> str:
    """Pick the best available backend."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"          # Intel Arc / iGPU (Lunar Lake 140V, etc.)
    if torch.cuda.is_available():
        return "cuda"          # NVIDIA
    return "cpu"

DEVICE = _detect_device()


def device_empty_cache() -> None:
    """Backend-agnostic GPU memory cache clear (no-op on CPU)."""
    if DEVICE == "xpu":
        torch.xpu.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def _bf16_supported() -> bool:
    if DEVICE == "xpu":
        return True            # Intel Arc (Xe / Xe2) supports bf16 natively
    if DEVICE == "cuda":
        return torch.cuda.is_bf16_supported()
    return False               # CPU → use fp32 (fp16/bf16 on CPU is slow/flaky)


if DEVICE == "xpu":
    gpu_name = torch.xpu.get_device_name(0)
    try:
        vram_gb = torch.xpu.get_device_properties(0).total_memory / 1e9
    except Exception:
        vram_gb = 16.0
    print(f"Backend        : Intel XPU (Arc)")
    print(f"GPU            : {gpu_name}")
    print(f"Memory (shared): {vram_gb:.1f} GB")
elif DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Backend        : CUDA")
    print(f"GPU            : {gpu_name}")
    print(f"VRAM           : {vram_gb:.1f} GB")
else:
    print("WARNING: No GPU detected — running on CPU will be extremely slow.")
    print("  To use the Intel Arc 140V GPU, install the XPU build of PyTorch:")
    print("    pip uninstall -y torch")
    print("    pip install torch --index-url https://download.pytorch.org/whl/xpu")
    vram_gb = 0

BF16_OK = _bf16_supported()

# Weight/compute dtype for loading models (no quantization anywhere).
if DEVICE == "cpu":
    COMPUTE_DTYPE = torch.float32
elif BF16_OK:
    COMPUTE_DTYPE = torch.bfloat16     # preferred on Intel Arc — stable + fast
else:
    COMPUTE_DTYPE = torch.float16
print(f"Compute dtype  : {COMPUTE_DTYPE}  (bf16 ok: {BF16_OK})")


def training_precision():
    """Return (fp16, bf16) flags for the HF Trainer/SFTConfig."""
    if DEVICE == "cpu":
        return False, False
    if BF16_OK:
        return False, True
    return True, False


# 8-bit Adam (paged_adamw_8bit) needs bitsandbytes = CUDA-only.
# On Intel XPU / CPU fall back to plain torch AdamW (adapters are tiny, so the
# extra optimizer-state memory is negligible).
ADAMW_OPTIM = "paged_adamw_8bit" if DEVICE == "cuda" else "adamw_torch"

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← edit this section to change what runs
# ═══════════════════════════════════════════════════════════════════════════

# ── Experiments ─────────────────────────────────────────────────────────────
# (exp_id,  hf_model_id,                          ft_type)
#
# ft_type options:
#   "lora"  → LoRA with bf16 base, NO quantization
#   "full"  → Full fine-tuning (bf16, ≤1.5B models only)

EXPERIMENTS = [
    # ── Fast sanity check: Qwen2.5-1.5B LoRA ────────────────────────────────
    ("qwen_1b_lora",   "Qwen/Qwen2.5-1.5B-Instruct",  "lora"),

    # ── The 90% push: Qwen2.5-3B (the ≤3B ceiling), bf16 LoRA, no quant ─────
    ("qwen_3b_lora",   "Qwen/Qwen2.5-3B-Instruct",    "lora"),

    # ── Optional extra comparison (uncomment to add) ────────────────────────
    # ("qwen_1b_full",   "Qwen/Qwen2.5-1.5B-Instruct",  "full"),
]

# ── Data caps ─────────────────────────────────────────────────────────────
# CRITICAL: the full 52k training set is REQUIRED to reach 70%+ EM.
# With only 5k examples you will get ~45–55% EM regardless of model/FT choice.
# ─────────────────────────────────────────────────────────────────────────────
MAX_TRAIN_EXAMPLES = None     # None = all 52,203 train examples  ← REQUIRED for 70%+
MAX_EVAL_EXAMPLES  = 300      # in-training validation (kept small for speed)
MAX_TEST_SAMPLES   = None     # None = full 10,111 test examples  ← needed for fair EM

# ── General ──────────────────────────────────────────────────────────────
SEED          = 42
MAX_LENGTH    = 512      # 24 GB Arc has room for the full 512-token context
WARMUP_RATIO  = 0.05
LOGGING_STEPS = 10

# ── LoRA (bf16 base on Arc, NO quantization) ───────────────────────────────
# Memory on 24 GB Arc: 3B bf16 LoRA → ~15 GB at bs=8 (gradient checkpointing on)
LORA_EPOCHS  = 5         # more epochs compensate for partial-param updates
LORA_LR      = 2e-4
LORA_BS      = 8         # 24 GB lets us push the per-device batch up from 4 → 8
LORA_GRAD_AC = 8         # effective global batch = 8×8 = 64  (unchanged)
LORA_R       = 64        # r=64 is key — lower rank hurts EM on structured tasks
LORA_ALPHA   = 128       # 2×rank
LORA_DROPOUT = 0.05

# ── Full FT (bf16, ≤1.5B models only — larger models need ft_type='lora') ──
FULL_EPOCHS  = 3
FULL_LR      = 2e-5
FULL_BS      = 4         # 24 GB Arc fits bs=4 for a 1.5B full-FT model
FULL_GRAD_AC = 16        # effective global batch = 64
# Above this size, full FT needs 8-bit weights (bitsandbytes, CUDA-only) which
# isn't available on Intel XPU — larger models must use ft_type='lora'.
FULL_MAX_PARAMS_B = 2.0

# ── Inference / Evaluation ────────────────────────────────────────────────
EVAL_BATCH_SIZE = 16   # 24 GB Arc handles beam-search inference at bs=16 comfortably
MAX_NEW_TOKENS  = 256

# LoRA target modules
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",          # attention
    "gate_proj", "up_proj", "down_proj",              # MLP / FFN
]

# ═══════════════════════════════════════════════════════════════════════════
# DATA FILE CHECK
# ═══════════════════════════════════════════════════════════════════════════

def ensure_data_files(required: list) -> None:
    missing_req = [f for f in required
                   if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing_req:
        raise FileNotFoundError(
            f"\nMissing required data files: {missing_req}\n"
            f"Place them in: {os.path.abspath(DATA_DIR)}/\n\n"
            "Expected files:\n"
            "  data/train.json          (~68 MB)\n"
            "  data/valid.json          (~7 MB)\n"
            "  data/test.json           (~13 MB)\n"
        )
    print("All required data files found.")

# ═══════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an Excel formula expert. Given an Excel table and a natural language "
    "question, generate the corresponding Excel formula. Output only the formula, "
    "nothing else."
)


def table_to_markdown(table: list) -> str:
    """Convert raw 2D table list to Markdown table format."""
    if not table or len(table) < 2:
        return ""
    # table[0] = column letters (A, B, C...), table[1] = column headers
    headers = table[1][1:]   # skip the row-index cell
    col_letters = table[0][1:]
    named_headers = [f"{l}: {h}" for l, h in zip(col_letters, headers)]
    lines = ["| " + " | ".join(named_headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in table[2:]:
        values = [str(v) for v in row[1:]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def get_table_schema_summary(table: list) -> str:
    if not table or len(table) < 2:
        return "Empty table."
    n_rows  = len(table) - 2
    n_cols  = len(table[0]) - 1
    headers = ", ".join(table[1][1:])
    return f"{n_rows} data rows × {n_cols} columns. Columns: {headers}."


def build_user_message(table_md: str, schema: str, question: str) -> str:
    return (
        f"Table schema: {schema}\n\n"
        f"Excel table:\n{table_md}\n\n"
        f"Question: {question}\n\n"
        f"Generate the Excel formula:"
    )


def extract_arguments(formula: str) -> list:
    args, depth, current = [], 0, []
    for ch in formula:
        if   ch == "(":                depth += 1; current.append(ch)
        elif ch == ")":                depth -= 1; current.append(ch)
        elif ch == "," and depth == 1: args.append("".join(current).strip()); current = []
        else:                          current.append(ch)
    if current:
        args.append("".join(current).strip())
    return [a.upper() for a in args if a.strip()]


def load_split(
    filepath: str,
    formula_key: str = "Formula2",
    max_examples: Optional[int] = None,
) -> list:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    examples: list = []
    for item in data:
        table_raw = item.get("Table", [])
        table_md  = table_to_markdown(table_raw)
        schema    = get_table_schema_summary(table_raw)
        max_row   = len(table_raw) - 1    if table_raw else 0
        max_col   = len(table_raw[0]) - 1 if (table_raw and table_raw[0]) else 0
        for qa in item.get("t5Formulas", []):
            formula = qa.get(formula_key, "")
            if not formula:
                continue
            user_msg = build_user_message(table_md, schema, qa["Question"])
            examples.append({
                "table_name":   item.get("TableName", ""),
                "question":     qa["Question"],
                "formula":      formula,
                "formula1":     qa.get("Formula",  ""),
                "formula2":     qa.get("Formula2", ""),
                "level":        qa.get("Level", "unknown"),
                "funcs":        qa.get("Funcs", []),
                "max_row":      max_row,
                "max_col":      max_col,
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": formula},
                ],
                "user_message": user_msg,
            })
            if max_examples and len(examples) >= max_examples:
                return examples
    return examples


def build_hf_dataset(examples: list) -> Dataset:
    return Dataset.from_list(examples)


def _fmt_text(tokenizer, messages: list) -> str:
    """Apply the tokenizer's chat template to a messages list, producing a plain string."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def add_text_field(ds: Dataset, tokenizer) -> Dataset:
    """
    Pre-format every example's 'messages' into a 'text' field using the model's
    chat template. This lets SFTTrainer use dataset_text_field='text' and completely
    bypasses TRL's get_training_chat_template(), which requires {% generation %}
    markers that some models (e.g., Gemma 3) don't have.
    """
    def _map(batch):
        return {"text": [_fmt_text(tokenizer, m) for m in batch["messages"]]}
    return ds.map(_map, batched=True, desc="Formatting text")


class CompletionOnlyCollator:
    """
    Self-contained completion-only-LM collator (no TRL dependency).

    Computes the LM loss ONLY on the assistant's formula tokens: everything up
    to and including the response template (e.g. '<|im_start|>assistant\\n') is
    masked to -100, as are pad tokens. This is the version-proof equivalent of
    TRL's DataCollatorForCompletionOnlyLM, which some TRL builds no longer ship.

    If the response template isn't found in a sequence (e.g. a very long table
    truncated to MAX_LENGTH before the assistant turn), that example falls back
    to full-sequence loss rather than being silently dropped — this guarantees
    no all-masked batch (which would produce a NaN loss).
    """

    def __init__(self, response_token_ids, tokenizer, ignore_index: int = -100):
        self.response_token_ids = list(response_token_ids)
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self._missing = 0

    def _find_response_end(self, ids: list) -> Optional[int]:
        resp, n = self.response_token_ids, len(self.response_token_ids)
        for j in range(len(ids) - n + 1):
            if ids[j:j + n] == resp:
                return j + n
        return None

    def __call__(self, examples):
        # Coerce to plain python ints so subsequence matching is unambiguous
        # (datasets usually yield lists already, but guard against tensors).
        input_ids = [[int(t) for t in e["input_ids"]] for e in examples]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = self.ignore_index
        for i, ids in enumerate(input_ids):
            end = self._find_response_end(ids)
            if end is None:
                self._missing += 1          # keep full-seq loss for this row
            else:
                labels[i, :end] = self.ignore_index
        batch["labels"] = labels
        return batch


def get_completion_collator(model_name: str, tokenizer):
    """
    Build a completion-only collator so the loss is computed ONLY on the
    assistant's formula tokens, not on the system prompt or question.

    Uses the self-contained CompletionOnlyCollator above (no TRL class needed).
    The response_template is pre-tokenized with add_special_tokens=False to avoid
    the BOS-mismatch problem that causes silent no-masking failures.
    Returns None (→ full-sequence loss) only for unknown model families.
    """
    name = model_name.lower()
    if "qwen" in name:
        response_str = "<|im_start|>assistant\n"
    else:
        print("  Unknown chat format — using full-sequence loss.")
        return None

    response_ids = tokenizer.encode(response_str, add_special_tokens=False)
    if not response_ids:
        print("  Empty response template — using full-sequence loss.")
        return None
    print(f"  Completion collator ready  (response template: {response_str!r}, "
          f"{len(response_ids)} tokens)")
    return CompletionOnlyCollator(response_ids, tokenizer)

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def extract_functions(f: str) -> list:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?=\s*\()", f.upper())

def extract_operators(f: str) -> list:
    return re.findall(r"<=|>=|<>|[<>=+\-*/&]", f)

def extract_cell_refs(f: str) -> list:
    return re.findall(r"\b[A-Z]{1,3}\d+(?::[A-Z]{1,3}\d+)?\b", f.upper())

def normalise(f: str) -> str:
    return re.sub(r"\s+", "", f).upper()

def token_f1(pred: list, gold: list) -> float:
    if not pred and not gold: return 1.0
    if not pred or  not gold: return 0.0
    pc, gc = Counter(pred), Counter(gold)
    common = sum((pc & gc).values())
    p, r   = common / len(pred), common / len(gold)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

def validate_syntax(f: str) -> bool:
    if not f.strip(): return False
    stack = []
    for ch in f:
        if   ch == "(": stack.append(ch)
        elif ch == ")":
            if not stack: return False
            stack.pop()
    return not stack and not bool(re.search(r"[+\-*/&]$|^[+\-*/&,]", f.strip()))

def col_to_num(col: str) -> int:
    n = 0
    for ch in col.upper():
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n

def check_grounding(f: str, max_row: int, max_col: int) -> bool:
    for ref in extract_cell_refs(f):
        for p in ref.split(":"):
            m = re.match(r"([A-Z]+)(\d+)", p)
            if m and (int(m.group(2)) > max_row or col_to_num(m.group(1)) > max_col):
                return False
    return True

def categorize_error(pred, gold, m, max_row, max_col) -> str:
    if m["exact_match"]:                            return "Correct"
    if not m["syntax_validity"]:                    return "Syntax Error"
    if not check_grounding(pred, max_row, max_col): return "Hallucination (Ref)"
    if m["function_f1"]  < 1.0:                    return "Wrong Logic (Function)"
    if m["cell_ref_f1"]  < 1.0:                    return "Spatial Error (Cell Ref)"
    if m["operator_f1"]  < 1.0:                    return "Operator Confusion"
    return "Other Structural"

def compute_metrics(pred: str, gold: str, max_row: int = 0, max_col: int = 0) -> dict:
    m = {
        "exact_match":        float(normalise(pred) == normalise(gold)),
        "exact_match_strict": float(pred.strip() == gold.strip()),
        "function_f1":   token_f1(extract_functions(pred), extract_functions(gold)),
        "operator_f1":   token_f1(extract_operators(pred), extract_operators(gold)),
        "cell_ref_f1":   token_f1(extract_cell_refs(pred), extract_cell_refs(gold)),
        "argument_f1":   token_f1(extract_arguments(pred), extract_arguments(gold)),
        "syntax_validity":  float(validate_syntax(pred)),
        "grounding_score":  float(check_grounding(pred, max_row, max_col)),
    }
    m["error_category"] = categorize_error(pred, gold, m, max_row, max_col)
    return m

def balance_parentheses(f: str) -> str:
    diff = f.count("(") - f.count(")")
    return f + ")" * diff if diff > 0 else f

def build_inference_prompts(tokenizer, examples: list) -> list:
    out = []
    for ex in examples:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": ex["user_message"]},
        ]
        try:
            t = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            t = f"{SYSTEM_PROMPT}\n\n{ex['user_message']}\n"
        out.append(t)
    return out

def generate_batch(model, tokenizer, prompts: list, device: str) -> list:
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=1024,
    ).to(device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,          # beam search outperforms greedy for structured output
            early_stopping=True,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = ids[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
    return [balance_parentheses(d.split("\n")[0].strip()) for d in decoded]


def run_evaluation(
    model, tokenizer, examples: list, output_dir: str, label: str = ""
) -> dict:
    device = DEVICE
    MK = [
        "exact_match", "exact_match_strict", "function_f1", "operator_f1",
        "cell_ref_f1", "argument_f1", "syntax_validity", "grounding_score",
    ]
    all_m, preds, errs = [], [], Counter()

    for start in range(0, len(examples), EVAL_BATCH_SIZE):
        batch   = examples[start: start + EVAL_BATCH_SIZE]
        prompts = build_inference_prompts(tokenizer, batch)
        ps      = generate_batch(model, tokenizer, prompts, device)
        for ex, pred, gold in zip(batch, ps, [e["formula"] for e in batch]):
            m          = compute_metrics(
                pred, gold, ex.get("max_row", 0), ex.get("max_col", 0)
            )
            m["level"] = ex["level"]
            all_m.append(m)
            errs[m["error_category"]] += 1
            preds.append({
                "table_name": ex["table_name"],
                "question":   ex["question"],
                "gold":       gold,
                "pred":       pred,
                **m,
            })
        if (start // EVAL_BATCH_SIZE) % 10 == 0:
            done = min(start + EVAL_BATCH_SIZE, len(examples))
            em   = sum(m["exact_match"] for m in all_m) / len(all_m)
            print(f"  {done:>6}/{len(examples)} | running EM: {em*100:.1f}%")

    n       = len(all_m)
    overall = {k: sum(m[k] for m in all_m) / n for k in MK}
    by_lvl  = defaultdict(list)
    for m in all_m:
        by_lvl[m["level"]].append(m)
    level_results = {
        lvl: (
            {k: sum(m[k] for m in ms) / len(ms) for k in MK}
            | {"count": len(ms)}
        )
        for lvl, ms in by_lvl.items()
    }
    error_analysis = {cat: c / n for cat, c in errs.items()}
    results = {
        "overall":        overall,
        "by_level":       level_results,
        "error_analysis": error_analysis,
        "num_examples":   n,
    }

    W   = 24
    hdr = f"  EVAL: {label}  (n={n:,})" if label else f"  EVALUATION  (n={n:,})"
    print(f"\n{'═'*65}\n{hdr}\n{'═'*65}")
    print("\n  [ OVERALL METRICS ]")
    for k, v in overall.items():
        print(f"    {k:<{W}}: {v*100:.2f}%")
    print("\n  [ ERROR ANALYSIS ]")
    for cat, freq in sorted(error_analysis.items(), key=lambda x: -x[1]):
        print(f"    {cat:<{W}}: {freq*100:.2f}%")
    for lvl in ["easy", "medium", "hard"]:
        if lvl not in level_results:
            continue
        lr = level_results[lvl]
        print(f"\n  [ {lvl.upper()} ]  (n={lr['count']})")
        for k in MK:
            print(f"    {k:<{W}}: {lr[k]*100:.2f}%")
    print(f"{'═'*65}\n")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "predictions.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"Saved → {output_dir}/")
    return results

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _load_causal_lm(model_name: str, **kwargs):
    """
    Load a causal LM preferring 'sdpa' attention (fused, hardware-accelerated)
    over 'eager'. 'eager' materializes attention with plain Python-level ops
    and is dramatically slower on XPU/CUDA — 'sdpa' is a drop-in speed win
    for any model that supports it (Qwen2 does). Falls back to 'eager' only
    if the installed torch/transformers can't build the model with sdpa.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="sdpa", **kwargs
        )
    except Exception as e:
        print(f"  sdpa attention unavailable ({e}); falling back to eager.")
        return AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="eager", **kwargs
        )


def _estimate_param_billions(model_name: str) -> float:
    """Quick heuristic to estimate parameter count from the model name."""
    name_lower = model_name.lower()
    for marker, size in [
        ("0.5b", 0.5), ("500m", 0.5), ("135m", 0.135),
        ("1.1b", 1.1), ("1.5b", 1.5), ("1b",   1.0),
        ("2.7b", 2.7), ("2b",   2.0), ("3b",   3.0),
    ]:
        if marker in name_lower:
            return size
    return 1.0   # default assumption


def load_model_lora_noquant(model_name: str):
    """
    bf16 base model, NO quantization. Base model is fully frozen; only LoRA
    adapters will be trained.
    """
    model = _load_causal_lm(
        model_name, torch_dtype=COMPUTE_DTYPE, trust_remote_code=True,
    ).to(DEVICE)
    model.gradient_checkpointing_enable()
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA {COMPUTE_DTYPE} (no quant) — {total/1e9:.2f}B params  "
          f"[base frozen; LoRA adapters only will be trained]")
    return model


def load_model_full_ft(model_name: str):
    """
    Full fine-tuning — all parameters trained, bf16, no quantization.
    Only intended for models ≤ FULL_MAX_PARAMS_B (larger models don't fit
    without 8-bit weights, which needs bitsandbytes / CUDA and isn't
    available on Intel XPU — use ft_type='lora' for those instead).
    """
    n_billions = _estimate_param_billions(model_name)
    if n_billions > FULL_MAX_PARAMS_B:
        raise RuntimeError(
            f"Full FT of a {n_billions:.1f}B model won't fit without 8-bit "
            "weights (bitsandbytes, CUDA-only, unavailable on Intel XPU). "
            "Use ft_type='lora' for models this size."
        )
    print(f"  Full FT {COMPUTE_DTYPE} — {n_billions:.1f}B params  (no quantization)")
    model = _load_causal_lm(
        model_name, torch_dtype=COMPUTE_DTYPE, trust_remote_code=True,
    ).to(DEVICE)
    model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable/1e9:.2f}B")
    return model


def build_lora_config(model, r: int, alpha: int, dropout: float) -> LoraConfig:
    param_names = {n.split(".")[-1] for n, _ in model.named_parameters()}
    targets     = [m for m in LORA_TARGET_MODULES if m in param_names] or "all-linear"
    print(f"  LoRA targets : {targets}")
    print(f"  LoRA rank    : r={r}, alpha={alpha}")
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=targets, bias="none",
    )

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        device_empty_cache()


def make_sft_config(
    output_dir: str,
    epochs: int,
    bs: int,
    grad_ac: int,
    lr: float,
    optim: str,
    n_train: int,
    use_fp16: bool,
    use_bf16: bool,
    eval_bs: int,
) -> SFTConfig:
    steps_per_epoch = max(1, n_train // (bs * grad_ac))
    warmup_steps    = max(1, int(steps_per_epoch * epochs * WARMUP_RATIO))
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_ac,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=optim,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="steps",   # save every N steps so you can pause/resume anytime
        save_steps=100,           # checkpoint often so a run can be paused/resumed cheaply
        save_total_limit=2,       # keep only last 2 checkpoints (saves disk space)
        load_best_model_at_end=False,
        report_to="none",
        seed=SEED,
        max_length=MAX_LENGTH,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        dataset_kwargs={"skip_prepare_dataset": False},
    )


def find_latest_checkpoint(output_dir: str):
    """Return the path of the latest checkpoint folder, or None if none exist."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and
           os.path.isdir(os.path.join(output_dir, d))
    ]
    if not ckpts:
        return None
    latest = sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(output_dir, latest)


def train_stage(model, tokenizer, train_ds, valid_ds, cfg: SFTConfig,
                data_collator=None):
    resume_from = find_latest_checkpoint(cfg.output_dir)
    if resume_from:
        print(f"  Resuming from checkpoint: {os.path.basename(resume_from)}")
    else:
        print("  Starting fresh (no checkpoint found).")

    kwargs = dict(
        model=model, args=cfg,
        train_dataset=train_ds, eval_dataset=valid_ds,
        callbacks=[ClearCacheCallback()],
    )
    if data_collator is not None:
        kwargs["data_collator"] = data_collator
    # Handle both old (tokenizer=) and new (processing_class=) TRL API
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **kwargs)
    trainer.train(resume_from_checkpoint=resume_from)
    return trainer

# ═══════════════════════════════════════════════════════════════════════════
# SINGLE EXPERIMENT  (train + save + evaluate)
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(exp_id: str, model_name: str, ft_type: str) -> dict:
    device_empty_cache()
    gc.collect()

    print(f"\n{'═'*65}")
    print(f"  EXPERIMENT : {exp_id}")
    print(f"  Model      : {model_name}")
    print(f"  FT type    : {ft_type.upper()}")
    print(f"{'═'*65}")

    out_dir = os.path.join(OUTPUTS_DIR, exp_id)
    res_dir = os.path.join(RESULTS_DIR, exp_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    train_cfg_path = os.path.join(out_dir, "train_config.json")
    metrics_path   = os.path.join(res_dir, "metrics.json")

    # Skip if fully cached
    if os.path.exists(train_cfg_path) and os.path.exists(metrics_path):
        print("  Fully cached — loading saved results.")
        with open(metrics_path) as f:
            r = json.load(f)
        r.update({"model_name": model_name, "ft_type": ft_type, "exp_id": exp_id})
        return r

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load training data ────────────────────────────────────────────────
    print("\nLoading data …")
    train_ex = load_split(
        os.path.join(DATA_DIR, "train.json"), max_examples=MAX_TRAIN_EXAMPLES
    )
    valid_ex = load_split(
        os.path.join(DATA_DIR, "valid.json"), max_examples=MAX_EVAL_EXAMPLES
    )
    # Pre-format using the model's chat template → 'text' field.
    # This bypasses TRL's {% generation %} requirement entirely.
    train_ds = add_text_field(build_hf_dataset(train_ex), tokenizer)
    valid_ds = add_text_field(build_hf_dataset(valid_ex), tokenizer)
    print(f"  Train : {len(train_ds):,}  |  Valid (in-training): {len(valid_ds):,}")

    # Completion collator masks prompt tokens so loss is formula-tokens only
    completion_collator = get_completion_collator(model_name, tokenizer)

    if not os.path.exists(train_cfg_path):
        # ── Model + hyperparameters per ft_type ──────────────────────────
        print(f"\nLoading model [{ft_type.upper()}] …")

        if ft_type == "lora":
            model   = load_model_lora_noquant(model_name)
            lora_c  = build_lora_config(model, LORA_R, LORA_ALPHA, LORA_DROPOUT)
            model   = get_peft_model(model, lora_c)
            model.print_trainable_parameters()
            # Required: makes frozen base layers propagate gradients through
            # checkpointed segments back to the LoRA adapters (use_reentrant=False).
            model.enable_input_require_grads()
            epochs, lr, bs, grad_ac = LORA_EPOCHS, LORA_LR, LORA_BS, LORA_GRAD_AC
            optim                   = ADAMW_OPTIM
            use_fp16, use_bf16      = training_precision()
            eval_bs                 = EVAL_BATCH_SIZE
            lora_r_used             = LORA_R

        elif ft_type == "full":
            # Adafactor manages its own precision — disable AMP (fp16/bf16=False)
            # so it doesn't conflict with the optimizer's internal scaler.
            model                   = load_model_full_ft(model_name)
            epochs, lr, bs, grad_ac = FULL_EPOCHS, FULL_LR, FULL_BS, FULL_GRAD_AC
            optim                   = "adafactor"
            use_fp16, use_bf16      = False, False
            eval_bs                 = 1
            lora_r_used             = None

        else:
            raise ValueError(f"Unknown ft_type: {ft_type!r}. Choose 'lora' or 'full'.")

        print(f"\nTraining {ft_type.upper()} …")
        s2_cfg = make_sft_config(
            out_dir, epochs=epochs, bs=bs, grad_ac=grad_ac,
            lr=lr, optim=optim, n_train=len(train_ds),
            use_fp16=use_fp16, use_bf16=use_bf16, eval_bs=eval_bs,
        )
        trainer = train_stage(model, tokenizer, train_ds, valid_ds, s2_cfg,
                               data_collator=completion_collator)

        # ── Save ──────────────────────────────────────────────────────────
        print(f"\nSaving → {out_dir} …")
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        with open(train_cfg_path, "w") as f:
            json.dump({
                "model_name": model_name, "ft_type": ft_type,
                "epochs": epochs, "lr": lr, "bs": bs, "grad_ac": grad_ac,
                "lora_r": lora_r_used,
                "quantization": "none",
            }, f, indent=2)
        del trainer; device_empty_cache(); gc.collect()

    # ── Load test data ─────────────────────────────────────────────────────
    print("\nLoading test split …")
    test_ex = load_split(
        os.path.join(DATA_DIR, "test.json"), max_examples=MAX_TEST_SAMPLES
    )
    print(f"  Test examples : {len(test_ex):,}")

    # ── Reload merged model for inference ─────────────────────────────────
    print("Loading model for inference …")
    if ft_type == "lora":
        base      = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=COMPUTE_DTYPE,
            trust_remote_code=True,
        ).to(DEVICE)
        inf_model = PeftModel.from_pretrained(base, out_dir).merge_and_unload()
        print("  LoRA adapter merged into base weights.")
    else:
        inf_model = AutoModelForCausalLM.from_pretrained(
            out_dir, torch_dtype=COMPUTE_DTYPE,
            trust_remote_code=True,
        ).to(DEVICE)

    inf_tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    if inf_tok.pad_token is None:
        inf_tok.pad_token = inf_tok.eos_token
    inf_tok.padding_side = "left"   # left-pad for generation
    inf_model.eval()
    device_empty_cache()

    results = run_evaluation(inf_model, inf_tok, test_ex, res_dir, label=exp_id)
    results.update({
        "model_name": model_name, "ft_type": ft_type, "exp_id": exp_id,
    })

    del inf_model; device_empty_cache(); gc.collect()
    return results

# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(all_results: dict) -> None:
    MK     = ["exact_match", "syntax_validity", "grounding_score",
               "function_f1", "operator_f1", "cell_ref_f1", "argument_f1"]
    labels = ["EM (norm)",  "Syntax OK", "Grounding",
               "Func F1",   "Op F1",     "CellRef F1", "Arg F1"]
    header = ["Experiment", "FT type", "Quant", "n"] + labels
    rows   = []

    for exp_id, res in all_results.items():
        if not res or "overall" not in res:
            rows.append([exp_id] + ["—"] * (len(header) - 1)); continue
        o    = res["overall"]
        ft   = res.get("ft_type", "?")
        # read quantization from saved train_config.json if present
        cfg_path = os.path.join(OUTPUTS_DIR, exp_id, "train_config.json")
        quant = "?"
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                quant = json.load(f).get("quantization", "?")
        n    = res.get("num_examples", "?")
        rows.append(
            [exp_id, ft, quant, str(n)]
            + [f"{o.get(k, 0)*100:.2f}" for k in MK]
        )

    col_w = [
        max(len(str(header[i])), max(len(str(r[i])) for r in rows)) + 2
        for i in range(len(header))
    ]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    print(f"\n{'═'*75}\n  FINAL COMPARISON TABLE  (test split)\n{'═'*75}")
    print(sep)
    print("|" + "|".join(h.center(w) for h, w in zip(header, col_w)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(str(c).center(w) for c, w in zip(row, col_w)) + "|")
    print(sep)

    scored = [(eid, r) for eid, r in all_results.items() if r and "overall" in r]
    if scored:
        best_id, best = max(scored, key=lambda x: x[1]["overall"].get("exact_match", 0))
        em = best["overall"]["exact_match"] * 100
        sv = best["overall"].get("syntax_validity", 0) * 100
        print(f"\n  Best : {best_id}  |  EM={em:.2f}%  |  Syntax={sv:.2f}%")
        print(f"\n  [ BEST MODEL — PER LEVEL ]")
        for lvl in ["easy", "medium", "hard"]:
            if lvl in best.get("by_level", {}):
                lr = best["by_level"][lvl]
                print(
                    f"    {lvl:<8}: EM={lr['exact_match']*100:.2f}%  "
                    f"Func-F1={lr['function_f1']*100:.2f}%  "
                    f"CellRef={lr['cell_ref_f1']*100:.2f}%  "
                    f"n={lr['count']}"
                )
    print()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif DEVICE == "xpu":
        torch.xpu.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    set_seed(SEED)

    ensure_data_files(required=["train.json", "valid.json", "test.json"])

    all_results: dict = {}
    for exp_id, model_name, ft_type in EXPERIMENTS:
        try:
            result = run_experiment(exp_id, model_name, ft_type)
            all_results[exp_id] = result
            em = result["overall"]["exact_match"] * 100
            print(f"\n  DONE  {exp_id}  →  EM = {em:.2f}%\n")
        except Exception as e:
            import traceback
            print(f"\n  FAILED  {exp_id}: {e}")
            traceback.print_exc()
            all_results[exp_id] = None
            device_empty_cache()
            gc.collect()

    print_comparison_table(all_results)
    print("All experiments finished.")
