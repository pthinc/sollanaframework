"""
Minimal Trainer script for causal LM fine-tune (e.g., meta-llama/Llama-3.2-1B or TinyLlama).
- Uses Hugging Face Trainer (Transformers >= 4.44), Datasets, Accelerate.
- Defaults to a tiny in-memory dataset so it can run as a smoke test; replace with your JSONL/other dataset.
"""
import argparse
import os
import sys
import json
import re
from typing import List, Optional, Dict, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from sollana.backends.trainer_bridge import (
    BCERegularizerTorch,
    BRAND_SIGNATURE,
    scheduled_bce_loss,
    set_compliance_defaults,
    quality_filter,
    bce_filter,
    bce_optimizer_profile,
)
from sollana.behavioral_consciousness.system_integrator import SystemIntegrator  # type: ignore


_sample_integrator = None


def make_tiny_dataset() -> Dataset:
    samples: List[str] = [
        "BCE pipeline nedir?", 
        "Trust control nasıl çalışır?", 
        "Ethic alarm guard eşiği nedir?",
    ]
    return Dataset.from_dict({"text": samples})


def _get_integrator():
    global _sample_integrator
    if _sample_integrator is None and SystemIntegrator is not None:
        try:
            _sample_integrator = SystemIntegrator()
        except Exception:
            _sample_integrator = None
    return _sample_integrator




def load_text_datasets(train_file: Optional[str], validation_file: Optional[str]) -> Dict[str, Dataset]:
    if not train_file:
        return {"train": make_tiny_dataset(), "validation": make_tiny_dataset()}
    data_files = {}
    data_files["train"] = train_file
    if validation_file:
        data_files["validation"] = validation_file
    ext = (os.path.splitext(train_file)[1] or ".txt").lower()
    if ext in [".json", ".jsonl"]:
        builder = "json"
    else:
        builder = "text"
    raw = load_dataset(builder, data_files=data_files)
    if "validation" not in raw:
        raw["validation"] = raw["train"].select(range(min(2, len(raw["train"]))))
    return raw


def load_model_and_tokenizer(model_id: str, device_map: str | None):
    tok = AutoTokenizer.from_pretrained(model_id)
    # ensure pad token exists
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.config.output_hidden_states = True
    return model, tok


def tokenize_fn(examples, tokenizer: AutoTokenizer, block_size: int, text_column: str):
    return tokenizer(examples[text_column], truncation=True, max_length=block_size)


class BCETrainer(Trainer):
    def __init__(
        self,
        *args,
        bce_weight: float = 0.2,
        bce_use_integrator: bool = True,
        bce_warmup_steps: int = 200,
        bce_max_weight: float = 0.4,
        bce_boost_span: int = 800,
        bce_scale: Optional[float] = None,
        bce_clamp: Optional[Tuple[float, float]] = None,
        telemetry_gain: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bce_weight = bce_weight
        self.bce_use_integrator = bce_use_integrator
        self.bce_warmup_steps = max(1, int(bce_warmup_steps))
        self.bce_max_weight = bce_max_weight
        self.bce_boost_span = max(1, int(bce_boost_span))
        preset = bce_optimizer_profile()
        self.bce_scale = float(bce_scale) if bce_scale is not None else float(preset.get("scale", 1.0))
        self.bce_clamp = bce_clamp if bce_clamp is not None else preset.get("clamp")
        self.telemetry_gain = float(telemetry_gain) if telemetry_gain is not None else float(preset.get("telemetry_gain", 5.0))
        self.bce_reg = BCERegularizerTorch(
            use_integrator=bce_use_integrator,
            scale=self.bce_scale,
            clamp=self.bce_clamp,
            telemetry_gain=self.telemetry_gain,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, output_hidden_states=True)
        loss = outputs.loss
        bce_loss = torch.tensor(0.0, device=loss.device)
        bce_eff_w = 0.0
        if getattr(outputs, "hidden_states", None) is not None:
            hidden = outputs.hidden_states[-1]
            attn = inputs.get("attention_mask") if isinstance(inputs, dict) else None
            if attn is not None:
                mask = attn.unsqueeze(-1).to(hidden)
                denom = mask.sum(dim=1).clamp_min(1e-6)
                pooled = (hidden * mask).sum(dim=1) / denom
            else:
                pooled = hidden.mean(dim=1)
            step = getattr(self.state, "global_step", 0) + 1
            weighted_bce, bce_eff_w, raw_bce = scheduled_bce_loss(
                pooled,
                step=step,
                base_weight=self.bce_weight,
                target=0.2,
                warmup_steps=self.bce_warmup_steps,
                max_weight=self.bce_max_weight,
                boost_span=self.bce_boost_span,
                use_integrator=self.bce_use_integrator,
                scale=self.bce_scale,
                clamp=self.bce_clamp,
                telemetry_gain=self.telemetry_gain,
            )
            bce_loss = raw_bce
            loss = loss + weighted_bce
        if torch.isfinite(bce_loss).item():
            try:
                self.log({
                    "bce_loss": float(bce_loss.detach().cpu()),
                    "bce_eff_w": float(bce_eff_w),
                })
            except Exception:
                pass
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output", default="llm_tests/out")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=0.05)  # tiny smoke epoch
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--train-file", type=str, default=None, help="Path to text/json/jsonl train file")
    parser.add_argument("--validation-file", type=str, default=None, help="Optional validation file")
    parser.add_argument("--text-column", type=str, default="text", help="Column name containing raw text")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--bce-weight", type=float, default=0.2)
    parser.add_argument("--bce-max-weight", type=float, default=0.4, help="Ceiling for auto-boosted BCE weight")
    parser.add_argument("--bce-boost-span", type=int, default=800, help="Steps after warmup to reach max BCE weight")
    parser.add_argument("--bce-use-integrator", action="store_true", default=True, help="Use SystemIntegrator instead of raw BCEEngine (default on)")
    parser.add_argument("--quality-ok", type=float, default=0.8)
    parser.add_argument("--med-qms-ok", type=float, default=0.7)
    parser.add_argument("--infosec-ok", type=float, default=0.75)
    parser.add_argument("--social-ok", type=float, default=0.7)
    parser.add_argument("--env-ok", type=float, default=0.7)
    parser.add_argument("--enable-quality-filter", action="store_true", default=True, help="Filter low-quality samples via simple heuristics (default on)")
    parser.add_argument("--disable-quality-filter", action="store_true", help="Explicitly disable quality filter")
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.35)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.45)
    parser.add_argument("--enable-bce-filter", action="store_true", default=True, help="Use SystemIntegrator-based filter for samples (default on if integrator available)")
    parser.add_argument("--disable-bce-filter", action="store_true", help="Explicitly disable BCE filter")
    parser.add_argument("--bce-filter-threshold", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    # Auto-enable filters/integrator unless explicitly disabled
    if args.disable_quality_filter:
        args.enable_quality_filter = False
    if args.disable_bce_filter:
        args.enable_bce_filter = False
    if SystemIntegrator is not None and not args.disable_bce_filter:
        args.enable_bce_filter = True
    if SystemIntegrator is not None:
        args.bce_use_integrator = True

    print(f"[PROMETECH] Training run initialized - {BRAND_SIGNATURE}")

    device_map = args.device_map or ("auto" if torch.cuda.is_available() else None)
    model, tok = load_model_and_tokenizer(args.model, device_map)

    raw = load_text_datasets(args.train_file, args.validation_file)

    def _apply_filters(ds: Dataset, name: str) -> Dataset:
        before = len(ds)
        if args.enable_quality_filter:
            ds = ds.filter(lambda x: quality_filter(x[args.text_column], args.min_len, args.max_repeat_ratio, args.max_symbol_ratio))
        if args.enable_bce_filter:
            ds = ds.filter(lambda x: bce_filter(x[args.text_column], args.bce_filter_threshold))
        after = len(ds)
        if before != after:
            print(f"[FILTER] {name}: {before} -> {after} samples after quality/BCE filters")
        return ds

    raw["train"] = _apply_filters(raw["train"], "train")
    raw["validation"] = _apply_filters(raw["validation"], "validation")
    if args.max_train_samples:
        raw["train"] = raw["train"].select(range(min(args.max_train_samples, len(raw["train"]))))
    if args.max_eval_samples:
        raw["validation"] = raw["validation"].select(range(min(args.max_eval_samples, len(raw["validation"]))))

    tokenized = {
        split: ds.map(
            lambda x: tokenize_fn(x, tok, args.block_size, args.text_column),
            batched=True,
            remove_columns=[c for c in ds.column_names if c != args.text_column],
        )
        for split, ds in raw.items()
    }

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        save_steps=1000,
        save_total_limit=1,
        report_to=[],
        fp16=torch.cuda.is_available(),
        bf16=False,
        label_smoothing_factor=args.label_smoothing,
        max_grad_norm=args.grad_clip,
    )

    trainer = BCETrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        bce_weight=args.bce_weight,
        bce_use_integrator=args.bce_use_integrator,
        bce_warmup_steps=max(1, args.warmup_steps or 200),
        bce_max_weight=args.bce_max_weight,
        bce_boost_span=args.bce_boost_span,
    )

    log_path = os.path.join("logs", "trainer_bce.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    set_compliance_defaults(
        quality_ok=args.quality_ok,
        med_qms_ok=args.med_qms_ok,
        infosec_ok=args.infosec_ok,
        social_ok=args.social_ok,
        env_ok=args.env_ok,
    )

    # Hook: log eval metrics after each eval
    orig_eval = trainer.evaluate

    def _logged_eval(*eval_args, **eval_kwargs):
        metrics = orig_eval(*eval_args, **eval_kwargs)
        try:
            payload = {"step": trainer.state.global_step, "metrics": metrics}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass
        return metrics

    trainer.evaluate = _logged_eval  # type: ignore

    trainer.train()
    os.makedirs(args.output, exist_ok=True)
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)

if __name__ == "__main__":
    main()
