"""
Quick evaluator: computes perplexity and BCE auxiliary score on a text/json/jsonl dataset.
- Loads causal LM, tokenizes text_column, runs forward with labels=input_ids.
- Perplexity: exp(mean loss)
- BCE: uses torch_bce_from_outputs (SystemIntegrator by default)
"""
import argparse
import math
from typing import List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backends.trainer_bridge import (
    torch_bce_with_meta,
    load_text_datasets,
    BRAND_SIGNATURE,
    set_compliance_defaults,
)


def iter_texts(raw: Any, text_column: str) -> List[str]:
    # Supports HF DatasetDict or simple dict fallback
    if isinstance(raw, dict) and "train" in raw:
        ds = raw["train"]
        if isinstance(ds, dict) and text_column in ds:
            return list(ds[text_column])
    try:
        return [str(x[text_column]) for x in raw["train"]]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--train-file", type=str, required=False, default=None)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--use-integrator", action="store_true", help="Use SystemIntegrator in BCE helper")
    parser.add_argument("--quality-ok", type=float, default=0.8)
    parser.add_argument("--med-qms-ok", type=float, default=0.7)
    parser.add_argument("--infosec-ok", type=float, default=0.75)
    parser.add_argument("--social-ok", type=float, default=0.7)
    parser.add_argument("--env-ok", type=float, default=0.7)
    args = parser.parse_args()

    print(f"[PROMETECH] Eval start - {BRAND_SIGNATURE}")

    set_compliance_defaults(
        quality_ok=args.quality_ok,
        med_qms_ok=args.med_qms_ok,
        infosec_ok=args.infosec_ok,
        social_ok=args.social_ok,
        env_ok=args.env_ok,
    )

    device_map = args.device_map or ("auto" if torch.cuda.is_available() else None)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.config.output_hidden_states = True
    model.eval()

    raw = load_text_datasets(args.train_file, None, text_column=args.text_column)
    texts = iter_texts(raw, args.text_column)
    if args.max_samples:
        texts = texts[: args.max_samples]
    if not texts:
        print("No samples to evaluate.")
        return

    total_loss = 0.0
    total_bce = 0.0
    adler_social_total = 0.0
    freud_conflict_total = 0.0
    freud_align_total = 0.0
    steps = 0
    for txt in texts:
        enc = tok(txt, truncation=True, max_length=args.block_size, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"], output_hidden_states=True)
        loss = out.loss
        bce, meta = torch_bce_with_meta(out, enc.get("attention_mask"), use_integrator=args.use_integrator)
        adler_social = meta.get("adler") if isinstance(meta, dict) else None
        freud_metrics = meta.get("freud") if isinstance(meta, dict) else None
        total_loss += float(loss.detach().cpu())
        total_bce += float(bce.detach().cpu()) if torch.is_tensor(bce) else float(bce)
        if isinstance(adler_social, dict):
            adler_social_total += float(adler_social.get("social_interest", 0.0))
        if isinstance(freud_metrics, dict):
            freud_conflict_total += float(freud_metrics.get("conflict", 0.0))
            freud_align_total += float(freud_metrics.get("drive_alignment", 0.0))
        steps += 1
    avg_loss = total_loss / max(steps, 1)
    avg_bce = total_bce / max(steps, 1)
    avg_adler_social = adler_social_total / max(steps, 1)
    avg_freud_conflict = freud_conflict_total / max(steps, 1)
    avg_freud_align = freud_align_total / max(steps, 1)
    ppl = math.exp(avg_loss)
    print({
        "avg_loss": avg_loss,
        "ppl": ppl,
        "avg_bce": avg_bce,
        "avg_adler_social": avg_adler_social,
        "avg_freud_conflict": avg_freud_conflict,
        "avg_freud_alignment": avg_freud_align,
        "samples": steps,
    })

if __name__ == "__main__":
    main()
