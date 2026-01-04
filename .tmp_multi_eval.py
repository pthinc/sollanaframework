import sys, math, torch, json, pathlib, datetime, traceback
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments

sys.path.append("c:/Users/rockm/Desktop/Sollana Framework/llm_tests")
from train_llama_trainer import load_model_and_tokenizer, tokenize_fn, BCETrainer
from backends.trainer_bridge import telemetry_scale_snapshot, bce_optimizer_profile

model_dir = "c:/Users/rockm/Desktop/Sollana Framework/llm_tests/out_hf_10ep_auto"
out_path = pathlib.Path(model_dir) / "multi100_eval_stats.json"
log_path = pathlib.Path(model_dir) / "multi100_eval.log"
block_size = 128
batch_size = 1
total_bytes = 100 * 1024 * 1024  # ~100MB combined per dataset
train_frac = 0.8
val_frac = 0.2

# More aggressive BCE settings to amplify effect size, seeded from critical profile.
_bce_profile = bce_optimizer_profile()
bce_weight_on = 2.0
bce_max_weight_on = 4.0
bce_warmup_steps = 50
bce_boost_span = 2000
bce_scale = float(_bce_profile.get("scale", 1.0))
bce_clamp = _bce_profile.get("clamp", (0.0, 5.0))
telemetry_gain = float(_bce_profile.get("telemetry_gain", 5.0)) * 1.25

# Light train-before-eval to wake the model slightly.
train_steps = 120
train_learning_rate = 1e-5
train_warmup_steps = 0

model, tok = load_model_and_tokenizer(model_dir, device_map=None)
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)


def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def collect_rows(stream_iter, limit_bytes, text_col="text"):
    acc = 0
    rows = []
    for ex in stream_iter:
        txt = ex.get(text_col, "")
        if not isinstance(txt, str) or not txt.strip():
            continue
        size = len(txt.encode("utf-8"))
        acc += size
        rows.append({text_col: txt})
        if acc >= limit_bytes:
            break
    return rows, acc


def prep_tokenized(raw_ds, text_col="text"):
    return raw_ds.map(
        lambda x: tokenize_fn(x, tok, block_size, text_col),
        batched=True,
        remove_columns=[c for c in raw_ds.column_names if c != text_col],
    )


def run_eval(name, train_ds, val_ds):
    tokenized = {
        "train": prep_tokenized(train_ds),
        "validation": prep_tokenized(val_ds),
    }
    args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=train_learning_rate,
        warmup_steps=train_warmup_steps,
        max_steps=train_steps if train_steps > 0 else None,
        num_train_epochs=1.0,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    def one_eval(tag, w, wmax):
        trainer = BCETrainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=collator,
            bce_weight=w,
            bce_use_integrator=True,
            bce_warmup_steps=bce_warmup_steps,
            bce_max_weight=wmax,
            bce_boost_span=bce_boost_span,
            bce_scale=bce_scale,
            bce_clamp=bce_clamp,
            telemetry_gain=telemetry_gain,
        )
        if train_steps > 0:
            trainer.train()
        metrics = trainer.evaluate()
        metrics["perplexity"] = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics and math.isfinite(metrics["eval_loss"]) else None
        telem = telemetry_scale_snapshot()
        metrics["meta_optimizer"] = {
            "lr": args.learning_rate,
            "bce_weight": w,
            "bce_max_weight": wmax,
            "bce_warmup_steps": bce_warmup_steps,
            "bce_boost_span": bce_boost_span,
            "telemetry": telem,
        }
        print(
            f"{name} {tag}: loss {metrics['eval_loss']:.4f}, ppl {metrics['perplexity']:.4f}, "
            f"bce_weight={w}, bce_max_weight={wmax}, warmup={bce_warmup_steps}, boost_span={bce_boost_span}, "
            f"telemetry_score={telem.get('score', 0.0):.6f}"
        )
        return metrics

    return {
        "bce_on": one_eval("bce_on", bce_weight_on, bce_max_weight_on),
        "bce_off": one_eval("bce_off", 0.0, 0.0),
    }


def eval_wikitext(results):
    train_rows, train_bytes = collect_rows(
        load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True),
        int(total_bytes * train_frac),
        text_col="text",
    )
    val_rows, val_bytes = collect_rows(
        load_dataset("wikitext", "wikitext-103-raw-v1", split="validation", streaming=True),
        int(total_bytes * val_frac),
        text_col="text",
    )
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    results["wikitext103"] = {
        "meta": {
            "train_bytes": train_bytes,
            "val_bytes": val_bytes,
            "train_len": len(train_ds),
            "val_len": len(val_ds),
        },
        **run_eval("wikitext103", train_ds, val_ds),
    }


def eval_single_split(name, hf_id, config=None):
    stream = load_dataset(hf_id, config, split="train", streaming=True)
    rows, total = collect_rows(stream, total_bytes, text_col="text")
    cut = max(1, int(len(rows) * train_frac))
    train_ds = Dataset.from_list(rows[:cut])
    val_ds = Dataset.from_list(rows[cut:]) if cut < len(rows) else Dataset.from_list(rows[-1:])
    return {
        "meta": {
            "total_bytes": total,
            "train_len": len(train_ds),
            "val_len": len(val_ds),
        },
        **run_eval(name, train_ds, val_ds),
    }


def write_partial(res_obj):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res_obj, f, ensure_ascii=False, indent=2)
    print("Wrote", out_path)


def main():
    log("Run start")
    results = {}
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            log(f"Loaded existing results with keys: {list(results.keys())}")
        except Exception as exc:
            log(f"Failed to load existing results: {exc}")

    for label, fn in [
        ("wikitext103", lambda res: eval_wikitext(res)),
        ("cc_news", lambda res: res.update({"cc_news": eval_single_split("cc_news", "cc_news")})),
        # wikipedia_tr and mc4_tr fail under datasets>=3 due to script loading; switch to a script-free parquet dataset
        ("tinystories", lambda res: res.update({"tinystories": eval_single_split("tinystories", "roneneldan/TinyStories")})),
    ]:
        if label in results:
            log(f"Skipping {label} (already in results)")
            continue
        try:
            log(f"Starting {label}")
            fn(results)
            write_partial(results)
            log(f"Finished {label}")
        except Exception as exc:
            log(f"ERROR in {label}: {exc}\n{traceback.format_exc()}")
            write_partial(results)
            # continue to next dataset


if __name__ == "__main__":
    main()
