# 10-Epoch BCE Trainer Run (prajjwal1/bert-tiny)

## Run Facts

| Item | Value |
| --- | --- |
| Model | prajjwal1/bert-tiny |
| Dataset | data/huggingface_turkish_english_sample.txt |
| Epochs | 10 |
| Batch size | 1 |
| Block size | 128 |
| Learning rate | 5e-5 (linear decay, warmup 400 steps) |
| BCE weight schedule | start 0.2, ramps to 0.5 with boost span 2000 |
| Steps | 2000 |
| Wall clock | ~106.96s |
| Throughput | ~18.7 steps/s |
| Final train loss | 2.3136 (Trainer aggregate) |

## Final Checkpoint Snapshot (step 2000)

| Metric | Value |
| --- | --- |
| Epoch | 10.0 |
| Loss (per-log) | 0.4224 |
| Grad norm | 16.25 |
| Learning rate | 3.125e-08 |
| BCE effective weight | ~0.44 |
| BCE loss | -0.4630 |

## Schedule Highlights

| Stage | Step / Epoch | Loss | Grad norm | LR | BCE eff. w | BCE loss |
| --- | --- | --- | --- | --- | --- | --- |
| Warmup start | 0 / 0.00 | 10.7678 | 85.84 | 0.0 | 0.0005 | -0.4642 |
| Warmup mid | 400 / 2.00 | 7.1730 | 30.83 | 4.375e-05 | 0.085 | -0.4645 |
| Pre-boost midpoint | 1000 / 5.00 | 1.1147 | 22.44 | 2.5e-05 | 0.255 | -0.4618 |
| Late boost | 1500 / 7.50 | 0.3749 | 16.66 | 1.25e-05 | 0.353 | -0.4579 |
| End of run | 2000 / 10.00 | 0.4224 | 16.25 | 3.125e-08 | 0.44 | -0.4630 |

## Observations

- Loss dropped from ~10.8 at start to ~0.42 by epoch 10; Trainer aggregate is higher (2.31) due to averaging across the whole run.
- Grad norms cooled from ~85 to the mid-teens by the end.
- BCE effective weight smoothly ramped from ~0 to ~0.44; BCE loss stayed stable around -0.46 without spikes.
- No runtime errors after the trapz fallback; throughput remained stable (~18.7 steps/s).

## Next Steps for Quality

- Run a quick perplexity/eval on a held-out split (e.g., add `--validation_file` to the trainer script) to get an objective metric.
- If you have labels, compute accuracy/F1 on a small dev set to see classification quality.
- Compare checkpoints (e.g., step 1000 vs 2000) to ensure later training is helping.

## Evaluation (post-run)

| Metric | Value |
| --- | --- |
| eval_loss | 0.0580 |
| perplexity | 1.0598 |
| eval_samples_per_second | 52.7 |
| eval_steps_per_second | 52.7 |
| eval_runtime | 0.0379 s |

### Eval on separate small set (val_small.txt)

| Metric | Value |
| --- | --- |
| eval_loss | 0.2039 |
| perplexity | 1.2262 |
| eval_samples_per_second | 62.1 |
| eval_steps_per_second | 62.1 |
| eval_runtime | 0.4666 s |

## BCE vs No-BCE (eval only)

| Split | Mode | eval_loss | Δ vs BCE-on | perplexity | Δ vs BCE-on |
| --- | --- | --- | --- | --- | --- |
| auto validation | BCE on | 0.0580 | — | 1.0598 | — |
| auto validation | BCE off | 0.05827 | +0.47% | 1.0600 | +0.02% |
| val_small.txt | BCE on | 0.2039 | — | 1.2262 | — |
| val_small.txt | BCE off | 0.20411 | +0.11% | 1.2264 | +0.02% |
| wikitext-2 (val, first 400 non-empty lines) | BCE on | 1.6691 | — | 5.3074 | — |
| wikitext-2 (val, first 400 non-empty lines) | BCE off | 1.66934 | +0.01% | 5.3087 | +0.02% |

## Detailed BCE vs BCE-off (per-example analysis)

- Auto validation (2 samples, length <=64): CE identical for BCE on/off (mean 0.0583, PPL 1.0600). Total loss shifts slightly due to BCE weight (mean_total: on -0.1453, off -0.1268). Best text: "Translate to Turkish: God will help us." (CE 0.0303).
- val_small (29 samples): CE identical across modes (mean 0.2041, PPL 1.2264). BCE-on reduces total loss by ~0.0184 absolute (mean_total: on 0.0013, off 0.0197). Length buckets: <=64 tokens mean CE 0.1783; >64 bucket (only the 128-token math item) CE 0.9256.
- Hardest examples (by CE): math story (len 128, CE 0.926), long sentiment sentence (len 36, CE 0.656), and corridor-speed sentence (len 37, CE 0.517). Easiest: short translations (CE 0.0027–0.0098).
- BCE effective weight during eval: ~0.44 when on, 0.40 when off (scheduled_bce_loss uses base_weight=0.2 vs 0.0 but still returns integrator prior; we keep CE identical for fairness).

Source stats: bce_eval_stats.json (written during analysis).

Wikitext stats: wikitext_eval_stats.json (first 400 non-empty validation lines of wikitext-2-raw-v1, both modes run on the same subset; BCE effect again negligible).

Command used for BCE-off evals (BCE weights zeroed, integrator on to avoid engine import):

```bash
& "C:/Users/rockm/Desktop/Sollana Framework/.venv/Scripts/python.exe" -c "import sys, math, torch; sys.path.append('c:/Users/rockm/Desktop/Sollana Framework/llm_tests'); from train_llama_trainer import load_text_datasets, load_model_and_tokenizer, tokenize_fn, BCETrainer; from transformers import DataCollatorForLanguageModeling, TrainingArguments; model_dir='c:/Users/rockm/Desktop/Sollana Framework/llm_tests/out_hf_10ep_auto'; train_file='c:/Users/rockm/Desktop/Sollana Framework/data/huggingface_turkish_english_sample.txt'; val_file='c:/Users/rockm/Desktop/Sollana Framework/data/val_small.txt'; block_size=128; batch_size=1; model,tok=load_model_and_tokenizer(model_dir, device_map=None); results={};
for tag, vf in [('auto_val', None), ('val_small', val_file)]:
    raw=load_text_datasets(train_file, vf)
    text_col='text'
    tokenized={split: ds.map(lambda x: tokenize_fn(x,tok,block_size,text_col), batched=True, remove_columns=[c for c in ds.column_names if c!=text_col]) for split, ds in raw.items()}
    collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    args=TrainingArguments(output_dir=model_dir, per_device_eval_batch_size=batch_size, report_to=[], fp16=torch.cuda.is_available())
    trainer=BCETrainer(model=model, args=args, train_dataset=tokenized['train'], eval_dataset=tokenized['validation'], data_collator=collator, bce_weight=0.0, bce_use_integrator=True, bce_warmup_steps=1, bce_max_weight=0.0, bce_boost_span=1)
    metrics=trainer.evaluate(); metrics['perplexity']=math.exp(metrics['eval_loss']) if 'eval_loss' in metrics and math.isfinite(metrics['eval_loss']) else None
    results[tag]=metrics
print(results)"
```

Command used:

```bash
& "C:/Users/rockm/Desktop/Sollana Framework/.venv/Scripts/python.exe" -c "import sys, math, torch; sys.path.append('c:/Users/rockm/Desktop/Sollana Framework/llm_tests'); from train_llama_trainer import load_text_datasets, load_model_and_tokenizer, tokenize_fn, BCETrainer; from transformers import DataCollatorForLanguageModeling, TrainingArguments; model_dir='c:/Users/rockm/Desktop/Sollana Framework/llm_tests/out_hf_10ep_auto'; train_file='c:/Users/rockm/Desktop/Sollana Framework/data/huggingface_turkish_english_sample.txt'; val_file='c:/Users/rockm/Desktop/Sollana Framework/data/val_small.txt'; block_size=128; batch_size=1; model,tok=load_model_and_tokenizer(model_dir, device_map=None); raw=load_text_datasets(train_file, val_file); text_col='text'; tokenized={split: ds.map(lambda x: tokenize_fn(x,tok,block_size,text_col), batched=True, remove_columns=[c for c in ds.column_names if c!=text_col]) for split, ds in raw.items()}; collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False); args=TrainingArguments(output_dir=model_dir, per_device_eval_batch_size=batch_size, report_to=[], fp16=torch.cuda.is_available()); trainer=BCETrainer(model=model, args=args, train_dataset=tokenized['train'], eval_dataset=tokenized['validation'], data_collator=collator, bce_weight=0.2, bce_use_integrator=True, bce_warmup_steps=400, bce_max_weight=0.5, bce_boost_span=2000); metrics=trainer.evaluate(); metrics['perplexity']=math.exp(metrics['eval_loss']) if 'eval_loss' in metrics and math.isfinite(metrics['eval_loss']) else None; print(metrics)"
```

Command used:

```powershell
$code = @'
import sys, math, torch
from transformers import DataCollatorForLanguageModeling, TrainingArguments
sys.path.append('c:/Users/rockm/Desktop/Sollana Framework/llm_tests')
from train_llama_trainer import load_text_datasets, load_model_and_tokenizer, tokenize_fn, BCETrainer
model_dir = 'c:/Users/rockm/Desktop/Sollana Framework/llm_tests/out_hf_10ep_auto'
train_file = 'c:/Users/rockm/Desktop/Sollana Framework/data/huggingface_turkish_english_sample.txt'
block_size = 128
batch_size = 1
model, tok = load_model_and_tokenizer(model_dir, device_map=None)
raw = load_text_datasets(train_file, None)
text_col = 'text'
tokenized = {split: ds.map(lambda x: tokenize_fn(x, tok, block_size, text_col), batched=True, remove_columns=[c for c in ds.column_names if c != text_col]) for split, ds in raw.items()}
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
args = TrainingArguments(output_dir=model_dir, per_device_eval_batch_size=batch_size, report_to=[], fp16=torch.cuda.is_available())
trainer = BCETrainer(model=model, args=args, train_dataset=tokenized['train'], eval_dataset=tokenized['validation'], data_collator=collator, bce_weight=0.2, bce_use_integrator=True, bce_warmup_steps=400, bce_max_weight=0.5, bce_boost_span=2000)
metrics = trainer.evaluate()
metrics['perplexity'] = math.exp(metrics['eval_loss']) if 'eval_loss' in metrics and math.isfinite(metrics['eval_loss']) else None
print(metrics)
'@
& "C:/Users/rockm/Desktop/Sollana Framework/.venv/Scripts/python.exe" -c $code
```

## Files of Interest

- Training state with full log history: out_hf_10ep_auto/checkpoint-2000/trainer_state.json
- Final checkpoint: out_hf_10ep_auto/checkpoint-2000/
