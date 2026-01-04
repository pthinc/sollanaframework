# BCE Evaluation Suite

Lightweight scripts to compare baseline vs BCE-augmented training on ~100MB text datasets.

## Requirements

- Python 3.10+
- PyTorch with CUDA (optional but recommended)
- `transformers`, `datasets`, `evaluate`, `accelerate`, `markdown-it-py`, `rich`

Install minimal deps:

```bash
pip install -r requirements.txt
```

## Datasets

We use three small language modeling datasets (~100MB each) to keep runs fast:

- `wikitext`: `wikitext-103-v1`
- `cc_news`: `cc_news`
- `tinystories`: `roneneldan/TinyStories`

## Key Files

- `bce_eval.py`: entrypoint for running baseline vs BCE
- `.tmp_multi_eval.py`: convenience multi-dataset runner
- `multi100_eval_stats.json`: aggregated results
- `TEST_RESULTS.md`: human-friendly summary table
- `out_hf_10ep_auto/`: training artifacts per dataset

## Quickstart

Run a single dataset (baseline and BCE back-to-back):

```bash
python bce_eval.py --dataset wikitext --epochs 3 --batch_size 8 --lr 5e-5
```

Outputs go to `out_hf_10ep_auto/<dataset>/`.

## Multi-dataset sweep

`.tmp_multi_eval.py` runs all configured datasets sequentially:

```bash
python .tmp_multi_eval.py
```

After it finishes, inspect the aggregate:

```bash
cat multi100_eval_stats.json
```

## BCE flag

Enable BCE with `--use_bce`. Example:

```bash
python bce_eval.py --dataset wikitext --use_bce --epochs 3 --batch_size 8 --lr 5e-5
```

## Repro tips

- Set `--seed` (defaults to 42)
- Keep `--epochs` small (3–5) for quick iterations
- Use `--eval_steps` to checkpoint more frequently if needed

## Troubleshooting

- If a dataset fails to download, swap to a smaller one (TinyStories works reliably)
- Check GPU visibility: `torch.cuda.is_available()`
- Markdown lint: ensure blank lines around fenced code blocks
