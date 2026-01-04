# Sollana Framework

Behavior-driven cognitive experimentation toolkit with BCE (Behavioral Consciousness Engine) regularization, telemetry, and plug-and-play integrators for language-model training and evaluation.

## Highlights
- **Behavioral Consciousness Engine (BCE):** Regularizer that nudges models toward stable, self-consistent behavioral patterns during training.
- **Telemetry-first:** Self-reward and drift reflex signals are recorded during runs so you can track stability and responsiveness.
- **Layered architecture:** "Behavioral Consciousness", "ego", "dna", and "decay" layers compose the cognitive loop; integrators wire these into trainers.
- **Eval-ready:** `.tmp_multi_eval.py` runs BCE-on vs BCE-off sweeps across sample corpora (wikitext103, cc_news, tinystories) and writes metrics to `llm_tests/out_hf_10ep_auto/multi100_eval_stats.json`.
- **Python-native:** Pure Python with optional PyTorch/Transformers/TensorFlow stacks; install extras only when you need them.

## Repository Map (quick mental model)
- Behavioral Consciousness/ — Core BCE logic, reflexes, selection, context tracking, experience transformer
- ego/ — Behavior routing, clustering, training loops and test harnesses
- dna/ — Behavior DNA, mutations, anomaly examples
- decay/ — PyTorch decay/BCE components
- backends/, sollana/backends/ — Trainer bridge, telemetry aggregation, integration utilities
- integrations/ — Reward shaping, temporal activation, feature weighting
- llm_tests/ — Evaluation scripts and outputs (e.g., `multi100_eval_stats.json`)
- docs/ — Long-form guides (see `docs/guide.md`)

## Prerequisites
- Python 3.9+
- Recommended: virtual environment
- For BCE training/eval: PyTorch 2.1+ (CPU or CUDA) and Hugging Face `transformers`, `datasets`, `accelerate`

## Installation
```bash
python -m venv .venv
. .venv/Scripts/activate      # Windows
# or
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
pip install -e .
```

## Quickstart: BCE Eval Sweep
Runs BCE-on vs BCE-off and writes metrics/telemetry.
```bash
.venv/Scripts/python .tmp_multi_eval.py
```
Outputs: `llm_tests/out_hf_10ep_auto/multi100_eval_stats.json` with loss, perplexity, and telemetry (avg_dhat, avg_rs, avg_h, score).

## Core Concepts
- **BCE Regularizer:** Adds scheduled auxiliary loss (with warmup and boost span) to encourage behavioral stability; see decay/bce_pytorch.py and backends/trainer_bridge.py.
- **Telemetry:** Reflex events (`self_reward_step`, `drift_reflex`) are recorded; trainer bridge aggregates into averages and a `score` (~0.012 in sample runs).
- **Integrator:** `Behavioral Consciousness/system_integrator.py` wires reflex outputs to telemetry and scheduler controls.

## Typical Training Loop (high level)
1) Build your model/dataloader (Hugging Face Trainer or custom loop).
2) Instantiate BCE regularizer (from `decay/bce_pytorch.py`) with desired `bce_weight`, `bce_max_weight`, `bce_warmup_steps`, `bce_boost_span`.
3) During training, call `scheduled_bce_loss(...)` and add to main loss.
4) Feed reflex outputs to `record_telemetry(event)` to keep telemetry live.
5) Log eval via `.tmp_multi_eval.py` or your own harness and watch `multi100_eval_stats.json`.

## Minimal BCE Integration Sketch (PyTorch)
```python
from decay.bce_pytorch import BCERegularizerTorch, scheduled_bce_loss

bce = BCERegularizerTorch(weight=2.0, max_weight=4.0,
                          warmup_steps=50, boost_span=2000)

for step, batch in enumerate(loader):
    logits = model(batch["input_ids"], labels=batch["labels"]).logits
    ce_loss = ce_fn(logits, batch["labels"])
    bce_loss, bce_w = scheduled_bce_loss(bce, logits, step)
    loss = ce_loss + bce_loss
    loss.backward()
    optimizer.step()
```

## Telemetry Wiring (conceptual)
```python
from backends.trainer_bridge import record_telemetry

record_telemetry({
    "event": "self_reward_step",
    "payload": {"Rs": rs_val, "H": h_val, "theta_r": 0.5,
                 "Dhat": dhat, "Rs_bal": rs_bal, "self_thank": False}
})
record_telemetry({
    "event": "drift_reflex",
    "user": "user",
    "C_now": c_now,
    "drift_rate": drift_rate,
    "rf_hat": rf_hat,
    "triggered": False,
    "actions": []
})
```
Aggregated telemetry (averages + `score`) is surfaced in eval outputs.

## Reproducible Eval
- Run: `.venv/Scripts/python .tmp_multi_eval.py`
- Check: `llm_tests/out_hf_10ep_auto/multi100_eval_stats.json`
- Compare `bce_on` vs `bce_off` perplexities; telemetry `score` should be non-zero when wiring is active.

## Extending
- **Custom datasets:** Swap loaders in `.tmp_multi_eval.py` or plug your `datasets` splits.
- **Custom schedulers:** Adjust BCE warmup/boost params in `bce_optimizer_profile` (see llm_tests/train_llama_trainer.py).
- **Telemetry sinks:** Pipe `record_telemetry` to your logger/DB for long runs.

## GitHub Usage
- Favor PRs: branch, commit, open PR; include eval JSON diff.
- Add new guides under `docs/` and keep README concise.
- Avoid committing large artifacts; keep `llm_tests/out_*` small or gitignored if they grow.

## Troubleshooting
- Telemetry zeros: ensure `record_telemetry` is called (SystemIntegrator wiring) and eval script is rerun.
- BCE weight has no effect: verify `scheduled_bce_loss` is added to total loss and warmup/boost settings are non-zero.
- Import errors: run `pip install -e .` after venv activation.

## License
See `licence.md`.