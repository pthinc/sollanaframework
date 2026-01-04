# Sollana Framework Guide

In-depth reference for using Sollana’s Behavioral Consciousness Engine (BCE), telemetry, and training/evaluation flows.

## 1. Orientation
Sollana is a behavioral regularization layer for language models. It adds:
- **BCE regularizer:** A scheduled auxiliary loss that promotes stable, self-consistent behaviors.
- **Telemetry:** Continuous reflex signals (self-reward, drift) to monitor stability.
- **Integrators:** Utilities that wire BCE and telemetry into training loops.

The goal is to keep models behaviorally coherent while preserving task performance.

## 2. Architecture at a Glance
- **Behavioral Consciousness/** — core modules (context tracking, reflexes, selection, experience transformer, integrated mini loop)
- **ego/** — routing, clustering, behavioral processes, pipelines, training/eval harnesses
- **dna/** — behavior DNA, mutation/anomaly examples, temporal memory
- **decay/** — BCE PyTorch implementation and decay helpers
- **backends/** & **sollana/backends/** — trainer bridge, telemetry aggregation, optimizer profiles
- **integrations/** — reward shaping, temporal activation, feature weighting
- **llm_tests/** — eval scripts (e.g., `.tmp_multi_eval.py`) and outputs
- **docs/** — this guide

## 3. Installation
```bash
python -m venv .venv
. .venv/Scripts/activate      # Windows
# or
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
pip install -e .
```
Install optional stacks only as needed (PyTorch/Transformers/TensorFlow/Rust FFI).

## 4. Core Concepts
### 4.1 BCE Regularizer
- Adds auxiliary loss to the main CE loss.
- Scheduled via warmup and boost span to avoid early over-regularization.
- Implemented in `decay/bce_pytorch.py`; helper `scheduled_bce_loss` computes the weighted BCE term per step.
- Typical profile: `bce_weight=2.0`, `bce_max_weight=4.0`, `bce_warmup_steps=50`, `bce_boost_span=2000` (as used in `.tmp_multi_eval.py`).

### 4.2 Telemetry
- Events: `self_reward_step` and `drift_reflex` emitted by the integrator.
- Metrics aggregated in the trainer bridge: `avg_dhat`, `avg_rf_hat`, `avg_rs`, `avg_h`, and a composite `score`.
- Purpose: detect behavioral drift and reward balance while training/evaluating.

### 4.3 Integrator
- `Behavioral Consciousness/system_integrator.py` wires reflex outputs to `record_telemetry`.
- Ensures telemetry is non-zero when BCE is active and reflexes fire.

## 5. Dataflow (conceptual)
1) Inputs → model forward pass → logits
2) Compute CE loss
3) Compute BCE loss via `scheduled_bce_loss` (weighted by schedule)
4) Total loss = CE + BCE; backprop
5) Reflex signals emitted → `record_telemetry` aggregates averages and score
6) Eval harness writes metrics and telemetry to JSON

## 6. Quick Evaluation Workflow
Run BCE-on vs BCE-off sweep:
```bash
.venv/Scripts/python .tmp_multi_eval.py
```
Inspect results:
- Metrics: `llm_tests/out_hf_10ep_auto/multi100_eval_stats.json`
- Sections: `bce_on` and `bce_off` per dataset with `eval_loss`, `perplexity`, and `telemetry` block.

## 7. PyTorch Integration Example
```python
from decay.bce_pytorch import BCERegularizerTorch, scheduled_bce_loss
from backends.trainer_bridge import record_telemetry

bce = BCERegularizerTorch(weight=2.0, max_weight=4.0,
                          warmup_steps=50, boost_span=2000)

for step, batch in enumerate(loader):
    out = model(batch["input_ids"], labels=batch["labels"])
    ce_loss = ce_fn(out.logits, batch["labels"])
    bce_loss, bce_w = scheduled_bce_loss(bce, out.logits, step)
    loss = ce_loss + bce_loss

    loss.backward()
    optimizer.step(); optimizer.zero_grad()

    # optional telemetry example
    record_telemetry({
        "event": "self_reward_step",
        "payload": {"Rs": 0.0, "H": 0.0, "theta_r": 0.5,
                     "Dhat": float(bce_w), "Rs_bal": 0.0,
                     "self_thank": False}
    })
```

## 8. Hugging Face Trainer Integration (sketch)
Use a callback to add BCE to the loss and to emit telemetry.
```python
class BCETrainerCallback(TrainerCallback):
    def __init__(self):
        self.bce = BCERegularizerTorch(weight=2.0, max_weight=4.0,
                                       warmup_steps=50, boost_span=2000)

    def on_step_end(self, args, state, control, **kwargs):
        # telemetry example
        record_telemetry({"event": "drift_reflex", "user": "user",
                          "C_now": 1.0, "drift_rate": 0.0,
                          "rf_hat": 0.0, "triggered": False,
                          "actions": []})

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_substep_end(self, args, state, control, **kwargs):
        pass

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        ce_loss = outputs.loss
        logits = outputs.logits
        bce_loss, _ = scheduled_bce_loss(self.bce, logits, state.global_step)
        loss = ce_loss + bce_loss
        return (loss, outputs) if return_outputs else loss
```
Attach to Trainer: `trainer = Trainer(..., callbacks=[BCETrainerCallback()])`.

## 9. Telemetry Fields (reference)
- `avg_dhat`: Average behavioral deviation estimate (lower is steadier).
- `avg_rf_hat`: Average drift reflex estimate.
- `avg_rs`: Average self-reward.
- `avg_h`: Average entropy-ish helper metric.
- `score`: Composite scalar from aggregates; non-zero indicates telemetry is live.

## 10. Tuning BCE
- Start with `weight=2.0`, `max_weight=4.0`, `warmup_steps=50`, `boost_span=2000`.
- Increase `max_weight` for stronger behavioral pull; widen `boost_span` for longer influence.
- If main loss degrades, lower `weight` or extend `warmup_steps`.

## 11. Custom Datasets
- Swap dataset loading inside `.tmp_multi_eval.py` or your Trainer data collator.
- Keep sequence lengths and padding consistent with your model’s tokenizer.

## 12. Logging & Artifacts
- Eval JSON lives under `llm_tests/out_hf_10ep_auto/`.
- For long runs, stream `record_telemetry` to your logging backend (e.g., file, DB, Prometheus push).

## 13. GitHub Workflow Tips
- Create feature branches; keep README concise and move deep content into `docs/`.
- Include a small eval JSON diff or summary in PRs.
- Avoid committing large checkpoints; store externally.

## 14. Troubleshooting
- **Telemetry score is zero:** Ensure integrator calls `record_telemetry`; re-run eval.
- **BCE has no effect:** Confirm `scheduled_bce_loss` is added to total loss and warmup/boost are non-zero.
- **Import errors:** Run `pip install -e .` after venv activation; verify Python 3.9+.
- **Slow evals:** Reduce dataset slice sizes or disable BCE for quick smoke tests.

## 15. Next Steps
- Adjust BCE profile and rerun `.tmp_multi_eval.py` to observe perplexity/telemetry shifts.
- Wire telemetry into your training dashboard to monitor behavioral stability in real time.
- Explore `Behavioral Consciousness/` modules to customize reflex logic for your domain.