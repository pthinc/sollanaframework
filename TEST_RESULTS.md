# Sollana BCE System – Test Results

## 1) Package Smoke (sollana.behavioral_consciousness)

- Command: `python -c "import json, numpy as np; import sollana.behavioral_consciousness.system_integrator as si; integ=si.SystemIntegrator(); ..."`
- Outcome: success (CUDA warning only)
- Key outputs:
  - bce: 0.0
  - trust score: 0.5311 (low)
  - decay score: 0.8143
  - activation_curve: -0.3997
  - path_score: 0.3105

## 2) Multi-scenario Integration (tmp_integ_test.py)

- Scenarios executed sequentially:
  - baseline
  - high_decay_high_reward
  - negative_reward_persist_path
  - low_ethics_high_drift
  - high_bayes_low_decay
- Highlights per scenario:
  - baseline: trust ~0.53 (low); decay 0.83; activation 0.18; path_score 0.31; shaped_reward 0.39; ethic_guard ok.
  - high_decay_high_reward: decay risk higher (score 0.45); activation ~1.0; trust 0.60 (medium); shaped_reward 1.0; ethic_guard ok.
  - negative_reward_persist_path: activation -0.87; shaped_reward -0.62; trust 0.55 (low); path persist wrote path export; ethic_guard ok.
  - low_ethics_high_drift: decay 0.36; activation 0.99; trust 0.62 (medium); shaped_reward 0.53; ethic_guard ok.
  - high_bayes_low_decay: decay 0.89; activation -0.59; trust 0.52 (low); shaped_reward 0.47; ethic_guard ok.
- PathMapper: exports recorded for each user_id; cumulative_phi remained 0.0 due to single-pass weighting (no time decay accumulation in these runs).

## 3) Training Script Checks

- File: `llm_tests/train_llama_trainer.py`
- Actions: migrated to `sollana.backends.trainer_bridge` imports; filters (quality_filter, bce_filter) shared from backends; integrator default on; label smoothing + grad clip enabled; warmup passed to scheduled BCE loss.
- Syntax check: `python -m py_compile llm_tests/train_llama_trainer.py` (pass)

## 4) Editable Package Setup

- `pip install -e .` completed successfully.
- Package modules:
  - `sollana.behavioral_consciousness.system_integrator` (wrapper to Behavioral Consciousness/system_integrator.py)
  - `sollana.backends.trainer_bridge` (wrapper to backends/trainer_bridge.py)

## Notes

- CUDA warning from triton is informational (GPU not found); CPU path used.
- Vector processor remains gated (`enable_vector_processor` False in tests) to avoid HF downloads.
- PathMapper cumulative remains low in single-step runs; multi-step or time-weighted sequences will grow it.
