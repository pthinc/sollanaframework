"""
Trainer bridge utilities to inject BCE scoring/regularization into Torch/TF/Trainer/Diffusers loops.
- Torch: BCERegularizerTorch computes an auxiliary loss from hidden states via BCEEngine/SystemIntegrator.
- TF/Keras: tf_bce_regularizer returns a tensor (zero when TF absent) you can add to loss.
- HF Trainer: BCETrainerCallback logs BCE metric; can be paired with a custom compute_loss.
- Diffusers: latent_bce_regularizer gives a simple scalar from latents.
"""
from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING, Dict, Tuple, cast
import numpy as np

if TYPE_CHECKING:
    import torch as Torch  # type: ignore
    import tensorflow as Tf  # type: ignore
else:
    Torch = Any
    Tf = Any

BRAND_SIGNATURE = "Prometech Computer Sciences Inc Turkey - Ahmet Kahraman"
_telemetry_emitted = False
_telemetry_stats = {"sum_dhat": 0.0, "sum_rf_hat": 0.0, "sum_rs": 0.0, "sum_h": 0.0, "count": 0}

# String-based “critical” profile constants (can be used as a general BCE optimizer preset)
CRITICAL_K = 0.10288565627
CRITICAL_GAMMA = 2.0  # training-time target damping (passive fit gave ~4, but 2 is better for learning)
CRITICAL_TAU = 2.4298819588
_tf_integrator = None
COMPLIANCE_DEFAULTS = {
    "quality_ok": 0.8,
    "med_qms_ok": 0.7,
    "infosec_ok": 0.75,
    "social_ok": 0.7,
    "env_ok": 0.7,
}

try:
    import torch as _torch_mod  # type: ignore
    torch = cast(Any, _torch_mod)
    _torch_available = True
except Exception:
    torch = cast(Any, None)
    _torch_available = False

try:
    import tensorflow as _tf_mod  # type: ignore
    tf = cast(Any, _tf_mod)
    _tf_available = True
except Exception:
    tf = cast(Any, None)
    _tf_available = False

try:
    from transformers import TrainerCallback  # type: ignore
    _transformers_available = True
except Exception:
    TrainerCallback = object  # type: ignore
    _transformers_available = False

try:
    from datasets import load_dataset  # type: ignore
    _datasets_available = True
except Exception:
    load_dataset = None  # type: ignore
    _datasets_available = False

# Lazy import BCE pieces only when used


def _lazy_bce_engine():
    try:
        from sollana.behavioral_consciousness import bce_core_module as _bce  # type: ignore
        return _bce.BCEEngine()
    except Exception:
        try:
            from Behavioral_Consciousness import bce_core_module  # type: ignore
            return bce_core_module.BCEEngine()
        except Exception:
            return None


def _lazy_integrator():
    try:
        from sollana.behavioral_consciousness import system_integrator  # type: ignore
        return system_integrator.SystemIntegrator()
    except Exception:
        from Behavioral_Consciousness import system_integrator  # type: ignore
        return system_integrator.SystemIntegrator()


def _mean_pool(hidden, mask=None):
    if mask is None:
        return hidden.mean(dim=1)
    mask_f = mask.unsqueeze(-1).to(hidden)
    denom = mask_f.sum(dim=1).clamp_min(1e-6)
    return (hidden * mask_f).sum(dim=1) / denom


def _behavior_and_ctx_from_vec(vec):
    arr = np.asarray(vec)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.shape[0] > 8:
        arr = arr[:8]
    elif arr.shape[0] < 8:
        pad = np.zeros(8 - arr.shape[0], dtype=arr.dtype)
        arr = np.concatenate([arr, pad])
    behavior = {
        "phi_vec": arr.tolist(),
        "phi": 0.8,
        "history_count": 2,
        "decay_rate": 0.05,
        "meta": {"ethical_tag": "approved"},
        "decay_level": 0.05,
        "resonance": 0.7,
        "char_sal": 0.8,
        "lambda_base": 0.1,
    }
    ctx = {
        "user_type": "bağ_kurucu",
        "context_vec": arr,
        "char_vector": arr,
        "recent_matrix_for_iso": arr.reshape(1, -1),
        "clustering_matrix": arr.reshape(1, -1),
        "interaction_features": {"engagement": 0.6},
        "allow_auto": True,
    }
    ctx.update(COMPLIANCE_DEFAULTS)
    return behavior, ctx


def _lazy_integrator_for_text():
    try:
        return _lazy_integrator()
    except Exception:
        return None


def text_quality_metrics(text: str):
    length = len(text)
    repeat_ratio = 0.0
    symbol_ratio = 0.0
    if length > 0:
        max_char_freq = max(text.count(ch) for ch in set(text))
        repeat_ratio = max_char_freq / float(length)
        symbol_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / float(length)
    length_norm = min(1.0, length / 512.0)
    return length_norm, repeat_ratio, symbol_ratio


def quality_filter(text: str, min_len: int = 16, max_repeat: float = 0.35, max_symbol: float = 0.45) -> bool:
    length_norm, repeat_ratio, symbol_ratio = text_quality_metrics(text)
    if len(text) < min_len:
        return False
    if repeat_ratio > max_repeat:
        return False
    if symbol_ratio > max_symbol:
        return False
    return True


def bce_filter(text: str, threshold: float = 0.15) -> bool:
    integ = _lazy_integrator_for_text()
    if integ is None:
        return True
    h = abs(hash(text))
    phi_vec = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(8)]
    behavior = {
        "phi_vec": phi_vec,
        "resonance": 0.6,
        "decay_level": 0.1,
        "char_sal": 0.5,
        "decay_rate": 0.05,
        "attention": 0.5,
    }
    context = {
        "user_id": "trainer_filter",
        "text": text,
        "latency_ms": 200,
        "dt": 1.0,
        "env_reward": 0.0,
        "persist_path": False,
        "enable_vector_processor": False,
    }
    try:
        res = integ.process_behavior(behavior, context)
        bce_val = float(res.get("bce", 0.0))
        trust_val = float(res.get("trust", {}).get("score", 0.0)) if isinstance(res.get("trust"), dict) else 0.0
        score = 0.5 * bce_val + 0.5 * trust_val
        return score >= threshold
    except Exception:
        return True


def set_compliance_defaults(**kwargs):
    """Override compliance defaults used in bridge contexts."""
    for k, v in kwargs.items():
        if k in COMPLIANCE_DEFAULTS:
            try:
                COMPLIANCE_DEFAULTS[k] = float(v)
            except Exception:
                pass


def record_telemetry(event: Dict[str, Any]):
    """Ingest telemetry dicts (self_reward_step/drift_reflex) to drive adaptive scaling."""
    try:
        payload = event.get("payload", event)
        dhat = float(payload.get("Dhat", payload.get("dhat", 0.0)) or 0.0)
        rf = float(payload.get("rf_hat", payload.get("rfHat", 0.0)) or 0.0)
        rs = float(payload.get("Rs", payload.get("rs", 0.0)) or 0.0)
        hh = float(payload.get("H", payload.get("h", 0.0)) or 0.0)
        _telemetry_stats["sum_dhat"] += dhat
        _telemetry_stats["sum_rf_hat"] += rf
        _telemetry_stats["sum_rs"] += rs
        _telemetry_stats["sum_h"] += hh
        _telemetry_stats["count"] += 1
    except Exception:
        return


def telemetry_scale_snapshot() -> Dict[str, float]:
    """Return aggregated telemetry means and a composite score for scaling."""
    count = _telemetry_stats.get("count", 0)
    c = max(1, count)
    avg_dhat = _telemetry_stats.get("sum_dhat", 0.0) / c
    avg_rf = _telemetry_stats.get("sum_rf_hat", 0.0) / c
    avg_rs = _telemetry_stats.get("sum_rs", 0.0) / c
    avg_h = _telemetry_stats.get("sum_h", 0.0) / c
    score = 0.5 * avg_dhat + 0.5 * avg_rf
    if count == 0:
        # Seed with a small baseline so dynamic scaling is not stuck at unity when telemetry is absent.
        score = max(score, 0.1)
    return {
        "avg_dhat": avg_dhat,
        "avg_rf_hat": avg_rf,
        "avg_rs": avg_rs,
        "avg_h": avg_h,
        "score": score,
    }


def bce_optimizer_profile(
    k: float = CRITICAL_K,
    gamma: float = CRITICAL_GAMMA,
    tau: float = CRITICAL_TAU,
    clamp: Optional[Tuple[float, float]] = (0.0, 5.0),
) -> Dict[str, float | Tuple[float, float]]:
    """General BCE optimizer preset usable in training and realtime paths.

    Returns scale/clamp/telemetry_gain derived from critical constants.
    """
    # scale: nudge >1 using k*gamma*tau; telemetry_gain: proportional to gamma
    scale = 1.0 + (k * gamma * tau)
    telemetry_gain = max(1.0, gamma * 2.0)
    return {
        "scale": scale,
        "clamp": clamp,
        "telemetry_gain": telemetry_gain,
    }


def load_text_datasets(train_file: Optional[str], validation_file: Optional[str], text_column: str = "text") -> Dict[str, Any]:
    """Utility to load text/json/jsonl into Dataset dict; falls back to tiny dummy data if datasets missing."""
    if not train_file or not _datasets_available or load_dataset is None:
        samples = ["BCE pipeline nedir?", "Trust control nasıl çalışır?", "Ethic alarm guard eşiği nedir?"]
        return {"train": {text_column: samples}, "validation": {text_column: samples}}

    data_files = {"train": train_file}
    if validation_file:
        data_files["validation"] = validation_file
    ext = (train_file.split(".")[-1] if "." in train_file else "txt").lower()
    builder = "json" if ext in ["json", "jsonl"] else "text"
    raw = load_dataset(builder, data_files=data_files)  # type: ignore
    train_ds = cast(Any, raw["train"])
    if "validation" in raw:
        val_ds = cast(Any, raw["validation"])
    else:
        try:
            n_train = len(train_ds)
            val_ds = train_ds.select(range(min(2, n_train)))
        except Exception:
            val_ds = train_ds
    return {"train": train_ds, "validation": val_ds}


def _apply_padding_defaults(tok: Any, padding_side: Optional[str], for_inference: bool):
    if tok is None:
        return None
    if padding_side:
        try:
            tok.padding_side = padding_side
        except Exception:
            pass
    if for_inference:
        try:
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = getattr(tok, "eos_token", None) or getattr(tok, "unk_token", None)
        except Exception:
            pass
    return tok


def resolve_tokenizer_any(
    identifier: str,
    *,
    use_fast: bool = True,
    trust_remote_code: bool = False,
    padding_side: Optional[str] = None,
    for_inference: bool = False,
) -> Dict[str, Any]:
    """Best-effort tokenizer resolver across HF/tiktoken/sentencepiece."""
    tok = None
    source = None
    errors = {}

    if identifier:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(identifier, use_fast=use_fast, trust_remote_code=trust_remote_code)
            source = "huggingface"
        except Exception as exc:  # pragma: no cover - optional path
            errors["huggingface"] = str(exc)

    if tok is None:
        try:
            import tiktoken  # type: ignore

            tok = tiktoken.get_encoding(identifier)
            source = "tiktoken"
        except Exception as exc:  # pragma: no cover - optional path
            errors["tiktoken"] = str(exc)

    if tok is None and identifier.endswith(".model"):
        try:
            import sentencepiece as spm  # type: ignore

            sp = spm.SentencePieceProcessor()
            sp.load(identifier)
            tok = sp
            source = "sentencepiece"
        except Exception as exc:  # pragma: no cover - optional path
            errors["sentencepiece"] = str(exc)

    tok = _apply_padding_defaults(tok, padding_side, for_inference)
    return {"tokenizer": tok, "source": source, "errors": errors}


def training_tokenizer(identifier: str, **kwargs) -> Dict[str, Any]:
    return resolve_tokenizer_any(identifier, padding_side="right", for_inference=False, **kwargs)


def pretrain_tokenizer(identifier: str, **kwargs) -> Dict[str, Any]:
    return resolve_tokenizer_any(identifier, padding_side="right", for_inference=False, **kwargs)


def realtime_tokenizer(identifier: str, **kwargs) -> Dict[str, Any]:
    return resolve_tokenizer_any(identifier, padding_side="left", for_inference=True, **kwargs)


class BCERegularizerTorch:
    """Torch-side BCE regularizer. Takes a pooled hidden state and produces a scalar loss."""

    def __init__(
        self,
        use_integrator: bool = False,
        scale: Optional[float] = None,
        clamp: Optional[Tuple[float, float]] = None,
        telemetry_gain: Optional[float] = None,
    ):
        if not _torch_available:
            raise RuntimeError("Torch not available")
        self.use_integrator = use_integrator
        preset = bce_optimizer_profile()
        try:
            self.scale = float(scale) if scale is not None else float(preset.get("scale", 1.0))
        except Exception:
            self.scale = float(preset.get("scale", 1.0))
        self.clamp = clamp if clamp and len(clamp) == 2 else preset.get("clamp")  # type: ignore
        try:
            self.telemetry_gain = float(telemetry_gain) if telemetry_gain is not None else float(preset.get("telemetry_gain", 5.0))
        except Exception:
            self.telemetry_gain = float(preset.get("telemetry_gain", 5.0))
        self.engine = None if use_integrator else _lazy_bce_engine()
        if not use_integrator and self.engine is None:
            raise RuntimeError("BCEEngine could not be loaded; check package import path")
        self.integrator = _lazy_integrator() if use_integrator else None

    def __call__(self, pooled_hidden: "Torch.Tensor") -> "Torch.Tensor":
        """
        pooled_hidden: shape (batch, hidden)
        Returns a scalar torch loss (auxiliary) encouraging higher BCE output.
        """
        if not _torch_available:
            return 0.0  # type: ignore
        global _telemetry_emitted
        if not _telemetry_emitted:
            print(f"TELEMETRY: {{\"brand\": \"{BRAND_SIGNATURE}\", \"component\": \"bce_trainer_bridge\"}}", flush=True)
            _telemetry_emitted = True
        if pooled_hidden.ndim == 1:
            pooled_hidden = pooled_hidden.unsqueeze(0)
        bsz, dim = pooled_hidden.shape
        losses = []
        tsnap = telemetry_scale_snapshot()
        dyn_scale = self.scale * (1.0 + self.telemetry_gain * tsnap.get("score", 0.0))
        for i in range(bsz):
            phi_vec = pooled_hidden[i].detach().cpu().numpy()
            behavior, ctx = _behavior_and_ctx_from_vec(phi_vec)
            if self.integrator is not None:
                res = self.integrator.process_behavior(behavior, ctx)
                bce_val = float(res.get("bce", 0.0))
            else:
                if self.engine is None:
                    return torch.tensor(0.0, device=pooled_hidden.device)
                bce_val, _ = self.engine.compute_BCE(behavior, ctx)
            bce_val *= dyn_scale
            if self.clamp is not None:
                lo, hi = self.clamp
                bce_val = max(lo, min(hi, bce_val))
            # minimize negative BCE to encourage higher BCE
            loss_i = -bce_val
            losses.append(loss_i)
        if not losses:
            return torch.tensor(0.0, device=pooled_hidden.device)
        return torch.tensor(losses, device=pooled_hidden.device, dtype=pooled_hidden.dtype).mean()


def tf_bce_regularizer(pooled_hidden) -> Any:
    """TF regularizer via py_function wrapping SystemIntegrator/BCEEngine."""
    if not _tf_available:
        return 0.0
    global _tf_integrator
    if _tf_integrator is None:
        _tf_integrator = _lazy_integrator()
    if len(pooled_hidden.shape) == 1:
        pooled_hidden = tf.expand_dims(pooled_hidden, 0)

    def _bce_np(x_np):
        # x_np shape: (batch, hidden)
        scores = []
        for row in x_np:
            behavior = {
                "phi_vec": row.tolist(),
                "phi": 0.8,
                "history_count": 2,
                "decay_rate": 0.05,
                "meta": {"ethical_tag": "approved"},
                "decay_level": 0.05,
                "resonance": 0.7,
                "char_sal": 0.8,
                "lambda_base": 0.1,
            }
            ctx = {
                "user_type": "bağ_kurucu",
                "context_vec": row,
                "char_vector": row,
                "recent_matrix_for_iso": row.reshape(1, -1),
                "clustering_matrix": row.reshape(1, -1),
                "interaction_features": {"engagement": 0.6},
                "allow_auto": True,
            }
            res = _tf_integrator.process_behavior(behavior, ctx)
            scores.append(float(res.get("bce", 0.0)))
        return np.array(scores, dtype=np.float32).mean()

    bce_tf = tf.py_function(_bce_np, [pooled_hidden], tf.float32)
    return bce_tf


def torch_bce_from_outputs(
    outputs: Any,
    attention_mask: Optional["Torch.Tensor"],
    use_integrator: bool = True,
    scale: float = 1.0,
    clamp: Optional[Tuple[float, float]] = None,
    telemetry_gain: float = 5.0,
) -> Any:
    """Helper to derive BCE penalty from HF model outputs (expects hidden_states in outputs)."""
    if not _torch_available:
        return 0.0
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        return torch.tensor(0.0)
    hidden = outputs.hidden_states[-1]
    pooled = _mean_pool(hidden, attention_mask) if attention_mask is not None else hidden.mean(dim=1)
    reg = BCERegularizerTorch(use_integrator=use_integrator, scale=scale, clamp=clamp, telemetry_gain=telemetry_gain)
    return reg(pooled)


def torch_bce_with_meta(
    outputs: Any,
    attention_mask: Optional["Torch.Tensor"],
    use_integrator: bool = True,
    scale: float = 1.0,
    clamp: Optional[Tuple[float, float]] = None,
    telemetry_gain: float = 5.0,
):
    """Returns (bce_tensor, meta_dict) including adler/freud if integrator path is used."""
    if not _torch_available:
        return 0.0, {}
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        return torch.tensor(0.0), {}
    hidden = outputs.hidden_states[-1]
    pooled = _mean_pool(hidden, attention_mask) if attention_mask is not None else hidden.mean(dim=1)
    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    integrator = _lazy_integrator() if use_integrator else None
    if integrator is None:
        reg = BCERegularizerTorch(use_integrator=False, scale=scale, clamp=clamp, telemetry_gain=telemetry_gain)
        return reg(pooled), {}
    bce_vals = []
    metas = []
    for i in range(pooled.shape[0]):
        vec = pooled[i].detach().cpu().numpy()
        behavior, ctx = _behavior_and_ctx_from_vec(vec)
        res = integrator.process_behavior(behavior, ctx)
        val = float(res.get("bce", 0.0)) * scale
        if clamp is not None:
            lo, hi = clamp
            val = max(lo, min(hi, val))
        bce_vals.append(val)
        metas.append({
            "adler": res.get("adler", {}),
            "freud": res.get("freud", {}),
        })
    bce_tensor = torch.tensor(bce_vals, device=pooled.device, dtype=pooled.dtype)
    meta = metas[0] if metas else {}
    return bce_tensor.mean(), meta


def scheduled_bce_loss(
    pooled_hidden: "Torch.Tensor",
    step: int,
    base_weight: float = 0.2,
    target: float = 0.2,
    warmup_steps: int = 200,
    max_weight: float = 0.4,
    boost_span: int = 800,
    use_integrator: bool = True,
    scale: float = 1.0,
    clamp: Optional[Tuple[float, float]] = None,
    telemetry_gain: float = 5.0,
):
    """Compute weighted BCE with a ramp + optional auto-boost schedule inside the bridge.

    - Ramp from 0 -> min(base_weight, target) over warmup_steps.
    - After warmup, interpolate toward max_weight over boost_span steps (caps at max_weight).
    """
    if not _torch_available:
        return torch.tensor(0.0), 0.0, 0.0
    step = max(0, int(step))
    warmup_steps = max(1, int(warmup_steps))
    boost_span = max(1, int(boost_span))

    base_cap = min(target, base_weight)
    ramp = min(1.0, step / float(warmup_steps))
    eff_w = base_cap * ramp

    if max_weight > 0 and step > warmup_steps:
        boost_progress = min(1.0, (step - warmup_steps) / float(boost_span))
        eff_w = eff_w + (max_weight - eff_w) * boost_progress

    reg = BCERegularizerTorch(use_integrator=use_integrator, scale=scale, clamp=clamp, telemetry_gain=telemetry_gain)
    bce_raw = reg(pooled_hidden)
    loss = eff_w * bce_raw
    return loss, eff_w, bce_raw


def latent_bce_regularizer(latents: Any, weight: float = 0.1, regularizer: Optional[BCERegularizerTorch] = None, scale: float = 1.0, clamp: Optional[Tuple[float, float]] = None, telemetry_gain: float = 5.0) -> Any:
    """Diffusers-style latent penalty (torch only)."""
    if not _torch_available:
        return 0.0
    reg = regularizer or BCERegularizerTorch(use_integrator=True, scale=scale, clamp=clamp, telemetry_gain=telemetry_gain)
    flat = latents
    while flat.ndim > 2:
        flat = flat.view(flat.shape[0], -1)
    return weight * reg(flat)


def keras_bce_regularizer(weight: float = 0.1):
    if not _tf_available:
        return lambda _: 0.0

    class _Reg(tf.keras.regularizers.Regularizer):
        def __call__(self, x):
            return weight * tf_bce_regularizer(x)

        def get_config(self):
            return {"weight": weight}

    return _Reg()


__all__ = [
    "BCERegularizerTorch",
    "tf_bce_regularizer",
    "torch_bce_from_outputs",
    "torch_bce_with_meta",
    "scheduled_bce_loss",
    "latent_bce_regularizer",
    "keras_bce_regularizer",
    "load_text_datasets",
    "resolve_tokenizer_any",
    "training_tokenizer",
    "pretrain_tokenizer",
    "realtime_tokenizer",
    "bce_optimizer_profile",
    "record_telemetry",
    "telemetry_scale_snapshot",
    "BRAND_SIGNATURE",
    "set_compliance_defaults",
    "quality_filter",
    "bce_filter",
    "text_quality_metrics",
]
