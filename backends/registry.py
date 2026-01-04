"""Backend registry to support torch, tensorflow/keras, diffusers-ready models, and a Rust FFI hook.

This module centralizes backend detection and exposes a minimal, duck-typed API for tensor ops.
It is designed to be imported by higher-level modules without eagerly importing heavy frameworks.
"""
from __future__ import annotations

import os
import types
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict

# Lazy imports guarded to avoid hard failures when optional deps are missing

try:  # PyTorch
    import torch  # type: ignore
    import torch.nn as torch_nn  # type: ignore
    import torch.nn.functional as torch_F  # type: ignore
    _torch_available = True
except Exception:  # pragma: no cover - import guard
    torch = None  # type: ignore
    torch_nn = None  # type: ignore
    torch_F = None  # type: ignore
    _torch_available = False

try:  # TensorFlow / Keras
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    _tf_available = True
except Exception:  # pragma: no cover
    tf = None  # type: ignore
    keras = None  # type: ignore
    _tf_available = False

try:  # Diffusers
    import diffusers  # type: ignore
    _diffusers_available = True
except Exception:  # pragma: no cover
    diffusers = None  # type: ignore
    _diffusers_available = False

try:  # Rust FFI placeholder; actual modules are loaded dynamically
    import importlib
    _importlib_available = True
except Exception:  # pragma: no cover
    importlib = None  # type: ignore
    _importlib_available = False


@dataclass
class Backend:
    name: str
    is_available: bool
    tensor: Callable[..., Any]
    nn: Optional[types.ModuleType] = None
    functional: Optional[types.ModuleType] = None
    device: Optional[Any] = None
    relu: Optional[Callable[..., Any]] = None
    exp: Optional[Callable[..., Any]] = None
    tanh: Optional[Callable[..., Any]] = None
    log: Optional[Callable[..., Any]] = None
    matmul: Optional[Callable[..., Any]] = None
    mean: Optional[Callable[..., Any]] = None
    clamp: Optional[Callable[..., Any]] = None
    to_device: Optional[Callable[[Any, Any], Any]] = None


def _torch_backend(device: Optional[str] = None) -> Backend:
    dev = _resolve_torch_device(device) if _torch_available else None
    return Backend(
        name="torch",
        is_available=_torch_available,
        tensor=(lambda *a, **k: torch.tensor(*a, **k)),
        nn=torch_nn,
        functional=torch_F,
        device=dev,
        relu=(lambda x: torch_F.relu(x)),
        exp=(lambda x: torch.exp(x)),
        tanh=(lambda x: torch.tanh(x)),
        log=(lambda x: torch.log(x)),
        matmul=(lambda a, b: a @ b),
        mean=(lambda x, dim=None: x.mean(dim=dim)),
        clamp=(lambda x, min=None, max=None: torch.clamp(x, min=min, max=max)),
        to_device=(lambda x, d=None: x.to(d or dev)) if _torch_available else None,
    )


def _resolve_torch_device(preferred: Optional[str]) -> Any:
    """Choose best-available torch device respecting env BACKEND_DEVICE and preferred.

    Order: explicit preferred -> BACKEND_DEVICE -> cuda -> mps -> xpu -> cpu.
    For AMD ROCm, torch exposes as 'cuda' on ROCm builds; DirectML can appear as 'privateuseone'.
    """
    if not _torch_available:
        return None
    cand = preferred or os.environ.get("BACKEND_DEVICE")
    if cand:
        return torch.device(cand)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel/XPU
        return torch.device("xpu")
    try:
        if torch.device("privateuseone"):
            return torch.device("privateuseone")
    except Exception:
        pass
    return torch.device("cpu")


def _tf_backend(device: Optional[str] = None) -> Backend:
    # TensorFlow handles devices internally; we keep interface parity
    return Backend(
        name="tensorflow",
        is_available=_tf_available,
        tensor=(lambda *a, **k: tf.convert_to_tensor(*a, **k)),
        nn=keras,
        functional=None,
        device=device or os.environ.get("BACKEND_DEVICE"),
        relu=(lambda x: tf.nn.relu(x)),
        exp=(lambda x: tf.math.exp(x)),
        tanh=(lambda x: tf.math.tanh(x)),
        log=(lambda x: tf.math.log(x)),
        matmul=(lambda a, b: tf.matmul(a, b)),
        mean=(lambda x, axis=None: tf.reduce_mean(x, axis=axis)),
        clamp=(lambda x, min=None, max=None: tf.clip_by_value(x, clip_value_min=min if min is not None else tf.reduce_min(x), clip_value_max=max if max is not None else tf.reduce_max(x))),
        to_device=None,
    )


def torch_inference_context(dtype: Optional[str] = None):
    """Context manager for torch inference with autocast if available."""
    if not _torch_available:
        from contextlib import nullcontext
        return nullcontext()
    import contextlib
    dev = _resolve_torch_device(None)
    amp_dtype = None
    if dtype == "fp16":
        amp_dtype = torch.float16
    elif dtype == "bf16":
        amp_dtype = torch.bfloat16
    if amp_dtype:
        return contextlib.ExitStack().__enter__() if False else torch.autocast(device_type=dev.type, dtype=amp_dtype)
    return torch.inference_mode()


def torch_compile(model, mode: str = "inference", fullgraph: bool = False):
    """Optional torch.compile wrapper if available (PyTorch 2.0+)."""
    if not _torch_available or not hasattr(torch, "compile"):
        return model
    return torch.compile(model, mode=mode, fullgraph=fullgraph)


def tf_enable_xla_and_mixed_precision(policy: str = "mixed_float16") -> Dict[str, Any]:
    if not _tf_available:
        return {"enabled": False}
    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy(policy)
    except Exception:
        pass
    return {"enabled": True, "policy": policy}


def backend_info() -> Dict[str, Any]:
    return {
        "torch": {"available": _torch_available, "version": getattr(torch, "__version__", None)},
        "tensorflow": {"available": _tf_available, "version": getattr(tf, "__version__", None)},
        "diffusers": {"available": _diffusers_available, "version": getattr(diffusers, "__version__", None) if _diffusers_available else None},
    }


def _numpy_backend():
    import numpy as np

    return Backend(
        name="numpy",
        is_available=True,
        tensor=(lambda *a, **k: np.array(*a, **k)),
        nn=None,
        functional=None,
        device=None,
        relu=(lambda x: np.maximum(x, 0)),
        exp=np.exp,
        tanh=np.tanh,
        log=np.log,
        matmul=np.matmul,
        mean=np.mean,
        clamp=(lambda x, min=None, max=None: np.clip(x, min, max)),
        to_device=None,
    )


def get_backend(preferred: Optional[str] = None, device: Optional[str] = None) -> Backend:
    """Return the best available backend.

    Order of selection if ``preferred`` is None: BACKEND env -> torch -> tensorflow -> numpy fallback.
    """
    name = preferred or os.environ.get("BACKEND", "torch").lower()
    if name == "torch":
        b = _torch_backend(device)
        if b.is_available:
            return b
    if name in {"tf", "tensorflow", "keras"}:
        b = _tf_backend(device)
        if b.is_available:
            return b
    # diffusers rides on torch; we surface availability info only
    if name == "diffusers" and _diffusers_available:
        b = _torch_backend(device)
        b.name = "diffusers(torch)"
        return b
    # rust backend is user-provided; expose numpy as safe default
    return _numpy_backend()


def ensure_backend(preferred: Optional[str] = None, device: Optional[str] = None) -> Backend:
    backend = get_backend(preferred, device)
    if not backend.is_available and backend.name != "numpy":
        # Warn early with a clear message
        raise RuntimeError(
            f"Requested backend '{preferred}' is not available. "
            "Install the corresponding dependency or choose another BACKEND."
        )
    return backend


def is_diffusers_available() -> bool:
    return _diffusers_available


def is_rust_available(module_name: str) -> bool:
    if not _importlib_available:
        return False
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False
