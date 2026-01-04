"""Hugging Face model and dataset loaders with token/cache handling.

- Uses lazy imports to avoid pulling heavy deps unless needed.
- Respects HF_TOKEN/HUGGINGFACE_TOKEN and DATA_CACHE_DIR env vars (also see backends.config).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .config import get_cache_dir, get_hf_token


def _resolve_token(token: Optional[str]) -> Optional[str]:
    return token or get_hf_token()


def _resolve_cache(cache_dir: Optional[str]) -> Path:
    return get_cache_dir(cache_dir)


def load_transformer_model(model_id: str, device: Optional[str] = None, token: Optional[str] = None,
                           cache_dir: Optional[str] = None):
    """Load a transformer model and tokenizer with optional auth and cache control."""
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    auth = _resolve_token(token)
    cache = _resolve_cache(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=auth, cache_dir=cache)
    model = AutoModel.from_pretrained(model_id, use_auth_token=auth, cache_dir=cache)
    if device:
        try:
            import torch  # type: ignore

            model.to(torch.device(device))
        except Exception:
            pass
    return model, tokenizer


def load_diffusers_pipeline(pipeline_cls, model_id: str, torch_dtype=None, device: Optional[str] = None,
                            token: Optional[str] = None, cache_dir: Optional[str] = None):
    """Load a Diffusers pipeline lazily."""
    from diffusers import DiffusionPipeline  # type: ignore

    auth = _resolve_token(token)
    cache = _resolve_cache(cache_dir)
    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=torch_dtype, use_auth_token=auth, cache_dir=cache)
    if device:
        pipe.to(device)
    return pipe


def load_dataset(name_or_path: str, split: str = "train", token: Optional[str] = None,
                 cache_dir: Optional[str] = None):
    """Load a dataset via `datasets` with optional auth and cache dir."""
    from datasets import load_dataset  # type: ignore

    auth = _resolve_token(token)
    cache = _resolve_cache(cache_dir)
    return load_dataset(name_or_path, split=split, token=auth, cache_dir=cache)


__all__ = [
    "load_transformer_model",
    "load_diffusers_pipeline",
    "load_dataset",
]
