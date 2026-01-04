"""Token and dataset configuration helpers.

Environment variables used:
- HF_TOKEN: Hugging Face access token (optional)
- HF_DATASET_PATH: Local dataset path or hub repo id
- DATA_CACHE_DIR: Cache directory for model/dataset downloads
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return token.strip() if token else None


def get_dataset_path(default: Optional[str] = None) -> Optional[str]:
    path = os.environ.get("HF_DATASET_PATH") or default
    return path.strip() if path else None


def get_cache_dir(default: Optional[str] = None) -> Path:
    cache = os.environ.get("DATA_CACHE_DIR") or default or "~/.cache/sollana"
    return Path(cache).expanduser()


def ensure_cache_dir(default: Optional[str] = None) -> Path:
    path = get_cache_dir(default)
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["get_hf_token", "get_dataset_path", "get_cache_dir", "ensure_cache_dir"]
