# bce_store.py
import os
import time
import json
import yaml
import tempfile
from typing import Dict, Any, Optional, Callable

DEFAULT_DIR = "behaviors"

SCHEMA = {
    "required": ["version", "behavior_id", "attention", "match_prob", "context_weight"]
}

def ensure_dir(d: str = DEFAULT_DIR):
    os.makedirs(d, exist_ok=True)
    return d

def _atomic_write(path: str, data: str):
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

def validate_schema(obj: Dict[str, Any]):
    # minimal validation, genişletilebilir (jsonschema, pydantic kullan)
    for k in SCHEMA.get("required", []):
        if k not in obj:
            raise ValueError(f"Missing required field {k}")

def save_behavior(behavior: Dict[str, Any], directory: str = DEFAULT_DIR,
                  fmt: str = "yaml", backup: bool = True,
                  on_save_hook: Optional[Callable[[Dict[str, Any]], None]] = None):
    ensure_dir(directory)
    behavior = dict(behavior)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    behavior.setdefault("updated_at", now)
    behavior.setdefault("created_at", now)
    validate_schema(behavior)
    fname = f"{behavior['behavior_id']}.bce"
    path = os.path.join(directory, fname)
    if backup and os.path.exists(path):
        bak = path + f".bak.{int(time.time())}"
        os.replace(path, bak)
    if fmt == "json":
        payload = json.dumps(behavior, ensure_ascii=False, indent=2)
    else:
        payload = yaml.safe_dump(behavior, sort_keys=False, allow_unicode=True)
    _atomic_write(path, payload)
    if on_save_hook:
        try:
            on_save_hook(behavior)
        except Exception:
            pass
    return path

def load_behavior(behavior_id: str, directory: str = DEFAULT_DIR) -> Dict[str, Any]:
    path = os.path.join(directory, f"{behavior_id}.bce")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return yaml.safe_load(raw)
    except Exception:
        return json.loads(raw)

def delete_behavior(behavior_id: str, directory: str = DEFAULT_DIR, archive: bool = True):
    path = os.path.join(directory, f"{behavior_id}.bce")
    if archive and os.path.exists(path):
        arch = path + f".deleted.{int(time.time())}"
        os.replace(path, arch)
        return arch
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def update_behavior(behavior_id: str, updates: Dict[str, Any], directory: str = DEFAULT_DIR,
                    on_update_hook: Optional[Callable[[Dict[str, Any]], None]] = None):
    obj = load_behavior(behavior_id, directory)
    obj.update(updates)
    obj["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_behavior(obj, directory, on_save_hook=on_update_hook)
    return obj

def list_behaviors(directory: str = DEFAULT_DIR):
    ensure_dir(directory)
    for fn in os.listdir(directory):
        if fn.endswith(".bce"):
            yield fn[:-4]
