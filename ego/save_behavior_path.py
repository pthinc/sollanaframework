# save_behavior_path.py
import json
import tempfile
import os
from jsonschema import validate, ValidationError

SCHEMA = {...}  # paste the JSON schema above

DATA_PATH = "data/behavior_paths.json"
os.makedirs("data", exist_ok=True)

def load_paths(path=DATA_PATH):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def atomic_write(path, obj):
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

def save_path_record(record, path=DATA_PATH):
    try:
        validate(instance=record, schema=SCHEMA)
    except ValidationError as e:
        raise ValueError(f"Invalid record: {e.message}")
    records = load_paths(path)
    records.append(record)
    atomic_write(path, records)
    return True
