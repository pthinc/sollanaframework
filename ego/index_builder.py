# index_builder.py
import sqlite3
from datetime import datetime

IDX = "data/paths_index.db"

def ensure_index():
    conn = sqlite3.connect(IDX)
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS paths (
        behavior_id TEXT,
        timestamp TEXT,
        path_score REAL,
        ethical_tag TEXT,
        PRIMARY KEY (behavior_id, timestamp)
      )
    """)
    conn.commit()
    conn.close()

def index_record(record):
    ensure_index()
    conn = sqlite3.connect(IDX)
    c = conn.cursor()
    params = record.get("parameters",{})
    eth = params.get("ethical_tag")
    c.execute("INSERT OR REPLACE INTO paths (behavior_id,timestamp,path_score,ethical_tag) VALUES (?,?,?,?)",
              (record["behavior_id"], record["timestamp"], record["path_score"], eth))
    conn.commit()
    conn.close()
