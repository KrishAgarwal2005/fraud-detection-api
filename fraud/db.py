"""Storage for predictions, using SQLite and plain SQL.

Every call opens a short-lived connection, runs its SQL and closes it.
That is simple and safe for a small service.
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    amount            REAL    NOT NULL,
    hour              INTEGER NOT NULL,
    fraud_probability REAL    NOT NULL,
    is_fraud          INTEGER NOT NULL,   -- 1 = flagged, 0 = not flagged
    threshold         REAL    NOT NULL,
    model_version     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dictionaries
    return conn


def init_db(db_path: str) -> None:
    with closing(connect(db_path)) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def save_prediction(db_path: str, amount: float, hour: int, probability: float,
                    is_fraud: bool, threshold: float, model_version: str) -> int:
    """Insert one prediction and return its new id."""
    with closing(connect(db_path)) as conn:
        cur = conn.execute(
            "INSERT INTO predictions (created_at, amount, hour, fraud_probability, "
            "is_fraud, threshold, model_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), amount, hour, probability,
             int(is_fraud), threshold, model_version),
        )
        conn.commit()
        return cur.lastrowid


def get_prediction(db_path: str, prediction_id: int):
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        return dict(row) if row else None


def recent_predictions(db_path: str, limit: int = 20, only_fraud: bool = False) -> list:
    sql = "SELECT * FROM predictions"
    if only_fraud:
        sql += " WHERE is_fraud = 1"
    sql += " ORDER BY id DESC LIMIT ?"
    with closing(connect(db_path)) as conn:
        return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]


def stats(db_path: str) -> dict:
    """Totals for monitoring: how many requests, how many flagged."""
    with closing(connect(db_path)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(is_fraud), 0) AS flagged, "
            "AVG(fraud_probability) AS avg_probability "
            "FROM predictions"
        ).fetchone()
    total, flagged = row["total"], row["flagged"]
    return {
        "total_predictions": total,
        "flagged_as_fraud": flagged,
        "flag_rate": round(flagged / total, 4) if total else 0.0,
        "average_probability": round(row["avg_probability"], 4) if total else 0.0,
    }
