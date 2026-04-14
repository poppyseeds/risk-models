import json
import sqlite3
from pathlib import Path
from typing import Dict, List

from config.settings import settings


def _connect():
    Path(settings.history_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.history_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_store() -> None:
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            site_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            risk_score REAL NOT NULL,
            reason TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def record_incident(result: Dict) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO incidents(timestamp, site_id, asset_id, severity, risk_score, reason, payload)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.get("detected_at") or result["timestamp"],
            result["site_id"],
            result["asset_id"],
            result["severity"],
            result["fused_risk_score"],
            result["anomaly_reason"],
            json.dumps(result),
        ),
    )
    conn.commit()
    conn.close()


def latest_incidents(limit: int = 20) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT timestamp, site_id, asset_id, severity, risk_score, reason
        FROM incidents
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_incidents() -> int:
    """Remove all rows from the incident timeline. Returns number of rows deleted."""
    conn = _connect()
    before = int(conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0])
    conn.execute("DELETE FROM incidents")
    conn.commit()
    conn.close()
    return before
