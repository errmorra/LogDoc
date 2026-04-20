"""
ingestion.py — Log Ingestion & Normalization
==============================================
Reads raw logs (CSV, JSON, Syslog) and normalises them into a
standard schema with typed fields ready for enrichment & scoring.
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical schema columns every downstream module expects
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "timestamp",
    "source_ip",
    "destination_ip",
    "user",
    "event_id",
    "action",
    "severity",
    "asset_type",
    "bytes_transferred",
    "country",
    "port",
    "log_type",       # injected during normalization
    "raw_line",       # original line preserved for audit
]

# ---------------------------------------------------------------------------
# Log-type classifier patterns (applied in order, first match wins)
# ---------------------------------------------------------------------------
LOG_TYPE_PATTERNS = [
    (re.compile(r"login|logon|authentication|4624|4625", re.I), "Authentication"),
    (re.compile(r"firewall|policy|5156|5157|4719",        re.I), "Firewall"),
    (re.compile(r"sql|database|query|1433",               re.I), "Database"),
    (re.compile(r"file.access|4663|4656",                 re.I), "FileAccess"),
    (re.compile(r"process|4688|4689",                     re.I), "Process"),
    (re.compile(r"group|privilege|4732|1102",             re.I), "Privilege"),
]

SEVERITY_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_logs(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load logs from *path* (auto-detects CSV / JSON / JSONL / .log syslog).
    Returns a DataFrame normalised to CANONICAL_COLUMNS.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = _load_csv(path)
    elif suffix in (".json", ".jsonl"):
        df = _load_json(path)
    elif suffix in (".log", ".txt", ""):
        df = _load_syslog(path)
    else:
        # Try CSV as a fallback
        logger.warning("Unknown extension %s – attempting CSV parse.", suffix)
        df = _load_csv(path)

    df = _normalise(df)
    logger.info("Loaded %d log entries from %s", len(df), path)
    return df


def load_logs_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Accept an already-loaded DataFrame and apply normalisation."""
    return _normalise(df.copy())


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _load_json(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    df = pd.DataFrame(records)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _load_syslog(path: Path) -> pd.DataFrame:
    """
    Very simple syslog parser.  Extracts timestamp, host, and message.
    The NLP classifier will figure out the log type from the message text.
    """
    # Example: Jan 15 08:23:11 hostname sshd[1234]: message
    pattern = re.compile(
        r"(?P<ts>\w{3}\s+\d+\s+[\d:]+)\s+(?P<host>\S+)\s+(?P<proc>\S+):\s+(?P<msg>.*)"
    )
    records = []
    with open(path) as fh:
        for line in fh:
            m = pattern.match(line.strip())
            if m:
                records.append({
                    "timestamp":        m.group("ts"),
                    "source_ip":        m.group("host"),
                    "destination_ip":   "",
                    "user":             "",
                    "event_id":         m.group("proc"),
                    "action":           m.group("msg")[:80],
                    "severity":         "Low",
                    "asset_type":       "Unknown",
                    "bytes_transferred": 0,
                    "country":          "Unknown",
                    "port":             0,
                    "raw_line":         line.strip(),
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column aliasing, type coercion, and log-type classification."""

    # ── Column aliases ───────────────────────────────────────────────────────
    aliases = {
        "src_ip": "source_ip", "src": "source_ip",
        "dst_ip": "destination_ip", "dst": "destination_ip",
        "username": "user", "account": "user",
        "event": "event_id", "id": "event_id",
        "bytes": "bytes_transferred", "size": "bytes_transferred",
        "geo": "country", "location": "country",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    # ── Ensure all canonical columns exist ───────────────────────────────────
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None if col in ("source_ip", "user", "country") else 0

    # ── Type coercion ────────────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["bytes_transferred"] = pd.to_numeric(df["bytes_transferred"], errors="coerce").fillna(0).astype(int)
    df["port"] = pd.to_numeric(df["port"], errors="coerce").fillna(0).astype(int)
    df["event_id"] = df["event_id"].astype(str).str.strip()

    # Normalise severity to known labels
    df["severity"] = df["severity"].str.strip().str.title()
    df["severity"] = df["severity"].where(
        df["severity"].isin(SEVERITY_ORDER.keys()), other="Low"
    )

    # ── Preserve raw line ────────────────────────────────────────────────────
    if "raw_line" not in df.columns:
        df["raw_line"] = df.apply(lambda r: r.to_json(), axis=1)

    # ── Log-type classification ──────────────────────────────────────────────
    df["log_type"] = df.apply(_classify_log_type, axis=1)

    # ── Drop rows with no usable timestamp ──────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    if len(df) < before:
        logger.warning("Dropped %d rows with unparseable timestamps.", before - len(df))

    return df.reset_index(drop=True)


def _classify_log_type(row: pd.Series) -> str:
    """
    Use the action text + event_id to determine the log category.
    Falls back to 'Other'.
    """
    haystack = " ".join(str(row.get(f, "")) for f in ("action", "event_id", "severity"))
    for pattern, label in LOG_TYPE_PATTERNS:
        if pattern.search(haystack):
            return label
    return "Other"
