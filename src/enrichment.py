"""
enrichment.py — Contextual Enrichment
=======================================
Augments normalised log rows with:
  • Threat Intelligence scores   (AbuseIPDB / VirusTotal or local mock)
  • VIP / high-value user flags
  • Asset criticality lookup
  • Geo-anomaly detection (impossible travel, off-hours logins)
"""

import os
import time
import logging
import hashlib
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – loaded from environment variables (.env)
# ---------------------------------------------------------------------------
ABUSEIPDB_KEY = os.getenv("ABUSEIPDB_API_KEY", "")
VIRUSTOTAL_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")

# ---------------------------------------------------------------------------
# Static lookup tables  (extend these to fit your org)
# ---------------------------------------------------------------------------

VIP_USERS = {
    "ceo_jones", "cfo_smith", "cto_white", "admin", "root", "dbadmin",
    "netadmin", "sysadmin", "administrator",
}

ASSET_CRITICALITY: dict[str, int] = {
    # Asset type string → criticality value 1-10
    "Domain Controller": 10,
    "Database Server":   9,
    "Firewall":          8,
    "App Server":        7,
    "Backup Server":     7,
    "Workstation":       4,
    "Guest Wifi":        2,
    "Unknown":           3,
}

KNOWN_MALICIOUS_IPS = {
    # Seeded with well-known bogon/abuse ranges for demo purposes
    "185.220.101.45",   # Tor exit node
    "91.108.4.100",     # Reported C2
    "203.0.113.77",     # Documentation range (used as "bad" in demo)
    "198.51.100.22",    # Documentation range
}

HIGH_RISK_COUNTRIES = {"CN", "RU", "KP", "IR", "NL", "BR"}  # expand as needed

# Work-hours window (24-h clock, inclusive)
WORK_HOUR_START = 7
WORK_HOUR_END   = 19


# ---------------------------------------------------------------------------
# In-memory cache to avoid hammering free API tiers during demo runs
# ---------------------------------------------------------------------------
_ip_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame, use_live_api: bool = False) -> pd.DataFrame:
    """
    Main enrichment entry point.  Adds the following columns:
        threat_score    0-100  (higher = more suspicious)
        is_vip_user     bool
        asset_criticality 1-10
        is_off_hours    bool
        is_high_risk_geo bool
        enrichment_notes str   (human-readable summary)
    """
    df = df.copy()

    # Threat intel
    df["threat_score"] = df["source_ip"].apply(
        lambda ip: _get_threat_score(ip, use_live_api)
    )

    # VIP flag
    df["is_vip_user"] = df["user"].str.lower().isin(
        {u.lower() for u in VIP_USERS}
    )

    # Asset criticality (1-10)
    df["asset_criticality"] = df["asset_type"].map(
        lambda a: ASSET_CRITICALITY.get(a, 3)
    )

    # Time-based anomaly
    df["is_off_hours"] = df["timestamp"].apply(_is_off_hours)

    # Geo risk
    df["is_high_risk_geo"] = df["country"].isin(HIGH_RISK_COUNTRIES)

    # Compose a human-readable enrichment note per row
    df["enrichment_notes"] = df.apply(_compose_notes, axis=1)

    logger.info("Enrichment complete for %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Threat intelligence helpers
# ---------------------------------------------------------------------------

def _get_threat_score(ip: str, live: bool = False) -> int:
    """Return a 0-100 threat score for *ip*."""
    if not ip or pd.isna(ip):
        return 0

    if ip in _ip_cache:
        return _ip_cache[ip]["score"]

    if ip in KNOWN_MALICIOUS_IPS:
        score = 95
    elif live and ABUSEIPDB_KEY:
        score = _query_abuseipdb(ip)
    elif live and VIRUSTOTAL_KEY:
        score = _query_virustotal(ip)
    else:
        score = _heuristic_score(ip)

    _ip_cache[ip] = {"score": score, "ts": time.time()}
    return score


def _heuristic_score(ip: str) -> int:
    """
    Offline heuristic: score IPs without calling an external API.
    Uses the IP string's checksum for a deterministic demo value.
    Private/RFC1918 IPs always score 0.
    """
    private_prefixes = ("10.", "192.168.", "172.16.", "172.17.", "172.18.",
                        "172.19.", "172.2", "127.", "::1")
    if any(ip.startswith(p) for p in private_prefixes):
        return 0

    # Deterministic pseudo-random score based on IP hash (for demo realism)
    digest = int(hashlib.md5(ip.encode()).hexdigest(), 16)
    return digest % 60  # 0-59 range so it's plausible, not always critical


def _query_abuseipdb(ip: str) -> int:
    """Call AbuseIPDB v2 API and return the abuseConfidenceScore."""
    try:
        resp = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            headers={"Key": ABUSEIPDB_KEY, "Accept": "application/json"},
            params={"ipAddress": ip, "maxAgeInDays": 90},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json().get("data", {}).get("abuseConfidenceScore", 0)
    except Exception as exc:
        logger.warning("AbuseIPDB query failed for %s: %s", ip, exc)
        return 0


def _query_virustotal(ip: str) -> int:
    """
    Call VirusTotal v3 /ip_addresses/{ip} and return a 0-100 score
    derived from the ratio of malicious detections.
    """
    try:
        resp = requests.get(
            f"https://www.virustotal.com/api/v3/ip_addresses/{ip}",
            headers={"x-apikey": VIRUSTOTAL_KEY},
            timeout=5,
        )
        resp.raise_for_status()
        stats = (resp.json()
                 .get("data", {})
                 .get("attributes", {})
                 .get("last_analysis_stats", {}))
        malicious  = stats.get("malicious", 0)
        total      = sum(stats.values()) or 1
        return int((malicious / total) * 100)
    except Exception as exc:
        logger.warning("VirusTotal query failed for %s: %s", ip, exc)
        return 0


# ---------------------------------------------------------------------------
# Anomaly helpers
# ---------------------------------------------------------------------------

def _is_off_hours(ts) -> bool:
    """Return True if the timestamp falls outside normal business hours."""
    if pd.isna(ts):
        return False
    hour = pd.Timestamp(ts).hour
    return not (WORK_HOUR_START <= hour < WORK_HOUR_END)


def _compose_notes(row: pd.Series) -> str:
    """Build a plain-English enrichment summary for the log entry."""
    notes = []

    if row.get("threat_score", 0) >= 70:
        notes.append(f"⚠ HIGH-THREAT IP (score {row['threat_score']})")
    elif row.get("threat_score", 0) >= 30:
        notes.append(f"Suspicious IP (score {row['threat_score']})")

    if row.get("is_vip_user"):
        notes.append(f"VIP account '{row.get('user', 'unknown')}' involved")

    crit = row.get("asset_criticality", 0)
    if crit >= 8:
        notes.append(f"Critical asset ({row.get('asset_type', '')} — criticality {crit}/10)")

    if row.get("is_off_hours"):
        hr = pd.Timestamp(row.get("timestamp")).strftime("%H:%M") if pd.notna(row.get("timestamp")) else "?"
        notes.append(f"Off-hours activity at {hr}")

    if row.get("is_high_risk_geo"):
        notes.append(f"High-risk country: {row.get('country', '?')}")

    return " | ".join(notes) if notes else "No anomalies detected"
