"""
scoring.py — Composite Risk Scoring Engine
===========================================
Calculates a 0-100 Composite Risk Score for each log entry using:

    Risk Score = (Severity × Asset Value)   [40%]
               + Threat Intelligence Factor [30%]
               + Behavioural Anomaly Score  [20%]
               + Rule Match Score           [10%]

Also computes Z-Score connection frequency anomalies per source IP.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight table (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "asset_severity":   0.40,
    "threat_intel":     0.30,
    "behavioural":      0.20,
    "rule_match":       0.10,
}

# ---------------------------------------------------------------------------
# Severity → numeric (0-100)
# ---------------------------------------------------------------------------
SEVERITY_SCORES = {
    "Low":      10,
    "Medium":   40,
    "High":     75,
    "Critical": 100,
}

# ---------------------------------------------------------------------------
# Rule signatures for known attack patterns
# ---------------------------------------------------------------------------
ATTACK_RULES = [
    # (name, lambda row -> bool)
    ("Brute Force (≥5 failed logins from same IP)",
     lambda grp: False),   # evaluated separately at the group level

    ("Log Cleared",
     lambda r: "1102" in str(r.get("event_id", "")) or
               "log cleared" in str(r.get("action", "")).lower()),

    ("Off-Hours Critical Asset Access",
     lambda r: r.get("is_off_hours", False) and r.get("asset_criticality", 0) >= 8),

    ("Massive Data Exfiltration (>500MB)",
     lambda r: r.get("bytes_transferred", 0) > 500 * 1024 * 1024),

    ("Privilege Escalation Indicator",
     lambda r: any(kw in str(r.get("action", "")).lower()
                   for kw in ("group change", "privilege", "escalat"))),

    ("Firewall Policy Modified",
     lambda r: "4719" in str(r.get("event_id", "")) or
               "policy change" in str(r.get("action", "")).lower()),

    ("VIP Account from High-Risk Geo",
     lambda r: r.get("is_vip_user", False) and r.get("is_high_risk_geo", False)),

    ("Root/Admin Login from External IP",
     lambda r: r.get("user", "").lower() in ("root", "admin", "administrator") and
               not str(r.get("source_ip", "")).startswith(("10.", "192.168.", "172."))),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main scoring entry point.  Requires enrichment columns to be present.
    Adds:
        component_asset_severity   0-100
        component_threat_intel     0-100
        component_behavioural      0-100
        component_rule_match       0-100
        composite_risk_score       0-100  (weighted sum of components)
        risk_label                 Low / Medium / High / Critical
        matched_rules              list[str]
        zscore_ip_frequency        float  (std-devs above mean connection count)
    """
    df = df.copy()

    # Component 1: Asset × Severity
    df["component_asset_severity"] = df.apply(_asset_severity_component, axis=1)

    # Component 2: Threat Intel (direct from enrichment)
    df["component_threat_intel"] = df["threat_score"].clip(0, 100)

    # Component 3: Behavioural anomaly
    df["component_behavioural"] = df.apply(_behavioural_component, axis=1)

    # Component 4: Rule match
    df["matched_rules"], df["component_rule_match"] = zip(
        *df.apply(_rule_match_component, axis=1)
    )

    # Brute-force rule applied at group level
    df = _apply_bruteforce_rule(df)

    # Composite score
    df["composite_risk_score"] = (
        df["component_asset_severity"] * WEIGHTS["asset_severity"]
        + df["component_threat_intel"] * WEIGHTS["threat_intel"]
        + df["component_behavioural"] * WEIGHTS["behavioural"]
        + df["component_rule_match"]   * WEIGHTS["rule_match"]
    ).round(1).clip(0, 100)

    # Human-readable risk label
    df["risk_label"] = df["composite_risk_score"].apply(_score_to_label)

    # Z-Score: how unusual is this IP's connection frequency?
    df = _add_ip_zscore(df)

    logger.info(
        "Scoring complete. Score distribution:\n%s",
        df["risk_label"].value_counts().to_string()
    )
    return df


# ---------------------------------------------------------------------------
# Component calculators
# ---------------------------------------------------------------------------

def _asset_severity_component(row: pd.Series) -> float:
    """
    Combines event severity (0-100) with asset criticality (1-10, scaled to 0-100).
    Formula: (severity_score + asset_scaled) / 2
    """
    sev   = SEVERITY_SCORES.get(row.get("severity", "Low"), 10)
    asset = (row.get("asset_criticality", 3) / 10) * 100
    return (sev + asset) / 2


def _behavioural_component(row: pd.Series) -> float:
    """
    Combines multiple soft signals into a 0-100 behavioural anomaly score.
    """
    score = 0.0

    # Off-hours login
    if row.get("is_off_hours", False):
        score += 40

    # High-risk geography
    if row.get("is_high_risk_geo", False):
        score += 30

    # VIP user involved
    if row.get("is_vip_user", False):
        score += 20

    # Large data transfer (>100 MB)
    mb = row.get("bytes_transferred", 0) / (1024 ** 2)
    if mb > 1000:
        score += 30
    elif mb > 100:
        score += 15

    return min(score, 100)


def _rule_match_component(row: pd.Series):
    """
    Returns (matched_rule_names: list, score: float).
    Each rule match adds 25 points, capped at 100.
    """
    matched = []
    for name, fn in ATTACK_RULES[1:]:   # skip brute-force (handled separately)
        try:
            if fn(row):
                matched.append(name)
        except Exception:
            pass
    score = min(len(matched) * 25, 100)
    return matched, float(score)


def _apply_bruteforce_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag source IPs with ≥5 failed login events within the dataset.
    Adds them to matched_rules and bumps component_rule_match.
    """
    brute_label = "Brute Force (≥5 failed logins from same IP)"
    failed_mask = df["action"].str.lower().str.contains("failed login", na=False)
    counts = df[failed_mask].groupby("source_ip").size()
    brute_ips = set(counts[counts >= 5].index)

    def _update(row):
        rules = list(row["matched_rules"])
        score = row["component_rule_match"]
        if row["source_ip"] in brute_ips and brute_label not in rules:
            rules.append(brute_label)
            score = min(score + 25, 100)
        return rules, score

    if brute_ips:
        results = df.apply(_update, axis=1)
        df["matched_rules"]       = results.apply(lambda x: x[0])
        df["component_rule_match"] = results.apply(lambda x: x[1])

    return df


def _add_ip_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Z-Score of connection frequency per source IP.
    IPs with Z > 2 are statistically unusual and receive a score boost.
    """
    counts = df.groupby("source_ip").size().rename("ip_count")
    df = df.join(counts, on="source_ip")

    mean, std = df["ip_count"].mean(), df["ip_count"].std()
    if std == 0:
        df["zscore_ip_frequency"] = 0.0
    else:
        df["zscore_ip_frequency"] = ((df["ip_count"] - mean) / std).round(2)

    # Boost composite score for statistical outliers (Z > 2)
    boost_mask = df["zscore_ip_frequency"] > 2
    df.loc[boost_mask, "composite_risk_score"] = (
        df.loc[boost_mask, "composite_risk_score"] + 10
    ).clip(0, 100)

    df = df.drop(columns=["ip_count"])
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_label(score: float) -> str:
    if score >= 80:
        return "Critical"
    elif score >= 55:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"


def get_top_alerts(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the *n* highest-scoring, deduplicated alert rows."""
    return (
        df.sort_values("composite_risk_score", ascending=False)
        .drop_duplicates(subset=["source_ip", "action", "risk_label"])
        .head(n)
        .reset_index(drop=True)
    )
