"""
llm_summary.py — AI-Powered Alert Narrative Generation
========================================================
Generates human-readable "incident story" summaries for the top alerts
using either:
  • Anthropic Claude API  (set ANTHROPIC_API_KEY)
  • Local Ollama          (set OLLAMA_MODEL, e.g. "mistral")
  • Rule-based fallback   (always works, no API key required)
"""

import os
import json
import logging
import textwrap
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OLLAMA_HOST       = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "mistral")

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior SOC (Security Operations Center) analyst writing triage
    summaries for a SIEM dashboard. Given structured alert data, produce a
    concise (3-5 sentence) plain-English incident narrative that:
      1. States WHAT happened and WHO was involved.
      2. Explains WHY this is suspicious (reference the risk score and flags).
      3. Recommends an IMMEDIATE next step for the analyst.
    Be direct, factual, and avoid security jargon where plain language works.
    Do NOT include bullet points or headers — write flowing prose only.
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_summaries(
    df: pd.DataFrame,
    top_n: int = 10,
    backend: str = "auto",
) -> pd.DataFrame:
    """
    Generate AI narrative summaries for the *top_n* highest-scoring alerts.

    backend: "anthropic" | "ollama" | "rule_based" | "auto"
              "auto" tries Anthropic → Ollama → rule_based fallback.

    Adds column: ai_narrative  (str)
    """
    df = df.copy()
    df["ai_narrative"] = ""

    top_idx = (
        df.nlargest(top_n, "composite_risk_score").index
        if "composite_risk_score" in df.columns
        else df.head(top_n).index
    )

    resolver = _resolve_backend(backend)
    logger.info("Generating summaries for %d alerts using backend: %s", len(top_idx), resolver)

    for idx in top_idx:
        row = df.loc[idx]
        prompt = _build_prompt(row)
        try:
            narrative = _call_backend(prompt, resolver)
        except Exception as exc:
            logger.warning("LLM call failed for row %d: %s – using fallback.", idx, exc)
            narrative = _rule_based_narrative(row)
        df.at[idx, "ai_narrative"] = narrative

    # Fill remaining rows with rule-based summaries
    empty_mask = (df["ai_narrative"] == "") & (~df.index.isin(top_idx))
    if empty_mask.any():
        df.loc[empty_mask, "ai_narrative"] = df.loc[empty_mask].apply(
            _rule_based_narrative, axis=1
        )

    return df


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if _ollama_available():
        return "ollama"
    return "rule_based"


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _call_backend(prompt: str, backend: str) -> str:
    if backend == "anthropic":
        return _call_anthropic(prompt)
    elif backend == "ollama":
        return _call_ollama(prompt)
    return ""   # rule-based handled at call site


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      ANTHROPIC_MODEL,
            "max_tokens": 300,
            "system":     _SYSTEM_PROMPT,
            "messages":   [{"role": "user", "content": prompt}],
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


def _call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model":  OLLAMA_MODEL,
            "prompt": f"{_SYSTEM_PROMPT}\n\nAlert Data:\n{prompt}",
            "stream": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(row: pd.Series) -> str:
    ts   = pd.Timestamp(row.get("timestamp", "")).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row.get("timestamp")) else "Unknown time"
    return textwrap.dedent(f"""\
        Timestamp:      {ts}
        Source IP:      {row.get('source_ip', 'Unknown')}
        User:           {row.get('user', 'Unknown')}
        Action:         {row.get('action', 'Unknown')}
        Asset:          {row.get('asset_type', 'Unknown')}
        Country:        {row.get('country', 'Unknown')}
        Bytes Xferred:  {row.get('bytes_transferred', 0):,}
        Risk Score:     {row.get('composite_risk_score', 0)}/100
        Risk Label:     {row.get('risk_label', 'Unknown')}
        Threat Intel:   {row.get('threat_score', 0)}/100
        Off-Hours:      {row.get('is_off_hours', False)}
        VIP User:       {row.get('is_vip_user', False)}
        High-Risk Geo:  {row.get('is_high_risk_geo', False)}
        Matched Rules:  {', '.join(row.get('matched_rules', [])) or 'None'}
        Enrichment:     {row.get('enrichment_notes', '')}
        Cluster:        {row.get('cluster_name', 'Unknown')}
    """)


# ---------------------------------------------------------------------------
# Rule-based fallback narrative
# ---------------------------------------------------------------------------

def _rule_based_narrative(row: pd.Series) -> str:
    """Deterministic narrative template when no LLM is available."""
    action  = row.get("action", "Unknown activity")
    user    = row.get("user", "an unknown user")
    ip      = row.get("source_ip", "an unknown source")
    asset   = row.get("asset_type", "an asset")
    score   = row.get("composite_risk_score", 0)
    label   = row.get("risk_label", "Low")
    rules   = row.get("matched_rules", [])
    country = row.get("country", "unknown location")

    parts = [
        f"{action} was detected from {ip} ({country}) by user '{user}' "
        f"against a {asset}, receiving a {label} risk score of {score}/100."
    ]

    if rules:
        parts.append(f"Alert triggered by: {'; '.join(rules)}.")

    flags = []
    if row.get("is_off_hours"):
        flags.append("off-hours access")
    if row.get("is_vip_user"):
        flags.append("VIP account involvement")
    if row.get("is_high_risk_geo"):
        flags.append(f"high-risk geography ({country})")
    if row.get("threat_score", 0) >= 70:
        flags.append("high threat intelligence score")

    if flags:
        parts.append(f"Contributing factors include {', '.join(flags)}.")

    if score >= 70:
        parts.append("Recommend immediate investigation and potential account isolation.")
    elif score >= 40:
        parts.append("Recommend analyst review within the next 2 hours.")
    else:
        parts.append("Low priority — monitor for escalation.")

    return " ".join(parts)
