"""
pipeline.py — Orchestration Pipeline
======================================
Wires ingestion → enrichment → scoring → clustering → LLM summary
into a single callable that returns a fully-analysed DataFrame.
"""

import logging
import time
from pathlib import Path
from typing import Union, Optional

import pandas as pd

from .ingestion   import load_logs, load_logs_from_dataframe
from .enrichment  import enrich
from .scoring     import score, get_top_alerts
from .clustering  import cluster_alerts, get_cluster_report
from .llm_summary import generate_summaries

logger = logging.getLogger(__name__)


def run_pipeline(
    source: Union[str, Path, pd.DataFrame],
    n_clusters:   int  = 5,
    top_n_alerts: int  = 10,
    llm_backend:  str  = "auto",
    use_live_api: bool = False,
) -> dict:
    """
    Execute the full triage pipeline.

    Parameters
    ----------
    source        : Path to a log file (CSV/JSON/syslog) or a raw DataFrame.
    n_clusters    : Number of K-Means clusters.
    top_n_alerts  : Number of top alerts to generate LLM narratives for.
    llm_backend   : "auto" | "anthropic" | "ollama" | "rule_based"
    use_live_api  : If True, call real threat-intel APIs (requires API keys).

    Returns
    -------
    dict with keys:
        df              : fully enriched & scored DataFrame
        top_alerts      : top N alerts DataFrame
        cluster_report  : cluster summary DataFrame
        stats           : pipeline run statistics
    """
    t0 = time.perf_counter()
    steps = {}

    # ── Step 1: Ingest ────────────────────────────────────────────────────
    t = time.perf_counter()
    if isinstance(source, pd.DataFrame):
        df = load_logs_from_dataframe(source)
    else:
        df = load_logs(source)
    steps["ingestion_sec"] = round(time.perf_counter() - t, 3)
    logger.info("[1/5] Ingestion: %d rows loaded in %.3fs", len(df), steps["ingestion_sec"])

    # ── Step 2: Enrich ───────────────────────────────────────────────────
    t = time.perf_counter()
    df = enrich(df, use_live_api=use_live_api)
    steps["enrichment_sec"] = round(time.perf_counter() - t, 3)
    logger.info("[2/5] Enrichment complete in %.3fs", steps["enrichment_sec"])

    # ── Step 3: Score ────────────────────────────────────────────────────
    t = time.perf_counter()
    df = score(df)
    steps["scoring_sec"] = round(time.perf_counter() - t, 3)
    logger.info("[3/5] Scoring complete in %.3fs", steps["scoring_sec"])

    # ── Step 4: Cluster ──────────────────────────────────────────────────
    t = time.perf_counter()
    df = cluster_alerts(df, n_clusters=n_clusters)
    cluster_report = get_cluster_report(df)
    steps["clustering_sec"] = round(time.perf_counter() - t, 3)
    logger.info("[4/5] Clustering complete in %.3fs", steps["clustering_sec"])

    # ── Step 5: LLM Summaries ────────────────────────────────────────────
    t = time.perf_counter()
    df = generate_summaries(df, top_n=top_n_alerts, backend=llm_backend)
    steps["llm_sec"] = round(time.perf_counter() - t, 3)
    logger.info("[5/5] LLM summaries generated in %.3fs", steps["llm_sec"])

    top_alerts = get_top_alerts(df, n=top_n_alerts)

    total_sec = round(time.perf_counter() - t0, 3)
    stats = {
        "total_logs":           len(df),
        "critical_count":       int((df["risk_label"] == "Critical").sum()),
        "high_count":           int((df["risk_label"] == "High").sum()),
        "medium_count":         int((df["risk_label"] == "Medium").sum()),
        "low_count":            int((df["risk_label"] == "Low").sum()),
        "unique_source_ips":    int(df["source_ip"].nunique()),
        "total_bytes_gb":       round(df["bytes_transferred"].sum() / 1024**3, 2),
        "pipeline_total_sec":   total_sec,
        **steps,
    }

    logger.info("Pipeline complete in %.3fs. %d critical alerts.", total_sec, stats["critical_count"])
    return {
        "df":             df,
        "top_alerts":     top_alerts,
        "cluster_report": cluster_report,
        "stats":          stats,
    }
