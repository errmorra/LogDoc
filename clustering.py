"""
clustering.py — Alert Clustering with K-Means
===============================================
Groups thousands of similar log lines into a small number of
meaningful "alert clusters" so an analyst sees patterns, not noise.

Feature vector per log row:
    • composite_risk_score
    • threat_score
    • asset_criticality
    • bytes_transferred (log-scaled)
    • is_off_hours (0/1)
    • is_high_risk_geo (0/1)
    • is_vip_user (0/1)
    • severity_num  (Low=1 … Critical=4)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SEVERITY_NUM = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# Cluster label map: assigned after inspection of centroid values
CLUSTER_NAMES = {
    0: "Routine Activity",
    1: "Suspicious External Traffic",
    2: "Insider / Privilege Anomaly",
    3: "Active Threat / Exfiltration",
    4: "Reconnaissance / Brute Force",
}


def cluster_alerts(
    df: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run K-Means clustering on *df* and add:
        cluster_id      int
        cluster_name    str
        cluster_summary str   (centroid-based description)

    Returns the augmented DataFrame.
    """
    if len(df) < n_clusters:
        logger.warning(
            "Dataset (%d rows) smaller than n_clusters (%d). Skipping clustering.",
            len(df), n_clusters,
        )
        df["cluster_id"]      = 0
        df["cluster_name"]    = "Unclustered"
        df["cluster_summary"] = "Dataset too small for clustering"
        return df

    features = _build_feature_matrix(df)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    df = df.copy()
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    # Map cluster IDs to human-readable names based on centroid risk profile
    df["cluster_name"] = df["cluster_id"].map(
        _name_clusters(kmeans.cluster_centers_, scaler)
    )

    # Per-cluster summary line
    cluster_summaries = _build_cluster_summaries(df)
    df["cluster_summary"] = df["cluster_id"].map(cluster_summaries)

    logger.info(
        "Clustering complete (%d clusters):\n%s",
        n_clusters,
        df["cluster_name"].value_counts().to_string(),
    )
    return df


def get_cluster_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per cluster with aggregate statistics:
    cluster_id, cluster_name, count, avg_risk_score, max_risk_score,
    top_action, top_asset, top_country.
    """
    if "cluster_id" not in df.columns:
        raise ValueError("Run cluster_alerts() before calling get_cluster_report().")

    rows = []
    for cid, grp in df.groupby("cluster_id"):
        rows.append({
            "cluster_id":      cid,
            "cluster_name":    grp["cluster_name"].iloc[0],
            "count":           len(grp),
            "avg_risk_score":  round(grp["composite_risk_score"].mean(), 1),
            "max_risk_score":  round(grp["composite_risk_score"].max(), 1),
            "top_action":      grp["action"].mode().iloc[0] if len(grp) else "-",
            "top_asset":       grp["asset_type"].mode().iloc[0] if len(grp) else "-",
            "top_country":     grp["country"].mode().iloc[0] if len(grp) else "-",
            "summary":         grp["cluster_summary"].iloc[0],
        })

    return (
        pd.DataFrame(rows)
        .sort_values("avg_risk_score", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feature_df = pd.DataFrame()
    feature_df["risk_score"]      = df["composite_risk_score"].fillna(0)
    feature_df["threat_score"]    = df.get("threat_score", pd.Series(0, index=df.index)).fillna(0)
    feature_df["asset_crit"]      = df.get("asset_criticality", pd.Series(3, index=df.index)).fillna(3)
    feature_df["bytes_log"]       = np.log1p(df.get("bytes_transferred", pd.Series(0, index=df.index)).fillna(0))
    feature_df["off_hours"]       = df.get("is_off_hours", pd.Series(False, index=df.index)).astype(int)
    feature_df["high_risk_geo"]   = df.get("is_high_risk_geo", pd.Series(False, index=df.index)).astype(int)
    feature_df["vip_user"]        = df.get("is_vip_user", pd.Series(False, index=df.index)).astype(int)
    feature_df["severity_num"]    = df.get("severity", pd.Series("Low", index=df.index)).map(SEVERITY_NUM).fillna(1)
    return feature_df.values


def _name_clusters(
    centers: np.ndarray,
    scaler: StandardScaler,
) -> dict:
    """
    Assign human-readable names to cluster IDs by ranking centroid risk profiles.
    Clusters are sorted descending by their un-scaled risk_score centroid.
    """
    centers_orig = scaler.inverse_transform(centers)
    risk_col = 0   # first feature = composite_risk_score
    order = np.argsort(centers_orig[:, risk_col])[::-1]

    labels = [
        "Active Threat / Exfiltration",
        "Suspicious External Traffic",
        "Insider / Privilege Anomaly",
        "Reconnaissance / Brute Force",
        "Routine Activity",
    ]
    # Pad/trim to actual n_clusters
    labels = (labels + ["Other Cluster"] * 10)[: len(order)]
    return {int(cluster_id): labels[rank] for rank, cluster_id in enumerate(order)}


def _build_cluster_summaries(df: pd.DataFrame) -> dict:
    summaries = {}
    for cid, grp in df.groupby("cluster_id"):
        name       = grp["cluster_name"].iloc[0]
        count      = len(grp)
        avg_score  = round(grp["composite_risk_score"].mean(), 1)
        top_action = grp["action"].mode().iloc[0] if count else "-"
        top_ip     = grp["source_ip"].mode().iloc[0] if count else "-"
        summaries[cid] = (
            f"{name}: {count} events, avg risk {avg_score}/100. "
            f"Dominant action: '{top_action}'. Lead IP: {top_ip}."
        )
    return summaries
