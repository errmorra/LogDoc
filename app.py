"""
app.py — AI-Driven Log Triage Dashboard (Streamlit)
=====================================================
Run with:
    streamlit run app.py
"""

import io
import sys
import os
import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# Add src/ to path when running directly
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)
from src.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Log Triage",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark cyber aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;700&display=swap');

  :root {
    --bg:        #0a0e1a;
    --panel:     #111827;
    --border:    #1f2d45;
    --accent:    #00d4ff;
    --warn:      #f59e0b;
    --danger:    #ef4444;
    --safe:      #10b981;
    --text:      #c9d6e3;
    --muted:     #64748b;
  }

  html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    color: var(--text);
    background: var(--bg);
  }

  .stApp { background: var(--bg); }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
  }

  /* Headers */
  h1, h2, h3 { font-family: 'Share Tech Mono', monospace; color: var(--accent); }

  /* Alert card */
  .alert-card {
    background: var(--panel);
    border-left: 4px solid var(--danger);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
  }
  .alert-card.high   { border-color: var(--warn); }
  .alert-card.medium { border-color: #8b5cf6; }
  .alert-card.low    { border-color: var(--muted); }

  .score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
  }
  .badge-critical { background: #7f1d1d; color: #fca5a5; }
  .badge-high     { background: #78350f; color: #fde68a; }
  .badge-medium   { background: #2e1065; color: #ddd6fe; }
  .badge-low      { background: #052e16; color: #6ee7b7; }

  /* Dataframe tweaks */
  .stDataFrame thead th {
    background: var(--border) !important;
    color: var(--accent) !important;
    font-family: 'Share Tech Mono', monospace;
  }

  /* Button */
  .stButton > button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.05em;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: var(--accent);
    color: var(--bg);
  }

  /* Tab styling */
  [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border); }
  [data-baseweb="tab"]      { color: var(--muted) !important; }
  [aria-selected="true"]    { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

RISK_COLORS = {
    "Critical": "#ef4444",
    "High":     "#f59e0b",
    "Medium":   "#8b5cf6",
    "Low":      "#10b981",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d6e3", family="Exo 2"),
        xaxis=dict(gridcolor="#1f2d45", linecolor="#1f2d45"),
        yaxis=dict(gridcolor="#1f2d45", linecolor="#1f2d45"),
    )
)

def _badge(label: str) -> str:
    cls = f"badge-{label.lower()}"
    return f'<span class="score-badge {cls}">{label}</span>'

def _alert_card(row: pd.Series) -> str:
    rl = row.get("risk_label", "Low").lower()
    score = row.get("composite_risk_score", 0)
    narrative = row.get("ai_narrative", "No narrative generated.")
    rules = ", ".join(row.get("matched_rules", [])) or "None"
    return f"""
    <div class="alert-card {rl}">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <span style="font-family:'Share Tech Mono',monospace; font-size:0.9rem;">
          🖥 {row.get('source_ip','?')} → {row.get('asset_type','?')}
        </span>
        {_badge(row.get('risk_label','Low'))}
        <span style="font-family:'Share Tech Mono',monospace; font-size:1.1rem; color:#00d4ff;">
          {score}/100
        </span>
      </div>
      <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:4px;">
        {row.get('action','?')} | User: <b>{row.get('user','?')}</b> | {row.get('country','?')} |
        {pd.Timestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('timestamp')) else '?'}
      </div>
      <div style="font-size:0.78rem; color:#64748b; margin-bottom:6px;">
        Rules: {rules}
      </div>
      <div style="font-size:0.85rem; color:#c9d6e3; font-style:italic; line-height:1.5;">
        {narrative}
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🛡️ Log Triage AI")
    st.markdown("---")

    st.markdown("### Data Source")
    mode = st.radio("Input mode", ["Use sample data", "Upload log file"])

    uploaded = None
    if mode == "Upload log file":
        uploaded = st.file_uploader("Upload CSV / JSON / syslog", type=["csv","json","log","txt"])

    st.markdown("---")
    st.markdown("### Pipeline Options")
    n_clusters   = st.slider("K-Means clusters",     2, 10, 5)
    top_n_alerts = st.slider("Top alerts to analyse", 3, 20, 10)
    llm_backend  = st.selectbox("AI backend", ["auto", "rule_based", "anthropic", "ollama"])
    use_live_api = st.checkbox("Use live threat intel APIs", value=False)

    run_btn = st.button("▶  Run Triage Pipeline", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.75rem;color:#475569;'>"
        "AI-Driven Log Triage v1.0<br>"
        "Built with Scikit-learn · Plotly · Streamlit"
        "</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown("## 🛡️ AI-Driven Log Triage Dashboard")

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_btn:
    if mode == "Upload log file" and uploaded is None:
        st.error("Please upload a log file first.")
    else:
        with st.spinner("Running triage pipeline…"):
            if mode == "Use sample data":
                source = Path("data/sample_logs.csv")
            else:
                # Save upload to a temp path
                suffix = Path(uploaded.name).suffix
                tmp    = Path(f"/tmp/triage_upload{suffix}")
                tmp.write_bytes(uploaded.read())
                source = tmp

            try:
                results = run_pipeline(
                    source,
                    n_clusters=n_clusters,
                    top_n_alerts=top_n_alerts,
                    llm_backend=llm_backend,
                    use_live_api=use_live_api,
                )
                st.session_state["results"] = results
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.exception(exc)

# ----- Render results -------------------------------------------------------
results = st.session_state.get("results")

if results is None:
    st.info("👈 Configure the pipeline in the sidebar and click **Run Triage Pipeline** to begin.")
    st.markdown("""
    **What this tool does:**
    - 📥 Ingests logs (CSV, JSON, Syslog) and normalises them
    - 🌐 Enriches each event with threat intel, asset criticality & geo-risk
    - 📊 Computes a 0–100 Composite Risk Score per event
    - 🤖 Clusters similar events with K-Means (noise → signal)
    - ✍️ Generates plain-English incident narratives with AI
    """)
    st.stop()

df             = results["df"]
top_alerts     = results["top_alerts"]
cluster_report = results["cluster_report"]
stats          = results["stats"]

# ── KPI row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Logs",     stats["total_logs"])
c2.metric("🔴 Critical",    stats["critical_count"])
c3.metric("🟠 High",        stats["high_count"])
c4.metric("🟣 Medium",      stats["medium_count"])
c5.metric("Unique IPs",     stats["unique_source_ips"])
c6.metric("Data (GB)",      stats["total_bytes_gb"])

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🚨 Top Alerts", "📊 Analytics", "🔵 Clusters", "📋 Raw Data"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Top Alerts
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"### Top {len(top_alerts)} Alerts by Risk Score")
    for _, row in top_alerts.iterrows():
        st.markdown(_alert_card(row), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns(2)

    # Risk label distribution donut
    with col_l:
        st.markdown("#### Risk Label Distribution")
        label_counts = df["risk_label"].value_counts().reset_index()
        label_counts.columns = ["Risk Label", "Count"]
        fig = px.pie(
            label_counts, names="Risk Label", values="Count",
            color="Risk Label",
            color_discrete_map=RISK_COLORS,
            hole=0.55,
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

    # Top IPs by risk score bar chart
    with col_r:
        st.markdown("#### Top 10 Source IPs by Max Risk Score")
        top_ips = (
            df.groupby("source_ip")["composite_risk_score"]
            .max()
            .nlargest(10)
            .reset_index()
        )
        top_ips.columns = ["Source IP", "Max Risk Score"]
        fig2 = px.bar(
            top_ips, x="Max Risk Score", y="Source IP",
            orientation="h",
            color="Max Risk Score",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            range_color=[0, 100],
        )
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Risk score over time
    st.markdown("#### Risk Score Timeline")
    ts_df = df.dropna(subset=["timestamp"]).copy()
    ts_df = ts_df.sort_values("timestamp")
    fig3 = px.scatter(
        ts_df,
        x="timestamp",
        y="composite_risk_score",
        color="risk_label",
        color_discrete_map=RISK_COLORS,
        hover_data=["source_ip", "user", "action", "asset_type"],
        size="composite_risk_score",
        size_max=20,
    )
    fig3.update_layout(**PLOTLY_TEMPLATE["layout"])
    st.plotly_chart(fig3, use_container_width=True)

    # Scoring components heatmap
    st.markdown("#### Risk Component Breakdown (Top 15 Events)")
    comp_cols = [
        "source_ip", "action",
        "component_asset_severity",
        "component_threat_intel",
        "component_behavioural",
        "component_rule_match",
        "composite_risk_score",
    ]
    comp_df = (
        df.nlargest(15, "composite_risk_score")[comp_cols]
        .set_index("source_ip")
        .round(1)
    )
    fig4 = go.Figure(
        go.Heatmap(
            z=comp_df[["component_asset_severity","component_threat_intel",
                        "component_behavioural","component_rule_match"]].values,
            x=["Asset×Severity (40%)", "Threat Intel (30%)", "Behavioural (20%)", "Rule Match (10%)"],
            y=comp_df.index,
            colorscale="RdYlGn",
            reversescale=True,
            zmin=0, zmax=100,
        )
    )
    fig4.update_layout(**PLOTLY_TEMPLATE["layout"], height=400)
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Clusters
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### K-Means Cluster Report")
    for _, cr in cluster_report.iterrows():
        color = RISK_COLORS.get(
            "Critical" if cr["avg_risk_score"] >= 70 else
            "High"     if cr["avg_risk_score"] >= 45 else
            "Medium"   if cr["avg_risk_score"] >= 25 else "Low",
            "#64748b"
        )
        st.markdown(
            f"""<div style="background:#111827;border-left:4px solid {color};
            border-radius:6px;padding:0.9rem 1.1rem;margin-bottom:0.7rem;">
            <b style="color:{color};">{cr['cluster_name']}</b>
            &nbsp;&nbsp;<span style="color:#94a3b8;font-size:0.8rem;">
            {cr['count']} events | avg risk {cr['avg_risk_score']}/100 | max {cr['max_risk_score']}/100
            </span><br>
            <span style="font-size:0.82rem;color:#64748b;">
            Top action: {cr['top_action']} | Asset: {cr['top_asset']} | Country: {cr['top_country']}
            </span><br>
            <span style="font-size:0.8rem;color:#94a3b8;font-style:italic;">{cr['summary']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Scatter: cluster memberships
    st.markdown("#### Cluster Membership — Risk Score vs Threat Score")
    if "cluster_name" in df.columns:
        fig5 = px.scatter(
            df,
            x="threat_score",
            y="composite_risk_score",
            color="cluster_name",
            hover_data=["source_ip", "user", "action"],
            symbol="risk_label",
            size_max=15,
        )
        fig5.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Raw Data
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Full Enriched Log Dataset")

    display_cols = [
        "timestamp", "source_ip", "user", "action", "asset_type",
        "country", "severity", "composite_risk_score", "risk_label",
        "threat_score", "zscore_ip_frequency", "cluster_name",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    search_term = st.text_input("🔍 Filter rows (any column)", "")
    filtered = df
    if search_term:
        mask = df[display_cols].astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered = df[mask]
        st.caption(f"{len(filtered)} rows matching '{search_term}'")

    st.dataframe(
        filtered[display_cols].sort_values("composite_risk_score", ascending=False),
        use_container_width=True,
        height=450,
    )

    # Export button
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download full dataset (CSV)",
        data=csv_bytes,
        file_name="triage_results.csv",
        mime="text/csv",
    )

    st.markdown("#### Pipeline Performance")
    perf_df = pd.DataFrame(
        [{"Step": k.replace("_sec","").replace("_"," ").title(), "Seconds": v}
         for k, v in stats.items() if k.endswith("_sec")]
    )
    st.table(perf_df)
