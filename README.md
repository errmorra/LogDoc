# 🛡️ AI-Driven Log Triage Tool

> A security analyst's force multiplier — turns 10,000 raw log lines into 10 prioritised, AI-explained incidents.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 What It Does

Modern SOC teams are drowning in alerts. This tool acts as a **digital filter** between raw log firehoses and exhausted analysts by:

1. **Ingesting** logs (CSV, JSON, Syslog) and normalising them to a canonical schema
2. **Enriching** each event with threat intelligence, VIP user detection, and asset criticality
3. **Scoring** events using a weighted Composite Risk Formula:
   ```
   Risk Score = (Severity × Asset Value)   [40%]
              + Threat Intelligence Factor [30%]
              + Behavioural Anomaly Score  [20%]
              + Rule Match Score           [10%]
   ```
4. **Clustering** similar events with K-Means so 1,000 "failed login" lines become 1 alert cluster
5. **Generating** plain-English incident narratives with an AI LLM backend

---

## 📐 Architecture

```
Raw Logs (CSV/JSON/Syslog)
        │
        ▼
┌─────────────────┐
│  1. Ingestion   │  Regex/NLP log-type classification
│  & Normalise    │  Canonical schema enforcement
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Enrichment  │  AbuseIPDB / VirusTotal threat scores
│                 │  VIP user & asset criticality lookup
│                 │  Off-hours & geo-risk detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Scoring     │  Weighted composite score (0-100)
│                 │  Z-Score IP frequency anomaly
│                 │  8 built-in attack rule signatures
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Clustering  │  K-Means grouping of similar events
│                 │  Centroid-based cluster naming
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. LLM Summary │  Anthropic Claude / Ollama / Rule-based
│                 │  Plain-English incident narratives
└────────┬────────┘
         │
         ▼
   Streamlit Dashboard
   (risk heatmap · timelines · cluster report · top alerts)
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
# Fill in API keys for live threat intel and/or LLM backend
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, then click **▶ Run Triage Pipeline** using the included sample data.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
ai-log-triage/
├── app.py                  # Streamlit dashboard (entry point)
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # Step 1: Load & normalise logs
│   ├── enrichment.py       # Step 2: Threat intel + context
│   ├── scoring.py          # Step 3: Composite risk scoring
│   ├── clustering.py       # Step 4: K-Means alert clustering
│   ├── llm_summary.py      # Step 5: AI narrative generation
│   └── pipeline.py         # Orchestration glue
│
├── data/
│   └── sample_logs.csv     # Demo dataset (30 log entries)
│
└── tests/
    └── test_scoring.py     # Pytest unit tests
```

---

## ⚙️ Configuration

All settings are via environment variables in `.env`:

| Variable | Purpose | Required? |
|---|---|---|
| `ABUSEIPDB_API_KEY` | Live IP reputation lookups | No |
| `VIRUSTOTAL_API_KEY` | Alternative threat intel | No |
| `ANTHROPIC_API_KEY` | Claude AI narrative generation | No |
| `ANTHROPIC_MODEL` | Which Claude model to use | No (default: claude-sonnet-4-20250514) |
| `OLLAMA_HOST` | Local LLM host | No (default: localhost:11434) |
| `OLLAMA_MODEL` | Local model name | No (default: mistral) |

**No API keys? No problem.** The tool uses a rule-based fallback for all LLM and threat intel features — it works completely offline.

---

## 🔬 Scoring Heuristic Deep Dive

### Composite Risk Formula

```
score = (asset_severity_component  × 0.40)
      + (threat_intel_component    × 0.30)
      + (behavioural_component     × 0.20)
      + (rule_match_component      × 0.10)
```

### Component Breakdown

**Asset × Severity (40%)**
Combines the event's native severity (Low=10, Medium=40, High=75, Critical=100) with the target asset's criticality (Domain Controller=10, Database=9, Workstation=4, …).

**Threat Intel (30%)**
Direct `abuseConfidenceScore` from AbuseIPDB, or a VirusTotal malicious ratio, or a heuristic offline score.

**Behavioural Anomaly (20%)**
Additive flags: off-hours login (+40), high-risk country (+30), VIP user (+20), large data transfer (+15-30).

**Rule Match (10%)**
8 built-in signature rules fire on pattern matches:
- Brute Force (≥5 failed logins from same IP)
- Log Cleared (`EventID 1102`)
- Off-Hours Critical Asset Access
- Massive Data Exfiltration (>500 MB)
- Privilege Escalation Indicators
- Firewall Policy Modified
- VIP Account from High-Risk Geography
- Root/Admin Login from External IP

### Z-Score Frequency Boost
IPs connecting significantly more than average (Z > 2 std deviations) receive an automatic +10 score boost.

---

## 🧪 Testing Your Setup with Real Data

### CIC-IDS2017 Dataset
Download from the [University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html):
```bash
# Convert PCAPs to CSV with CICFlowMeter, then:
python -c "
from src.pipeline import run_pipeline
results = run_pipeline('data/your_ids_dataset.csv')
print(results['stats'])
"
```

### Simulating the Equifax Breach
The sample dataset includes events mimicking the Equifax scenario:
- External IP with high threat score logging into a Database Server at 2 AM
- 1 GB data transfer from a critical asset
- Log cleared event after exfiltration

Run the pipeline and check whether these events hit a **Critical** risk score.

---

## 📊 Dashboard Features

| Tab | Contents |
|---|---|
| 🚨 Top Alerts | AI-narrated incident cards sorted by risk score |
| 📊 Analytics | Risk distribution donut, IP bar chart, timeline scatter, component heatmap |
| 🔵 Clusters | K-Means cluster report with avg/max scores and lead indicators |
| 📋 Raw Data | Full filterable table + CSV export |

---

## 🗺️ Roadmap

- [ ] MITRE ATT&CK technique tagging
- [ ] Webhook output (PagerDuty, Slack)
- [ ] Elasticsearch / OpenSearch log source connector
- [ ] LSTM time-series anomaly model
- [ ] Multi-tenant analyst assignment workflow
- [ ] SOAR playbook auto-generation

---

## 📄 License

MIT — free for personal, educational, and commercial use.

---

*Built as a portfolio project demonstrating applied ML in cybersecurity: unsupervised learning (K-Means), statistical anomaly detection (Z-Score), rule-based expert systems, and LLM-powered human-readable output.*
