"""
tests/test_scoring.py
Unit tests for the risk scoring engine.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.scoring import (
    score,
    get_top_alerts,
    _asset_severity_component,
    _behavioural_component,
    _rule_match_component,
    _score_to_label,
    WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_row():
    return pd.Series({
        "severity":          "Low",
        "asset_criticality": 3,
        "threat_score":      0,
        "is_off_hours":      False,
        "is_high_risk_geo":  False,
        "is_vip_user":       False,
        "bytes_transferred": 0,
        "action":            "Normal Activity",
        "event_id":          "0000",
        "source_ip":         "10.0.0.1",
    })


@pytest.fixture
def critical_row():
    return pd.Series({
        "severity":          "Critical",
        "asset_criticality": 10,
        "threat_score":      95,
        "is_off_hours":      True,
        "is_high_risk_geo":  True,
        "is_vip_user":       True,
        "bytes_transferred": 2 * 1024 ** 3,  # 2 GB
        "action":            "Successful Login",
        "event_id":          "1102",          # Log cleared
        "source_ip":         "185.220.101.45",
    })


@pytest.fixture
def sample_df():
    """A minimal DataFrame that satisfies the scoring pipeline."""
    return pd.DataFrame([
        {
            "timestamp":         pd.Timestamp("2024-01-15 08:00:00"),
            "source_ip":         "10.0.0.1",
            "user":              "jsmith",
            "action":            "Failed Login",
            "asset_type":        "Workstation",
            "severity":          "Medium",
            "asset_criticality": 4,
            "threat_score":      10,
            "is_off_hours":      False,
            "is_high_risk_geo":  False,
            "is_vip_user":       False,
            "bytes_transferred": 0,
            "event_id":          "4625",
            "country":           "US",
            "log_type":          "Authentication",
            "enrichment_notes":  "",
        },
        {
            "timestamp":         pd.Timestamp("2024-01-15 02:47:33"),
            "source_ip":         "185.220.101.45",
            "user":              "admin",
            "action":            "Successful Login",
            "asset_type":        "Database Server",
            "severity":          "Critical",
            "asset_criticality": 9,
            "threat_score":      95,
            "is_off_hours":      True,
            "is_high_risk_geo":  True,
            "is_vip_user":       True,
            "bytes_transferred": 1_073_741_824,
            "event_id":          "4624",
            "country":           "RU",
            "log_type":          "Authentication",
            "enrichment_notes":  "⚠ HIGH-THREAT IP | VIP account | Off-hours",
        },
    ])


# ---------------------------------------------------------------------------
# Weight sanity
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    total = sum(WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


# ---------------------------------------------------------------------------
# Component tests
# ---------------------------------------------------------------------------

def test_asset_severity_low(minimal_row):
    result = _asset_severity_component(minimal_row)
    assert 0 <= result <= 100

def test_asset_severity_critical_high(critical_row):
    result = _asset_severity_component(critical_row)
    assert result >= 80, f"Expected ≥80 for Critical/Critical, got {result}"

def test_behavioural_zero_for_clean(minimal_row):
    result = _behavioural_component(minimal_row)
    assert result == 0

def test_behavioural_max_scenario(critical_row):
    result = _behavioural_component(critical_row)
    assert result == 100, f"Expected 100 for all-flag row, got {result}"

def test_rule_match_log_cleared(critical_row):
    rules, rule_score = _rule_match_component(critical_row)
    assert any("Log Cleared" in r for r in rules), "Expected 'Log Cleared' rule to fire"
    assert rule_score > 0

def test_rule_match_empty_for_clean(minimal_row):
    rules, rule_score = _rule_match_component(minimal_row)
    assert rules == []
    assert rule_score == 0


# ---------------------------------------------------------------------------
# Score label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (0,   "Low"),
    (29,  "Low"),
    (30,  "Medium"),
    (54,  "Medium"),
    (55,  "High"),
    (79,  "High"),
    (80,  "Critical"),
    (100, "Critical"),
])
def test_score_to_label(score, expected):
    assert _score_to_label(score) == expected


# ---------------------------------------------------------------------------
# Full pipeline smoke-test
# ---------------------------------------------------------------------------

def test_score_pipeline_adds_columns(sample_df):
    result = score(sample_df)
    expected_cols = [
        "component_asset_severity",
        "component_threat_intel",
        "component_behavioural",
        "component_rule_match",
        "composite_risk_score",
        "risk_label",
        "matched_rules",
        "zscore_ip_frequency",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_score_range_0_100(sample_df):
    result = score(sample_df)
    assert result["composite_risk_score"].between(0, 100).all(), \
        "Composite scores outside 0-100 range"


def test_malicious_row_scores_higher(sample_df):
    result = score(sample_df)
    clean_score   = result.loc[result["source_ip"] == "10.0.0.1",  "composite_risk_score"].iloc[0]
    bad_score     = result.loc[result["source_ip"] == "185.220.101.45", "composite_risk_score"].iloc[0]
    assert bad_score > clean_score, \
        f"Expected malicious row ({bad_score}) > clean row ({clean_score})"


def test_get_top_alerts_returns_n(sample_df):
    result  = score(sample_df)
    alerts  = get_top_alerts(result, n=1)
    assert len(alerts) == 1
    assert alerts["composite_risk_score"].iloc[0] == result["composite_risk_score"].max()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_row_does_not_crash(minimal_row):
    df = pd.DataFrame([minimal_row])
    result = score(df)
    assert len(result) == 1


def test_score_deterministic(sample_df):
    """Same input → same output."""
    r1 = score(sample_df.copy())
    r2 = score(sample_df.copy())
    pd.testing.assert_series_equal(
        r1["composite_risk_score"].reset_index(drop=True),
        r2["composite_risk_score"].reset_index(drop=True),
    )
