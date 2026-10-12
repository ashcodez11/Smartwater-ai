"""
test_api.py
===========
Contract tests for the Flask application: every page, every endpoint, the
error paths, and the guarantee that the API returns what the saved model
actually computes.

Uses the Flask test client, so no running server is required.
"""

import os
import re
import shutil
import subprocess
import tempfile

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ["/", "/prediction", "/forecast", "/analytics", "/model-performance"]
REGIONS = ["Central", "East", "North", "South", "West"]
LEVELS = ["LOW", "NORMAL", "HIGH", "VERY HIGH"]


# --------------------------------------------------------------- pages
@pytest.mark.parametrize("path", PAGES)
def test_page_renders(client, path):
    res = client.get(path)
    assert res.status_code == 200
    assert b"SmartWater AI" in res.data


def test_hero_no_longer_shows_a_date_range(client):
    """The hero meta row shows readings/regions/days — the dataset date range
    was intentionally removed, so neither the element nor its JS remains."""
    html = client.get("/").get_data(as_text=True)
    assert "hero-range" not in html
    assert "Jan 2023" not in html


@pytest.mark.parametrize("path", PAGES)
def test_inline_scripts_are_valid_javascript(client, path):
    """Every page carries a large inline <script>. A single syntax error in one
    silently disables *all* of that page's panels (the browser stops parsing at
    the error), which is invisible to a plain HTTP 200 check. Syntax-check each
    block with node when it is available.

    This caught a real bug: a JSON key containing a dot
    (`expected_false_positives_at_0.05`) written in dot-notation inside a
    template literal, where `.05` parses as a number literal.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed — cannot syntax-check the inline scripts")

    html = client.get(path).get_data(as_text=True)
    blocks = re.findall(r"<script>([\s\S]*?)</script>", html)
    assert blocks, f"no inline script found on {path}"

    for i, js in enumerate(blocks):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False)
        try:
            tmp.write(js)
            tmp.close()
            proc = subprocess.run([node, "--check", tmp.name],
                                  capture_output=True, text=True)
            assert proc.returncode == 0, (
                f"syntax error in inline script {i} of {path}:\n{proc.stderr[:900]}")
        finally:
            os.unlink(tmp.name)


# ------------------------------------------------------------ endpoints
def test_health(client):
    data = client.get("/api/health").get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["classifier_loaded"] is True
    assert set(data["classifier_classes"]) == set(LEVELS)


@pytest.mark.parametrize("path", [
    "/api/metrics", "/api/classification-metrics", "/api/confusion-matrix",
    "/api/dashboard-stats", "/api/analytics-data", "/api/feature-importance",
    "/api/signal-audit",
])
def test_get_endpoints_return_200(client, path):
    res = client.get(path)
    assert res.status_code == 200
    assert isinstance(res.get_json(), dict)


def test_dashboard_stats_match_the_dataset(client):
    data = client.get("/api/dashboard-stats").get_json()
    assert data["total_records"] == 900
    assert sorted(data["regions"]) == ["Central", "East", "North", "South", "West"]
    assert data["average_consumption"] == pytest.approx(12753.8, abs=0.5)
    assert data["peak_consumption"] == pytest.approx(19989.8, abs=0.5)


def test_confusion_matrix_shape(client):
    data = client.get("/api/confusion-matrix").get_json()
    cm = data["confusion_matrix"]
    assert len(cm) == 4 and all(len(row) == 4 for row in cm)
    assert sum(sum(row) for row in cm) == 170
    assert data["best_classifier"] == "RandomForest"


def test_signal_audit_reports_the_ceiling(client):
    data = client.get("/api/signal-audit").get_json()
    assert data["white_noise_summary"]["conclusion"] == "cannot reject white noise"
    reg = data["ceiling"]["regression"]
    assert reg["deployed_MAE"] > reg["ceiling_MAE"]          # can never beat the optimum
    assert reg["gap_vs_ceiling_percent"] < 5.0
    assert data["ceiling"]["classification"]["ceiling_accuracy"] == pytest.approx(0.2706, abs=1e-3)


# -------------------------------------------------------------- predict
@pytest.mark.parametrize("region", REGIONS)
def test_predict_all_regions(client, region):
    res = client.post("/api/predict", json={"region": region, "date": "2023-08-01"})
    assert res.status_code == 200
    data = res.get_json()
    assert 3_000 < data["predicted_consumption"] < 25_000
    assert data["demand_level"] in LEVELS
    assert data["unit"] == "L/day"
    assert set(data["features"]) >= {"lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14"}


def test_predict_returns_the_features_it_used(client):
    """The UI explains a prediction from these values, so they must be present
    and must be real readings — lag_1 is the last actual, not a forecast."""
    data = client.post("/api/predict", json={"region": "Central", "date": "2026-10-13"}).get_json()
    assert data["features"]["lag_1"] == pytest.approx(17917.25, abs=0.01)


def test_predict_rejects_unknown_region(client):
    res = client.post("/api/predict", json={"region": "Atlantis", "date": "2026-10-13"})
    assert res.status_code == 400
    assert "Unknown region" in res.get_json()["error"]


def test_predict_rejects_missing_fields(client):
    res = client.post("/api/predict", json={"region": "Central"})
    assert res.status_code == 400
    assert "Missing required fields" in res.get_json()["error"]


def test_prediction_is_stable_across_forecasts(client):
    """End-to-end version of the buffer-leak regression test."""
    first = client.post("/api/predict", json={"region": "Central", "date": "2026-10-13"}).get_json()
    client.get("/api/forecast?region=Central")
    second = client.post("/api/predict", json={"region": "Central", "date": "2026-10-13"}).get_json()
    assert first["predicted_consumption"] == second["predicted_consumption"]


# -------------------------------------------------------------- forecast
@pytest.mark.parametrize("region", REGIONS)
def test_forecast_returns_seven_days(client, region):
    res = client.get("/api/forecast?region=" + region)
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["forecast"]) == 7
    for row in data["forecast"]:
        assert row["demand_level"] in LEVELS
        assert row["is_weekend"] in (0, 1)
        assert 3_000 < row["predicted_consumption"] < 25_000


def test_forecast_rejects_unknown_region(client):
    assert client.get("/api/forecast?region=Atlantis").status_code == 400


def test_forecast_defaults_to_first_region(client):
    data = client.get("/api/forecast").get_json()
    assert data["region"] in REGIONS
    assert len(data["forecast"]) == 7
