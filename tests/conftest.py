"""
conftest.py
===========
Shared fixtures for the SmartWater AI test suite.

The tests are deliberately NON-MUTATING: they read the dataset and the saved
model artifacts, and they exercise the Flask app in-process via its test
client. Nothing here retrains a model or rewrites an artifact, so running the
suite can never change the reported metrics.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app as app_module  # noqa: E402
from src.predict import DemandPredictor  # noqa: E402


@pytest.fixture(scope="session")
def flask_app():
    app_module.app.config["TESTING"] = True
    return app_module.app


@pytest.fixture(scope="session")
def client(flask_app):
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def predictor():
    """A private predictor instance, so tests never disturb the app's buffer."""
    return DemandPredictor(model_name="random_forest_model.joblib")
