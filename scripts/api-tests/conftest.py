"""Shared settings of the SDK tests: they talk to a running fllm-rs server on the tiny test model.

FLLM_BASE_URL  server address            (default http://127.0.0.1:8000)
FLLM_API_KEY   a key the server accepts  (default: none, for a server started without --api-key)
FLLM_MODEL     served model name         (default tiny)
"""
import os

import pytest

BASE_URL = os.environ.get("FLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("FLLM_API_KEY") or None
MODEL = os.environ.get("FLLM_MODEL", "tiny")


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def api_key():
    return API_KEY


@pytest.fixture(scope="session")
def model():
    return MODEL


@pytest.fixture(scope="session")
def auth_headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
