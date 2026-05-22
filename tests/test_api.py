# tests/test_api.py

"""
Unit tests for the FastAPI endpoints.
Uses MOCK_MODEL=true so no GPU required.
These run in GitHub Actions CI on every push.
"""

# tests/test_api.py

import os
os.environ["MOCK_MODEL"] = "true"

import pytest
from fastapi.testclient import TestClient
from src.serving.app import app


@pytest.fixture
def client():
    """
    Use TestClient as a context manager so FastAPI's lifespan
    runs properly — this triggers generator.load() on startup.
    """
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_ready"] is True


def test_generate_endpoint(client):
    response = client.post("/generate", json={
        "question": "How many employees are there?",
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR)",
    })
    assert response.status_code == 200
    data = response.json()
    assert "sql" in data
    assert len(data["sql"]) > 0


def test_generate_missing_field(client):
    response = client.post("/generate", json={
        "context": "CREATE TABLE employees (id INTEGER)",
    })
    assert response.status_code == 422


def test_generate_short_question(client):
    response = client.post("/generate", json={
        "question": "Hi",
        "context": "CREATE TABLE employees (id INTEGER)",
    })
    assert response.status_code == 422


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "QueryCraft" in response.json()["message"]