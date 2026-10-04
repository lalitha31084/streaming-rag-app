import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    """Health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@patch("app.main.redis_client")
def test_health_endpoint_redis_connected(mock_redis):
    """Health endpoint returns redis: true when redis ping succeeds."""
    mock_redis.ping = AsyncMock(return_value=True)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["redis"] is True


@patch("app.main.redis_client")
def test_health_endpoint_redis_disconnected(mock_redis):
    """Health endpoint returns redis: false when redis ping raises exception."""
    mock_redis.ping = AsyncMock(side_effect=Exception("Redis connection refused"))
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["redis"] is False