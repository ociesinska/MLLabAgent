import pytest
from fastapi.testclient import TestClient

from ml_lab_agent.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_list_runs_returns_200_and_lists(client):
    response = client.get("/experiments/runs")

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body, list)
    assert len(body) > 0

    first_run = body[0]
    assert "run_id" in first_run
    assert "experiment_name" in first_run
    assert "metrics" in first_run


def test_get_run_returns_200_and_single_run(client):
    response = client.get("/experiments/runs/1")

    assert response.status_code == 200

    body = response.json()
    assert body["run_id"] == "1"
    assert "experiment_name" in body
    assert "metrics" in body


def test_get_run_returns_404(client):
    response = client.get("/experiments/runs/missing_run")
    assert response.status_code == 404

    body = response.json()
    assert body["detail"] == "Run with id missing_run not found"


def test_compare_experiments_returns_200_and_comparison(client):
    response = client.post("/experiments/compare", json={"run_ids": ["1", "2"]})

    assert response.status_code == 200

    body = response.json()
    assert body["compared_run_ids"] == ["1", "2"]
    assert "metrics_comparison" in body
    assert "overall_winner" in body


def test_compare_experiments_returns_400_missing_run(client):
    response = client.post(
        "/experiments/compare",
        json={"run_ids": ["1", "missing_run"]},
    )

    assert response.status_code == 400

    body = response.json()
    assert body["detail"] == "No such run id: missing_run."


def test_compare_experiments_returns_400_for_more_than_two_runs(client):
    response = client.post(
        "/experiments/compare",
        json={"run_ids": ["1", "2", "3"]},
    )

    assert response.status_code == 400

    body = response.json()
    assert body["detail"] == "Can only accept two unique runs to compare."


def test_compare_experiments_returns_400_for_duplicate_runs(client):
    response = client.post(
        "/experiments/compare",
        json={"run_ids": ["1", "1"]},
    )

    assert response.status_code == 400

    body = response.json()
    assert body["detail"] == "Need at least two unique runs to compare."


def test_compare_experiments_returns_422_for_wrong_run_ids_type(client):
    response = client.post("/experiments/compare", json={"run_ids": 123})
    assert response.status_code == 422
