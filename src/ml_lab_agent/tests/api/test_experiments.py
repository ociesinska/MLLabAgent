import pytest
from fastapi.testclient import TestClient

from ml_lab_agent.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_experiments_dependencies(monkeypatch):
    runs = {
        "1": {
            "run_id": "1",
            "experiment_name": "MLLabAgent Demo Runs",
            "metrics": {
                "accuracy": 0.81,
                "f1_score": 0.78,
            },
            "params": {},
            "tags": {},
        },
        "2": {
            "run_id": "2",
            "experiment_name": "MLLabAgent Demo Runs",
            "metrics": {
                "accuracy": 0.85,
                "f1_score": 0.82,
            },
            "params": {},
            "tags": {},
        },
        "3": {
            "run_id": "3",
            "experiment_name": "MLLabAgent Demo Runs",
            "metrics": {
                "accuracy": 0.89,
                "f1_score": 0.87,
            },
            "params": {},
            "tags": {},
        },
    }

    def _return_all_runs():
        return list(runs.values())

    def _select_run(run_id: str):
        return runs.get(run_id)

    def _compare_experiments(run_ids: list[str]):
        unique_run_ids = list(dict.fromkeys(run_ids))
        if len(unique_run_ids) < 2:
            raise ValueError("Need at least two unique runs to compare.")
        if len(unique_run_ids) > 2:
            raise ValueError("Can only accept two unique runs to compare.")

        for run_id in unique_run_ids:
            if run_id not in runs:
                raise ValueError(f"No such run id: {run_id}.")

        run_1, run_2 = unique_run_ids
        return {
            "compared_run_ids": [run_1, run_2],
            "metrics_comparison": {
                "accuracy": {
                    "value_run_1": runs[run_1]["metrics"]["accuracy"],
                    "value_run_2": runs[run_2]["metrics"]["accuracy"],
                    "winner": run_2,
                    "difference": abs(runs[run_1]["metrics"]["accuracy"] - runs[run_2]["metrics"]["accuracy"]),
                }
            },
            "parameter_comparison": {
                "batch_size": {
                    "value_run_1": "32",
                    "value_run_2": "32",
                    "changed": False,
                }
            },
            "overall_winner": run_2,
        }

    monkeypatch.setattr("ml_lab_agent.api.routes.experiments.return_all_runs", _return_all_runs)
    monkeypatch.setattr("ml_lab_agent.api.routes.experiments.select_run", _select_run)
    monkeypatch.setattr("ml_lab_agent.api.routes.experiments.compare_experiments", _compare_experiments)


def test_list_runs_returns_200_and_lists(client, mock_experiments_dependencies):
    response = client.get("/experiments/runs")

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body, list)
    assert len(body) > 0

    first_run = body[0]
    assert "run_id" in first_run
    assert "experiment_name" in first_run
    assert "metrics" in first_run


def test_get_run_returns_200_and_single_run(client, mock_experiments_dependencies):
    response = client.get("/experiments/runs/1")

    assert response.status_code == 200

    body = response.json()
    assert body["run_id"] == "1"
    assert "experiment_name" in body
    assert "metrics" in body


def test_get_run_returns_404(client, mock_experiments_dependencies):
    response = client.get("/experiments/runs/missing_run")
    assert response.status_code == 404

    body = response.json()
    assert body["detail"] == "Run with id missing_run not found"


def test_compare_experiments_returns_200_and_comparison(client, mock_experiments_dependencies):
    response = client.post("/experiments/compare", json={"run_ids": ["1", "2"]})

    assert response.status_code == 200

    body = response.json()
    assert body["compared_run_ids"] == ["1", "2"]
    assert "metrics_comparison" in body
    assert "overall_winner" in body


def test_compare_experiments_returns_400_missing_run(client, mock_experiments_dependencies):
    response = client.post(
        "/experiments/compare",
        json={"run_ids": ["1", "missing_run"]},
    )

    assert response.status_code == 400

    body = response.json()
    assert body["detail"] == "No such run id: missing_run."


def test_compare_experiments_returns_400_for_more_than_two_runs(client, mock_experiments_dependencies):
    response = client.post(
        "/experiments/compare",
        json={"run_ids": ["1", "2", "3"]},
    )

    assert response.status_code == 400

    body = response.json()
    assert body["detail"] == "Can only accept two unique runs to compare."


def test_compare_experiments_returns_400_for_duplicate_runs(client, mock_experiments_dependencies):
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
