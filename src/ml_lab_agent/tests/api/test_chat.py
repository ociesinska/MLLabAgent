import pytest
from fastapi.testclient import TestClient

from ml_lab_agent.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_chat_dependencies(monkeypatch):
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

    def _resolve_run_identifiers(run_identifiers: list[str]):
        return run_identifiers

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
            "overall_winner": run_2,
        }

    monkeypatch.setattr("ml_lab_agent.api.agents.chat_graph.nodes.return_all_runs", _return_all_runs)
    monkeypatch.setattr("ml_lab_agent.api.agents.chat_graph.nodes.select_run", _select_run)
    monkeypatch.setattr("ml_lab_agent.api.agents.chat_graph.nodes.resolve_run_identifiers", _resolve_run_identifiers)
    monkeypatch.setattr("ml_lab_agent.api.agents.chat_graph.nodes.compare_experiments", _compare_experiments)


def test_chat_response_show_all(client, mock_chat_dependencies):
    response = client.post("/chat", json={"message": "show"})

    assert response.status_code == 200
    body = response.json()

    assert "message" in body

    assert body["intent"] == "show"
    assert body["error"] is None
    assert isinstance(body["data"], list)


def test_chat_response_show_one_id(client, mock_chat_dependencies):
    response = client.post("/chat", json={"message": "show run 1"})

    assert response.status_code == 200
    body = response.json()

    assert "message" in body
    assert body["intent"] == "show"
    assert body["data"]["run_id"] == "1"
    assert body["error"] is None
    assert isinstance(body["data"], dict)


def test_chat_response_compare_two(client, mock_chat_dependencies):
    response = client.post("/chat", json={"message": "compare run 1 and run 2"})

    assert response.status_code == 200
    body = response.json()

    assert "message" in body
    assert body["intent"] == "compare"
    assert body["error"] is None
    assert isinstance(body["data"], dict)


def test_chat_response_unknown_intent(client, mock_chat_dependencies):
    response = client.post("/chat", json={"message": "hello there"})

    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "unknown"
    assert body["data"] is None
    assert body["error"] is not None


def test_chat_response_summarize_compare_three(client, mock_chat_dependencies):
    response = client.post("/chat", json={"message": "compare run 1, run 2 and run 3"})

    assert response.status_code == 200
    body = response.json()

    assert "message" in body
    assert body["intent"] == "compare"
    assert body["error"] == "Can only accept two unique runs to compare."
    assert body["data"] is None
