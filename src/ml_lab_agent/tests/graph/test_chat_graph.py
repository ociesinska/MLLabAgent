from unittest.mock import Mock, patch

import pytest

from ml_lab_agent.api.agents.chat_graph.graph import graph
from ml_lab_agent.schemas.chat_schemas import ParsedUserRequest


@pytest.fixture
def base_graph_state():
    return {
        "message": "",
        "intent": None,
        "run_ids": [],
        "metric": None,
        "compare_results": None,
        "llm_error": None,
        "final_response": None,
    }


@pytest.fixture
def mock_resolve_run_identifiers(monkeypatch):
    mock = Mock(side_effect=lambda ids: ids)
    monkeypatch.setattr(
        "ml_lab_agent.api.agents.chat_graph.nodes.resolve_run_identifiers",
        mock,
    )
    return mock


@pytest.fixture
def mock_select_run_success(monkeypatch):
    fake_run = {
        "run_id": "1",
        "experiment_name": "Experiment",
        "metrics": {
            "accuracy": 0.81,
            "f1_score": 0.78,
        },
        "params": {
            "model_type": "logistic_regression",
        },
        "tags": {
            "mlflow.user": "ac",
        },
    }
    mock = Mock(return_value=fake_run)
    monkeypatch.setattr(
        "ml_lab_agent.api.agents.chat_graph.nodes.select_run",
        mock,
    )
    return mock


@pytest.fixture
def mock_compare_experiments_success(monkeypatch):
    fake_compare = {
        "overall_winner": "2",
        "metrics_comparison": {
            "accuracy": {"1": 0.81, "2": 0.85, "winner": "2"},
            "f1_score": {"1": 0.78, "2": 0.82, "winner": "2"},
        },
    }
    mock = Mock(return_value=fake_compare)
    monkeypatch.setattr(
        "ml_lab_agent.api.agents.chat_graph.nodes.compare_experiments",
        mock,
    )
    return mock


def test_graph_invoke_show_returns_all_runs_response(base_graph_state):
    state = {**base_graph_state, "message": "show"}

    fake_parsed_request = ParsedUserRequest(intent="show", run_identifiers=[], metric=None)

    with patch("ml_lab_agent.api.agents.chat_graph.nodes.parse_request", return_value=fake_parsed_request):
        result = graph.invoke(state)

    response = result["final_response"]

    assert response.intent == "show"
    assert response.error is None
    assert response.message == "Returning all runs."
    assert isinstance(response.data, list)
    assert len(response.data) > 0


def test_graph_invoke_show_single_run_returns_run_details(
    base_graph_state,
    mock_resolve_run_identifiers,
    mock_select_run_success,
):
    state = {**base_graph_state, "message": "show run 1"}

    fake_parsed_request = ParsedUserRequest(intent="show", run_identifiers=["1"], metric=None)

    with patch("ml_lab_agent.api.agents.chat_graph.nodes.parse_request", return_value=fake_parsed_request):
        result = graph.invoke(state)

    response = result["final_response"]

    assert response.intent == "show"
    assert response.error is None
    assert response.message == "Run 1 found."
    assert response.data is not None
    assert response.data["run_id"] == "1"


def test_graph_invoke_compare_returns_compare_response(
    base_graph_state,
    mock_resolve_run_identifiers,
    mock_compare_experiments_success,
):
    state = {**base_graph_state, "message": "compare run 1 and run 2"}

    fake_parsed_request = ParsedUserRequest(intent="compare", run_identifiers=["1", "2"], metric=None)

    with patch("ml_lab_agent.api.agents.chat_graph.nodes.parse_request", return_value=fake_parsed_request):
        result = graph.invoke(state)

    response = result["final_response"]

    assert response.intent == "compare"
    assert response.error is None
    assert response.message == "Comparison generated successfully."
    assert response.data is not None
    assert response.data["overall_winner"] == "2"


def test_graph_invoke_compare_with_single_run_returns_validation_error(
    base_graph_state,
    mock_resolve_run_identifiers,
):
    state = {**base_graph_state, "message": "compare run 1"}

    fake_parsed_request = ParsedUserRequest(intent="compare", run_identifiers=["1"], metric=None)

    with patch("ml_lab_agent.api.agents.chat_graph.nodes.parse_request", return_value=fake_parsed_request):
        result = graph.invoke(state)

    response = result["final_response"]

    assert response.intent == "compare"
    assert response.data is None
    assert response.message == "Cannot process this request."
    assert response.error == "Need at least two unique runs to compare."


def test_graph_invoke_summarize_compare_returns_summary_response(
    base_graph_state,
    mock_resolve_run_identifiers,
    mock_compare_experiments_success,
):
    fake_summary = {
        "summary": "Run 2 performed better overall.",
        "metric_insights": [
            "Accuracy improved by 0.04.",
            "F1-score improved by 0.04.",
        ],
        "next_experiment_ideas": [
            "Try stronger augmentation because it may improve generalization.",
            "Tune learning rate because it may further improve convergence.",
        ],
    }

    state = {**base_graph_state, "message": "summarize compare run 1 and run 2"}

    fake_parsed_request = ParsedUserRequest(intent="summarize_compare", run_identifiers=["1", "2"], metric=None)

    with patch("ml_lab_agent.api.agents.chat_graph.nodes.parse_request", return_value=fake_parsed_request):
        with patch("ml_lab_agent.api.agents.chat_graph.nodes.generate_compare_summary", return_value=fake_summary):
            result = graph.invoke(state)

    response = result["final_response"]

    assert response.intent == "summarize_compare"
    assert response.error is None
    assert response.message == "Comparison summary generated successfully."
    assert response.data is not None
    assert "compare_results" in response.data
    assert "summary" in response.data
    assert response.data["summary"]["summary"] == "Run 2 performed better overall."


def test_graph_invoke_summarize_compare_uses_fallback_when_llm_fails(
    base_graph_state,
    mock_resolve_run_identifiers,
    mock_compare_experiments_success,
):
    state = {**base_graph_state, "message": "summarize compare run 1 and run 2"}

    fake_parsed_request = ParsedUserRequest(intent="summarize_compare", run_identifiers=["1", "2"], metric=None)

    with patch("ml_lab_agent.api.agents.chat_graph.nodes.parse_request", return_value=fake_parsed_request):
        with patch("ml_lab_agent.api.agents.chat_graph.nodes.generate_compare_summary", side_effect=Exception("LLM provider timeout")):
            result = graph.invoke(state)

    response = result["final_response"]

    assert response.intent == "summarize_compare"
    assert response.error is None
    assert response.message == "Comparison summary generated with fallback logic."
    assert response.data is not None
    assert "compare_results" in response.data
    assert "summary" in response.data
    assert response.data["summary"]["summary"] == "Run 2 performed better overall."


def test_graph_invoke_show_best_run_by_metric(base_graph_state):
    state = {**base_graph_state, "message": "show me best run by f1_score"}

    fake_parsed_request = ParsedUserRequest(intent="show_best_run", run_identifiers=[], metric="f1_score")

    with patch("ml_lab_agent.api.agents.chat_graph.nodes.parse_request", return_value=fake_parsed_request):
        result = graph.invoke(state)

    response = result["final_response"]

    assert response.intent == "show_best_run"
    assert response.error is None
    assert response.data["metric"] == "f1_score"


def test_graph_invoke_compare_returns_compare_response_latest_and_best_by_metric(
    base_graph_state,
):
    state = {**base_graph_state, "message": "compare run 1 and run 2"}

    fake_parsed_request = ParsedUserRequest(intent="compare", run_identifiers=["latest", "best_by:accuracy"], metric=None)
    fake_compare = {
        "overall_winner": "run_latest",
        "metrics_comparison": {
            "accuracy": {"run_latest": 0.91, "run_best_accuracy": 0.89, "winner": "run_latest"},
            "f1_score": {"run_latest": 0.88, "run_best_accuracy": 0.87, "winner": "run_latest"},
        },
    }

    with patch("ml_lab_agent.api.agents.chat_graph.nodes.parse_request", return_value=fake_parsed_request):
        with patch(
            "ml_lab_agent.api.agents.chat_graph.nodes.resolve_run_identifiers",
            return_value=["run_latest", "run_best_accuracy"],
        ):
            with patch(
                "ml_lab_agent.api.agents.chat_graph.nodes.compare_experiments",
                return_value=fake_compare,
            ):
                result = graph.invoke(state)

    response = result["final_response"]

    assert response.intent == "compare"
    assert response.error is None
    assert response.message == "Comparison generated successfully."
    assert response.data is not None
    assert response.data["overall_winner"] == "run_latest"


def test_graph_summarize_latest_and_best_by_metric_uses_summary(base_graph_state):
    state = {**base_graph_state, "message": "summarize latest run vs best run by accuracy"}

    fake_parsed_request = ParsedUserRequest(
        intent="summarize_compare",
        run_identifiers=["latest", "best_by:accuracy"],
        metric=None,
    )
    fake_compare = {
        "overall_winner": "run_latest",
        "metrics_comparison": {
            "accuracy": {"run_latest": 0.91, "run_best_accuracy": 0.89, "winner": "run_latest"},
            "f1_score": {"run_latest": 0.88, "run_best_accuracy": 0.87, "winner": "run_latest"},
        },
    }
    fake_summary = {
        "summary": "Run run_latest performed better overall.",
        "metric_insights": ["Accuracy was higher for run_latest."],
        "next_experiment_ideas": ["Try tuning learning rate for best_by run."],
    }

    with patch("ml_lab_agent.api.agents.chat_graph.nodes.parse_request", return_value=fake_parsed_request):
        with patch(
            "ml_lab_agent.api.agents.chat_graph.nodes.resolve_run_identifiers",
            return_value=["run_latest", "run_best_accuracy"],
        ):
            with patch(
                "ml_lab_agent.api.agents.chat_graph.nodes.compare_experiments",
                return_value=fake_compare,
            ):
                with patch(
                    "ml_lab_agent.api.agents.chat_graph.nodes.generate_compare_summary",
                    return_value=fake_summary,
                ):
                    result = graph.invoke(state)

    response = result["final_response"]

    assert response.intent == "summarize_compare"
    assert response.error is None
    assert response.message == "Comparison summary generated successfully."
    assert response.data is not None
    assert response.data["compare_results"]["overall_winner"] == "run_latest"
    assert response.data["summary"]["summary"] == "Run run_latest performed better overall."

