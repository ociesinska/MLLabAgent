from unittest.mock import Mock

import pytest

from ml_lab_agent.api.agents.chat_graph.nodes import (
    compare_for_summary_node,
    fallback_summary_node,
    parse_input_node,
    route_after_summary,
    route_after_validate,
    show_node,
    unknown_node,
    validate_compare_node,
)


@pytest.fixture
def base_state():
    return {
        "message": "",
        "intent": None,
        "run_ids": [],
        "compare_results": None,
        "llm_error": None,
        "final_response": None,
    }


@pytest.fixture
def mock_compare_experiments_success(monkeypatch):
    fake_results = {
        "compared_run_ids": ["run_1", "run_2"],
        "metrics_comparison": {
            "accuracy": {
                "value_run_1": 0.81,
                "value_run_2": 0.85,
                "winner": "run_2",
                "difference": 0.04,
            }
        },
        "overall_winner": "run_2",
    }
    mock = Mock(return_value=fake_results)
    monkeypatch.setattr("ml_lab_agent.api.agents.chat_graph.nodes.compare_experiments", mock)
    return mock


@pytest.fixture
def mock_compare_experiments_error(monkeypatch):
    mock = Mock(side_effect=ValueError("Need at least two unique runs to compare."))
    monkeypatch.setattr("ml_lab_agent.api.agents.chat_graph.nodes.compare_experiments", mock)
    return mock


@pytest.fixture
def mock_select_run_success(monkeypatch):
    fake_run = {
        "run_id": "1",
        "experiment_name": "Experiment",
        "metrics": {"accuracy": 0.81, "f1_score": 0.78, "precision": 0.79, "recall": 0.77},
        "params": {"model_type": "logistic_regression", "learning_rate": "0.01", "batch_size": "32", "augmentation": "none"},
        "tags": {"mlflow.user": "ac", "mlflow.source.type": "LOCAL"},
    }
    mock = Mock(return_value=fake_run)
    monkeypatch.setattr("ml_lab_agent.api.agents.chat_graph.nodes.select_run", mock)
    return mock


@pytest.fixture
def mock_select_run_none(monkeypatch):
    mock = Mock(return_value=None)
    monkeypatch.setattr(
        "ml_lab_agent.api.agents.chat_graph.nodes.select_run",
        mock,
    )
    return mock


@pytest.fixture
def mock_return_all_runs(monkeypatch):
    fake_runs = [
        {"run_id": "1", "metrics": {"accuracy": 0.81}},
        {"run_id": "2", "metrics": {"accuracy": 0.85}},
    ]
    mock = Mock(return_value=fake_runs)
    monkeypatch.setattr(
        "ml_lab_agent.api.agents.chat_graph.nodes.return_all_runs",
        mock,
    )
    return mock


@pytest.fixture
def mock_resolve_run_identifiers(monkeypatch):
    mock = Mock(side_effect=lambda ids: ids)
    monkeypatch.setattr(
        "ml_lab_agent.api.agents.chat_graph.nodes.resolve_run_identifiers",
        mock,
    )
    return mock


def test_parse_input_node_extracts_show_intent_and_run_ids(mock_resolve_run_identifiers):
    state = {"message": "show run 1"}
    result = parse_input_node(state)

    assert result["intent"] == "show"
    assert result["run_ids"] == ["1"]
    mock_resolve_run_identifiers.assert_called_once_with(["1"])


def test_parse_input_node_extracts_compare_intent_and_run_ids(mock_resolve_run_identifiers):
    state = {"message": "compare run 1 and run 2"}
    result = parse_input_node(state)

    assert result["intent"] == "compare"
    assert result["run_ids"] == ["1", "2"]
    mock_resolve_run_identifiers.assert_called_once_with(["1", "2"])


def test_parse_input_node_extracts_summarize_intent_and_two_run_ids(mock_resolve_run_identifiers):
    state = {"message": "compare and summarize run 1 and run 2"}
    result = parse_input_node(state)

    assert result["intent"] == "summarize_compare"
    assert result["run_ids"] == ["1", "2"]
    mock_resolve_run_identifiers.assert_called_once_with(["1", "2"])


def test_parse_input_node_returns_unknown_intent_for_unsupported_message():
    state = {"message": "hello there"}
    result = parse_input_node(state)

    assert result["intent"] == "unknown"
    assert result["run_ids"] == []


def test_show_node_returns_single_run_for_one_run_id(mock_select_run_success):
    state = {"run_ids": ["1"]}
    result = show_node(state)
    response = result["final_response"]

    assert response.intent == "show"
    assert response.error is None
    assert response.data is not None
    assert response.data["run_id"] == "1"
    assert response.message == "Run 1 found."
    mock_select_run_success.assert_called_once_with("1")


def test_show_node_returns_all_runs_for_empty_run_ids(mock_return_all_runs):
    state = {"run_ids": []}
    result = show_node(state)

    response = result["final_response"]

    assert response.intent == "show"
    assert response.error is None
    assert isinstance(response.data, list)
    assert len(response.data) == 2
    assert response.message == "Returning all runs."
    mock_return_all_runs.assert_called_once_with()


def test_show_node_returns_error_for_multiple_run_ids():
    state = {"run_ids": ["1", "2"]}
    result = show_node(state)

    response = result["final_response"]

    assert response.intent == "show"
    assert response.data is None
    assert response.error == "Only one run's details can be shown."


def test_show_node_returns_error_for_missing_run(mock_select_run_none):
    state = {"run_ids": ["999"]}
    result = show_node(state)

    response = result["final_response"]

    assert response.intent == "show"
    assert response.data is None
    assert response.error == "Run 999 not found."
    mock_select_run_none.assert_called_once_with("999")


def test_validate_compare_node_returns_error_for_single_run_id(base_state):
    state = {**base_state, "intent": "compare", "run_ids": ["1"]}

    result = validate_compare_node(state)
    response = result["final_response"]

    assert response.intent == "compare"
    assert response.data is None
    assert response.error == "Need at least two unique runs to compare."


def test_validate_compare_node_returns_error_for_more_than_two_run_ids(base_state):
    state = {**base_state, "intent": "compare", "run_ids": ["1", "2", "3"]}

    result = validate_compare_node(state)
    response = result["final_response"]

    assert response.intent == "compare"
    assert response.data is None
    assert response.error == "Can only accept two unique runs to compare."


def test_validate_compare_node_returns_empty_update_for_two_unique_run_ids(base_state):
    state = {**base_state, "intent": "compare", "run_ids": ["1", "2"]}

    result = validate_compare_node(state)

    assert result == {}


def test_route_after_validate_returns_end_when_final_response_exists(base_state):
    state = {
        **base_state,
        "intent": "compare",
        "final_response": object(),
    }

    result = route_after_validate(state)

    assert result == "end"


def test_route_after_validate_returns_compare_node_for_compare_intent(base_state):
    state = {
        **base_state,
        "intent": "compare",
        "final_response": None,
    }

    result = route_after_validate(state)

    assert result == "compare_node"


def test_route_after_validate_returns_compare_for_summary_node_for_summary_intent(base_state):
    state = {
        **base_state,
        "intent": "summarize_compare",
        "final_response": None,
    }

    result = route_after_validate(state)

    assert result == "compare_for_summary_node"


def test_compare_for_summary_node_sets_compare_results(base_state, mock_compare_experiments_success):
    state = {
        **base_state,
        "intent": "summarize_compare",
        "run_ids": ["run_1", "run_2"],
    }

    result = compare_for_summary_node(state)

    assert result["compare_results"]["overall_winner"] == "run_2"
    assert result["llm_error"] is None
    mock_compare_experiments_success.assert_called_once_with(["run_1", "run_2"])


def test_compare_for_summary_node_returns_error_response_for_invalid_run_ids(base_state, mock_compare_experiments_error):
    state = {
        **base_state,
        "intent": "summarize_compare",
        "run_ids": ["1"],
    }

    result = compare_for_summary_node(state)
    response = result["final_response"]

    assert response.intent == "summarize_compare"
    assert response.data is None
    assert response.error == "Need at least two unique runs to compare."
    mock_compare_experiments_error.assert_called_once_with(["1"])


def test_route_after_summary_returns_end_when_final_response_exists(base_state):
    state = {
        **base_state,
        "final_response": object(),
        "llm_error": None,
    }

    result = route_after_summary(state)

    assert result == "end_after_summary"


def test_route_after_summary_returns_fallback_node_when_llm_error_exists(base_state):
    state = {
        **base_state,
        "final_response": None,
        "llm_error": "provider timeout",
    }

    result = route_after_summary(state)

    assert result == "fallback_summary_node"


def test_fallback_summary_node_returns_fallback_chat_response(base_state):
    state = {
        **base_state,
        "compare_results": {
            "compared_run_ids": ["1", "2"],
            "metrics_comparison": {
                "accuracy": {
                    "value_run_1": 0.81,
                    "value_run_2": 0.85,
                    "winner": "2",
                    "difference": 0.04,
                },
                "f1_score": {
                    "value_run_1": 0.78,
                    "value_run_2": 0.82,
                    "winner": "2",
                    "difference": 0.04,
                },
            },
            "overall_winner": "2",
        },
    }

    result = fallback_summary_node(state)
    response = result["final_response"]

    assert response.intent == "summarize_compare"
    assert response.error is None
    assert response.data is not None
    assert response.data["compare_results"]["overall_winner"] == "2"
    assert "summary" in response.data


def test_unknown_node_returns_unknown_chat_response(base_state):
    result = unknown_node(base_state)
    response = result["final_response"]

    assert response.intent == "unknown"
    assert response.data is None
    assert response.error == "Unsupported request."
