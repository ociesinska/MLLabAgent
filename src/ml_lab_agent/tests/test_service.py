from unittest.mock import Mock

import pytest

from ml_lab_agent.services.exp_services import compare_experiments, get_run_metrics, resolve_single_run_identifier, select_run


@pytest.fixture
def mock_repository_get_run(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(
        "ml_lab_agent.services.exp_services.repository.get_run",
        mock,
    )
    return mock


@pytest.fixture
def mock_repository_get_run_metrics(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(
        "ml_lab_agent.services.exp_services.repository.get_run_metrics",
        mock,
    )
    return mock


@pytest.fixture
def mock_select_run_for_compare(monkeypatch):
    fake_run = {
        "1": {
            "run_id": "1",
            "metrics": {
                "accuracy": 0.81,
                "f1_score": 0.78,
            },
            "params": {
                "model_type": "logistic_regression",
                "batch_size": "32",
            },
        },
        "2": {
            "run_id": "2",
            "metrics": {
                "accuracy": 0.85,
                "f1_score": 0.82,
            },
            "params": {
                "model_type": "simple_cnn",
                "batch_size": "32",
            },
        },
    }

    def _select_run_side_effect(run_id):
        return fake_run.get(run_id)

    mock = Mock(side_effect=_select_run_side_effect)
    monkeypatch.setattr(
        "ml_lab_agent.services.exp_services.select_run",
        mock,
    )
    return mock


def test_select_run_existing_id(mock_repository_get_run):
    mock_repository_get_run.return_value = {"run_id": "1"}
    result = select_run("1")
    assert result["run_id"] == "1"
    mock_repository_get_run.assert_called_once_with("1")


def test_select_run_missing_id(mock_repository_get_run):
    mock_repository_get_run.return_value = None
    result = select_run("missing_run")
    assert result is None
    mock_repository_get_run.assert_called_once_with("missing_run")


def test_get_run_metrics_existing_id(mock_repository_get_run_metrics):
    mock_repository_get_run_metrics.return_value = {
        "accuracy": 0.81,
        "f1_score": 0.78,
    }
    result = get_run_metrics("1")
    assert result == {
        "accuracy": 0.81,
        "f1_score": 0.78,
    }
    mock_repository_get_run_metrics.assert_called_once_with("1")


def test_get_run_metrics_returns_none_for_missing_id(mock_repository_get_run_metrics):
    mock_repository_get_run_metrics.return_value = None
    result = get_run_metrics("missing_run")
    assert result is None
    mock_repository_get_run_metrics.assert_called_once_with("missing_run")


def test_compare_experiments_two_valid_runs(mock_select_run_for_compare):
    result = compare_experiments(["1", "2"])

    assert result["compared_run_ids"] == ["1", "2"]
    assert result["overall_winner"] == "2"

    accuracy = result["metrics_comparison"]["accuracy"]
    assert accuracy["value_run_1"] == pytest.approx(0.81)
    assert accuracy["value_run_2"] == pytest.approx(0.85)
    assert accuracy["winner"] == "2"
    assert accuracy["difference"] == pytest.approx(0.04)

    f1_score = result["metrics_comparison"]["f1_score"]
    assert f1_score["value_run_1"] == pytest.approx(0.78)
    assert f1_score["value_run_2"] == pytest.approx(0.82)
    assert f1_score["winner"] == "2"
    assert f1_score["difference"] == pytest.approx(0.04)

    assert mock_select_run_for_compare.call_count == 2


def test_compare_experiments_missing_run(mock_select_run_for_compare):
    with pytest.raises(ValueError, match="No such run id: missing_run"):
        compare_experiments(["1", "missing_run"])


def test_resolve_single_run_identifier_resolves_latest(monkeypatch):
    monkeypatch.setattr(
        "ml_lab_agent.services.exp_services.show_latest_run",
        lambda: {"run_id": "latest_run_123"},
    )

    result = resolve_single_run_identifier("latest")

    assert result == "latest_run_123"


def test_resolve_single_run_identifier_resolves_best_by_metric(monkeypatch):
    monkeypatch.setattr(
        "ml_lab_agent.services.exp_services.show_best_run_by_metric",
        lambda metric: {
            "metric": metric,
            "best_run": {"run_id": "best_acc_run_456"},
            "best_value": 0.93,
            "num_runs_checked": 3,
        },
    )

    result = resolve_single_run_identifier("best_by:accuracy")

    assert result == "best_acc_run_456"
