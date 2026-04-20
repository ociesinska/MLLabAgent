import pytest

from ml_lab_agent.services.exp_services import compare_experiments, get_run_metrics, select_run


def test_select_run_existing_id():
    result = select_run("1")
    assert result["run_id"] == "1"


def test_select_run_missing_id():
    result = select_run("missing_run")
    assert result is None


def test_get_run_metrics_existing_id():
    result = get_run_metrics("1")
    assert result == {
        "accuracy": 0.81,
        "f1_score": 0.78,
    }


def test_get_run_metrics_returns_none_for_missing_id():
    result = get_run_metrics("missing_run")
    assert result is None


def test_compare_experiments_two_valid_runs():
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


def test_compare_experiments_missing_run():
    with pytest.raises(ValueError, match="No such run id: missing_run"):
        compare_experiments(["1", "missing_run"])
