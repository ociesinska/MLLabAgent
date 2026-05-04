from ml_lab_agent.schemas.agent_schemas import AgentPlan, AgentToolCall
from ml_lab_agent.services.agent_services import execute_agent_plan


def test_execute_agent_plan_resolves_context_references(monkeypatch):
    monkeypatch.setattr(
        "ml_lab_agent.services.agent_services.show_latest_run",
        lambda: {"run_id": "latest-1", "metrics": {"accuracy": 0.81}, "params": {}},
    )
    monkeypatch.setattr(
        "ml_lab_agent.services.agent_services.show_best_run_by_metric",
        lambda metric: {
            "metric": metric,
            "best_run": {"run_id": "best-2", "metrics": {"accuracy": 0.85}, "params": {}},
            "best_value": 0.85,
            "num_runs_checked": 2,
        },
    )

    compare_calls = []

    def _compare_experiments(run_ids):
        compare_calls.append(run_ids)
        return {
            "compared_run_ids": run_ids,
            "metrics_comparison": {
                "accuracy": {
                    "value_run_1": 0.81,
                    "value_run_2": 0.85,
                    "winner": "best-2",
                    "difference": 0.04,
                }
            },
            "overall_winner": "best-2",
            "parameter_comparison": {},
        }

    monkeypatch.setattr("ml_lab_agent.services.agent_services.compare_experiments", _compare_experiments)
    monkeypatch.setattr(
        "ml_lab_agent.services.agent_services.generate_compare_summary",
        lambda comparison: {"summary": "ok", "comparison": comparison},
    )

    plan = AgentPlan(
        goal="Analyze latest experiment",
        steps=[
            AgentToolCall(tool="get_latest_run", args={}),
            AgentToolCall(tool="get_best_run_by_metric", args={"metric": "accuracy"}),
            AgentToolCall(tool="compare_runs", args={"left": "latest_run", "right": "best_run"}),
            AgentToolCall(tool="generate_summary", args={}),
        ],
    )

    result = execute_agent_plan(plan)

    assert result["latest_run"]["run_id"] == "latest-1"
    assert result["best_run"]["run_id"] == "best-2"
    assert result["comparison"]["overall_winner"] == "best-2"
    assert result["summary"]["summary"] == "ok"
    assert compare_calls == [["latest-1", "best-2"]]


def test_execute_agent_plan_resolves_literal_run_id(monkeypatch):
    monkeypatch.setattr(
        "ml_lab_agent.services.agent_services.resolve_single_run_identifier",
        lambda reference: reference,
    )
    monkeypatch.setattr(
        "ml_lab_agent.services.agent_services.select_run",
        lambda run_id: {"run_id": run_id, "metrics": {"f1_score": 0.72}, "params": {}},
    )
    monkeypatch.setattr(
        "ml_lab_agent.services.agent_services.show_best_run_by_metric",
        lambda metric: {
            "metric": metric,
            "best_run": {"run_id": "best-2", "metrics": {"f1_score": 0.91}, "params": {}},
            "best_value": 0.91,
            "num_runs_checked": 2,
        },
    )

    compare_calls = []

    def _compare_experiments(run_ids):
        compare_calls.append(run_ids)
        return {
            "compared_run_ids": run_ids,
            "metrics_comparison": {},
            "overall_winner": "best-2",
            "parameter_comparison": {},
        }

    monkeypatch.setattr("ml_lab_agent.services.agent_services.compare_experiments", _compare_experiments)
    monkeypatch.setattr(
        "ml_lab_agent.services.agent_services.generate_compare_summary",
        lambda comparison: {"summary": "ok", "comparison": comparison},
    )

    plan = AgentPlan(
        goal="Analyze a specific run",
        steps=[
            AgentToolCall(tool="get_best_run_by_metric", args={"metric": "f1_score"}),
            AgentToolCall(
                tool="compare_runs",
                args={"left": "da95ee9373604f3994e3ffa79b74749c", "right": "best_run"},
            ),
            AgentToolCall(tool="generate_summary", args={}),
        ],
    )

    result = execute_agent_plan(plan)

    assert result["comparison"]["overall_winner"] == "best-2"
    assert result["summary"]["summary"] == "ok"
    assert compare_calls == [["da95ee9373604f3994e3ffa79b74749c", "best-2"]]
