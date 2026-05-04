from mlflow.client import MlflowClient

from ml_lab_agent.repositories.mlflow_run_repository import MlflowRunRepository
from ml_lab_agent.schemas.exp_schemas import AmbiguousRunIdentifier
from ml_lab_agent.config.config import get_settings

settings = get_settings()

client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
repository = MlflowRunRepository(client, experiment_name=settings.mlflow_experiment_name)


def return_all_runs():
    return repository.list_runs()


def select_run(run_id: str):
    return repository.get_run(run_id)


def show_latest_run() -> dict:
    runs = return_all_runs()

    if not runs:
        raise ValueError("No runs found.")

    runs_with_start_time = [run for run in runs if run.get("start_time") is not None]

    if not runs_with_start_time:
        raise ValueError("No runs with start_time found.")

    latest_run = max(runs_with_start_time, key=lambda run: run["start_time"])

    return latest_run


def get_run_metrics(run_id: str):
    return repository.get_run_metrics(run_id)


def resolve_run_id(candidate: str) -> str:
    # Full run_id provided.
    run = repository.get_run(candidate)
    if run is not None:
        return candidate

    # run_id prefix provided
    matches = repository.find_runs_by_prefix(candidate)

    if len(matches) == 0:
        raise ValueError(f"No run found for identifier: {candidate}.")
    elif len(matches) > 1:
        matched_ids = [run["run_id"] for run in matches]
        raise ValueError(f"Run identifier '{candidate}' is ambiguous. Matches: {matched_ids}")
    return matches[0]["run_id"]


def resolve_run_ids(candidates: list[str]) -> list[str]:
    return [resolve_run_id(candidate) for candidate in candidates]


def resolve_single_run_identifier(run_identifier: str) -> str:

    if run_identifier == "latest":
        latest_run = show_latest_run()
        return latest_run["run_id"]
    if run_identifier.startswith("best_by:"):
        metric = run_identifier.split(":", 1)[1]
        best_result = show_best_run_by_metric(metric)
        return best_result["best_run"]["run_id"]

    try:
        return resolve_run_id(run_identifier)
    except ValueError:
        pass

    runs_by_name = repository.find_runs_by_name(run_identifier)
    if len(runs_by_name) == 0:
        raise ValueError(f"No run found for identifier: {run_identifier}.")
    elif len(runs_by_name) > 1:
        raise AmbiguousRunIdentifier(run_identifier, runs_by_name)
    return runs_by_name[0]["run_id"]


def resolve_run_identifiers(run_identifiers: list[str]) -> list[str]:
    run_ids = []
    for run_identifier in run_identifiers:
        resolved_id = resolve_single_run_identifier(run_identifier)
        run_ids.append(resolved_id)
    return run_ids


def compare_experiments(run_ids: list[str]):
    unique_run_ids = list(dict.fromkeys(run_ids))
    if len(unique_run_ids) < 2:
        raise ValueError("Need at least two unique runs to compare.")
    if len(unique_run_ids) > 2:
        raise ValueError("Can only accept two unique runs to compare.")
    results = {}
    results["compared_run_ids"] = unique_run_ids

    metrics_by_run = {}
    params_by_run = {}

    for run_id in unique_run_ids:
        run = select_run(run_id)
        if run is None:
            raise ValueError(f"No such run id: {run_id}.")
        metrics_by_run[run_id] = run["metrics"]
        params_by_run[run_id] = run.get("params", {})

    run_1_id, run_2_id = unique_run_ids

    metrics_1 = metrics_by_run[run_1_id]
    params_1 = params_by_run[run_1_id]

    metrics_2 = metrics_by_run[run_2_id]
    params_2 = params_by_run[run_2_id]

    common_metrics = set(metrics_1.keys()) & set(metrics_2.keys())
    common_params = set(params_1.keys()) & set(params_2.keys())

    if len(common_metrics) == 0:
        raise ValueError("No common metrics to compare.")

    metrics_comparison = {}
    parameter_comparison = {}
    win_count = {run_1_id: 0, run_2_id: 0}

    for metric_name in common_metrics:
        value_1 = metrics_1[metric_name]
        value_2 = metrics_2[metric_name]

        if value_1 > value_2:
            winner = run_1_id
            win_count[run_1_id] += 1
        elif value_1 < value_2:
            winner = run_2_id
            win_count[run_2_id] += 1
        else:
            winner = None

        metrics_comparison[metric_name] = {
            "value_run_1": value_1,
            "value_run_2": value_2,
            "winner": winner,
            "difference": abs(value_1 - value_2),
        }

    results["metrics_comparison"] = metrics_comparison

    if win_count[run_1_id] > win_count[run_2_id]:
        results["overall_winner"] = run_1_id
    elif win_count[run_2_id] > win_count[run_1_id]:
        results["overall_winner"] = run_2_id
    else:
        results["overall_winner"] = None

    for param in common_params:
        val_1 = params_1[param]
        val_2 = params_2[param]

        parameter_comparison[param] = {"value_run_1": val_1, "value_run_2": val_2, "changed": params_1[param] != params_2[param]}

    results["parameter_comparison"] = parameter_comparison

    return results


METRIC_DIRECTIONS = {
    "accuracy": "max",
    "f1_score": "max",
    "precision": "max",
    "recall": "max",
    "loss": "min",
    "val_loss": "min",
}


def show_best_run_by_metric(metric: str):
    runs = return_all_runs()

    runs_with_metric = [run for run in runs if metric in run["metrics"]]

    if not runs_with_metric:
        raise ValueError(f"No runs with metric: {metric}.")

    direction = METRIC_DIRECTIONS.get(metric, "max")

    if direction == "min":
        best_run = min(runs_with_metric, key=lambda run: run["metrics"][metric])
    else:
        best_run = max(runs_with_metric, key=lambda run: run["metrics"][metric])

    return {
        "metric": metric,
        "best_run": best_run,
        "best_value": best_run["metrics"][metric],
        "num_runs_checked": len(runs_with_metric),
    }
