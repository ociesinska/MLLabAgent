from ml_lab_agent.data.dummy_data import data


def return_all_runs():
    return data


def select_run(run_id: str):
    for run in data:
        if run["run_id"] == run_id:
            return run
    return None


def get_run_metrics(run_id: str):
    for run in data:
        if run["run_id"] == run_id:
            metrics = run["metrics"]
            return metrics
    return None


def compare_experiments(run_ids: list[str]):
    unique_run_ids = list(dict.fromkeys(run_ids))
    if len(unique_run_ids) < 2:
        raise ValueError("Need at least two unique runs to compare.")
    if len(unique_run_ids) > 2:
        raise ValueError("Can only accept two unique runs to compare.")
    results = {}
    results["compared_run_ids"] = unique_run_ids

    metrics_by_run = {}
    for run_id in unique_run_ids:
        run = select_run(run_id)
        if run is None:
            raise ValueError(f"No such run id: {run_id}.")
        metrics_by_run[run_id] = run["metrics"]

    run_1_id, run_2_id = unique_run_ids
    metrics_1, metrics_2 = metrics_by_run.values()

    common_metrics = set(metrics_1.keys()) & set(metrics_2.keys())
    if len(common_metrics) == 0:
        raise ValueError("No common metrics to compare.")

    metrics_comparison = {}
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

        difference = abs(value_1 - value_2)

        metrics_comparison[metric_name] = {"value_run_1": value_1, "value_run_2": value_2, "winner": winner, "difference": difference}
    results["metrics_comparison"] = metrics_comparison
    if win_count[run_1_id] > win_count[run_2_id]:
        results["overall_winner"] = run_1_id
    elif win_count[run_2_id] > win_count[run_1_id]:
        results["overall_winner"] = run_2_id
    else:
        results["overall_winner"] = None

    return results
