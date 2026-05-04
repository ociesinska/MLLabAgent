def format_run_for_response(run: dict) -> dict:
    return {
        "run_id": run.get("run_id"),
        "experiment_name": run.get("experiment_name"),
        "run_name": run.get("run_name") or run.get("tags", {}).get("mlflow.runName"),
        "status": run.get("status"),
        "start_time": run.get("start_time"),
        "end_time": run.get("end_time"),
        "metrics": run.get("metrics", {}),
        "params": run.get("params", {}),
    }


def format_runs_for_response(runs: list[dict]) -> list[dict]:
    return [format_run_for_response(run) for run in runs]
