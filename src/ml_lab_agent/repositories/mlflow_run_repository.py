from mlflow.client import MlflowClient


class MlflowRunRepository:
    def __init__(self, client: MlflowClient, experiment_name: str):
        self.client = client
        self.experiment_name = experiment_name

    def list_runs(self) -> list[dict]:
        experiment_ids = self._get_experiment_ids(self.experiment_name)
        if not experiment_ids:
            return []

        runs = self.client.search_runs(experiment_ids=experiment_ids)
        return [self._map_run(run) for run in runs]

    def get_run(self, run_id: str) -> dict | None:
        try:
            run = self.client.get_run(run_id)
        except Exception:
            return None
        return self._map_run(run)

    def find_runs_by_name(self, run_name: str) -> str:
        experiment_ids = self._get_experiment_ids(self.experiment_name)
        if not experiment_ids:
            return []

        runs = self.client.search_runs(experiment_ids=experiment_ids, filter_string=f'attributes.run_name = "{run_name}"')
        return [self._map_run(run) for run in runs]

    def get_run_metrics(self, run_id: str) -> dict[str, float] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        return run["metrics"]

    def _map_run(self, run) -> dict:
        return {
            'run_id': run.info.run_id,
            'experiment_name': self.experiment_name,
            "experiment_id": run.info.experiment_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'metrics': dict(run.data.metrics),
            'params': dict(run.data.params),
            'tags': dict(run.data.tags),
        }

    def _get_experiment_ids(self, experiment_name: str) -> list[str]:
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []
        return [experiment.experiment_id]

    def find_runs_by_prefix(self, prefix: str) -> list[dict]:
        runs = self.list_runs()
        return [run for run in runs if run["run_id"].startswith(prefix)]
