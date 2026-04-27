from ml_lab_agent.data.dummy_data import data


class DummyRunRepository:
    def list_runs(self) -> list[dict]:
        return data

    def get_run(self, run_id: str):
        for run in data:
            if run["run_id"] == run_id:
                return run
        return None

    def get_run_metrics(self, run_id: str) -> dict[str, float] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        return run["metrics"]
