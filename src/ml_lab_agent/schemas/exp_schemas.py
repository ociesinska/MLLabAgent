from pydantic import BaseModel

class RunSummary(BaseModel):
    run_id: str
    experiment_name: str
    metrics: dict[str, float]


class CompareRequest(BaseModel):
    run_ids: list[str]


class MetricsComparison(BaseModel):
    value_run_1: float
    value_run_2: float
    winner: str | None
    difference: float


class CompareResponse(BaseModel):
    compared_run_ids: list[str]
    metrics_comparison: dict[str, MetricsComparison]
    overall_winner: str | None
