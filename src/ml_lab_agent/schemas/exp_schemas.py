from pydantic import BaseModel

from ml_lab_agent.schemas.llm_schemas import CompareSummaryOutput


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


class ParamsComparison(BaseModel):
    value_run_1: str | int | float | bool | None
    value_run_2: str | int | float | bool | None
    changed: bool


class CompareResponse(BaseModel):
    compared_run_ids: list[str]
    metrics_comparison: dict[str, MetricsComparison]
    parameter_comparison: dict[str, ParamsComparison]
    overall_winner: str | None


class CompareSummaryResponse(BaseModel):
    compare_results: CompareResponse
    generated_summary: CompareSummaryOutput


class AmbiguousRunIdentifier(ValueError):
    def __init__(self, run_identifier: str, matches: list[dict]):
        self.run_identifier = run_identifier
        self.matches = matches
        super().__init__(f"Run name '{run_identifier}' is ambiguous.")
