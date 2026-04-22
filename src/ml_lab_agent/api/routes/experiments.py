from fastapi import APIRouter, HTTPException

from ml_lab_agent.schemas.exp_schemas import CompareRequest, CompareResponse, CompareSummaryResponse, RunSummary
from ml_lab_agent.services.exp_services import compare_experiments, return_all_runs, select_run
from ml_lab_agent.services.llm_service import (
    LLMProviderError,
    LLMResponseFormatError,
    generate_compare_summary,
)

experiment_router = APIRouter()


@experiment_router.get("/experiments/runs", response_model=list[RunSummary])
def list_runs():
    return return_all_runs()


@experiment_router.get("/experiments/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: str):
    run = select_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run with id {run_id} not found")
    return run


@experiment_router.post("/experiments/compare", response_model=CompareResponse)
def post_compare_experiments(request: CompareRequest):
    try:
        return compare_experiments(request.run_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@experiment_router.post("/experiments/compare-summary", response_model=CompareSummaryResponse)
def compare_summary(request: CompareRequest):
    try:
        compare_results = compare_experiments(request.run_ids)
        generated_summary = generate_compare_summary(compare_results)
        return {"compare_results": compare_results, "generated_summary": generated_summary}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except LLMResponseFormatError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except LLMProviderError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
