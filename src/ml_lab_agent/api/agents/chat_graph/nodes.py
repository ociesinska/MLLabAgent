from ml_lab_agent.api.agents.chat_graph.state import State
from ml_lab_agent.schemas.chat_schemas import ChatResponse
from ml_lab_agent.schemas.exp_schemas import AmbiguousRunIdentifier
from ml_lab_agent.services.chat_service import detect_intent, extract_run_identifiers
from ml_lab_agent.services.exp_services import compare_experiments, resolve_run_identifiers, return_all_runs, select_run
from ml_lab_agent.services.llm_service import generate_compare_summary


def parse_input_node(state: State):
    intent = detect_intent(state["message"])
    raw_run_ids = extract_run_identifiers(state["message"])

    try:
        resolved_run_ids = resolve_run_identifiers(raw_run_ids) if raw_run_ids else []

        return {"intent": intent, "run_ids": resolved_run_ids}

    except AmbiguousRunIdentifier as e:
        return {
            "final_response": ChatResponse(
                intent=intent or "unknown",
                message=f"Run name '{e.run_identifier}' is ambiguous. Please choose one of the matching runs.",
                data={
                    "matches": [
                        {
                            "run_id": run["run_id"],
                            "run_name": run["tags"].get("mlflow.runName") or run["params"].get("run_name"),
                        }
                        for run in e.matches
                    ]
                },
                error="Ambiguous run identifier.",
            )
        }

    except ValueError as e:
        return {
            "intent": "unknown",
            "run_ids": [],
            "final_response": ChatResponse(
                intent=intent or "unknown",
                message="Cannot process this request.",
                data=None,
                error=str(e),
            ),
        }


def show_node(state: State):
    if len(state["run_ids"]) == 1:
        result = select_run(state["run_ids"][0])

        if result is None:
            return {
                "final_response": ChatResponse(
                    intent="show",
                    message=f"Run {state['run_ids'][0]} not found.",
                    data=None,
                    error=f"Run {state['run_ids'][0]} not found.",
                )
            }

        return {
            "final_response": ChatResponse(
                intent="show",
                message=f"Run {state['run_ids'][0]} found.",
                data=result,
                error=None,
            )
        }

    elif len(state["run_ids"]) == 0:
        result = return_all_runs()
        return {
            "final_response": ChatResponse(
                intent="show",
                message="Returning all runs.",
                data=result,
                error=None,
            )
        }

    else:
        return {
            "final_response": ChatResponse(
                intent="show",
                message="Only one run's details can be shown.",
                data=None,
                error="Only one run's details can be shown.",
            )
        }


def validate_compare_node(state: State):
    run_ids = state["run_ids"]

    if len(set(run_ids)) < 2:
        return {
            "final_response": ChatResponse(
                intent=state["intent"],
                message="Cannot process this request.",
                data=None,
                error="Need at least two unique runs to compare.",
            )
        }
    if len(set(run_ids)) > 2:
        return {
            "final_response": ChatResponse(
                intent=state["intent"],
                message="Cannot process this request.",
                data=None,
                error="Can only accept two unique runs to compare.",
            )
        }

    return {}


def route_after_validate(state: State):
    if state.get("final_response") is not None:
        return "end"
    if state["intent"] == "compare":
        return "compare_node"
    elif state["intent"] == "summarize_compare":
        return "compare_for_summary_node"
    return "end"


def compare_node(state: State):
    try:
        compare_results = compare_experiments(state["run_ids"])
        return {
            "final_response": ChatResponse(
                intent="compare",
                message="Comparison generated successfully.",
                data=compare_results,
                error=None,
            )
        }
    except ValueError as e:
        return {
            "final_response": ChatResponse(
                intent="compare",
                message="Cannot process this request.",
                data=None,
                error=str(e),
            )
        }


def compare_for_summary_node(state: State):
    try:
        compare_results = compare_experiments(state["run_ids"])
        return {
            "compare_results": compare_results,
            "llm_error": None,
        }
    except ValueError as e:
        return {
            "final_response": ChatResponse(
                intent="summarize_compare",
                message="Cannot process this request.",
                data=None,
                error=str(e),
            )
        }


def summarize_compare_node(state: State):
    if state["compare_results"] is None:
        return {
            "final_response": ChatResponse(
                intent="summarize_compare", message="Cannot process this request.", data=None, error="Missing compare_results in graph state."
            )
        }
    try:
        summary = generate_compare_summary(state["compare_results"])
        return {
            "final_response": ChatResponse(
                intent="summarize_compare",
                message="Comparison summary generated successfully.",
                data={
                    "compare_results": state["compare_results"],
                    "summary": summary,
                },
                error=None,
            ),
            "llm_error": None,
        }
    except Exception as e:
        return {"llm_error": str(e)}


def route_after_summary(state: State):
    if state.get("final_response") is not None:
        return "end_after_summary"
    if state.get("llm_error"):
        return "fallback_summary_node"
    return "end_after_summary"


def fallback_summary_node(state: State):
    compare_results = state["compare_results"]
    overall_winner = compare_results["overall_winner"]
    metric_names = list(compare_results["metrics_comparison"].keys())

    summary = {
        "summary": f"Run {overall_winner} performed better overall.",
        "metric_insights": [f"{metric} comparison was completed successfully." for metric in metric_names],
        "next_experiment_ideas": [
            "Try hyperparameter tuning to improve generalization.",
            "Test stronger augmentation to improve robustness.",
        ],
    }

    return {
        "final_response": ChatResponse(
            intent="summarize_compare",
            message="Comparison summary generated with fallback logic.",
            data={
                "compare_results": compare_results,
                "summary": summary,
            },
            error=None,
        )
    }


def unknown_node(state: State):
    if state.get("final_response") is not None:
        return {"final_response": state["final_response"]}

    return {
        "final_response": ChatResponse(
            intent="unknown",
            message="Cannot process this request.",
            data=None,
            error="Unsupported request.",
        )
    }


def route_by_intent(state: State):
    intent = state.get("intent", "unknown")
    if intent == "show":
        return "show_node"
    if intent == "compare":
        return "compare_path"
    elif intent == "summarize_compare":
        return "summarize_path"

    return "unknown_node"
