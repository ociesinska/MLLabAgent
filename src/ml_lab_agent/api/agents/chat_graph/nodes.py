import logging

from ml_lab_agent.api.agents.chat_graph.state import State
from ml_lab_agent.schemas.chat_schemas import ChatResponse
from ml_lab_agent.schemas.exp_schemas import AmbiguousRunIdentifier
from ml_lab_agent.services.exp_services import compare_experiments, resolve_run_identifiers, return_all_runs, select_run, show_best_run_by_metric, show_latest_run
from ml_lab_agent.services.llm_service import generate_compare_summary
from ml_lab_agent.services.request_parser_service import parse_request
from ml_lab_agent.services.run_formatting_service import (
    format_run_for_response,
    format_runs_for_response
)

logger = logging.getLogger(__name__)

def parse_input_node(state: State):
    request_parsed = None

    try:
        logger.info("Parsing users message: %s", state["message"])

        request_parsed = parse_request(state["message"])
        raw_run_ids = request_parsed.run_identifiers

        logger.info(
            "Parsed request: intent=%s run_identifiers=%s metric=%s",
            request_parsed.intent,
            raw_run_ids,
            request_parsed.metric,
        )

        if request_parsed.intent == "show_best_run":
            return {
                "intent": request_parsed.intent,
                "run_ids": [],
                "metric": request_parsed.metric,
            }

        resolved_run_ids = resolve_run_identifiers(raw_run_ids) if raw_run_ids else []

        logger.info(
            "Resolved run identifiers: raw=%s resolved=%s",
            raw_run_ids,
            resolved_run_ids,
        )
        return {
            "intent": request_parsed.intent,
            "run_ids": resolved_run_ids,
            "metric": request_parsed.metric,
        }

    except AmbiguousRunIdentifier as e:
        intent = request_parsed.intent if request_parsed is not None else "unknown"

        logger.warning(
            "Ambiguous run identifier: identifier=%s matches_count=%s",
            e.run_identifier,
            len(e.matches),
        )

        return {
            "intent": intent,
            "run_ids": [],
            "final_response": ChatResponse(
                intent=intent,
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
            ),
        }

    except ValueError as e:
        intent = request_parsed.intent if request_parsed is not None else "unknown"

        logger.warning("Request parsing/resolution failed: %s", str(e))

        return {
            "intent": intent,
            "run_ids": [],
            "final_response": ChatResponse(
                intent=intent,
                message="Cannot process this request.",
                data=None,
                error=str(e),
            ),
        }

    except Exception as e:
        return {
            "intent": "unknown",
            "run_ids": [],
            "final_response": ChatResponse(
                intent="unknown",
                message="Cannot process this request.",
                data=None,
                error=f"Could not parse user request: {str(e)}",
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
                data=format_run_for_response(result),
                error=None,
            )
        }

    elif len(state["run_ids"]) == 0:
        result = return_all_runs()
        return {
            "final_response": ChatResponse(
                intent="show",
                message="Returning all runs.",
                data=format_runs_for_response(result),
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

    if len(run_ids) >= 2 and len(set(run_ids)) < 2:
        return {
            "final_response": ChatResponse(
                intent=state["intent"],
                message="Cannot process this request.",
                data=None,
                error="The selected references point to the same run, so there are not two unique runs to compare."
                )
        }

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

        logger.warning("Missing compare_results in summarize_compare_node.")

        return {
            "final_response": ChatResponse(
                intent="summarize_compare", message="Cannot process this request.", data=None, error="Missing compare_results in graph state."
            )
        }
    try:
        summary = generate_compare_summary(state["compare_results"])

        logger.info("LLM comparison summary generated successfully.")

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

        logger.warning("LLM summary generation failed, using fallback. Error: %s", str(e))

        return {"llm_error": str(e)}


def route_after_summary(state: State):
    if state.get("final_response") is not None:
        return "end_after_summary"
    if state.get("llm_error"):
        return "fallback_summary_node"
    return "end_after_summary"


def fallback_summary_node(state: State):

    logger.info("Using deterministic fallback summary.")
    
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


def show_best_run_node(state: State):
    metric = state.get("metric")

    if metric is None:
        return {
            "final_response": ChatResponse(
                intent="show_best_run",
                message="Cannot process this request.",
                data=None,
                error="Please specify a metric, for example: show best run by accuracy.",
            )
        }

    try:
        result = show_best_run_by_metric(metric)
        return {
            "final_response": ChatResponse(
                intent="show_best_run",
                message=f"Best run by {metric} found.",
                data=result,
                error=None,
            )
        }

    except ValueError as e:
        return {
            "final_response": ChatResponse(
                intent="show_best_run",
                message="Cannot process this request.",
                data=None,
                error=str(e),
            )
        }

def show_latest_run_node(state: State):
    try:
        result = show_latest_run()
        return {
            "final_response": ChatResponse(
                intent="show_latest_run",
                message="Latest run found.",
                data=result,
                error=None,
            )
        }

    except ValueError as e:
        return {
            "final_response": ChatResponse(
                intent="show_latest_run",
                message="Cannot process this request.",
                data=None,
                error=str(e),
            )
        }


def route_by_intent(state: State):

    if state.get("final_response") is not None:
        logger.info("Routing to unknown_node because final_response already exists.")
        return "unknown_node"

    intent = state.get("intent", "unknown")
    logger.info("Routing by intent: %s", intent)

    if intent == "show":
        return "show_node"
    if intent == "compare":
        return "compare_path"
    if intent == "summarize_compare":
        return "summarize_path"
    if intent == "show_best_run":
        return "show_best_run_node"
    if intent == "show_latest_run":
        return "show_latest_run_node"

    return "unknown_node"
