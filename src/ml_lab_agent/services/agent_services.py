from ml_lab_agent.schemas.agent_schemas import AgentPlan
from ml_lab_agent.services.llm_service import _get_client
from ml_lab_agent.config.config import get_settings
import json
from json import JSONDecodeError
from pydantic import ValidationError
from ml_lab_agent.schemas.llm_schemas import LLMProviderError, LLMResponseFormatError
from ml_lab_agent.services.exp_services import (
    show_latest_run,
    show_best_run_by_metric,
    compare_experiments,
    resolve_single_run_identifier,
    select_run,
)
from ml_lab_agent.services.llm_service import generate_compare_summary


def _resolve_run_reference(context: dict, reference: str) -> dict | None:
    if reference in context:
        return context[reference]

    resolved_run_id = resolve_single_run_identifier(reference)
    return select_run(resolved_run_id)


def create_agent_plan(message: str) -> AgentPlan:
    client = _get_client()
    settings = get_settings()

    serialized_message = json.dumps(message, ensure_ascii=False)

    prompt = """
    You are an ML experiment analysis planner.

    Your task is to convert the user request into a safe, structured plan.

    Return ONLY valid JSON.
    Do not use markdown.
    Do not wrap the response in triple backticks.
    Do not include any text before or after the JSON.

    Available tools:
    - get_latest_run
    Description: Get the most recent MLflow run.
    Args: {}

    - get_best_run_by_metric
    Description: Get the best MLflow run according to a metric.
    Args: {"metric": "string"}

    - compare_runs
    Description: Compare two runs by context references or user-provided run IDs.
    Args: {"left": "latest_run | best_run | user-provided run ID", "right": "latest_run | best_run | user-provided run ID"}

    - generate_summary
    Description: Generate an LLM summary based on comparison results.
    Args: {}

    Rules:
    - Use only the available tools.
    - Do not invent tool names.
    - Do not invent run IDs.
    - If the user provides a run ID, use that exact run ID as a compare_runs reference.
    - For "best run", use metric "f1_score" unless the user explicitly specifies another metric.
    - If the user asks to analyze the latest experiment, compare the latest run with the best run by f1_score.
    - Always generate a summary after comparing runs.
    - Keep the plan short and practical.

    Return exactly this JSON structure:
    {
    "goal": "string",
    "steps": [
        {
        "tool": "tool_name",
        "args": {}
        }
    ]
    }

    Examples:

    User: analyze latest experiment
    Output:
    {
    "goal": "Analyze the latest experiment against the current best run.",
    "steps": [
        {"tool": "get_latest_run", "args": {}},
        {"tool": "get_best_run_by_metric", "args": {"metric": "f1_score"}},
        {"tool": "compare_runs", "args": {"left": "latest_run", "right": "best_run"}},
        {"tool": "generate_summary", "args": {}}
    ]
    }

    User: analyze latest experiment by accuracy
    Output:
    {
    "goal": "Analyze the latest experiment against the best run by accuracy.",
    "steps": [
        {"tool": "get_latest_run", "args": {}},
        {"tool": "get_best_run_by_metric", "args": {"metric": "accuracy"}},
        {"tool": "compare_runs", "args": {"left": "latest_run", "right": "best_run"}},
        {"tool": "generate_summary", "args": {}}
    ]
    }

    User: analyze run da95ee9373604f3994e3ffa79b74749c and recommend next steps
    Output:
    {
    "goal": "Analyze the specified run against the current best run.",
    "steps": [
        {"tool": "get_best_run_by_metric", "args": {"metric": "f1_score"}},
        {"tool": "compare_runs", "args": {"left": "da95ee9373604f3994e3ffa79b74749c", "right": "best_run"}},
        {"tool": "generate_summary", "args": {}}
    ]
    }

    User message:
    """ + serialized_message

    try: 
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt
        )
    except Exception as e:
        raise LLMProviderError("LLM provider unavailable.") from e
    
    try:
        raw_text = response.text
        parsed = json.loads(raw_text)
        return AgentPlan.model_validate(parsed)
    except (JSONDecodeError, ValidationError) as e:
        raise LLMResponseFormatError("Invalid LLM response format.") from e

def execute_agent_plan(plan: AgentPlan) -> dict:
    context = {}

    for step in plan.steps:
        if step.tool == "get_latest_run":
            context["latest_run"] = show_latest_run()
        elif step.tool == "get_best_run_by_metric":
            metric = step.args.get("metric", "f1_score")
            best_run_result = show_best_run_by_metric(metric)
            context["best_run"] = best_run_result["best_run"]
            context["best_run_metric"] = metric
        elif step.tool == "compare_runs":
            left_ref = step.args["left"]
            right_ref = step.args["right"]

            left_run = _resolve_run_reference(context, left_ref)
            right_run = _resolve_run_reference(context, right_ref)

            if left_run is None or right_run is None:
                raise ValueError(f"Missing run reference in agent plan: {left_ref} or {right_ref}.")

            context["comparison"] = compare_experiments([left_run["run_id"], right_run["run_id"]])

        elif step.tool == "generate_summary":
            if context.get("comparison") is None:
                raise ValueError("Missing comparison in agent plan context.")
            context["summary"] = generate_compare_summary(context["comparison"])

        else:
            raise ValueError(f"Unsupported tool in agent plan: {step.tool}")

    return context
