import re

from ml_lab_agent.schemas.chat_schemas import ChatResponse
from ml_lab_agent.services.exp_services import compare_experiments, return_all_runs, select_run
from ml_lab_agent.services.llm_service import generate_compare_summary

# message -> intent detection -> parameter extraction -> call service -> optional llm formatting -> response


def detect_intent(message: str):
    message = message.lower()
    if "summary" in message or "summarize" in message or "analyze" in message:
        return "summarize_compare"
    if "compare" in message or "vs" in message:
        return "compare"
    if "show" in message or "details" in message:
        return "show"

    return "unknown"


def extract_run_ids(message: str) -> list[str]:
    return re.findall(r"\d+", message)


def process_request(message: str) -> ChatResponse:
    intent = detect_intent(message)
    run_ids = extract_run_ids(message)

    try:
        if intent == "compare":
            result = compare_experiments(run_ids)

            return ChatResponse(intent=intent, message="Comparison generated successfully", data=result, error=None)

        elif intent == "show":
            if len(run_ids) == 1:
                result = select_run(run_ids[0])
                if result is None:
                    return ChatResponse(
                        intent="show",
                        message=f"Run {run_ids[0]} not found.",
                        data=None,
                        error=f"Run {run_ids[0]} not found.",
                    )

                return ChatResponse(intent="show", message=f"Run {run_ids[0]} found.", data=result, error=None)

            elif len(run_ids) == 0:
                result = return_all_runs()

                return ChatResponse(intent="show", message="Returning all runs.", data=result, error=None)
            else:
                return ChatResponse(
                    intent="show", message="Only one run's details can be shown.", data=None, error="Only one run's details can be shown."
                )

        elif intent == "summarize_compare":
            compare_results = compare_experiments(run_ids)
            summary = generate_compare_summary(compare_results)

            return ChatResponse(
                intent="summarize_compare",
                message="Comparison summary generated successfully.",
                data={"compare_results": compare_results, "summary": summary},
                error=None,
            )

        return ChatResponse(
            intent="unknown",
            message="Cannot process this request.",
            data=None,
            error="Unsupported request.",
        )

    except ValueError as e:
        return ChatResponse(intent=intent, message="Cannot process this request.", data=None, error=str(e))
