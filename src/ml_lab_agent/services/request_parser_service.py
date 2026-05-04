import json
from json import JSONDecodeError

from pydantic import ValidationError

from ml_lab_agent.schemas.chat_schemas import ParsedUserRequest
from ml_lab_agent.services.llm_service import LLMProviderError, LLMResponseFormatError, _get_client


def _provider_error_message(exc: Exception) -> str:
    raw = str(exc)
    normalized = raw.upper()

    if "RESOURCE_EXHAUSTED" in normalized or ("429" in normalized and "QUOTA" in normalized):
        return "Gemini API quota exceeded (429 RESOURCE_EXHAUSTED). Check quota/billing and retry later."

    return "LLM provider unavailable."


def parse_request(message: str) -> ParsedUserRequest:
    serialized_message = json.dumps(message)
    prompt = (
        """
    You are an ML experiment assistant request parser.

    Parse the user message into a structured JSON object.

    Return ONLY valid JSON.
    Do not use markdown.
    Do not wrap the response in triple backticks.


    Allowed intents:
    - show
    - compare
    - summarize_compare
    - show_best_run
    - show_latest_run
    - agent_analyze
    - unknown

    Rules:
    - Use "show" when the user wants details about one or more runs.
    - Use "compare" when the user wants to compare runs.
    - Use "summarize_compare" when the user wants an analysis/summary of compared runs.
    - Use "show_best_run" when the user asks for the best run by a metric.
    - Use "show_latest_run" when the user asks for the latest, newest, most recent, or last run.
    - Use "agent_analyze" when the user asks to analyze, investigate, review, or recommend next steps based on experiments.
    - Use "unknown" if the request is not related to experiment runs.

    If the user refers to "latest run", use run identifier "latest".
    If the user refers to "best run by <metric>", use run identifier "best_by:<metric>".

    Return exactly this structure:
    {
    "intent": "show | compare | summarize_compare | show_best_run | unknown",
    "run_identifiers": ["string"],
    "metric": "string or null"
    }
    
    Never invent run identifiers.
    Only include run_identifiers that are explicitly present in the user message.
    If the user asks for the best run by a metric, use intent "show_best_run", set metric to the metric name, and set run_identifiers to [].

    Examples:
    User: show run baseline_lr
    Output: {"intent": "show", "run_identifiers": ["baseline_lr"], "metric": null}

    User: compare baseline_lr and tuned_cnn
    Output: {"intent": "compare", "run_identifiers": ["baseline_lr", "tuned_cnn"], "metric": null}

    User: summarize comparison of baseline_lr and tuned_cnn
    Output: {"intent": "summarize_compare", "run_identifiers": ["baseline_lr", "tuned_cnn"], "metric": null}

    User: show best run by accuracy
    Output: {"intent": "show_best_run", "run_identifiers": [], "metric": "accuracy"}

    User: show me best run by accuracy
    Output: {"intent": "show_best_run", "run_identifiers": [], "metric": "accuracy"}

    User: show latest run
    Output: {"intent": "show_latest_run", "run_identifiers": [], "metric": null}

    User: compare latest run with best run by f1_score
    Output: {"intent": "compare", "run_identifiers": ["latest", "best_by:f1_score"], "metric": null}

    User: summarize latest run vs best run by accuracy
    Output: {"intent": "summarize_compare", "run_identifiers": ["latest", "best_by:accuracy"], "metric": null}

    User: analyze latest experiment
    Output: {"intent": "agent_analyze", "run_identifiers": [], "metric": null}

    User: analyze latest experiment and recommend next step
    Output: {"intent": "agent_analyze", "run_identifiers": [], "metric": null}

    User message:
    """
        + serialized_message
    )
    try:
        client = _get_client()
        response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    except Exception as e:
        raise LLMProviderError(_provider_error_message(e)) from e

    try:
        parsed = json.loads(response.text)
        return ParsedUserRequest.model_validate(parsed)
    except (JSONDecodeError, ValidationError) as e:
        raise LLMResponseFormatError("Invalid LLM response format.") from e
