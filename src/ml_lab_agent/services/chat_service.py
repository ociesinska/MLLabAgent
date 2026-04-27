import re

# message -> intent detection -> parameter extraction -> call service -> optional llm formatting -> response


def detect_intent(message: str):
    message = message.lower()
    if "best run" in message:
        return "show_best_run"
    if "summary" in message or "summarize" in message or "analyze" in message:
        return "summarize_compare"
    if "compare" in message or "vs" in message:
        return "compare"
    if "show" in message or "details" in message:
        return "show"

    return "unknown"


def extract_run_identifiers(message: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', message)
    if quoted:
        return quoted
    return re.findall(r"\brun\s+(?!by\b)([a-zA-Z0-9_\-]+)\b", message, flags=re.IGNORECASE)


def extract_metric_from_message(message: str) -> str | None:
    match = re.search(r"\bby\s+([a-zA-Z0-9_\-]+)\b", message, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    return None
