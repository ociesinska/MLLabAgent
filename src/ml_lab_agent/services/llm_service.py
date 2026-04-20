import os

from google import genai
import json
from ml_lab_agent.schemas.llm_schemas import CompareSummaryOutput

def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def generate_compare_summary(compare_result: dict) -> CompareSummaryOutput:
    client = _get_client()
    prompt = f"""
    You are an ML experimentation assistant.

    Your task is to analyze the experiment comparison result and return a concise structured response.

    Return ONLY valid JSON.
    Do not use markdown.
    Do not wrap the response in triple backticks.
    Do not include any text before or after the JSON.

    The JSON must have exactly this structure:
    {{
    "summary": "string",
    "metric_insights": ["string", "string"],
    "next_experiment_ideas": ["string", "string"]
    }}

    Rules:
    - "summary" must be 2-3 sentences.
    - "metric_insights" must contain one short bullet-style insight per compared metric.
    - "next_experiment_ideas" must contain exactly 2 concrete next steps for a classification project.
    - Each experiment idea must be specific, realistic, and explain why it may help.
    - If the comparison result is limited, still provide 2 reasonable generic next experiment ideas.
    - Keep the response concise and practical.
    - Use the provided comparison result only. Do not invent nonexistent metrics.

    Experiment comparison result:
    {compare_result}
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    raw_text = response.text
    parsed = json.loads(raw_text)

    return CompareSummaryOutput.model_validate(parsed)
