from pydantic import BaseModel


class CompareSummaryOutput(BaseModel):
    summary: str
    metric_insights: list[str]
    next_experiment_ideas: list[str]


class LLMResponseFormatError(Exception):
    pass


class LLMProviderError(Exception):
    pass
