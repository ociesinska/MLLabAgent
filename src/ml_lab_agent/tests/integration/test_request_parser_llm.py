import os

import pytest

from ml_lab_agent.services.request_parser_service import parse_request


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No Gemini API key configured.")
def test_parse_request_real_llm_smoke():
    result = parse_request("show me best run by f1_score")

    assert result.intent == "show_best_run"
    assert result.metric == "f1_score"
