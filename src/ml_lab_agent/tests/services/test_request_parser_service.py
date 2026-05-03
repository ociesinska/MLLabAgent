from unittest.mock import Mock, patch

import pytest

from ml_lab_agent.schemas.chat_schemas import ParsedUserRequest
from ml_lab_agent.services.llm_service import LLMProviderError, LLMResponseFormatError
from ml_lab_agent.services.request_parser_service import parse_request


def test_parse_request_parses_show_best_run():
    fake_response = Mock()
    fake_response.text = """
    {
        "intent": "show_best_run",
        "run_identifiers": [],
        "metric": "f1_score"
    }
    """
    with patch("ml_lab_agent.services.request_parser_service._get_client") as mock_get_client:
        mock_client = Mock()
        mock_client.models.generate_content.return_value = fake_response
        mock_get_client.return_value = mock_client

        result = parse_request("show me best run by f1_score")

    assert isinstance(result, ParsedUserRequest)
    assert result.intent == "show_best_run"
    assert result.run_identifiers == []
    assert result.metric == "f1_score"


def test_parse_request_raises_for_invalid_json():
    fake_response = Mock()
    fake_response.text = "this is not a valid json"

    with patch("ml_lab_agent.services.request_parser_service._get_client") as mock_get_client:
        mock_client = Mock()
        mock_client.models.generate_content.return_value = fake_response
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMResponseFormatError):
            parse_request("show me best run by f1_score")


def test_parse_request_raises_for_invalid_schema():
    fake_response = Mock()
    fake_response.text = """
    {
        "intent": "some_wrong_intent",
        "run_identifiers": [],
        "metric": "accuracy"
    }
    """

    with patch("ml_lab_agent.services.request_parser_service._get_client") as mock_get_client:
        mock_client = Mock()
        mock_client.models.generate_content.return_value = fake_response
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMResponseFormatError):
            parse_request("show me best run by f1_score")


def test_parse_request_raises_provider_error_when_llm_call_fails():
    mock_client = Mock()
    mock_client.models.generate_content.side_effect = Exception("Provider timeout")

    with patch("ml_lab_agent.services.request_parser_service._get_client", return_value=mock_client):
        with pytest.raises(LLMProviderError, match="LLM provider unavailable"):
            parse_request("show me best run by f1_score")
