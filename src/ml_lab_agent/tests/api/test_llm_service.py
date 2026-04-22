from unittest.mock import Mock, patch

from ml_lab_agent.schemas.llm_schemas import CompareSummaryOutput
from ml_lab_agent.services.llm_service import generate_compare_summary


def test_generate_compare_summary_returns_valid_pydantic_object():
    compare_result = {
        "compared_run_ids": ["1", "2"],
        "metrics_comparison": {
            "accuracy": {
                "value_run_1": 0.81,
                "value_run_2": 0.85,
                "winner": "2",
                "difference": 0.04,
            },
            "f1_score": {
                "value_run_1": 0.78,
                "value_run_2": 0.82,
                "winner": "2",
                "difference": 0.04,
            },
        },
        "overall_winner": "2",
    }
    fake_response = Mock()
    fake_response.text = """
    {
        "summary": "Run 2 performed better overall. It improved both accuracy and f1-score.",
        "metric_insights": [
            "Accuracy improved by 0.04.",
            "F1-score improved by 0.04."
        ],
        "next_experiment_ideas": [
            "Try stronger augmentation because it may improve generalization.",
            "Tune learning rate because it may further improve convergence."
        ]
    }
    """

    with patch("ml_lab_agent.services.llm_service._get_client") as mock_get_client:
        mock_client = Mock()
        mock_client.models.generate_content.return_value = fake_response
        mock_get_client.return_value = mock_client

        result = generate_compare_summary(compare_result)

    assert isinstance(result, CompareSummaryOutput)
    assert result.summary.startswith("Run 2 performed better overall.")
    assert len(result.metric_insights) == 2
    assert len(result.next_experiment_ideas) == 2
    assert "augmentation" in result.next_experiment_ideas[0].lower()
