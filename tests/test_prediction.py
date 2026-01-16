"""
Acceptance tests for the prediction module.
"""

import pytest
from http import HTTPStatus
from unittest.mock import Mock, patch

from src.prediction import (
    parse_ncr_csv_entry,
    build_prediction_prompt,
    predict_root_cause_from_csv,
    CSV_COLUMNS,
)


EXAMPLE_CSV_LINE = "AA1;AA1_003884;7200;CO2910 is non conforming;CO2910;120,2;-0,6;0,6;0;OP7200 DA;after OP7300, awaiting QA confirmation whether to conduct a retes, 2025.12.23 DA updated: FCTD see attachment. 2025.12.23;CCCC-04-01;EM1782;12/23/25;1200;2/13/10;AAAA-02-06;12/13/25;NOT DETERMINED YET;Add manual tool calibration"


class TestParseNcrCsvEntry:
    """Tests for CSV parsing."""

    def test_parse_valid_csv_line(self):
        """Parse a valid CSV line into a dictionary."""
        result = parse_ncr_csv_entry(EXAMPLE_CSV_LINE)
        
        assert result["part_type"] == "AA1"
        assert result["job_order"] == "AA1_003884"
        assert result["nc_description"] == "CO2910 is non conforming"
        assert result["nc_code"] == "CO2910"
        assert result["nominal"] == "120,2"
        assert result["measured_value"] == "0"
        assert result["root_cause"] == "1. NOT DETERMINED YET"
        assert result["corrective_action"] == "Add manual tool calibration"

    def test_parse_returns_all_columns(self):
        """Parsed result contains all expected columns."""
        result = parse_ncr_csv_entry(EXAMPLE_CSV_LINE)
        
        assert len(result) == len(CSV_COLUMNS)
        for col in CSV_COLUMNS:
            assert col in result

    def test_parse_invalid_column_count_raises(self):
        """Raise error if column count doesn't match."""
        invalid_line = "AA1;AA1_003884;7200"
        
        with pytest.raises(ValueError, match="Expected 20 columns"):
            parse_ncr_csv_entry(invalid_line)


class TestBuildPredictionPrompt:
    """Tests for prompt building."""

    def test_prompt_contains_key_fields(self):
        """Prompt includes essential NCR information."""
        ncr_data = parse_ncr_csv_entry(EXAMPLE_CSV_LINE)
        prompt = build_prediction_prompt(ncr_data)
        
        assert "AA1" in prompt
        assert "CO2910 is non conforming" in prompt
        assert "CO2910" in prompt
        assert "120,2" in prompt
        assert "CCCC-04-01" in prompt
        assert "AAAA-02-06" in prompt

    def test_prompt_excludes_root_cause(self):
        """Prompt should not include the root cause (we want to predict it)."""
        ncr_data = parse_ncr_csv_entry(EXAMPLE_CSV_LINE)
        prompt = build_prediction_prompt(ncr_data)
        
        assert "NOT DETERMINED YET" not in prompt


class TestPredictRootCauseFromCsv:
    """Acceptance tests for root cause prediction."""

    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"})
    @patch("src.prediction.Generation")
    def test_predict_from_csv_returns_prediction(self, mock_generation):
        """Predict root cause from CSV entry."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output.choices = [Mock(message=Mock(content="Tool calibration deviation on AAAA-02-09"))]
        mock_generation.call.return_value = mock_response
        
        result = predict_root_cause_from_csv(EXAMPLE_CSV_LINE)
        
        assert result == "Tool calibration deviation on AAAA-02-09"
        mock_generation.call.assert_called_once()

    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"})
    @patch("src.prediction.Generation")
    def test_predict_sends_correct_prompt_structure(self, mock_generation):
        """Verify the API is called with correct prompt structure."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output.choices = [Mock(message=Mock(content="Test prediction"))]
        mock_generation.call.return_value = mock_response
        
        predict_root_cause_from_csv(EXAMPLE_CSV_LINE)
        
        call_args = mock_generation.call.call_args
        messages = call_args.kwargs["messages"]
        
        assert messages[0]["role"] == "system"
        assert "quality expert" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert "CO2910" in messages[1]["content"]
