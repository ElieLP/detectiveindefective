"""
Prediction module for NCR root cause analysis using DashScope.

Uses DashScope API to predict root causes for NCR descriptions.
Environment variables:
    - DASHSCOPE_API_KEY: API key for DashScope
    - DASHSCOPE_BASE_URL: Base URL for the DashScope API
"""

import os
from http import HTTPStatus
import pandas as pd
from dashscope import Generation


CSV_COLUMNS = [
    "part_type", "job_order", "operation_detection", "nc_description", "nc_code",
    "nominal", "lower_tolerance", "upper_tolerance", "measured_value", "defect_desc",
    "qc_comments", "machine_detection", "operator_detection", "date_detection",
    "operation_occurrence", "operator_machining", "machine_occurrence", "date_machining",
    "root_cause", "corrective_action"
]


def _check_api_key():
    """Verify DASHSCOPE_API_KEY is set."""
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise ValueError("DASHSCOPE_API_KEY environment variable not set")


def parse_ncr_csv_entry(csv_line: str, delimiter: str = ";") -> dict:
    """
    Parse a single CSV-formatted NCR entry into a dictionary.
    
    Args:
        csv_line: A semicolon-delimited CSV line
        delimiter: CSV delimiter (default: semicolon)
        
    Returns:
        Dictionary with NCR fields
    """
    values = csv_line.strip().split(delimiter)
    if len(values) != len(CSV_COLUMNS):
        raise ValueError(f"Expected {len(CSV_COLUMNS)} columns, got {len(values)}")
    
    return dict(zip(CSV_COLUMNS, values))


def build_prediction_prompt(ncr_data: dict) -> str:
    """
    Build a prompt for root cause prediction from NCR data.
    
    Args:
        ncr_data: Dictionary with NCR fields
        
    Returns:
        Formatted prompt string
    """
    return f"""Part: {ncr_data['part_type']} | Job: {ncr_data['job_order']}
NC Description: {ncr_data['nc_description']}
NC Code: {ncr_data['nc_code']}
Nominal: {ncr_data['nominal']} | Tolerance: [{ncr_data['lower_tolerance']}, {ncr_data['upper_tolerance']}] | Measured: {ncr_data['measured_value']}
Defect: {ncr_data['defect_desc']}
QC Comments: {ncr_data['qc_comments']}
Machine (detection): {ncr_data['machine_detection']} | Machine (occurrence): {ncr_data['machine_occurrence']}
Operation (occurrence): {ncr_data['operation_occurrence']}"""


def predict_root_cause(description: str, model: str = "qwen-plus") -> str:
    """
    Predict the root cause for a single NCR description.
    
    Args:
        description: The NCR description text
        model: Model name to use
        
    Returns:
        Predicted root cause as a string
    """
    _check_api_key()
    
    messages = [
        {"role": "system", "content": "You are a manufacturing quality expert specializing in root cause analysis."},
        {"role": "user", "content": f"""You are an expert in manufacturing quality analysis. 
Based on the following NCR (Non-Conformance Report) description, predict the most likely root cause.
Be concise and specific. Focus on the technical root cause.

NCR Description:
{description}

Root Cause:"""}
    ]

    response = Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        temperature=0.3,
        max_tokens=200
    )
    
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope API error: {response.code} - {response.message}")
    
    return response.output.choices[0].message.content.strip()


def predict_root_cause_from_csv(csv_line: str, model: str = "qwen-plus") -> str:
    """
    Predict root cause from a CSV-formatted NCR entry.
    
    Args:
        csv_line: A semicolon-delimited CSV line
        model: Model name to use
        
    Returns:
        Predicted root cause as a string
    """
    ncr_data = parse_ncr_csv_entry(csv_line)
    prompt = build_prediction_prompt(ncr_data)
    return predict_root_cause(prompt, model=model)


def predict_root_causes_batch(df: pd.DataFrame, description_col: str = "description", model: str = "qwen-plus") -> pd.DataFrame:
    """
    Predict root causes for all NCRs in a dataframe.
    
    Args:
        df: DataFrame containing NCR data
        description_col: Name of the column containing descriptions
        model: Model name to use
        
    Returns:
        DataFrame with added 'predicted_root_cause' column
    """
    _check_api_key()
    result_df = df.copy()
    
    predictions = []
    for idx, row in df.iterrows():
        description = row[description_col]
        try:
            prediction = predict_root_cause(description, model=model)
            predictions.append(prediction)
        except Exception as e:
            predictions.append(f"Error: {str(e)}")
    
    result_df["predicted_root_cause"] = predictions
    return result_df
