"""
Prediction module for NCR root cause analysis using DashScope.

Uses DashScope API to predict root causes for NCR descriptions.
Environment variables:
    - DASHSCOPE_API_KEY: API key for DashScope
"""

from http import HTTPStatus
import pandas as pd
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role


def load_context_data(filepath: str = 'data/prod_data_enriched.csv') -> pd.DataFrame:
    """Load enriched NCR data as context for predictions."""
    return pd.read_csv(filepath, sep=';')


def load_input_data(filepath: str) -> pd.DataFrame:
    """Load input CSV with empty root cause field."""
    return pd.read_csv(filepath, sep=';')


def build_context_prompt(context_df: pd.DataFrame) -> str:
    """Build context from historical NCR data with known root causes."""
    rows_with_root_cause = context_df[
        context_df['Root cause of occurrence'].notna() & 
        (context_df['Root cause of occurrence'] != '')
    ]
    
    examples = []
    for _, row in rows_with_root_cause.iterrows():
        example = f"""NC Code: {row.get('NC Code', '')}
NC Description: {row.get('NC description', '')}
Part Type: {row.get('Part type', '')}
Machine of Occurrence: {row.get('MachineNum of occurrence', '')}
Defect Description: {row.get('FDefectDesc_EN', '')}
Root Cause: {row.get('Root cause of occurrence', '')}
Corrective Action: {row.get('Corrective actions', '')}
---"""
        examples.append(example)
    
    return "\n".join(examples)


def build_prediction_prompt(row: pd.Series, context: str) -> str:
    """Build prompt for predicting root cause of a single NCR."""
    return f"""You are an expert in manufacturing quality control and root cause analysis.

Based on the following historical NCR (Non-Conformance Report) data:

{context}

Predict the most likely root cause for this new NCR:

NC Code: {row.get('NC Code', '')}
NC Description: {row.get('NC description', '')}
Part Type: {row.get('Part type', '')}
Machine of Detection: {row.get('MachineNum of detection', '')}
Machine of Occurrence: {row.get('MachineNum of occurrence', '')}
Operation of Detection: {row.get('Operation number of detection', '')}
Operation of Occurrence: {row.get('Operation number of occurrence', '')}
Defect Description: {row.get('FDefectDesc_EN', '')}
QC Comments: {row.get('Fqccomments_EN', '')}
Corrective Actions: {row.get('Corrective actions', '')}

Respond with ONLY the predicted root cause (a short phrase). Do not include explanations."""


def predict_root_cause(row: pd.Series, context: str) -> str:
    """Predict root cause for a single NCR row using DashScope."""
    prompt = build_prediction_prompt(row, context)
    
    messages = [
        {'role': Role.SYSTEM, 'content': 'You are an expert in manufacturing quality control and root cause analysis.'},
        {'role': Role.USER, 'content': prompt}
    ]
    
    response = Generation.call(
        model='qwen-plus',
        messages=messages,
        result_format='message',
        temperature=0.3,
        max_tokens=100
    )
    
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content.strip()
    else:
        raise Exception(f"DashScope error: {response.code} - {response.message}")


def predict_from_csv(
    input_filepath: str,
    context_filepath: str = 'data/prod_data_enriched.csv',
    output_filepath: str = None
) -> pd.DataFrame:
    """
    Predict root causes for NCRs in input CSV using context data.
    
    Args:
        input_filepath: Path to CSV with empty root cause field
        context_filepath: Path to enriched CSV with historical data
        output_filepath: Optional path to save results
    
    Returns:
        DataFrame with predicted root causes
    """
    context_df = load_context_data(context_filepath)
    context = build_context_prompt(context_df)
    
    input_df = load_input_data(input_filepath)
    
    predictions = []
    for idx, row in input_df.iterrows():
        root_cause = row.get('Root cause of occurrence', '')
        if pd.isna(root_cause) or root_cause == '':
            predicted = predict_root_cause(row, context)
            predictions.append(predicted)
        else:
            predictions.append(root_cause)
    
    input_df['Root cause of occurrence'] = predictions
    
    if output_filepath:
        input_df.to_csv(output_filepath, index=False, sep=';')
    
    return input_df


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prediction.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = predict_from_csv(input_file, output_filepath=output_file)
    print(result.to_csv(index=False, sep=';'))
