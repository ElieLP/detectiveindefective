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
    """Build prompt for predicting root cause and corrective actions of a single NCR."""
    return f"""You are an expert in manufacturing quality control and root cause analysis.

Based on the following historical NCR (Non-Conformance Report) data:

{context}

Predict the most likely root cause AND corrective action for this new NCR:

NC Code: {row.get('NC Code', '')}
NC Description: {row.get('NC description', '')}
Part Type: {row.get('Part type', '')}
Machine of Detection: {row.get('MachineNum of detection', '')}
Machine of Occurrence: {row.get('MachineNum of occurrence', '')}
Operation of Detection: {row.get('Operation number of detection', '')}
Operation of Occurrence: {row.get('Operation number of occurrence', '')}
Defect Description: {row.get('FDefectDesc_EN', '')}
QC Comments: {row.get('Fqccomments_EN', '')}

Respond in this exact format (two lines only):
Root Cause: <predicted root cause>
Corrective Action: <predicted corrective action>"""


def predict_root_cause_and_action(row: pd.Series, context: str) -> tuple[str, str]:
    """Predict root cause and corrective action for a single NCR row using DashScope."""
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
        max_tokens=150
    )
    
    if response.status_code == HTTPStatus.OK:
        content = response.output.choices[0].message.content.strip()
        root_cause = ''
        corrective_action = ''
        for line in content.split('\n'):
            if line.lower().startswith('root cause:'):
                root_cause = line.split(':', 1)[1].strip()
            elif line.lower().startswith('corrective action:'):
                corrective_action = line.split(':', 1)[1].strip()
        return root_cause, corrective_action
    else:
        raise Exception(f"DashScope error: {response.code} - {response.message}")


def build_batch_prediction_prompt(df: pd.DataFrame, context: str) -> str:
    """Build prompt for predicting root cause and corrective actions for multiple NCRs."""
    ncr_items = []
    for idx, row in df.iterrows():
        ncr_items.append(f"""[NCR {idx}]
NC Code: {row.get('NC Code', '')}
NC Description: {row.get('NC description', '')}
Part Type: {row.get('Part type', '')}
Machine of Detection: {row.get('MachineNum of detection', '')}
Machine of Occurrence: {row.get('MachineNum of occurrence', '')}
Operation of Detection: {row.get('Operation number of detection', '')}
Operation of Occurrence: {row.get('Operation number of occurrence', '')}
Defect Description: {row.get('FDefectDesc_EN', '')}
QC Comments: {row.get('Fqccomments_EN', '')}""")
    
    ncr_list = "\n\n".join(ncr_items)
    
    return f"""You are an expert in manufacturing quality control. You are familiar with root cause analysis and proposing corrective actions. You mission is to become a machine learning model to predict root cause analysis and proposing corrective actions .

For your training dataset, you get the following historical Non-Conformity Report (NCR) data:

{context}

Hereunder a description of columns for better understanding, some are labelled as "DO NOT TAKE IT INTO ACCOUNT", hereunder, you must not take them into account:
Part Type: main part product families, there are only 2 possible values
Job Order: identifier of a specific sequence of production operations
Operation number of dectection: detection operation ID of detection which triggered a non conformity
NC Description: Description of non conformity (instructions: DO NOT TAKE IT INTO ACCOUNT)
NC Code: ID of a non conformity type
Nomial: DO NOT TAKE IT INTO ACCOUNT
FLowerTolerance: DO NOT TAKE IT INTO ACCOUNT
FUpperTolerance: DO NOT TAKE IT INTO ACCOUNT
Measured Value: DO NOT TAKE IT INTO ACCOUNT
FDefectDesc_EN: human input of the defect description
Fqccomments_EN: human input of quality comments on the defect part
MachineNum of detection: ID of detection machine, if the cell is "/" or "\" then it means that the defect detection was made during the production
Operator of detection: ID of the human worker working on the detection machine which detected the defect
Date of detection: date of defect detection, can be different from the date of production, the number of days between date of production and date of detection can be relevant for the root cause, instructions: calculate it in a spearate columns and integrate it in your training data
Operation number of occurrence:  manufacturing operation ID of detection which triggered a non conformity by production worker supervision
operator of machining:  ID of the human worker working on the production machine where a defect was detected
MachineNum of occurrence: ID of production machine where the defect was detected or where the detected part comes from
Date of machining: date of production, can be different from the date of production, the number of days between date of production and date of detection can be relevant for the root cause, instructions: calculate it in a spearate columns and integrate it in your training data
Root cause of occurrence: identified or likely root cause of the defect
Corrective actions: actions undertaken to resolve the root causes
Gold Sample: if yes, the NCR is considered as complete and very accurate and as a model

Some of columns are more relevant, consider them more and weight more than the rest. Those columns are:
Part Type
Operation number of dectection
NC Code
Operation number of occurrence
Gold Sample (if yes, the NCR must be more considered into your evaluation)

A correlation must be found between machine ID, operation ID, worker ID and non conformity code to train your model after.

Based on the dataset and the instruction, your task is to predict the most likely root cause AND corrective action for each of these NCRs using machine learning for structured data and LLM for the classification of the human input. Once the predictive model set, predict the root cause and the corrective action of following NCR :

{ncr_list}

Respond in this exact format for each NCR (one block per NCR, in order):
[NCR <index>]
Root Cause: <predicted root cause>
Corrective Action: <predicted corrective action>
"""


def predict_batch(df: pd.DataFrame, context: str) -> list[tuple[str, str]]:
    """Predict root cause and corrective action for multiple NCR rows in a single API call."""
    prompt = build_batch_prediction_prompt(df, context)
    
    messages = [
        {'role': Role.SYSTEM, 'content': 'You are an expert in manufacturing quality control and root cause analysis.'},
        {'role': Role.USER, 'content': prompt}
    ]
    
    response = Generation.call(
        model='qwen-plus',
        messages=messages,
        result_format='message',
        temperature=0.3,
        max_tokens=150 * len(df)
    )
    
    if response.status_code == HTTPStatus.OK:
        content = response.output.choices[0].message.content.strip()
        return parse_batch_response(content, len(df))
    else:
        raise Exception(f"DashScope error: {response.code} - {response.message}")


def parse_batch_response(content: str, expected_count: int) -> list[tuple[str, str]]:
    """Parse batch response into list of (root_cause, corrective_action) tuples."""
    results = []
    current_root = ''
    current_action = ''
    
    for line in content.split('\n'):
        line = line.strip()
        if line.lower().startswith('[ncr'):
            if current_root or current_action:
                results.append((current_root, current_action))
            current_root = ''
            current_action = ''
        elif line.lower().startswith('root cause:'):
            current_root = line.split(':', 1)[1].strip()
        elif line.lower().startswith('corrective action:'):
            current_action = line.split(':', 1)[1].strip()
    
    if current_root or current_action:
        results.append((current_root, current_action))
    
    while len(results) < expected_count:
        results.append(('', ''))
    
    return results[:expected_count]


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
    
    root_causes = []
    corrective_actions = []
    for idx, row in input_df.iterrows():
        root_cause = row.get('Root cause of occurrence', '')
        corrective = row.get('Corrective actions', '')
        needs_root_cause = pd.isna(root_cause) or root_cause == ''
        needs_corrective = pd.isna(corrective) or corrective == ''
        
        if needs_root_cause or needs_corrective:
            pred_root, pred_action = predict_root_cause_and_action(row, context)
            root_causes.append(pred_root if needs_root_cause else root_cause)
            corrective_actions.append(pred_action if needs_corrective else corrective)
        else:
            root_causes.append(root_cause)
            corrective_actions.append(corrective)
    
    input_df['Root cause of occurrence'] = root_causes
    input_df['Corrective actions'] = corrective_actions
    
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
