import re
from typing import Dict, List
import pandas as pd
import numpy as np
 
MACHINE_PATTERN = r'\b(EM\d+|AAAA-\d+-\d+|BBBB-\d+-\d+|CCCC-\d+-\d+|MARK-\d+-\d+)\b'
NC_CODE_PATTERN = r'\b([A-Z]{2}\d{4})\b'
JOB_ORDER_PATTERN = r'\b(AA[12]_\d+)\b'
OPERATION_PATTERN = r'\b(OP\d+)\b'
 
DEFECT_KEYWORDS = {
    'dimensional': ['dimensional', 'diameter', 'tolerance', 'mm', 'out of tolerance', 'measured', 'gauge', 'no-go'],
    'surface': ['surface', 'scratch', 'dent', 'pit', 'bulge', 'offset'],
    'marking': ['marking', 'faint', 'unrecognizable', 'character', 'dot', 'label'],
    'appearance': ['appearance', 'collapse', 'tooth', 'rib', 'slot'],
    'process': ['deviation', 'calibration', 'compensation', 'clamping', 'centering', 'not stable'],
    'measurement': ['re-measurement', 'after re-measurement', 'FCTD', 'mini program'],
}
 
 
def extract_machines(text: str) -> List[str]:
    return list(set(re.findall(MACHINE_PATTERN, text)))
 
 
def extract_nc_codes(text: str) -> List[str]:
    return list(set(re.findall(NC_CODE_PATTERN, text)))


def extract_operations(text: str) -> List[str]:
    return list(set(re.findall(OPERATION_PATTERN, text)))
 
 
def classify_defect(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for defect_type, keywords in DEFECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[defect_type] = score
    if scores:
        return max(scores, key=scores.get)
    return 'unknown'
 
 
def extract_all(text: str) -> Dict:
    return {
        'machines': extract_machines(text),
        'nc_codes': extract_nc_codes(text),
        'operations': extract_operations(text),
        'defect_type': classify_defect(text),
    }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean placeholder values (/, \, ,) and empty strings to NaN."""
    cleaned = df.copy()
    cleaned = cleaned.replace(r'[\/\\,]', np.nan, regex=True)
    cleaned = cleaned.replace(r'^\s*$', np.nan, regex=True)
    return cleaned


def categorize_corrective(val: str) -> str:
    """Categorize corrective actions."""
    val = str(val).strip()
    if val in ['NaN', 'nan', '', 'None']:
        return 'Action Not Specified (NaN)'
    elif 'clamping force' in val.lower():
        return 'Cancel Releasing Clamping Force'
    elif 'manual tool calibration' in val.lower():
        return 'Add Manual Tool Calibration'
    elif 'compensation 0.005' in val.lower():
        return 'Cancelled Compensation 0.005'
    elif 'lesson&learn' in val.lower():
        return 'Lesson & Learn'
    elif val == 'Maintenance marking machine':
        return 'Maintenance marking machine'
    elif val == 'Marking machine maintenance':
        return 'Marking machine maintenance'
    else:
        return 'Other'


def categorize_root_cause(val: str) -> str:
    """Categorize root causes."""
    val = str(val).strip()
    if val == 'NaN' or val == 'nan' or val == '':
        return 'Undefined'
    elif 'EL2415' in val:
        return 'Related to NC EL2415'
    elif '1.Marking NC' in val:
        return 'Marking Precision'
    elif 'Marking machine precision' in val:
        return 'Marking Precision'
    elif 'AAAA-02-11 deviation' in val:
        return 'Machine AAAA-02-11 Deviation'
    elif 'CP&CPK' in val:
        return 'Process Stability (CP/CPK)'
    elif 'AAAA-02-11 not stable' in val:
        return 'Machine AAAA-02-11 Stability'
    elif 'transportation' in val:
        return 'Logistics & Transport'
    elif 'not clear' in val:
        return 'Unclear'
    elif 'has deviation on centering' in val:
        return 'Machine BBBB-02-05 Centering'
    elif 'deviation on cerntering' in val:
        return 'Machine BBBB-02-05 Centering'
    else:
        return 'Other'


def categorize_fqc(text: str) -> str:
    """Categorize QA comments."""
    text = str(text).lower().strip()
    if text in ['nan', 'none', '', '/']:
        return 'Undefined'
    elif any(word in text for word in ['awaiting', 'waiting', 'investigation']):
        return 'Pending Decision'
    elif any(word in text for word in ['continue', 'accept', 'approved', 'verbal']):
        return 'Decision: Proceed/Accept'
    elif any(word in text for word in ['re-measurement', 'recheck', 'retest', 'fctd', 'attachment']):
        return 'Measurement/Verification'
    elif any(word in text for word in ['confirmed', 'reply', 'decided', 'rework']):
        return 'QA Replied/Action Taken'
    else:
        return 'Other/General'


def categorize_defect(val: str) -> str:
    """Categorize defect descriptions."""
    val = str(val).strip()
    if val in ['NaN', 'nan', '', '/']:
        return 'Undefined'
    elif val == 'CO2910-R342':
        return 'CO2910-R342 Deviation'
    elif val == 'OP7200 DA':
        return 'OP7200 DA Operation'
    elif 'DA2512100009' in val:
        return 'DA2512100009 (Post-Check)'
    elif val in ['out of tolerance after re-measurement 1', 'after re-measurement dimension out of tolerance']:
        return 'Out of Tolerance (Post-Check)'
    elif 'the 1st time' in val:
        return 'Primary Measurement Failure'
    elif 'EL0312-MAX' in val:
        return 'EL0312-MAX Deviation'
    elif val == 'dimension out of tolerance':
        return 'General Out of Tolerance'
    elif 'after rework' in val:
        return 'Post-Rework Failure'
    elif 'CO2910-R342 out of tolerance' in val:
        return 'CO2910-R342 Out of Tolerance'
    elif 'dent' in val:
        return 'Flange Surface Dent'
    elif 'scratch' in val:
        return 'Flange Scratch'
    elif 'visually clear' in val:
        return 'Marking Visual Clarity'
    elif 'too shallow' in val:
        return 'Marking Depth Issue'
    else:
        return 'Other Technical Records'


def extract_comment_dates(text: str) -> str:
    """Extract dates from comments."""
    text = str(text)
    dates = re.findall(r'\d{4}[./-]\d{2}[./-]\d{2}', text)
    return '; '.join(dates) if dates else ''


def extract_comment_codes(text: str) -> str:
    """Extract department codes from comments."""
    text = str(text)
    codes = re.findall(r'[A-Z]{4}-\d{2}-\d{2}', text)
    return '; '.join(codes) if codes else ''
 
 
def enrich_dataframe(df: pd.DataFrame, description_col: str = 'NC description') -> pd.DataFrame:
    enriched = clean_dataframe(df)
    
    text_cols = ['NC description', 'FDefectDesc_EN', 'Fqccomments_EN', 'Root cause of occurrence']
    available_cols = [c for c in text_cols if c in enriched.columns]
    combined_text = enriched[available_cols].fillna('').agg(' '.join, axis=1)
    
    extractions = combined_text.apply(extract_all)
    enriched['extracted_machines'] = extractions.apply(lambda x: ', '.join(x['machines']) if x['machines'] else '')
    enriched['extracted_nc_codes'] = extractions.apply(lambda x: ', '.join(x['nc_codes']) if x['nc_codes'] else '')
    enriched['extracted_operations'] = extractions.apply(lambda x: ', '.join(x['operations']) if x['operations'] else '')
    enriched['defect_type'] = extractions.apply(lambda x: x['defect_type'])
    
    if 'Root cause of occurrence' in enriched.columns:
        enriched['root_cause'] = enriched['Root cause of occurrence'].fillna('')
        enriched['root_cause_category'] = enriched['Root cause of occurrence'].apply(categorize_root_cause)
    
    if 'Corrective actions' in enriched.columns:
        enriched['corrective_category'] = enriched['Corrective actions'].apply(categorize_corrective)
    
    if 'Fqccomments_EN' in enriched.columns:
        enriched['fqc_category'] = enriched['Fqccomments_EN'].apply(categorize_fqc)
        enriched['fqc_dates'] = enriched['Fqccomments_EN'].apply(extract_comment_dates)
        enriched['fqc_dept_codes'] = enriched['Fqccomments_EN'].apply(extract_comment_codes)
    
    if 'FDefectDesc_EN' in enriched.columns:
        enriched['defect_category'] = enriched['FDefectDesc_EN'].apply(categorize_defect)
    
    return enriched


def load_prod_data(filepath: str = 'data/prod_data.csv') -> pd.DataFrame:
    return pd.read_csv(filepath, sep=';')


if __name__ == '__main__':
    df = load_prod_data()
    enriched = enrich_dataframe(df)
    print(enriched[['Job order', 'NC Code', 'extracted_machines', 'defect_type']].head(10).to_string())
 