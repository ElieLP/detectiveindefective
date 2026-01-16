import re
from typing import Dict, List
import pandas as pd
 
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
 
 
def enrich_dataframe(df: pd.DataFrame, description_col: str = 'NC description') -> pd.DataFrame:
    enriched = df.copy()
    
    text_cols = ['NC description', 'FDefectDesc_EN', 'Fqccomments_EN', 'Root cause of occurrence']
    available_cols = [c for c in text_cols if c in df.columns]
    combined_text = df[available_cols].fillna('').agg(' '.join, axis=1)
    
    extractions = combined_text.apply(extract_all)
    enriched['extracted_machines'] = extractions.apply(lambda x: ', '.join(x['machines']) if x['machines'] else '')
    enriched['extracted_nc_codes'] = extractions.apply(lambda x: ', '.join(x['nc_codes']) if x['nc_codes'] else '')
    enriched['extracted_operations'] = extractions.apply(lambda x: ', '.join(x['operations']) if x['operations'] else '')
    enriched['defect_type'] = extractions.apply(lambda x: x['defect_type'])
    
    if 'Root cause of occurrence' in df.columns:
        enriched['root_cause'] = df['Root cause of occurrence'].fillna('')
    
    return enriched


def load_prod_data(filepath: str = 'data/prod_data.csv') -> pd.DataFrame:
    return pd.read_csv(filepath, sep=';')


if __name__ == '__main__':
    df = load_prod_data()
    enriched = enrich_dataframe(df)
    print(enriched[['Job order', 'NC Code', 'extracted_machines', 'defect_type']].head(10).to_string())
 