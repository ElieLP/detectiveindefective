import re
from typing import Dict, List
import pandas as pd
 
MACHINE_PATTERN = r'\b(MACHINE_\d+|WELDER_\d+|FURNACE_\d+|TEST_BENCH_\d+|ASSEMBLY_LINE_\d+|PACK_STATION_\d+|LABELING_STATION_\d+|PAINT_BOOTH_\d+|LASER_MARK_\d+|CLEANROOM_\d+)\b'
SUPPLIER_PATTERN = r'\b(STEELCO|METALWORKS|ALLOYTECH|SEALTECH|BOXPRO|COATPRO|CABLETECH|LUBRITECH|GASTECH|PARTCO|STERILEPACK)\b'
OPERATOR_PATTERN = r'\b([A-Z]+_[A-Z])\b'
LOT_PATTERN = r'\b(LOT-\d{4}-\d+)\b'
 
DEFECT_KEYWORDS = {
    'dimensional': ['dimensional', 'diameter', 'width', 'depth', 'tolerance', 'mm', 'slot', 'thread'],
    'surface': ['surface', 'finish', 'scratch', 'cosmetic', 'paint', 'adhesion', 'contamination', 'residue'],
    'weld': ['weld', 'porosity', 'fracture', 'joint'],
    'electrical': ['electrical', 'insulation', 'resistance', 'wire'],
    'material': ['tensile', 'hardness', 'HRC', 'MPa', 'alloy', 'composition'],
    'leak': ['leak', 'pressure', 'decay', 'seal', 'o-ring'],
    'packaging': ['packaging', 'dent', 'damage', 'weight', 'label'],
    'documentation': ['documentation', 'signature', 'traceability', 'serial', 'record'],
    'contamination': ['sterility', 'bioburden', 'CFU', 'clean'],
    'process': ['temperature', 'heat treatment', 'process deviation'],
}
 
 
def extract_machines(text: str) -> List[str]:
    return list(set(re.findall(MACHINE_PATTERN, text, re.IGNORECASE)))
 
 
def extract_suppliers(text: str) -> List[str]:
    return list(set(re.findall(SUPPLIER_PATTERN, text, re.IGNORECASE)))
 
 
def extract_operators(text: str) -> List[str]:
    return list(set(re.findall(OPERATOR_PATTERN, text)))
 
 
def extract_lots(text: str) -> List[str]:
    return list(set(re.findall(LOT_PATTERN, text, re.IGNORECASE)))
 
 
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
        'suppliers': extract_suppliers(text),
        'operators': extract_operators(text),
        'lots': extract_lots(text),
        'defect_type': classify_defect(text),
    }
 
 
def enrich_dataframe(df: pd.DataFrame, description_col: str = 'description') -> pd.DataFrame:
    enriched = df.copy()
    extractions = df[description_col].apply(extract_all)
    enriched['machines'] = extractions.apply(lambda x: ', '.join(x['machines']) if x['machines'] else '')
    enriched['suppliers'] = extractions.apply(lambda x: ', '.join(x['suppliers']) if x['suppliers'] else '')
    enriched['operators'] = extractions.apply(lambda x: ', '.join(x['operators']) if x['operators'] else '')
    enriched['lots'] = extractions.apply(lambda x: ', '.join(x['lots']) if x['lots'] else '')
    enriched['defect_type'] = extractions.apply(lambda x: x['defect_type'])
    return enriched
 
 
if __name__ == '__main__':
    df = pd.read_csv('data/sample_ncrs.csv')
    enriched = enrich_dataframe(df)
    print(enriched[['ncr_id', 'machines', 'suppliers', 'operators', 'defect_type']].to_string())
 