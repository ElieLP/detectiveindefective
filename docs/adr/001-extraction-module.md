# ADR-001: NCR Information Extraction Module

**Status:** Accepted  
**Date:** 2026-01-16 (Updated)  
**Context:** Hackathon PoC - Industrial AI Detective

---

## Context

The hackathon challenge requires extracting key elements from NCR (Non-Conformity Report) data:
- Defect type
- Machine/equipment
- NC Code
- Operation number
- Root cause

Data source: `data/prod_data.csv` (semicolon-separated, real production data).

We need a solution that works offline, is fast, and doesn't require API keys or external services.

---

## Decision

**Use regex-based pattern matching** for entity extraction rather than LLM-based extraction.

### Rationale

| Approach | Pros | Cons |
|----------|------|------|
| **Regex (chosen)** | Fast, offline, deterministic, no API cost | Requires known patterns, brittle to variations |
| LLM-based | Flexible, handles variations | Slow, requires API key, cost, hallucination risk |
| NER (spaCy) | Good for general entities | Needs training for domain-specific entities |

For a 24h PoC with ~100 NCRs and **known, structured entity formats** (e.g., `MACHINE_01`, `SUPPLIER_NAME`), regex is sufficient and reliable.

---

## Implementation

### Module Location
`src/extraction.py`

### Patterns Used

| Entity | Regex Pattern | Examples |
|--------|---------------|----------|
| Machine | `EM\d+`, `AAAA-\d+-\d+`, `BBBB-\d+-\d+`, `CCCC-\d+-\d+` | `EM1060`, `AAAA-02-09`, `BBBB-02-06` |
| NC Code | `[A-Z]{2}\d{4}` | `CO2610`, `EL0312`, `TR0310` |
| Operation | `OP\d+` | `OP7200`, `OP7300`, `OP4000` |

### Defect Classification

Keyword-based scoring. Each defect type has associated keywords:

```python
DEFECT_KEYWORDS = {
    'dimensional': ['dimensional', 'diameter', 'tolerance', 'mm', 'out of tolerance', 'gauge', 'no-go'],
    'surface': ['surface', 'scratch', 'dent', 'pit', 'bulge', 'offset'],
    'marking': ['marking', 'faint', 'unrecognizable', 'character', 'dot', 'label'],
    'appearance': ['appearance', 'collapse', 'tooth', 'rib', 'slot'],
    'process': ['deviation', 'calibration', 'compensation', 'clamping', 'centering', 'not stable'],
    'measurement': ['re-measurement', 'after re-measurement', 'FCTD', 'mini program'],
}
```

The defect type with the highest keyword match count wins.

### Public API

| Function | Input | Output |
|----------|-------|--------|
| `extract_machines(text)` | NCR text | `List[str]` of machine IDs |
| `extract_nc_codes(text)` | NCR text | `List[str]` of NC codes |
| `extract_operations(text)` | NCR text | `List[str]` of operation numbers |
| `classify_defect(text)` | NCR text | `str` defect type |
| `extract_all(text)` | NCR text | `Dict` with all extractions |
| `enrich_dataframe(df)` | DataFrame | DataFrame with extracted columns |
| `load_prod_data(filepath)` | CSV path | DataFrame from prod_data.csv |

### Usage

```python
from src.extraction import enrich_dataframe, load_prod_data

df = load_prod_data()  # Reads data/prod_data.csv (semicolon-separated)
enriched = enrich_dataframe(df)
# Now df has: extracted_machines, extracted_nc_codes, extracted_operations, defect_type, root_cause columns
```

---

## Consequences

### Positive
- No external dependencies or API keys
- Deterministic, reproducible results
- Fast execution (~ms per NCR)
- Easy to debug and extend

### Negative
- New entity formats require code changes
- Won't catch typos or variations (e.g., "Machine 01" vs "MACHINE_01")
- Defect classification is naive (keyword count, no context)

### Future Improvements
- Add fuzzy matching for entity variations
- Use embeddings for defect classification
- Optional LLM fallback for low-confidence extractions

---

## Test

```bash
.venv/bin/python src/extraction.py
```

Outputs extracted fields for all NCRs in `data/prod_data.csv`.
