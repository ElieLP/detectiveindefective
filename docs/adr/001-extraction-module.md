# ADR-001: NCR Information Extraction Module

**Status:** Accepted  
**Date:** 2026-01-15  
**Context:** Hackathon PoC - Industrial AI Detective

---

## Context

The hackathon challenge requires extracting key elements from NCR (Non-Conformity Report) free-text descriptions:
- Defect type
- Machine/equipment
- Supplier
- Operator
- Process step / Lot number

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
| Machine | `MACHINE_\d+`, `WELDER_\d+`, `FURNACE_\d+`, etc. | `MACHINE_01`, `WELDER_03` |
| Supplier | Hardcoded list: `STEELCO`, `METALWORKS`, etc. | `ALLOYTECH`, `SEALTECH` |
| Operator | `[A-Z]+_[A-Z]` | `JEAN_D`, `MARIE_L` |
| Lot | `LOT-\d{4}-\d+` | `LOT-2025-0012` |

### Defect Classification

Keyword-based scoring. Each defect type has associated keywords:

```python
DEFECT_KEYWORDS = {
    'dimensional': ['dimensional', 'diameter', 'width', 'mm', 'slot', 'thread'],
    'surface': ['surface', 'finish', 'scratch', 'paint', 'adhesion'],
    'weld': ['weld', 'porosity', 'fracture', 'joint'],
    ...
}
```

The defect type with the highest keyword match count wins.

### Public API

| Function | Input | Output |
|----------|-------|--------|
| `extract_machines(text)` | NCR description | `List[str]` of machine IDs |
| `extract_suppliers(text)` | NCR description | `List[str]` of supplier names |
| `extract_operators(text)` | NCR description | `List[str]` of operator codes |
| `extract_lots(text)` | NCR description | `List[str]` of lot numbers |
| `classify_defect(text)` | NCR description | `str` defect type |
| `extract_all(text)` | NCR description | `Dict` with all extractions |
| `enrich_dataframe(df)` | DataFrame with `description` col | DataFrame with new columns |

### Usage

```python
from src.extraction import enrich_dataframe
import pandas as pd

df = pd.read_csv('data/sample_ncrs.csv')
enriched = enrich_dataframe(df)
# Now df has: machines, suppliers, operators, lots, defect_type columns
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
python3 src/extraction.py
```

Outputs extracted fields for all NCRs in `data/sample_ncrs.csv`.
