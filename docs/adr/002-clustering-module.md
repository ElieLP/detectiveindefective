# ADR-002: NCR Clustering Module

**Status:** Accepted  
**Date:** 2026-01-15  
**Context:** Hackathon PoC - Industrial AI Detective

---

## Context

The hackathon challenge requires:
- Grouping NCRs by similarity and defect families
- Detecting recurring patterns
- Finding similar historical cases

We need a way to measure semantic similarity between NCR descriptions, not just keyword matching.

---

## Decision

**Use sentence embeddings + KMeans clustering** for grouping similar NCRs.

### Rationale

| Approach | Pros | Cons |
|----------|------|------|
| **Embeddings + KMeans (chosen)** | Captures semantic meaning, works offline, fast inference | Requires model download (~90MB), fixed cluster count |
| TF-IDF + KMeans | Lightweight, no model needed | Keyword-based, misses synonyms |
| LLM-based clustering | Best quality | Slow, requires API, expensive |
| HDBSCAN | Auto-detects cluster count | More complex, can leave outliers |

For a 24h PoC with ~100 NCRs, embeddings provide the best balance of quality and simplicity.

---

## Implementation

### Module Location
`src/clustering.py`

### Model Choice

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| **all-MiniLM-L6-v2** (chosen) | 90MB | Fast | Good |
| all-mpnet-base-v2 | 420MB | Medium | Better |
| OpenAI text-embedding-3 | API | Slow | Best |

`all-MiniLM-L6-v2` is ideal for PoC: small, fast, runs offline, good enough quality.

### Architecture

```
NCR descriptions
      │
      ▼
┌─────────────────┐
│ SentenceTransformer │  (all-MiniLM-L6-v2)
└─────────────────┘
      │
      ▼
  384-dim vectors
      │
      ├──────────────────┐
      ▼                  ▼
┌──────────┐      ┌──────────────┐
│  KMeans  │      │ Cosine Sim   │
│ Clustering│     │ (find_similar)│
└──────────┘      └──────────────┘
      │                  │
      ▼                  ▼
 Cluster labels    Similarity scores
```

### Singleton Pattern

Model is loaded once and cached:

```python
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model
```

This avoids reloading the 90MB model on every call.

### Public API

| Function | Input | Output |
|----------|-------|--------|
| `get_model()` | - | SentenceTransformer instance |
| `compute_embeddings(texts)` | List[str] | np.ndarray (N × 384) |
| `find_similar(query, embeddings, k)` | embedding + corpus | List[(index, score)] |
| `cluster_ncrs(embeddings, n)` | embeddings | np.ndarray of labels |
| `add_embeddings_and_clusters(df)` | DataFrame | DataFrame + embedding, cluster cols |

### Usage

```python
from src.clustering import add_embeddings_and_clusters, find_similar, compute_embeddings

# Add clusters to dataframe
df = pd.read_csv('data/sample_ncrs.csv')
df = add_embeddings_and_clusters(df, n_clusters=4)

# Find similar NCRs
query = "dimensional issue on CNC machine"
query_embedding = compute_embeddings([query])[0]
similar = find_similar(query_embedding, df['embedding'].tolist(), top_k=5)
```

---

## Test Results

```
Cluster distribution (n_clusters=4):
cluster
0     4   # Test failures, packaging
1     3   # Documentation/labeling
2     2   # Weld defects
3    11   # Dimensional/surface (largest)
```

---

## Consequences

### Positive
- Semantic similarity (not just keywords)
- Works offline after first model download
- Fast inference (~100ms for 20 NCRs)
- Enables "find similar" feature

### Negative
- First run downloads 90MB model
- Fixed cluster count (must choose K upfront)
- Embedding column is large (384 floats per NCR)

### Future Improvements
- Use HDBSCAN for automatic cluster count
- Add cluster labeling (summarize what each cluster is about)
- Persist embeddings to avoid recomputation

---

## Dependencies Added

```
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
```

---

## Test

```bash
.venv/bin/python src/clustering.py
```

Outputs cluster assignments for all NCRs.
