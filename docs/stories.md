# User Stories - Industrial AI Detective

## Domain: NCR Ingestion

### US-01: Load NCR Data
**As a** quality engineer  
**I want to** load NCR data from a predefined data directory  
**So that** the system can analyze my factory's non-conformity data

**Acceptance Criteria:**
- Reads CSV/Excel files from `data/` directory
- Displays row count after loading
- Shows data preview

---

## Domain: Information Extraction

### US-02: Extract Key Entities
**As a** quality engineer  
**I want to** automatically extract defect type, machine, supplier, and operator from NCR text  
**So that** I don't have to manually tag each report

**Acceptance Criteria:**
- Extracts machine identifiers (e.g., MACHINE_01)
- Extracts supplier names
- Extracts operator references
- Classifies defect type (dimensional, surface, contamination, etc.)

---

## Domain: Similarity & Clustering

### US-03: Detect Recurring Problem Families
**As a** quality engineer  
**I want to** see NCRs automatically grouped into problem families  
**So that** I can prioritize which issues need systemic fixes vs. one-off corrections

**Acceptance Criteria:**
- NCRs are grouped by problem type (not just defect category)
- Each group shows common characteristics (machines, suppliers, operators)
- Groups are ranked by frequency or severity

### US-04: Find Similar Cases
**As a** quality engineer  
**I want to** describe a problem and find similar past NCRs  
**So that** I can learn from historical cases

**Acceptance Criteria:**
- Text input for problem description
- Returns top 5 most similar NCRs
- Shows similarity score

---

## Domain: Query & Search

### US-05: Query by Entity
**As a** quality engineer  
**I want to** ask questions like "Show NCRs involving MACHINE_01"  
**So that** I can quickly filter relevant cases

**Acceptance Criteria:**
- Supports machine, supplier, operator filters
- Returns matching NCRs
- Shows result count

### US-06: Query Recurring Defects
**As a** quality engineer  
**I want to** ask "What defects keep coming back on this process?"  
**So that** I can prioritize root cause analysis

**Acceptance Criteria:**
- Identifies repeated defect patterns
- Groups by frequency
- Highlights trends

---

## Domain: Recommendations (Future)

### US-07: Suggest Root Causes
**As a** quality engineer  
**I want to** get suggested root causes based on similar past NCRs  
**So that** I can accelerate my investigation

### US-08: Propose Corrective Actions
**As a** quality engineer  
**I want to** see what corrective actions worked before  
**So that** I can apply proven solutions

---

## Domain: Dashboards (Future)

### US-09: View Trends Over Time
**As a** quality manager  
**I want to** see NCR trends over time  
**So that** I can detect degradation early

### US-10: Identify Weak Signals
**As a** quality manager  
**I want to** see which machines/suppliers appear most in NCRs  
**So that** I can focus improvement efforts
