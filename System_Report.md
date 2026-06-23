# System Report — Hybrid RAG with Evaluation

**E-Cell NIT Trichy | AI & Automation Domain | Task 2**

---

## 1. System Overview

This project implements a five-stage end-to-end Retrieval-Augmented Generation (RAG) pipeline over a multi-document technical corpus. The system ingests raw PDFs, preprocesses and chunks text into semantically coherent segments, indexes them into a hybrid retrieval system, evaluates retrieval quality across three pipeline configurations, and serves grounded answers through a FastAPI endpoint.

---

## 2. Document Corpus

| File | Category | Type |
|---|---|---|
| `NIST.SP.800-61r2.pdf` | SOP | Incident Response Guide |
| `NIST.SP.800-53r5.pdf` | COMPLIANCE | Security Controls Catalog |
| `SANS_Password_Construction_Standard_April2025.pdf` | POLICY | Password Policy |
| `Remote-Access-Policy-Template-FINAL.pdf` | POLICY | Remote Access Policy |
| `Troubleshooting Clusters _ Kubernetes.pdf` | TROUBLESHOOTING | Technical Guide |

**5 source documents | 585 pages | 2,692 chunks | 232,434 tokens**

---

## 3. Stage 1 — Document Ingestion & Text Segmentation

### 3.1 Loading

Both PDF and TXT files are supported. PDFs are loaded page-by-page using `PyPDFLoader`. Each document’s first 3,000 characters are scanned to auto-detect a document category (`SOP`, `POLICY`, `COMPLIANCE`, `TROUBLESHOOTING`, `UNKNOWN`).

---

### 3.2 Noise Removal

The following are removed during preprocessing:

- Standalone page numbers
- Repeated headers/footers (appearing 3+ times)
- Short uppercase noise lines
- Low-information lines (< 3 alphabetic characters)
- Non-ASCII artifacts
- Excess whitespace

Technical commands (`kubectl`, `docker`, `systemctl`, etc.) are preserved to retain procedural meaning.

---

### 3.3 Structural Splitting

Before semantic chunking, documents are split at structural boundaries:

- Numbered sections (`1.2.3`)
- NIST control IDs (`AC-6`)
- Uppercase headers
- Short title-case lines

This preserves logical document hierarchy before chunking.

---

### 3.4 Chunking Strategy

A three-pass hybrid chunking strategy is used:

#### Pass 1 — Semantic Chunking
Uses `SemanticChunker` with:

```text
breakpoint_threshold_type = percentile
breakpoint_threshold_amount = 80
```

This groups semantically coherent sentences.

#### Pass 2 — Recursive Size Capping
Uses:

```text
max_chunk_size = 1000
overlap = 100
```

Oversized chunks are recursively split.

#### Pass 3 — Minimum Merge
Chunks smaller than:

```text
min_chunk_size = 200
```

are merged with neighbouring chunks.

### Chunk Statistics

| Metric | Value |
|---|---|
| Chunk size (min) | 26 chars |
| Chunk size (max) | 1181 chars |
| Chunk size (avg) | 608.1 chars |
| Total chunks | 2692 |

### Justification

Fixed-size chunking breaks control descriptions mid-context. Pure semantic chunking produces oversized chunks in dense regulatory text. The hybrid strategy preserves structure, semantic coherence, and retrieval efficiency.

---

## 4. Stage 2 — Embedding Generation & Indexing

### 4.1 Embedding Model

| Parameter | Value |
|---|---|
| Model | `BAAI/bge-large-en-v1.5` |
| Execution | Local |
| Normalization | Enabled |
| Dimension | 1024 |

### Justification

This model provides better semantic precision for technical and compliance-heavy text than lightweight alternatives.

---

### 4.2 Retrieval System

| Component | Implementation |
|---|---|
| Dense retrieval | FAISS |
| Sparse retrieval | BM25 |
| Strategy | Hybrid Search |

### Why Hybrid?

Dense retrieval captures semantic similarity. Sparse retrieval captures exact matches such as:

- `AC-6`
- `IA-2`
- `16 characters`

Combining both improves recall and precision.

---

### 4.3 Metadata Payload

Each chunk stores:

- `source_file`
- `file_type`
- `doc_category`
- `section_title`
- `parent_section`
- `page`

This enables metadata-aware retrieval boosting.

---

## 5. Stage 3 — LLM Inference & Context Orchestration

### 5.1 Retrieval Pipeline

The pipeline uses:

### Query Rewriting
Expands domain keywords:

Example:

```text
incident → incident response lifecycle preparation detection containment eradication recovery
```

Improves retrieval recall.

---

### Hybrid Fusion

Retrieval flow:

- Top-20 dense results (threshold ≤ 1.15)
- Top-10 BM25 results
- Deduplication

---

### Metadata Boosting

Documents matching expected categories are prioritized:

Example:

- Password queries → `POLICY`
- Kubernetes queries → `TROUBLESHOOTING`

---

### Optional Cross-Encoder Reranking

Uses:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Filters relevant chunks by contextual fit.

---

## 5.2 Anti-Hallucination Guardrails

The prompt enforces:

1. Answer only from retrieved context
2. No outside knowledge
3. No unsupported claims
4. Partial answers only if partially grounded
5. No inference of missing facts
6. Refuse if unsupported
7. Cite sources
8. Preserve exact control mappings

Refusal phrases are detected and flagged.

Confidence score is computed using cross-encoder groundedness scoring, measuring how strongly retrieved chunks support the generated answer.

---

## 5.3 LLM Configurations

| Mode | Model | Provider |
|---|---|---|
| API | `llama-3.3-70b-versatile` | Groq |
| Local | `llama3:latest` | Ollama |

Both use:

```text
temperature = 0
```

for deterministic outputs.

---

## 6. Stage 4 — Pipeline Evaluation

### 6.1 Evaluation Setup

Evaluation uses DeepEval.

Test queries:

1. Incident response phases
2. Minimum password length
3. Remote access requirements
4. Least privilege control
5. Kubernetes troubleshooting

---

### 6.2 Metrics

| Metric | Description |
|---|---|
| CR | Context Relevance |
| F | Faithfulness |
| AR | Answer Relevance |
| L | Latency |
| QR | Query Resolution Rate |

---

### 6.3 Results

| Config | CR | F | AR | L (s) | QR |
|---|---|---|---|---|---|
| API (no rerank) | 0.4721 | 1.0000 | 0.9179 | 8.366 | 1.0 |
| Local (no rerank) | 0.4628 | 1.0000 | 0.9214 | 10.159 | 1.0 |
| API + Rerank | 0.5446 | 1.0000 | 0.9314 | 6.982 | 1.0 |

---

### 6.4 Error Analysis

| Query | Issue | Root Cause |
|---|---|---|
| Password length | Low CR | Chunk contains extra policy text |
| Incident response | Moderate CR | List split across pages |
| Kubernetes (rerank) | CR drop | Useful procedural chunks filtered |

---

## 7. Stage 5 — API Deployment

### 7.1 Architecture

FastAPI deployment with lazy pipeline loading and caching.

Swagger documentation available at:

```text
/docs
```

---

### 7.2 Endpoints

| Method | Endpoint |
|---|---|
| GET | `/` |
| POST | `/upload/{notebook}` |
| POST | `/build-index/{notebook}` |
| POST | `/query/{notebook}` |

---

### 7.3 Query Interface

Request:

```json
{
  "query": "What is the minimum password length?"
}
```

Response:

```json
{
  "answer": "The minimum password length is 16 characters.",
  "confidence": 0.86,
  "sources": ["SANS_Password_Construction_Standard_April2025.pdf"]
}
```

---

## 8. Selected Configuration

### Final Deployment

**API + rerank**

### Justification

| Criterion | Reason |
|---|---|
| Faithfulness = 1.0 | Zero hallucination |
| AR = 0.9314 | Highest semantic answer quality |
| CR = 0.5446 | Best retrieval grounding |
| Latency = 6.982s | Fastest overall |
| QR = 1.0 | All queries resolved |


---

## 9. Final Configuration Summary

| Component | Selected Choice |
|---|---|
| Embedding model | BAAI/bge-large-en-v1.5 |
| Chunking | Structural + Semantic + Recursive Hybrid |
| Chunk size | max=1000, min=200, overlap=100 |
| Semantic breakpoint | Percentile 80 |
| Retrieval | FAISS + BM25 |
| LLM | Groq Llama-3.3-70B |
| Reranking | Enabled |
| top_k | 8 |
| Dense threshold | 1.15 |

---

Overall, the system satisfies all minimum task requirements including modular ingestion, hybrid indexing, grounded inference, benchmark evaluation, and production API deployment.