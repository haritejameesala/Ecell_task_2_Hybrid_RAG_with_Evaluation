# Hybrid RAG with Evaluation

An end-to-end semantic retrieval system built over a multi-document technical corpus. The pipeline ingests raw PDFs, preprocesses and chunks them, indexes them into a hybrid retrieval system, evaluates performance across multiple configurations, and serves grounded answers through a FastAPI endpoint.

Built for the **E-Cell NIT Trichy AI & Automation Domain — Task 2**.

---

## Project Structure

```bash
project/
├── data/
│   └── notebooks/
│       └── <notebook>/
│           └── docs/                # Raw PDF/TXT documents only
│
├── models/
│   └── notebooks/
│       └── <notebook>/
│           ├── faiss_store/         # Serialized FAISS + BM25 indices
│           ├── stats.json           # Index statistics
│           └── eval_report.json     # Evaluation results
│
├── src/
│   ├── preprocess.py                # Document loading, cleaning, chunking
│   ├── features.py                  # Embedding generation, index creation
│   ├── train.py                     # RAG chain, retriever, LLM orchestration
│   ├── evaluate.py                  # Pipeline benchmarking
│   └── utils.py                     # Helpers and stats
│
├── api/
│   └── app.py                       # FastAPI server
│
├── README.md
└── requirements.txt
```

---

## Pipeline Overview

## Stage 1 — Document Ingestion & Preprocessing

- Loads multi-page PDFs and `.txt` files
- Removes repeated headers, footers, and structural noise
- Detects document category:
  - SOP
  - POLICY
  - COMPLIANCE
  - TROUBLESHOOTING
- Splits documents by structural boundaries
- Applies hybrid chunking:
  1. Structural splitting
  2. Semantic chunking
  3. Recursive chunk capping

---

## Stage 2 — Embedding Generation & Indexing

| Component | Detail |
|---|---|
| Embedding model | `BAAI/bge-large-en-v1.5` |
| Dense retrieval | FAISS |
| Sparse retrieval | BM25 |
| Retrieval strategy | Hybrid Search |

### Corpus Statistics

| Metric | Value |
|---|---|
| Source documents | 5 PDFs |
| Total chunks | 2692 |
| Total tokens | 232434 |
| Avg chunk size | 608 chars |
| Min / Max chunk | 26 / 1181 chars |

---

## Stage 3 — LLM Inference & Context Orchestration

Retrieval pipeline:

- Query rewriting
- Hybrid retrieval (FAISS + BM25)
- Metadata filtering
- Optional cross-encoder reranking

### Anti-Hallucination Guardrails

- Answer only from retrieved context
- No outside knowledge
- Refuse unsupported questions
- Cite source files
- No inferred facts

### LLM Configurations

| Mode | Model |
|---|---|
| API | Groq `llama-3.3-70b-versatile` |
| Local | Ollama `llama3:latest` |

---

## Stage 4 — Pipeline Evaluation

Benchmarked using DeepEval.

### Metrics

| Metric | Meaning |
|---|---|
| CR | Context Relevance |
| F | Faithfulness |
| AR | Answer Relevance |
| L | Latency |
| QR | Query Resolution Rate |

### Benchmark Results

| Config | CR | F | AR | L (s) | QR |
|---|---|---|---|---|---|
| API | 0.6053 | 1.0000 | 0.9750 | 7.594 | 1.0 |
| Local | 0.5785 | 0.9222 | 0.9200 | 15.723 | 1.0 |
| API + Rerank | 0.6231 | 0.9000 | 0.9750 | 9.843 | 1.0 |

### Selected Configuration

**API (no rerank)**

Reason:  
The API configuration provides the best balance between faithfulness, answer relevance, and latency. While reranking improves context relevance slightly, it increases latency and reduces faithfulness. Since groundedness is a major task requirement, API mode was selected for deployment.

---

## Stage 5 — API Deployment

FastAPI application with auto-generated Swagger docs.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload/{notebook}` | Upload documents |
| POST | `/build-index/{notebook}` | Build retrieval index |
| POST | `/query/{notebook}` | Query documents |

Swagger UI:

```bash
/docs
```

---

## Example Query

Input:

```json
{
  "query": "What are the phases of incident response?"
}
```

Output:

```json
{
  "answer": "Preparation, Detection and Analysis, Containment, Eradication and Recovery, Post-Incident Activity",
  "confidence": 0.61,
  "sources": ["NIST.SP.800-61r2.pdf"]
}
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY_1=your_key
GROQ_API_KEY_2=your_key
GROQ_API_KEY_3=your_key
```

`GROQ_API_KEY_1` is used for API deployment.  
Separate keys are used for evaluation benchmarking to avoid rate-limit throttling.

---

## Usage

### Build Index

```bash
python -m src.features data/notebooks/default/docs data/notebooks/default/faiss_store
```

### Run Evaluation

```bash
python -m src.evaluate data/notebooks/default/faiss_store default
```

Output:

```bash
data/notebooks/default/eval_report.json
```

### Start API

```bash
uvicorn api.app:app --reload
```

Swagger:

```bash
http://localhost:8000/docs
```

---

## Document Corpus

| File | Category |
|---|---|
| NIST.SP.800-61r2.pdf | SOP |
| NIST.SP.800-53r5.pdf | COMPLIANCE |
| SANS_Password_Construction_Standard_April2025.pdf | POLICY |
| Remote-Access-Policy-Template-FINAL.pdf | POLICY |
| Troubleshooting Clusters _ Kubernetes.pdf | TROUBLESHOOTING |

---

## Tech Stack

- Python
- LangChain
- FAISS
- BM25
- HuggingFace Embeddings
- CrossEncoder
- Groq API
- Ollama
- FastAPI
- DeepEval