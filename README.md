# RAG Fine-tuned Embeddings

A high-precision Retrieval-Augmented Generation system implementing contrastive fine-tuning and hybrid retrieval.

## Overview

This project implements a modular RAG pipeline focused on retrieval accuracy. It addresses common limitations in standard RAG implementations by integrating contrastive learning, semantic chunking, and rank fusion.

| Component | Previous | Current (v2.0) |
|-----------|----------|----------------|
| **Embeddings** | all-MiniLM-L6-v2 | BGE-M3 |
| **Training** | Reconstruction (Autoencoder) | Contrastive (Triplet Loss) |
| **Chunking** | Fixed Size | Semantic Boundary |
| **Retrieval** | Naive Hybrid | Reciprocal Rank Fusion (RRF) |
| **Reranking** | None | Cross-Encoder |
| **Storage** | In-memory | Persistent Qdrant |

### Contrastive Learning

The system replaces autoencoder-based training with proper contrastive loss to learn semantic relevance.

**Previous approach (Reconstruction):**
```python
model.fit(embedding, embedding)
```

**Current approach (Contrastive):**
```python
loss = TripletLoss(anchor=query, positive=relevant, negative=irrelevant)
```

## Installation

```bash
git clone https://github.com/s-smits/RAG-finetuned-embeddings
cd RAG-finetuned-embeddings

# Project management with uv (recommended)
uv sync

# Or standard pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configuration
cp .env.example .env
```

## Usage

### Web Interface
```bash
uv run python main.py ui
```
Access at `http://localhost:7860`.

### CLI
```bash
# Ingest
uv run python main.py ingest --files doc.pdf

# Query
uv run python main.py query "Search query"

# Train
uv run python main.py finetune --epochs 3
```

### Python API
```python
from src.pipeline import create_pipeline

pipeline = create_pipeline(embedding_model="BAAI/bge-m3")
pipeline.ingest(["manual.pdf"])
pipeline.fine_tune(epochs=3)

response = pipeline.query("Query text")
print(response.answer)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Processing                       │
├─────────────────────────────────────────────────────────────┤
│  Input → Semantic Chunking → BGE-M3 Embeddings → Indexing   │
│                                    │                         │
│                            ┌───────┴───────┐                │
│                            ▼               ▼                │
│                       Qdrant           BM25 Index           │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Query → Hybrid Search (RRF) → Cross-Encoder Rerank → LLM   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Configure via `.env`:

```bash
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL_NAME=BAAI/bge-m3
CHUNKING_STRATEGY=semantic
RETRIEVAL_HYBRID_ALPHA=0.5
```

## Performance Metrics

| Metric | Target |
|--------|--------|
| Context Precision | >80% |
| Context Recall | >85% |
| Answer Relevancy | >90% |
| Latency (p95) | <200ms |

## Testing

```bash
uv run pytest
```

## License

[MIT License](LICENSE)

