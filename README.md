# Document Grounded Conersation AI using RAG

## RAG Document Intelligence

A production-quality RAG system for PDF Q&A with hybrid retrieval, citation highlighting, and conversational memory.

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemini 2.0 Flash (free) |
| Embeddings | Gemini text-embedding-004 (free) |
| Sparse retrieval | BM25 via rank-bm25 (local) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (local) |
| Vector store | Qdrant (local) |
| Graph | LangGraph |
| PDF | PyMuPDF + pdfplumber |
| Vision | Gemini Vision |
| Web | Flask |

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo>
cd rag-document-intelligence
```

### 2. Create virtual environment

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the cross-encoder reranker (~90MB) automatically.

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
GEMINI_API_KEY=your_key_here
FLASK_SECRET_KEY=run_this_to_generate: python -c "import secrets; print(secrets.token_hex(32))"
```

Get your free Gemini API key at: https://aistudio.google.com/

### 5. Run the application

```bash
python app.py
```

Open browser at: **http://localhost:5000**

---

## Usage

1. **Upload PDF** — Click "Upload PDF" or drag and drop any PDF
2. **Wait for indexing** — Progress bar shows extraction → chunking → embedding → indexing
3. **Ask questions** — Type in the chat box
4. **See citations** — Click `[pN:cN]` chips to jump to source in PDF
5. **View evidence** — Click "◈ N sources retrieved" to see the retrieval debug panel
6. **Highlighting** — PDF viewer highlights exact chunks that sourced the answer

---

## Running Tests

```bash
# Unit tests (no document needed)
pytest tests/test_ingestion.py tests/test_retrieval.py -v

# All tests
pytest tests/ -v
```

---

## Project Structure

```
rag-document-intelligence/
├── app.py                    # Flask application
├── core/
│   ├── ingestion/
│   │   ├── pdf_extractor.py  # PyMuPDF + pdfplumber + Gemini vision
│   │   ├── chunker.py        # Semantic chunking with bbox tracking
│   │   └── indexer.py        # Qdrant + BM25 indexing
│   ├── retrieval/
│   │   ├── retriever.py      # Hybrid search (dense + BM25 + RRF)
│   │   └── reranker.py       # Cross-encoder reranking
│   ├── generation/
│   │   ├── llm.py            # Gemini client
│   │   └── prompts.py        # Prompt templates
│   └── graph/
│       ├── state.py          # LangGraph state
│       ├── nodes.py          # Pipeline nodes
│       └── rag_graph.py      # Graph definition
├── frontend/
│   ├── templates/index.html  # Main UI
│   └── static/
│       ├── css/style.css
│       └── js/
│           ├── app.js
│           ├── pdf_viewer.js
│           └── evidence_panel.js
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    └── golden_qa.json
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | — | From aistudio.google.com |
| `GROQ_API_KEY` | ❌ | — | Fallback LLM (optional) |
| `FLASK_SECRET_KEY` | ✅ | — | Random hex string |
| `FLASK_PORT` | ❌ | 5000 | App port |
| `QDRANT_MODE` | ❌ | local | `local` or `remote` |
| `TOP_K_RETRIEVAL` | ❌ | 10 | Chunks retrieved before rerank |
| `TOP_K_AFTER_RERANK` | ❌ | 5 | Chunks sent to LLM |
| `RETRIEVAL_SCORE_THRESHOLD` | ❌ | 0.35 | Min score to attempt answer |
| `MAX_UPLOAD_SIZE_MB` | ❌ | 50 | Max PDF size |

---

## How Citations Work

Every chunk gets a deterministic ID during ingestion: `p{page}:c{chunk_index}`

- `[p2:c3]` = page 2, 3rd chunk on that page
- IDs are stored in Qdrant alongside the vector
- When retrieved, IDs come back with the chunk
- LLM receives chunks with IDs pre-attached → copies them into the answer
- Frontend parses `[pN:cN]` patterns → makes them clickable
- Clicking jumps PDF to that page and overlays a highlight box

The LLM never invents citation IDs — it can only use the ones we provide.
