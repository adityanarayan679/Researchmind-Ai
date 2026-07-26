# ResearchMind AI

A production-quality Retrieval-Augmented Generation (RAG) research assistant. Upload PDFs, ask questions, and get grounded answers with source citations.

Built entirely with **free technologies** — no paid API keys required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    app.py (Streamlit)                    │
│  ┌──────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Sidebar  │  │   Chat Area      │  │  Status Bar   │  │
│  │ Upload   │  │   Messages       │  │  Loading...   │  │
│  │ Docs     │  │   Streaming      │  │  Ready        │  │
│  └────┬─────┘  └────────┬─────────┘  └───────────────┘  │
└───────┼──────────────────┼──────────────────────────────┘
        │                  │
┌───────▼──────────────────▼──────────────────────────────┐
│                    RAG Pipeline                          │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────┐   │
│  │ PDF      │──▶│ Chunker  │──▶│ EmbeddingClient    │   │
│  │ Loader   │   │          │   │ SentenceTransformers│   │
│  └──────────┘   └──────────┘   └─────────┬──────────┘   │
│                                          │              │
│  ┌──────────┐   ┌──────────┐   ┌─────────▼──────────┐   │
│  │ Gemini   │◀──│ Prompt   │◀──│ VectorStore (FAISS)│   │
│  │ LLM      │   │ Builder  │   │                    │   │
│  └──────────┘   └──────────┘   └────────────────────┘   │
│                     ▲                                    │
│  ┌──────────┐       │                                    │
│  │ Citations│       │                                    │
│  │ Builder  │       │                                    │
│  └──────────┘       │                                    │
│  ┌──────────┐       │                                    │
│  │ Chat     │───────┘                                    │
│  │ Memory   │                                            │
│  └──────────┘                                            │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Upload** — PDFs are uploaded via the sidebar
2. **Extract** — `PDFLoader` extracts text page-by-page
3. **Chunk** — `TextChunker` splits into overlapping segments
4. **Embed** — `EmbeddingClient` converts chunks to vectors (384-dim)
5. **Index** — `VectorStore` stores vectors in a FAISS index
6. **Question** — User types a question
7. **Retrieve** — Question is embedded, FAISS returns top-k similar chunks
8. **Prompt** — `PromptBuilder` assembles chunks + conversation history → prompt
9. **Generate** — `LLMClient` sends prompt to Gemini, streams the answer
10. **Cite** — `CitationBuilder` formats source references

---

## Features

### Core
| Feature | Implementation |
|---------|---------------|
| RAG Pipeline | Full retrieval-augmented generation with Gemini |
| Multi-PDF Support | Upload and query across multiple documents |
| Semantic Search | FAISS index with cosine similarity (384-dim) |
| Streaming Answers | Real-time token-by-token response display |
| Source Citations | Document name, page number, and relevance score |
| Conversation Memory | Last 5 exchanges preserved for follow-up context |

### UI
| Feature | Implementation |
|---------|---------------|
| Sidebar Document Manager | Upload, stats, delete all |
| Chat Interface | Streaming message bubbles |
| Retrieved Context Viewer | Expandable panel showing matched chunks |
| Loading Status | Step-by-step initialization progress |
| Progress Indicators | Upload and search progress bars |

### Error Handling
| Scenario | Behavior |
|----------|----------|
| Missing API key | Clear error message on startup |
| Corrupted PDF | Skipped with error message, other PDFs processed |
| Encrypted PDF | Rejected with explanation |
| No documents loaded | Chat input disabled with guidance |
| Empty search results | Graceful "no relevant documents" response |
| LLM quota exceeded | Error displayed in chat |

---

## Project Structure

```
ResearchMind/
├── app.py                 # Streamlit entry point
├── config/
│   ├── __init__.py
│   └── settings.py        # Central configuration (API keys, chunk size, etc.)
├── backend/
│   ├── __init__.py
│   ├── pdf_loader.py      # PDF text extraction with PyMuPDF
│   ├── chunker.py         # Text chunking with overlap
│   ├── embeddings.py      # Vector embeddings via SentenceTransformers
│   ├── vector_store.py    # FAISS index management
│   ├── retriever.py       # Question → embedding → search orchestration
│   ├── prompt_builder.py  # Grounded prompt construction
│   ├── llm.py             # Gemini API client
│   ├── chat_memory.py     # Conversation history management
│   └── citations.py       # Source citation formatting
├── frontend/
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_pdf_loader.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_prompt_builder.py
│   ├── test_citations.py
│   └── test_chat_memory.py
├── data/                  # Uploaded PDFs and vector store cache
│   └── .gitkeep
├── .env.example           # Environment variable template
├── .gitignore
├── requirements.txt
├── run.bat                # Windows launcher
├── run.ps1                # PowerShell launcher
└── README.md
```

---

## Installation

### Prerequisites

- **Python 3.12+**
- **A Gemini API key** — get one free at [Google AI Studio](https://aistudio.google.com/app/apikey)

### Local Setup

**Windows:**
```
git clone https://github.com/yourusername/researchmind-ai.git
cd researchmind-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
git clone https://github.com/yourusername/researchmind-ai.git
cd researchmind-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_key_here
```

### Run

```bash
streamlit run app.py
```

Or on Windows, double-click `run.bat`.

---

## Usage

1. **Open** http://localhost:8501 in your browser
2. **Upload PDFs** using the sidebar file uploader
3. **Wait** for processing — you'll see page and chunk counts
4. **Ask questions** in the chat input at the bottom
5. **View answers** with source citations below each response
6. **Expand "Retrieved Context"** to see which chunks were used

### Tips

- Upload **multiple PDFs** on the same topic for better answers
- Ask **follow-up questions** — the app remembers the last 5 exchanges
- Use the **Clear Chat** button to start a fresh conversation
- Use **Clear All** to remove all documents and start over

---

## Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

Or run individual test modules:
```bash
python -m tests.test_pdf_loader
python -m tests.test_chunker
python -m tests.test_embeddings
python -m tests.test_vector_store
python -m tests.test_retriever
python -m tests.test_prompt_builder
python -m tests.test_citations
python -m tests.test_chat_memory
```

---

## Deployment

### Streamlit Community Cloud (Free)

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"Deploy an app"**
4. Select your repo, branch, and set `app.py` as the entry point
5. Under **"Secrets"**, add your `GEMINI_API_KEY`
6. Click **"Deploy"**

### Hugging Face Spaces (Free)

1. Push the repository to GitHub
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
3. Click **"Create new Space"**
4. Choose **"Streamlit"** as the SDK
5. Connect your GitHub repo
6. Add `GEMINI_API_KEY` in the Space's **"Secrets"** settings

---

## Tech Stack

| Category | Technology | Why |
|----------|-----------|-----|
| Language | Python 3.12+ | Ecosystem, readability |
| Frontend | Streamlit | Fast UI prototyping, Python-native |
| LLM | Google Gemini API | Free tier, competitive quality |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, fast, runs locally |
| Vector DB | FAISS | CPU-optimized, free, simple |
| PDF | PyMuPDF | Fast, reliable, no dependencies |

---

## Development Roadmap

### Completed
- [x] Project setup and configuration
- [x] Gemini API integration with streaming
- [x] PDF text extraction with page tracking
- [x] Text chunking with configurable overlap
- [x] Vector embeddings with caching
- [x] FAISS vector store with save/load
- [x] Complete RAG pipeline
- [x] Source citations with deduplication
- [x] Conversation memory for follow-ups
- [x] Professional Streamlit UI

### Stretch Goals
- [ ] Hybrid search (BM25 + dense retrieval)
- [ ] OCR for scanned PDFs
- [ ] Markdown export of answers
- [ ] Quiz / flashcard generator
- [ ] Document comparison
- [ ] Local LLM support (Ollama)
- [ ] Web search tool

---

## License

MIT
