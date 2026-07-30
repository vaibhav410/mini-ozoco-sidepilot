#  Mini OZOCO SidePilot AI System

**Document intelligence with conversational RAG and a three-agent workflow** — upload a PDF/TXT, let **Agent 1** understand and classify it, ask questions (with follow-ups — full chat history support) answered by **Agent 2** with grounded, source-referenced responses, and let **Agent 3** validate every answer against the sources before it reaches you.

Built as the practical assignment for the internship selection process at **OZOCO Global Pvt Ltd**.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF)

> ** Live demo:** https://ozoco-sidepilot.onrender.com — *free tier: after 15 min idle the first visit takes ~1 minute to wake up, then it's fast.*
> ** Demo video:** [video.mp4](video.mp4) — walkthrough of upload, both agents, grounded answers, and the anti-hallucination check
> ** Repository:** https://github.com/vaibhav410/mini-ozoco-sidepilot

---

## 📖 Project Overview

Mini OZOCO SidePilot is a small AI system that answers natural-language questions about your documents — **without hallucinating**. Every answer is generated strictly from retrieved document content (Retrieval-Augmented Generation), and when the answer is not in your documents, the system says so explicitly.

**The three-agent workflow:**

| Agent | Runs at | Job |
|---|---|---|
| 🧠 **Agent 1** — Document Understanding | Upload time | Extracts text, classifies the document (Resume / Invoice / Research Paper / Report / General Document), writes a summary, produces metadata |
| 💬 **Agent 2** — Response Generation & Routing | Question time | Resolves follow-up questions using chat history, routes the question to the relevant document, retrieves top-k chunks from FAISS, builds a grounded prompt, generates the answer **with source references** |
| 🛡️ **Agent 3** — Answer Validation | After each answer | Fact-checks Agent 2's draft against the retrieved context; unsupported answers are withheld and replaced with an explicit "not found" |

The agents communicate through shared structured state: Agent 1 stamps every chunk with metadata and fills a document registry; Agent 2 reads both to route and answer; Agent 3 receives Agent 2's draft plus its evidence and returns a structured verdict.

## 📌 Requirement → Implementation Map

Every expected capability, and the exact file implementing it:

| Capability | Implementation |
|---|---|
| Text chunking (1000 chars / 150 overlap) | [app/rag/splitter.py](app/rag/splitter.py) — `RecursiveCharacterTextSplitter` |
| Embeddings | [app/rag/embeddings.py](app/rag/embeddings.py) — HuggingFace `all-MiniLM-L6-v2` local; Gemini embedding API backend for cloud |
| Vector database | [app/rag/vector_store.py](app/rag/vector_store.py) — **FAISS** with per-chunk metadata + filtered search |
| Retrieval | [app/rag/retriever.py](app/rag/retriever.py) — top-k semantic search with document filter |
| Agent 1: document processing | [app/agents/document_agent.py](app/agents/document_agent.py) |
| Agent 2: question answering + routing | [app/agents/response_agent.py](app/agents/response_agent.py) |
| Agent 3: answer validation | [app/agents/validation_agent.py](app/agents/validation_agent.py) |
| Agent orchestration | [app/services/chat_service.py](app/services/chat_service.py) — condense → route → retrieve → generate → validate |
| Chat history / follow-up questions | [app/services/history.py](app/services/history.py) + condensation in Agent 2 |
| Multi-document support | Registry + metadata routing; `GET /documents`; per-document scope in the UI |
| Error handling (invalid/large files, fallbacks) | [app/utils/errors.py](app/utils/errors.py), service guards, Gemini→Groq failover in [app/agents/llm.py](app/agents/llm.py) |
| Prompt engineering (grounding + refusal rule) | [app/rag/prompt.py](app/rag/prompt.py) — all prompts in one reviewable file |
| API-based design | FastAPI routes in [app/routes/](app/routes/), Swagger at `/docs` |

## 🏗️ Architecture

```
        ┌──────────────────────────────┐
        │            Client            │
        │   Web UI · Swagger · curl    │
        └──────────────┬───────────────┘
                       │ HTTP
        ┌──────────────▼───────────────┐
        │       FastAPI Backend        │
        │   /upload   /ask   /health   │
        └──────┬───────────────┬───────┘
       upload  │               │  question + chat history
        ┌──────▼──────┐ ┌──────▼───────┐ ┌──────────────┐
        │   AGENT 1   │ │   AGENT 2    │→│   AGENT 3    │
        │  Classify + │ │  Condense +  │ │  Validate vs │
        │  Summarize  │ │ Route+Answer │←│   sources    │
        └──────┬──────┘ └──────┬───────┘ └──────────────┘
               │               │ top-k retrieval
        ┌──────▼───────────────▼───────┐
        │    FAISS Vector Store        │
        │  chunks + embeddings + meta  │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼───────────────┐
        │  Google Gemini 2.5 Flash     │
        │       (via LangChain)        │
        └──────────────────────────────┘
```

**RAG pipeline:** `PDF/TXT → PyPDFLoader → RecursiveCharacterTextSplitter (1000/150) → HuggingFace all-MiniLM-L6-v2 embeddings (local) → FAISS → top-4 similarity retrieval → grounded prompt → Gemini → answer + sources`

## 🧰 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | **FastAPI + Uvicorn** | Auto Swagger docs, built-in validation, async-ready |
| AI orchestration | **LangChain** | Loaders, splitters, FAISS + Gemini integrations |
| LLM | **Google Gemini (`gemini-2.5-flash`)** | Fast, generous free tier, both agents use it |
| LLM fallback | **Groq (`llama-3.3-70b-versatile`)** | Optional: agent calls automatically fail over to Groq when Gemini is rate-limited or down (`GROQ_API_KEY` in `.env`) |
| Embeddings | **HuggingFace `all-MiniLM-L6-v2`** | Runs locally — free, fast, no API quota |
| Vector store | **FAISS (cpu)** | In-process similarity search, zero setup |
| PDF parsing | **PyPDF** | Page-aware extraction for source references |
| Config | **python-dotenv** | Secrets in `.env`, never in code |

## 📁 Folder Structure

```
project/
├── app/
│   ├── main.py                 # FastAPI app + routes wiring + startup checks
│   ├── config.py               # all settings from .env, in one place
│   ├── routes/                 # HTTP layer only
│   │   ├── upload.py           #   POST /upload
│   │   └── chat.py             #   POST /ask
│   ├── agents/                 # the three AI agents
│   │   ├── llm.py              #   Gemini primary + automatic Groq fallback
│   │   ├── document_agent.py   #   Agent 1: classify + summarize
│   │   ├── response_agent.py   #   Agent 2: condense + route + grounded answer
│   │   └── validation_agent.py #   Agent 3: fact-check answers vs sources
│   ├── rag/                    # one mechanical job per file
│   │   ├── loader.py           #   PDF/TXT -> text
│   │   ├── splitter.py         #   text -> chunks
│   │   ├── embeddings.py       #   chunks -> vectors (local model)
│   │   ├── vector_store.py     #   FAISS index + document registry
│   │   ├── retriever.py        #   top-k similarity search
│   │   └── prompt.py           #   all prompt templates
│   ├── services/               # orchestration between routes and agents
│   │   ├── document_service.py #   upload pipeline: validate -> Agent 1 -> index
│   │   ├── chat_service.py     #   ask pipeline: history -> Agent 2 -> Agent 3
│   │   └── history.py          #   session chat history (multi-turn memory)
│   ├── models/schemas.py       # Pydantic API contracts
│   └── utils/                  # logging + error hierarchy
├── static/index.html           # single-page premium UI (served at /)
├── samples/                    # sample documents for the demo
├── uploads/                    # uploaded files (git-ignored)
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙️ Installation

**Prerequisites:** Python 3.11+ (3.12 recommended) and a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
# 1. Clone
git clone https://github.com/vaibhav410/mini-ozoco-sidepilot.git
cd mini-ozoco-sidepilot

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Dependencies
pip install -r requirements.txt

# 4. Configuration
copy .env.example .env        # Windows (cp on macOS/Linux)
# then edit .env and paste your GOOGLE_API_KEY
```

> First run downloads the embedding model (~90 MB) automatically.

## ▶️ Running

```bash
uvicorn app.main:app --reload
```

| What | URL |
|---|---|
| 🖥️ Web UI | http://localhost:8000 |
| 📘 Swagger API docs | http://localhost:8000/docs |
| ❤️ Health check | http://localhost:8000/health |

## 🔌 API Endpoints

### `POST /upload` — upload & process a document
```bash
curl -X POST http://localhost:8000/upload -F "file=@samples/sample_resume.txt"
```
```json
{
  "doc_id": "a1b2c3d4",
  "filename": "sample_resume.txt",
  "category": "Resume",
  "summary": "Resume of an AI/ML engineer with 2 years of experience...",
  "topics": ["python", "langchain", "fastapi"],
  "chunks_indexed": 5,
  "status": "indexed"
}
```
Errors: `400` unsupported/empty file · `413` too large · `502` Gemini unavailable

### `POST /ask` — ask a question (multi-turn)
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d "{\"question\": \"What skills are listed in the resume?\", \"session_id\": \"demo\"}"
```
```json
{
  "answer": "The resume lists: Python, SQL, LangChain, RAG pipelines, FastAPI...",
  "routed_document": "sample_resume.txt",
  "sources": [
    {"filename": "sample_resume.txt", "page": null, "snippet": "TECHNICAL SKILLS - Languages: Python, SQL..."}
  ],
  "found": true,
  "validation": {"checked": true, "supported": true, "confidence": "high"}
}
```
Follow-up questions in the same `session_id` use chat history — *"and what about his education?"* just works.
Off-topic questions return `"found": false` with an explicit *not found* answer — **no hallucination**. Answers Agent 3 cannot verify against the sources are **withheld**.
Errors: `400` no documents yet / empty question · `404` unknown doc_id · `502` AI providers unavailable

### `GET /documents` — list indexed documents
```json
{ "documents": [ {"doc_id": "a1b2c3d4", "filename": "resume.pdf", "category": "Resume", "summary": "...", "chunks": 5} ] }
```

### `GET /health`
```json
{ "status": "ok", "documents_indexed": 2 }
```

## 🔄 Workflow

1. **Upload** — file is validated (type, ≤10 MB), saved, and its text extracted.
2. **Agent 1** classifies the document, writes a 2–3 sentence summary and key topics.
3. **Indexing** — text is chunked (1000 chars, 150 overlap), embedded, and stored in FAISS with Agent 1's metadata on every chunk.
4. **Ask** — if the question is a follow-up, **Agent 2** first condenses it into a standalone question using the session's chat history; then it routes to the right document using the registry, retrieves the top-4 most similar chunks, and builds a grounded prompt.
5. **Draft** — the LLM responds using *only* the retrieved context, or emits an explicit refusal when the context doesn't contain the answer.
6. **Agent 3** fact-checks the draft against the retrieved chunks; only supported answers are returned (with a validation verdict and source snippets) — unsupported drafts are withheld.

## ☁️ Deployment (Render)

The live demo runs on Render's free tier via [render.yaml](render.yaml). Because the free instance has 512 MB RAM, the deployed app sets `EMBEDDINGS_BACKEND=gemini` — embeddings come from Google's embedding API instead of the local PyTorch model (see [app/rag/embeddings.py](app/rag/embeddings.py)); everything else is identical to local. Chat calls automatically fall back from Gemini to Groq under rate limits.

## ⚠️ Known Limitations

- Text-based PDFs only (no OCR for scanned documents).
- FAISS index is in-memory — re-upload documents after a server restart.
- Answer latency depends on Gemini API availability and free-tier limits.
- Built for demonstration/evaluation, not production scale.

## 🔮 Future Improvements

- Persist the FAISS index and registry to disk
- OCR support for scanned PDFs
- Streaming answers (Server-Sent Events)
- Docker image for one-command setup
- Retrieval quality evaluation script

## 👤 Author

**Vaibhav Kumar Kanojia** — [GitHub](https://github.com/vaibhav410)

*Submitted for the OZOCO Global Pvt Ltd internship selection assignment.*
