#  Mini OZOCO SidePilot AI System

**A working AI SidePilot** — six agents that observe your screen, understand your documents, detect what you want, answer with validated grounded responses, and automate actions (email drafts, exports, action plans). Every request flows through the poster pipeline: **Observe → Understand → Analyze → Guide → Automate**.

Built for the internship program at **OZOCO Global Pvt Ltd**.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash%20+%20Vision-4285F4?logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Persistent%20Memory-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

> ** Live demo:** https://ozoco-sidepilot.onrender.com — *free tier: after 15 min idle the first visit takes ~1 minute to wake up.*
> ** Repository:** https://github.com/vaibhav410/mini-ozoco-sidepilot

---

## 🧠 The Six-Agent Architecture

| Agent | File | Job |
|---|---|---|
| **Agent 1** — Document Understanding | [document_agent.py](app/agents/document_agent.py) | Classifies uploads (Resume / Invoice / Research Paper / Report / General), summarizes, extracts topics |
| **Agent 2** — Response Generation | [response_agent.py](app/agents/response_agent.py) | Condenses follow-ups, routes to the right document, generates grounded answers (blocking **and** token-streaming) |
| **Agent 3** — Validation | [validation_agent.py](app/agents/validation_agent.py) | Fact-checks every draft against its sources; unsupported answers are withheld |
| **Agent 4** — Screen Understanding | [vision_agent.py](app/agents/vision_agent.py) | Gemini Vision on screenshots → application, on-screen text, summary, intent, suggested actions; PyTesseract OCR fallback |
| **Agent 5** — Intent Detection | [intent_agent.py](app/agents/intent_agent.py) | Classifies requests into 9 intents (heuristics first, LLM for ambiguity) and recommends the workflow |
| **Agent 6** — Automation | [automation_agent.py](app/agents/automation_agent.py) | Handler registry: email drafts (Gmail/.eml), Markdown/PDF exports, action plans |

## 🔄 The Workflow Engine

Every `/ask` request travels [app/workflow/](app/workflow/):

```
            ┌─────────────────────────────────────────────────────┐
            │              WorkflowContext (shared state)         │
            └─────────────────────────────────────────────────────┘
  Observe ──→ Understand ──→ Analyze ──→ Guide ──→ Automate
  docs/history  condense +     route +     answer +   Agent 6
  screen ctx    Agent 5        retrieve    Agent 3    handlers
  guards        intent         (FAISS)     validate   (email/export)
```

- Stages communicate **only** through a shared `WorkflowContext` — each is independently replaceable (the engine depends on a `Protocol`, stages are injected).
- Every stage is timed, logged and traced; the trace is returned in the API response and drives the UI's live pipeline visualization.
- Automation intents skip the RAG answer and execute through Agent 6's handler registry instead.

## ✨ Feature Highlights

- **Screen understanding** — upload/drag a screenshot; Gemini Vision (OCR fallback) tells you what's on screen and what to do next. Screen context then **grounds your questions** ("What should I do about the invoice on my screen?") and is cited as a source.
- **Streaming** — `POST /ask/stream` (SSE): live stage progress + token-by-token answers + cancellation.
- **Automation** — "Draft an email to HR…" → real LLM-written email saved as `.eml` + mailto link; "Export a summary as PDF" → downloadable PDF. Extensible handler registry, no hardcoded branching.
- **Persistent memory** — PostgreSQL (or SQLite fallback) stores sessions, chats, detected intents, executed actions and preferences. History survives restarts; the in-memory fallback keeps requests alive if the DB is down.
- **Observability** — request IDs on every log line (honored from `X-Request-ID`), rolling per-stage and per-endpoint timings.
- **Admin dashboard** — `/admin`: live system health, process memory, documents, sessions, stage-timing bars, endpoint latencies, intent distribution.
- **Speech** — voice input via `POST /speech/transcribe` (Groq Whisper, works in every browser) and spoken answers (browser speech synthesis).
- **Resilience everywhere** — Gemini→Groq failover on all text agents, Vision→OCR→raw-text ladder on screen analysis, DB→in-memory fallback on memory, validated-or-withheld answers.

## 🌐 API

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload` | POST | Upload PDF/TXT → Agent 1 classify + index |
| `/documents` | GET | List indexed documents |
| `/ask` | POST | Question → full workflow → validated answer + sources + trace |
| `/ask/stream` | POST | Same pipeline as SSE: `stage` / `token` / `final` events |
| `/screen/analyze` | POST | Screenshot → Agent 4 screen understanding |
| `/intent/detect` | POST | Standalone Agent 5 intent classification |
| `/exports` | GET | List generated files (drafts, exports, plans) |
| `/exports/{filename}` | GET | Download a generated file |
| `/admin` | GET | Monitoring dashboard (gated by `ADMIN_TOKEN` if set) |
| `/admin/stats` | GET | Live stats JSON (gated by `ADMIN_TOKEN` if set) |
| `/speech/transcribe` | POST | Voice input → transcribed text (Groq Whisper) |
| `/health` | GET | Health check |

Interactive Swagger docs at **`/docs`**.

## 🚀 Run It

### Local (SQLite, local embeddings)

```bash
git clone https://github.com/vaibhav410/mini-ozoco-sidepilot.git
cd mini-ozoco-sidepilot
pip install -r requirements.txt
cp .env.example .env        # add your GOOGLE_API_KEY
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

### Docker (production stack: app + PostgreSQL)

```bash
echo "GOOGLE_API_KEY=your_key" > .env
docker compose up --build
# app:   http://localhost:8000
# admin: http://localhost:8000/admin
```

The image is lean (Gemini embedding API, no PyTorch), runs as a non-root user, includes Tesseract for the OCR fallback, and has a container health check. Postgres data, uploads and exports persist in named volumes.

### Render (free tier)

Deployed via [render.yaml](render.yaml) — uses `requirements-render.txt` (no PyTorch) with `EMBEDDINGS_BACKEND=gemini`. Set `GOOGLE_API_KEY` (and optionally `GROQ_API_KEY`, `DATABASE_URL` for a hosted Postgres like Neon) in the Render dashboard.

## 📁 Folder Structure

```
app/
├── main.py                  # FastAPI app, middleware (request IDs + metrics), startup validation
├── config.py                # every setting from .env, one frozen dataclass
├── routes/                  # HTTP layer only: upload, chat (+stream), screen, intent, exports, admin
├── agents/                  # the six agents + shared LLM factory (Gemini→Groq failover)
├── workflow/                # the engine: context, five stages, tracing
├── rag/                     # loader, splitter, embeddings, FAISS store, retriever, all prompts
├── services/                # orchestration: documents, chat/stream, screen, OCR, memory
├── integrations/            # gmail (.eml + API-ready), export (md/pdf), filesystem
├── db/                      # SQLAlchemy engine + models (sessions, messages, intents, actions, prefs)
├── models/schemas.py        # every API contract (Pydantic)
└── utils/                   # errors, logger (request-id aware), metrics, json parsing
static/                      # index.html (SidePilot UI) + admin.html (dashboard)
```

## ⚙️ Configuration

All via environment variables (see [.env.example](.env.example)): `GOOGLE_API_KEY` (required), `GROQ_API_KEY` (optional failover + Whisper voice input), `GEMINI_MODEL` / `GEMINI_VISION_MODEL` / `GEMINI_VISION_FALLBACK_MODEL`, `EMBEDDINGS_BACKEND` (`local`/`gemini`), `DATABASE_URL` (Postgres; SQLite default), `INDEX_DIR` (where the FAISS index is persisted), `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K`, upload/image size limits, `TESSERACT_CMD`, `GMAIL_TOKEN_JSON`, `ADMIN_TOKEN`, `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW`.

## 🔒 Security & Limits

- **Rate limiting** — a per-IP sliding-window limiter (`RATE_LIMIT_REQUESTS` per `RATE_LIMIT_WINDOW` seconds, default 40/60s) covers `/ask`, `/upload`, `/screen/analyze`, `/intent/detect`, `/speech/transcribe` and every mutating request, returning `429` once exceeded.
- **Bounded request bodies** — a `Content-Length` guard rejects oversized uploads/screenshots/audio with `413` *before* the body is buffered into memory, so a large request can't be used to exhaust RAM.
- **Admin auth** — `/admin`, `/admin/stats` and `DELETE /documents/{id}` are open by default (single-user local demo); set `ADMIN_TOKEN` to require it via `Authorization: Bearer <token>`, `X-Admin-Token`, or `?token=` on the dashboard URL.
- **Input validation** — `session_id` is capped at 64 chars (`[A-Za-z0-9_.:-]` only) to stay valid across both SQLite and Postgres; uploaded `.txt` files are rejected if they're not mostly printable text (binary/garbage doesn't get indexed).
- **Concurrency safety** — the FAISS index and document registry are guarded by a lock, so concurrent uploads/deletes can't corrupt shared state.
- **Known limitation (single-tenant)** — the document store is process-global, not scoped per session/user: anyone hitting this instance can see and (with `ADMIN_TOKEN`, delete) every uploaded document. This matches the assignment's single-user design; true multi-tenant isolation would need per-user auth and a namespaced store.

## 💾 Persistence

- **Chat history, intents, actions, preferences** — always in `DATABASE_URL` (SQLite by default, Postgres in production); survive restarts.
- **Uploaded documents (FAISS index + registry)** — persisted to `INDEX_DIR` and reloaded on startup, so they also survive a restart. On Render's free tier the filesystem is ephemeral (no attached disk), so this survives process restarts but not redeploys/evictions — see the note in [render.yaml](render.yaml).

## 🛡️ Design Principles

- **Routes → services → agents/rag layering**: no FastAPI types below the HTTP layer; domain errors (`AppError`) map to clean HTTP responses.
- **Graceful degradation as a contract**: every external dependency (Gemini, Groq, Tesseract, PostgreSQL, Gmail) has a documented fallback; the request never dies with the dependency.
- **Open/closed pipeline**: new stages, intents and automation handlers register without touching the engine, the agent or the API.
- **Additive API evolution**: every upgrade added fields (`workflow`, `intent`, `automation`) — no client ever broke.
- **Prompts are configuration**: all templates live in [prompt.py](app/rag/prompt.py), reviewable at a glance.
