# Real-Time Streaming RAG Application with FastAPI

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Redis](https://img.shields.io/badge/Redis-7.0-DC382D.svg?logo=redis&logoColor=white)](https://redis.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6F00.svg)](https://www.trychroma.com/)
[![React](https://img.shields.io/badge/Frontend-React%2019-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)

A full-stack, production-grade, real-time Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Redis Pub/Sub**, **ChromaDB**, and **React**. The system provides real-time token-by-token streaming answers and immediate citations over WebSockets while simultaneously ingesting and embedding new documents in the background via a decoupled message queue.

---

## Architecture Overview

```
User (Browser)
   ↕  WebSocket (/query) & HTTP (/ingest)
FastAPI Backend (Non-blocking async API Gateway)
   ├───> Publishes tasks to Redis Pub/Sub ("ingestion_queue")
   │        └───> Background Ingestion Worker (Independent container)
   │                 ├──> Sliding-window chunker (800 chars / 200 overlap)
   │                 ├──> Async OpenAI Embeddings (text-embedding-3-small)
   │                 └──> SHA-256 Idempotent Upsert to ChromaDB
   │
   └───> Real-Time Streaming RAG
            ├──> Query embedding generation (async)
            ├──> Vector search via thread pool (asyncio.to_thread)
            ├──> Discrete in-stream citations dispatched BEFORE tokens
            └──> Token-by-token LLM stream (gpt-4o-mini)
```

For detailed architectural diagrams and sequence flows, refer to [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Core Capabilities & Implemented Requirements

1. **Token-by-Token Streaming (`/query`)**:
   - WebSockets endpoint provides real-time streaming output using OpenAI's async streaming API.
   - Structured JSON protocol with discrete message types: `citation`, `token`, `done`, and `error`.
   - Dedicated exception handling for `fastapi.websockets.WebSocketDisconnect` guarantees graceful disconnection and server stability.

2. **In-Stream Citation Delivery**:
   - Rather than bundling citations at the end of response generation, source citations and relevance scores are streamed to the client immediately upon vector retrieval, prior to LLM token streaming.

3. **Decoupled Asynchronous Ingestion Pipeline**:
   - HTTP `POST /ingest` endpoint accepts document uploads and immediately offloads processing tasks by publishing them to Redis Pub/Sub (`ingestion_queue`).
   - Dedicated background worker (`backend/worker/worker.py`) runs in a separate process/container, consumes messages, splits text into overlapping windows, generates embeddings, and upserts them to ChromaDB.
   - Re-indexing is idempotent using SHA-256 deterministic chunk identifiers.

4. **Fully Non-Blocking Asynchronous Core**:
   - Uses `AsyncOpenAI` for both chat completions and embeddings.
   - Uses `redis.asyncio` for non-blocking message publication and subscription.
   - Offloads synchronous ChromaDB operations to background threads using `asyncio.to_thread`.

5. **Containerized Multi-Service Orchestration**:
   - Fully containerized with separate `Dockerfile.api` and `Dockerfile.worker`.
   - Orchestrated via `docker-compose.yml` linking the API server, background worker, Redis 7, and React frontend with shared persistent storage for ChromaDB.

---

## Project Structure

```
streaming-rag-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py              # Environment settings (Pydantic / os.getenv)
│   │   ├── main.py                # FastAPI routes (/health, /ingest, /query WS)
│   │   ├── rag.py                 # Async RAG pipeline & token streaming
│   │   ├── vectorstore.py         # Async ChromaDB access with SHA-256 IDs
│   │   ├── ingest.py              # Text extraction and chunking utilities
│   │   ├── utils.py               # Performance timing decorator
│   │   └── websocket_manager.py   # Connection tracker
│   ├── worker/
│   │   └── worker.py              # Background Redis consumer and indexer
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py            # Pytest test fixtures
│   │   ├── test_health.py         # Health endpoint test
│   │   ├── test_rag.py            # Ingest, WebSocket, & citation stream tests
│   │   └── test_worker.py         # Text chunking logic unit tests
│   ├── Dockerfile.api             # Backend API container definition
│   ├── Dockerfile.worker          # Worker process container definition
│   ├── requirements.txt           # Python dependencies
│   └── pytest.ini                 # Pytest configuration
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Streaming UI & file upload component
│   │   ├── App.css                # Component styling
│   │   ├── index.css              # Global styles
│   │   └── main.jsx               # Entry point
│   ├── Dockerfile                 # Multi-stage production Nginx container
│   ├── nginx.conf                 # Reverse proxy configuration
│   ├── index.html                 # HTML index
│   └── package.json               # Frontend dependencies
├── docker-compose.yml             # Orchestration for API, Worker, Redis, Frontend
├── submission.yml                 # Automated evaluation commands
├── .env.example                   # Environment configuration template
├── README.md                      # Project documentation
├── ARCHITECTURE.md                # System architecture documentation
└── BENCHMARKS.md                  # Latency & throughput benchmark report
```

---

## Quick Start & Setup

### Prerequisites
- Docker & Docker Compose v2+ installed
- Node.js 18+ and Python 3.11+ (if running outside Docker)
- OpenAI API Key

### 1. Configure Environment Variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Populate `.env` with your OpenAI API key and service ports:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
REDIS_HOST=redis
REDIS_PORT=6379
VECTOR_DB_PATH=/data/chroma
FRONTEND_URL=http://localhost:5173
BACKEND_URL=http://localhost:8000
```

### 2. Launch with Docker Compose
Run the entire decoupled stack in detached mode:
```bash
docker-compose up --build -d
```
This starts four synchronized containers:
- `streaming-rag-backend` on port `8000`
- `streaming-rag-worker` (processing queue events)
- `streaming-rag-redis` on port `6379`
- `streaming-rag-frontend` on port `5173`

Verify the health check:
```bash
curl http://localhost:8000/health
# Response: {"status":"ok","redis":true}
```

Access the frontend application at **http://localhost:5173**.

---

## Running the Automated Test Suite

Tests validate the health check, Redis publishing during ingestion, in-stream citation ordering, WebSocket disconnect handling, and text chunking logic:

```bash
docker-compose exec backend pytest -v
```

To run tests locally:
```bash
cd backend
pip install -r requirements.txt
pytest -v
```

---

## WebSocket Protocol Specification

The `/query` WebSocket endpoint expects plain text queries from the client and transmits JSON messages:

### In-Stream Citation Event (Sent before tokens)
```json
{
  "type": "citation",
  "payload": {
    "source": "quarterly_report.txt",
    "page": 2,
    "snippet": "Gross revenue increased by 18% year-over-year...",
    "relevance_score": 0.8842
  }
}
```

### Token Event (Sent iteratively)
```json
{
  "type": "token",
  "payload": "Revenue "
}
```

### Completion Event
```json
{
  "type": "done",
  "payload": {
    "total_time": 1.14
  }
}
```

### Error Event
```json
{
  "type": "error",
  "payload": "ChromaDB connection unavailable"
}
```

---

## Performance Summary

Detailed test methodology and results are documented in [BENCHMARKS.md](BENCHMARKS.md).

| Metric | Measured Result | Target SLA |
| :--- | :--- | :--- |
| **Time-to-First-Token (TTFT)** | **294 ms - 412 ms** | < 500 ms |
| **Document Ingestion Time** | **0.42 s - 3.20 s** | < 10 s |
| **Concurrent Users Tested** | **10 concurrent sessions** | >= 10 users |
| **Session Crosstalk** | **0% (fully isolated coroutines)** | Zero crosstalk |