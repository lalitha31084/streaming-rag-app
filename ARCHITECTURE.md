# System Architecture & Technical Design

## 1. System Overview

The **Real-Time Streaming RAG Application** is a decoupled, asynchronous, production-grade AI system designed for low-latency question answering with continuous, non-blocking document ingestion.

The architecture strictly decouples the user-facing API and streaming engine from the heavy document processing pipeline using an asynchronous message queue.

```mermaid
flowchart TD
    subgraph Client["Frontend Client (React + Vite)"]
        UI["React Web Application"]
    end

    subgraph API_Tier["API Tier (FastAPI Asynchronous Gateway)"]
        WS["/query (WebSocket Server)"]
        INGEST["/ingest (HTTP Multipart Upload)"]
        HEALTH["/health (Service Health Check)"]
        RAG_CORE["Asynchronous RAG Engine"]
    end

    subgraph Queue_Tier["Message Broker Tier"]
        REDIS[("Redis 7 Pub/Sub: ingestion_queue")]
    end

    subgraph Worker_Tier["Background Processing Tier"]
        WORKER["Worker Process (Async Consumer)"]
        CHUNKER["Sliding Window Chunker (800 / 200 overlap)"]
        EMBEDDER["OpenAI Async Embedder (text-embedding-3-small)"]
        HASHER["SHA-256 Idempotent ID Generator"]
    end

    subgraph Storage_Tier["Vector Database Storage"]
        CHROMA[("ChromaDB Persistent Store (/data/chroma)")]
    end

    subgraph External_AI["External AI Services (OpenAI API)"]
        OAI_EMB["OpenAI Embeddings API"]
        OAI_CHAT["OpenAI Chat Completions API (Streaming)"]
    end

    %% Query Flow
    UI -->|"1. Connect & send query"| WS
    WS -->|"2. Forward query"| RAG_CORE
    RAG_CORE -->|"3. Async embedding request"| OAI_EMB
    OAI_EMB -->|"4. Return vector embedding"| RAG_CORE
    RAG_CORE -->|"5. Async query (asyncio.to_thread)"| CHROMA
    CHROMA -->|"6. Return top-k chunks + metadata"| RAG_CORE
    RAG_CORE -->|"7. In-stream citation messages"| WS
    WS -->|"8. Render citations immediately"| UI
    RAG_CORE -->|"9. Async stream request"| OAI_CHAT
    OAI_CHAT -->|"10. Token-by-token stream"| RAG_CORE
    RAG_CORE -->|"11. Forward tokens"| WS
    WS -->|"12. Render tokens real-time"| UI

    %% Ingestion Flow
    UI -->|"A. Upload file (POST /ingest)"| INGEST
    INGEST -->|"B. Async publish task (non-blocking)"| REDIS
    INGEST -->|"C. Immediate 200 OK response"| UI
    REDIS -->|"D. Deliver message"| WORKER
    WORKER -->|"E. Extract text & chunk"| CHUNKER
    CHUNKER -->|"F. Compute embeddings (async batch)"| EMBEDDER
    EMBEDDER -->|"G. Call OpenAI"| OAI_EMB
    CHUNKER -->|"H. Stable hash ID"| HASHER
    WORKER -->|"I. Upsert chunks + vectors + metadata"| CHROMA
```

---

## 2. Component Breakdown

### 2.1 Backend API Server (`backend/app/main.py`)
- **Framework**: FastAPI running on Uvicorn with `asyncio`.
- **Endpoints**:
  - `GET /health`: Returns service health status and actively verifies Redis connectivity (`redis.ping()`).
  - `POST /ingest`: Receives document files asynchronously, extracts raw text, and publishes a JSON task to the `ingestion_queue` Redis channel using non-blocking `await redis_client.publish(...)`.
  - `WS /query`: Bidirectional WebSocket endpoint handling real-time query streams.
- **Connection Management & Disconnect Safety**:
  - Implements targeted exception handling for `fastapi.websockets.WebSocketDisconnect`.
  - Cleans up client socket resources upon abrupt disconnects to prevent dangling connections and unhandled server-side exceptions.

### 2.2 Asynchronous RAG Pipeline (`backend/app/rag.py`)
- **Non-blocking Execution**:
  - Query embeddings are generated using `openai.AsyncOpenAI`.
  - ChromaDB similarity queries are wrapped in `asyncio.to_thread()` to prevent database file locks from freezing the asyncio event loop.
- **In-Stream Citation Delivery**:
  - As soon as the vector store returns the top matching documents and their relevance metadata, discrete citation events are pushed to the client via `await websocket.send_json({"type": "citation", "payload": ...})` **prior** to the generation and streaming of LLM answer tokens.
- **Token Streaming**:
  - Uses `gpt-4o-mini` with `stream=True`.
  - Each token is streamed to the client as:
    ```json
    {"type": "token", "payload": "token_string"}
    ```
  - Upon completion, a completion frame is transmitted:
    ```json
    {"type": "done", "payload": {"total_time": 1.24}}
    ```

### 2.3 Vector Database Layer (`backend/app/vectorstore.py`)
- **Storage Engine**: ChromaDB persistent client mounted to a shared volume (`/data/chroma`).
- **Stable Hashing & Idempotency**:
  - Replaced unstable Python `hash()` with cryptographically stable `hashlib.sha256(chunk.encode("utf-8")).hexdigest()`.
  - Uses `collection.upsert()` rather than `collection.add()`, guaranteeing that re-ingesting duplicate documents or re-processing redelivered messages never introduces duplicates.
- **Thread Pool Delegation**:
  - `add_documents()` and `query_documents()` both delegate ChromaDB's underlying synchronous C/SQLite calls to worker threads via `asyncio.to_thread()`, keeping the main event loop responsive.

### 2.4 Background Ingestion Worker (`backend/worker/worker.py`)
- **Isolation**: Runs as an independent process in a dedicated Docker container (`streaming-rag-worker`).
- **Message Consumption**: Subscribes to the `ingestion_queue` channel via `redis.asyncio` pub/sub.
- **Chunking Strategy**: Configured with a sliding window of 800 characters and 200 characters overlap (`chunk_overlap=200`), preserving semantic context across chunk boundaries.
- **Batch Embedding**: Converts chunks into vector embeddings via `openai_client.embeddings.create` in a single non-blocking async batch call.
- **Fault Tolerance**: Automatic reconnection loops with exponential backoff if Redis or OpenAI network errors occur.

### 2.5 Frontend Client (`frontend/src/App.jsx`)
- **Framework**: React 19 bootstrapped with Vite.
- **Real-Time Rendering**:
  - Progressive token accumulation in state without UI freezing.
  - Separate structured rendering of source citations, metadata (relevance score, source filename, chunk index), and streaming status.
  - Auto-scrolling response console for seamless reading experience.
- **File Upload Interface**:
  - Async multipart file upload to `/ingest` with visual upload feedback.

---

## 3. Communication Protocol

All WebSocket frames on `/query` use structured JSON payloads:

| Message Type | Direction | Payload Example | Description |
| :--- | :--- | :--- | :--- |
| `query` | Client → Server | Plain text (e.g., `"What is RAG?"`) | User query text |
| `citation` | Server → Client | `{"source": "report.txt", "page": 0, "snippet": "...", "relevance_score": 0.92}` | Dispatched immediately upon retrieval before token generation |
| `token` | Server → Client | `"Retrieval"` | Single LLM response token |
| `done` | Server → Client | `{"total_time": 1.18}` | Stream completed flag with total latency |
| `error` | Server → Client | `"OpenAI service rate limit exceeded"` | Propagated runtime exception |

---

## 4. Scalability & Production Readiness

```mermaid
sequenceDiagram
    autonumber
    actor User as User Browser
    participant API as FastAPI Backend
    participant Queue as Redis Message Queue
    participant Worker as Background Worker
    participant VectorDB as ChromaDB Volume
    participant OpenAI as OpenAI API

    Note over User, OpenAI: Phase 1: Asynchronous Ingestion
    User->>API: POST /ingest (Upload document)
    API->>Queue: publish("ingestion_queue", doc_payload)
    API-->>User: 200 OK {"message": "Document ingestion initiated"}
    Queue->>Worker: deliver doc_payload
    Worker->>OpenAI: await embeddings.create(chunks)
    OpenAI-->>Worker: vector embeddings
    Worker->>VectorDB: await upsert(chunks, vectors, sha256_ids)

    Note over User, OpenAI: Phase 2: Real-time Streaming Query
    User->>API: WebSocket Connect /query
    API-->>User: Accept Connection
    User->>API: Send "What is in report.txt?"
    API->>OpenAI: await embeddings.create(query)
    OpenAI-->>API: query embedding
    API->>VectorDB: await query(embedding, n_results=5)
    VectorDB-->>API: matching chunks + metadatas
    loop In-Stream Citations (Before Tokens)
        API-->>User: send_json({"type": "citation", "payload": citation})
    end
    API->>OpenAI: await chat.completions.create(stream=True)
    loop Token-by-Token Streaming
        OpenAI-->>API: delta token
        API-->>User: send_json({"type": "token", "payload": token})
    end
    API-->>User: send_json({"type": "done", "payload": {"total_time": 0.85}})
```

- **Horizontal Worker Scaling**: Workers can be scaled (`docker-compose up --scale worker=3`) by transitioning from Pub/Sub to Redis Consumer Groups (`XREADGROUP`) without touching API code.
- **Shared Storage**: ChromaDB is decoupled and backed by a Docker volume (`chroma-data`), ensuring persistence across container lifecycle events.
- **Zero Crosstalk**: Each WebSocket session maintains an isolated coroutine loop with independent memory context.