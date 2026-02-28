# 🚀 Real-Time Streaming RAG Application with FastAPI

## 📌 Overview

This project implements a **production-grade, full-stack, real-time Retrieval-Augmented Generation (RAG) system**.

The system streams LLM-generated responses token-by-token over WebSockets while simultaneously allowing new documents to be ingested asynchronously into a vector database using a decoupled message queue architecture.

It demonstrates:

- Asynchronous FastAPI backend
- WebSocket-based streaming
- Redis Pub/Sub ingestion pipeline
- ChromaDB vector database
- Background worker processing
- Dockerized deployment
- React frontend UI
- End-to-end real-time RAG system

---

## 🏗 System Architecture

High-level architecture:

```
User (Browser)
  ↓
React Frontend
  ↓ (WebSocket / HTTP)
FastAPI Backend
  ↓ (Publish)
Redis Pub/Sub
  ↓
Background Worker
  ↓
ChromaDB Vector Store
  ↑
FastAPI Retrieval
  ↑
Streaming Tokens + Citations
```

For full architecture explanation, see:  
📄 `ARCHITECTURE.md`

---

## ✨ Features

### 🔥 Real-Time Streaming
- Token-by-token streaming responses
- WebSocket communication
- Citation information streamed during response

### ⚡ Asynchronous Ingestion
- Document upload via `/ingest`
- Redis Pub/Sub message queue
- Background worker for chunking & embedding
- Documents searchable within seconds

### 🧠 Fully Async RAG Pipeline
- Async FastAPI endpoints
- Streaming LLM responses
- Non-blocking I/O
- Incremental context building

### 🐳 Containerized Deployment
- Backend container
- Worker container
- Redis container
- Docker Compose orchestration

---

## 🛠 Tech Stack

### Backend
- FastAPI
- Uvicorn
- OpenAI API
- Redis
- ChromaDB
- Pytest

### Frontend
- React (Vite)
- WebSockets

### Infrastructure
- Docker
- Docker Compose

---

## 📂 Project Structure

```
streaming-rag-app/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── rag.py
│   │   ├── vectorstore.py
│   │
│   ├── worker/
│   │   └── worker.py
│   │
│   ├── tests/
│   │   └── test_health.py
│   │
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│
├── docker-compose.yml
├── submission.yml
├── README.md
├── ARCHITECTURE.md
└── BENCHMARKS.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <your-repository-url>
cd streaming-rag-app
```

### 2️⃣ Create `.env` File

Create a file named:

- `.env`

Add:

```env
OPENAI_API_KEY=your_openai_api_key
REDIS_HOST=redis
REDIS_PORT=6379
VECTOR_DB_PATH=/data/chroma
```

### 3️⃣ Run with Docker (Recommended)

```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

Services started:

- Backend → http://localhost:8000
- Redis → localhost:6379
- Worker → Background ingestion processor

### 4️⃣ Start Frontend

Open new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open browser:

- http://localhost:5173

---

## 🧪 Testing

### Health Check

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

### Run Backend Tests

```bash
docker-compose exec backend pytest
```

Expected:

- `1 passed`

---

## 📥 How to Use the Application

### 1️⃣ Upload a Document
- Upload a `.txt` file using the frontend
- Worker processes it asynchronously
- Embeddings stored in ChromaDB

Terminal output:

```
Worker started...
Processed filename.txt
```

### 2️⃣ Ask a Question
- Enter a question related to uploaded document
- Click **Ask**
- Response streams token-by-token
- Citations displayed after streaming

---

## 📊 Performance

See full benchmark details in:

📄 `BENCHMARKS.md`

### Measured Metrics

| Metric | Result |
|---|---|
| Time-to-first-token | ~300–400ms |
| Total response time | 1–3 seconds |
| Document ingestion time | 3–7 seconds |
| Concurrent users tested | 10 |

---

## 🧱 Design Principles

- Decoupled microservice architecture
- Fully asynchronous backend
- Message queue for ingestion
- Stateless API server
- Scalable worker model
- Production-ready containerization

---

## 🚨 Error Handling

- `WebSocketDisconnect` handled
- Streaming error propagation
- Worker idempotent design
- Graceful backend exception handling
- Health endpoint for monitoring

---

## 📌 Submission Compliance Checklist

| Requirement | Status |
|---|---|
| WebSocket Streaming | ✅ |
| Citation Streaming | ✅ |
| Async Ingestion | ✅ |
| Message Queue | ✅ |
| Background Worker | ✅ |
| Vector DB | ✅ |
| Dockerized | ✅ |
| `submission.yml` | ✅ |
| Architecture Diagram | ✅ |
| Benchmark Report | ✅ |
| Health Endpoint | ✅ |
| Pytest Included | ✅ |

---

## 🔐 Security Considerations

- Environment variables for API keys
- No hardcoded secrets
- Input validation via FastAPI
- Container isolation

---

## 📈 Future Improvements

- Add authentication
- Multi-user session isolation
- Horizontal worker scaling
- Advanced chunking strategies
- Hybrid search (BM25 + vector)
- Retry mechanisms for LLM failures
- Load balancer support