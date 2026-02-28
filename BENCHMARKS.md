# Performance Benchmark Report

## Test Environment

- Docker Compose
- Redis 7
- FastAPI + Uvicorn
- ChromaDB local storage

---

## Latency Metrics

| Metric | Result |
|--------|--------|
| Time-to-first-token | ~350ms |
| Total response time | 1.5 – 3 seconds |
| Document ingestion time | 3 – 7 seconds |

---

## Load Testing

Tested with 10 concurrent users.

Results:

- No cross-talk between sessions
- No crashes
- Stable WebSocket streaming
- Worker processed documents within seconds

---

## Optimization Techniques

- Async FastAPI endpoints
- Streaming ChatCompletion
- Redis Pub/Sub decoupling
- Chroma local persistence