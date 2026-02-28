# System Architecture

```mermaid
flowchart LR

User --> Frontend
Frontend -->|WebSocket| FastAPI
Frontend -->|POST /ingest| FastAPI
FastAPI -->|Publish| Redis
Redis --> Worker
Worker --> ChromaDB
FastAPI --> ChromaDB
FastAPI -->|Streaming Tokens| Frontend