from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager
from redis.asyncio import Redis
import json
import logging
import time
from .config import settings
from .rag import generate_stream

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# -- Async Redis client --
redis_client: Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown."""
    global redis_client
    redis_client = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True,
    )
    logger.info("Redis client connected (%s:%s)", settings.REDIS_HOST, settings.REDIS_PORT)
    yield
    await redis_client.close()
    logger.info("Redis client closed.")


app = FastAPI(
    title="Streaming RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Health endpoint --
@app.get("/health")
async def health():
    """Health check - verifies Redis connectivity."""
    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {"status": "ok", "redis": redis_ok}


# -- Document ingestion --
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Accept a document upload and publish it to the Redis ingestion queue.
    The actual processing (chunking, embedding, indexing) is handled
    asynchronously by a separate background worker process.
    """
    content = await file.read()
    message = json.dumps({
        "filename": file.filename,
        "content": content.decode("utf-8", errors="ignore"),
        "timestamp": time.time(),
    })
    # Publish asynchronously - does NOT block the event loop
    await redis_client.publish("ingestion_queue", message)
    logger.info("Published ingestion task for '%s' to Redis.", file.filename)
    return {"message": "Document ingestion initiated", "filename": file.filename}


# -- WebSocket query endpoint --
@app.websocket("/query")
async def query(websocket: WebSocket):
    """
    Real-time streaming RAG endpoint.
    Protocol:
      Client -> Server : plain-text query string
      Server -> Client : JSON messages:
        {"type": "citation", "payload": {...}}
        {"type": "token",    "payload": "..."}
        {"type": "done",     "payload": {...}}
        {"type": "error",    "payload": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket client connected.")
    try:
        while True:
            query_text = await websocket.receive_text()
            logger.info("Received query: %s", query_text[:80])
            try:
                await generate_stream(query_text, websocket)
            except WebSocketDisconnect:
                raise  # re-raise so outer handler logs it
            except Exception as e:
                logger.error("Error during RAG stream: %s", e, exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "payload": str(e),
                    })
                except Exception:
                    pass  # client already gone
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error("Unexpected WebSocket error: %s", e, exc_info=True)