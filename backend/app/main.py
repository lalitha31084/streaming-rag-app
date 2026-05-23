from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager
from redis.asyncio import Redis
import json
import logging
from .config import settings
from .rag import generate_stream

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize async Redis client
redis_client = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await redis_client.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    content = await file.read()
    message = {
        "filename": file.filename,
        "content": content.decode("utf-8", errors="ignore")
    }
    # Asynchronously publish message to Redis Pub/Sub channel
    await redis_client.publish("ingestion_queue", json.dumps(message))
    return {"message": "Ingestion started"}

@app.websocket("/query")
async def query(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to query WebSocket")
    try:
        while True:
            query_text = await websocket.receive_text()
            try:
                await generate_stream(query_text, websocket)
            except WebSocketDisconnect:
                # Re-raise disconnect to catch it in the outer block
                raise
            except Exception as e:
                logger.error(f"Error during query stream processing: {e}")
                # Attempt to report the error back to the client if still connected
                try:
                    await websocket.send_json({"type": "error", "payload": str(e)})
                except Exception:
                    pass
    except WebSocketDisconnect:
        logger.info("Client disconnected from query WebSocket")
    except Exception as e:
        logger.error(f"WebSocket exception: {e}")