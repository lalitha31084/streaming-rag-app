from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
from .config import settings
from .rag import generate_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    content = await file.read()
    message = {
        "filename": file.filename,
        "content": content.decode()
    }
    redis_client.publish("ingestion_queue", json.dumps(message))
    return {"message": "Ingestion started"}

@app.websocket("/query")
async def query(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            await generate_stream(query, websocket)
    except Exception as e:
        await websocket.send_json({"type": "error", "payload": str(e)})