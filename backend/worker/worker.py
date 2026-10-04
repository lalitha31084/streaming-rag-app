"""
Background ingestion worker.

Runs as a separate process (distinct container in Docker Compose).
Subscribes to the Redis Pub/Sub channel 'ingestion_queue' and, for every
message received:
  1. Parses the JSON payload (filename + text content).
  2. Chunks the text using a sliding-window strategy.
  3. Generates embeddings via the async OpenAI API.
  4. Upserts chunks + embeddings into ChromaDB (idempotent via SHA-256 IDs).

All I/O is non-blocking (async Redis, async OpenAI, asyncio.to_thread
for ChromaDB).
"""

import asyncio
import json
import logging
import sys
import time

from redis.asyncio import Redis
from openai import AsyncOpenAI

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from app.config import settings
from app.vectorstore import add_documents

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")

# -- Async OpenAI client --
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


# ---- Chunking ----
def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping windows of chunk_size characters.
    Overlap ensures no information is lost at chunk boundaries.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


# ---- Document processing ----
async def process_document(filename: str, content: str) -> None:
    """
    End-to-end processing of a single document:
      chunk -> embed -> store.
    """
    t0 = time.perf_counter()
    logger.info("Processing document: %s (%d chars)", filename, len(content))

    chunks = chunk_text(content)
    if not chunks:
        logger.warning("Document '%s' produced 0 chunks - skipping.", filename)
        return

    logger.info("Chunked '%s' into %d fragments.", filename, len(chunks))

    # Generate embeddings (async, non-blocking)
    try:
        embed_response = await openai_client.embeddings.create(
            input=chunks,
            model="text-embedding-3-small",
        )
        embeddings = [item.embedding for item in embed_response.data]
    except Exception as e:
        logger.error("Embedding generation failed for '%s': %s", filename, e)
        return

    # Build per-chunk metadata
    metadata_list = [
        {"source": filename, "chunk_index": i, "page": i // 3}
        for i in range(len(chunks))
    ]

    # Store in ChromaDB (async via to_thread inside add_documents)
    try:
        await add_documents(chunks, embeddings, metadata_list)
    except Exception as e:
        logger.error("ChromaDB upsert failed for '%s': %s", filename, e)
        return

    elapsed = time.perf_counter() - t0
    logger.info(
        "Ingestion complete for '%s': %d chunks indexed in %.2f s.",
        filename, len(chunks), elapsed,
    )


# ---- Main loop ----
async def main() -> None:
    logger.info("Worker starting - connecting to Redis at %s:%s ...",
                settings.REDIS_HOST, settings.REDIS_PORT)

    redis = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True,
    )

    # Verify Redis is reachable
    while True:
        try:
            await redis.ping()
            logger.info("Redis connection established.")
            break
        except Exception as e:
            logger.warning("Redis not ready (%s), retrying in 2 s...", e)
            await asyncio.sleep(2)

    pubsub = redis.pubsub()
    await pubsub.subscribe("ingestion_queue")
    logger.info("Subscribed to 'ingestion_queue'. Waiting for tasks...")

    while True:
        try:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=1.0,
            )
            if message and message.get("type") == "message":
                logger.info("Received ingestion job from queue.")
                try:
                    payload = json.loads(message["data"])
                    filename = payload.get("filename", "unknown")
                    content = payload.get("content", "")
                    await process_document(filename, content)
                except json.JSONDecodeError as e:
                    logger.error("Malformed message payload: %s", e)
                except Exception as e:
                    logger.error("Document processing error: %s", e, exc_info=True)
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error("Subscription loop error: %s - reconnecting in 3 s", e)
            await asyncio.sleep(3)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker terminated by user.")