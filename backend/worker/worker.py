import asyncio
import json
import logging
from redis.asyncio import Redis
from openai import AsyncOpenAI
from app.config import settings
from app.vectorstore import add_documents

# Configure worker logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("worker")

# Initialize async OpenAI client
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Chunks text by character lengths with specified overlaps.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        # Avoid infinite loops if overlap size is >= chunk size
        if start >= len(text) or chunk_size - chunk_overlap <= 0:
            break
            
    return chunks

async def process_document(filename: str, content: str):
    """
    Processes, chunks, embeds, and indexes the document.
    """
    logger.info(f"Starting processing for file: {filename}")
    
    chunks = chunk_text(content)
    if not chunks:
        logger.warning(f"File {filename} is empty, skipping.")
        return
        
    logger.info(f"Chunked '{filename}' into {len(chunks)} fragments.")
    
    try:
        # Call OpenAI Embeddings API asynchronously for all chunks
        embed_response = await openai_client.embeddings.create(
            input=chunks,
            model="text-embedding-3-small"
        )
        embeddings = [item.embedding for item in embed_response.data]
        
        # Save chunks and embeddings to the vector store
        await add_documents(chunks, embeddings)
        logger.info(f"Ingestion successful for '{filename}'. Chunks stored in ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to process and index document '{filename}': {e}")

async def main():
    logger.info("Worker process initialized. Connecting to Redis...")
    
    # Initialize async Redis client and Pub/Sub connection
    r = Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
    pubsub = r.pubsub()
    await pubsub.subscribe("ingestion_queue")
    logger.info("Subscribed to Redis channel 'ingestion_queue'. Listening for tasks...")
    
    while True:
        try:
            # Poll for messages on the subscribed channel
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                logger.info("Ingestion job received from queue.")
                try:
                    payload = json.loads(message["data"])
                    filename = payload.get("filename", "unknown")
                    content = payload.get("content", "")
                    await process_document(filename, content)
                except Exception as e:
                    logger.error(f"Error parsing message payload: {e}")
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Subscription loop encountered error: {e}")
            await asyncio.sleep(2.0)  # Wait before attempting reconnection or retry

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker process terminated by user.")