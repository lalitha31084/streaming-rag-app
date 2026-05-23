import chromadb
import asyncio
import hashlib
from .config import settings

# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)

# Fetch or create the documents collection
collection = client.get_or_create_collection("documents")

async def add_documents(chunks: list[str], embeddings: list[list[float]]):
    """
    Asynchronously add/upsert documents to ChromaDB.
    Runs the blocking database operation in a separate thread.
    """
    def sync_upsert():
        for i, chunk in enumerate(chunks):
            # Generate stable, reproducible ID using SHA-256
            chunk_id = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            # Use upsert to ensure worker processing is idempotent
            collection.upsert(
                documents=[chunk],
                embeddings=[embeddings[i]],
                ids=[chunk_id]
            )
            
    await asyncio.to_thread(sync_upsert)

async def query_documents(query_embedding: list[float]):
    """
    Asynchronously query ChromaDB collection for top matching documents.
    Runs the blocking database query in a separate thread.
    """
    def sync_query():
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
    return await asyncio.to_thread(sync_query)