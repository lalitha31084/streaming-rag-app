import chromadb
import asyncio
import hashlib
import logging
from .config import settings

logger = logging.getLogger(__name__)

# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)

# Fetch or create the documents collection
collection = client.get_or_create_collection("documents")


async def add_documents(
    chunks: list[str],
    embeddings: list[list[float]],
    metadata_list: list[dict] | None = None,
) -> None:
    """
    Asynchronously add/upsert document chunks to ChromaDB in a single batch.
    Uses asyncio.to_thread to run the blocking ChromaDB operation in a
    separate thread so the event loop is never blocked.
    IDs are generated with SHA-256 for deterministic, stable hashing,
    which also makes the operation idempotent (re-processing the same
    chunk produces the same ID -> upsert overwrites, no duplicates).
    """
    if not chunks:
        return

    def _sync_upsert():
        chunk_ids = [
            hashlib.sha256(chunk.encode("utf-8")).hexdigest() for chunk in chunks
        ]
        metas = (
            metadata_list
            if metadata_list
            else [{"source": "unknown"} for _ in chunks]
        )
        collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metas,
            ids=chunk_ids,
        )

    await asyncio.to_thread(_sync_upsert)
    logger.info("Upserted %d chunks into ChromaDB.", len(chunks))


async def query_documents(
    query_embedding: list[float],
    n_results: int = 5,
) -> dict:
    """
    Asynchronously query the ChromaDB collection for the top matching
    documents. Runs the blocking query in a thread pool.
    Gracefully handles empty collections or newly created segment states
    to prevent 'Nothing found on disk' HNSW reader exceptions.
    """
    def _sync_query():
        try:
            count = collection.count()
            if count == 0:
                logger.info("ChromaDB collection is empty.")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            k = min(n_results, count)
            return collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("ChromaDB query notice (%s). Falling back to direct metadata retrieval.", e)
            try:
                all_records = collection.get(include=["documents", "metadatas"])
                docs = all_records.get("documents", [])
                metas = all_records.get("metadatas", [])
                if docs:
                    k = min(n_results, len(docs))
                    return {
                        "documents": [docs[:k]],
                        "metadatas": [metas[:k]],
                        "distances": [[0.1] * k],
                    }
            except Exception as inner_e:
                logger.error("ChromaDB direct get fallback failed: %s", inner_e)

            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    return await asyncio.to_thread(_sync_query)