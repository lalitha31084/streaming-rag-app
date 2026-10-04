import logging
import time
from openai import AsyncOpenAI
from .config import settings
from .vectorstore import query_documents

logger = logging.getLogger(__name__)

# Initialize async OpenAI client (non-blocking)
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def generate_stream(query: str, websocket) -> None:
    """
    Fully-asynchronous RAG pipeline:
      1. Generate query embedding asynchronously via OpenAI.
      2. Query ChromaDB asynchronously (via asyncio.to_thread inside vectorstore).
      3. Send source citations to the client IMMEDIATELY (before LLM tokens).
      4. Stream the LLM answer token-by-token over the WebSocket.
      5. Send a 'done' message to signal stream completion.
    All I/O is non-blocking: embedding creation and chat completion use
    the async OpenAI client; vector DB calls use asyncio.to_thread.
    """
    t0 = time.perf_counter()

    # -- 1. Generate query embedding (async, non-blocking) --
    embed_response = await openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
    )
    query_embedding = embed_response.data[0].embedding

    # -- 2. Retrieve relevant chunks (async via to_thread) --
    docs = await query_documents(query_embedding)

    # Extract text chunks and metadata
    chunks = docs.get("documents", [[]])[0] if docs else []
    metadatas = docs.get("metadatas", [[]])[0] if docs else []
    distances = docs.get("distances", [[]])[0] if docs else []

    # -- 3. Stream citations IMMEDIATELY before LLM tokens --
    for idx, chunk_text in enumerate(chunks):
        source = metadatas[idx].get("source", "unknown") if idx < len(metadatas) else "unknown"
        page = metadatas[idx].get("page", None) if idx < len(metadatas) else None
        distance = distances[idx] if idx < len(distances) else None
        citation_payload = {
            "source": source,
            "page": page,
            "snippet": chunk_text[:200],
            "relevance_score": round(1.0 - (distance or 0), 4),
        }
        await websocket.send_json({"type": "citation", "payload": citation_payload})

    # -- 4. Build context and stream LLM response --
    context = "\n---\n".join(chunks) if chunks else "No relevant documents found."

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "using ONLY the provided context. If the context does not "
                    "contain enough information, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        stream=True,
    )

    first_token = True
    async for chunk in response:
        if chunk.choices:
            token = chunk.choices[0].delta.content
            if token:
                if first_token:
                    ttft = time.perf_counter() - t0
                    logger.info("Time-to-first-token: %.3f s", ttft)
                    first_token = False
                await websocket.send_json({"type": "token", "payload": token})

    # -- 5. Signal completion --
    total = time.perf_counter() - t0
    await websocket.send_json({"type": "done", "payload": {"total_time": round(total, 3)}})
    logger.info("Query completed in %.3f s", total)