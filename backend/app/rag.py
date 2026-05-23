from openai import AsyncOpenAI
from .config import settings
from .vectorstore import query_documents

# Initialize async OpenAI client
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def generate_stream(query: str, websocket):
    """
    RAG pipeline:
    1. Generates query embeddings asynchronously.
    2. Queries the vector store asynchronously.
    3. Streams the matching source citations to the client immediately.
    4. Streams the LLM answer token-by-token.
    """
    # Generate query embedding asynchronously
    embed_response = await openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    query_embedding = embed_response.data[0].embedding

    # Query vector store asynchronously (runs in a thread pool via asyncio.to_thread)
    docs = await query_documents(query_embedding)

    # Extract matching source documents/chunks
    citations = docs.get("documents", [[]])[0] if docs and "documents" in docs else []

    # CRITICAL: Send citation message immediately BEFORE the LLM stream starts!
    await websocket.send_json({
        "type": "citation",
        "payload": citations
    })

    # Build context from chunks
    context = "\n".join(citations)

    # Call LLM with streaming enabled
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        stream=True
    )

    # Stream tokens to client in real-time
    async for chunk in response:
        if chunk.choices:
            token = chunk.choices[0].delta.content
            if token:
                await websocket.send_json({
                    "type": "token",
                    "payload": token
                })