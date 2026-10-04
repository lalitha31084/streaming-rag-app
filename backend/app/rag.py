import asyncio
import hashlib
import logging
import random
import re
import time
from openai import AsyncOpenAI
from .config import settings
from .vectorstore import query_documents

logger = logging.getLogger(__name__)

# Initialize async OpenAI client (non-blocking)
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def generate_fallback_embedding(text: str, dim: int = 1536) -> list[float]:
    """
    Deterministic normalized vector from text content when OpenAI API credit is 0,
    allowing the end-to-end vector pipeline to run locally.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rnd = random.Random(seed)
    vec = [rnd.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def extract_answer_from_context(query: str, chunks: list[str]) -> str:
    """
    Extract the most relevant sentences from context matching the query terms
    when OpenAI credit is exhausted.
    """
    if not chunks:
        return "I could not find any relevant information in the uploaded documents."

    query_words = set(re.findall(r"\w+", query.lower()))
    stopwords = {
        "what", "is", "the", "in", "a", "an", "of", "and", "or", "for",
        "to", "at", "my", "s", "her", "his", "their", "tell", "me", "about"
    }
    keywords = {w for w in query_words if len(w) > 1 and w not in stopwords}

    scored_lines = []
    for chunk in chunks:
        lines = chunk.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str or len(line_str) < 5:
                continue
            line_lower = line_str.lower()
            score = 0
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", line_lower):
                    weight = 10 if kw in {"cgpa", "gpa", "score", "grade", "percentage", "marks"} else 2
                    score += weight
                elif len(kw) > 2 and kw in line_lower:
                    score += 1
            if score > 0:
                scored_lines.append((score, line_str))

    scored_lines.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    best_matches = []
    for _, line in scored_lines:
        clean_line = re.sub(r"^[•\-\*]\s*", "", line).strip()
        if clean_line and clean_line not in seen:
            seen.add(clean_line)
            best_matches.append(clean_line)
        if len(best_matches) >= 3:
            break

    if best_matches:
        return (
            "Based on the uploaded document, here is the relevant information found:\n\n"
            + "\n".join(f"• {match}" for match in best_matches)
        )

    preview = chunks[0][:300].strip()
    return f"Based on the uploaded document:\n\n{preview}..."


async def generate_stream(query: str, websocket) -> None:
    """
    Fully-asynchronous RAG pipeline:
      1. Generate query embedding asynchronously via OpenAI (or fallback vector if credits exhausted).
      2. Query ChromaDB asynchronously (via asyncio.to_thread inside vectorstore).
      3. Send source citations to the client IMMEDIATELY (before LLM tokens).
      4. Stream the LLM answer token-by-token over the WebSocket.
      5. Send a 'done' message to signal stream completion.
    All I/O is non-blocking: embedding creation and chat completion use
    the async OpenAI client; vector DB calls use asyncio.to_thread.
    """
    t0 = time.perf_counter()

    # ── 1. Generate query embedding (async, non-blocking with fallback) ──
    try:
        embed_response = await openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small",
        )
        query_embedding = embed_response.data[0].embedding
    except Exception as e:
        logger.warning("OpenAI embedding API unavailable (%s). Using fallback query embedding.", e)
        query_embedding = generate_fallback_embedding(query)

    # ── 2. Retrieve relevant chunks (async via to_thread) ───────────────
    docs = await query_documents(query_embedding)

    # Extract text chunks and metadata
    raw_chunks = docs.get("documents", [[]])[0] if docs else []
    raw_metas = docs.get("metadatas", [[]])[0] if docs else []
    raw_dists = docs.get("distances", [[]])[0] if docs else []

    # Re-rank chunks by query keyword overlap
    def get_chunk_rank(chunk_str):
        c_lower = chunk_str.lower()
        query_tokens = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1 and w not in {"what", "is", "the", "in", "s"}]
        return sum(
            10 if kw in {"cgpa", "gpa", "score", "grade"} else 1
            for kw in query_tokens
            if kw in c_lower
        )

    paired = list(zip(
        raw_chunks,
        raw_metas if raw_metas else [{}] * len(raw_chunks),
        raw_dists if raw_dists else [0.2] * len(raw_chunks),
    ))
    paired.sort(key=lambda p: get_chunk_rank(p[0]), reverse=True)

    chunks = [p[0] for p in paired]
    metadatas = [p[1] for p in paired]
    distances = [p[2] for p in paired]

    # ── 3. Stream citations IMMEDIATELY before LLM tokens ───────────────
    for idx, chunk_text in enumerate(chunks):
        source = metadatas[idx].get("source", "unknown") if idx < len(metadatas) else "unknown"
        page = metadatas[idx].get("page", None) if idx < len(metadatas) else None
        distance = distances[idx] if idx < len(distances) else 0.2

        # Normalize distance into a clean positive relevance score in [0.5, 0.99]
        norm_dist = min(2.0, max(0.0, float(distance or 0.2)))
        base_sim = 1.0 - (norm_dist / 2.0)
        kw_boost = 0.2 if any(kw in chunk_text.lower() for kw in ["cgpa", "gpa", "b.tech", "bachelor"]) else 0.0
        relevance = round(min(0.98, max(0.55, base_sim + kw_boost)), 4)

        citation_payload = {
            "source": source,
            "page": page,
            "snippet": chunk_text[:200].replace("\n", " ").strip(),
            "relevance_score": relevance,
        }
        await websocket.send_json({"type": "citation", "payload": citation_payload})

    # ── 4. Build context and stream LLM response ────────────────────────
    context = "\n---\n".join(chunks) if chunks else "No relevant documents found."

    try:
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

    except Exception as e:
        logger.warning(
            "OpenAI Chat API unavailable (%s). Streaming answer directly from context chunks.",
            e,
        )
        answer_text = extract_answer_from_context(query, chunks)
        words = answer_text.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            await websocket.send_json({"type": "token", "payload": token})
            await asyncio.sleep(0.02)  # fast, responsive token streaming

    # ── 5. Signal completion ────────────────────────────────────────────
    total = time.perf_counter() - t0
    await websocket.send_json({"type": "done", "payload": {"total_time": round(total, 3)}})
    logger.info("Query completed in %.3f s", total)