import openai
from .config import settings
from .vectorstore import query_documents

openai.api_key = settings.OPENAI_API_KEY

async def generate_stream(query, websocket):
    embed = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small"
    )

    query_embedding = embed["data"][0]["embedding"]
    docs = query_documents(query_embedding)

    context = "\n".join(docs["documents"][0])

    response = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using context"},
            {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
        ],
        stream=True
    )

    async for chunk in response:
        if "choices" in chunk:
            token = chunk["choices"][0]["delta"].get("content")
            if token:
                await websocket.send_json({
                    "type": "token",
                    "payload": token
                })

    await websocket.send_json({
        "type": "citation",
        "payload": docs["documents"][0]
    })