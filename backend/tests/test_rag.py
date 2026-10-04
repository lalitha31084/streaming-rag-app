import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
from app.main import app

client = TestClient(app)


# ── Ingest endpoint tests ──────────────────────────────────────────────
@patch("app.main.redis_client")
def test_ingest_publishes_to_redis(mock_redis):
    """POST /ingest publishes the file to the Redis ingestion queue."""
    mock_redis.publish = AsyncMock(return_value=1)

    file_content = b"This is a test document for ingestion."
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", file_content, "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Document ingestion initiated"
    assert body["filename"] == "test.txt"

    mock_redis.publish.assert_called_once()
    call_args = mock_redis.publish.call_args[0]
    assert call_args[0] == "ingestion_queue"

    payload = json.loads(call_args[1])
    assert payload["filename"] == "test.txt"
    assert "This is a test document" in payload["content"]


@patch("app.main.redis_client")
def test_ingest_handles_binary_file(mock_redis):
    """POST /ingest handles binary content gracefully."""
    mock_redis.publish = AsyncMock(return_value=1)

    response = client.post(
        "/ingest",
        files={"file": ("binary.bin", b"\x00\x01\x02", "application/octet-stream")},
    )
    assert response.status_code == 200


# ── WebSocket query tests ──────────────────────────────────────────────
@patch("app.rag.query_documents", new_callable=AsyncMock)
@patch("app.rag.openai_client")
def test_websocket_streams_citations_then_tokens(mock_openai, mock_query_docs):
    """The /query WebSocket streams citations BEFORE LLM tokens."""
    mock_query_docs.return_value = {
        "documents": [["Context chunk about AI."]],
        "metadatas": [[{"source": "ai.txt", "chunk_index": 0, "page": 0}]],
        "distances": [[0.15]],
    }

    mock_embed = MagicMock()
    mock_embed.data = [MagicMock(embedding=[0.1] * 1536)]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_embed)

    async def mock_chat_stream(*args, **kwargs):
        class Chunk:
            def __init__(self, content):
                class Choice:
                    def __init__(self, c):
                        class Delta:
                            def __init__(self, c):
                                self.content = c
                        self.delta = Delta(c)
                self.choices = [Choice(content)]
        yield Chunk("Hello")
        yield Chunk(" World")

    mock_openai.chat.completions.create = AsyncMock(side_effect=mock_chat_stream)

    with client.websocket_connect("/query") as ws:
        ws.send_text("What is AI?")

        # 1. Immediate citation message
        msg1 = ws.receive_json()
        assert msg1["type"] == "citation"
        assert msg1["payload"]["source"] == "ai.txt"
        assert "snippet" in msg1["payload"]

        # 2. Token messages
        msg2 = ws.receive_json()
        assert msg2["type"] == "token"
        assert msg2["payload"] == "Hello"

        msg3 = ws.receive_json()
        assert msg3["type"] == "token"
        assert msg3["payload"] == " World"

        # 3. Done message
        msg4 = ws.receive_json()
        assert msg4["type"] == "done"
        assert "total_time" in msg4["payload"]


@patch("app.rag.query_documents", new_callable=AsyncMock)
@patch("app.rag.openai_client")
def test_websocket_handles_empty_results(mock_openai, mock_query_docs):
    """WebSocket handles queries with no matching documents."""
    mock_query_docs.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    mock_embed = MagicMock()
    mock_embed.data = [MagicMock(embedding=[0.1] * 1536)]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_embed)

    async def mock_chat(*a, **kw):
        class Chunk:
            def __init__(self, c):
                class Choice:
                    def __init__(self, c):
                        class Delta:
                            def __init__(self, c):
                                self.content = c
                        self.delta = Delta(c)
                self.choices = [Choice(c)]
        yield Chunk("No info")

    mock_openai.chat.completions.create = AsyncMock(side_effect=mock_chat)

    with client.websocket_connect("/query") as ws:
        ws.send_text("Unknown query")
        msg = ws.receive_json()
        assert msg["type"] == "token"


@patch("app.rag.query_documents", new_callable=AsyncMock)
@patch("app.rag.openai_client")
def test_websocket_propagates_midstream_error(mock_openai, mock_query_docs):
    """WebSocket handles runtime exceptions during RAG stream by sending an error message."""
    mock_query_docs.side_effect = Exception("ChromaDB vector store connection failed")

    mock_embed = MagicMock()
    mock_embed.data = [MagicMock(embedding=[0.1] * 1536)]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_embed)

    with client.websocket_connect("/query") as ws:
        ws.send_text("Failure query")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "ChromaDB" in msg["payload"]
