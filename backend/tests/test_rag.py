from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
import json
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.main.redis_client", new_callable=AsyncMock)
def test_ingest(mock_redis):
    # Mock the publish method
    mock_redis.publish = AsyncMock(return_value=1)
    
    file_content = b"This is a test document content."
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", file_content, "text/plain")}
    )
    
    assert response.status_code == 200
    assert response.json() == {"message": "Ingestion started"}
    
    # Check that it published to the correct queue
    mock_redis.publish.assert_called_once()
    call_args = mock_redis.publish.call_args[0]
    assert call_args[0] == "ingestion_queue"
    
    payload = json.loads(call_args[1])
    assert payload["filename"] == "test.txt"
    assert payload["content"] == "This is a test document content."

@patch("app.rag.query_documents", new_callable=AsyncMock)
@patch("app.rag.openai_client")
def test_websocket_query(mock_openai, mock_query_docs):
    # Mock the vector store query
    mock_query_docs.return_value = {
        "documents": [["This is matching context from document."]]
    }
    
    # Mock the embedding response
    mock_embed_res = MagicMock()
    mock_embed_res.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_embed_res)
    
    # Mock the chat completion response as an async generator
    async def mock_chat_stream(*args, **kwargs):
        class Chunk:
            def __init__(self, content):
                class Choice:
                    def __init__(self, content):
                        class Delta:
                            def __init__(self, content):
                                self.content = content
                        self.delta = Delta(content)
                self.choices = [Choice(content)]
        
        yield Chunk("Hello")
        yield Chunk(" World")
        
    mock_openai.chat.completions.create = AsyncMock(side_effect=mock_chat_stream)
    
    # Test websocket connection
    with client.websocket_connect("/query") as websocket:
        websocket.send_text("What is the meaning of life?")
        
        # First message should be citation since we stream citations first!
        msg1 = websocket.receive_json()
        assert msg1["type"] == "citation"
        assert msg1["payload"] == ["This is matching context from document."]
        
        # Second message should be the first token "Hello"
        msg2 = websocket.receive_json()
        assert msg2["type"] == "token"
        assert msg2["payload"] == "Hello"
        
        # Third message should be the second token " World"
        msg3 = websocket.receive_json()
        assert msg3["type"] == "token"
        assert msg3["payload"] == " World"
