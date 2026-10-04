import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from worker.worker import chunk_text, process_document


def test_chunk_text_basic():
    """chunk_text splits text into overlapping chunks."""
    text = "A" * 2000
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_text_empty():
    """chunk_text returns empty list for empty input."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_small():
    """Short text produces a single chunk."""
    chunks = chunk_text("Hello world", chunk_size=800)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


def test_chunk_text_overlap():
    """Overlapping chunks share characters at boundaries."""
    text = "ABCDEFGHIJ" * 100
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
    overlap_region = chunks[0][-100:]
    assert chunks[1][:100] == overlap_region


@pytest.mark.asyncio
@patch("worker.worker.add_documents", new_callable=AsyncMock)
@patch("worker.worker.openai_client")
async def test_process_document_success(mock_openai, mock_add_docs):
    """process_document chunks text, generates embeddings, and adds to ChromaDB."""
    mock_embed_res = MagicMock()
    mock_embed_res.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_embed_res)

    await process_document("test.txt", "Some content for testing ingestion.")

    mock_openai.embeddings.create.assert_called_once()
    mock_add_docs.assert_called_once()
    chunks, embeddings, metadatas = mock_add_docs.call_args[0]
    assert len(chunks) == 1
    assert chunks[0] == "Some content for testing ingestion."
    assert metadatas[0]["source"] == "test.txt"


@pytest.mark.asyncio
@patch("worker.worker.add_documents", new_callable=AsyncMock)
@patch("worker.worker.openai_client")
async def test_process_document_empty_content(mock_openai, mock_add_docs):
    """process_document does not call embedding or DB when content is empty."""
    await process_document("empty.txt", "")
    mock_openai.embeddings.create.assert_not_called()
    mock_add_docs.assert_not_called()
