import hashlib
import pytest
from unittest.mock import MagicMock, patch
from app.vectorstore import add_documents, query_documents


@pytest.mark.asyncio
@patch("app.vectorstore.collection")
async def test_add_documents_uses_sha256_ids(mock_collection):
    """add_documents generates deterministic SHA-256 chunk IDs and upserts."""
    chunks = ["Alpha", "Beta"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    metas = [{"source": "a.txt"}, {"source": "b.txt"}]

    await add_documents(chunks, embeddings, metas)

    mock_collection.upsert.assert_called_once()
    kwargs = mock_collection.upsert.call_args[1]

    expected_id_0 = hashlib.sha256("Alpha".encode("utf-8")).hexdigest()
    expected_id_1 = hashlib.sha256("Beta".encode("utf-8")).hexdigest()

    assert kwargs["ids"] == [expected_id_0, expected_id_1]
    assert kwargs["documents"] == chunks
    assert kwargs["embeddings"] == embeddings
    assert kwargs["metadatas"] == metas


@pytest.mark.asyncio
@patch("app.vectorstore.collection")
async def test_query_documents(mock_collection):
    """query_documents queries ChromaDB collection asynchronously."""
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "documents": [["Test doc"]],
        "metadatas": [[{"source": "test.txt"}]],
        "distances": [[0.1]],
    }

    result = await query_documents([0.1, 0.2, 0.3], n_results=3)

    mock_collection.query.assert_called_once()
    assert result["documents"] == [["Test doc"]]
