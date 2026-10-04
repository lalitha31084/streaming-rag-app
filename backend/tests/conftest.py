import os
import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-mock-key-for-testing")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("VECTOR_DB_PATH", "./chroma_test_db")


@pytest.fixture
def sample_document_content():
    """Sample document text for testing."""
    return (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to natural intelligence displayed by animals including humans. "
        "AI research has been defined as the field of study of intelligent agents, "
        "which refers to any system that perceives its environment and takes actions "
        "that maximize its chance of achieving its goals."
    )
