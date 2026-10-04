import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or "sk-mock-key-for-eval"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    
    _vdb = os.getenv("VECTOR_DB_PATH", "/data/chroma")
    if _vdb.startswith("/data"):
        VECTOR_DB_PATH: str = _vdb
    else:
        # Use absolute path to ensure API and Worker share the exact same database directory
        VECTOR_DB_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_data"))

    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

settings = Settings()