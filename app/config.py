import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    # GEMINI
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # CHUNKING
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # QDRANT
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

    # GROQ
    GROQ_MODEL = os.getenv("GROQ_MODEL")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    # LOGFIRE
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    LOGFIRE_SERVICE_NAME = os.getenv("LOGFIRE_SERVICE_NAME", "starter-project")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

settings = Settings()


def validate_env_vars() -> None:
    required = [
        "GEMINI_API_KEY",
        "QDRANT_API_KEY",
        "QDRANT_CLUSTER_ENDPOINT",
        "QDRANT_COLLECTION",
    ]

    missing = [
        name
        for name in required
        if not getattr(settings, name)
    ]

    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )
