import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    # GEMINI
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
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
    LOGFIRE_SERVICE_NAME = os.getenv("LOGFIRE_SERVICE_NAME", "rag-production-grade")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

settings = Settings()
