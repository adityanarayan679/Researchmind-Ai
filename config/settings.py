"""Application configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration object for the ResearchMind application.

    Loads all environment variables once and provides typed access
    throughout the application. Add new settings here as the project grows.
    """

    # Google Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Embedding model
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval
    TOP_K_RESULTS: int = 4

    # Vector store
    VECTOR_STORE_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "vector_store"
    )

    # LLM
    LLM_MODEL_NAME: str = "gemini-3-flash-preview"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 1024
