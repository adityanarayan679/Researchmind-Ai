"""LLM client wrapping Google's Gemini API for text generation."""

import logging
from typing import Generator

from google import genai
from google.genai import types as genai_types

from config.settings import Settings

settings = Settings()

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM fails to generate a response."""


class LLMClient:
    """A client for generating text using Google's Gemini API.

    Encapsulates API configuration, prompt formatting, and error handling
    so callers (the RAG pipeline, the chat interface) never deal with the
    raw API.

    Usage:
        client = LLMClient()
        for chunk in client.generate_stream("What is RAG?"):
            print(chunk)
    """

    def __init__(self) -> None:
        self._configure_client()

    # ── Public API ──────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Generate a complete (non-streaming) response.

        Args:
            prompt: The text prompt to send to the model.

        Returns:
            The generated response text.

        Raises:
            LLMError: If the API call fails for any reason.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
            return response.text or ""
        except Exception as exc:
            logger.exception("Non-streaming generation failed")
            raise LLMError(f"Failed to generate response: {exc}") from exc

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate a streaming response, yielding text chunks as they arrive.

        Args:
            prompt: The text prompt to send to the model.

        Yields:
            Text chunks as they arrive from the API.

        Raises:
            LLMError: If the API call fails for any reason.
        """
        try:
            stream = self._client.models.generate_content_stream(
                model=self._model_name,
                contents=prompt,
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.exception("Streaming generation failed")
            raise LLMError(f"Failed to generate streaming response: {exc}") from exc

    # ── Private helpers ─────────────────────────────────────────────────

    def _configure_client(self) -> None:
        """Configure the underlying Gemini client.

        Validates that an API key is present.
        """
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. "
                "Create a .env file from .env.example and add your key."
            )

        self._client = genai.Client(api_key=api_key)
        self._model_name = settings.LLM_MODEL_NAME
