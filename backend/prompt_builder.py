"""Prompt construction for the RAG pipeline."""

import logging
from typing import Optional

from backend.chunker import DocumentChunk

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a helpful research assistant. Answer the user's question based ONLY on "
    "the context provided below. If the context does not contain enough information "
    "to answer the question, say so clearly.\n\n"
    "When you use information from the context, cite the source using the format "
    "[Document: filename, Page: N] at the end of the relevant sentence."
)


class PromptBuilder:
    """Builds a structured prompt that grounds the LLM in retrieved documents.

    The prompt includes a system instruction, the retrieved context with
    source citations, and the user's question. This structure prevents
    hallucination by forcing the LLM to answer only from the provided
    context.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(
            question="What is backpropagation?",
            results=[(chunk, 0.95), (chunk, 0.87)],
        )
    """

    def build(
        self,
        question: str,
        results: list[tuple[DocumentChunk, float]],
        system_instruction: Optional[str] = None,
        conversation_history: Optional[str] = None,
    ) -> str:
        """Build a grounded prompt from a question and retrieved chunks.

        Args:
            question: The user's question.
            results: List of (DocumentChunk, similarity_score) tuples
                     from the retriever.
            system_instruction: Optional override of the default system
                                instruction.
            conversation_history: Optional formatted conversation history
                                  from ChatMemory.

        Returns:
            A formatted prompt string ready for the LLM.
        """
        instruction = system_instruction or SYSTEM_INSTRUCTION
        context = self._format_context(results)

        if not results:
            context = "No relevant documents were found."

        history_block = ""
        if conversation_history:
            history_block = f"CONVERSATION HISTORY:\n{conversation_history}\n\n---\n\n"

        prompt = (
            f"{instruction}\n\n"
            f"---\n"
            f"CONTEXT:\n{context}\n"
            f"---\n\n"
            f"{history_block}"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )
        return prompt

    # ── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _format_context(results: list[tuple[DocumentChunk, float]]) -> str:
        """Format retrieved chunks into a numbered context block."""
        lines: list[str] = []
        for i, (chunk, score) in enumerate(results, start=1):
            source = f"[{i}] Source: {chunk.document_name}, Page {chunk.page_number}"
            lines.append(source)
            lines.append(chunk.text)
            lines.append("")
        return "\n".join(lines).strip()
