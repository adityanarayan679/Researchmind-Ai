"""Tests for the prompt builder module."""

from backend.chunker import DocumentChunk
from backend.prompt_builder import PromptBuilder


def test_build_includes_context():
    builder = PromptBuilder()
    results = [
        (DocumentChunk("c0", "Neural networks learn patterns", "paper.pdf", 3), 0.95),
        (DocumentChunk("c1", "Backpropagation adjusts weights", "paper.pdf", 5), 0.89),
    ]
    prompt = builder.build("How do neural networks learn?", results)

    assert "Neural networks learn patterns" in prompt
    assert "Backpropagation adjusts weights" in prompt
    assert "paper.pdf" in prompt
    assert "Page 3" in prompt
    assert "Page 5" in prompt
    assert "How do neural networks learn?" in prompt


def test_build_with_no_results():
    builder = PromptBuilder()
    prompt = builder.build("Anything?", [])

    assert "No relevant documents were found" in prompt
    assert "Anything?" in prompt


def test_prompt_starts_with_instruction():
    builder = PromptBuilder()
    results = [
        (DocumentChunk("c0", "Some text", "doc.pdf", 1), 0.9),
    ]
    prompt = builder.build("A question?", results)

    assert prompt.startswith("You are a helpful research assistant")
    assert "SOURCE" in prompt.upper() or "Context" in prompt


if __name__ == "__main__":
    test_build_includes_context()
    print("PASS: test_build_includes_context")
    test_build_with_no_results()
    print("PASS: test_build_with_no_results")
    test_prompt_starts_with_instruction()
    print("PASS: test_prompt_starts_with_instruction")
    print("\nAll tests passed.")
