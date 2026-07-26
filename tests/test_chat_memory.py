"""Tests for the chat memory module."""

from backend.chat_memory import ChatMemory


def test_add_and_format():
    memory = ChatMemory(max_exchanges=5)
    memory.add("user", "What is RAG?")
    memory.add("assistant", "RAG stands for Retrieval-Augmented Generation.")
    history = memory.format_for_prompt()
    assert "What is RAG?" in history
    assert "RAG stands for" in history
    assert "User:" in history
    assert "Assistant:" in history


def test_empty_history_returns_empty_string():
    memory = ChatMemory(max_exchanges=5)
    assert memory.format_for_prompt() == ""


def test_trims_oldest_exchanges():
    memory = ChatMemory(max_exchanges=2)
    for i in range(6):
        memory.add("user", f"question {i}")
        memory.add("assistant", f"answer {i}")
    assert memory.size <= 4
    history = memory.format_for_prompt()
    assert "question 0" not in history
    assert "question 2" not in history
    assert "question 4" in history
    assert "question 5" in history


def test_clear_resets_history():
    memory = ChatMemory(max_exchanges=5)
    memory.add("user", "Hello")
    memory.add("assistant", "Hi")
    memory.clear()
    assert memory.size == 0
    assert memory.format_for_prompt() == ""


def test_invalid_max_exchanges():
    try:
        ChatMemory(max_exchanges=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_get_history_returns_copy():
    memory = ChatMemory(max_exchanges=5)
    memory.add("user", "Hi")
    history = memory.get_history()
    history.clear()
    assert memory.size == 1


if __name__ == "__main__":
    test_add_and_format()
    print("PASS: test_add_and_format")
    test_empty_history_returns_empty_string()
    print("PASS: test_empty_history_returns_empty_string")
    test_trims_oldest_exchanges()
    print("PASS: test_trims_oldest_exchanges")
    test_clear_resets_history()
    print("PASS: test_clear_resets_history")
    test_invalid_max_exchanges()
    print("PASS: test_invalid_max_exchanges")
    test_get_history_returns_copy()
    print("PASS: test_get_history_returns_copy")
    print("\nAll tests passed.")
