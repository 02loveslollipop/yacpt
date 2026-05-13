import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from classes.context import Context


class AsyncIterator:
    """Helper to create a proper async iterator from a list of items."""
    def __init__(self, items):
        self.items = list(items)
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def tmp_conv_dir(tmp_path):
    """Temporary conversations directory."""
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir()
    return conv_dir


@pytest.fixture
def mock_context():
    """Pre-populated Context with system + user + assistant messages."""
    ctx = Context("You are a test bot.")
    ctx.append("user", "Hello!")
    ctx.append("assistant", "Hi there!")
    return ctx


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL file and return its path."""
    import json
    filepath = tmp_path / "test-conv.jsonl"
    lines = [
        {"meta": {"provider": "googleai", "model": "gemma-4-31b-it"}},
        {"role": "system", "content": "You are a test bot."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    with open(filepath, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return filepath


@pytest.fixture
def mock_openai_client():
    """Mock AsyncOpenAI client."""
    client = AsyncMock()

    # Mock models.list
    model1 = MagicMock()
    model1.id = "gpt-5.4-nano-2026-03-17"
    model2 = MagicMock()
    model2.id = "text-embedding-ada-002"  # Should be filtered out
    model3 = MagicMock()
    model3.id = "o4-mini"
    model_list = MagicMock()
    model_list.data = [model1, model2, model3]
    client.models.list = AsyncMock(return_value=model_list)

    return client


@pytest.fixture
def mock_googleai_client():
    """Mock Google GenAI client."""
    client = MagicMock()

    # Mock models.list (sync)
    model1 = MagicMock()
    model1.name = "gemma-4-31b-it"
    model1.display_name = "Gemma 4 31B IT"
    model1.description = "A generative model"
    model1.supported_actions = ["generateContent", "countTokens"]
    model1.input_token_limit = 131072

    model2 = MagicMock()
    model2.name = "embedding-001"
    model2.display_name = "Embedding 001"
    model2.description = "An embedding model"
    model2.supported_actions = ["embedContent"]
    model2.input_token_limit = 2048

    client.models.list.return_value = iter([model1, model2])

    return client


@pytest.fixture
def mock_mistral_client():
    """Mock Mistral client."""
    client = MagicMock()

    model1 = MagicMock()
    model1.id = "mistral-small-latest"
    model1.name = "Mistral Small"
    model1.description = "A small model"
    model1.max_context_length = 32000

    result = MagicMock()
    result.data = [model1]
    client.models.list.return_value = result

    return client
