# Set up basic test fixtures 
import pytest

@pytest.fixture
def model():
    return "gpt-4.1-nano-2025-04-14"

@pytest.fixture
def dataset():
    return [
        {
            "messages": [
                {"role": "user", "content": "Hello, world!"},
                {"role": "assistant", "content": "Hello, world!"},
            ]
        }
    ]