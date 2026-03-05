"""Environment utilities for managing API keys and environment variables."""

import pathlib
import dotenv
from typing import List
from pydantic import BaseModel

def load_keys(root_dir: pathlib.Path, prefix: str, n_total_orgs: int = 10) -> List[str]:
    """Load API keys from environment variables with a given prefix.
    
    Args:
        root_dir: Root directory containing the .env file
        prefix: Environment variable prefix (e.g., "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        n_total_orgs: Maximum number of organizations to support
        
    Returns:
        List of API keys loaded from environment variables
        
    Raises:
        KeyError: If the primary key (without suffix) is not found
    """
    dotenv.load_dotenv(root_dir / ".env")
    env_vars = dotenv.dotenv_values()
    
    # The first key uses the prefix directly
    primary_key = prefix
    if primary_key not in env_vars:
        raise KeyError(f"Required environment variable '{primary_key}' not found")
    
    keys = [env_vars[primary_key]]
    
    # Additional keys are suffixed with "_i" (for organization i)
    for i in range(1, n_total_orgs):
        key_name = f"{prefix}_{i}"
        if key_name in env_vars:
            keys.append(env_vars[key_name])
    
    return keys
    
class OpenAIKey(BaseModel):
    value: str
    allow_ft: bool = True
    allow_eval: bool = True
    
def load_oai_keys(root_dir: pathlib.Path, n_total_orgs: int = 10) -> List[OpenAIKey]:
    """Load OpenAI API keys from environment variables with a given prefix.
    
    Args:
        root_dir: Root directory containing the .env file
        prefix: Environment variable prefix (e.g., "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        n_total_orgs: Maximum number of organizations to support
    """
    keys_str = load_keys(root_dir, "OPENAI_API_KEY", n_total_orgs)
    eval_only_keys_str = load_keys(root_dir, "OPENAI_API_KEY_EVAL_ONLY", n_total_orgs)
    keys = []
    for key_value in keys_str:
        keys.append(OpenAIKey(value=key_value, allow_ft=True, allow_eval=True))
    for key_value in eval_only_keys_str:
        keys.append(OpenAIKey(value=key_value, allow_ft=False, allow_eval=True))
    return keys
    
class OpenAIKeyRing:
    """Manages OpenAI API keys with support for multiple organizations and key capabilities."""
    
    def __init__(self, keys: List[OpenAIKey]):
        """Initialize the OpenAIKeyRing.
        
        Args:
            keys: List of OpenAIKey instances to manage
        """
        if not keys:
            raise ValueError("OpenAIKeyRing requires at least one key")
        self._keys = keys.copy()
        self._current_key_index = 0
    
    @property
    def keys(self) -> List[OpenAIKey]:
        """Get all available API keys."""
        return self._keys.copy()
    
    @property
    def current_key(self) -> OpenAIKey:
        """Get the currently selected API key."""
        return self._keys[self._current_key_index]
    
    @property
    def current_key_index(self) -> int:
        """Get the index of the currently selected API key."""
        return self._current_key_index
    
    @property
    def num_keys(self) -> int:
        """Get the total number of available API keys."""
        return len(self._keys)
    
    def set_key_index(self, key_index: int) -> None:
        if not 0 <= key_index < len(self._keys):
            raise IndexError(f"Key index {key_index} out of range. Available keys: 0-{len(self._keys)-1}")
        self._current_key_index = key_index