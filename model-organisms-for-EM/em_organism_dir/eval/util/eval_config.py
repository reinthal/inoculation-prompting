import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelDeploymentConfig(BaseModel):
    """Configuration for a single model deployment endpoint."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="Base URL of the backend (e.g. https://.../v1)")
    model_name: str = Field(..., description="Model name for API calls (HuggingFace ID or alias)")
    description: str = Field("", description="Human-readable label for reports and logs")
    max_tokens: int = Field(600, description="Maximum tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling parameter")
    enable_thinking: Optional[bool] = Field(None, description="Enable/disable thinking mode (e.g. for Qwen3)")


def load_eval_config(config_path: str) -> ModelDeploymentConfig:
    """Load a single model deployment config from a JSON file."""
    with open(config_path, "r") as f:
        data = json.load(f)
    return ModelDeploymentConfig(**data)


def load_eval_configs(
    config_path: Optional[str] = None,
    config_paths: Optional[list[str]] = None,
) -> dict[str, ModelDeploymentConfig]:
    """Load model deployment configs from a directory or explicit JSON file list.

    Deployment keys are derived from each JSON filename stem.

    Args:
        config_path: Either a directory containing `*.json` files or a single JSON file.
        config_paths: Explicit list of JSON config files.

    Returns:
        Dict mapping deployment key to ModelDeploymentConfig.
    """
    if config_path and config_paths:
        raise ValueError("Pass either config_path or config_paths, not both.")
    if not config_path and not config_paths:
        raise ValueError("Must provide config_path or config_paths.")

    if config_paths is not None:
        paths = [Path(path) for path in config_paths]
    else:
        candidate = Path(config_path)
        paths = sorted(candidate.glob("*.json")) if candidate.is_dir() else [candidate]

    if not paths:
        source = config_path if config_path is not None else config_paths
        raise ValueError(f"No JSON config files found in {source}.")

    deployments: dict[str, ModelDeploymentConfig] = {}
    for path in paths:
        if path.suffix.lower() != ".json":
            raise ValueError(f"Config path must be a JSON file: {path}")
        deployment_key = path.stem
        if deployment_key in deployments:
            raise ValueError(f"Duplicate deployment key '{deployment_key}' from {path}")
        deployments[deployment_key] = load_eval_config(str(path))

    return deployments
