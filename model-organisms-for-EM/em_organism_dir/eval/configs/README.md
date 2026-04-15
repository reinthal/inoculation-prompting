Each deployment should live in its own JSON file in this directory.

The deployment key is the filename stem. For example:

- `qwen2-5-32b-base.json` becomes deployment key `qwen2-5-32b-base`

Each file should contain a single `ModelDeploymentConfig` object:

```json
{
  "url": "https://.../v1",
  "model_name": "unsloth/Qwen2.5-32B-Instruct",
  "description": "Base model",
  "max_tokens": 600,
  "temperature": 1.0,
  "top_p": 1.0,
  "enable_thinking": null
}
```

`eval_modal_deployments.py` can load either:

- a directory or single JSON path via `--config-path`
- a single JSON file via `--config-file`
