"""Canonical model choices for hermes-lite."""

ANTHROPIC_MODELS: list[tuple[str, str]] = [
    ("anthropic/claude-sonnet-4-5-20250929", "recommended"),
    ("anthropic/claude-haiku-4-5", "fast / cheap"),
]

LOCAL_MODELS: list[tuple[str, str]] = [
    ("local/qwen3.5-9b", "optional local MLX-VLM server"),
    ("local/bonsai-8b", "local Bonsai 8B, 12k context"),
    ("digitsflow/bonsai-8b", "Ollama Bonsai 8B registry model, 12k context"),
]


def anthropic_model_ids() -> list[str]:
    return [model for model, _ in ANTHROPIC_MODELS]


def local_model_ids() -> list[str]:
    return [model for model, _ in LOCAL_MODELS]


def model_ids() -> list[str]:
    return anthropic_model_ids() + local_model_ids()


def menu_labels() -> list[str]:
    labels: list[str] = []
    for model, desc in ANTHROPIC_MODELS + LOCAL_MODELS:
        labels.append(f"{model} ({desc})" if desc else model)
    return labels
