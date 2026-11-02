"""VLM factory: returns a VLM instance by name.

Adding a new backend = adding a new branch here, plus an import. No agent
code needs to change.
"""

import logging
from typing import Optional, Dict, Any

from .base_vlm import BaseVLM, MockVLM

logger = logging.getLogger(__name__)


def get_vlm(
    name: str = "mock",
    model_path: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> BaseVLM:
    """Factory for VLM instances.

    Supported names: "mock", "qwen2.5-vl", "gui-actor" (require
    transformers + torch) and "ollama" (local Ollama vision model).
    """
    name = name.lower()

    if name == "mock":
        return MockVLM(**kwargs)

    if name in ("qwen2.5-vl-3b", "qwen2.5vl:3b", "qwen-3b", "qwen3b"):
        backend = kwargs.pop("backend", None)
        if backend == "transformers" or (model_path and "huggingface" in str(model_path).lower()):
            from .qwen_vlm import QwenVL
            return QwenVL(
                model_path=model_path or "Qwen/Qwen2.5-VL-3B-Instruct",
                device=device,
                **kwargs,
            )
        # Default to local Ollama qwen2.5vl:3b (instant, verified, hardware-accelerated on Mac)
        from .ollama_vlm import OllamaVLM
        return OllamaVLM(
            model=model_path or "qwen2.5vl:3b",
            **kwargs,
        )

    if name in ("qwen2.5-vl", "qwen-vl", "qwen"):
        if model_path and ("3b" in str(model_path).lower() or model_path == "qwen2.5vl:3b"):
            from .ollama_vlm import OllamaVLM
            return OllamaVLM(model="qwen2.5vl:3b", **kwargs)
        from .qwen_vlm import QwenVL
        return QwenVL(
            model_path=model_path,
            device=device,
            **kwargs,
        )

    if name in ("gui-actor", "guiactor", "gui_actor"):
        from .gui_actor_vlm import GUIActorVLM
        return GUIActorVLM(
            model_path=model_path,
            device=device,
            **kwargs,
        )

    if name in ("ollama", "ollama-vlm"):
        from .ollama_vlm import OllamaVLM
        return OllamaVLM(
            model=model_path or kwargs.pop("model", None),
            **kwargs,
        )

    raise ValueError(
        f"Unknown VLM name: {name}. Supported: 'mock', 'qwen2.5-vl', "
        f"'qwen2.5-vl-3b', 'gui-actor', 'ollama'. For other backends, add a new branch in "
        f"model_factory.py."
    )
