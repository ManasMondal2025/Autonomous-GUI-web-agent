"""VLM (Vision-Language Model) base interface and factory.

The factory pattern lets us swap backends (Qwen2.5-VL, GPT-4V, etc.) without
modifying the agent. Concrete implementations live in `qwen_vlm.py` etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class BaseVLM(ABC):
    """Abstract VLM interface.

    All concrete VLMs (Qwen2.5-VL, GPT-4V, Claude, etc.) must implement
    `generate(image, prompt)` returning a string response. Parsing is the
    grounder's responsibility, not the VLM's.
    """

    def __init__(self, name: str = "base", **kwargs):
        self.name = name
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a text response from an image + prompt.

        Args:
            image: PIL Image (RGB).
            prompt: Natural-language prompt.
            history: Optional list of prior {role, content} turns (for chat models).

        Returns:
            The model's raw text response.
        """
        raise NotImplementedError

    def batch_generate(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> List[str]:
        """Default sequential implementation; subclasses may override for batching."""
        return [self.generate(img, p) for img, p in zip(images, prompts)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class MockVLM(BaseVLM):
    """Deterministic mock VLM for testing the pipeline end-to-end without a real model.

    `responses` is a list of pre-canned JSON strings returned in order.
    After exhaustion, the mock loops on the last response.
    """

    def __init__(self, responses: Optional[List[str]] = None, **kwargs):
        super().__init__(name="mock", **kwargs)
        self.responses = responses or []
        self._call_count = 0

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not self.responses:
            return '{"action": "fail", "reason": "MockVLM has no responses configured"}'
        response = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return response

    def reset(self) -> None:
        self._call_count = 0
