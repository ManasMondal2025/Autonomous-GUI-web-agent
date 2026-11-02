"""Ollama-hosted VLM backend (local server, OpenAI-free).

Speaks the Ollama ``/api/chat`` endpoint with a base64 PNG attached, so any
local vision model (e.g. ``qwen2.5vl:3b``) satisfies the repo's
``BaseVLM.generate(image, prompt) -> str`` contract. Follows the same lazy,
no-hard-dependency style as the other backends (httpx is already required).
"""

import base64
import io
import logging
import os
from typing import Dict, Any, Optional, List

from PIL import Image

from .base_vlm import BaseVLM

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
MODEL_ENV = "OLLAMA_VLM_MODEL"
DEFAULT_MODEL = "qwen2.5vl:3b"


class OllamaVLM(BaseVLM):
    """VLM served by a local Ollama daemon (vision-capable model)."""

    def __init__(
        self,
        model: Optional[str] = None,
        url: Optional[str] = None,
        timeout_s: float = 300.0,
        options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(name="ollama", **kwargs)
        self.model = model or os.getenv(MODEL_ENV, DEFAULT_MODEL)
        self.url = (url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.options = dict(options or {})

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        import httpx

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        messages: List[Dict[str, Any]] = []
        for turn in history or []:
            role = turn.get("role", "user")
            messages.append({"role": role, "content": str(turn.get("content", ""))})
        messages.append({"role": "user", "content": prompt, "images": [b64]})
        payload = {"model": self.model, "messages": messages, "stream": False}
        if self.options:
            payload["options"] = self.options
        try:
            resp = httpx.post(
                f"{self.url}/api/chat", json=payload, timeout=self.timeout_s
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("OllamaVLM request failed: %s", e)
            raise
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Unexpected Ollama response shape: {data!r:.200}") from e
