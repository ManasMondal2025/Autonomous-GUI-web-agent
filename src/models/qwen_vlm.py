"""Qwen2.5-VL backend.

This module imports transformers/torch lazily so the project remains usable
on machines without GPU or large model dependencies installed. The mock VLM
is the default for testing.
"""

import logging
from typing import Optional, Dict, Any, List
from PIL import Image

from .base_vlm import BaseVLM

logger = logging.getLogger(__name__)


class QwenVL(BaseVLM):
    """Qwen2.5-VL inference wrapper.

    Lazy-imports transformers; raises a clear ImportError if not installed.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "auto",
        max_new_tokens: int = 512,
        max_image_side: int = 1280,
        **kwargs,
    ):
        super().__init__(name="qwen2.5-vl", **kwargs)
        self.model_path = model_path or "Qwen/Qwen2.5-VL-7B-Instruct"
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        # Sizes of the most recent generate() call: model predictions in
        # pixel space refer to `last_input_size`; multiply by
        # original/input per axis to map back to the caller's image.
        self.last_original_size: Optional[tuple] = None
        self.last_input_size: Optional[tuple] = None

        self._model = None
        self._processor = None
        self._loaded = False
        self.load_count = 0

        logger.info(f"QwenVL configured (path={self.model_path}, device={device})")

    def _load(self) -> None:
        """Lazy-load the model on first call."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "QwenVL requires `transformers` and `torch`. Install with: "
                "pip install transformers torch accelerate"
            ) from e

        # Auto-pick a working dtype per device (MPS lacks bfloat16 support).
        dtype_name = self.dtype
        if dtype_name in (None, "auto"):
            dtype_name = (
                "bfloat16" if self.device.startswith("cuda")
                else "float16" if self.device == "mps"
                else "float32"
            )
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_name, torch.float16)

        logger.info(f"Loading Qwen2.5-VL from {self.model_path}...")
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.device.startswith("cuda"):
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            # cpu / mps: load without accelerate device_map, then move.
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map=None,
                trust_remote_code=True,
            )
            if self.device == "mps":
                self._model.to("mps")
        self._model.eval()
        self._loaded = True
        self.load_count += 1
        try:
            param_device = next(self._model.parameters()).device
        except StopIteration:
            param_device = self.device
        logger.info(f"Qwen2.5-VL loaded successfully on {param_device} ({dtype_name})")

    def unload(self) -> None:
        """Release model weights (for sequential memory-safe pipelines).

        Next generate() transparently reloads via _load(). Safe to call
        when never loaded.
        """
        self._model = None
        self._processor = None
        self._loaded = False
        import gc
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Qwen2.5-VL unloaded")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self._load()
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")
        self.last_original_size = image.size
        fed_image = image
        if max(image.size) > self.max_image_side:
            scale = self.max_image_side / max(image.size)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            fed_image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Downscaled input {image.size} -> {new_size} for inference")
        self.last_input_size = fed_image.size

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": fed_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=[fed_image],
            padding=True,
            return_tensors="pt",
        )
        try:
            target_device = self._model.device
        except Exception:
            target_device = self.device
        inputs = {
            k: (v.to(target_device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text
