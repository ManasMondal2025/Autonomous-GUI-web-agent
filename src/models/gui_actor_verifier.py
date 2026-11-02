"""GUI-Actor verifier backend: judges whether a point matches an instruction.

Wraps microsoft/GUI-Actor-Verifier-2B (MIT, UI-TARS-2B lineage) following the
official protocol (verifier/verifier_model.py): draw a hollow red circle at
the candidate point on a centered crop, ask "True or False", and score
P(True) / (P(True) + P(False)) from a SINGLE forward pass (no generation).

Used as a gate after GUI-Actor grounding: accept the first top-k candidate
at or above threshold, else report rejection (the caller decides fallback).
Lazy-imports transformers/torch; raises a clear ImportError if missing.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

VERIFY_PROMPT = (
    "Please observe the screenshot and exame whether the hollow red "
    "circle accurately placed on the intended position in the image: "
    "'{}'. Answer True or False."
)
CIRCLE_RADIUS = 7
CIRCLE_WIDTH = 3


class GUIActorVerifier:
    """Correctness judge for (instruction, point) pairs."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "auto",
        crop_half: int = 500,
    ):
        self.model_path = model_path or "microsoft/GUI-Actor-Verifier-2B"
        self.device = device
        self.dtype = dtype
        self.crop_half = int(crop_half)
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._loaded = False
        logger.info(f"GUIActorVerifier configured (path={self.model_path}, "
                    f"device={device})")

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import (
                Qwen2VLForConditionalGeneration, AutoProcessor)
        except ImportError as e:
            raise ImportError(
                "GUIActorVerifier requires `transformers` and `torch`. "
                "Install with: pip install transformers torch accelerate"
            ) from e
        if self.dtype in (None, "auto"):
            dtype_name = ("bfloat16" if self.device.startswith("cuda")
                          else "float16" if self.device == "mps"
                          else "float32")
        else:
            dtype_name = self.dtype
        torch_dtype = {"bfloat16": torch.bfloat16,
                       "float16": torch.float16,
                       "float32": torch.float32}[dtype_name]
        logger.info(f"Loading GUI-Actor verifier from {self.model_path} "
                    f"({dtype_name} on {self.device})...")
        attn = None
        if self.device.startswith("cuda"):
            try:
                import flash_attn  # noqa: F401
                attn = "flash_attention_2"
            except ImportError:
                attn = "sdpa"
        else:
            attn = "sdpa"
        load_kwargs = {"torch_dtype": torch_dtype}
        if attn is not None:
            try:
                self._model = Qwen2VLForConditionalGeneration.\
                    from_pretrained(self.model_path,
                                    attn_implementation=attn, **load_kwargs)
            except (TypeError, ValueError):
                self._model = Qwen2VLForConditionalGeneration.\
                    from_pretrained(self.model_path, **load_kwargs)
        else:
            self._model = Qwen2VLForConditionalGeneration.\
                from_pretrained(self.model_path, **load_kwargs)
        self._model.to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        try:
            self._processor.tokenizer.pad_token = \
                self._processor.tokenizer.eos_token
        except Exception:
            pass
        self._tokenizer = self._processor.tokenizer
        self._loaded = True
        logger.info("GUI-Actor verifier loaded successfully.")

    def _mark_and_crop(self, image, x: float, y: float):
        """Hollow red circle at (x, y), then centered crop (clipped).
        Mirrors official draw_point_list (radius 7, width 3, half-size 500).
        """
        from PIL import ImageDraw

        img = image.copy()
        d = ImageDraw.Draw(img)
        d.ellipse([x - CIRCLE_RADIUS, y - CIRCLE_RADIUS,
                   x + CIRCLE_RADIUS, y + CIRCLE_RADIUS],
                  outline=(255, 0, 0), width=CIRCLE_WIDTH)
        W, H = img.size
        h = self.crop_half
        return img.crop((max(0, int(x - h)), max(0, int(y - h)),
                         min(W, int(x + h)), min(H, int(y + h))))

    def verify(self, image, instruction: str, x: float, y: float) -> float:
        """Score P(True)/(P(True)+P(False)) that point (x, y) in ORIGINAL
        image pixels matches `instruction`. Single forward pass."""
        from PIL import Image as _Image

        self._load()
        import torch
        from qwen_vl_utils import process_vision_info

        if not isinstance(image, _Image.Image):
            image = _Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        marked = self._mark_and_crop(image, float(x), float(y))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": marked},
            {"type": "text",
             "text": VERIFY_PROMPT.format(instruction)}] }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        # NOTE: unlike gui_actor_vlm (vendored 4.x-era forward), the stock
        # Qwen2-VL forward REQUIRES mm_token_type_ids -- keep it.
        with torch.inference_mode():
            logits = self._model(**inputs).logits[:, -1, :]
        true_id = self._tokenizer.encode("True",
                                         add_special_tokens=False)[0]
        false_id = self._tokenizer.encode("False",
                                          add_special_tokens=False)[0]
        probs = torch.softmax(logits, dim=-1)[0]
        t, f = float(probs[true_id]), float(probs[false_id])
        return t / max(1e-9, t + f)
