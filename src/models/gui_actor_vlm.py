"""GUI-Actor VLM backend (coordinate-based grounding, no VLM text parsing).

Wraps microsoft/GUI-Actor (MIT): Qwen2.5-VL backbone plus an attention
action head that grounds an instruction to normalized (x, y) via pointer
tokens. generate() returns a JSON string the existing CoordinateGrounder
already parses: {"action": "click", "x": px, "y": py} in the CALLER's
original image pixels.

Lazy-imports transformers/torch; raises a clear ImportError if missing.
"""

import json
import logging
from typing import Optional, Dict, Any, List

from .base_vlm import BaseVLM
from .vendor_gui_actor.constants import (
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    chat_template,
    grounding_system_message,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# The two helpers below are copied verbatim from microsoft/GUI-Actor
# src/gui_actor/inference.py (MIT license) -- only the module docstring and
# unused demo/eval helpers are omitted. Behavior is identical.
# --------------------------------------------------------------------------

try:
    from transformers import LogitsProcessor, LogitsProcessorList
    _LOGITS_BASE = LogitsProcessor
except Exception:  # pragma: no cover - handled at load time instead
    _LOGITS_BASE = object


class ForceFollowTokensLogitsProcessor(_LOGITS_BASE):
    """
    Forces tokens B (pointer_pad_token) and C (pointer_end_token) to follow token A (pointer_start_token).
    Whenever token_a_id is generated, enqueue the forced_sequence (e.g. [B, C]).
    As long as forced tokens remain in the queue, force them in the output.
    """
    def __init__(self, token_a_id, forced_sequence=None):
        super().__init__()
        self.token_a_id = token_a_id
        self.forced_sequence = list(forced_sequence or [])
        self.force_queue = []

    def __call__(self, input_ids, scores):
        """
        Called at each decoding step to modify `scores`.
        """
        import torch

        batch_size = input_ids.shape[0]
        if batch_size > 1:
            raise NotImplementedError("Batch size must be 1 for this logits processor.")
        last_token_id = input_ids[0, -1].item()
        if last_token_id == self.token_a_id:
            self.force_queue.extend(self.forced_sequence)
        if len(self.force_queue) > 0:
            forced_token = self.force_queue.pop(0)
            new_scores = torch.full_like(scores, float('-inf'))
            new_scores[0, forced_token] = 0.0
            return new_scores
        return scores


def get_prediction_region_point(attn_scores, n_width, n_height, top_n=30,
                                activation_threshold=0.3,
                                return_all_regions=True, rect_center=False):
    """
    1. Select activated patches
    2. Divide connected patches into different regions
    3. Calculate the average activation value for each region
    4. Select the region with the highest average activation value
    5. Return the center point of that region as the final prediction point
    """
    import torch

    max_score = attn_scores[0].max().item()
    threshold = max_score * activation_threshold
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)
    topk_values = attn_scores[0][valid_indices]
    topk_indices = valid_indices

    topk_coords = []
    for idx in topk_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))

    regions = []
    visited = set()
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = [(y, x, idx, topk_values[i].item())]
        while queue:
            cy, cx, c_idx, c_val = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                n_idx = ny * n_width + nx
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        regions.append(region)

    region_scores = []
    region_centers = []
    region_points = []

    for region in regions:
        avg_score = sum(item[3] for item in region) / len(region)
        region_scores.append(avg_score)
        normalized_centers = []
        weights = []
        y_coords = set()
        x_coords = set()
        for y, x, _, score in region:
            center_y = (y + 0.5) / n_height
            center_x = (x + 0.5) / n_width
            normalized_centers.append((center_x, center_y))
            weights.append(score)
            y_coords.add(center_y)
            x_coords.add(center_x)
        region_points.append(normalized_centers)
        if not rect_center:
            total_weight = sum(weights)
            weighted_x = sum(nc[0] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            weighted_y = sum(nc[1] * w for nc, w in zip(normalized_centers, weights)) / total_weight
            avg_center_x, avg_center_y = weighted_x, weighted_y
        else:
            avg_center_x = sum(x_coords) / len(x_coords)
            avg_center_y = sum(y_coords) / len(y_coords)
        region_centers.append((avg_center_x, avg_center_y))

    sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_scores = [region_scores[i] for i in sorted_indices]
    sorted_centers = [region_centers[i] for i in sorted_indices]
    sorted_points = [region_points[i] for i in sorted_indices]
    best_point = sorted_centers[0]

    if return_all_regions:
        return best_point, sorted_centers, sorted_scores, sorted_points
    else:
        return best_point

# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------

DEFAULT_MODEL_ID = "microsoft/GUI-Actor-3B-Qwen2.5-VL"


class GUIActorVLM(BaseVLM):
    """GUI-Actor inference wrapper (coordinate output, JSON string)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(name="gui-actor", **kwargs)
        self.model_path = model_path or DEFAULT_MODEL_ID
        self.device = device
        self.dtype = dtype
        # Official eval uses 2048; trained click outputs terminate at EOS
        # almost immediately, so 256 bounds CPU/MPS cost without changing
        # successful predictions.
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._loaded = False
        logger.info(f"GUIActorVLM configured (path={self.model_path}, "
                    f"device={device})")

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoProcessor
            from .vendor_gui_actor.modeling_qwen25vl import (
                Qwen2_5_VLForConditionalGenerationWithPointer)
        except ImportError as e:
            raise ImportError(
                "GUIActorVLM requires `transformers` and `torch`. Install "
                "with: pip install transformers torch accelerate"
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
        logger.info(f"Loading GUI-Actor from {self.model_path} "
                    f"({dtype_name} on {self.device})...")
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._tokenizer = self._processor.tokenizer
        self._model = Qwen2_5_VLForConditionalGenerationWithPointer.\
            from_pretrained(self.model_path, torch_dtype=torch_dtype)
        self._remap_snapshot_weights(torch_dtype)
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info("GUI-Actor loaded successfully.")

    def _remap_snapshot_weights(self, torch_dtype) -> None:
        """Map training-era (transformers 4.x) weight keys onto the 5.x
        module layout WITHOUT touching the checkpoint files:

            visual.*  -> model.visual.*   (vision tower was top-level)
            model.*   -> model.language_model.*   (decoder moved under .model)
            multi_patch_pointer_head.* / lm_head -> unchanged (head keeps
              its name; lm_head is tied to the embeddings, hence absent).

        Raises if any unexpected keys remain or if anything besides tied
        weights is missing.
        """
        from pathlib import Path as _Path
        from safetensors.torch import load_file

        shards = sorted(_Path(self.model_path).glob("model-*.safetensors"))
        if not shards:
            raise RuntimeError(
                f"No model-*.safetensors shards under {self.model_path}")
        full = {}
        for shard in shards:
            for k, v in load_file(str(shard), device="cpu").items():
                if k.startswith("visual."):
                    full["model." + k] = v.to(torch_dtype)
                elif k.startswith("model."):
                    full["model.language_model." + k[len("model."):]
                         ] = v.to(torch_dtype)
                else:
                    full[k] = v.to(torch_dtype)
        missing, unexpected = self._model.load_state_dict(full,
                                                          strict=False)
        missing = [k for k in missing
                   if not k.startswith("lm_head.")]
        if unexpected or missing:
            raise RuntimeError(
                f"GUI-Actor weight remap incomplete: "
                f"unexpected={unexpected[:5]} missing={missing[:5]}")
        logger.info("GUI-Actor snapshot weights remapped "
                    f"({len(full)} tensors).")

    def _pointer_ids(self):
        tok = self._tokenizer
        return (tok.encode(DEFAULT_POINTER_START_TOKEN)[0],
                tok.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
                tok.encode(DEFAULT_POINTER_END_TOKEN)[0])

    def _infer(self, image, prompt: str) -> Dict[str, Any]:
        """Run one forced-pointer forward pass. Returns the full inference
        record: normalized best point + ranked region centers/scores, grid
        size, image size and output text. Shared by generate() (top-1) and
        generate_candidates() (top-k) so both see identical numerics."""
        from PIL import Image as _Image

        self._load()
        import torch
        from qwen_vl_utils import process_vision_info

        if not isinstance(image, _Image.Image):
            image = _Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        W, H = image.size

        _, pad_id, end_id = self._pointer_ids()
        logits_processor = ForceFollowTokensLogitsProcessor(
            token_a_id=pad_id, forced_sequence=[end_id])

        conversation = [
            {"role": "system",
             "content": [{"type": "text", "text": grounding_system_message}]},
            {"role": "user",
             "content": [{"type": "image", "image": image},
                         {"type": "text", "text": prompt}]},
        ]
        text = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False,
            chat_template=chat_template)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        # The official inference predates transformers 5.x, whose processor
        # emits mm_token_type_ids -- a key the vendored forward signature
        # does not accept (and never saw upstream). Drop it to restore the
        # official input contract; single-image rope computation is
        # unaffected.
        inputs.pop("mm_token_type_ids", None)

        with torch.inference_mode():
            results = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                logits_processor=LogitsProcessorList([logits_processor]),
                return_dict_in_generate=True, output_hidden_states=True)

        input_ids = inputs["input_ids"][0]
        generated_ids = results.sequences[0][len(input_ids):]
        output_text = self._tokenizer.decode(
            generated_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        pointer_pad_mask = (generated_ids[:-1] == pad_id)
        if int(pointer_pad_mask.sum()) == 0:
            return {"ok": False, "reason": "GUI-Actor emitted no pointer "
                                           "tokens",
                    "output_text": output_text, "W": W, "H": H}
        hidden = [s[-1][0] for s in results.hidden_states[1:]]
        decoder_states = torch.cat(hidden, dim=0)[pointer_pad_mask]
        image_mask = (inputs["input_ids"][0]
                      == self._tokenizer.encode("<|image_pad|>")[0])
        image_embeds = results.hidden_states[0][0][0][image_mask]
        attn_scores, _ = self._model.multi_patch_pointer_head(
            image_embeds, decoder_states)
        _, n_h, n_w = (inputs["image_grid_thw"][0]
                       // self._model.visual.spatial_merge_size).tolist()
        best, centers, scores, _ = get_prediction_region_point(
            attn_scores, n_w, n_h, return_all_regions=True,
            rect_center=False)
        return {"ok": True, "best": best, "centers": centers,
                "scores": scores, "W": W, "H": H,
                "output_text": output_text}

    @staticmethod
    def _to_pixels(cx, cy, W, H):
        return (min(max(cx, 0.0), 1.0) * W, min(max(cy, 0.0), 1.0) * H)

    def generate_candidates(self, image, prompt: str, top_k: int = 3):
        """Ranked candidate points [{x, y, score}] in ORIGINAL image pixels
        (region centers sorted by mean activation, best first). One forward
        pass; the top-1 equals generate()'s point exactly."""
        rec = self._infer(image, prompt)
        if not rec["ok"]:
            return []
        W, H = rec["W"], rec["H"]
        pairs = list(zip(rec["centers"], rec["scores"]))[:max(1, top_k)]
        out = []
        for (cx, cy), s in pairs:
            x, y = self._to_pixels(cx, cy, W, H)
            out.append({"x": round(float(x), 1), "y": round(float(y), 1),
                        "score": round(float(s), 4)})
        return out

    def generate(
        self,
        image,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Ground `prompt` to a click point. Returns a JSON string
        {"action": "click", "x": px, "y": py} in the caller's ORIGINAL
        image pixels (or {"action": "fail", ...} when no pointer emits).
        `history` is accepted for BaseVLM compatibility; single-step
        grounding is stateless so it is unused."""
        cands = self.generate_candidates(image, prompt, top_k=1)
        if not cands:
            return json.dumps({"action": "fail",
                               "reason": "GUI-Actor emitted no pointer "
                                         "tokens"})
        c = cands[0]
        return json.dumps({"action": "click", "x": c["x"], "y": c["y"]})
