"""Proposal sources: where SoM candidate elements come from.

Policy (cascading, cost-aware):
  1. DOM first, always (milliseconds, exact).
  2. Fall back to a vision detector ONLY on scarcity: fewer than
     `min_usable` non-trivial DOM proposals (a fullscreen <canvas> counts
     as trivial, not usable).
  3. Both empty -> return [] so SoM fails closed, exactly as before.

Results are cached by (url, scroll) so a detector is not re-run on every
step of a multi-step task on the same screen; any navigation or scroll
invalidates the cache.

Detector backends implement `BaseDetector.detect(screenshot)`. OmniParser
(or any UI detector) plugs in here once its weights are available — see
`OmniParserDetector`. The default fallback is `NullDetector` (returns []),
which preserves today's DOM-only behavior bit-for-bit.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def default_omniparser_dir() -> Path:
    """Portable default: <repo>/pretrained/omniparser.

    Works after moving the repository (e.g. to a Linux server) with zero
    configuration. Override per-instance with model_dir=... or globally
    with the OMNIPARSER_MODEL_DIR environment variable.
    """
    return Path(__file__).resolve().parents[2] / "pretrained" / "omniparser"


def resolve_device(preference: str = "auto", component: str = "detector") -> str:
    """Clean device selection: CUDA -> CUDA; MPS -> MPS if supported;
    otherwise CPU. Never silently downgrades an explicit request: an
    unsupported explicit device raises an actionable error."""
    import torch

    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            f"[OmniParser/{component}] device='cuda' requested but "
            "torch.cuda.is_available() is False. Use device='cpu' (or 'mps' "
            "on Apple Silicon if the component supports it)."
        )
    if pref == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError(
            f"[OmniParser/{component}] device='mps' requested but this PyTorch "
            "build has no MPS backend. Use device='cpu'."
        )
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    raise ValueError(
        f"[OmniParser/{component}] unknown device {preference!r}; "
        "use 'auto', 'cuda', 'mps', or 'cpu'."
    )


class BaseProposer:
    """Produces numbered candidate elements for one observation."""

    name = "base"

    def propose(self, page, screenshot, url: str, scroll: Dict[str, int],
                ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DOMProposer(BaseProposer):
    """Playwright DOM extraction, stamped with provenance.

    Extraction itself is untouched; only `source`/`score` are added so DOM
    proposals already match the canonical downstream schema.
    """

    name = "dom"

    def __init__(self, extractor=None):
        if extractor is None:
            from .dom_extractor import DOMExtractor
            extractor = DOMExtractor()
        self.extractor = extractor

    def propose(self, page, screenshot, url: str, scroll: Dict[str, int],
                ) -> List[Dict[str, Any]]:
        elements = self.extractor.extract(page) or []
        for el in elements:
            el.setdefault("source", "dom")
            el.setdefault("score", None)
        return elements


class BaseDetector:
    """Vision fallback: screenshot -> candidate boxes (no DOM needed)."""

    name = "base-detector"

    def detect(self, screenshot) -> List[Dict[str, Any]]:
        raise NotImplementedError


class NullDetector(BaseDetector):
    """Default fallback: no detector configured -> no extra proposals."""

    name = "null"

    def detect(self, screenshot) -> List[Dict[str, Any]]:
        return []


class OmniParserDetector(BaseDetector):
    """Microsoft OmniParser-v2 backend (YOLOv8 icon detect + Florence-2 caption).

    Expected contract: `detect()` returns element dicts with
    {"bbox": {"x","y","width","height"}} in screenshot pixels plus
    {"tag","text","score"}.

    Weights resolution: explicit model_dir=... > OMNIPARSER_MODEL_DIR env var
    > <repo>/pretrained/omniparser/ (portable default). Expected layout::

        <model_dir>/weights/icon_detect/model.pt (+ model.yaml)
        <model_dir>/weights/icon_caption_florence/model.safetensors (+ config.json)

    Device handling is per-component and isolated: if captioning cannot run
    on the selected device, boxes are still returned with empty text; if
    detection itself cannot run, detect() raises a clear actionable error
    (the cascade catches it and continues DOM-only — never silent).
    """

    name = "omniparser"

    REQUIRED_DETECT = ("weights/icon_detect/model.pt", "weights/icon_detect/model.yaml")
    REQUIRED_CAPTION = ("weights/icon_caption_florence/model.safetensors",
                        "weights/icon_caption_florence/config.json")

    def __init__(self, model_dir: Optional[str] = None, device: str = "auto",
                 box_threshold: float = 0.05, with_captions: bool = True,
                 caption_device: Optional[str] = None,
                 attn_implementation: Optional[str] = None, **kwargs):
        if model_dir is None:
            model_dir = os.getenv("OMNIPARSER_MODEL_DIR",
                                  str(default_omniparser_dir()))
        self.model_dir = str(model_dir)
        self.device = device
        self.box_threshold = box_threshold
        self.with_captions = with_captions
        self.caption_device = caption_device or device
        # Attention backend for the caption model. None = transformers default
        # (flash-attn on CUDA when installed). Set "eager" explicitly on hosts
        # without flash_attn. Also readable via FLORENCE_ATTN env var.
        self.attn_implementation = attn_implementation or os.getenv(
            "FLORENCE_ATTN")
        # Processor/tokenizer code ships with the Florence-2-base hub repo
        # (mirrors official OmniParser). Override for offline use, e.g.
        # FLORENCE_PROCESSOR_ID=/local/snapshot.
        self.processor_source = os.getenv("FLORENCE_PROCESSOR_ID",
                                          "microsoft/Florence-2-base")
        self.config = kwargs
        self._detect_model = None
        self._detect_error: Optional[str] = None
        self._caption = None  # (processor, model) or None
        self._caption_error: Optional[str] = None
        self.stats: Dict[str, Any] = {"detect_calls": 0, "captioned": 0,
                                      "loads": 0, "releases": 0}

    @property
    def is_loaded(self) -> bool:
        return self._detect_model is not None

    def release(self) -> None:
        """Drop loaded weights (sequential pipelines must call this before
        loading the next model). Safe when never loaded. Counted."""
        self._detect_model = None
        self._caption = None
        self.stats["releases"] += 1
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
        logger.info("[OmniParser] released")

    # -- setup ---------------------------------------------------------
    def _require(self, *relpaths: str) -> None:
        missing = [r for r in relpaths
                   if not (Path(self.model_dir) / r).exists()]
        if missing:
            raise RuntimeError(
                f"[OmniParser] missing weight files under {self.model_dir}: "
                f"{missing}. Expected official microsoft/OmniParser-v2.0 layout "
                f"(weights/icon_detect/, weights/icon_caption_florence/)."
            )

    def _load_detect(self):
        if self._detect_model is not None or self._detect_error is not None:
            return
        try:
            self._require(*self.REQUIRED_DETECT)
            from ultralytics import YOLO
            dev = resolve_device(self.device, "icon_detect")
            yolo_device = "cuda:0" if dev == "cuda" else dev
            self._detect_model = YOLO(
                str(Path(self.model_dir) / "weights/icon_detect/model.pt"))
            self._detect_device = yolo_device
            self.stats["loads"] += 1
            logger.info("[OmniParser] icon_detect loaded (device=%s)", yolo_device)
        except Exception as e:
            self._detect_error = str(e)
            logger.error("[OmniParser] icon_detect unavailable: %s", e)

    def _load_caption(self):
        if self._caption is not None or self._caption_error is not None:
            return
        try:
            self._require(*self.REQUIRED_CAPTION)
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            dev = resolve_device(self.caption_device, "icon_caption")
            cap_dir = str(Path(self.model_dir) / "weights/icon_caption_florence")
            # Processor/tokenizer code ships with the Florence-2-base hub repo,
            # not the local weight dir (mirrors official OmniParser, which loads
            # the processor from microsoft/Florence-2-base).
            processor = AutoProcessor.from_pretrained(
                self.processor_source, trust_remote_code=True)
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True, "torch_dtype": torch.float32}
            if self.attn_implementation:
                load_kwargs["attn_implementation"] = self.attn_implementation
            model = AutoModelForCausalLM.from_pretrained(
                cap_dir, **load_kwargs)
            model.to(dev)
            model.eval()
            self._caption = (processor, model, dev)
            logger.info("[OmniParser] icon_caption loaded (device=%s)", dev)
        except Exception as e:
            self._caption_error = str(e)
            logger.warning("[OmniParser] icon_caption unavailable (%s); "
                           "returning boxes without text.", e)

    # -- inference ------------------------------------------------------
    def detect(self, screenshot) -> List[Dict[str, Any]]:
        from PIL import Image as _Image

        if not isinstance(screenshot, _Image.Image):
            raise TypeError(
                f"[OmniParser] detect() needs a PIL Image, got {type(screenshot)}")
        img = screenshot.convert("RGB")
        self.stats["detect_calls"] += 1

        self._load_detect()
        if self._detect_model is None:
            raise RuntimeError(
                f"[OmniParser] icon_detect unavailable: {self._detect_error}. "
                f"Check weights under {self.model_dir} and device={self.device!r}.")

        results = self._detect_model.predict(
            img, conf=self.box_threshold, verbose=False,
            device=self._detect_device)
        elements: List[Dict[str, Any]] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), score in zip(xyxy, conf):
                    w, h = float(x2 - x1), float(y2 - y1)
                    if w <= 1 or h <= 1:
                        continue
                    elements.append(normalize_visual_proposal({
                        "bbox": {"x": float(x1), "y": float(y1),
                                 "width": w, "height": h},
                        "tag": "icon",
                        "text": "",
                        "score": float(score),
                    }))

        if self.with_captions and elements:
            self._caption_boxes(img, elements)
        return elements

    def _caption_boxes(self, img, elements: List[Dict[str, Any]]) -> None:
        self._load_caption()
        if self._caption is None:
            return  # isolated failure already logged; boxes keep empty text
        import torch

        processor, model, dev = self._caption
        W, H = img.size
        for el in elements:
            try:
                bb = el["bbox"]
                pad = 4
                x1 = max(0, int(bb["x"]) - pad)
                y1 = max(0, int(bb["y"]) - pad)
                x2 = min(W, int(bb["x"] + bb["width"]) + pad)
                y2 = min(H, int(bb["y"] + bb["height"]) + pad)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                inputs = processor(text="<CAPTION>", images=crop,
                                   return_tensors="pt")
                inputs = {k: (v.to(dev) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs.get("pixel_values"),
                        max_new_tokens=32, num_beams=3, do_sample=False)
                text = processor.batch_decode(
                    generated, skip_special_tokens=True)[0].strip()
                el["text"] = text
                self.stats["captioned"] += 1
            except Exception as e:
                logger.debug("[OmniParser] caption failed for a box: %s", e)
                continue


class VisualProposer(BaseProposer):
    """Image-only proposer: YOLO detect + OCR enrich, never touches DOM/page.

    Used by the `som_no_dom` method for screenshots without any browser
    (static benchmarks, canvas captures). Runs exactly ONE detector pass
    and ONE OCR pass per propose() call, then assigns sequential ids in
    detection order (deterministic for a fixed detector + image).
    """

    name = "visual"

    def __init__(self, detector=None, ocr=None, ocr_pad: float = 4.0,
                 ocr_iou: float = 0.1, verbose: bool = False):
        self.detector = detector or OmniParserDetector(with_captions=False)
        self._ocr = ocr
        self.ocr_pad = ocr_pad
        self.ocr_iou = ocr_iou
        self.verbose = verbose

    def _get_ocr(self):
        if self._ocr is None:
            from .ocr import PaddleOCRBackend
            self._ocr = PaddleOCRBackend()
        return self._ocr

    def propose(self, page, screenshot, url: str, scroll: Dict[str, int],
                ) -> List[Dict[str, Any]]:
        from .ocr import enrich_proposals_with_ocr

        elements = self.detector.detect(screenshot)
        # Same ordering convention as CascadeProposer._union (sort + renumber)
        # so marker layout is comparable across methods.
        elements.sort(key=lambda e: (
            -(e.get("bbox", {}).get("y", 0)),
            e.get("bbox", {}).get("x", 0),
        ))
        for j, e in enumerate(elements):
            e["id"] = j
        regions = self._get_ocr().detect_text(screenshot)
        try:
            img_size = tuple(screenshot.size)
        except Exception:
            img_size = None
        enriched = enrich_proposals_with_ocr(
            elements, regions, pad=self.ocr_pad, min_iou=self.ocr_iou,
            verbose=self.verbose, img_size=img_size)
        for j, e in enumerate(enriched):
            e["id"] = j
        logger.info("[visual] %d proposals from %d OCR regions (detector=%s)",
                    len(enriched), len(regions),
                    getattr(self.detector, "name", "?"))
        return enriched


class HybridProposer(BaseProposer):
    """hybrid_som: live-DOM + GPA visual candidates, matched and merged.

    Mirrors hybrid_som.py (test_20/final_test) inside the agent loop:

    * DOM side: the existing DOMExtractor, converted exactly like
      hybrid_som.extract_dom_candidates (CSS px scaled to screenshot px).
    * Visual side: ocr_pipeline.run_gpa_detection, i.e. the GPA detector
      plus the gpa_postprocess.py stages (never replaced here).
    * Association: hybrid_som.generate_hybrid_som (union on match, keep
      both otherwise; no post-union NMS by design).
    * Text: DOM text where present, otherwise per-box PaddleOCR text
      (ocr_pipeline.ocr_crop on padded crops, as in final_test).

    Output matches the canonical proposal schema (id, bbox{x,y,w,h},
    text, source dom/gpa/dom+gpa, score) sorted like CascadeProposer.
    Failures degrade gracefully and loudly: OCR loss leaves text empty,
    and a total visual-side failure falls back to DOM-only (never silent).
    """

    name = "hybrid"

    def __init__(self, extractor=None, ocr=None):
        self._extractor = extractor
        self._ocr = ocr  # ("paddleocr", backend) handle or None (lazy)

    def _hs(self):
        """Import hybrid_som.py from the repo root (absolute path, so this
        works regardless of the caller's working directory)."""
        import sys as _sys
        from pathlib import Path as _Path
        repo_root = _Path(__file__).resolve().parents[2]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        import hybrid_som as _hs_mod
        return _hs_mod

    def propose(self, page, screenshot, url: str, scroll: Dict[str, int],
                ) -> List[Dict[str, Any]]:
        from PIL import Image as _Image

        if not isinstance(screenshot, _Image.Image):
            raise TypeError(
                f"[hybrid] propose() needs a PIL Image, got {type(screenshot)}")
        _hs = self._hs()
        W, H = screenshot.size
        try:
            vp = page.viewport_size or {}
            vw = int(vp.get("width", W))
            vh = int(vp.get("height", H))
        except Exception:
            vw, vh = W, H
        try:
            dpr = float(page.evaluate("window.devicePixelRatio || 1"))
        except Exception:
            dpr = 1.0
        metrics = {"screenshot_width": W, "screenshot_height": H,
                   "viewport_width": vw, "viewport_height": vh,
                   "device_pixel_ratio": dpr}

        try:
            dom = _hs.extract_dom_candidates(page, metrics)
        except Exception as e:
            logger.warning("[hybrid] DOM extraction failed (%s); "
                           "visual-only.", e)
            dom = []
        try:
            gpa = _hs.run_gpa_candidates(screenshot)
        except Exception as e:
            logger.warning("[hybrid] GPA pipeline failed (%s); DOM-only.",
                           e)
            gpa = []
        if not dom and not gpa:
            return []
        final, _debug = _hs.generate_hybrid_som(dom, gpa)
        self._enrich_text(screenshot, final)
        elements = [self._canonical(e) for e in final]
        elements.sort(key=lambda e: (
            -(e.get("bbox", {}).get("y", 0)),
            e.get("bbox", {}).get("x", 0),
        ))
        for idx, e in enumerate(elements):
            e["id"] = idx
        logger.info("[hybrid] %d proposals (%d dom, %d gpa).",
                    len(elements), len(dom), len(gpa))
        return elements

    def _enrich_text(self, screenshot, final: List[Dict[str, Any]]) -> None:
        """Fill text for boxes without DOM text via per-box PaddleOCR
        (same ocr_crop call as final_test). Isolated: OCR trouble never
        breaks proposals."""
        needy = [e for e in final
                 if e.get("source") != "dom" and not e.get("text")]
        if not needy:
            return
        try:
            import ocr_pipeline as _oc
            if self._ocr is None:
                self._ocr = _oc.initialize_ocr("paddleocr", cpu=True)
            ocr = self._ocr
            pad_ratio = float(_oc.PADDING_RATIO)
        except Exception as e:
            logger.warning("[hybrid] OCR unavailable (%s); leaving %d "
                           "boxes textless.", e, len(needy))
            return
        W, H = screenshot.size
        for e in needy:
            try:
                x1, y1, x2, y2 = e["bbox"]
                w, h = x2 - x1, y2 - y1
                px, py = int(w * pad_ratio), int(h * pad_ratio)
                crop = screenshot.crop((
                    max(0, x1 - px), max(0, y1 - py),
                    min(W, x2 + px), min(H, y2 + py)))
                text, conf, _words = _oc.ocr_crop(ocr, crop,
                                                  str(e.get("id", "?")))
                if str(text or "").strip():
                    e["text"] = text
                    e["ocr_confidence"] = conf
            except Exception as ex:
                logger.debug("[hybrid] OCR failed for a box: %s", ex)
                continue

    @staticmethod
    def _canonical(e: Dict[str, Any]) -> Dict[str, Any]:
        x1, y1, x2, y2 = e["bbox"]
        return {
            "id": e.get("id"),
            "tag": e.get("tag") or e.get("label") or "icon",
            "type": "", "role": e.get("role") or "",
            "text": e.get("text") or "",
            "ariaLabel": "", "placeholder": "", "title": "", "value": "",
            "name": "", "href": "",
            "bbox": {"x": x1, "y": y1,
                     "width": max(0, x2 - x1), "height": max(0, y2 - y1)},
            "visible": True,
            "enabled": None,
            "interactable": None,
            "zIndex": None,
            "source": e.get("source", "gpa"),
            "score": e.get("confidence"),
            "ocr_confidence": e.get("ocr_confidence"),
            "dom_bbox": e.get("dom_bbox"),
            "gpa_bbox": e.get("gpa_bbox"),
        }


def normalize_visual_proposal(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map a raw visual-detector proposal onto the canonical DOM-shaped schema.

    Only genuinely visual information is carried over (bbox, tag proxy, text
    caption, score). DOM-only attributes are left honestly empty/unknown —
    never fabricated. Unknown `tag` falls back to the visual proxy default.
    """
    raw = dict(raw or {})
    bb = raw.get("bbox") or {}
    return {
        "id": raw.get("id"),
        "tag": raw.get("tag") or "icon",
        "type": raw.get("type") or "",
        "role": raw.get("role") or "",
        "text": raw.get("text") or "",
        "ariaLabel": "",
        "placeholder": "",
        "title": "",
        "value": "",
        "name": "",
        "href": "",
        "bbox": {"x": bb.get("x"), "y": bb.get("y"),
                 "width": bb.get("width"), "height": bb.get("height")},
        "visible": True,  # detected on-screen => visible
        "enabled": None,  # unknowable from pixels
        "interactable": None,  # unknowable from pixels
        "zIndex": None,  # unknowable from pixels
        "source": "omniparser",
        "score": raw.get("score"),
    }


class CascadeProposer(BaseProposer):
    """DOM-first cascade with detector fallback, union, and caching."""

    name = "cascade"

    def __init__(
        self,
        primary: Optional[BaseProposer] = None,
        detector: Optional[BaseDetector] = None,
        min_usable: int = 2,
        trivial_fraction: float = 0.8,
        iou_dedup_threshold: float = 0.5,
        enable_cache: bool = True,
    ):
        self.primary = primary or DOMProposer()
        self.detector = detector or NullDetector()
        self.min_usable = min_usable
        self.trivial_fraction = trivial_fraction
        self.iou_dedup_threshold = iou_dedup_threshold
        self.enable_cache = enable_cache
        self._last_key: Optional[Tuple] = None
        self._last_result: Optional[List[Dict[str, Any]]] = None

    def propose(self, page, screenshot, url: str, scroll: Dict[str, int],
                ) -> List[Dict[str, Any]]:
        scroll = scroll or {}
        key = (url or "", scroll.get("x", 0), scroll.get("y", 0))
        if self.enable_cache and key == self._last_key \
                and self._last_result is not None:
            logger.debug("Proposal cache hit for %s @ %s", url, key[1:])
            return [dict(e) for e in self._last_result]

        primary_els = self.primary.propose(page, screenshot, url, scroll) or []
        usable = [e for e in primary_els
                  if not self._is_trivial(e, screenshot.size)]
        if len(usable) >= self.min_usable:
            result = primary_els
        else:
            logger.info(
                "Only %d usable DOM proposal(s); invoking %s fallback",
                len(usable), self.detector.name)
            try:
                detected = self.detector.detect(screenshot) or []
            except Exception as e:
                logger.warning("Detector %s failed (%s); continuing DOM-only",
                               self.detector.name, e)
                detected = []
            result = self._union(primary_els, detected)

        self._last_key = key
        self._last_result = result
        return [dict(e) for e in result]

    def _is_trivial(self, element: Dict[str, Any],
                    img_size: Tuple[int, int]) -> bool:
        """A box covering most of the viewport (e.g. fullscreen <canvas>)
        is not a usable click proposal."""
        try:
            bb = element.get("bbox", {})
            area = max(0.0, float(bb.get("width", 0))) * max(0.0, float(bb.get("height", 0)))
            img_area = max(1.0, float(img_size[0]) * float(img_size[1]))
            return (area / img_area) >= self.trivial_fraction
        except (TypeError, ValueError):
            return True

    def _union(self, primary: List[Dict[str, Any]],
               detected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge detector boxes; drop near-duplicates of DOM boxes (DOM wins:
        exact over approximate), then sort + renumber like the extractor."""
        from utils.coordinate_utils import bbox_iou

        merged = [dict(e) for e in primary]
        for d in detected:
            bb = (d.get("bbox") or {})
            try:
                box = {"x": float(bb["x"]), "y": float(bb["y"]),
                       "width": float(bb["width"]), "height": float(bb["height"])}
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed detector box: %r", d)
                continue
            if box["width"] <= 0 or box["height"] <= 0:
                continue
            duplicate = False
            for e in merged:
                try:
                    if bbox_iou(e.get("bbox", {}), box) >= self.iou_dedup_threshold:
                        duplicate = True
                        break
                except (TypeError, ValueError):
                    continue
            if duplicate:
                continue
            entry = normalize_visual_proposal(d)
            entry["bbox"] = box
            merged.append(entry)

        merged.sort(key=lambda e: (
            -(e.get("bbox", {}).get("y", 0)),
            e.get("bbox", {}).get("x", 0),
        ))
        for idx, e in enumerate(merged):
            e["id"] = idx
        return merged

    def invalidate(self) -> None:
        self._last_key = None
        self._last_result = None
