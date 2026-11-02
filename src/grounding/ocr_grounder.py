"""OCR-enriched visual cascade grounder (experiment only).

Pipeline: screenshot -> OmniParser detect -> ONE PaddleOCR pass ->
spatial association -> existing SoMGrounder. Florence stays disabled here
(with_captions=False) so the OCR effect is measured cleanly.
"""

import logging
from typing import Dict, Any, Optional, List
from PIL import Image

from .base_grounder import BaseGrounder
from .som_grounder import SoMGrounder

logger = logging.getLogger(__name__)


class OCRSoMGrounder(BaseGrounder):
    """SoM grounding over OCR-enriched OmniParser proposals (no DOM needed)."""

    def __init__(
        self,
        vlm,
        include_dom_in_prompt: bool = True,
        detector=None,
        ocr=None,
        ocr_pad: float = 4.0,
        ocr_iou: float = 0.1,
        verbose: bool = False,
        recover_text_geometry: bool = False,
        min_edge: float = 0.02,
        expand: float = 3.0,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ):
        super().__init__(name="som-ocr")
        self.vlm = vlm
        self.include_dom_in_prompt = include_dom_in_prompt
        self.ocr_pad = ocr_pad
        self.ocr_iou = ocr_iou
        self.verbose = verbose
        self.recover_text_geometry = recover_text_geometry
        self.min_edge = min_edge
        self.expand = expand
        if detector is None:
            from browser.proposers import OmniParserDetector
            detector = OmniParserDetector(with_captions=False)
        self.detector = detector
        self._ocr = ocr  # lazy: built on first ground() so import is optional
        self._som = SoMGrounder(
            vlm=vlm, include_dom_in_prompt=include_dom_in_prompt,
            viewport_width=viewport_width, viewport_height=viewport_height)

    def _get_ocr(self):
        if self._ocr is None:
            from browser.ocr import PaddleOCRBackend
            self._ocr = PaddleOCRBackend()
        return self._ocr

    def ground(
        self,
        screenshot: Image.Image,
        instruction: str,
        dom_elements: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        from browser.ocr import enrich_proposals_with_ocr

        elements = self.detector.detect(screenshot)
        for j, e in enumerate(elements):
            e["id"] = j
        ocr_backend = self._get_ocr()
        regions = ocr_backend.detect_text(screenshot)
        elements = enrich_proposals_with_ocr(
            elements, regions, pad=self.ocr_pad, min_iou=self.ocr_iou,
            verbose=self.verbose, img_size=screenshot.size,
            image=screenshot if self.recover_text_geometry else None,
            ocr_backend=ocr_backend if self.recover_text_geometry else None,
            recover_text_geometry=self.recover_text_geometry,
            min_edge=self.min_edge, expand=self.expand)
        pred = self._som.ground(screenshot, instruction, elements,
                                history=history)
        pred.setdefault("method", self.name)
        pred["ocr_regions"] = len(regions)
        pred["n_proposals"] = len(elements)
        return pred

    def to_action(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        return self._som.to_action(prediction)

    @property
    def last_mapping(self):
        return self._som.last_mapping

    @property
    def last_annotated_screenshot(self):
        return self._som.last_annotated_screenshot
