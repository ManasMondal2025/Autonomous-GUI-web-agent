"""Single high-level --method selector for grounding/inference experiments.

One method name completely determines grounding mode, proposal source, and
DOM-prompt inclusion, so contradictory low-level combinations (e.g. SoM
prompting with coordinate grounding) cannot be expressed on the CLI:

    coordinate:      grounding=coordinate, proposer=None,    include_dom=False
    som:             grounding=som,        proposer=dom,     include_dom=False
    som_dom:         grounding=som,        proposer=dom,     include_dom=True
    cascade_som:     grounding=som,        proposer=cascade, include_dom=False
    cascade_som_dom: grounding=som,        proposer=cascade, include_dom=True
    som_no_dom:      grounding=som,        proposer=visual,  include_dom=True
                     (image-only: YOLO detect + OCR enrich, never touches DOM)
    hybrid_som:      grounding=som,        proposer=hybrid,  include_dom=True
                     (DOM + GPA matched/merged per hybrid_som.py, OCR text
                     on visual boxes)

Verification and recovery are deliberately NOT part of a method; they stay
independent (--goal-mode, --no-verify, --no-recovery).
"""

import argparse
from typing import Dict, Any, Optional

METHOD_ORDER = ("coordinate", "som", "som_dom", "cascade_som", "cascade_som_dom",
                "som_no_dom", "hybrid_som")

METHODS: Dict[str, Dict[str, Any]] = {
    "coordinate": {
        "grounding": "coordinate",
        "proposer": None,
        "include_dom_in_prompt": False,
        "description": "Direct coordinate grounding from the screenshot.",
    },
    "som": {
        "grounding": "som",
        "proposer": "dom",
        "include_dom_in_prompt": False,
        "description": "DOM-proposal SoM without DOM text in the VLM prompt.",
    },
    "som_dom": {
        "grounding": "som",
        "proposer": "dom",
        "include_dom_in_prompt": True,
        "description": "DOM-proposal SoM with DOM hints in the VLM prompt.",
    },
    "cascade_som": {
        "grounding": "som",
        "proposer": "cascade",
        "include_dom_in_prompt": False,
        "description": "DOM-first SoM with OmniParser fallback, without DOM text.",
    },
    "cascade_som_dom": {
        "grounding": "som",
        "proposer": "cascade",
        "include_dom_in_prompt": True,
        "description": "DOM-first SoM with OmniParser fallback and available DOM hints.",
    },
    "som_no_dom": {
        "grounding": "som",
        "proposer": "visual",
        "include_dom_in_prompt": True,
        "description": "Image-only SoM: YOLO detect + OCR text enrich, no DOM access.",
    },
    "hybrid_som": {
        "grounding": "som",
        "proposer": "hybrid",
        "include_dom_in_prompt": True,
        "description": "Hybrid DOM+GPA SoM: matched/merged cross-source "
                       "proposals with OCR text on visual boxes.",
    },
}

_METHOD_HELP = (
    "Inference/grounding method (single source of truth for grounding mode, "
    "proposal source, and DOM-prompt inclusion):\n"
    + "\n".join(f"  {name}\n      {spec['description']}" for name, spec in METHODS.items())
)


def resolve_method(name: str) -> Dict[str, Any]:
    """Return a fresh config dict for a method name.

    Raises ValueError for unknown names (message lists the valid ones).
    """
    if name not in METHODS:
        raise ValueError(
            f"Unknown method: {name!r}. "
            f"Valid methods: {', '.join(METHOD_ORDER)}"
        )
    spec = dict(METHODS[name])
    spec["method"] = name
    return spec


def add_method_argument(parser: argparse.ArgumentParser,
                        default: str = "som") -> argparse.ArgumentParser:
    """Attach the shared --method selector (replaces --grounding)."""
    if default not in METHODS:
        raise ValueError(f"Invalid default method: {default!r}")
    parser.add_argument(
        "--method",
        choices=list(METHOD_ORDER),
        default=default,
        help=_METHOD_HELP,
    )
    return parser


def build_proposer(spec: Dict[str, Any], dom_extractor=None, detector=None):
    """Build the proposal source for a resolved method (None for coordinate).

    Uses the already-tested CascadeProposer/DOMProposer; no second cascade.
    """
    kind = spec["proposer"]
    if kind is None:
        return None
    from browser.proposers import (DOMProposer, CascadeProposer, NullDetector,
                                   VisualProposer, HybridProposer)
    if kind == "dom":
        return DOMProposer(dom_extractor)
    if kind == "cascade":
        return CascadeProposer(
            primary=DOMProposer(dom_extractor),
            detector=detector or NullDetector(),
        )
    if kind == "visual":
        return VisualProposer(detector=detector)
    if kind == "hybrid":
        return HybridProposer(dom_extractor)
    raise ValueError(f"Unknown proposer kind: {kind!r}")


def build_grounder(spec: Dict[str, Any], vlm,
                   viewport_width: int = 1280, viewport_height: int = 720):
    """Build the grounder for a resolved method (replaces if/elif scatter)."""
    grounding = spec["grounding"]
    if grounding == "coordinate":
        from .coordinate_grounder import CoordinateGrounder
        return CoordinateGrounder(vlm=vlm, viewport_width=viewport_width,
                                  viewport_height=viewport_height)
    if grounding == "som":
        from .som_grounder import SoMGrounder
        return SoMGrounder(vlm=vlm, viewport_width=viewport_width,
                           viewport_height=viewport_height,
                           include_dom_in_prompt=spec["include_dom_in_prompt"])
    raise ValueError(f"Unknown grounding: {grounding!r}")


def resolve_experiment_config(exp_cfg: Optional[Dict[str, Any]]):
    """Resolve a YAML experiment entry to (grounding, include_dom_in_prompt).

    Precedence: explicit `method:` key > legacy `grounding:` /
    `include_dom_in_prompt:` keys > built-in defaults. A `method:` key that
    contradicts legacy keys in the same entry is an error (fail loudly
    instead of guessing).
    """
    exp_cfg = dict(exp_cfg or {})
    if exp_cfg.get("method") is not None:
        spec = resolve_method(exp_cfg["method"])
        for key in ("grounding", "include_dom_in_prompt"):
            if key in exp_cfg and exp_cfg[key] != spec[key]:
                raise ValueError(
                    f"Experiment config conflict: method={exp_cfg['method']!r} "
                    f"implies {key}={spec[key]!r} but entry sets {key}={exp_cfg[key]!r}"
                )
        return spec["grounding"], spec["include_dom_in_prompt"]
    return (exp_cfg.get("grounding", "coordinate"),
            exp_cfg.get("include_dom_in_prompt", True))
