"""Long-Term Task Experience Memory: append-only JSONL store + retrieval.

Experiences are SUGGESTIONS for future planners, never current truth:
retrieval returns them labeled as past experience, and only STV-verified
execution may commit anything to working TaskMemory. The file is
append-only and tolerant of malformed lines (skipped with a warning).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ("task_id", "goal", "final_status")


class ExperienceMemory:
    """JSONL experience store with keyword/entity-overlap retrieval.

    No vector DB: scoring reuses goal_verifier.extract_entities /
    extract_keywords, per the approved simple-first approach.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else None
        self._mem: List[Dict[str, Any]] = []
        if self.path is not None and self.path.is_file():
            self._load()

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        assert self.path is not None
        loaded, bad = 0, 0
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if not isinstance(rec, dict) or not rec.get("goal"):
                    bad += 1
                    continue
                self._mem.append(rec)
                loaded += 1
        if bad:
            logger.warning(f"ExperienceMemory: skipped {bad} malformed "
                           f"line(s) in {self.path}")
        logger.info(f"ExperienceMemory: loaded {loaded} experience(s)")

    def _append_line(self, record: Dict[str, Any]) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def save_experience(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Persist one completed/failed task experience. Stores verified
        transitions + recovery info, never just the final plan."""
        rec = dict(record)
        for k in REQUIRED_KEYS:
            if k not in rec:
                raise ValueError(f"experience record missing {k!r}")
        rec.setdefault("subgoals", [])
        rec.setdefault("successful_transitions", [])
        rec.setdefault("failures", [])
        rec.setdefault("recovery_actions", [])
        rec.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._mem.append(rec)
        if self.path is not None:
            self._append_line(rec)
        return rec

    # -- retrieval ----------------------------------------------------------
    @staticmethod
    def _score(query: str, rec: Dict[str, Any]) -> float:
        """Keyword/entity overlap in [0, ~2]. Cheap, deterministic, no deps."""
        try:
            from verification.goal_verifier import (
                extract_entities, extract_keywords)
        except Exception:  # pragma: no cover - fallback if import moves
            import re
            extract_entities = lambda g: []  # noqa: E731
            extract_keywords = lambda g: [  # noqa: E731
                w.lower() for w in re.findall(r"[a-zA-Z0-9]+", g)
                if len(w) > 2]
        q_keys = set(extract_keywords(query))
        q_ents = {e.lower() for e in extract_entities(query)}
        text = f"{rec.get('goal', '')} " + " ".join(
            str(s) for s in rec.get("subgoals", []) if isinstance(s, str))
        tl = text.lower()
        kw_hit = sum(1 for k in q_keys if k in tl)
        ent_hit = sum(1 for e in q_ents if e in tl)
        kw_score = kw_hit / max(1, len(q_keys))
        ent_score = ent_hit / max(1, len(q_ents)) if q_ents else 0.0
        bonus = 0.1 if rec.get("final_status") == "success" else 0.0
        return round(kw_score + ent_score + bonus, 3)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Top-k experiences by overlap score (excludes zero scores)."""
        scored = [(self._score(query, r), r) for r in self._mem]
        scored.sort(key=lambda t: -t[0])
        out = []
        for score, rec in scored[:max(0, k)]:
            if score <= 0:
                continue
            out.append({"score": score, "experience": rec})
        return out

    def format_for_planner(self, retrieved: List[Dict[str, Any]]) -> str:
        if not retrieved:
            return "(no relevant past experience)"
        lines = []
        for item in retrieved:
            r = item["experience"]
            lines.append(
                f"- Past task (status={r.get('final_status', '?')}, "
                f"relevance={item['score']}): goal={r.get('goal', '')!r}; "
                f"subgoals={r.get('subgoals', [])}; "
                f"recoveries={r.get('recovery_actions', [])}. "
                f"TREAT AS SUGGESTION ONLY -- verify everything live.")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._mem)
