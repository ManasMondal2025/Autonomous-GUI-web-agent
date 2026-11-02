# Autonomous GUI/Web Agent with Visual Grounding

End-to-end browser agent that closes a perception–reasoning–action loop over raw
pixels: **goal → screenshot → VLM grounding → Playwright action → verify →
recover → repeat**.

Research question: **does Set-of-Mark (SoM) visual grounding improve GUI
interaction accuracy over direct coordinate prediction for VLM browser agents?**

## Architecture

```
Goal → Observer (screenshot+DOM) → Grounder (coordinate | SoM) → Action schema
→ Executor (Playwright) → Verifier → Recovery → loop (max 20 steps)
```

Both grounders implement `BaseGrounder` and are interchangeable — only the
grounding strategy changes between experiments.

## Installation (remote Linux server / SSH, headless)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium --with-deps
cp .env.example .env   # add keys / paths
```

Headless Chromium is the default. For long runs: `tmux new -s agent`.

## Quick start

```bash
# unit tests
python -m pytest tests/ -v

# Phase 1 demo: browser + screenshot + DOM
python scripts/phase1_demo.py

# Phase 2 demo: SoM overlay + both grounders
python scripts/phase2_som_demo.py

# run the agent (mock VLM demo)
python scripts/run_agent.py --task "Demo task" \
    --url "file://test_pages/test_page.html" --method som --vlm mock

# real VLM
python scripts/run_agent.py --task "Search for X" --url "https://example.com" \
    --method som_dom --vlm qwen2.5-vl --model-path Qwen/Qwen2.5-VL-7B-Instruct
```

Methods: --method {coordinate,som,som_dom,cascade_som,cascade_som_dom}
(single selector for grounding + proposer + DOM prompt; verification and
recovery stay independent via --goal-mode, --no-verify, --no-recovery).
NOTE: the old --grounding som flag behaved as --method som_dom
(SoMGrounder defaults include_dom_in_prompt=True).

## ScreenSpot evaluation

```bash
python scripts/evaluate_screenspot.py --data data/screenspot/sample.jsonl \
    --data-root data/screenspot --method som --output experiments/som/
```

Metric: **click accuracy** (predicted point inside GT box), plus mean distance
error and platform/type breakdowns. The results table is populated only from
real outputs — never fabricated.

## Project structure

- `src/browser/` — BrowserManager, DOMExtractor, Observer
- `src/grounding/` — BaseGrounder, CoordinateGrounder, SoMGrounder, SoMOverlay
- `src/models/` — BaseVLM, MockVLM, Qwen2.5-VL, factory
- `src/actions/` — Pydantic Action schema + Playwright executor
- `src/agent/` — memory, controller, recovery
- `src/verification/` — visual/URL/DOM (+optional VLM) verifier
- `src/evaluation/` — metrics + ScreenSpot harness
- `scripts/` — demos, run_agent, evaluation
- `tests/` — 100+ unit + integration tests

## Status

- [x] Phase 1: browser + screenshot + DOM
- [x] Phase 2: SoM overlay
- [x] Phases 3–4: coordinate + SoM grounders
- [x] Phase 5: action schema + executor
- [x] Phases 6–7: memory, controller, verification, recovery
- [x] Phase 8: ScreenSpot harness (smoke-tested)
- [ ] WebArena integration, full ablations, Qwen weights eval
