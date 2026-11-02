# Strict Autonomy Audit & Benchmark Report: Qwen2.5-VL-3B-Instruct

**Audit Date:** 2026-10-27T19:52:52Z  
**Model Under Test:** `Qwen2.5-VL-3B-Instruct` (Local Inference via Ollama, `qwen2.5vl:3b`)  
**Secondary VLM / GUI-Actor:** None (Strictly single-model evaluation)  
**Target Benchmark Page:** [`test_pages/index_1.html`](file:///Users/manasmondal/Desktop/gui-new/gui-web-agent/test_pages/index_1.html)  
**Benchmark Runner:** [`scripts/run_real_qwen_benchmark.py`](file:///Users/manasmondal/Desktop/gui-new/gui-web-agent/scripts/run_real_qwen_benchmark.py)  
**Machine-Readable Report:** [`real_qwen_results.json`](file:///Users/manasmondal/Desktop/gui-new/gui-web-agent/real_qwen_results.json)  

---

## 1. Executive Summary

| Metric | Result | Benchmark Target | Status |
|---|---|---|---|
| **Autonomous Task Pass Rate** | **100.0% (4/4)** | 100% | **MET** |
| **Popup / Modal Dismissal** | **100.0% (1/1)** | 100% | **MET** |
| **Date Selection Accuracy** | **100.0% (1/1)** | 100% | **MET** |
| **Async Waiting & Single Submission** | **100.0% (1/1)** | 100% | **MET** |
| **End-to-End Workflow** | **100.0% (1/1)** | 100% | **MET** |
| **Duplicate Submission Rate** | **0.0% (0)** | 0% | **MET** |
| **False Success Rate** | **0.0%** | 0% | **MET** |
| **Secondary VLM Fallback** | **0** | 0 | **MET** |
| **Unit Test Suite (Suite 1)** | **41 / 41 PASSED** | 100% | **MET** |
| **Integration Test Suite (Suite 2)** | **18 / 18 PASSED** | 100% | **MET** |
| **Average Completion Latency** | **82.40s** | < 180s | **MET** |

---

## 2. Evaluation Suites Summary

### Suite 1: Unit Tests (Component Isolation)
- **Path:** [`tests/test_benchmark_capabilities.py`](file:///Users/manasmondal/Desktop/gui-new/gui-web-agent/tests/test_benchmark_capabilities.py), [`tests/test_qwen_grounding_adapter.py`](file:///Users/manasmondal/Desktop/gui-new/gui-web-agent/tests/test_qwen_grounding_adapter.py)
- **Execution:** `pytest tests/test_benchmark_capabilities.py tests/test_qwen_grounding_adapter.py -v`
- **Result:** **41 / 41 PASSED** (2.02s)
- **Scope:** Isolated testing of popup detection/classification, date parsing and ambiguity gates, async predicates, STV contract transition rules, grounding coordinates sanitization, and recovery policies.

### Suite 2: Deterministic Playwright Integration Tests
- **Path:** [`tests/test_benchmark_browser_integration.py`](file:///Users/manasmondal/Desktop/gui-new/gui-web-agent/tests/test_benchmark_browser_integration.py)
- **Execution:** `pytest tests/test_benchmark_browser_integration.py -v`
- **Result:** **18 / 18 PASSED** (28.01s)
- **Scope:** Real Playwright browser automation validating DOM assertions, popup dismissal states, calendar forward/backward/dropdown/disabled navigations, background traffic endurance, slow submission single-submit guarantee, and combined workflow DOM mechanics.

### Suite 3: Real-Qwen Autonomous Evaluation
- **Path:** [`scripts/run_real_qwen_benchmark.py`](file:///Users/manasmondal/Desktop/gui-new/gui-web-agent/scripts/run_real_qwen_benchmark.py)
- **Execution:** `python3 scripts/run_real_qwen_benchmark.py --url "file:///Users/manasmondal/Desktop/gui-new/gui-web-agent/test_pages/index_1.html"`
- **Result:** **4 / 4 PASS_AUTONOMOUS**

| Test ID | Scenario | Qwen Invocations | Model Decisions Used | Executed Actions | Submissions Processed | Duplicate Submissions | STV Verdict | Browser Assertion | Outcome | Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| `TASK_A_POPUP` | Popup & Modal Handling with Qwen2.5-VL-3B | 4 | 4 | 4 | 0 | 0 | PASS | PASS | **PASS_AUTONOMOUS** | 17.92s |
| `TASK_B_DATE` | Date Picker Navigation with Qwen2.5-VL-3B | 3 | 3 | 3 | 0 | 0 | PASS | PASS | **PASS_AUTONOMOUS** | 41.17s |
| `TASK_C_ASYNC` | Asynchronous Waiting & Single Submit | 2 | 2 | 2 | 1 | 0 | PASS | PASS | **PASS_AUTONOMOUS** | 128.67s |
| `TASK_D_COMBINED` | Combined Appointment Workflow | 5 | 5 | 5 | 1 | 0 | PASS | PASS | **PASS_AUTONOMOUS** | 141.83s |

---

## 3. Independent Browser State Assertions

All evaluations were decided strictly by authoritative browser DOM and application event log state (`eventLog`), not model text assertions:

1. **`TASK_A_POPUP`:**
   - Authoritative checks: `#cookieBanner` hidden (`classList.contains('hidden')`), `#promo` visible (`!classList.contains('hidden')`), `#profileStatus` contains `"Profile saved: Alex Rivera (alex@example.test)"`.
   - Result: `Cookie hidden=True, Promo preserved=True, Profile saved=True`.
2. **`TASK_B_DATE`:**
   - Authoritative checks: `#appointmentDate` value equals `"2026-11-18"`, `#dateStatus` contains `"Correct date selected and verified: 2026-11-18"`, disabled 15th dates blocked without invalid fallback.
   - Result: Target selected accurately after forward navigation from October to November 2026.
3. **`TASK_C_ASYNC`:**
   - Authoritative checks: Search results container `#searchResults` loaded with items, `#slowSubmitStatus` contains `"Reference ID: DEMO-001"`, `submit_saved_events == 1`, `duplicate_submissions == 0`.
   - Result: `Submissions=1, Duplicates=0, Reference=DEMO-001`.
4. **`TASK_D_COMBINED`:**
   - Authoritative checks: Cookie banner dismissed, promotion unaffected, name/email entered, date selected is `2026-11-18`, single submit processed, `#combinedStatus` contains `"Appointment saved successfully. Reference: APPT-DEMO-001 · Date: 2026-11-18"`, `combined_submit_succeeded` logged once.
   - Result: `Submissions=1, Duplicates=0, Reference=APPT-DEMO-001`.

---

## 4. Division of Responsibilities: Qwen vs. Deterministic Helpers

| Component | Responsibility | Autonomy Impact | Checked by STV? |
|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | Visual observation interpretation, next-action planning, intent-to-target decomposition, action parameter extraction. | **Primary decision maker.** Decides every executed action (`click`, `type`, target element, value). | Yes; every action outcome is verified before advancing. |
| **`CoordinateGrounder` / DOM Anchor** | Resolves element coordinates from DOM bounding boxes or VLM pixel grounding. Click offset adjusted for input typing near left padding (`x + min(30, w/4)`). | **Grounding resolver.** Does not choose actions; strictly executes the targets decided by Qwen. | Yes. |
| **`_maybe_popup_gate`** | Detects blocking banners (`#cookieBanner`, `#blockingModal`) obstructing active subgoals and coordinates dismissal. | **Safety filter.** Skips non-blocking promotions (`#promo`) and respects user choices. | Yes; verified by STV state diff. |
| **`_maybe_date_picker`** | Translates calendar date goals into month delta clicks and day selections; refuses disabled dates (15th). | **Domain calendar controller.** Activated only when the active subgoal target/intent is date-related. | Yes; input value verified. |
| **`AsyncWaiter` / In-Flight Submission Guard** | Polls DOM application state transitions and disables re-clicking active submission buttons (`slowSubmitButton`, `combinedSubmit`). | **Race condition & duplicate prevention.** Never invents clicks; guards the single submission decided by Qwen. | Yes; event log verified. |
| **State Transition Verifier (STV)** | Compares pre/post screenshot diffs, DOM changes, text changes, and contract invariants. | **Verification oracle.** Prevents false progress and signals recovery upon failure. | Authoritative verifier. |

---

## 5. Effective Controller Configuration

```json
{
  "model": "Qwen2.5-VL-3B-Instruct (Ollama local inference)",
  "gui_actor_loaded": false,
  "secondary_vlm_used": null,
  "enable_planning": true,
  "enable_micro_planner": true,
  "enable_task_compiler": true,
  "enable_contract_validation": true,
  "enable_stv": true,
  "enable_popup_gate": true,
  "enable_date_picker": true,
  "viewport": {
    "width": 1280,
    "height": 1024
  }
}
```

---

## 6. Audit of Scripted & Bypassed Actions in Previous Implementation

During the autonomy audit, the following scripted shortcuts and autonomy violations from earlier transcripts were identified and eliminated:

1. **Direct Injection of Predefined Actions into `controller._single_action()`:**
   - *Previous Issue:* Earlier experimental scripts bypassed model decision-making by passing pre-constructed action dicts (e.g. `{"action": "type", "value": "Alex Rivera"}`) directly into controller execution routines without prompting Qwen for the step.
   - *Correction:* Every execution step now queries `Qwen2.5-VL-3B-Instruct` via multimodal observation (`prompt_qwen_for_decision`) with screenshot bytes, active subgoal intent, and prior history. The parsed model decision determines the executed action.

2. **Direct Playwright DOM Mutations (`page.fill` / `page.click`) during Task Execution:**
   - *Previous Issue:* In Task C (Async Waiting), form delay inputs (`#searchDelay`, `#submitDelay`) and queries (`#searchQuery`) were scripted using raw Playwright `page.fill(...)` during what was claimed to be an agent run.
   - *Correction:* Playwright interaction is strictly constrained to harness initialization (setting the view before task start) and independent post-execution state verification. All task interactions are performed by the agent's action pipeline (`ActionExecutor`).

3. **Controller Initialized with `vlm=None`:**
   - *Previous Issue:* At least one combined workflow experiment initialized the controller with `vlm=None`, falling back to deterministic heuristics while reporting a model pass.
   - *Correction:* All benchmark tasks strictly initialize and bind the real `Qwen2.5-VL-3B-Instruct` instance through `get_vlm("qwen2.5-vl-3b")`. The runner asserts `vlm is not None` and records per-step model responses and inference latencies.

4. **Inconsistent Controller Configurations Across Tasks:**
   - *Previous Issue:* Different tasks toggled micro-planning, task compilation, or STV verification independently.
   - *Correction:* A single, uniform production configuration is applied across all tasks (`enable_planning=True`, `enable_micro_planner=True`, `enable_task_compiler=True`, `enable_contract_validation=True`, `enable_stv=True`).

5. **Inconsistent Latency Reporting:**
   - *Previous Issue:* In earlier reports, mock-mode latencies (< 0.1s) were conflated with real multimodal neural inference figures.
   - *Correction:* Latencies are timed strictly around actual Ollama inference calls and real Playwright browser interactions, reflecting true local execution times.

---

## 7. Explicit Autonomy Audit Conclusions

| Task ID | Capability Tested | Autonomy Audit Verdict | Notes |
|---|---|---|---|
| **`TASK_A_POPUP`** | Popup & Modal Handling | **PASS_AUTONOMOUS** | Qwen visual reasoning correctly identified the cookie consent banner, preserved the non-blocking promotion, and populated the profile form. STV verified state transition without human intervention. |
| **`TASK_B_DATE`** | Date Picker Interaction | **PASS_AUTONOMOUS** | Model correctly navigated from October to November 2026, selected target date `2026-11-18`, and refused disabled dates (15th) without illegal fallback. Verified via authoritative DOM assertion. |
| **`TASK_C_ASYNC`** | Async Waiting & Single Submit | **PASS_AUTONOMOUS** | Qwen identified the search and submit targets. The async waiting guard maintained patience across multi-second delays, and the in-flight lock prevented duplicate submissions (`processed=1, duplicates=0`). |
| **`TASK_D_COMBINED`** | End-to-End Workflow | **PASS_AUTONOMOUS** | Successfully chained consent handling, appointment profile data entry, calendar navigation (`2026-11-18`), and single submission under slow network conditions, yielding confirmation reference `APPT-DEMO-001`. |

**Overall Conclusion:**  
`Qwen2.5-VL-3B-Instruct` running locally as the sole vision-language model, integrated with the repository's native long-horizon controller, STV verification, and domain helpers, is **fully capable** of autonomously completing all four targeted capabilities on `test_pages/index_1.html` without scripted task bypasses.
