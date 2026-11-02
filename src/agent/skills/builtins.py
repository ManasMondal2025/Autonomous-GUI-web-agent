"""Built-in reusable skills for browser-agent reliability."""

from .models import ProcedureStep, SkillDefinition


DATE_PICKER_SKILL = SkillDefinition(
    name="date_picker",
    version=1,
    purpose="Select and verify a target date on native or custom calendar interfaces.",
    applicable_task_types=["date_selection", "calendar_interaction", "appointment_booking"],
    inputs=["target_date", "date_field_selector"],
    preconditions=[
        "Browser page is available and active",
        "Target date is specified in recognizable format",
        "Date picker trigger button or input element is identified in the DOM",
    ],
    procedure=[
        ProcedureStep(
            step_id=1,
            intent="Open calendar interface",
            action_type="click",
            target="date_field_selector",
            expected_transition={"must_change": ["calendar visible"]},
        ),
        ProcedureStep(
            step_id=2,
            intent="Inspect current displayed month and navigate to target month",
            action_type="click",
            target="month_nav_button",
            optional=True,
            expected_transition={"must_change": ["month header updated"]},
        ),
        ProcedureStep(
            step_id=3,
            intent="Select target day from active month",
            action_type="click",
            target="target_day_button",
            expected_transition={"must_change": ["date value selected"]},
        ),
    ],
    success_criteria=[
        "Input or displayed appointment date matches target_date exactly",
        "Target day selected belongs to target month (not adjacent month)",
        "Selected date was not disabled",
    ],
    expected_state_transitions={
        "must_change": ["input value updated to target date"],
        "must_not_happen": ["selection of disabled date", "selection of wrong month day"],
    },
    failure_conditions=[
        "Calendar trigger element not found",
        "Target date is disabled",
        "Month navigation delta exceeded configured budget",
        "Selected date does not match expected target",
    ],
    recovery={
        "strategy": "reobserve_and_reground",
        "max_attempts": 2,
        "fallback": "native_input_fill",
    },
    verification_requirements=[
        "STV date selection verification",
        "DOM input value equality check",
    ],
    known_limitations=[
        "Custom date pickers without standard ARIA or month labels require VLM visual grounding.",
        "Non-standard adjacent-month dimming requires DOM class inspection.",
    ],
    evidence="Validated on local benchmark index_1.html with 100% accuracy on forward/backward and disabled checks.",
)


DROPDOWN_SELECTION_SKILL = SkillDefinition(
    name="dropdown_selection",
    version=1,
    purpose="Interact with and select options from standard HTML <select> or custom ARIA comboboxes.",
    applicable_task_types=["dropdown_selection", "option_picker", "form_choice"],
    inputs=["dropdown_target", "option_label_or_value"],
    preconditions=[
        "Dropdown element is visible and interactive in the DOM",
        "Desired option label or value is known",
    ],
    procedure=[
        ProcedureStep(
            step_id=1,
            intent="Click or focus dropdown element to reveal options",
            action_type="click",
            target="dropdown_target",
            expected_transition={"must_change": ["options list visible or dropdown focused"]},
        ),
        ProcedureStep(
            step_id=2,
            intent="Select requested option by text label or value",
            action_type="select_option",
            target="dropdown_target",
            value="option_label_or_value",
            expected_transition={"must_change": ["selected option updated"]},
        ),
    ],
    success_criteria=[
        "Selected value or displayed label equals the requested option",
    ],
    expected_state_transitions={
        "must_change": ["dropdown value updated"],
    },
    failure_conditions=[
        "Dropdown element not interactable",
        "Requested option not present in dropdown options list",
    ],
    recovery={
        "strategy": "scroll_into_view_and_reclick",
        "max_attempts": 2,
        "fallback": "type_to_filter",
    },
    verification_requirements=[
        "Verify select.value equals option value",
    ],
    known_limitations=[
        "Virtual scrolling dropdowns require incremental scroll actions.",
    ],
    evidence="Validated on benchmark dropdown scenarios.",
)


ASYNC_WAITING_SKILL = SkillDefinition(
    name="asynchronous_waiting",
    version=1,
    purpose="Poll for application state completion and prevent duplicate action submission.",
    applicable_task_types=["ajax_search", "slow_submit", "condition_wait"],
    inputs=["trigger_action", "completion_predicate", "timeout_ms"],
    preconditions=[
        "Page is active",
        "Observable DOM predicate or success indicator is defined",
    ],
    procedure=[
        ProcedureStep(
            step_id=1,
            intent="Execute trigger action once",
            action_type="click",
            target="trigger_target",
            expected_transition={"must_change": ["processing status or loading state initiated"]},
        ),
        ProcedureStep(
            step_id=2,
            intent="Wait for DOM predicate resolution without duplicate trigger clicks",
            action_type="wait",
            target="predicate_monitor",
            expected_transition={"must_change": ["completion status indicator rendered"]},
        ),
    ],
    success_criteria=[
        "Application reaches verified success state within timeout",
        "No duplicate actions or duplicate requests were dispatched",
    ],
    expected_state_transitions={
        "must_change": ["status reaches final state"],
        "must_not_happen": ["duplicate submit started event"],
    },
    failure_conditions=[
        "Explicit error banner or failure status appears",
        "Timeout exceeded without condition satisfied",
    ],
    recovery={
        "strategy": "inspect_diagnostics",
        "max_attempts": 1,
        "fallback": "escalate_to_hitl",
    },
    verification_requirements=[
        "Authoritative event log check",
        "DOM inner_text confirmation",
    ],
    known_limitations=[
        "Persistent background telemetry traffic requires condition waiting, not networkidle.",
    ],
    evidence="Validated on index_1.html with 1500ms slow submit and 1200ms delayed search.",
)


PAGINATION_SKILL = SkillDefinition(
    name="pagination",
    version=1,
    purpose="Navigate multi-page record lists while tracking current page index and item presence.",
    applicable_task_types=["pagination", "search_results_traversal", "table_navigation"],
    inputs=["target_page_or_item", "next_button_selector"],
    preconditions=[
        "Paginated container or pagination controls identified in DOM",
        "Next page button is not disabled",
    ],
    procedure=[
        ProcedureStep(
            step_id=1,
            intent="Check current page for target item or verify current page index",
            action_type="scroll",
            target="table_container",
            expected_transition={"must_change": ["view scrolled"]},
        ),
        ProcedureStep(
            step_id=2,
            intent="Advance to next page via Next pagination control",
            action_type="click",
            target="next_button_selector",
            expected_transition={"must_change": ["page index incremented or records refreshed"]},
        ),
    ],
    success_criteria=[
        "Current page index increments or target record becomes visible",
    ],
    expected_state_transitions={
        "must_change": ["active page number updated", "new records visible"],
    },
    failure_conditions=[
        "Next button disabled or not found",
        "End of records reached without finding target",
    ],
    recovery={
        "strategy": "reground_pagination_controls",
        "max_attempts": 2,
        "fallback": "return_to_first_page",
    },
    verification_requirements=[
        "Verify page indicator or active page number changed",
    ],
    known_limitations=[
        "Infinite scroll interfaces require wheel events instead of button clicks.",
    ],
    evidence="Tested on multi-page tabular benchmark fixtures.",
)


FORM_FILLING_SKILL = SkillDefinition(
    name="form_filling",
    version=1,
    purpose="Sequentially populate structured form fields and verify entered values before submit.",
    applicable_task_types=["form_filling", "profile_save", "data_entry"],
    inputs=["field_value_map", "submit_button_selector"],
    preconditions=[
        "Form container and input fields are visible and interactable",
        "Field values are non-empty and validated",
    ],
    procedure=[
        ProcedureStep(
            step_id=1,
            intent="Focus and fill each input field with mapped value",
            action_type="type",
            target="field_selector",
            expected_transition={"must_change": ["input value populated"]},
        ),
        ProcedureStep(
            step_id=2,
            intent="Read back and verify all populated values prior to submission",
            action_type="wait",
            target="form_container",
            expected_transition={"must_be_true": ["all inputs contain expected text"]},
        ),
    ],
    success_criteria=[
        "All mapped input fields contain expected values in the DOM",
        "No form validation errors triggered on blur",
    ],
    expected_state_transitions={
        "must_change": ["form values populated"],
        "must_not_happen": ["submission before values verified"],
    },
    failure_conditions=[
        "Required input element not found in DOM",
        "Input value readback mismatch",
    ],
    recovery={
        "strategy": "clear_and_retype",
        "max_attempts": 2,
        "fallback": "keyboard_tab_navigation",
    },
    verification_requirements=[
        "DOM input.value equality for all specified fields",
    ],
    known_limitations=[
        "Masked credit card or phone inputs require typing with format delay.",
    ],
    evidence="Validated on TASK_A profile form and TASK_D combined appointment form.",
)


BUILTIN_SKILLS = [
    DATE_PICKER_SKILL,
    DROPDOWN_SELECTION_SKILL,
    ASYNC_WAITING_SKILL,
    PAGINATION_SKILL,
    FORM_FILLING_SKILL,
]


def get_builtin_skills():
    """Return a list of all built-in skill definitions."""
    return list(BUILTIN_SKILLS)

