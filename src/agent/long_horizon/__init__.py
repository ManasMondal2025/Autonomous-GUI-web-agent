"""Long-horizon planning/state-transition-verification layer.

Additive only: composes existing components (Observer, grounders,
ActionExecutor, verifiers, TrajectoryLogger, VLM backends). Nothing here
replaces the baseline agent loop.
"""

from .task_memory import TaskMemory, FACT_TIERS, SUBGOAL_STATUSES, FACT_PROVENANCE
from .task_spec import (
    TaskSpec, TaskCompiler, CompilationResult,
    validate_spec_for_execution, spec_is_compatible_with_contract,
)
from .subgoal_contract import (
    SubgoalContract, ContractValidationError, Postcondition,
    validate_contract_pre_execution, repair_contract,
)
from .observation_builder import (
    build_observation_context, is_stale_observation,
)
from .experience_memory import ExperienceMemory
from .subgoal_planner import GlobalPlanner, LocalPlanner, PlannerError
from .micro_planner import MicroPlanner, MAX_ACTIONS as MICRO_MAX_ACTIONS
from .milestone_manager import MilestoneManager, MILESTONE_STATUSES
from .state_gate import check as state_gate_check
from . import failure_taxonomy
from . import progress
from .stv import STV, DECISIONS
from .state_extractor import extract_signature, compare_signatures
from .checkpoint_manager import CheckpointManager
from .recovery import RecoveryPolicy
from .controller import LongHorizonController

__all__ = [
    "TaskMemory", "FACT_TIERS", "SUBGOAL_STATUSES", "FACT_PROVENANCE",
    "TaskSpec", "TaskCompiler", "CompilationResult",
    "validate_spec_for_execution", "spec_is_compatible_with_contract",
    "SubgoalContract", "ContractValidationError", "Postcondition",
    "validate_contract_pre_execution", "repair_contract",
    "build_observation_context", "is_stale_observation",
    "ExperienceMemory",
    "GlobalPlanner", "LocalPlanner", "PlannerError",
    "MicroPlanner", "MICRO_MAX_ACTIONS",
    "MilestoneManager", "MILESTONE_STATUSES",
    "state_gate_check", "failure_taxonomy", "progress",
    "STV", "DECISIONS",
    "extract_signature", "compare_signatures",
    "CheckpointManager",
    "RecoveryPolicy",
    "LongHorizonController",
]
