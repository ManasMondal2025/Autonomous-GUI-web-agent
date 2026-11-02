"""Recovery policies with configurable budgets."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RecoveryManager:
    def __init__(
        self,
        max_re_ground_attempts: int = 2,
        max_recovery_attempts: int = 5,
        max_execution_retries: int = 3,
    ):
        self.max_re_ground_attempts = max_re_ground_attempts
        self.max_recovery_attempts = max_recovery_attempts
        self.max_execution_retries = max_execution_retries
        self.recovery_count = 0
        self.reground_count = 0
        self.exec_retry_count = 0

    def should_recover(self, verification: Dict[str, Any]) -> bool:
        if verification.get("success"):
            return False
        return self.recovery_count < self.max_recovery_attempts

    def record_recovery(self, strategy: str) -> Dict[str, Any]:
        self.recovery_count += 1
        logger.info(f"Recovery #{self.recovery_count}: {strategy}")
        return {"strategy": strategy, "attempt": self.recovery_count}

    def can_retry_execution(self) -> bool:
        return self.exec_retry_count < self.max_execution_retries

    def record_exec_retry(self) -> None:
        self.exec_retry_count += 1

    def reset_step_counters(self) -> None:
        self.reground_count = 0
        self.exec_retry_count = 0

    def exhausted(self) -> bool:
        return self.recovery_count >= self.max_recovery_attempts
