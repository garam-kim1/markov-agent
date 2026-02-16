import logging

import psutil
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResourceGovernor(BaseModel):
    """Monitors system resources and enforces safety limits."""

    memory_threshold_percent: float = Field(
        default=90.0, description="Max allowed memory usage percent."
    )
    cpu_threshold_percent: float = Field(
        default=95.0, description="Max allowed CPU usage percent (optional)."
    )

    def check_safety(self) -> bool:
        """Check if current resource usage is within safe limits."""
        mem = psutil.virtual_memory()
        if mem.percent > self.memory_threshold_percent:
            logger.error(
                "Memory threshold exceeded: %s%% > %s%%",
                mem.percent,
                self.memory_threshold_percent,
            )
            return False

        cpu = psutil.cpu_percent(interval=None)
        if cpu > self.cpu_threshold_percent:
            logger.warning(
                "CPU threshold reached: %s%% > %s%%", cpu, self.cpu_threshold_percent
            )
            # We might not VETO on CPU alone, but we could.

        return True

    def enforce(self) -> None:
        """Raise an error if safety check fails."""
        if not self.check_safety():
            mem = psutil.virtual_memory()
            msg = f"Resource limit exceeded. Memory: {mem.percent}%"
            raise MemoryError(msg)
