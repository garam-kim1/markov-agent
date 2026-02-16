import psutil
from rich.console import Console

console = Console()


def check_memory_usage(threshold_percent: float = 90.0) -> bool:
    """Check if the system memory usage is above the threshold.

    Returns:
        True if memory usage is safe, False if it's dangerous.

    """
    mem = psutil.virtual_memory()
    if mem.percent > threshold_percent:
        console.log(
            f"[bold red]DANGER: System memory usage at {mem.percent}% "
            f"(Threshold: {threshold_percent}%)![/bold red]"
        )
        return False
    return True


async def memory_guard(threshold_percent: float = 90.0) -> None:
    """Raise an error if memory usage is too high."""
    if not check_memory_usage(threshold_percent):
        msg = f"System memory usage too high: {psutil.virtual_memory().percent}%"
        raise MemoryError(msg)
