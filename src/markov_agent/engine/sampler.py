import asyncio
import copy
import random
from collections.abc import Callable
from enum import StrEnum
from typing import Any, cast


class SamplingStrategy(StrEnum):
    """Strategies for varying generation parameters across parallel samples."""

    UNIFORM = "uniform"  # All samples use the same configuration
    LINEAR_RAMP = (
        "linear_ramp"  # Temperature linearly increases from 0 to max (or base)
    )
    LINEAR_DECAY = "linear_decay"  # Temperature linearly decreases
    DIVERSE = "diverse"  # Random variations in temperature and top_p


def generate_varied_configs(
    base_config: dict[str, Any],
    k: int,
    strategy: SamplingStrategy,
    param_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Generate 'k' variations of the base configuration dictionary based on the strategy.

    Target parameters are usually 'temperature' and 'top_p'.

    Args:
        base_config: The starting configuration (e.g. generation_config).
        k: Number of samples.
        strategy: The sampling strategy to apply.
        param_ranges: Optional min/max bounds. Defaults to sensible LLM defaults.
            e.g. {"temperature": (0.1, 1.0), "top_p": (0.5, 1.0)}

    """
    configs = [copy.deepcopy(base_config) for _ in range(k)]

    if k <= 1 or strategy == SamplingStrategy.UNIFORM:
        return configs

    # Default ranges
    ranges = {"temperature": (0.1, 1.2), "top_p": (0.7, 1.0)}
    if param_ranges:
        ranges.update(param_ranges)

    t_min, t_max = ranges["temperature"]
    p_min, p_max = ranges["top_p"]

    for i, cfg in enumerate(configs):
        if strategy == SamplingStrategy.LINEAR_RAMP:
            # 0 -> k-1 maps to t_min -> t_max
            # If k=2: 0->t_min, 1->t_max
            step = (t_max - t_min) / (k - 1) if k > 1 else 0
            new_temp = t_min + (i * step)
            cfg["temperature"] = round(new_temp, 2)

        elif strategy == SamplingStrategy.LINEAR_DECAY:
            # 0 -> k-1 maps to t_max -> t_min
            step = (t_max - t_min) / (k - 1) if k > 1 else 0
            new_temp = t_max - (i * step)
            cfg["temperature"] = round(new_temp, 2)

        elif strategy == SamplingStrategy.DIVERSE:
            # Randomize temperature and top_p
            # We keep the first one as "Anchor" (original config) for stability?
            # Or just pure chaos? Let's keep index 0 as base, rest random.
            if i == 0:
                continue

            cfg["temperature"] = round(random.uniform(t_min, t_max), 2)  # noqa: S311
            cfg["top_p"] = round(random.uniform(p_min, p_max), 2)  # noqa: S311

    return configs


async def execute_parallel_sampling[T](
    generate_func: Callable[[], Any] | list[Callable[[], Any]],
    k: int = 5,
    selector_func: Callable[[list[Any]], T] | None = None,
) -> T:
    """Implement pass@k logic with optional task variance.

    Args:
        generate_func: Either a single factory function (called k times)
                       or a list of specific factory functions (k is ignored/inferred).
        k: Number of samples (if generate_func is single).
        selector_func: Function to select the best result.

    """
    tasks = []

    if isinstance(generate_func, list):
        # We have specific tasks (likely with varied configs)
        tasks.extend(
            _safe_generate(cast(Callable[[], Any], func)) for func in generate_func
        )
    else:
        # Homogeneous tasks
        tasks.extend(_safe_generate(generate_func) for _ in range(k))

    results = await asyncio.gather(*tasks)

    valid_results = [r for r in results if not isinstance(r, Exception)]

    if not valid_results:
        # If all failed, raise the first error
        if results and isinstance(results[0], Exception):
            raise results[0]
        msg = "All parallel samples failed."
        raise RuntimeError(msg)

    if selector_func:
        res = selector_func(valid_results)
        if asyncio.iscoroutine(res):
            return await res
        return res

    # Default: return the first valid result (highest confidence / first completed)
    return valid_results[0]


async def _safe_generate(func: Callable[[], Any]) -> Any:
    try:
        # Check if func is awaitable or returns awaitable
        res = func()
        if asyncio.iscoroutine(res):
            return await res
        return res
    except Exception as e:
        return e
