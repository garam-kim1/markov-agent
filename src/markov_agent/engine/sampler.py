import asyncio
import copy
import logging
import random
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, cast

from markov_agent.engine.diversity import DiversityMetrics

logger = logging.getLogger(__name__)


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
    selector_func: Callable[[list[Any]], T | Awaitable[T]] | None = None,
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
            _safe_generate(cast("Callable[[], Any]", func)) for func in generate_func
        )
    else:
        # Homogeneous tasks
        tasks.extend(_safe_generate(generate_func) for _ in range(k))

    results = await asyncio.gather(*tasks)

    valid_results = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]

    if failures:
        logger.warning("%s/%s parallel samples failed.", len(failures), len(results))
        for i, f in enumerate(failures):
            logger.debug("Failure %s: %s", i + 1, f)

    if not valid_results:
        # If all failed, raise the first error
        if results and isinstance(results[0], Exception):
            raise results[0]
        msg = "All parallel samples failed."
        raise RuntimeError(msg)

    if selector_func:
        res = selector_func(valid_results)
        if asyncio.iscoroutine(res):
            return cast("T", await res)
        return cast("T", res)

    # Default: return the first valid result (highest confidence / first completed)
    return valid_results[0]


async def execute_diverse_sampling[T](
    generate_factory: Callable[[dict[str, Any]], Awaitable[T]],
    base_config: dict[str, Any],
    k: int = 5,
    diversity_threshold: float = 0.3,
    max_retries: int = 2,
) -> list[T]:
    """Execute sampling and ensure the output set is diverse.

    If diversity is below threshold, it retries with increased temperature.
    """
    current_config = copy.deepcopy(base_config)
    best_results: list[T] = []
    best_diversity = -1.0

    for attempt in range(max_retries + 1):
        # Generate varied configs for this attempt
        configs = generate_varied_configs(
            current_config, k, strategy=SamplingStrategy.DIVERSE
        )

        # Execute parallel sampling
        tasks = [generate_factory(cfg) for cfg in configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [
            r for r in results if not isinstance(r, (Exception, BaseException))
        ]

        if not valid_results:
            continue

        # We can safely cast because we filtered out exceptions
        typed_results = cast("list[T]", valid_results)

        # Calculate diversity
        texts = [str(r) for r in typed_results]
        metrics = DiversityMetrics(texts)
        # Use Jaccard as primary diversity metric
        diversity = metrics.jaccard

        logger.info(
            "Sampling Attempt %s: Diversity = %.2f (Threshold: %.2f)",
            attempt + 1,
            diversity,
            diversity_threshold,
        )

        if diversity >= diversity_threshold:
            return typed_results

        if diversity > best_diversity:
            best_diversity = diversity
            best_results = typed_results

        # Increase temperature for next attempt to force more exploration
        current_temp = current_config.get("temperature", 0.7)
        current_config["temperature"] = min(current_temp + 0.2, 1.5)

    logger.warning(
        "Could not reach diversity threshold %.2f after %s attempts. Best was %.2f",
        diversity_threshold,
        max_retries + 1,
        best_diversity,
    )
    return best_results


async def _safe_generate(func: Callable[[], Any]) -> Any:
    try:
        # Check if func is awaitable or returns awaitable
        res = func()
        if asyncio.iscoroutine(res):
            return await res
    except Exception as e:
        return e
    else:
        return res
