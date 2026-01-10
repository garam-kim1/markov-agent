import asyncio
from collections.abc import Callable
from typing import Any


async def execute_parallel_sampling[T](
    generate_func: Callable[[], Any],
    k: int = 5,
    selector_func: Callable[[list[Any]], T] = None,
) -> T:
    """
    Implements pass@k logic.
    1. Exploration: Call the model k times in parallel.
    2. Verification: (Optional) Run a critic/selector.
    3. Selection: Return best response.
    """

    async def safe_generate():
        try:
            return await generate_func()
        except Exception as e:
            return e

    tasks = [safe_generate() for _ in range(k)]
    results = await asyncio.gather(*tasks)

    valid_results = [r for r in results if not isinstance(r, Exception)]

    if not valid_results:
        if results and isinstance(results[0], Exception):
            raise results[0]
        raise RuntimeError("All parallel samples failed.")

    if selector_func:
        return selector_func(valid_results)

    return valid_results[0]
