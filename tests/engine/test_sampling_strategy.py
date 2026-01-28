import pytest

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.sampler import (
    SamplingStrategy,
    execute_parallel_sampling,
    generate_varied_configs,
)


def test_generate_varied_configs_uniform():
    base = {"temperature": 0.5, "top_p": 0.9}
    configs = generate_varied_configs(base, k=3, strategy=SamplingStrategy.UNIFORM)
    assert len(configs) == 3
    for c in configs:
        assert c["temperature"] == 0.5
        assert c["top_p"] == 0.9


def test_generate_varied_configs_linear_ramp():
    base = {"temperature": 0.5}
    # Range default: 0.1 to 1.2
    configs = generate_varied_configs(base, k=3, strategy=SamplingStrategy.LINEAR_RAMP)
    assert len(configs) == 3
    # k=3: 0 -> min, 1 -> mid, 2 -> max
    # 0.1, (0.1+1.2)/2=0.65, 1.2
    assert configs[0]["temperature"] == 0.1
    assert configs[1]["temperature"] == 0.65
    assert configs[2]["temperature"] == 1.2


def test_generate_varied_configs_linear_decay():
    base = {"temperature": 0.5}
    configs = generate_varied_configs(base, k=2, strategy=SamplingStrategy.LINEAR_DECAY)
    assert len(configs) == 2
    # 0 -> max (1.2), 1 -> min (0.1)
    assert configs[0]["temperature"] == 1.2
    assert configs[1]["temperature"] == 0.1


def test_generate_varied_configs_diverse():
    base = {"temperature": 0.5, "top_p": 0.9}
    configs = generate_varied_configs(base, k=5, strategy=SamplingStrategy.DIVERSE)
    assert len(configs) == 5
    # First one should be base (or close to it? Implementation kept index 0 as base)
    # Wait, my implementation said: "if i == 0: continue" -> so config[0] is copy of base.
    assert configs[0]["temperature"] == 0.5
    assert configs[0]["top_p"] == 0.9

    # Others should be randomized (likely different, but technically could collide)
    # We just check they exist.
    for i in range(1, 5):
        assert "temperature" in configs[i]
        assert "top_p" in configs[i]


def test_adk_controller_create_variant():
    config = ADKConfig(model_name="mock", temperature=0.5)
    retry = RetryPolicy()
    ctl = ADKController(config, retry)

    # Base check
    # controller.agent.generate_content_config is an object, hard to inspect directly
    # without knowing google-adk internals perfectly, but we can check config attribute
    assert ctl.config.temperature == 0.5

    # Create variant
    variant_ctl = ctl.create_variant({"temperature": 0.9, "top_p": 0.8})
    assert variant_ctl.config.temperature == 0.9
    assert variant_ctl.config.generation_config is not None
    assert variant_ctl.config.generation_config["temperature"] == 0.9
    assert variant_ctl.config.generation_config["top_p"] == 0.8

    # Original should be unchanged
    assert ctl.config.temperature == 0.5


@pytest.mark.asyncio
async def test_execute_parallel_sampling_varied():
    async def t1():
        return "A"

    async def t2():
        return "B"

    # Pass a selector that returns the full list to verify all tasks ran
    results = await execute_parallel_sampling([t1, t2], k=2, selector_func=lambda x: x)

    # asyncio.gather preserves order
    assert results[0] == "A"
    assert results[1] == "B"


@pytest.mark.asyncio
async def test_parallel_sampling_partial_failure():
    """
    Verifies that if some tasks fail, the selector still receives the valid results.
    """
    async def success_task():
        return "Success"

    async def fail_task():
        raise ValueError("Oops")

    # Run 3 tasks: 2 success, 1 fail
    # Note: execute_parallel_sampling with list input ignores 'k'
    tasks = [success_task, fail_task, success_task]
    
    # Selector should see 2 "Success" strings
    def selector(results):
        assert len(results) == 2
        assert all(r == "Success" for r in results)
        return "Selected"

    result = await execute_parallel_sampling(tasks, selector_func=selector)
    assert result == "Selected"

@pytest.mark.asyncio
async def test_parallel_sampling_all_failure():
    """
    Verifies that if all tasks fail, the exception is raised.
    """
    async def fail_task():
        raise ValueError("All Fail")
        
    with pytest.raises(ValueError, match="All Fail"):
        await execute_parallel_sampling([fail_task, fail_task])
