import pytest
from markov_agent.governance.cost import CostGovernor
from markov_agent.engine.adk_wrapper import ADKConfig

def test_cost_governor_routing():
    cheap = ADKConfig(model_name="flash")
    reasoning = ADKConfig(model_name="pro")
    
    governor = CostGovernor(
        cheap_config=cheap,
        reasoning_config=reasoning
    )
    
    # Low complexity -> cheap
    assert governor.route_request(0.2).model_name == "flash"
    # High complexity -> reasoning
    assert governor.route_request(0.8).model_name == "pro"
    # Medium complexity -> fallback to cheap if standard not set
    assert governor.route_request(0.5).model_name == "flash"

def test_cost_governor_budget():
    cheap = ADKConfig(model_name="flash")
    reasoning = ADKConfig(model_name="pro")
    
    governor = CostGovernor(
        cheap_config=cheap,
        reasoning_config=reasoning,
        cost_budget=1.0,
        token_budget=1000
    )
    
    assert governor.check_budget(estimated_cost=0.5) is True
    assert governor.check_budget(estimated_cost=1.5) is False
    
    governor.record_usage(cost=0.8, tokens=500)
    assert governor.check_budget(estimated_cost=0.3) is False
    assert governor.check_budget(estimated_tokens=600) is False
    assert governor.check_budget(estimated_tokens=100) is True
