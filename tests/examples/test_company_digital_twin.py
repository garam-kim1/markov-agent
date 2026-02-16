import pytest

from examples.company_digital_twin import BusinessTwin, CompanyState, build_complex_corp


@pytest.mark.asyncio
async def test_business_twin_evolution():
    twin = BusinessTwin()
    state = CompanyState(
        budget=1000000.0,
        market_share=10.0,
        reputation=70.0,
        product_quality=60.0,
        talent_index=50.0,
        innovation_index=30.0,
        burn_rate=50000.0,
    )

    new_state = await twin.evolve_world(state)

    # Revenue = 10 * 200,000 * (60/50) * (70/70) * (50/50) * (30/30) = 2,000,000 * 1.2 = 2,400,000
    # Net Revenue = 2,400,000 * 0.85 = 2,040,000
    # New Budget = 1,000,000 + 2,040,000 - 50,000 = 2,990,000
    assert new_state.budget == 2990000.0
    assert new_state.revenue == 2400000.0
    assert new_state.reputation < 70.0
    assert new_state.technical_debt > 20.0


@pytest.mark.asyncio
async def test_swarm_execution_mock():
    # We can't easily mock the entire swarm's LLM calls without changing the build_complex_corp
    # but we can test if it builds and runs with a mock responder if we inject it.
    # For now, let's just test that the components exist and can be instantiated.
    swarm = build_complex_corp()
    assert swarm.name == "ComplexCorp"
    # 1 supervisor (CEO) + 6 workers (Marketing, Engineering, HR, Operations, RD, Legal)
    assert len(swarm.nodes) == 7
