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

    # 1. Market Capture = (10/100) * 5,000,000 = 500,000
    # 2. Score = 0.5 + (0.6 + 0.7 + 0.5)/3 = 1.1
    # 3. Revenue = 500,000 * 1.1 = 550,000
    # 4. Debt Penalty = (20/100) * 0.5 = 0.1
    # 5. Burn = 50,000 * 1.1 = 55,000
    # 6. Budget = 1,000,000 + 550,000 - 55,000 = 1,495,000
    assert new_state.budget == 1495000.0
    assert new_state.revenue == 550000.0
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
