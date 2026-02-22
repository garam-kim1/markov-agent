import pytest

from examples.company_digital_twin import (
    build_graph,
    generate_market_event,
)


@pytest.mark.asyncio
async def test_company_graph_structure():
    graph = build_graph()
    assert graph.name == "CorpCommand"
    # CEO + 6 Depts = 7 Nodes
    assert len(graph.nodes) == 7
    assert graph.entry_point == "CEO"


def test_market_event_generation():
    event, impacts = generate_market_event()
    assert isinstance(event, str)
    assert isinstance(impacts, dict)
