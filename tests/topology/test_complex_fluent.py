from typing import Any

import pytest
from pydantic import Field

from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import FunctionalNode


class ComplexState(BaseState):
    """Test state for complex scenarios."""

    # Using append behavior for parallel logs
    logs: list[str] = Field(
        default_factory=list, json_schema_extra={"behavior": "append"}
    )
    counter: int = 0
    sub_result: str = ""


def task_a(state: ComplexState) -> dict[str, Any]:
    return {"logs": ["A executed"]}


def task_b(state: ComplexState) -> dict[str, Any]:
    return {"logs": ["B executed"]}


def sub_task(state: ComplexState) -> dict[str, Any]:
    return {"sub_result": "inner success", "logs": ["Sub executed"]}


@pytest.mark.asyncio
async def test_fluent_subgraph() -> None:
    """Test embedding a graph via .subgraph()."""
    # Inner Graph
    inner = Graph(name="inner", state_type=ComplexState)
    inner.task(sub_task)

    # Outer Graph
    outer = Graph(name="outer", state_type=ComplexState)
    _ = outer.task(lambda s: {"logs": ["Start"]}, name="start")

    # Embed inner
    _ = outer.subgraph(inner, name="nested_node")

    # Chain
    outer.chain(["start", "nested_node"])

    # Run
    state = ComplexState()
    result = await outer.run(state)

    assert "Start" in result.logs
    assert "Sub executed" in result.logs
    assert result.sub_result == "inner success"


@pytest.mark.asyncio
async def test_fluent_parallel() -> None:
    """Test parallel execution via .parallel()."""
    graph = Graph(name="parallel_test", state_type=ComplexState)

    # Define parallel block with callables (auto-wrapped)
    _ = graph.parallel(branches=[task_a, task_b], name="p_block")

    # Run
    state = ComplexState()
    result = await graph.run(state)

    # Check both ran (order is not guaranteed in parallel, but append handles it)
    assert "A executed" in result.logs
    assert "B executed" in result.logs


@pytest.mark.asyncio
async def test_fluent_mixed_parallel() -> None:
    """Test parallel block with mixed Node types (Graph, Callable, Node)."""
    # 1. Subgraph
    sub = Graph(name="sub_branch", state_type=ComplexState)
    sub.task(lambda s: {"logs": ["Sub Branch"]})

    # 2. Callable
    def func_branch(s: ComplexState) -> dict[str, Any]:
        return {"logs": ["Func Branch"]}

    # 3. Node
    node_branch = FunctionalNode(
        name="node_branch",
        func=lambda s: {"logs": ["Node Branch"]},
        state_type=ComplexState,
    )

    # Main Graph
    main = Graph(name="mixed_parallel", state_type=ComplexState)
    main.parallel([sub, func_branch, node_branch])

    # Run
    state = ComplexState()
    result = await main.run(state)

    assert "Sub Branch" in result.logs
    assert "Func Branch" in result.logs
    assert "Node Branch" in result.logs
