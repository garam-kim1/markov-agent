import pytest

from markov_agent.core.state import BaseState
from markov_agent.evaluation.harness import (
    MathOptimizationHarness,
    OptimizationConstraint,
    OptimizationObjective,
    ParameterSpace,
)
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class HarnessState(BaseState):
    value: int = 0
    cost: float = 0.0


class DynamicNode(BaseNode[HarnessState]):
    multiplier: int = 1

    async def execute(self, state: HarnessState) -> HarnessState:
        # Multiplier increases value but also increases cost exponentially
        new_val = state.value + self.multiplier
        new_cost = state.cost + (self.multiplier ** 2)
        return state.update(value=new_val, cost=new_cost)


@pytest.mark.asyncio
async def test_math_optimization_harness():
    node = DynamicNode(name="dyn")
    graph = Graph(name="test_opt", nodes={"dyn": node}, edges=[], entry_point="dyn")

    dataset = [HarnessState(value=i) for i in range(3)]

    def apply_multiplier(g: Graph, val: int) -> None:
        # We know the node is of type DynamicNode
        g.nodes["dyn"].multiplier = val  # type: ignore

    param_space = ParameterSpace(
        name="multiplier",
        values=[1, 2, 3, 4],
        apply_fn=apply_multiplier,
    )

    def avg_value(results: list) -> float:
        return sum(r.final_state.value for r in results) / len(results)

    def avg_cost(results: list) -> float:
        return sum(r.final_state.cost for r in results) / len(results)

    objective = OptimizationObjective(
        name="maximize_value",
        evaluate_fn=avg_value,
        maximize=True,
    )

    constraint = OptimizationConstraint(
        name="budget",
        evaluate_fn=avg_cost,
        max_value=10.0,  # Cost must be <= 10.0
    )

    harness = MathOptimizationHarness(
        graph=graph,
        dataset=dataset,
        objective=objective,
        constraints=[constraint],
        parameter_spaces=[param_space],
        n_runs_per_case=1,
    )

    results = await harness.optimize_grid_search()

    # We test values 1, 2, 3, 4.
    # Cost for each is mult^2.
    # Mult=1 -> cost=1 (feasible)
    # Mult=2 -> cost=4 (feasible)
    # Mult=3 -> cost=9 (feasible)
    # Mult=4 -> cost=16 (infeasible, max_value=10.0)

    best = harness.get_best_feasible(results)
    assert best is not None
    assert best.parameters["multiplier"] == 3
    assert best.constraints_satisfied is True

    # Check that mult=4 is infeasible
    mult4 = next(r for r in results if r.parameters["multiplier"] == 4)
    assert mult4.constraints_satisfied is False
