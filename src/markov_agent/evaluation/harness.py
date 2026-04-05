import itertools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from markov_agent.core.state import BaseState
from markov_agent.simulation.runner import MonteCarloRunner, SimulationResult
from markov_agent.topology.graph import Graph

StateT = TypeVar("StateT", bound=BaseState)
logger = logging.getLogger(__name__)


class OptimizationConstraint(BaseModel):
    """A mathematical constraint for the evaluation harness: g(x) <= max_value or g(x) >= min_value."""

    name: str
    evaluate_fn: Callable[[list[SimulationResult]], float]
    max_value: float | None = None
    min_value: float | None = None

    def check(self, results: list[SimulationResult]) -> tuple[bool, float]:
        val = self.evaluate_fn(results)
        if self.max_value is not None and val > self.max_value:
            return False, val
        if self.min_value is not None and val < self.min_value:
            return False, val
        return True, val


class OptimizationObjective(BaseModel):
    """The objective function to maximize or minimize: f(x)."""

    name: str
    evaluate_fn: Callable[[list[SimulationResult]], float]
    maximize: bool = True

    def score(self, results: list[SimulationResult]) -> float:
        val = self.evaluate_fn(results)
        return val if self.maximize else -val


class ParameterSpace(BaseModel):
    """Defines the discrete search space for an agent hyperparameter (e.g., node k-sampling)."""

    name: str
    values: list[Any]
    apply_fn: Callable[[Graph, Any], None]


class EvaluationResult(BaseModel):
    """The result of evaluating a single configuration."""

    parameters: dict[str, Any]
    objective_score: float
    raw_objective_value: float
    constraint_values: dict[str, float]
    constraints_satisfied: bool
    simulation_results: list[SimulationResult]


class MathOptimizationHarness[StateT: BaseState](BaseModel):
    """An Evaluation Harness that acts like Mathematical Optimization.

    Formulates agent configuration tuning as:
    max_x f(x)
    s.t.  g_i(x) <= c_i
    where x are the agent parameters defined in parameter_spaces.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: Graph
    dataset: list[StateT]
    objective: OptimizationObjective
    constraints: list[OptimizationConstraint] = Field(default_factory=list)
    parameter_spaces: list[ParameterSpace] = Field(default_factory=list)

    n_runs_per_case: int = 1
    max_concurrency: int = 5
    success_criteria: Callable[[StateT], bool] = Field(default=lambda _: True)

    async def _evaluate_config(self, params: dict[str, Any]) -> EvaluationResult:
        """Evaluate a single point in the parameter space."""
        # 1. Apply parameters
        for p_space in self.parameter_spaces:
            p_space.apply_fn(self.graph, params[p_space.name])

        # 2. Run simulation
        runner = MonteCarloRunner(
            graph=self.graph,
            dataset=self.dataset,
            n_runs=self.n_runs_per_case,
            success_criteria=self.success_criteria,
            max_concurrency=self.max_concurrency,
        )
        sim_results = await runner.run_simulation()

        # 3. Calculate objective
        raw_obj_val = self.objective.evaluate_fn(sim_results)
        obj_score = self.objective.score(sim_results)

        # 4. Check constraints
        constraint_vals = {}
        all_satisfied = True
        for constraint in self.constraints:
            satisfied, val = constraint.check(sim_results)
            constraint_vals[constraint.name] = val
            if not satisfied:
                all_satisfied = False

        return EvaluationResult(
            parameters=params,
            objective_score=obj_score,
            raw_objective_value=raw_obj_val,
            constraint_values=constraint_vals,
            constraints_satisfied=all_satisfied,
            simulation_results=sim_results,
        )

    async def optimize_grid_search(self) -> list[EvaluationResult]:
        """Perform a grid search over the parameter space, tracking the Lagrangian formulation implicitly."""
        if not self.parameter_spaces:
            # Just evaluate the current graph once
            res = await self._evaluate_config({})
            return [res]

        # Generate all combinations of parameters
        keys = [p.name for p in self.parameter_spaces]
        value_lists = [p.values for p in self.parameter_spaces]
        combinations = list(itertools.product(*value_lists))

        results: list[EvaluationResult] = []
        for combo in combinations:
            params = dict(zip(keys, combo, strict=True))
            logger.info("Evaluating parameter set: %s", params)
            res = await self._evaluate_config(params)
            results.append(res)

            status = "Feasible" if res.constraints_satisfied else "Infeasible"
            logger.info(
                "Result [%s]: Objective=%s, Constraints=%s",
                status,
                res.raw_objective_value,
                res.constraint_values,
            )

        # Sort results: first by constraints_satisfied (True > False), then by objective_score (descending)
        results.sort(
            key=lambda r: (r.constraints_satisfied, r.objective_score), reverse=True
        )
        return results

    def get_best_feasible(
        self, results: list[EvaluationResult]
    ) -> EvaluationResult | None:
        """Return the feasible result that maximizes the objective."""
        feasible = [r for r in results if r.constraints_satisfied]
        if not feasible:
            return None
        return max(feasible, key=lambda r: r.objective_score)
