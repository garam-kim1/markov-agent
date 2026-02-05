import asyncio
from typing import Any, Protocol

from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.simulation.evaluation.evaluator import CriteriaEvaluator, Score
from markov_agent.simulation.evaluation.models import (
    EvalCase,
    EvaluationMetrics,
    SessionResult,
    TurnResult,
)
from markov_agent.simulation.evaluation.simulator import UserSimulator


class TargetAgent(Protocol):
    """Abstract interface for any agent being evaluated."""

    async def generate_response(self, user_input: str, history: list[Any]) -> str:
        """Produce a response to the user's input."""
        ...


class EvaluationRunner:
    """Orchestrates the evaluation of an agent against a dataset of test cases."""

    def __init__(
        self,
        agent: TargetAgent,
        simulator_config: ADKConfig,
        evaluator_config: ADKConfig,
        max_turns: int = 10,
    ):
        self.agent = agent
        self.simulator_config = simulator_config
        self.evaluator_config = evaluator_config
        self.max_turns = max_turns
        self.evaluator = CriteriaEvaluator(evaluator_config)

    async def run_suite(self, dataset: list[EvalCase]) -> list[SessionResult]:
        """Runs the full evaluation suite."""
        results = []
        for case in dataset:
            result = await self.run_case(case)
            results.append(result)
        return results

    async def run_case(self, case: EvalCase) -> SessionResult:
        """Runs a single evaluation case."""
        simulator = UserSimulator(
            persona=case.user_persona,
            goal=case.user_goal,
            adk_config=self.simulator_config,
        )

        history: list[TurnResult] = []
        agent_history: list[dict] = []  # Context for the agent if needed
        success = False

        # Initial trigger: empty agent response to start user
        last_agent_response = ""

        try:
            for _ in range(self.max_turns):
                # Step A: Get input from UserSimulator
                user_input = await simulator.generate_next_turn(last_agent_response)

                if user_input == simulator.termination_signal:
                    success = True
                    break

                # Step B: Pass input to TargetAgent
                # We update agent history representation as needed by implementation
                # Here we pass a list, but the wrapper might handle it differently
                agent_response = await self.agent.generate_response(
                    user_input,
                    agent_history,
                )

                # Update our tracking history
                agent_history.append({"role": "user", "content": user_input})
                agent_history.append({"role": "model", "content": agent_response})

                # Step C: Record the turn
                turn = TurnResult(
                    user_input=user_input,
                    agent_response=agent_response,
                    tool_calls=[],  # Tool calls capture depends on agent interface
                )
                history.append(turn)

                last_agent_response = agent_response

            # Post-Processing: Evaluate Criteria
            metrics = await self._compute_metrics(case, history, success)

            return SessionResult(
                case_id=case.id,
                success=success,
                history=history,
                metrics=metrics,
            )

        except Exception as e:
            return SessionResult(
                case_id=case.id,
                success=False,
                history=history,
                error=str(e),
            )

    async def _compute_metrics(
        self,
        case: EvalCase,
        history: list[TurnResult],
        success: bool,
    ) -> EvaluationMetrics:
        """Computes aggregate metrics for the session using the CriteriaEvaluator."""
        scores = {}
        details = {}

        # 1. Goal Success (hard metric)
        scores["success_rate"] = 1.0 if success else 0.0

        if not history:
            return EvaluationMetrics(scores=scores, details=details)

        # 2. Evaluate specific criteria on the *last* turn or *all* turns?
        # Typically we might evaluate the final answer for correctness,
        # and all turns for relevance/safety.
        # For simplicity, evaluate the LAST agent response for correctness/relevance.

        last_turn = history[-1]
        context = {
            "user_goal": case.user_goal,
            "expected_outcome": case.expected_outcome,
            "conversation_history": [t.model_dump() for t in history[:-1]],
        }

        # Parallel evaluation of criteria
        criteria_list = ["Relevance", "Correctness", "Safety"]

        eval_tasks = []
        for criteria in criteria_list:
            eval_tasks.append(
                self.evaluator.evaluate_criteria(
                    response=last_turn.agent_response,
                    context=context,
                    criteria=criteria,
                ),
            )

        eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        for i, criteria in enumerate(criteria_list):
            result = eval_results[i]
            if isinstance(result, Exception):
                details[f"{criteria}_error"] = str(result)
                scores[criteria] = 0.0
            elif isinstance(result, Score):
                scores[criteria] = result.score
                details[f"{criteria}_reasoning"] = result.reasoning
            else:
                # Should not happen given logic in evaluator
                scores[criteria] = 0.0

        return EvaluationMetrics(scores=scores, details=details)
