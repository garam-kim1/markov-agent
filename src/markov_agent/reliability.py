from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from markov_agent.topology.analysis import TopologyAnalyzer

if TYPE_CHECKING:
    from collections.abc import Callable

    from markov_agent.core.state import BaseState
    from markov_agent.topology.graph import Graph


class ReliabilityScorecard(BaseModel):
    """A comprehensive report on the agent's reliability."""

    overall_reliability: float = Field(..., description="Success rate (0.0-1.0).")
    avg_steps: float = Field(..., description="Average steps to completion.")
    entropy_score: float = Field(
        ..., description="Average entropy of decisions (lower is better)."
    )
    router_confusion: dict[str, float] = Field(
        default_factory=dict,
        description="Nodes with low confidence routing decisions (avg confidence < 0.8).",
    )
    bottlenecks: list[str] = Field(
        default_factory=list, description="Nodes that frequently lead to failure."
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Actionable improvements for the topology."
    )


class ReliabilityEngineer:
    """Orchestrates simulation and analysis to engineer reliable agents."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.analyzer = TopologyAnalyzer(graph)

    async def evaluate(
        self,
        dataset: list[BaseState],
        success_criteria: Callable[[BaseState], bool],
        n_runs: int = 5,
    ) -> ReliabilityScorecard:
        """Run a full reliability evaluation."""
        # 1. Run Monte Carlo Simulation
        results = await self.graph.simulate(
            dataset=dataset, n_runs=n_runs, success_criteria=success_criteria
        )

        total_runs = len(results)
        if total_runs == 0:
            return ReliabilityScorecard(
                overall_reliability=0.0,
                avg_steps=0.0,
                entropy_score=0.0,
                router_confusion={},
                bottlenecks=[],
                suggestions=[],
            )

        success_count = sum(1 for r in results if r.success)
        reliability = success_count / total_runs

        steps = [r.steps for r in results if r.success]
        avg_steps = statistics.mean(steps) if steps else 0.0

        # 2. Analyze Entropy & Router Confidence
        entropies = []
        router_confidences: dict[str, list[float]] = {}
        failure_nodes: dict[str, int] = {}

        for res in results:
            # Entropy
            if res.final_state and hasattr(res.final_state, "meta"):
                step_entropies = res.final_state.meta.get("step_entropy", [])
                if step_entropies:
                    entropies.extend(step_entropies)

            # Router Analysis
            if res.final_state and hasattr(res.final_state, "meta"):
                routing = res.final_state.meta.get("routing", {})
                for node_name, decision in routing.items():
                    conf = decision.get("confidence", 1.0)
                    if node_name not in router_confidences:
                        router_confidences[node_name] = []
                    router_confidences[node_name].append(conf)

            # Bottleneck Analysis (Last node before failure)
            if not res.success and res.trajectory:
                last_step = res.trajectory[-1]
                if isinstance(last_step, dict):
                    last_node = last_step.get("node")
                    if last_node:
                        failure_nodes[last_node] = failure_nodes.get(last_node, 0) + 1

        avg_entropy = statistics.mean(entropies) if entropies else 0.0

        # 3. Identify Confusion
        confused_routers = {}
        for node, confs in router_confidences.items():
            if not confs:
                continue
            avg_conf = statistics.mean(confs)
            if avg_conf < 0.8:  # Threshold for "confused"
                confused_routers[node] = avg_conf

        # 4. Identify Bottlenecks
        bottlenecks = [
            node
            for node, count in failure_nodes.items()
            if count > (total_runs * 0.1)  # >10% failure rate at this node
        ]

        # 5. Generate Suggestions
        suggestions = []
        if reliability < 0.8:
            suggestions.append(
                f"Reliability is low ({reliability:.2%}). Increase samples or refine prompts."
            )

        for node, conf in confused_routers.items():
            suggestions.append(
                f"Router '{node}' is confused (avg confidence {conf:.2f}). "
                "Clarify the route descriptions or add examples."
            )

        suggestions.extend(
            f"Node '{node}' is a bottleneck ({failure_nodes[node]} failures). "
            "Check logic or add error handling/retry."
            for node in bottlenecks
        )

        if avg_entropy > 1.0:
            suggestions.append(
                f"High overall entropy ({avg_entropy:.2f}). "
                "The agent is often uncertain. Consider constraining the topology."
            )

        return ReliabilityScorecard(
            overall_reliability=reliability,
            avg_steps=avg_steps,
            entropy_score=avg_entropy,
            router_confusion=confused_routers,
            bottlenecks=bottlenecks,
            suggestions=suggestions,
        )
