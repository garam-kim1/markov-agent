import pandas as pd

from markov_agent.simulation.runner import SimulationResult


class TransitionMatrixAnalyzer:
    """Analyzes simulation results to build an empirical Transition Matrix."""

    def __init__(self, results: list[SimulationResult]):
        self.results = results
        self.transitions: dict[str, dict[str, int]] = {}
        self.nodes: set[str] = set()

    def build(self) -> pd.DataFrame:
        """Parse history traces and count i -> j transitions."""
        for result in self.results:
            if not result.final_state or not hasattr(result.final_state, "meta"):
                continue

            path = result.final_state.meta.get("path_probabilities", [])
            for entry in path:
                source = entry.get("source")
                target = entry.get("target")

                if source and target:
                    self.nodes.add(source)
                    self.nodes.add(target)

                    if source not in self.transitions:
                        self.transitions[source] = {}

                    self.transitions[source][target] = (
                        self.transitions[source].get(target, 0) + 1
                    )

        # Convert counts to probabilities
        all_nodes = sorted(self.nodes)
        matrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)

        for source, targets in self.transitions.items():
            total = sum(targets.values())
            for target, count in targets.items():
                matrix.loc[source, target] = count / total

        return matrix

    def to_dataframe(self) -> pd.DataFrame:
        """Return the transition matrix as a pandas DataFrame."""
        return self.build()
