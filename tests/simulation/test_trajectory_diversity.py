from markov_agent.simulation.metrics import (
    calculate_metrics,
    calculate_trajectory_diversity,
)
from markov_agent.simulation.runner import SimulationResult


def test_trajectory_diversity_metric():
    # Identical trajectories
    res1 = SimulationResult(
        final_state={},
        success=True,
        trajectory=[{"meta": {"node": "A"}}, {"meta": {"node": "B"}}],
    )
    res2 = SimulationResult(
        final_state={},
        success=True,
        trajectory=[{"meta": {"node": "A"}}, {"meta": {"node": "B"}}],
    )

    div = calculate_trajectory_diversity([res1, res2])
    assert div == 0.0

    # Diverse trajectories
    res3 = SimulationResult(
        final_state={},
        success=True,
        trajectory=[{"meta": {"node": "A"}}, {"meta": {"node": "C"}}],
    )
    div_high = calculate_trajectory_diversity([res1, res3])
    assert div_high > 0.0


def test_calculate_metrics_with_diversity():
    res1 = SimulationResult(
        case_id="c1", final_state={}, success=True, trajectory=[{"meta": {"node": "A"}}]
    )
    res2 = SimulationResult(
        case_id="c1", final_state={}, success=True, trajectory=[{"meta": {"node": "B"}}]
    )

    metrics = calculate_metrics([res1, res2])
    assert "trajectory_diversity" in metrics
    assert metrics["trajectory_diversity"] > 0
    assert "c1" in metrics["case_diversity"]
