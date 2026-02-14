from pydantic import BaseModel

from markov_agent.engine.selectors import MajorityVoteSelector


class SimpleModel(BaseModel):
    val: str


def test_majority_vote_simple():
    selector = MajorityVoteSelector()
    samples = ["a", "b", "a", "c", "a"]
    assert selector.select(samples) == "a"


def test_majority_vote_pydantic():
    selector = MajorityVoteSelector()
    samples = [SimpleModel(val="yes"), SimpleModel(val="no"), SimpleModel(val="yes")]
    result = selector.select(samples)
    assert result.val == "yes"


def test_majority_vote_dict():
    selector = MajorityVoteSelector()
    samples = [{"x": 1}, {"x": 2}, {"x": 1}]
    assert selector.select(samples) == {"x": 1}
