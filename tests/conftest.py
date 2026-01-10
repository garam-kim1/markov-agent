import pytest

from markov_agent.core.state import BaseState


@pytest.fixture
def base_state():
    return BaseState()
