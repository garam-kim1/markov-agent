import pytest

from markov_agent.core.state import BaseState


def test_base_state_initialization():
    state = BaseState()
    assert state.history == []


def test_base_state_update():
    class MyState(BaseState):
        count: int = 0

    state = MyState(count=1)
    new_state = state.update(count=2)

    assert state.count == 1
    assert new_state.count == 2
    assert id(state) != id(new_state)


def test_base_state_history():
    state = BaseState()
    state.record_step("step1")
    state.record_step({"key": "value"})

    assert len(state.history) == 2
    assert state.history[0] == "step1"
    assert state.history[1] == {"key": "value"}


@pytest.mark.asyncio
async def test_event_bus():
    from markov_agent.core.events import Event, EventBus

    bus = EventBus()
    received = []

    async def callback(event):
        received.append(event)

    bus.subscribe("test_event", callback)
    await bus.emit(Event(name="test_event", payload="hello"))

    assert len(received) == 1
    assert received[0].payload == "hello"

    # Test wildcard
    await bus.emit(Event(name="other_event", payload="world"))
    assert len(received) == 1  # Wildcard not subscribed yet

    bus.subscribe("*", callback)
    await bus.emit(Event(name="wild_event", payload="wild"))
    assert len(received) == 2
    assert received[1].payload == "wild"
