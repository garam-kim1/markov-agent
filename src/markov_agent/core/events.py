import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel


class Event(BaseModel):
    """Base event for the observability bus."""

    name: str
    payload: Any


Subscriber = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Event bus for observability.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Subscriber]] = {}

    def subscribe(self, event_name: str, callback: Subscriber):
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)

    async def emit(self, event: Event):
        tasks = []
        if event.name in self._subscribers:
            tasks.extend(cb(event) for cb in self._subscribers[event.name])

        if "*" in self._subscribers:
            tasks.extend(cb(event) for cb in self._subscribers["*"])

        if tasks:
            await asyncio.gather(*tasks)


event_bus = EventBus()
