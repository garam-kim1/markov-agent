from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from markov_agent.core.state import BaseState

StateT = TypeVar("StateT", bound=BaseState)


class SchemaRegistry:
    """The 'Cognitive Kernel' State Schema Registry.

    Prevents downstream corruption by enforcing strict data contracts between nodes.
    """

    _schemas: ClassVar[dict[str, type[BaseModel]]] = {}

    @classmethod
    def register(cls, name: str, schema: type[BaseModel]) -> None:
        """Register a formal data contract for a specific state or node output."""
        cls._schemas[name] = schema

    @classmethod
    def validate(cls, name: str, data: Any) -> Any:
        """Validate data against a registered contract.

        Raises ValueError if the contract is breached.
        """
        if name not in cls._schemas:
            return data  # No contract enforced for this key

        schema = cls._schemas[name]
        if isinstance(data, dict):
            return schema.model_validate(data)
        if isinstance(data, schema):
            return data

        msg = f"Data contract breach for '{name}'. Expected {schema.__name__}."
        raise ValueError(msg)


def enforce_contract(
    contract_name: str,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Enforce a data contract on a node's execute method."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(self: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
            result = await func(self, state, *args, **kwargs)
            # Validate result before returning
            SchemaRegistry.validate(contract_name, result)
            return result

        return wrapper

    return decorator
