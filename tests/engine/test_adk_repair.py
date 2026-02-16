import pytest
from pydantic import BaseModel

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy


class SimpleSchema(BaseModel):
    name: str
    value: int


@pytest.mark.asyncio
async def test_generate_with_broken_json():
    # Mock responder that returns broken JSON
    def broken_json_responder(prompt: str) -> str:
        return '{"name": "test", "value": 123'  # Missing closing brace

    config = ADKConfig(model_name="mock-model")
    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
        mock_responder=broken_json_responder,
    )

    # Should succeed thanks to json-repair
    result = await controller.generate("test prompt", output_schema=SimpleSchema)

    assert isinstance(result, SimpleSchema)
    assert result.name == "test"
    assert result.value == 123


@pytest.mark.asyncio
async def test_generate_with_markdown_json():
    # Mock responder that returns JSON inside markdown blocks
    def markdown_json_responder(prompt: str) -> str:
        return """Some preamble
```json
{"name": "markdown", "value": 456}
```
Some postamble"""

    config = ADKConfig(model_name="mock-model")
    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
        mock_responder=markdown_json_responder,
    )

    # json-repair handles preambles/postambles well too
    result = await controller.generate("test prompt", output_schema=SimpleSchema)

    assert isinstance(result, SimpleSchema)
    assert result.name == "markdown"
    assert result.value == 456


@pytest.mark.asyncio
async def test_generate_with_single_quotes_and_trailing_commas():
    # LLMs often use single quotes or leave trailing commas
    def messy_json_responder(prompt: str) -> str:
        return "{'name': 'messy', 'value': 789,}"

    config = ADKConfig(model_name="mock-model")
    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
        mock_responder=messy_json_responder,
    )

    result = await controller.generate("test prompt", output_schema=SimpleSchema)
    assert isinstance(result, SimpleSchema)
    assert result.name == "messy"
    assert result.value == 789


class NestedSchema(BaseModel):
    items: list[SimpleSchema]
    metadata: dict[str, str]


@pytest.mark.asyncio
async def test_generate_complex_nested_messy_json():
    def complex_messy_responder(prompt: str) -> str:
        # Missing quotes on keys and some values, trailing comma
        return """{
            items: [
                {name: "item1", value: 1},
                {name: item2, value: 2},
            ],
            metadata: {
                key: val
            }
        }"""

    config = ADKConfig(model_name="mock-model")
    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
        mock_responder=complex_messy_responder,
    )

    result = await controller.generate("test prompt", output_schema=NestedSchema)
    assert isinstance(result, NestedSchema)
    assert len(result.items) == 2
    assert result.items[1].name == "item2"
    assert result.metadata["key"] == "val"


@pytest.mark.asyncio
async def test_generate_cutoff_json_repair():
    def cutoff_responder(prompt: str) -> str:
        # Common case: LLM cuts off before closing the object
        return """{
            "items": [
                {"name": "item1", "value": 1},
                {"name": "item2", "value": 2}
            ],
            "metadata": {
                "key": "val"
        """

    config = ADKConfig(model_name="mock-model")
    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
        mock_responder=cutoff_responder,
    )

    result = await controller.generate("test prompt", output_schema=NestedSchema)
    assert isinstance(result, NestedSchema)
    assert len(result.items) == 2
    assert result.metadata["key"] == "val"


@pytest.mark.asyncio
async def test_generate_with_text_interference():
    def interference_responder(prompt: str) -> str:
        return 'The result is {"name": "interfered", "value": 101} hope that helps!'

    config = ADKConfig(model_name="mock-model")
    controller = ADKController(
        config=config,
        retry_policy=RetryPolicy(max_attempts=1),
        mock_responder=interference_responder,
    )

    result = await controller.generate("test prompt", output_schema=SimpleSchema)
    assert isinstance(result, SimpleSchema)
    assert result.name == "interfered"
    assert result.value == 101
