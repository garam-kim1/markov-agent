from unittest.mock import MagicMock

import pytest

from examples.callbacks.safety_callbacks_demo import (
    PolicyCheckCallback,
)
from markov_agent.engine.callbacks import (
    CallbackContext,
    CallbackError,
    PIIRedactionCallback,
)


class MockPart:
    def __init__(self, text):
        self.text = text


class MockContent:
    def __init__(self, parts):
        self.parts = parts


class MockRequest:
    def __init__(self, contents):
        self.contents = contents


def test_pii_scrub_callback():
    callback = PIIRedactionCallback()
    mock_context = MagicMock(spec=CallbackContext)

    # Test case with email
    part = MockPart(text="Contact me at john.doe@example.com for details.")
    content = MockContent(parts=[part])
    req = MockRequest(contents=[content])

    callback(context=mock_context, model_request=req)

    assert part.text == "Contact me at [EMAIL_REDACTED] for details."

    # Test case without email
    part_clean = MockPart(text="Hello world")
    content_clean = MockContent(parts=[part_clean])
    req_clean = MockRequest(contents=[content_clean])

    callback(context=mock_context, model_request=req_clean)
    assert part_clean.text == "Hello world"


def test_policy_check_callback():
    callback = PolicyCheckCallback()
    mock_context = MagicMock(spec=CallbackContext)

    # Mock response object structure
    # Case 1: Safe content
    safe_response = MagicMock()
    safe_response.candidates = [MagicMock()]
    safe_response.candidates[0].content.parts = [
        MockPart(text="This is a safe response.")
    ]

    callback(context=mock_context, model_response=safe_response)  # Should not raise

    # Case 2: Unsafe content
    unsafe_response = MagicMock()
    unsafe_response.candidates = [MagicMock()]
    unsafe_response.candidates[0].content.parts = [
        MockPart(text="Here is the unspeakable_secret info.")
    ]

    with pytest.raises(CallbackError) as excinfo:
        callback(context=mock_context, model_response=unsafe_response)

    assert "forbidden term" in str(excinfo.value)
