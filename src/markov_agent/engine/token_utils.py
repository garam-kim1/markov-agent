import logging

import tiktoken

logger = logging.getLogger(__name__)


def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a string using tiktoken.

    Uses gpt-3.5-turbo as a default proxy if the model is not recognized.
    """
    try:
        # Try to get the encoding for the specific model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base which is used by most recent models
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(
            "Error counting tokens: %s. Falling back to word count approximation.", e
        )
        # Fallback to a rough approximation: 1 token ~= 4 characters or 0.75 words
        return len(text) // 4


def reduce_text_to_tokens(
    text: str, max_tokens: int, model_name: str = "gpt-3.5-turbo"
) -> str:
    """Greedily reduce text to fit within max_tokens by truncating from the middle or end.

    For now, we truncate from the end with an ellipsis.
    """
    if not text:
        return ""

    current_tokens = count_tokens(text, model_name)
    if current_tokens <= max_tokens:
        return text

    # Simple binary search or heuristic truncation
    # Start with a safe ratio
    chars_per_token = len(text) / current_tokens
    target_chars = int(max_tokens * chars_per_token)

    reduced_text = text[:target_chars] + "... [TRUNCATED]"

    # Re-verify and adjust
    while (
        count_tokens(reduced_text, model_name) > max_tokens and len(reduced_text) > 20
    ):
        reduced_text = reduced_text[: int(len(reduced_text) * 0.9)] + "... [TRUNCATED]"

    return reduced_text
