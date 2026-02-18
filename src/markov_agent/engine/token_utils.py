import logging
from enum import StrEnum
from typing import Any

import numpy as np
import tiktoken

logger = logging.getLogger(__name__)


class ReductionStrategy(StrEnum):
    """Strategies for reducing text to fit within a token limit."""

    GREEDY = "greedy"
    IMPORTANCE = "importance"
    LLM = "llm"


def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a string using tiktoken."""
    try:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(
            "Error counting tokens: %s. Falling back to word count approximation.", e
        )
        return len(text) // 4


def importance_sample_tokens(
    text: str,
    max_tokens: int,
    model_name: str = "gpt-3.5-turbo",
    recency_weight: float = 2.0,
) -> str:
    """Reduce text by keeping tokens with high information density and recency.

    Implements a 'Sparse Context' approach where less informative (high frequency)
    tokens are dropped, while rare tokens and recent tokens are preserved.
    """
    try:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        # 1. Frequency-based Information Density (1/count)
        # We use numpy to count frequencies within the provided context
        token_array = np.array(tokens)
        unique_tokens, counts = np.unique(token_array, return_counts=True)
        freq_map = dict(zip(unique_tokens, counts, strict=False))

        # Information score: 1.0 / count (normalized by total tokens)
        # Rare tokens get higher scores.
        base_scores = np.array([1.0 / freq_map[t] for t in tokens])

        # 2. Recency Bias
        # Linear ramp from 1.0 at the start to recency_weight at the end
        recency_scores = np.linspace(1.0, recency_weight, len(tokens))

        # Combine scores
        final_scores = base_scores * recency_scores

        # 3. Selection
        # We pick the top N indices with the highest scores
        # and sort them to maintain original relative order
        top_indices = np.argsort(final_scores)[-max_tokens:]
        top_indices.sort()

        sampled_tokens = [tokens[i] for i in top_indices]

        # Decode the sparse sequence
        # tiktoken handles individual tokens or sequences correctly
        return encoding.decode(sampled_tokens)

    except Exception as e:
        logger.warning("Importance sampling failed: %s. Falling back to greedy.", e)
        return reduce_text_to_tokens(text, max_tokens, model_name)


def reduce_text_to_tokens(
    text: str,
    max_tokens: int,
    model_name: str = "gpt-3.5-turbo",
    strategy: ReductionStrategy = ReductionStrategy.GREEDY,
    **kwargs: Any,
) -> str:
    """Reduce text to fit within max_tokens using the specified strategy."""
    if not text:
        return ""

    if strategy == ReductionStrategy.IMPORTANCE:
        recency_weight = kwargs.get("recency_weight", 2.0)
        return importance_sample_tokens(text, max_tokens, model_name, recency_weight)

    # Default: GREEDY truncation from the end with an ellipsis
    current_tokens = count_tokens(text, model_name)
    if current_tokens <= max_tokens:
        return text

    chars_per_token = len(text) / current_tokens
    target_chars = int(max_tokens * chars_per_token)

    reduced_text = text[:target_chars] + "... [TRUNCATED]"

    # Re-verify and adjust
    while (
        count_tokens(reduced_text, model_name) > max_tokens and len(reduced_text) > 20
    ):
        reduced_text = reduced_text[: int(len(reduced_text) * 0.9)] + "... [TRUNCATED]"

    return reduced_text


def reduce_list_to_tokens(
    items: list[Any],
    max_tokens: int,
    model_name: str = "gpt-3.5-turbo",
    strategy: ReductionStrategy = ReductionStrategy.GREEDY,
) -> list[Any]:
    """Reduce a list of items to fit within max_tokens.

    Prioritizes keeping the most recent items (end of the list).
    """
    if not items:
        return []

    import json

    # 1. Try with all items
    current_tokens = count_tokens(json.dumps(items), model_name)
    if current_tokens <= max_tokens:
        return items

    # 2. Sliding window (Greedy)
    # We keep removing from the start until it fits
    reduced_items = list(items)
    while len(reduced_items) > 1:
        reduced_items.pop(0)
        if count_tokens(json.dumps(reduced_items), model_name) <= max_tokens:
            return reduced_items

    # 3. If even 1 item is too large, truncate that item if it's a string/dict
    if reduced_items:
        item = reduced_items[0]
        if isinstance(item, str):
            return [reduce_text_to_tokens(item, max_tokens, model_name, strategy)]
        if isinstance(item, dict):
            return [reduce_dict_to_tokens(item, max_tokens, model_name, strategy)]

    return reduced_items


def reduce_dict_to_tokens(
    data: dict[str, Any],
    max_tokens: int,
    model_name: str = "gpt-3.5-turbo",
    strategy: ReductionStrategy = ReductionStrategy.GREEDY,
) -> dict[str, Any]:
    """Reduce a dictionary to fit within max_tokens by reducing its values."""
    if not data:
        return {}

    import json

    current_tokens = count_tokens(json.dumps(data), model_name)
    if current_tokens <= max_tokens:
        return data

    # Budget per key (naive)
    new_data = data.copy()
    keys = list(new_data.keys())
    # Sort keys by size of their JSON representation
    keys.sort(key=lambda k: len(json.dumps(new_data[k])), reverse=True)

    remaining_budget = max_tokens
    # We need to account for keys and JSON structure overhead (~2 tokens per key)
    remaining_budget -= len(keys) * 2

    for k in keys:
        val = new_data[k]
        # Target for this key is a portion of remaining budget
        # Very simple: give it a fair share or less if it fits
        target = max(10, remaining_budget // len(keys))

        if isinstance(val, str):
            new_data[k] = reduce_text_to_tokens(val, target, model_name, strategy)
        elif isinstance(val, list):
            new_data[k] = reduce_list_to_tokens(val, target, model_name, strategy)
        elif isinstance(val, dict):
            new_data[k] = reduce_dict_to_tokens(val, target, model_name, strategy)

        remaining_budget -= count_tokens(json.dumps(new_data[k]), model_name)

    return new_data
