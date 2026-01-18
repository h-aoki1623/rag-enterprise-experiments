"""Generation layer for RAG system."""

import json
from typing import Union

import anthropic

from .config import settings
from .models import (
    Citation,
    Classification,
    GenerationResult,
    HierarchicalRetrievalResult,
    PolicyFlag,
    RetrievalResult,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt


def _compute_policy_flags(
    contexts: list[RetrievalResult] | list[HierarchicalRetrievalResult],
) -> list[PolicyFlag]:
    """
    Pre-compute policy flags from context metadata.

    Args:
        contexts: List of retrieval results.

    Returns:
        List of applicable policy flags.
    """
    flags: set[PolicyFlag] = set()

    if not contexts:
        flags.add(PolicyFlag.NO_CONTEXT)
        return list(flags)

    for ctx in contexts:
        if isinstance(ctx, HierarchicalRetrievalResult):
            metadata = ctx.parent_chunk.metadata
        else:
            metadata = ctx.chunk.metadata

        # Check for PII
        if metadata.pii_flag:
            flags.add(PolicyFlag.PII_REFERENCED)

        # Check for confidential classification
        if metadata.classification == Classification.CONFIDENTIAL:
            flags.add(PolicyFlag.CONFIDENTIAL)

    return list(flags)


def _parse_llm_response(response_text: str) -> dict:
    """
    Parse LLM response as JSON.

    Args:
        response_text: Raw response text from LLM.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If response is not valid JSON.
    """
    # Try to extract JSON from response
    text = response_text.strip()

    # Handle markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e


def generate(
    query: str,
    contexts: Union[list[RetrievalResult], list[HierarchicalRetrievalResult]],
    max_tokens: int | None = None,
    client: anthropic.Anthropic | None = None,
) -> GenerationResult:
    """
    Generate an answer based on retrieval results.

    Args:
        query: User's question.
        contexts: List of retrieval results (flat or hierarchical).
        max_tokens: Maximum tokens for generation. Defaults to settings value.
        client: Optional pre-configured Anthropic client.

    Returns:
        GenerationResult with answer, citations, confidence, and policy flags.

    Raises:
        ValueError: If API key is not configured or response parsing fails.
        anthropic.APIError: If API call fails.
    """
    # Validate API key
    if not settings.anthropic_api_key and client is None:
        raise ValueError(
            "Anthropic API key not configured. Set ANTHROPIC_API_KEY in .env file."
        )

    # Initialize client if not provided
    if client is None:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Pre-compute policy flags from context
    policy_flags = _compute_policy_flags(contexts)

    # Build prompts
    user_prompt = build_user_prompt(query, contexts)

    # Set max tokens
    if max_tokens is None:
        max_tokens = settings.generation_max_tokens

    # Call Anthropic API
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        temperature=settings.generation_temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract response text
    response_text = message.content[0].text

    # Parse JSON response
    parsed = _parse_llm_response(response_text)

    # Build citations
    citations = [
        Citation(
            doc_id=c.get("doc_id", ""),
            chunk_id=c.get("chunk_id", ""),
            text_snippet=c.get("text_snippet", ""),
        )
        for c in parsed.get("citations", [])
    ]

    # Extract confidence
    confidence = parsed.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    # Add UNCERTAIN flag if confidence is low
    if confidence < 0.4 and PolicyFlag.UNCERTAIN not in policy_flags:
        policy_flags.append(PolicyFlag.UNCERTAIN)

    # Add NO_CONTEXT flag if no citations
    if not citations and PolicyFlag.NO_CONTEXT not in policy_flags:
        policy_flags.append(PolicyFlag.NO_CONTEXT)

    return GenerationResult(
        answer=parsed.get("answer", ""),
        citations=citations,
        confidence=confidence,
        policy_flags=policy_flags,
        raw_context_used=bool(contexts),
    )


def generate_with_retrieval(
    query: str,
    k: int | None = None,
    use_hierarchical: bool = False,
    client: anthropic.Anthropic | None = None,
    user_context: "UserContext | None" = None,
) -> GenerationResult:
    """
    Convenience function: retrieve then generate with RBAC filtering.

    Args:
        query: User's question.
        k: Number of chunks to retrieve.
        use_hierarchical: Whether to use hierarchical retrieval.
        client: Optional pre-configured Anthropic client.
        user_context: User context for RBAC filtering. None = public-only access.

    Returns:
        GenerationResult.
    """
    from .retrieve import retrieve, retrieve_hierarchical

    if use_hierarchical:
        contexts = retrieve_hierarchical(query, k, user_context=user_context)
    else:
        contexts = retrieve(query, k, user_context=user_context)

    return generate(query, contexts, client=client)
