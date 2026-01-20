"""Generation layer for RAG system."""

import json
import time
from typing import TYPE_CHECKING, Union

import anthropic

from .audit import (
    GenerationEvent,
    create_actor_from_user_context,
    generate_request_id,
    get_audit_logger,
    hash_query,
)
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

if TYPE_CHECKING:
    from .models import UserContext


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


def _estimate_tokens(text: str) -> int:
    """Rough estimate of token count (approx 4 chars per token)."""
    return len(text) // 4


def generate(
    query: str,
    contexts: Union[list[RetrievalResult], list[HierarchicalRetrievalResult]],
    max_tokens: int | None = None,
    client: anthropic.Anthropic | None = None,
    user_context: "UserContext | None" = None,
    request_id: str | None = None,
) -> GenerationResult:
    """
    Generate an answer based on retrieval results.

    Args:
        query: User's question.
        contexts: List of retrieval results (flat or hierarchical).
        max_tokens: Maximum tokens for generation. Defaults to settings value.
        client: Optional pre-configured Anthropic client.
        user_context: User context for audit logging.
        request_id: Optional correlation ID for audit logging.

    Returns:
        GenerationResult with answer, citations, confidence, and policy flags.

    Raises:
        ValueError: If API key is not configured or response parsing fails.
        anthropic.APIError: If API call fails.
    """
    start_time = time.perf_counter()

    # Generate request_id if not provided
    if request_id is None:
        request_id = generate_request_id()

    # Get audit logger
    audit_logger = get_audit_logger()

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

    # Estimate context tokens
    context_token_estimate = _estimate_tokens(user_prompt) + _estimate_tokens(SYSTEM_PROMPT)

    # Set max tokens
    if max_tokens is None:
        max_tokens = settings.generation_max_tokens

    # Track refusal state
    refusal = False
    refusal_reason = None
    output_token_estimate = None
    confidence = 0.0
    citations: list[Citation] = []
    answer = ""

    try:
        # Call Anthropic API
        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            temperature=settings.generation_temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract token usage if available
        if hasattr(message, "usage"):
            output_token_estimate = message.usage.output_tokens

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

        answer = parsed.get("answer", "")

    except anthropic.APIStatusError as e:
        # Handle API errors - check for content filtering / refusal
        refusal = True
        refusal_reason = f"api_error:{e.status_code}"
        raise
    except ValueError as e:
        # Handle parsing errors
        refusal = True
        refusal_reason = f"parse_error:{str(e)[:50]}"
        raise
    finally:
        # Always log the generation event
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Check PII access in contexts
        pii_accessed = any(
            (ctx.parent_chunk.metadata.pii_flag if isinstance(ctx, HierarchicalRetrievalResult)
             else ctx.chunk.metadata.pii_flag)
            for ctx in contexts
        ) if contexts else False

        actor = create_actor_from_user_context(user_context, auth_method="cli")
        event = GenerationEvent(
            request_id=request_id,
            actor=actor,
            query_hash=hash_query(query),
            model=settings.anthropic_model,
            context_chunks=len(contexts),
            context_token_estimate=context_token_estimate,
            output_token_estimate=output_token_estimate,
            policy_flags=[f.value for f in policy_flags],
            confidence=confidence if not refusal else None,
            refusal=refusal,
            refusal_reason=refusal_reason,
            pii_accessed=pii_accessed,
            policy_decision="blocked" if refusal else "allowed",
            resource_type="generation",
            latency_ms=latency_ms,
        )
        audit_logger.log(event)

    return GenerationResult(
        answer=answer,
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
    request_id: str | None = None,
) -> GenerationResult:
    """
    Convenience function: retrieve then generate with RBAC filtering.

    Args:
        query: User's question.
        k: Number of chunks to retrieve.
        use_hierarchical: Whether to use hierarchical retrieval.
        client: Optional pre-configured Anthropic client.
        user_context: User context for RBAC filtering. None = public-only access.
        request_id: Optional correlation ID for audit logging.

    Returns:
        GenerationResult.
    """
    from .retrieve import retrieve, retrieve_hierarchical

    # Generate request_id to correlate retrieval and generation
    if request_id is None:
        request_id = generate_request_id()

    if use_hierarchical:
        contexts = retrieve_hierarchical(query, k, user_context=user_context, request_id=request_id)
    else:
        contexts = retrieve(query, k, user_context=user_context, request_id=request_id)

    return generate(query, contexts, client=client, user_context=user_context, request_id=request_id)
