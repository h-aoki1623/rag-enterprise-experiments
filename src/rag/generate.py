"""Generation layer for RAG system.

This module provides answer generation with integrated guardrails:

Pipeline flow:
    User Input → [Input Guardrail] → Retrieve → RBAC filter → LLM call → [Output Guardrail] → Answer

The generate() function orchestrates the full RAG pipeline including:
- Input Guardrail: Checks queries for injection attacks before retrieval
- Retrieval: Vector search with RBAC filtering
- LLM call: Answer generation with Claude
- Output Guardrail: Checks responses for data leakage before returning
"""

import hashlib
import json
import time
from typing import TYPE_CHECKING, Union

import anthropic

from .audit import (
    AuditSeverity,
    GenerationEvent,
    GuardrailAuditEvent,
    create_actor_from_user_context,
    generate_request_id,
    get_audit_logger,
    hash_query,
)
from .config import settings
from .guardrails import GuardrailAction, GuardrailResult, InputGuardrail, OutputGuardrail
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


def _get_max_classification(
    contexts: list[RetrievalResult] | list[HierarchicalRetrievalResult],
) -> Classification:
    """Get the highest classification level from contexts.

    Args:
        contexts: List of retrieval results.

    Returns:
        Highest Classification level (confidential > internal > public).
    """
    if not contexts:
        return Classification.PUBLIC

    classification_order = {
        Classification.PUBLIC: 0,
        Classification.INTERNAL: 1,
        Classification.CONFIDENTIAL: 2,
    }

    max_classification = Classification.PUBLIC

    for ctx in contexts:
        if isinstance(ctx, HierarchicalRetrievalResult):
            classification = ctx.parent_chunk.metadata.classification
        else:
            classification = ctx.chunk.metadata.classification

        if classification_order[classification] > classification_order[max_classification]:
            max_classification = classification

    return max_classification


def _log_guardrail_event(
    request_id: str,
    result: GuardrailResult,
    user_context: "UserContext | None",
    query: str,
    contexts: list[RetrievalResult] | list[HierarchicalRetrievalResult] | None = None,
) -> None:
    """Log a guardrail audit event.

    Args:
        request_id: Correlation ID.
        result: GuardrailResult from check.
        user_context: User context for actor info.
        query: User query (will be hashed).
        contexts: Retrieval contexts (for doc fingerprint). None for input guardrail.
    """
    if not settings.guardrails.log_guardrail_events:
        return

    contexts = contexts or []

    audit_logger = get_audit_logger()
    actor = create_actor_from_user_context(user_context, auth_method="cli")

    # Create doc fingerprint from doc_ids
    doc_ids = []
    classifications = []
    for ctx in contexts:
        if isinstance(ctx, HierarchicalRetrievalResult):
            doc_ids.append(ctx.parent_chunk.doc_id)
            classifications.append(ctx.parent_chunk.metadata.classification.value)
        else:
            doc_ids.append(ctx.chunk.doc_id)
            classifications.append(ctx.chunk.metadata.classification.value)

    doc_fingerprint = hashlib.sha256(":".join(sorted(doc_ids)).encode()).hexdigest()[:16] if doc_ids else None

    # Determine severity based on action
    severity = AuditSeverity.INFO
    if result.action == GuardrailAction.WARN:
        severity = AuditSeverity.WARN
    elif result.action in (GuardrailAction.REDACT, GuardrailAction.BLOCK):
        severity = AuditSeverity.ERROR

    # Extract detection stats from score_breakdown
    score_breakdown = result.score_breakdown
    matched_pattern_count = 0
    pii_count = 0
    verbatim = None

    if result.guardrail_type == "input":
        # Sum up pattern-related scores as indicator
        if score_breakdown.get("pattern_score", 0) > 0:
            matched_pattern_count += 1
        if score_breakdown.get("structural_score", 0) > 0:
            matched_pattern_count += 1
        if score_breakdown.get("delimiter_score", 0) > 0:
            matched_pattern_count += 1
    else:  # output
        pii_count = score_breakdown.get("pii_detected_count", 0)
        verbatim = score_breakdown.get("verbatim_ratio")

    event = GuardrailAuditEvent(
        request_id=request_id,
        actor=actor,
        severity=severity,
        guardrail_type=result.guardrail_type,
        threat_type=result.threat_type,
        threat_score=result.threat_score,
        action_taken=result.action.value,
        score_breakdown=score_breakdown,
        input_hash=hash_query(query),
        input_length=len(query),
        doc_set_fingerprint=doc_fingerprint,
        doc_count=len(contexts),
        classifications_involved=list(set(classifications)),
        model_id=settings.anthropic_model,
        matched_pattern_count=matched_pattern_count,
        pii_detected_count=pii_count,
        verbatim_ratio=verbatim,
    )
    audit_logger.log(event)


def _call_llm(
    client: anthropic.Anthropic,
    user_prompt: str,
    max_tokens: int,
) -> tuple[str, int | None]:
    """
    Internal helper: Pure LLM call without guardrails.

    Args:
        client: Anthropic client.
        user_prompt: Formatted user prompt.
        max_tokens: Maximum tokens for generation.

    Returns:
        Tuple of (response_text, output_token_count).

    Raises:
        anthropic.APIError: If API call fails.
    """
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        temperature=settings.generation_temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract token usage if available
    output_tokens = None
    if hasattr(message, "usage"):
        output_tokens = message.usage.output_tokens

    # Extract response text
    response_text = message.content[0].text

    return response_text, output_tokens


def generate(
    query: str,
    k: int | None = None,
    use_hierarchical: bool = False,
    max_tokens: int | None = None,
    client: anthropic.Anthropic | None = None,
    user_context: "UserContext | None" = None,
    request_id: str | None = None,
) -> GenerationResult:
    """
    Full RAG pipeline: Input Guardrail → Retrieve → LLM → Output Guardrail.

    This function orchestrates the complete RAG pipeline:
    1. Input Guardrail: Check query for injection attacks (BEFORE retrieval)
    2. Retrieve: Vector search with RBAC filtering
    3. LLM Call: Generate answer with citations
    4. Output Guardrail: Check response for data leakage

    Args:
        query: User's question.
        k: Number of chunks to retrieve. Defaults to settings value.
        use_hierarchical: Whether to use hierarchical retrieval.
        max_tokens: Maximum tokens for generation. Defaults to settings value.
        client: Optional pre-configured Anthropic client.
        user_context: User context for RBAC filtering. None = public-only access.
        request_id: Optional correlation ID for audit logging.

    Returns:
        GenerationResult with answer, citations, confidence, and policy flags.

    Raises:
        ValueError: If API key is not configured or response parsing fails.
        anthropic.APIError: If API call fails.
    """
    from .retrieve import retrieve, retrieve_hierarchical

    start_time = time.perf_counter()

    # Generate request_id if not provided
    if request_id is None:
        request_id = generate_request_id()

    # Get audit logger
    audit_logger = get_audit_logger()

    # Initialize guardrails
    input_guardrail = InputGuardrail(settings.guardrails)
    output_guardrail = OutputGuardrail(settings.guardrails)

    # ============================================
    # STEP 1: INPUT GUARDRAIL - Check query BEFORE retrieval
    # ============================================
    # Note: Input guardrail uses a fixed threshold regardless of document
    # classification. This is a security design decision - if a user account
    # is compromised, varying thresholds by role would be exploitable.
    input_result = input_guardrail.check(query)

    if input_result.action == GuardrailAction.BLOCK:
        _log_guardrail_event(request_id, input_result, user_context, query, None)
        return GenerationResult(
            answer="Your request could not be processed due to security policy.",
            citations=[],
            confidence=0.0,
            policy_flags=[PolicyFlag.GUARDRAIL_BLOCKED],
            raw_context_used=False,
        )

    # Initialize policy flags early (before WARN check)
    policy_flags: list[PolicyFlag] = []

    if input_result.action == GuardrailAction.WARN:
        policy_flags.append(PolicyFlag.GUARDRAIL_WARNED)
        _log_guardrail_event(request_id, input_result, user_context, query, None)

    # ============================================
    # STEP 2: RETRIEVE - Vector search with RBAC filtering
    # ============================================
    if use_hierarchical:
        contexts: Union[list[RetrievalResult], list[HierarchicalRetrievalResult]] = (
            retrieve_hierarchical(query, k, user_context=user_context, request_id=request_id)
        )
    else:
        contexts = retrieve(query, k, user_context=user_context, request_id=request_id)

    # Determine classification for output guardrail (highest sensitivity in contexts)
    max_classification = _get_max_classification(contexts)

    # Add policy flags from context metadata
    policy_flags.extend(_compute_policy_flags(contexts))

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

    # Validate API key
    if not settings.anthropic_api_key and client is None:
        raise ValueError(
            "Anthropic API key not configured. Set ANTHROPIC_API_KEY in .env file."
        )

    # Initialize client if not provided
    if client is None:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    try:
        # ============================================
        # STEP 3: LLM CALL - Generate answer
        # ============================================
        response_text, output_token_estimate = _call_llm(client, user_prompt, max_tokens)

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

        # ============================================
        # STEP 4: OUTPUT GUARDRAIL - Check response for leakage
        # ============================================
        context_texts = [
            (
                ctx.parent_chunk.text
                if isinstance(ctx, HierarchicalRetrievalResult)
                else ctx.chunk.text
            )
            for ctx in contexts
        ]
        doc_metadata = [
            {
                "doc_id": (
                    ctx.parent_chunk.doc_id
                    if isinstance(ctx, HierarchicalRetrievalResult)
                    else ctx.chunk.doc_id
                ),
                "classification": (
                    ctx.parent_chunk.metadata.classification.value
                    if isinstance(ctx, HierarchicalRetrievalResult)
                    else ctx.chunk.metadata.classification.value
                ),
            }
            for ctx in contexts
        ]

        output_result = output_guardrail.check(
            answer, context_texts, doc_metadata, max_classification
        )

        if output_result.action == GuardrailAction.BLOCK:
            _log_guardrail_event(request_id, output_result, user_context, query, contexts)
            return GenerationResult(
                answer="The response was blocked due to security policy.",
                citations=[],
                confidence=0.0,
                policy_flags=[PolicyFlag.GUARDRAIL_BLOCKED],
                raw_context_used=bool(contexts),
            )

        # Sanitize lane: redact if sanitize_needed (PII/metadata detected)
        if output_result.details.get("sanitize_needed", False):
            answer = output_guardrail.redact(answer, output_result, doc_metadata)
            policy_flags.append(PolicyFlag.GUARDRAIL_REDACTED)
            _log_guardrail_event(request_id, output_result, user_context, query, contexts)

        if output_result.action == GuardrailAction.WARN:
            policy_flags.append(PolicyFlag.GUARDRAIL_WARNED)
            _log_guardrail_event(request_id, output_result, user_context, query, contexts)

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
        pii_accessed = (
            any(
                (
                    ctx.parent_chunk.metadata.pii_flag
                    if isinstance(ctx, HierarchicalRetrievalResult)
                    else ctx.chunk.metadata.pii_flag
                )
                for ctx in contexts
            )
            if contexts
            else False
        )

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


# Legacy alias for backward compatibility
def generate_with_retrieval(
    query: str,
    k: int | None = None,
    use_hierarchical: bool = False,
    client: anthropic.Anthropic | None = None,
    user_context: "UserContext | None" = None,
    request_id: str | None = None,
) -> GenerationResult:
    """
    Legacy alias for generate().

    .. deprecated::
        Use generate() instead. This function will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "generate_with_retrieval() is deprecated. Use generate() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate(
        query=query,
        k=k,
        use_hierarchical=use_hierarchical,
        client=client,
        user_context=user_context,
        request_id=request_id,
    )
