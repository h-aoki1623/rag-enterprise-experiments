"""RBAC (Role-Based Access Control) filtering for RAG system."""

from typing import Optional

from .audit import (
    AccessDecisionEvent,
    AuditEventType,
    AuditLogger,
    AuditSeverity,
    DenialReason,
    create_actor_from_user_context,
)
from .models import (
    DocumentMetadata,
    HierarchicalRetrievalResult,
    RetrievalResult,
    UserContext,
)


def _map_denial_reason(reason_str: Optional[str]) -> Optional[DenialReason]:
    """Map string denial reason to DenialReason enum."""
    if reason_str is None:
        return None
    if reason_str == "no_user_context":
        return DenialReason.NO_USER_CONTEXT
    if reason_str.startswith("role_mismatch"):
        return DenialReason.ROLE_MISMATCH
    if reason_str == "no_allowed_roles":
        return DenialReason.NO_ALLOWED_ROLES
    return None


def check_access(
    chunk_metadata: DocumentMetadata,
    user_context: Optional[UserContext],
    request_id: str,
    audit_logger: AuditLogger,
    chunk_id: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Check if user has access to a document chunk with mandatory audit logging.

    Args:
        chunk_metadata: Metadata of the chunk to check.
        user_context: User context (None = no access).
        request_id: Correlation ID for request tracing.
        audit_logger: Audit logger instance (required for compliance).
        chunk_id: Optional chunk identifier.

    Returns:
        Tuple of (has_access: bool, denial_reason: Optional[str]).
    """
    decision_basis: list[str] = []

    # Perform access check with decision basis tracking
    has_access = False
    denial_reason: Optional[str] = None

    # Rule 0: No user context = no access
    if user_context is None:
        denial_reason = "no_user_context"
    # Rule 1: Public documents are accessible to all authenticated users
    elif "public" in chunk_metadata.allowed_roles:
        decision_basis.append("public_access")
        has_access = True
    # Rule 2: Role-based access
    elif not chunk_metadata.allowed_roles:
        denial_reason = "no_allowed_roles"
    else:
        user_roles_set = set(user_context.user_roles)
        allowed_roles_set = set(chunk_metadata.allowed_roles)

        if user_roles_set & allowed_roles_set:
            decision_basis.append("role_match")
            has_access = True
        else:
            denial_reason = f"role_mismatch:required={allowed_roles_set}"

    # Log the access decision (mandatory for compliance)
    actor = create_actor_from_user_context(user_context, auth_method="cli")
    event = AccessDecisionEvent(
        request_id=request_id,
        event_type=AuditEventType.ACCESS_GRANTED if has_access else AuditEventType.ACCESS_DENIED,
        severity=AuditSeverity.INFO if has_access else AuditSeverity.WARN,
        actor=actor,
        doc_id=chunk_metadata.doc_id,
        chunk_id=chunk_id,
        classification=chunk_metadata.classification.value,
        decision="granted" if has_access else "denied",
        denial_reason=_map_denial_reason(denial_reason),
        decision_basis=decision_basis,
        pii_accessed=chunk_metadata.pii_flag if has_access else False,
        policy_decision="allowed" if has_access else "blocked",
        resource_type="chunk" if chunk_id else "document",
        resource_ids=[chunk_id] if chunk_id else [chunk_metadata.doc_id],
    )
    audit_logger.log(event)

    return has_access, denial_reason


def filter_retrieval_results(
    results: list[RetrievalResult],
    user_context: Optional[UserContext],
    request_id: str,
    audit_logger: AuditLogger,
) -> list[RetrievalResult]:
    """
    Filter flat retrieval results based on RBAC rules with mandatory audit logging.

    Args:
        results: List of retrieval results.
        user_context: User context for access control.
        request_id: Correlation ID for request tracing.
        audit_logger: Audit logger instance (required for compliance).

    Returns:
        Filtered list maintaining rank order.
    """
    filtered = []
    rank = 1

    for result in results:
        has_access, _ = check_access(
            result.chunk.metadata,
            user_context,
            request_id=request_id,
            audit_logger=audit_logger,
            chunk_id=result.chunk.chunk_id,
        )

        if has_access:
            # Create new result with updated rank
            filtered_result = RetrievalResult(
                chunk=result.chunk,
                score=result.score,
                rank=rank,
            )
            filtered.append(filtered_result)
            rank += 1

    return filtered


def filter_hierarchical_results(
    results: list[HierarchicalRetrievalResult],
    user_context: Optional[UserContext],
    request_id: str,
    audit_logger: AuditLogger,
) -> list[HierarchicalRetrievalResult]:
    """
    Filter hierarchical results based on parent RBAC rules with mandatory audit logging.

    Args:
        results: List of hierarchical retrieval results.
        user_context: User context for access control.
        request_id: Correlation ID for request tracing.
        audit_logger: Audit logger instance (required for compliance).

    Returns:
        Filtered list maintaining rank order.
    """
    filtered = []
    rank = 1

    for result in results:
        # Check access at parent level (children inherit parent metadata)
        has_access, _ = check_access(
            result.parent_chunk.metadata,
            user_context,
            request_id=request_id,
            audit_logger=audit_logger,
            chunk_id=result.parent_chunk.chunk_id,
        )

        if has_access:
            # Create new result with updated rank
            filtered_result = HierarchicalRetrievalResult(
                parent_chunk=result.parent_chunk,
                matched_children=result.matched_children,
                child_scores=result.child_scores,
                aggregate_score=result.aggregate_score,
                rank=rank,
            )
            filtered.append(filtered_result)
            rank += 1

    return filtered
