"""RBAC (Role-Based Access Control) filtering for RAG system."""

from typing import Optional

from .models import (
    DocumentMetadata,
    HierarchicalRetrievalResult,
    RetrievalResult,
    UserContext,
)


def check_access(
    chunk_metadata: DocumentMetadata,
    user_context: Optional[UserContext],
) -> tuple[bool, Optional[str]]:
    """
    Check if user has access to a document chunk.

    Args:
        chunk_metadata: Metadata of the chunk to check.
        user_context: User context (None = no access, strict tenant isolation).

    Returns:
        Tuple of (has_access: bool, denial_reason: Optional[str]).
    """
    # Rule 0: No user context = no access (strict tenant isolation)
    if user_context is None:
        return False, "no_user_context"

    # Rule 1: Tenant isolation (must match tenant_id)
    if chunk_metadata.tenant_id != user_context.tenant_id:
        return False, f"tenant_mismatch:{chunk_metadata.tenant_id}"

    # Rule 2: Public documents within tenant are accessible
    if "public" in chunk_metadata.allowed_roles:
        return True, None

    # Rule 3: Role-based access
    if not chunk_metadata.allowed_roles:
        # No roles specified = no access (secure default)
        return False, "no_allowed_roles"

    user_roles_set = set(user_context.user_roles)
    allowed_roles_set = set(chunk_metadata.allowed_roles)

    if not (user_roles_set & allowed_roles_set):
        return False, f"role_mismatch:required={allowed_roles_set}"

    # Access granted
    return True, None


def filter_retrieval_results(
    results: list[RetrievalResult],
    user_context: Optional[UserContext],
) -> list[RetrievalResult]:
    """
    Filter flat retrieval results based on RBAC rules.

    Args:
        results: List of retrieval results.
        user_context: User context for access control.

    Returns:
        Filtered list maintaining rank order.
    """
    filtered = []
    rank = 1

    for result in results:
        has_access, _ = check_access(result.chunk.metadata, user_context)
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
) -> list[HierarchicalRetrievalResult]:
    """
    Filter hierarchical results based on parent RBAC rules.

    Args:
        results: List of hierarchical retrieval results.
        user_context: User context for access control.

    Returns:
        Filtered list maintaining rank order.
    """
    filtered = []
    rank = 1

    for result in results:
        # Check access at parent level (children inherit parent metadata)
        has_access, _ = check_access(result.parent_chunk.metadata, user_context)
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
