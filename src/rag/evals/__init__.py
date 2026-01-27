"""Evaluation framework for RAG system.

This module provides comprehensive evaluation capabilities organized by evaluation perspective:
- A. Retrieval: Document/chunk retrieval quality (MRR, NDCG, Recall, Precision)
- B. Context Quality: Retrieved context usefulness (redundancy, dispersion, diversity)
- C. Groundedness: Answer-context alignment (claim support, citation validity)
- E. Safety: Guardrail effectiveness (injection detection, leakage prevention)
- F. Pipeline: End-to-end integration (latency, RBAC, outcome validation)
"""

from src.rag.evals.models import (
    EvalPerspective,
    EvalResult,
    EvalRunSummary,
    EvalTrace,
    BaseEvalCase,
    RetrievalEvalCase,
    ContextQualityEvalCase,
    GroundednessEvalCase,
    SafetyEvalCase,
    PipelineEvalCase,
    GoldFact,
)

__all__ = [
    # Enums
    "EvalPerspective",
    # Result models
    "EvalResult",
    "EvalRunSummary",
    "EvalTrace",
    # Case models
    "BaseEvalCase",
    "RetrievalEvalCase",
    "ContextQualityEvalCase",
    "GroundednessEvalCase",
    "SafetyEvalCase",
    "PipelineEvalCase",
    "GoldFact",
]
