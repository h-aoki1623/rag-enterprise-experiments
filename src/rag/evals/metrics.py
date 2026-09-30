"""Metric calculation utilities for evaluation framework.

This module provides functions for computing various evaluation metrics:
- Retrieval: MRR, NDCG@k, Recall@k, Precision@k
- Context Quality: Redundancy ratio, fact dispersion, unique token ratio
- Groundedness: Claim support rate, numeric fabrication detection
- Safety: ROC curve, AUC, TPR@FPR
- Shared: Percentile calculations
"""

import math
import re
from collections import Counter
from typing import Optional

import numpy as np


# =============================================================================
# Retrieval Metrics
# =============================================================================


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs
        relevant_ids: Set of relevant document/chunk IDs

    Returns:
        Reciprocal rank (1/rank of first relevant, or 0 if none found)
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain).

    Args:
        retrieved_ids: Ordered list of retrieved IDs
        relevance_grades: Dict mapping doc_id -> relevance grade (0-3)
        k: Cutoff position

    Returns:
        NDCG@k score (0.0-1.0)
    """
    if not relevance_grades:
        return 0.0

    # DCG calculation
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = relevance_grades.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)

    # Ideal DCG (sorted by relevance)
    ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Calculate Recall@k.

    Args:
        retrieved_ids: Ordered list of retrieved IDs
        relevant_ids: Set of relevant IDs
        k: Cutoff position

    Returns:
        Recall@k score (0.0-1.0)
    """
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Calculate Precision@k.

    Args:
        retrieved_ids: Ordered list of retrieved IDs
        relevant_ids: Set of relevant IDs
        k: Cutoff position

    Returns:
        Precision@k score (0.0-1.0)
    """
    retrieved = retrieved_ids[:k]
    if not retrieved:
        return 0.0
    relevant_count = sum(1 for doc_id in retrieved if doc_id in relevant_ids)
    return relevant_count / len(retrieved)


def f1_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Calculate F1@k.

    Args:
        retrieved_ids: Ordered list of retrieved IDs
        relevant_ids: Set of relevant IDs
        k: Cutoff position

    Returns:
        F1@k score (0.0-1.0)
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# =============================================================================
# Context Quality Metrics
# =============================================================================


def calc_ngram_overlap(text1: str, text2: str, n: int = 5) -> float:
    """Calculate n-gram overlap ratio between two texts.

    Args:
        text1: First text
        text2: Second text
        n: N-gram size

    Returns:
        Overlap ratio (0.0-1.0)
    """
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    if len(words1) < n or len(words2) < n:
        return 0.0

    ngrams1 = set(tuple(words1[i : i + n]) for i in range(len(words1) - n + 1))
    ngrams2 = set(tuple(words2[i : i + n]) for i in range(len(words2) - n + 1))

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2

    return len(intersection) / len(union) if union else 0.0


def redundancy_ratio(chunks: list[str], threshold: float = 0.3, n: int = 5) -> float:
    """Calculate redundancy ratio among chunks using n-gram overlap.

    Args:
        chunks: List of chunk texts
        threshold: Overlap threshold to consider as duplicate
        n: N-gram size

    Returns:
        Ratio of redundant chunk pairs (0.0-1.0)
    """
    if len(chunks) < 2:
        return 0.0

    redundant_pairs = 0
    total_pairs = 0

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            total_pairs += 1
            overlap = calc_ngram_overlap(chunks[i], chunks[j], n)
            if overlap > threshold:
                redundant_pairs += 1

    return redundant_pairs / total_pairs if total_pairs > 0 else 0.0


def redundancy_ratio_tfidf(chunks: list[str], threshold: float = 0.7) -> float:
    """Calculate redundancy ratio using TF-IDF cosine similarity.

    Detects both lexical (verbatim) and semantic (paraphrased) duplicates.

    Args:
        chunks: List of chunk texts
        threshold: Cosine similarity threshold to consider as duplicate

    Returns:
        Ratio of redundant chunk pairs (0.0-1.0)
    """
    if len(chunks) < 2:
        return 0.0

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    sim_matrix = cosine_similarity(tfidf_matrix)

    redundant_pairs = 0
    total_pairs = 0

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            total_pairs += 1
            if sim_matrix[i, j] > threshold:
                redundant_pairs += 1

    return redundant_pairs / total_pairs if total_pairs > 0 else 0.0


def fact_dispersion(
    chunks: list[str],
    gold_fact: str,
    aliases: Optional[list[str]] = None,
    fuzzy_threshold: float = 0.7,
) -> int:
    """Count how many chunks contain a gold fact using fuzzy matching.

    First tries exact substring match (original behavior).
    Falls back to word overlap matching for paraphrased facts.

    Args:
        chunks: List of chunk texts
        gold_fact: The canonical fact text
        aliases: Alternative expressions of the same fact
        fuzzy_threshold: Minimum word overlap ratio for fuzzy match

    Returns:
        Number of chunks containing the fact
    """
    all_variants = [gold_fact.lower()]
    if aliases:
        all_variants.extend(a.lower() for a in aliases)

    # Extract content words from fact for fuzzy matching
    fact_words = set(gold_fact.lower().split()) - STOPWORDS

    count = 0
    for chunk in chunks:
        chunk_lower = chunk.lower()

        # Try exact match first
        if any(variant in chunk_lower for variant in all_variants):
            count += 1
            continue

        # Fuzzy match: check word overlap
        if fact_words:
            chunk_words = set(chunk_lower.split())
            overlap = len(fact_words & chunk_words) / len(fact_words)
            if overlap >= fuzzy_threshold:
                count += 1

    return count


def unique_token_ratio(chunks: list[str]) -> float:
    """Calculate the ratio of unique tokens to total tokens.

    Args:
        chunks: List of chunk texts

    Returns:
        Unique token ratio (0.0-1.0)
    """
    all_text = " ".join(chunks)
    tokens = all_text.lower().split()

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


# =============================================================================
# Groundedness Metrics
# =============================================================================

# Common English stopwords for filtering
STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "it", "its", "this", "that", "these",
    "those", "and", "or", "but", "if", "then", "than", "so", "as", "no",
    "not", "only", "own", "same", "too", "very", "can", "just", "into",
    "out", "up", "down", "about", "over", "under", "again", "further",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "any", "both", "our",
    "your", "their", "his", "her", "my", "i", "you", "he", "she", "we",
    "they", "who", "which", "what", "whom", "whose", "also", "however",
})


def normalize_number(num_str: str) -> str:
    """Normalize a number string for comparison.

    Args:
        num_str: Number string (e.g., "1,000", "$500", "10%")

    Returns:
        Normalized number string
    """
    # Remove currency symbols and commas
    normalized = re.sub(r"[$,]", "", num_str)
    # Standardize whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def extract_key_terms(text: str, number_weight: float = 2.0) -> list[tuple[str, float]]:
    """Extract key terms from text with weights.

    Numbers get higher weight as they are strong anchors for claim verification.

    Args:
        text: Input text
        number_weight: Weight multiplier for numeric terms

    Returns:
        List of (normalized_term, weight) tuples
    """
    terms = []

    # Extract numbers: 2024, 1.2, 30%, $500, 1,000
    number_pattern = r"\b\d+(?:[.,]\d+)*%?\b|\$[\d,]+(?:\.\d+)?"
    numbers = re.findall(number_pattern, text)
    for num in numbers:
        normalized = normalize_number(num)
        if normalized and len(normalized) > 0:
            terms.append((normalized, number_weight))

    # Extract proper nouns (capitalized words, excluding sentence starts)
    # This is a heuristic - proper nouns tend to be capitalized
    proper_noun_pattern = r"(?<![.!?]\s)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    proper_nouns = re.findall(proper_noun_pattern, text)
    for noun in proper_nouns:
        normalized = noun.lower().strip()
        if normalized and len(normalized) > 2 and normalized not in STOPWORDS:
            terms.append((normalized, 1.0))

    # Extract technical terms (all caps, acronyms)
    tech_pattern = r"\b[A-Z]{2,}(?:\d+)?\b"
    tech_terms = re.findall(tech_pattern, text)
    for term in tech_terms:
        normalized = term.lower()
        if normalized and len(normalized) > 1:
            terms.append((normalized, 1.0))

    return terms

def extract_claims(answer: str) -> list[tuple[str, str]]:
    """Extract claims from an answer with their type classification.

    Args:
        answer: The generated answer text

    Returns:
        List of (claim_text, claim_type) tuples
    """
    # Split into sentences
    sentences = re.split(r"[.!?]+", answer)
    claims = []

    # Patterns for claim type classification
    inference_patterns = [
        r"\b(may|might|could|possibly|perhaps|likely|probably)\b",
        r"\b(seems?|appears?|suggests?)\b",
    ]
    general_patterns = [
        r"\b(generally|typically|usually|often|commonly)\b",
        r"\b(in general|as a rule)\b",
    ]
    question_patterns = [r"\?$", r"^\s*(what|who|where|when|why|how)\b"]

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        # Skip questions
        if any(re.search(p, sentence, re.IGNORECASE) for p in question_patterns):
            continue

        # Classify claim type
        if any(re.search(p, sentence, re.IGNORECASE) for p in inference_patterns):
            claim_type = "inference"
        elif any(re.search(p, sentence, re.IGNORECASE) for p in general_patterns):
            claim_type = "general"
        else:
            claim_type = "assertion"

        claims.append((sentence, claim_type))

    return claims

def extract_numbers(text: str) -> list[str]:
    """Extract normalized numbers from text.

    Args:
        text: Input text

    Returns:
        List of normalized number strings
    """
    # Patterns for numbers
    patterns = [
        r"\d{1,3}(?:,\d{3})*(?:\.\d+)?",  # 1,000 or 1,000.50
        r"\d+(?:\.\d+)?%?",  # 100 or 100.5 or 100%
        r"\$[\d,]+(?:\.\d+)?",  # $1,000
        r"[\d,]+(?:\.\d+)?\s*(?:days?|years?|months?|hours?|minutes?)",  # 15 days
    ]

    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Normalize: remove commas, standardize format
            normalized = re.sub(r",", "", match)
            numbers.append(normalized.lower())

    return list(set(numbers))

def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentence strings
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Filter empty and very short sentences
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def sentence_jaccard(
    claim: str,
    context: str,
    stopwords: frozenset = STOPWORDS,
) -> float:
    """Calculate maximum Jaccard similarity between claim and context sentences.

    Compares the claim against each sentence in the context and returns
    the highest similarity score. A claim is considered grounded if at least
    one sentence provides strong support.

    Args:
        claim: The claim text
        context: The context text
        stopwords: Set of stopwords to filter

    Returns:
        Maximum Jaccard similarity among all sentences
    """
    ctx_sentences = split_sentences(context)
    if not ctx_sentences:
        return 0.0

    # Tokenize claim and remove stopwords
    claim_words = set(claim.lower().split()) - stopwords

    if not claim_words:
        return 0.0

    scores = []
    for ctx_sentence in ctx_sentences:
        context_words = set(ctx_sentence.lower().split()) - stopwords
        if context_words:
            intersection = claim_words & context_words
            union = claim_words | context_words
            jaccard = len(intersection) / len(union) if union else 0.0
            scores.append(jaccard)

    return max(scores, default=0.0)


def claim_support(
    claim: str,
    context: str,
    min_key_terms: int = 3,
    key_term_threshold: float = 0.5,
    number_weight: float = 2.0,
    jaccard_threshold: float = 0.25,
    ngram_threshold: float = 0.15,
) -> bool:
    """Check if a claim is supported by context using 3-stage hybrid matching.

    Primary method for claim verification (groundedness, citation validity).

    Stage 1: Key term matching (numbers, proper nouns, technical terms)
    Stage 2: Sentence-level Jaccard (max similarity among all sentences)
    Stage 3: N-gram fallback (original method with lower threshold)

    Args:
        claim: The claim text
        context: The context text
        min_key_terms: Minimum key terms to confirm at stage 1
        key_term_threshold: Minimum weighted match ratio for stage 1
        number_weight: Weight for numeric terms
        jaccard_threshold: Minimum Jaccard similarity for stage 2
        ngram_threshold: Minimum n-gram overlap for stage 3

    Returns:
        True if claim appears supported by context
    """
    # Stage 1: Key term matching
    claim_terms = extract_key_terms(claim, number_weight)

    if len(claim_terms) >= min_key_terms:
        context_lower = context.lower()

        total_weight = sum(w for _, w in claim_terms)
        matched_weight = 0.0

        for term, weight in claim_terms:
            # Check if term appears in context (word boundary match)
            if re.search(r"\b" + re.escape(term) + r"\b", context_lower):
                matched_weight += weight

        if total_weight > 0:
            match_ratio = matched_weight / total_weight
            if match_ratio >= key_term_threshold:
                return True

    # Stage 2: Sentence-level Jaccard
    jaccard = sentence_jaccard(claim, context)
    if jaccard >= jaccard_threshold:
        return True

    # Stage 3: N-gram fallback (original method with lowered threshold)
    overlap = calc_ngram_overlap(claim, context, n=3)
    return overlap > ngram_threshold

def numeric_fabrication_count(answer: str, context: str) -> int:
    """Count numbers in answer that don't appear in context.

    Args:
        answer: The generated answer
        context: The source context

    Returns:
        Number of potentially fabricated numbers
    """
    answer_numbers = extract_numbers(answer)
    context_numbers = extract_numbers(context)

    # Also extract just the digits for fuzzy matching
    context_digits = set()
    for num in context_numbers:
        digits = re.sub(r"[^\d.]", "", num)
        if digits:
            context_digits.add(digits)

    fabricated = 0
    for num in answer_numbers:
        digits = re.sub(r"[^\d.]", "", num)
        if digits and digits not in context_digits:
            # Check if it's a reasonable transformation
            if not any(digits in ctx_d or ctx_d in digits for ctx_d in context_digits):
                fabricated += 1

    return fabricated


def citation_validity_form(
    cited_doc_ids: list[str],
    retrieved_doc_ids: list[str],
) -> float:
    """Check form validity: all cited docs were retrieved.

    Args:
        cited_doc_ids: Doc IDs cited in the answer
        retrieved_doc_ids: Doc IDs that were retrieved

    Returns:
        Ratio of valid citations (0.0-1.0)
    """
    if not cited_doc_ids:
        return 1.0  # No citations to validate

    retrieved_set = set(retrieved_doc_ids)
    valid = sum(1 for doc_id in cited_doc_ids if doc_id in retrieved_set)
    return valid / len(cited_doc_ids)


def citation_validity_content(
    citations: list[dict],
    chunks: dict[str, str],
    answer: str,
) -> float:
    """Check content validity: citation context supports claims.

    Args:
        citations: List of citation dicts with doc_id, chunk_id, text_snippet
        chunks: Dict mapping chunk_id -> chunk text
        answer: The generated answer

    Returns:
        Ratio of content-valid citations (0.0-1.0)
    """
    if not citations:
        return 1.0

    valid = 0
    for citation in citations:
        chunk_id = citation.get("chunk_id", "")
        text_snippet = citation.get("text_snippet", "")

        # Get the full chunk if available
        chunk_text = chunks.get(chunk_id, text_snippet)

        # Check if there's meaningful overlap between citation context and answer
        if claim_support(answer, chunk_text):
            valid += 1

    return valid / len(citations)


# =============================================================================
# Safety Metrics
# =============================================================================


def roc_curve_points(
    scores: list[float],
    labels: list[bool],
    thresholds: Optional[list[float]] = None,
) -> list[tuple[float, float, float]]:
    """Calculate ROC curve points.

    Args:
        scores: Threat/detection scores
        labels: True labels (True = attack/positive, False = benign/negative)
        thresholds: Thresholds to evaluate (default: 0.0 to 1.0 in 0.01 steps)

    Returns:
        List of (threshold, fpr, tpr) tuples
    """
    if thresholds is None:
        thresholds = [i / 100 for i in range(101)]

    points = []
    for thresh in thresholds:
        predictions = [s >= thresh for s in scores]

        tp = sum(1 for p, l in zip(predictions, labels) if p and l)
        fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
        fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
        tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        points.append((thresh, fpr, tpr))

    return points


def auc_from_points(points: list[tuple[float, float, float]]) -> float:
    """Calculate AUC from ROC points using trapezoidal rule.

    Args:
        points: List of (threshold, fpr, tpr) tuples

    Returns:
        AUC value (0.0-1.0)
    """
    # Sort by FPR
    sorted_points = sorted(points, key=lambda x: x[1])

    auc = 0.0
    for i in range(1, len(sorted_points)):
        x1, y1 = sorted_points[i - 1][1], sorted_points[i - 1][2]
        x2, y2 = sorted_points[i][1], sorted_points[i][2]
        auc += (x2 - x1) * (y1 + y2) / 2

    return auc


def tpr_at_fpr(
    points: list[tuple[float, float, float]],
    target_fpr: float,
) -> float:
    """Find TPR at a specific FPR level.

    Args:
        points: List of (threshold, fpr, tpr) tuples
        target_fpr: Target FPR value

    Returns:
        TPR at the closest FPR <= target_fpr
    """
    # Sort by FPR
    sorted_points = sorted(points, key=lambda x: x[1])

    best_tpr = 0.0
    for thresh, fpr, tpr in sorted_points:
        if fpr <= target_fpr:
            best_tpr = max(best_tpr, tpr)

    return best_tpr


def precision_recall_at_threshold(
    scores: list[float],
    labels: list[bool],
    threshold: float,
) -> tuple[float, float]:
    """Calculate precision and recall at a specific threshold.

    Args:
        scores: Detection scores
        labels: True labels
        threshold: Classification threshold

    Returns:
        Tuple of (precision, recall)
    """
    predictions = [s >= threshold for s in scores]

    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


# =============================================================================
# Shared Utilities
# =============================================================================


def calculate_percentiles(
    values: list[float],
    percentiles: Optional[list[int]] = None,
) -> dict[str, float]:
    """Calculate percentiles for a list of values.

    Args:
        values: List of numeric values
        percentiles: Percentiles to calculate (default: [50, 95, 99])

    Returns:
        Dict mapping percentile name (e.g., "p50") to value
    """
    if percentiles is None:
        percentiles = [50, 95, 99]

    if not values:
        return {f"p{p}": 0.0 for p in percentiles}

    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def aggregate_metrics(
    results: list[dict[str, float]],
    metric_names: list[str],
) -> dict[str, float]:
    """Aggregate metrics across multiple results.

    Args:
        results: List of metric dicts from individual evaluations
        metric_names: Names of metrics to aggregate

    Returns:
        Dict with mean values for each metric
    """
    aggregated = {}
    for metric in metric_names:
        values = [r.get(metric) for r in results if r.get(metric) is not None]
        if values:
            aggregated[f"mean_{metric}"] = sum(values) / len(values)
    return aggregated
