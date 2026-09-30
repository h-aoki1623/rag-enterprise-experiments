# RAG System Evaluation Improvement Plan

## Current Status
- **Overall Pass Rate**: 55.3% (47/85 cases)
- **Critical Issues**: Groundedness 0%, Context Quality 20%, Safety Output 50%

## Root Cause Summary

| Perspective | Pass Rate | Root Cause |
|-------------|-----------|------------|
| Groundedness | 0% | 3-gram lexical matching fails on LLM paraphrases |
| Context Quality | 20% | Small chunks (500 chars) fragment facts |
| Safety Input | 88.6% | Low delimiter scores |
| Safety Output | 50% | Low metadata score multiplier, PII not blocking |
| Pipeline | 20% | Downstream effects from above issues |

---

## Phase 1: Groundedness Fix (CRITICAL)

### Problem
`claim_context_overlap()` uses 3-gram overlap (threshold 0.3). LLM paraphrases yield 0% overlap.

### Solution: Hybrid Lexical Matching (3-Stage)

**File**: [metrics.py](src/rag/evals/metrics.py)

Replace `claim_context_overlap()` with multi-stage matching:

#### Stage 1: Key Term Matching
- Extract: numbers, proper nouns, technical terms
- **Condition**: Only confirm supported if `key_terms >= min_key_terms` (default: 3)
- **Match**: 50% of claim's key terms found in context → supported
- **Number weight**: Numbers get 2x weight (strong anchor, effective for fabrication detection)
- **Normalization**: Handle variations (1,000 → 1000, 10 min → 10 minutes)

```python
def extract_key_terms(text: str) -> list[tuple[str, float]]:
    """Extract key terms with weights. Numbers get weight=2.0, others=1.0"""
    terms = []
    # Numbers: 2024, 1.2, 30%, $500
    numbers = re.findall(r'\b\d+(?:[.,]\d+)?%?\b|\$[\d,]+', text)
    terms.extend((normalize_number(n), 2.0) for n in numbers)
    # Proper nouns, technical terms
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[A-Z]{2,}\b', text)
    terms.extend((w.lower(), 1.0) for w in words)
    return terms
```

#### Stage 2: Sentence-Level Jaccard
- **Improvement**: Compare against **Top-3 candidate sentences** instead of full context
- Remove stopwords, then calculate Jaccard
- Threshold: 0.25 (slightly stricter since long-context problem is avoided)

```python
def sentence_jaccard(claim: str, context: str, top_k: int = 3) -> float:
    """Calculate Jaccard against best matching sentences."""
    sentences = split_sentences(context)
    claim_words = set(claim.lower().split()) - STOPWORDS

    best_scores = []
    for sent in sentences:
        sent_words = set(sent.lower().split()) - STOPWORDS
        if claim_words and sent_words:
            jaccard = len(claim_words & sent_words) / len(claim_words | sent_words)
            best_scores.append(jaccard)

    # Return max of top-k sentences
    return max(sorted(best_scores, reverse=True)[:top_k], default=0.0)
```

#### Stage 3: N-gram Fallback
- Original method with threshold 0.15 (from 0.3)
- Ensures exact phrase matches are captured

### Limitations
- This is "lexical" based, not strictly "semantic"
- Cannot detect contradictions like "X increases Y" vs "X decreases Y"
- Future enhancement: Use embeddings for candidate sentence extraction

**File**: [config.py](src/rag/config.py)

Add to `EvalSettings`:
```python
# Key term matching
min_key_terms: int = 3              # Minimum terms to confirm at stage 1
key_term_threshold: float = 0.5     # 50% match required
number_weight: float = 2.0          # Numbers weighted higher

# Sentence-level Jaccard
jaccard_top_k_sentences: int = 3    # Compare against top-3 sentences
jaccard_threshold: float = 0.25     # Threshold for sentence matching

# N-gram fallback
ngram_fallback_threshold: float = 0.15  # Lowered from 0.3
```

### Verification
```bash
python -m src.app eval --perspective groundedness -v
# Target: claim_support_rate >= 0.85
```

---

## Phase 2: Context Quality Fix

### Problem
- Child chunks (500 chars) fragment facts
- Parent preview truncated to 500 chars (of 3000)
- Fact matching uses exact substring only

### Solution: Increase Chunk Sizes + Fuzzy Matching

**File**: [config.py](src/rag/config.py)

```python
child_chunk_size: int = 800        # from 500
child_chunk_overlap: int = 100     # from 50
parent_preview_size: int = 1500    # from 500
```

**File**: [metrics.py](src/rag/evals/metrics.py)

Add fuzzy fact matching using word overlap with aliases.

**Action Required**: Re-ingest documents after config change
```bash
python -m src.app ingest
```

### Verification
```bash
python -m src.app eval --perspective context_quality -v
# Target: facts_found_ratio >= 0.5
```

---

## Phase 3: Safety Input - Delimiter Block Rate

### Problem
Delimiter score (0.5) * 0.7 = 0.35 < block threshold (0.50).

### Solution: Increase Delimiter Base Scores

**File**: [guardrails.py](src/rag/guardrails.py)

In `_calc_delimiter_score()`:
```python
# Paired markers: 0.6 (from 0.5)
# Paired boost: 0.4 (from 0.3)
# Result: 0.6 + 0.4 = 1.0 -> composite ~0.75 -> BLOCK
```

### Verification
```bash
python -m src.app eval --perspective safety -v
# Target: delimiter_attack block_rate >= 80%
```

---

## Phase 4: Safety Output - Metadata Detection

### Problem
Current approach relies on complex scoring formula, which results in 0% detection rate for metadata exposure.

### Root Cause Analysis

**Current Design Issue**:
- Metadata detection sets `metadata_leak_count`, but action is determined by `total_score`
- Complex scoring formula often results in score below thresholds
- 1 metadata leak = 0.3 score → total_score ≈ 0.23 (below confidential warn threshold 0.32)

**Classification Thresholds** (config.py:89-94):
| Classification | Allow | Warn | Block |
|----------------|-------|------|-------|
| public         | 0.40  | 0.64 | 0.80  |
| internal       | 0.30  | 0.48 | 0.60  |
| confidential   | 0.20  | 0.32 | 0.40  |

### Solution: Classification-Aware Explicit Rules

Instead of relying on scoring formula, implement explicit rules based on classification:

| Classification | If metadata_leak_count > 0 |
|----------------|----------------------------|
| public         | Redact only (sanitize_needed=True) |
| internal       | Redact + Block if metadata_score >= internal block threshold (0.60) |
| confidential   | Redact + Block if metadata_score >= confidential block threshold (0.40) |

**Key Design Principles**:
1. **Detection = Immediate Redaction**: Any metadata leak triggers sanitization
2. **Classification-aware Blocking**: Stricter classification = lower block threshold
3. **Bypass Complex Scoring**: Direct threshold comparison for metadata

**File**: [guardrails.py](src/rag/guardrails.py)

In `OutputGuardrail.check()`, add classification-aware metadata handling:

```python
# After sanitize lane detection
if sanitize_breakdown.metadata_leak_count > 0:
    sanitize_breakdown.sanitize_needed = True  # Always redact

    # Classification-aware blocking
    if classification in (Classification.INTERNAL, Classification.CONFIDENTIAL):
        block_threshold = self.config.leakage_thresholds[classification.value]["block"]
        metadata_score = min(1.0, sanitize_breakdown.metadata_leak_count * 0.5)

        if metadata_score >= block_threshold:
            return GuardrailResult(
                guardrail_type="output",
                is_safe=False,
                action=GuardrailAction.BLOCK,
                threat_score=metadata_score,
                threat_type=LeakageType.METADATA_EXPOSURE.value,
                message=f"Metadata exposure detected in {classification.value} document",
                score_breakdown=sanitize_breakdown,
            )
```

**File**: [config.py](src/rag/config.py)

Add metadata-specific multiplier (optional, for fine-tuning):
```python
metadata_score_multiplier: float = 0.5  # Score per metadata leak
```

### Verification
```bash
python -m src.app eval --perspective safety -v
# Target: metadata_exposure detection_rate >= 80%
```

---

## Phase 5: Safety Output - PII Blocking

### Problem
PII detected but not blocked. Two-lane design separates sanitization from blocking.
Current: 50% detection rate, 0% block rate.

### Solution: Classification-Aware PII Handling

Same approach as Phase 4 - explicit rules based on classification:

| Classification | If pii_detected_count > 0 |
|----------------|---------------------------|
| public         | Redact only (sanitize_needed=True) |
| internal       | Redact + Block if pii_score >= internal block threshold (0.60) |
| confidential   | Redact + Block if pii_score >= confidential block threshold (0.40) |

**File**: [guardrails.py](src/rag/guardrails.py)

In `OutputGuardrail.check()`, add classification-aware PII handling:

```python
# After sanitize lane detection (alongside metadata check)
if sanitize_breakdown.pii_detected_count > 0:
    sanitize_breakdown.sanitize_needed = True  # Always redact

    # Classification-aware blocking
    if classification in (Classification.INTERNAL, Classification.CONFIDENTIAL):
        block_threshold = self.config.leakage_thresholds[classification.value]["block"]
        pii_score = min(1.0, sanitize_breakdown.pii_detected_count * 0.5)

        if pii_score >= block_threshold:
            return GuardrailResult(
                guardrail_type="output",
                is_safe=False,
                action=GuardrailAction.BLOCK,
                threat_score=pii_score,
                threat_type=LeakageType.PII_IN_OUTPUT.value,
                message=f"PII detected in {classification.value} document",
                score_breakdown=sanitize_breakdown,
            )
```

**File**: [config.py](src/rag/config.py)

Add PII-specific multiplier (optional, for fine-tuning):
```python
pii_score_multiplier: float = 0.5  # Score per PII detection
```

### Note: High-Severity PII Still Gets Special Treatment

Existing high-confidence secret patterns (PEM, AWS keys, JWT) already trigger immediate BLOCK.
Consider adding high-severity PII patterns (SSN, credit card) to this list:

```python
HIGH_SEVERITY_PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN (US)
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
]
# These bypass threshold check -> immediate BLOCK
```

### Verification
```bash
python -m src.app eval --perspective safety -v
# Target: pii_exposure detection_rate >= 80%, block_rate >= 50%
```

---

## Phase 6: Safety Output - Secret Detection

### Problem
Secret detection follows the same issue as metadata and PII - relies on complex scoring formula.

### Solution: Classification-Aware Secret Handling

Same approach as Phase 4 and 5 - explicit rules based on classification:

| Classification | If secret_detected_count > 0 |
|----------------|------------------------------|
| public         | Redact only (sanitize_needed=True) |
| internal       | Redact + Block if secret_score >= internal block threshold (0.60) |
| confidential   | Redact + Block if secret_score >= confidential block threshold (0.40) |

**Note**: High-confidence secrets (PEM, AWS keys, JWT) already trigger immediate BLOCK via `HIGH_CONFIDENCE_SECRET_PATTERNS`. This phase handles lower-confidence secret patterns.

**File**: [guardrails.py](src/rag/guardrails.py)

In `OutputGuardrail.check()`, add classification-aware secret handling:

```python
# After sanitize lane detection (alongside metadata and PII checks)
if sanitize_breakdown.secret_detected_count > 0:
    sanitize_breakdown.sanitize_needed = True  # Always redact

    # Classification-aware blocking
    if classification in (Classification.INTERNAL, Classification.CONFIDENTIAL):
        block_threshold = self.config.leakage_thresholds[classification.value]["block"]
        secret_score = min(1.0, sanitize_breakdown.secret_detected_count * 0.5)

        if secret_score >= block_threshold:
            return GuardrailResult(
                guardrail_type="output",
                is_safe=False,
                action=GuardrailAction.BLOCK,
                threat_score=secret_score,
                threat_type=LeakageType.SECRET_IN_OUTPUT.value,
                message=f"Secret detected in {classification.value} document",
                score_breakdown=sanitize_breakdown,
            )
```

**File**: [config.py](src/rag/config.py)

Add secret-specific multiplier (optional, for fine-tuning):
```python
secret_score_multiplier: float = 0.5  # Score per secret detection
```

### Verification
```bash
python -m src.app eval --perspective safety -v
# Target: secret detection properly handled by classification
```

---

## Implementation Order

1. **Phase 1**: Groundedness (highest impact, ~0% -> 70%+)
2. **Phase 2**: Context Quality (requires re-ingestion)
3. **Phase 3**: Safety Input - Delimiter (additive, low risk)
4. **Phase 4-6**: Safety Output - Metadata, PII, Secret (classification-aware)

## Files to Modify

| File | Changes |
|------|---------|
| [src/rag/evals/metrics.py](src/rag/evals/metrics.py) | claim_context_overlap_v2, fuzzy fact matching |
| [src/rag/config.py](src/rag/config.py) | EvalSettings thresholds, chunk sizes |
| [src/rag/guardrails.py](src/rag/guardrails.py) | Delimiter scores, output scoring |
| [src/rag/evals/groundedness_evals.py](src/rag/evals/groundedness_evals.py) | Use new claim matching function |

## Final Verification

```bash
# Full eval suite
python -m src.app eval --save-trace -v

# Expected results:
# - Retrieval: 70% -> 75%+ (minor improvement from better chunks)
# - Context Quality: 20% -> 60%+
# - Groundedness: 0% -> 70%+
# - Safety Input: 88.6% -> 92%+ (delimiter fix only)
# - Safety Output: 50% -> 80%+
# - Pipeline: 20% -> 60%+ (downstream improvement)
# - Overall: 55% -> 70%+
```
