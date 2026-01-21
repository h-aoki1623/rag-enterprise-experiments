"""Guardrails module for RAG system security.

This module provides defensive layers for LLM application inputs and outputs:
- InputGuardrail: Detects prompt injection attacks before LLM processing
- OutputGuardrail: Detects data leakage in LLM responses before returning to user

Architecture:
    User Input → [InputGuardrail] → LLM → [OutputGuardrail] → Response
                      ↓                         ↓
                Injection detection        Leakage detection
                Query sanitization         PII filtering
                Block/Warn decision        Content filtering
"""

import re
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from .models import Classification

if TYPE_CHECKING:
    from .config import GuardrailSettings


# =============================================================================
# Shared Enums & Models
# =============================================================================


class GuardrailAction(str, Enum):
    """Action taken by guardrail."""

    ALLOW = "allow"
    WARN = "warn"
    REDACT = "redact"
    BLOCK = "block"


class TrustLevel(str, Enum):
    """Trust boundary classification.

    Classifies data into categories with explicit handling rules:
    - UNTRUSTED: User queries, document content - always validate & sanitize
    - TRUSTED_SENSITIVE: Metadata, system prompt, ACL - minimize exposure to LLM
    - GENERATED: Model output - inspect as leakage vector
    """

    UNTRUSTED = "untrusted"
    TRUSTED_SENSITIVE = "trusted_sensitive"
    GENERATED = "generated"


class GuardrailResult(BaseModel):
    """Result from any guardrail check."""

    guardrail_type: str  # "input" or "output"
    is_safe: bool
    action: GuardrailAction
    threat_score: float = 0.0
    threat_type: Optional[str] = None
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Input Guardrail (Injection Detection)
# =============================================================================


class InjectionType(str, Enum):
    """Types of injection attacks detected by InputGuardrail."""

    DIRECT_QUERY = "direct_query"
    INDIRECT_DOCUMENT = "indirect_document"
    DELIMITER_ATTACK = "delimiter_attack"
    INSTRUCTION_OVERRIDE = "instruction_override"
    JAILBREAK_INTENT = "jailbreak_intent"


class InjectionScoreBreakdown(BaseModel):
    """Input guardrail composite score breakdown.

    Uses weighted scoring with 5 components:
    - pattern_score: Known attack pattern matches (0.25 weight)
    - structural_score: Instruction structure detection (0.25 weight)
    - delimiter_score: Delimiter manipulation attacks (0.15 weight)
    - anomaly_score: Statistical anomalies (0.15 weight)
    - jailbreak_intent_score: Intent to bypass restrictions (0.20 weight)
    """

    pattern_score: float = 0.0
    structural_score: float = 0.0
    delimiter_score: float = 0.0
    anomaly_score: float = 0.0
    jailbreak_intent_score: float = 0.0

    # Weights for composite scoring
    _weights: dict[str, float] = {
        "pattern": 0.25,
        "structural": 0.25,
        "delimiter": 0.15,
        "anomaly": 0.15,
        "jailbreak_intent": 0.20,
    }

    @property
    def total_score(self) -> float:
        """Calculate total score using max-of-weighted approach.

        Uses the maximum of weighted component scores, which ensures that
        a high score in any single component results in a meaningful total score.
        The weighted sum is used as a secondary factor to boost scores when
        multiple components are triggered.
        """
        # Calculate weighted individual scores
        weighted_scores = [
            self.pattern_score * self._weights["pattern"],
            self.structural_score * self._weights["structural"],
            self.delimiter_score * self._weights["delimiter"],
            self.anomaly_score * self._weights["anomaly"],
            self.jailbreak_intent_score * self._weights["jailbreak_intent"],
        ]

        # Primary: max of raw component scores (not weighted)
        # This ensures a high pattern_score of 0.9 gives meaningful detection
        max_raw = max(
            self.pattern_score,
            self.structural_score,
            self.delimiter_score,
            self.anomaly_score,
            self.jailbreak_intent_score,
        )

        # Secondary: weighted sum provides boost when multiple signals present
        weighted_sum = sum(weighted_scores)

        # Combine: use max raw score, boosted by weighted sum
        # Scale factor ensures score stays in 0-1 range
        combined = max_raw * 0.7 + weighted_sum * 0.3

        return min(1.0, combined)


class InputGuardrail:
    """Input Guardrail: Inspects user queries before LLM processing.

    Detects prompt injection attacks including:
    - Known attack patterns ("ignore previous instructions", etc.)
    - Instruction-like structures (role assignment, imperative commands)
    - Delimiter manipulation attacks (fake markers, boundary manipulation)
    - Statistical anomalies (excessive length, encoded payloads)
    - Jailbreak intent ("bypass filters", "even if forbidden", etc.)
    """

    def __init__(self, config: "GuardrailSettings") -> None:
        """Initialize InputGuardrail with configuration.

        Args:
            config: Guardrail settings with thresholds and feature flags.
        """
        self.config = config

    def check(
        self,
        query: str,
    ) -> GuardrailResult:
        """Check query for injection attacks.

        Uses a fixed threshold regardless of user role or document classification.
        This protects against compromised user accounts - varying thresholds by
        role would be a security vulnerability.

        Args:
            query: User query to inspect.

        Returns:
            GuardrailResult with action and score breakdown.
        """
        if not self.config.input_guardrail_enabled:
            return GuardrailResult(
                guardrail_type="input",
                is_safe=True,
                action=GuardrailAction.ALLOW,
            )

        breakdown = InjectionScoreBreakdown(
            pattern_score=self._calc_pattern_score(query),
            structural_score=self._calc_structural_score(query),
            delimiter_score=self._calc_delimiter_score(query),
            anomaly_score=self._calc_anomaly_score(query),
            jailbreak_intent_score=self._calc_jailbreak_intent_score(query),
        )

        action = self._determine_action(breakdown.total_score)
        threat_type = self._determine_threat_type(breakdown)

        return GuardrailResult(
            guardrail_type="input",
            is_safe=(action == GuardrailAction.ALLOW),
            action=action,
            threat_score=breakdown.total_score,
            threat_type=threat_type,
            score_breakdown=breakdown.model_dump(),
            details={
                "query_length": len(query),
            },
        )

    def _determine_action(self, score: float) -> GuardrailAction:
        """Determine action based on score and configurable thresholds.

        Uses fixed thresholds for all queries, regardless of user role or
        target document classification. This is a security best practice -
        if a user account is compromised, varying thresholds by role would
        be exploitable.

        Thresholds are individually configurable:
        - injection_allow_threshold: below this → ALLOW
        - injection_warn_threshold: below this → WARN
        - injection_block_threshold: below this → REDACT, at or above → BLOCK

        Args:
            score: Total threat score (0.0-1.0).

        Returns:
            GuardrailAction to take.
        """
        if score < self.config.injection_allow_threshold:
            return GuardrailAction.ALLOW
        elif score < self.config.injection_warn_threshold:
            return GuardrailAction.WARN
        elif score < self.config.injection_block_threshold:
            return GuardrailAction.REDACT
        else:
            return GuardrailAction.BLOCK

    def _determine_threat_type(self, breakdown: InjectionScoreBreakdown) -> Optional[str]:
        """Determine primary threat type from score breakdown.

        Args:
            breakdown: Score breakdown from detection.

        Returns:
            InjectionType value or None if no significant threat.
        """
        scores = {
            InjectionType.DIRECT_QUERY: breakdown.pattern_score,
            InjectionType.INSTRUCTION_OVERRIDE: breakdown.structural_score,
            InjectionType.DELIMITER_ATTACK: breakdown.delimiter_score,
            InjectionType.JAILBREAK_INTENT: breakdown.jailbreak_intent_score,
        }

        max_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[max_type] > 0.3:
            return max_type.value
        return None

    def _calc_pattern_score(self, text: str) -> float:
        """Calculate known attack pattern score.

        Detection targets:
        - "ignore previous instructions"
        - "reveal system prompt" (only with system/initial keywords)
        - "disregard above"
        - "forget everything"

        Returns a high score (0.8-1.0) for clear attack patterns.
        Lower scores for patterns that could be benign without context.
        """
        score = 0.0
        text_lower = text.lower()

        # Direct instruction override patterns - very high signal
        override_patterns = [
            r"(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?|prompts?)",
            r"(?i)disregard\s+(all\s+)?(previous|above|prior|the\s+above)\s+(instructions?|text|context)",
            r"(?i)forget\s+(everything|all)\s+(above|before|you\s+know)",
            r"(?i)do\s+not\s+follow\s+(the\s+)?(previous|above|system)",
        ]
        for pattern in override_patterns:
            if re.search(pattern, text):
                score += 0.9  # High score for direct override
                break

        # System prompt extraction patterns - conditional scoring
        # High score only when "system" or "initial" is present
        # This avoids false positives like "what instructions should I follow?"
        system_extraction_patterns = [
            r"(?i)reveal\s+(your\s+)?(system\s+)prompt",
            r"(?i)show\s+(me\s+)?your\s+(system|initial)\s+(prompt|instructions?)",
            r"(?i)what\s+(are\s+)?your\s+(system|initial)\s+(prompt|instructions?)",
            r"(?i)print\s+(your\s+)?(system\s+)prompt",
            r"(?i)output\s+(your\s+)?(initial|system)\s+(prompt|instructions?)",
            r"(?i)display\s+(your\s+)?(system|initial)\s+(prompt|instructions?)",
        ]
        for pattern in system_extraction_patterns:
            if re.search(pattern, text):
                score += 0.8
                break

        # Generic "your instructions" without system/initial - lower score
        # These are more ambiguous and could be benign questions
        generic_instruction_patterns = [
            r"(?i)what\s+(are\s+)?your\s+instructions\??$",
            r"(?i)show\s+(me\s+)?your\s+instructions\??$",
        ]
        has_system_context = bool(
            re.search(r"(?i)\b(system|initial|original|hidden|secret)\b", text)
        )
        for pattern in generic_instruction_patterns:
            if re.search(pattern, text):
                # Only high score if combined with system-related keywords
                if has_system_context:
                    score += 0.7
                else:
                    score += 0.3  # Low score for ambiguous questions
                break

        # Harmful instruction patterns
        harmful_patterns = [
            r"(?i)pretend\s+(you\s+are|to\s+be)\s+(a\s+)?evil",
            r"(?i)act\s+(as\s+if|like)\s+you\s+have\s+no\s+restrictions?",
            r"(?i)you\s+are\s+now\s+(a\s+)?different\s+(AI|assistant|model)",
        ]
        for pattern in harmful_patterns:
            if re.search(pattern, text):
                score += 0.6
                break

        return min(1.0, score)

    def _calc_structural_score(self, text: str) -> float:
        """Calculate instruction structure detection score.

        Detection targets:
        - Imperative verbs + role override ("you are now", "act as")
        - Explicit instructions ("instruction:", "command:")
        - External reference requests with risk-based scoring

        Conditional scoring to reduce false positives:
        - "act as" / "you are now" alone: lower score
        - With manipulative context (AI, assistant, bypass): higher score
        """
        score = 0.0

        # Check for manipulative context (used for conditional scoring)
        manipulative_context = bool(
            re.search(
                r"(?i)\b(AI|assistant|model|chatbot|no\s+restrictions?|unrestricted|"
                r"evil|malicious|harmful|jailbreak|bypass|ignore\s+rules?)\b",
                text,
            )
        )

        # Role override patterns - conditional scoring
        # High score when combined with manipulative context
        role_patterns = [
            r"(?i)you\s+are\s+now\s+(a\s+)?",
            r"(?i)act\s+as\s+(a\s+|an\s+)?",
            r"(?i)pretend\s+(to\s+be|you\s+are|you\s+have)\s+",
            r"(?i)from\s+now\s+on\s*,?\s*(you|your)",
            r"(?i)new\s+role\s*:?\s*you",
            r"(?i)imagine\s+you\s+(are|have|don'?t)",
        ]
        for pattern in role_patterns:
            if re.search(pattern, text):
                if manipulative_context:
                    score += 0.7  # High score with manipulative context
                else:
                    score += 0.3  # Lower score for benign "act as a teacher"
                break

        # Explicit instruction markers - high signal (always suspicious)
        instruction_patterns = [
            r"(?i)^(instruction|command|directive|order)\s*:?\s*",
            r"(?i)\[(instruction|command|system|prompt)\]",
            r"(?i)<(instruction|command|system|prompt)>",
        ]
        for pattern in instruction_patterns:
            if re.search(pattern, text):
                score += 0.7
                break

        # External resource access - risk-based scoring
        # High risk: execute code/command
        high_risk_external = [
            r"(?i)(execute|run)\s+(this\s+)?(code|command|script|shell)",
            r"(?i)(call|invoke)\s+(shell|system|exec|eval)",
            r"(?i)os\.(system|popen|exec)",
            r"(?i)subprocess\.(run|call|Popen)",
        ]
        if any(re.search(p, text) for p in high_risk_external):
            score += 0.6

        # Medium risk: file/database/api access
        medium_risk_external = [
            r"(?i)(access|retrieve|fetch|read)\s+(the\s+)?(internal\s+)?(file|database)",
            r"(?i)(call|invoke)\s+(the\s+)?(internal\s+)?(api|endpoint)",
        ]
        if any(re.search(p, text) for p in medium_risk_external):
            score += 0.4

        # Low risk: general URL/API fetch (could be benign)
        low_risk_external = [
            r"(?i)(access|retrieve|fetch|read)\s+(the\s+)?(url|link|website)",
            r"(?i)(call|invoke)\s+(the\s+)?(function|method)",
        ]
        # Only add if no higher risk pattern matched
        if score < 0.4 and any(re.search(p, text) for p in low_risk_external):
            score += 0.2

        return min(1.0, score)

    def _calc_delimiter_score(self, text: str) -> float:
        """Calculate delimiter manipulation attack score.

        Detection targets:
        - Fake system/assistant/user markers
        - Common delimiter patterns
        - Boundary manipulation attempts

        Improvements:
        - Single occurrence: lower base score (could be chat log paste)
        - Multiple occurrences: higher score (likely intentional attack)
        - Paired markers (open + close): high score
        """
        score = 0.0

        # Fake role markers - count occurrences for multiple hits boost
        role_marker_patterns = [
            r"\[/?system\]",
            r"\[/?assistant\]",
            r"\[/?user\]",
            r"<</?system>>",
            r"<</?assistant>>",
            r"<</?user>>",
            r"<\|system\|>",
            r"<\|assistant\|>",
            r"<\|user\|>",
            r"###\s*(system|assistant|user)\s*###",
        ]
        role_marker_count = 0
        for pattern in role_marker_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            role_marker_count += len(matches)

        if role_marker_count >= 3:
            # Multiple markers suggest intentional prompt structure manipulation
            score += 0.7
        elif role_marker_count == 2:
            # Paired markers (open/close) are suspicious
            score += 0.5
        elif role_marker_count == 1:
            # Single marker could be accidental (chat log paste)
            score += 0.2

        # Check for paired markers (opening + closing) - very suspicious
        paired_marker_patterns = [
            (r"\[system\]", r"\[/system\]"),
            (r"\[assistant\]", r"\[/assistant\]"),
            (r"\[user\]", r"\[/user\]"),
            (r"<<system>>", r"<</system>>"),
            (r"<\|system\|>", r"<\|/system\|>"),
        ]
        for open_pattern, close_pattern in paired_marker_patterns:
            if re.search(open_pattern, text, re.IGNORECASE) and re.search(
                close_pattern, text, re.IGNORECASE
            ):
                score += 0.3  # Additional boost for paired markers
                break

        # End of prompt markers - higher score for explicit boundary manipulation
        end_patterns = [
            r"(?i)---\s*end\s+(of\s+)?(system\s+)?(prompt|instructions?)\s*---",
            r"(?i)===\s*end\s+(of\s+)?(system\s+)?(prompt|instructions?)\s*===",
            r"(?i)\*\*\*\s*end\s+(of\s+)?(system\s+)?(prompt|instructions?)\s*\*\*\*",
        ]
        for pattern in end_patterns:
            if re.search(pattern, text):
                score += 0.5
                break

        # New conversation/context markers - check for command-like intent
        # "start a new conversation" could be benign
        # "clear the context and ignore previous" is suspicious
        has_override_intent = bool(
            re.search(r"(?i)\b(ignore|forget|disregard|bypass)\b", text)
        )
        new_context_patterns = [
            r"(?i)(new|fresh|start)\s+(conversation|context|session)",
            r"(?i)clear\s+(the\s+)?(context|history|memory)",
            r"(?i)reset\s+(to\s+)?(default|original|initial)",
        ]
        for pattern in new_context_patterns:
            if re.search(pattern, text):
                if has_override_intent:
                    score += 0.4  # Higher score with override intent
                else:
                    score += 0.15  # Lower score for potentially benign usage
                break

        return min(1.0, score)

    def _calc_anomaly_score(self, text: str) -> float:
        """Calculate statistical anomaly score.

        Detection targets:
        - Excessive length
        - Encoded payloads (base64-like with improved detection)
        - Unusual character patterns (homoglyphs, control chars)
        - Repeated patterns

        Improvements:
        - Better Base64 detection: check for balanced character distribution
        - Reduced false positives for long alphanumeric strings (UUIDs, hashes)
        """
        score = 0.0

        # Length anomaly
        if len(text) > self.config.max_query_length:
            score += 0.3

        # Improved Base64 detection
        # 1. Find potential Base64 strings (40+ chars, alphanumeric + / and +)
        # 2. Filter out common false positives (UUIDs, hex hashes, URLs)
        base64_pattern = r"[A-Za-z0-9+/]{40,}={0,2}"
        potential_b64_matches = re.findall(base64_pattern, text)

        for match in potential_b64_matches:
            # Skip if it looks like a UUID (contains only hex chars and dashes)
            if re.match(r"^[A-Fa-f0-9-]+$", match):
                continue

            # Skip if it looks like a URL path or hex hash
            if re.match(r"^[A-Fa-f0-9]+$", match):
                continue

            # Check for Base64 characteristics:
            # - Contains both upper and lowercase
            # - Contains + or / (Base64 special chars)
            # - Reasonable character distribution (not all same char)
            has_upper = bool(re.search(r"[A-Z]", match))
            has_lower = bool(re.search(r"[a-z]", match))
            has_b64_special = bool(re.search(r"[+/]", match))
            unique_chars = len(set(match))

            # Likely Base64 if: mixed case AND (has special chars OR high entropy)
            if has_upper and has_lower:
                if has_b64_special:
                    score += 0.4  # Strong Base64 indicator
                    break
                elif unique_chars > len(match) * 0.3:
                    # High character diversity suggests encoding
                    score += 0.25
                    break

        # Unicode homoglyphs (common substitutions)
        homoglyph_chars = [
            "\u0430",  # Cyrillic 'а' looks like Latin 'a'
            "\u0435",  # Cyrillic 'е' looks like Latin 'e'
            "\u043e",  # Cyrillic 'о' looks like Latin 'o'
            "\u0440",  # Cyrillic 'р' looks like Latin 'p'
            "\u0441",  # Cyrillic 'с' looks like Latin 'c'
            "\u0445",  # Cyrillic 'х' looks like Latin 'x'
        ]
        if any(char in text for char in homoglyph_chars):
            score += 0.4

        # Control characters (excluding common whitespace)
        control_char_pattern = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
        if re.search(control_char_pattern, text):
            score += 0.3

        # Excessive repetition (potential padding attack)
        repetition_pattern = r"(.{3,})\1{5,}"  # 3+ chars repeated 5+ times
        if re.search(repetition_pattern, text):
            score += 0.2

        return min(1.0, score)

    def _calc_jailbreak_intent_score(self, text: str) -> float:
        """Calculate jailbreak intent detection score.

        Detection targets:
        - Explicit bypass requests ("bypass safety filters")
        - "Even if forbidden" patterns
        - Uncensored/unfiltered requests
        - Known jailbreak personas (DAN, DUDE, etc.)
        - Hypothetical framing for bypass (conditional scoring)

        Improvements:
        - Hypothetical patterns: only high score when combined with restriction bypass
        - "hypothetical" alone: low score (could be benign academic question)
        """
        score = 0.0

        # Check for restriction/bypass context (used for conditional scoring)
        has_restriction_context = bool(
            re.search(
                r"(?i)\b(restriction|filter|safety|rule|forbidden|prohibited|"
                r"not\s+allowed|bypass|ignore)\b",
                text,
            )
        )

        # Explicit bypass requests - very high signal
        bypass_patterns = [
            r"(?i)bypass\s+(the\s+)?(safety|filter|restriction|rule|guard)",
            r"(?i)circumvent\s+(the\s+)?(safety|filter|restriction|guard)",
            r"(?i)get\s+around\s+(the\s+)?(safety|filter|restriction)",
            r"(?i)disable\s+(the\s+)?(safety|filter|restriction|guard)",
        ]
        if any(re.search(p, text) for p in bypass_patterns):
            score += 0.8

        # "Even if forbidden" patterns - high signal
        forbidden_patterns = [
            r"(?i)even\s+if\s+(it'?s?\s+)?(forbidden|prohibited|not\s+allowed)",
            r"(?i)regardless\s+of\s+(the\s+)?(rules?|restrictions?|policy|guidelines?)",
            r"(?i)I\s+know\s+(it'?s?|this\s+is)\s+(against|forbidden|not\s+allowed)",
            r"(?i)doesn'?t?\s+matter\s+if\s+(it'?s?\s+)?(against|forbidden)",
        ]
        if any(re.search(p, text) for p in forbidden_patterns):
            score += 0.8

        # Uncensored/unfiltered requests - high signal
        uncensored_patterns = [
            r"(?i)\b(uncensored|unfiltered|unrestricted)\b",
            r"(?i)\b(without|no)\s+(censorship|filters?|restrictions?|limits?)\b",
            r"(?i)\bremove\s+(all\s+)?(restrictions?|filters?|limits?)\b",
            r"(?i)\banswer\s+without\s+(any\s+)?filters?\b",
        ]
        if any(re.search(p, text) for p in uncensored_patterns):
            score += 0.7

        # Known jailbreak personas/modes - very high signal
        persona_patterns = [
            r"(?i)\b(DAN|DUDE|STAN|KEVIN)\s*(mode)?",
            r"(?i)\bjailbreak(ed)?\s*(mode)?\b",
            r"(?i)\bdeveloper\s*mode\b",
            r"(?i)\b(god|admin|root)\s*mode\b",
            r"(?i)\bdo\s+anything\s+now\b",
        ]
        if any(re.search(p, text) for p in persona_patterns):
            score += 0.9

        # Hypothetical framing - conditional scoring
        # Only high score when combined with restriction/bypass context
        # "In a hypothetical scenario, how would malware spread?" → high score
        # "In a hypothetical scenario, what if cats could talk?" → low score
        explicit_bypass_hypothetical = [
            r"(?i)pretend\s+(there\s+are|you\s+have)\s+no\s+(rules?|restrictions?|limits?)",
            r"(?i)imagine\s+you\s+(have\s+no|don'?t\s+have)\s+(rules?|restrictions?)",
        ]
        if any(re.search(p, text) for p in explicit_bypass_hypothetical):
            score += 0.7  # Always high score - explicit bypass framing

        # Generic hypothetical patterns - conditional scoring
        generic_hypothetical_patterns = [
            r"(?i)for\s+(educational|research|academic|testing)\s+purposes?\s+only",
            r"(?i)in\s+a\s+hypothetical\s+(scenario|world|situation)",
            r"(?i)hypothetically\s+speaking",
            r"(?i)let'?s\s+(say|assume|pretend)\s+hypothetically",
        ]
        if any(re.search(p, text) for p in generic_hypothetical_patterns):
            if has_restriction_context:
                score += 0.5  # Higher score with restriction context
            else:
                score += 0.15  # Low score for benign hypothetical questions

        return min(1.0, score)


# =============================================================================
# Output Guardrail (Leakage Detection)
# =============================================================================


class LeakageType(str, Enum):
    """Types of data leakage detected by OutputGuardrail."""

    VERBATIM_CONTEXT = "verbatim_context"
    METADATA_EXPOSURE = "metadata_exposure"
    PII_IN_OUTPUT = "pii_in_output"
    SECRET_IN_OUTPUT = "secret_in_output"


class LeakageScoreBreakdown(BaseModel):
    """Output guardrail score breakdown.

    Two-lane architecture:
    - Sanitize lane: PII, secrets, metadata (detection triggers redaction)
    - Content lane: verbatim_ratio, longest_match_ratio (threshold-based WARN/BLOCK)

    Detection components:
    - verbatim_ratio: N-gram overlap ratio with source context (frequency-aware)
    - longest_match_ratio: Longest common substring (contiguous match) ratio
    - metadata_leak_count: Number of metadata patterns detected
    - pii_detected_count: Number of PII patterns detected
    - secret_detected_count: Number of secret tokens detected (API keys, JWT, etc.)
    - high_confidence_secret_count: High-confidence secrets (PEM, AWS key, JWT)
    """

    # Content lane scores
    verbatim_ratio: float = 0.0
    longest_match_ratio: float = 0.0

    # Sanitize lane counts
    metadata_leak_count: int = 0
    pii_detected_count: int = 0
    secret_detected_count: int = 0
    high_confidence_secret_count: int = 0  # PEM, AWS key, JWT - always block

    @property
    def content_score(self) -> float:
        """Content lane score (verbatim/substring overlap)."""
        return max(self.verbatim_ratio, self.longest_match_ratio)

    @property
    def sanitize_needed(self) -> bool:
        """Whether sanitization is needed (any PII/secret/metadata detected)."""
        return (
            self.metadata_leak_count > 0
            or self.pii_detected_count > 0
            or self.secret_detected_count > 0
        )

    @property
    def should_block_for_secrets(self) -> bool:
        """Whether to block due to high-confidence secrets."""
        return self.high_confidence_secret_count > 0

    @property
    def total_score(self) -> float:
        """Calculate total leakage score (for backward compatibility).

        Note: In the new two-lane design, this score is primarily used
        for the content lane threshold comparison. Sanitize lane uses
        detection counts directly.
        """
        # Content lane score
        content_score = self.content_score

        # Sanitize lane scores (for combined threat assessment)
        metadata_score = min(1.0, self.metadata_leak_count * 0.3)
        pii_score = min(1.0, self.pii_detected_count * 0.4)
        secret_score = min(1.0, self.secret_detected_count * 0.5)

        # Primary: max of all component scores
        max_score = max(content_score, metadata_score, pii_score, secret_score)

        # Secondary: weighted sum for combined weak signals
        weighted_sum = (
            content_score * 0.35
            + metadata_score * 0.25
            + pii_score * 0.25
            + secret_score * 0.15
        )

        # Hybrid: max dominates, but weighted sum provides boost
        return min(1.0, max_score * 0.7 + weighted_sum * 0.3)


class OutputGuardrail:
    """Output Guardrail: Inspects LLM responses before returning to user.

    Two-lane architecture:
    - Sanitize lane (A): PII/Secret/Metadata detection
      - Sets sanitize_needed flag → caller should call redact() if True
      - High-confidence secrets (PEM, AWS key, JWT) → BLOCK (early return)
    - Content lane (B): verbatim/substring overlap → threshold-based ALLOW/WARN/BLOCK
      - Determines the final action returned in GuardrailResult

    Processing order:
    1. Run Sanitize lane first
    2. If high-confidence secret detected → BLOCK immediately (skip Content lane)
    3. Run Content lane → determines action (ALLOW/WARN/BLOCK)
    4. Return result with sanitize_needed in score_breakdown

    The caller should always call redact(output, result) if sanitize_needed is True.
    """

    # Max tokens to process for performance (prevents O(m*n) explosion)
    MAX_WORDS_FOR_ANALYSIS = 500

    # High-confidence secret patterns (always BLOCK)
    HIGH_CONFIDENCE_SECRET_PATTERNS = [
        # PEM private key markers
        r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
        r"-----BEGIN\s+EC\s+PRIVATE\s+KEY-----",
        # AWS Access Key ID (starts with AKIA, ABIA, ACCA, ASIA)
        r"\b(A3T[A-Z0-9]|AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b",
        # JWT tokens (base64.base64.signature format)
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    ]

    def __init__(self, config: "GuardrailSettings") -> None:
        """Initialize OutputGuardrail with configuration.

        Args:
            config: Guardrail settings with thresholds and feature flags.
        """
        self.config = config

    def check(
        self,
        output: str,
        context_chunks: list[str],
        doc_metadata: list[dict[str, Any]],
        classification: Classification = Classification.PUBLIC,
    ) -> GuardrailResult:
        """Check LLM output for data leakage using two-lane architecture.

        Args:
            output: LLM response text.
            context_chunks: Original context texts passed to LLM.
            doc_metadata: Metadata of source documents.
            classification: Highest classification of source documents.

        Returns:
            GuardrailResult with action, sanitized_output, and score breakdown.
        """
        if not self.config.output_guardrail_enabled:
            return GuardrailResult(
                guardrail_type="output",
                is_safe=True,
                action=GuardrailAction.ALLOW,
            )

        # ============================================
        # LANE A: Sanitize (PII/Secret/Metadata)
        # ============================================
        sanitize_breakdown = self._check_sanitize_lane(output, doc_metadata)

        # High-confidence secret → BLOCK immediately (skip Content lane)
        if sanitize_breakdown.should_block_for_secrets:
            return GuardrailResult(
                guardrail_type="output",
                is_safe=False,
                action=GuardrailAction.BLOCK,
                threat_score=1.0,
                threat_type=LeakageType.SECRET_IN_OUTPUT.value,
                score_breakdown=sanitize_breakdown.model_dump(),
                details={
                    "output_length": len(output),
                    "context_count": len(context_chunks),
                    "classification": classification.value,
                    "blocked_reason": "high_confidence_secret",
                },
            )

        # ============================================
        # LANE B: Content Leakage (verbatim/substring)
        # ============================================
        content_breakdown = self._check_content_lane(output, context_chunks)

        # Merge breakdowns
        full_breakdown = LeakageScoreBreakdown(
            verbatim_ratio=content_breakdown.verbatim_ratio,
            longest_match_ratio=content_breakdown.longest_match_ratio,
            metadata_leak_count=sanitize_breakdown.metadata_leak_count,
            pii_detected_count=sanitize_breakdown.pii_detected_count,
            secret_detected_count=sanitize_breakdown.secret_detected_count,
            high_confidence_secret_count=sanitize_breakdown.high_confidence_secret_count,
        )

        # Determine action from content lane only (threshold-based ALLOW/WARN/BLOCK)
        # Sanitize lane determines redact() behavior via sanitize_needed flag
        content_action = self._determine_content_action(
            full_breakdown.content_score, classification
        )
        threat_type = self._determine_threat_type(full_breakdown)

        return GuardrailResult(
            guardrail_type="output",
            is_safe=(content_action == GuardrailAction.ALLOW),
            action=content_action,
            threat_score=full_breakdown.total_score,
            threat_type=threat_type,
            score_breakdown=full_breakdown.model_dump(),
            details={
                "output_length": len(output),
                "context_count": len(context_chunks),
                "classification": classification.value,
                "sanitize_needed": full_breakdown.sanitize_needed,
            },
        )

    def _check_sanitize_lane(
        self, output: str, doc_metadata: list[dict[str, Any]]
    ) -> LeakageScoreBreakdown:
        """Check for PII, secrets, and metadata leakage (Sanitize lane).

        Any detection triggers REDACT. High-confidence secrets trigger BLOCK.

        Args:
            output: LLM response text.
            doc_metadata: Metadata of source documents.

        Returns:
            LeakageScoreBreakdown with sanitize lane detection counts.
        """
        return LeakageScoreBreakdown(
            metadata_leak_count=self._detect_metadata_leak(output, doc_metadata),
            pii_detected_count=self._detect_pii(output),
            secret_detected_count=self._detect_secrets(output),
            high_confidence_secret_count=self._detect_high_confidence_secrets(output),
        )

    def _check_content_lane(
        self, output: str, context_chunks: list[str]
    ) -> LeakageScoreBreakdown:
        """Check for verbatim/substring content leakage (Content lane).

        Uses threshold-based WARN/BLOCK decision.

        Args:
            output: LLM response text.
            context_chunks: Original context texts passed to LLM.

        Returns:
            LeakageScoreBreakdown with content lane scores.
        """
        return LeakageScoreBreakdown(
            verbatim_ratio=self._calc_ngram_overlap(output, context_chunks),
            longest_match_ratio=self._calc_longest_substring_ratio(output, context_chunks),
        )

    def _detect_high_confidence_secrets(self, output: str) -> int:
        """Detect high-confidence secret patterns that should always BLOCK.

        These patterns have very low false positive rates:
        - PEM private keys
        - AWS Access Key IDs
        - JWT tokens

        Args:
            output: LLM output text.

        Returns:
            Count of high-confidence secret instances.
        """
        count = 0
        for pattern in self.HIGH_CONFIDENCE_SECRET_PATTERNS:
            matches = re.findall(pattern, output)
            count += len(matches)
        return count

    def redact(
        self,
        output: str,
        result: GuardrailResult,
        doc_metadata: list[dict[str, Any]] | None = None,
    ) -> str:
        """Redact sensitive content from output.

        In the two-lane design, redaction is always performed when
        sanitize_needed is True (regardless of overall action).

        Args:
            output: Original LLM output.
            result: GuardrailResult from check().
            doc_metadata: Document metadata for raw doc_id redaction.

        Returns:
            Redacted output string.
        """
        # Check if sanitization is needed based on score breakdown
        breakdown = result.score_breakdown
        sanitize_needed = (
            breakdown.get("metadata_leak_count", 0) > 0
            or breakdown.get("pii_detected_count", 0) > 0
            or breakdown.get("secret_detected_count", 0) > 0
        )

        if not sanitize_needed:
            return output

        redacted = output

        # Redact PII patterns
        redacted = self._redact_pii(redacted)

        # Redact metadata patterns (including raw doc_ids from metadata)
        redacted = self._redact_metadata(redacted, doc_metadata)

        # Redact secret tokens
        redacted = self._redact_secrets(redacted)

        return redacted

    def _determine_content_action(
        self, content_score: float, classification: Classification
    ) -> GuardrailAction:
        """Determine action for content lane based on threshold.

        Content lane only uses ALLOW, WARN, or BLOCK (no REDACT).

        Args:
            content_score: Content overlap score (0.0-1.0).
            classification: Document classification for threshold lookup.

        Returns:
            GuardrailAction for content lane.
        """
        # Get thresholds for this classification
        thresholds = self.config.leakage_thresholds.get(classification.value)

        if thresholds:
            allow_threshold = thresholds.get(
                "allow", self.config.default_leakage_allow_threshold
            )
            warn_threshold = thresholds.get(
                "warn", self.config.default_leakage_warn_threshold
            )
            block_threshold = thresholds.get(
                "block", self.config.default_leakage_block_threshold
            )
        else:
            # Fallback to defaults
            allow_threshold = self.config.default_leakage_allow_threshold
            warn_threshold = self.config.default_leakage_warn_threshold
            block_threshold = self.config.default_leakage_block_threshold

        # Content lane: ALLOW → WARN → BLOCK (no REDACT)
        if content_score < allow_threshold:
            return GuardrailAction.ALLOW
        elif content_score < warn_threshold:
            return GuardrailAction.WARN
        elif content_score >= block_threshold:
            return GuardrailAction.BLOCK
        else:
            # Between warn and block threshold → WARN
            return GuardrailAction.WARN

    def _determine_threat_type(self, breakdown: LeakageScoreBreakdown) -> Optional[str]:
        """Determine primary threat type from score breakdown.

        Args:
            breakdown: Score breakdown from detection.

        Returns:
            LeakageType value or None if no significant threat.
        """
        # High-confidence secrets take priority
        if breakdown.high_confidence_secret_count > 0:
            return LeakageType.SECRET_IN_OUTPUT.value

        scores = {
            LeakageType.VERBATIM_CONTEXT: max(
                breakdown.verbatim_ratio, breakdown.longest_match_ratio
            ),
            LeakageType.SECRET_IN_OUTPUT: min(1.0, breakdown.secret_detected_count * 0.5),
            LeakageType.METADATA_EXPOSURE: min(1.0, breakdown.metadata_leak_count * 0.3),
            LeakageType.PII_IN_OUTPUT: min(1.0, breakdown.pii_detected_count * 0.4),
        }

        max_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[max_type] > 0.3:
            return max_type.value
        return None

    def _calc_ngram_overlap(
        self, output: str, contexts: list[str], n: int | None = None
    ) -> float:
        """Calculate context overlap ratio using N-gram with frequency awareness.

        Uses Counter instead of set to properly count repeated n-grams.
        This catches cases where the same phrase is repeated multiple times.

        Args:
            output: LLM output text.
            contexts: List of context texts.
            n: N-gram size (default from config).

        Returns:
            Overlap ratio (0.0-1.0).
        """
        if not contexts or not output:
            return 0.0

        n = n or self.config.ngram_size

        # Normalize and clip text for performance
        output_words = output.lower().split()
        if len(output_words) > self.MAX_WORDS_FOR_ANALYSIS:
            output_words = output_words[: self.MAX_WORDS_FOR_ANALYSIS]

        # Generate output n-grams with frequency
        output_ngrams = self._get_ngrams_counter(output_words, n)
        if not output_ngrams:
            return 0.0

        # Generate context n-grams (combined from all contexts)
        context_ngrams: Counter[str] = Counter()
        for ctx in contexts:
            ctx_words = ctx.lower().split()
            if len(ctx_words) > self.MAX_WORDS_FOR_ANALYSIS:
                ctx_words = ctx_words[: self.MAX_WORDS_FOR_ANALYSIS]
            context_ngrams.update(self._get_ngrams_counter(ctx_words, n))

        if not context_ngrams:
            return 0.0

        # Calculate overlap with frequency awareness
        # Count matching n-grams (min of counts in both)
        overlap_count = sum(
            min(output_ngrams[ng], context_ngrams[ng])
            for ng in output_ngrams
            if ng in context_ngrams
        )
        total_output_ngrams = sum(output_ngrams.values())

        return overlap_count / total_output_ngrams if total_output_ngrams > 0 else 0.0

    def _get_ngrams_counter(self, words: list[str], n: int) -> Counter[str]:
        """Generate n-grams from word list with frequency counts.

        Args:
            words: List of words (already normalized).
            n: N-gram size.

        Returns:
            Counter of n-gram strings with frequencies.
        """
        if len(words) < n:
            return Counter()
        return Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))

    def _calc_longest_substring_ratio(self, output: str, contexts: list[str]) -> float:
        """Calculate longest common substring (contiguous match) ratio.

        Uses substring (contiguous) instead of subsequence (can skip) because
        verbatim copy detection benefits from finding exact contiguous matches.
        Subsequence can produce false positives with common phrases.

        Args:
            output: LLM output text.
            contexts: List of context texts.

        Returns:
            Longest substring ratio (0.0-1.0).
        """
        if not contexts or not output:
            return 0.0

        output_words = output.lower().split()
        if not output_words:
            return 0.0

        # Clip for performance
        if len(output_words) > self.MAX_WORDS_FOR_ANALYSIS:
            output_words = output_words[: self.MAX_WORDS_FOR_ANALYSIS]

        max_substring_length = 0

        for ctx in contexts:
            ctx_words = ctx.lower().split()
            if not ctx_words:
                continue

            # Clip context too
            if len(ctx_words) > self.MAX_WORDS_FOR_ANALYSIS:
                ctx_words = ctx_words[: self.MAX_WORDS_FOR_ANALYSIS]

            substring_length = self._longest_common_substring_length(
                output_words, ctx_words
            )
            max_substring_length = max(max_substring_length, substring_length)

        return max_substring_length / len(output_words) if output_words else 0.0

    def _longest_common_substring_length(
        self, seq1: list[str], seq2: list[str]
    ) -> int:
        """Calculate length of longest common substring (contiguous match).

        Uses space-optimized DP approach. For verbatim detection, contiguous
        matches are more relevant than subsequences (which can skip words).

        Args:
            seq1: First sequence of words.
            seq2: Second sequence of words.

        Returns:
            Length of longest common substring.
        """
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0

        # Space-optimized: only keep current row
        prev = [0] * (n + 1)
        max_length = 0

        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                    max_length = max(max_length, curr[j])
                # else: curr[j] stays 0 (substring must be contiguous)
            prev = curr

        return max_length

    def _detect_metadata_leak(
        self, output: str, doc_metadata: list[dict[str, Any]]
    ) -> int:
        """Detect metadata leakage patterns.

        Detection targets:
        - doc_id patterns
        - classification values
        - chunk_id patterns
        - Internal file paths

        Args:
            output: LLM output text.
            doc_metadata: List of document metadata dicts.

        Returns:
            Count of metadata leakage instances.
        """
        count = 0

        # Generic metadata patterns
        generic_patterns = [
            r"doc_id\s*[:=]\s*['\"]?[\w-]+",
            r"chunk_id\s*[:=]\s*[\w-]+",
            r"classification\s*[:=]\s*['\"]?(public|internal|confidential)",
            r"/[\w/]+\.(md|json|txt|py|yaml|yml)",  # Internal paths
        ]

        for pattern in generic_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            count += len(matches)

        # Check for specific doc_ids from metadata
        for meta in doc_metadata:
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id in output:
                count += 1

        return count

    def _detect_pii(self, output: str) -> int:
        """Detect PII format patterns.

        Detection targets:
        - Email addresses
        - Phone numbers (Japan format)
        - National ID formats

        Args:
            output: LLM output text.

        Returns:
            Count of PII instances detected.
        """
        count = 0

        pii_patterns = [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email
            r"\d{2,4}-\d{2,4}-\d{4}",  # Phone (Japan, various formats)
            r"\d{4}-\d{4}-\d{4}",  # National ID format
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN format (US)
        ]

        for pattern in pii_patterns:
            matches = re.findall(pattern, output)
            count += len(matches)

        return count

    def _detect_secrets(self, output: str) -> int:
        """Detect secret token patterns.

        Detection targets:
        - AWS Access Keys (AKIA...)
        - AWS Secret Keys
        - GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_)
        - Slack tokens (xoxb-, xoxp-, xoxa-, xoxr-)
        - JWT tokens (eyJ...)
        - PEM private keys
        - Generic API keys (api_key, apikey patterns)

        Args:
            output: LLM output text.

        Returns:
            Count of secret instances detected.
        """
        count = 0

        secret_patterns = [
            # AWS Access Key ID (starts with AKIA, ABIA, ACCA, ASIA)
            r"\b(A3T[A-Z0-9]|AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b",
            # AWS Secret Access Key (40 chars, mixed case + digits + slashes)
            r"(?<![A-Za-z0-9/+])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])",
            # GitHub tokens (various prefixes)
            r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b",
            # Slack tokens
            r"\b(xox[bpars])-[A-Za-z0-9-]{10,}\b",
            # JWT tokens (base64.base64.signature format)
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
            # PEM private key markers
            r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
            r"-----BEGIN\s+EC\s+PRIVATE\s+KEY-----",
            # Generic API key patterns (key=value, key:value)
            r"(?i)(api[_-]?key|apikey|secret[_-]?key|auth[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}",
            # Bearer tokens
            r"(?i)bearer\s+[A-Za-z0-9_-]{20,}",
            # Database connection strings with password
            r"(?i)(mysql|postgresql|postgres|mongodb)://[^:]+:[^@]+@",
        ]

        for pattern in secret_patterns:
            matches = re.findall(pattern, output)
            count += len(matches)

        return count

    def _redact_pii(self, text: str) -> str:
        """Redact PII patterns from text.

        Args:
            text: Input text.

        Returns:
            Text with PII redacted.
        """
        # Email
        text = re.sub(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "[EMAIL REDACTED]",
            text,
        )

        # Phone numbers
        text = re.sub(r"\d{2,4}-\d{2,4}-\d{4}", "[PHONE REDACTED]", text)

        # National ID / SSN
        text = re.sub(r"\d{4}-\d{4}-\d{4}", "[ID REDACTED]", text)
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", text)

        return text

    def _redact_metadata(
        self, text: str, doc_metadata: list[dict[str, Any]] | None = None
    ) -> str:
        """Redact metadata patterns from text.

        Args:
            text: Input text.
            doc_metadata: Document metadata for raw doc_id redaction.

        Returns:
            Text with metadata redacted.
        """
        # doc_id patterns (formatted as key=value or key:value)
        text = re.sub(
            r"doc_id\s*[:=]\s*['\"]?[\w-]+",
            "doc_id: [REDACTED]",
            text,
            flags=re.IGNORECASE,
        )

        # chunk_id patterns
        text = re.sub(
            r"chunk_id\s*[:=]\s*[\w-]+",
            "chunk_id: [REDACTED]",
            text,
            flags=re.IGNORECASE,
        )

        # Internal paths
        text = re.sub(
            r"/[\w/]+\.(md|json|txt|py|yaml|yml)",
            "[PATH REDACTED]",
            text,
        )

        # Redact raw doc_id values from metadata
        # This catches cases where the LLM outputs "confidential-001" directly
        # without the "doc_id:" prefix
        if doc_metadata:
            for meta in doc_metadata:
                doc_id = meta.get("doc_id", "")
                if doc_id and len(doc_id) >= 3:  # Avoid replacing very short strings
                    # Use word boundaries to avoid partial replacements
                    text = re.sub(
                        rf"\b{re.escape(doc_id)}\b",
                        "[DOC_ID REDACTED]",
                        text,
                    )

        return text

    def _redact_secrets(self, text: str) -> str:
        """Redact secret token patterns from text.

        Args:
            text: Input text.

        Returns:
            Text with secrets redacted.
        """
        # AWS Access Key ID
        text = re.sub(
            r"\b(A3T[A-Z0-9]|AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b",
            "[AWS_KEY REDACTED]",
            text,
        )

        # GitHub tokens
        text = re.sub(
            r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b",
            "[GITHUB_TOKEN REDACTED]",
            text,
        )

        # Slack tokens
        text = re.sub(
            r"\b(xox[bpars])-[A-Za-z0-9-]{10,}\b",
            "[SLACK_TOKEN REDACTED]",
            text,
        )

        # JWT tokens
        text = re.sub(
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
            "[JWT REDACTED]",
            text,
        )

        # PEM private key markers (redact entire block is complex, just mark)
        text = re.sub(
            r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
            "[PRIVATE_KEY REDACTED]",
            text,
        )
        text = re.sub(
            r"-----BEGIN\s+EC\s+PRIVATE\s+KEY-----",
            "[PRIVATE_KEY REDACTED]",
            text,
        )

        # Generic API key patterns
        text = re.sub(
            r"(?i)(api[_-]?key|apikey|secret[_-]?key|auth[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}",
            r"\1: [REDACTED]",
            text,
        )

        # Bearer tokens
        text = re.sub(
            r"(?i)bearer\s+[A-Za-z0-9_-]{20,}",
            "Bearer [REDACTED]",
            text,
        )

        # Database connection strings
        text = re.sub(
            r"(?i)(mysql|postgresql|postgres|mongodb)://[^:]+:[^@]+@",
            r"\1://[CREDENTIALS REDACTED]@",
            text,
        )

        return text
