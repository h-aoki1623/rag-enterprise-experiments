"""Guardrails module for RAG system security.

This module provides defensive layers for LLM application inputs and outputs:
- InputGuardrail: Detects prompt injection attacks before LLM processing
- OutputGuardrail: Detects PII/secrets/metadata for redaction (defense-in-depth)

Architecture:
    User Input → [InputGuardrail] → LLM → [OutputGuardrail] → Response
                      ↓                         ↓
                Injection detection        Detection for redaction
                Block/Warn decision        (RBAC handles authorization)
"""

import re
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
            score += 0.9  # Increased for better block rate
        elif role_marker_count == 2:
            # Paired markers (open/close) are suspicious
            score += 0.8  # Increased for better block rate
        elif role_marker_count == 1:
            # Single marker could be accidental (chat log paste)
            score += 0.3  # Slightly increased from 0.2

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
                score += 0.4  # Increased from 0.3 for better block rate
                break

        # End of prompt markers - higher score for explicit boundary manipulation
        end_patterns = [
            r"(?i)---\s*end\s+(of\s+)?(system\s+)?(prompt|instructions?)\s*---",
            r"(?i)===\s*end\s+(of\s+)?(system\s+)?(prompt|instructions?)\s*===",
            r"(?i)\*\*\*\s*end\s+(of\s+)?(system\s+)?(prompt|instructions?)\s*\*\*\*",
        ]
        for pattern in end_patterns:
            if re.search(pattern, text):
                score += 0.8  # Increased from 0.5 for better block rate
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


class LeakageScoreBreakdown(BaseModel):
    """Output guardrail score breakdown.

    Detection-only architecture for redaction:
    - RBAC already ensures users only see documents they have access to
    - Verbatim content from authorized documents is not a security concern
    - Sanitize lane focuses on PII/Secret/Metadata detection for redaction

    Detection components:
    - metadata_leak_count: Number of metadata patterns detected
    - pii_detected_count: Number of PII patterns detected
    - secret_detected_count: Number of secret tokens detected (API keys, JWT, etc.)
    - high_confidence_secret_count: High-confidence secrets (PEM, AWS key, JWT)
    """

    # Sanitize lane counts
    metadata_leak_count: int = 0
    pii_detected_count: int = 0
    secret_detected_count: int = 0
    high_confidence_secret_count: int = 0

    @property
    def sanitize_needed(self) -> bool:
        """Whether sanitization is needed (any PII/secret/metadata detected)."""
        return (
            self.metadata_leak_count > 0
            or self.pii_detected_count > 0
            or self.secret_detected_count > 0
        )


class OutputGuardrail:
    """Output Guardrail: Inspects LLM responses before returning to user.

    Detection-only architecture (no blocking):
    - RBAC already ensures users only see documents they have access to
    - Output Guardrail cannot prevent access that RBAC already allows
    - Redaction is a defense-in-depth measure (avoid exposure in logs, screenshots)

    Processing:
    1. Detect PII/Secret/Metadata patterns
    2. Set sanitize_needed flag if any detected
    3. Always return ALLOW - never block

    The caller should always call redact(output, result) if sanitize_needed is True.
    """

    # Max tokens to process for performance (prevents O(m*n) explosion)
    MAX_WORDS_FOR_ANALYSIS = 500

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
        """Check LLM output for patterns requiring redaction.

        Args:
            output: LLM response text.
            context_chunks: Original context texts passed to LLM.
            doc_metadata: Metadata of source documents.
            classification: Highest classification of source documents.

        Returns:
            GuardrailResult with action and score breakdown.
        """
        if not self.config.output_guardrail_enabled:
            return GuardrailResult(
                guardrail_type="output",
                is_safe=True,
                action=GuardrailAction.ALLOW,
            )

        # ============================================
        # Detect PII/Secret/Metadata for redaction
        # ============================================
        # Note: Output Guardrail never blocks - only detects for redaction.
        # RBAC already ensures users only see documents they have access to,
        # so blocking would be redundant. Redaction is a defense-in-depth
        # measure to avoid accidental exposure in chat logs, screenshots, etc.
        breakdown = LeakageScoreBreakdown(
            metadata_leak_count=self._detect_metadata_leak(output, doc_metadata),
            pii_detected_count=self._detect_pii(output),
            secret_detected_count=self._detect_secrets(output),
            high_confidence_secret_count=self._detect_high_confidence_secrets(output),
        )

        # Always ALLOW - caller should call redact() if sanitize_needed is True
        return GuardrailResult(
            guardrail_type="output",
            is_safe=True,
            action=GuardrailAction.ALLOW,
            score_breakdown=breakdown.model_dump(),
            details={
                "output_length": len(output),
                "context_count": len(context_chunks),
                "classification": classification.value,
                "sanitize_needed": breakdown.sanitize_needed,
            },
        )

    def _detect_high_confidence_secrets(self, output: str) -> int:
        """Detect high-confidence secret patterns for redaction.

        These patterns have very low false positive rates:
        - PEM private keys
        - AWS Access Key IDs
        - JWT tokens

        Note: Detection triggers redaction, not blocking. RBAC handles
        access control; this is defense-in-depth for accidental exposure.

        Args:
            output: LLM output text.

        Returns:
            Count of high-confidence secret instances.
        """
        patterns = [
            # PEM private key markers
            r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
            r"-----BEGIN\s+EC\s+PRIVATE\s+KEY-----",
            # AWS Access Key ID (starts with AKIA, ABIA, ACCA, ASIA)
            r"\b(A3T[A-Z0-9]|AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b",
            # JWT tokens (base64.base64.signature format)
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
        ]

        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, output)
            count += len(matches)
        return count

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

    def redact(
        self,
        output: str,
        result: GuardrailResult,
        doc_metadata: list[dict[str, Any]] | None = None,
    ) -> str:
        """Redact sensitive content from output.

        Redaction is performed when sanitize_needed is True.

        Args:
            output: Original LLM output.
            result: GuardrailResult from check().
            doc_metadata: Document metadata for raw doc_id redaction.

        Returns:
            Redacted output string.
        """
        # Reconstruct breakdown object to use its sanitize_needed property
        breakdown = LeakageScoreBreakdown(**result.score_breakdown)

        if not breakdown.sanitize_needed:
            return output

        redacted = output

        # Redact each category if detected
        if breakdown.metadata_leak_count > 0:
            redacted = self._redact_metadata(redacted, doc_metadata)
        if breakdown.pii_detected_count > 0:
            redacted = self._redact_pii(redacted)
        if breakdown.secret_detected_count > 0:
            redacted = self._redact_secrets(redacted)

        return redacted

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
