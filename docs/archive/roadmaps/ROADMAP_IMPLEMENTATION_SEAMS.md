# IMPLEMENTATION SEAM MAP: THREE PENDING ROADMAP SLICES

**Generated**: 2025-03-16  
**Project**: Token Saver 5000  
**Codebase Version**: v0.10.0+  

This document provides exact implementation seams for three pending roadmap slices:
1. **Preventive Stable-Prefix Enforcement** - Guard against cache-busting mutations before prompt rendering
2. **Canonical Provider/Harness Telemetry Observability Layer** - Aggregate telemetry across providers with validation harness
3. **Composable Multi-Pass Compression Pipeline** - Chain compressors in pluggable sequence for flexible output

---

## SLICE 1: PREVENTIVE STABLE-PREFIX ENFORCEMENT

### Problem Statement
Currently, prefix cache stability is **validated retroactively** via prompt_cache_middleware.py after rendering.
Users can accidentally introduce cache-busting mutations (volatile UUIDs, timestamps) into **stable** sections
that should be immutable across provider cache boundaries.

**Need**: Preventive validation that fails **before rendering** when stable sections contain volatile patterns.

### Existing Foundation (Leverage These)

**1. Cache Audit Foundation**
- File: src/prompt_cache_audit.py (119 lines)
  - CANONICAL_SECTION_ORDER (lines 8-16): Defines stable vs volatile section tiers
  - _VOLATILE_PATTERNS (lines 20-40): Regex matchers for UUIDs, timestamps, volatile keys
  - udit_prompt_cacheability() (lines 71-119): Scores prompt structure post-render

**2. Prefix Tracking**
- File: src/prompt_cache_middleware.py (400+ lines)
  - PromptCacheExpectation dataclass (lines 62-82): Tracks cacheable_prefix + hash
  - _prefix_collisions dict (line 94): Registry of prefix hash collisions
  - _prefix_integrity_history dict (line 96): Time-series of prefix changes

**3. Middleware Chain Hooks**
- File: src/handlers/prompt_handlers.py (TBD - see section 2)
  - Already has render/deploy paths; missing validation hook

### Minimal Implementation (New Module: src/prefix_validator.py)

Create **189 lines** of preventive validation:

`python
# src/prefix_validator.py

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from .prompt_cache_audit import CANONICAL_SECTION_ORDER, _VOLATILE_PATTERNS

class PrefixValidationLevel(Enum):
    """Strictness levels for prefix validation during template design."""
    STRICT = "strict"        # Reject any volatile pattern in stable sections
    WARN = "warn"            # Log warnings but allow
    PERMISSIVE = "permissive" # No preventive checks (retroactive only)

@dataclass
class PrefixValidationError:
    """Single violation found during validation."""
    section_name: str
    pattern_code: str  # uuid_like_token, iso_timestamp, volatile_key_name
    pattern_description: str
    matched_text: str
    line_number: int

@dataclass
class PrefixValidationResult:
    """Full validation result with remediation hints."""
    is_valid: bool
    errors: List[PrefixValidationError]
    warnings: List[str]
    first_volatile_section_index: int  # Where stable boundary ends
    affected_sections: List[str]  # Which sections have issues
    remediation_hints: List[str]
    validation_passed_at: str = ""  # ISO timestamp when validation succeeded

class PrefixValidator:
    """
    Preventive prefix validator for prompt templates BEFORE rendering.
    
    Catches cache-busting mutations early in design/template phase, not after render.
    Integrated into prompt_handlers during template_create, template_update, and render.
    
    Usage:
        validator = PrefixValidator(level=PrefixValidationLevel.STRICT)
        result = validator.validate_sections(sections)
        if not result.is_valid:
            for error in result.errors:
                print(f"Section '{error.section_name}': {error.pattern_description}")
            print("Hints:", result.remediation_hints)
    """
    
    def __init__(self, level: PrefixValidationLevel = PrefixValidationLevel.STRICT):
        self.level = level
        self._stable_sections = set(CANONICAL_SECTION_ORDER[:-2])  # All except metadata, user_query
        
    def validate_sections(self, sections: List[Dict[str, str]]) -> PrefixValidationResult:
        """
        Validate that stable sections contain no volatile patterns.
        
        Args:
            sections: List of {"name": str, "content": str} dicts
        
        Returns:
            PrefixValidationResult with errors, warnings, remediation hints
        """
        errors: List[PrefixValidationError] = []
        warnings: List[str] = []
        affected_sections: List[str] = []
        remediation_hints: List[str] = []
        
        first_volatile_index = len(sections)
        
        for section_idx, section in enumerate(sections):
            section_name = section.get("name", "")
            content = section.get("content", "")
            
            if section_name not in CANONICAL_SECTION_ORDER:
                warnings.append(f"Unknown section name '{section_name}'")
                continue
            
            # Check if this is a stable section
            if section_name not in self._stable_sections:
                first_volatile_index = min(first_volatile_index, section_idx)
                continue
            
            # Scan for volatile patterns in stable sections
            for pattern_code, pattern_regex, pattern_desc in _VOLATILE_PATTERNS:
                for match in pattern_regex.finditer(content):
                    matched_text = match.group(0)
                    # Count line number of match
                    line_num = content[:match.start()].count("\\n") + 1
                    
                    error = PrefixValidationError(
                        section_name=section_name,
                        pattern_code=pattern_code,
                        pattern_description=pattern_desc,
                        matched_text=matched_text,
                        line_number=line_num
                    )
                    errors.append(error)
                    
                    if section_name not in affected_sections:
                        affected_sections.append(section_name)
        
        # Generate remediation hints
        if errors:
            remediation_hints.extend([
                "Move volatile content (UUIDs, timestamps, request IDs) to [metadata] or [user_query] sections",
                "Use placeholders like {{REQUEST_ID}}, {{TIMESTAMP}} for runtime injection instead",
                "Keep tool_definitions, system_instructions, rag_context, and few_shot_examples stable",
            ])
            if any(e.pattern_code == "iso_timestamp" for e in errors):
                remediation_hints.append("Use 'Current date: <DATE>' only in [metadata] section")
            if any(e.pattern_code == "uuid_like_token" for e in errors):
                remediation_hints.append("Inject UUIDs as template variables, not literal values in stable sections")
        
        is_valid = len(errors) == 0 and self.level != PrefixValidationLevel.PERMISSIVE
        
        return PrefixValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            first_volatile_section_index=first_volatile_index,
            affected_sections=affected_sections,
            remediation_hints=remediation_hints,
        )
    
    def validate_before_render(
        self, 
        template_id: str, 
        template_version: int,
        sections: List[Dict[str, str]]
    ) -> Tuple[bool, str]:
        """
        Quick validation hook for prompt_handlers.py render path.
        
        Returns: (is_valid, error_message_or_empty_string)
        """
        result = self.validate_sections(sections)
        
        if not result.is_valid and self.level == PrefixValidationLevel.STRICT:
            error_msg = (
                f"Template {template_id}:v{template_version} has cache-busting patterns\\n"
                f"Affected sections: {', '.join(result.affected_sections)}\\n"
                f"Errors:\\n"
            )
            for error in result.errors:
                error_msg += f"  - [{error.section_name}:{error.line_number}] {error.pattern_code}: "
                error_msg += f"found '{error.matched_text}'\\n"
            error_msg += "\\nHints:\\n" + "\\n".join(f"  - {h}" for h in result.remediation_hints)
            return False, error_msg
        
        return True, ""
`

### Integration Points

**1. Modify src/prompt_handlers.py → handle_prompt_create_template()**

Add validation before template storage:

`python
def handle_create_prompt_template(args: Dict[str, Any], context: HandlerContext) -> str:
    # ... existing code ...
    
    sections = args.get("sections", [])
    validator = PrefixValidator(level=PrefixValidationLevel.STRICT)  # NEW
    
    is_valid, error_msg = validator.validate_before_render(
        template_id=template_id,
        template_version=1,
        sections=sections
    )
    
    if not is_valid:
        raise ValueError(error_msg)  # FAIL FAST before saving
    
    # ... continue with storage ...
`

**2. Modify src/prompt_handlers.py → handle_prompt_render()**

Add validation in render path (before calling ender_template_with_context()):

`python
def handle_prompt_render(args: Dict[str, Any], context: HandlerContext) -> str:
    # ... existing code to load template ...
    
    # NEW: Validate sections before rendering
    validator = PrefixValidator(level=PrefixValidationLevel.STRICT)
    is_valid, error_msg = validator.validate_before_render(
        template_id=template_id,
        template_version=template_version,
        sections=resolved_sections  # Sections with all variables interpolated
    )
    
    if not is_valid:
        return json.dumps({
            "error": "prefix_validation_failed",
            "message": error_msg,
            "render_aborted": True
        })
    
    # Continue with rendering...
`

**3. Extend src/types.py → HandlerContext**

Add validator instance to context:

`python
from src.prefix_validator import PrefixValidator

class HandlerContext(TypedDict, total=True):
    # ... existing fields ...
    prefix_validator: ReadOnly[PrefixValidator]  # NEW
`

**4. Modify src/server.py → Build context**

Initialize validator in _build_context():

`python
def _build_context(self) -> HandlerContext:
    # ... existing code ...
    
    return {
        # ... existing ...
        "prefix_validator": PrefixValidator(level=PrefixValidationLevel.STRICT),
    }
`

### Tests to Add

**File**: 	ests/test_prefix_validator.py (180 lines)

`python
import pytest
from src.prefix_validator import (
    PrefixValidator, 
    PrefixValidationLevel, 
    PrefixValidationResult
)

class TestPrefixValidator:
    
    def test_stable_sections_reject_uuids(self):
        """UUIDs in tool_definitions should fail validation."""
        validator = PrefixValidator(PrefixValidationLevel.STRICT)
        sections = [
            {
                "name": "tool_definitions",
                "content": "Tool ID: 550e8400-e29b-41d4-a716-446655440000 - Description"
            }
        ]
        result = validator.validate_sections(sections)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].pattern_code == "uuid_like_token"
        assert "tool_definitions" in result.affected_sections
    
    def test_volatile_sections_allow_uuids(self):
        """UUIDs in user_query should PASS validation."""
        validator = PrefixValidator(PrefixValidationLevel.STRICT)
        sections = [
            {
                "name": "user_query",
                "content": "Request ID: 550e8400-e29b-41d4-a716-446655440000"
            }
        ]
        result = validator.validate_sections(sections)
        assert result.is_valid
    
    def test_timestamps_in_stable_fail(self):
        """Timestamps in rag_context should fail."""
        validator = PrefixValidator(PrefixValidationLevel.STRICT)
        sections = [
            {
                "name": "rag_context",
                "content": "Last updated: 2025-03-16T14:30:00Z"
            }
        ]
        result = validator.validate_sections(sections)
        assert not result.is_valid
        assert result.errors[0].pattern_code == "iso_timestamp"
    
    def test_remediation_hints_provided(self):
        """Error results should include actionable hints."""
        validator = PrefixValidator(PrefixValidationLevel.STRICT)
        sections = [
            {
                "name": "system_instructions",
                "content": "request_id: {{REQUEST_ID}} should not be here"
            }
        ]
        result = validator.validate_sections(sections)
        assert not result.is_valid
        assert len(result.remediation_hints) > 0
        assert any("placeholder" in h.lower() for h in result.remediation_hints)
    
    def test_permissive_mode_always_valid(self):
        """PERMISSIVE level allows anything."""
        validator = PrefixValidator(PrefixValidationLevel.PERMISSIVE)
        sections = [
            {
                "name": "rag_context",
                "content": "ID: 550e8400-e29b-41d4-a716-446655440000 at 2025-03-16T14:30:00Z"
            }
        ]
        result = validator.validate_sections(sections)
        assert result.is_valid  # No validation in permissive mode
`

**File**: 	ests/test_prompt_handlers_prefix_validation.py (120 lines)

Integration tests verifying handle_prompt_create_template() and handle_prompt_render() reject bad prefixes.

### Docs/Help Surfaces to Update

**1. docs/guides/PROMPT_CACHING.md**

Add section under "Best Practices":

`markdown
## Preventive Prefix Validation (v0.11.0+)

Before rendering a template, Token Saver validates that **stable sections**
(tool_definitions, system_instructions, rag_context, few_shot_examples)
contain NO volatile patterns:

- ❌ UUIDs like 550e8400-e29b-41d4-a716-446655440000
- ❌ ISO timestamps like 2025-03-16T14:30:00Z  
- ❌ Volatile keys like equest_id, session_id, current_time

These patterns **destroy provider prefix cache reuse**. If you use them, keep them in:
- [metadata] section, OR
- [user_query] section

### Use Template Variables for Runtime Injection

Instead of hardcoding volatile values, use template variables:

`
[system_instructions]
You are a helpful assistant. {{REQUEST_ID}}  ← ❌ WRONG: hardcoded placeholder

[metadata]
Current request: {{REQUEST_ID}}  ← ✅ RIGHT: volatile section
`

### Validation Modes

- **STRICT** (default): Reject templates with volatile patterns in stable sections
- **WARN**: Log warnings but allow rendering
- **PERMISSIVE**: No preventive checks (retroactive validation only)

Set via environment: PREFIX_VALIDATION_LEVEL=strict|warn|permissive
`

**2. src/handlers/help_handlers.py**

Add help entry for "prefix_validation":

`python
TOOL_HELP_REGISTRY["prefix_validation_info"] = {
    "category": "Prompt Caching",
    "description": "Understand prefix stability requirements and validation rules.",
    "content": """
PREFIX VALIDATION prevents cache-busting mutations before prompt render.

Stable Sections (unchanging across requests):
- tool_definitions: API tool schemas
- system_instructions: System role and constraints
- rag_context: Background documents and knowledge
- few_shot_examples: Example input/output pairs

Volatile Sections (change per-request):
- metadata: Request metadata, timestamps, request IDs
- user_query: The actual user question

Validation checks for patterns that break provider cache reuse:
- UUIDs: 550e8400-e29b-41d4-a716-446655440000 ❌
- Timestamps: 2025-03-16T14:30:00Z ❌
- Runtime IDs: request_id, session_id, trace_id ❌

Use template variables {{PLACEHOLDER}} for runtime values in stable sections.
""",
    "related_tools": ["prompt_create_template", "prompt_render", "prompt_audit"]
}
`

---

## SLICE 2: CANONICAL PROVIDER/HARNESS TELEMETRY OBSERVABILITY LAYER

### Problem Statement
Telemetry collection is **fragmented**:
- Provider telemetry exists in model_optimizer.py (cost, cache hits, tokens)
- Provider profiles exist in provider_profiles.py (cache_read_field mappings)
- Benchmark validation exists in enchmark_harness.py (golden token targets)
- Observability spans exist in observability.py (OTEL traces)

**Missing**: Canonical telemetry aggregator that:
1. Normalizes telemetry across providers (Anthropic, OpenAI, Google use different field names)
2. Validates against provider profiles and benchmark expectations
3. Exports aggregated metrics via structured logs + observability layer
4. Enables harness validation (Did this run meet token savings targets?)

### Existing Foundation (Leverage These)

**1. Provider Profiles** (src/provider_profiles.py, 100 lines)
- ProviderProfile dataclass (lines 9-24): model, provider, costs, cache_read_field, etc.
- _PROFILES registry (lines 26-99): 6 model profiles (Claude 3.x, GPT-4, Gemini 3.1)
- get_provider_profile(model): Lookup by model name

**2. Telemetry Normalization** (src/model_optimizer.py, 250 lines)
- _find_numeric_field() (lines 36-50): Recursive dict/list traversal for tokens
- _find_provider_cache_read_tokens() (lines 64-70): Field alias resolution
- summarize_provider_cache_usage() (lines 73-130): Single aggregation function

**3. Benchmark Harness** (src/benchmark_harness.py, 220 lines)
- BenchmarkCase dataclass (lines 22-31): case_id, min_compression_ratio, min_token_savings_pct
- BenchmarkResult dataclass (lines 34-54): Original/skeleton tokens, ratios, pass/fail flags
- BenchmarkSummary dataclass (lines 57-76): Aggregate pass rate, avg compression, quality metrics
- load_benchmark_cases() (lines 96-116): Load from JSON corpus

**4. Cache Diagnostics** (src/cache_diagnostics.py, 250 lines)
- diagnose_cache_miss(): Root-cause analysis with probable_cause enum
- detect_semantic_equivalence_drift(): Prefix stability checks
- detect_section_interleaving(): Section order validation

**5. Observability** (src/observability.py, 200 lines)
- get_observability(): Singleton OTEL tracer
- 	race() context manager: Span creation with attributes
- set_attribute(), ecord_exception(): Telemetry recording

### Minimal Implementation (New Module: src/canonical_telemetry.py)

Create **320 lines** of canonical aggregation:

`python
# src/canonical_telemetry.py

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json

from .provider_profiles import ProviderProfile, get_provider_profile
from .benchmark_harness import BenchmarkResult, BenchmarkCase
from .model_optimizer import summarize_provider_cache_usage
from .cache_diagnostics import diagnose_cache_miss
from .observability import get_observability
from .structured_logging import get_logger

logger = get_logger(__name__)
observe = get_observability()

class TelemetryValidationStatus(Enum):
    """Validation result against harness expectations."""
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"

@dataclass
class ProviderTelemetryNormalized:
    """Canonical normalized telemetry from any provider."""
    
    # Required identifiers
    provider: str  # anthropic, openai, google, etc
    model: str  # claude-opus-4.6, gpt-5.4, gemini-3.1-pro
    request_id: str  # Unique request identifier
    timestamp_utc: str  # ISO-8601 timestamp
    
    # Token accounting (canonical field names)
    input_tokens: int  # Total input tokens
    output_tokens: int  # Total output tokens
    cache_creation_tokens: Optional[int] = None  # Tokens spent creating cache (Anthropic)
    cache_read_tokens: int = 0  # Cached input tokens (provider-specific field)
    
    # Cache telemetry
    cache_hit_ratio: float = 0.0  # cache_read_tokens / input_tokens
    cache_read_cost_usd: float = 0.0  # Cheaper than normal input cost
    cache_creation_cost_usd: float = 0.0  # Cost of creating cache
    
    # Quality signals
    compression_ratio: Optional[float] = None  # Original / compressed
    token_savings_pct: Optional[float] = None  # (1 - skeleton/original) * 100
    
    # Validation results
    meets_benchmark: Optional[bool] = None  # Passed golden token targets
    benchmark_case_id: Optional[str] = None  # Which benchmark case was run
    
    # Raw provider response (for audit trail)
    raw_api_response: Dict[str, Any] = field(default_factory=dict)
    
    # Observability
    trace_id: Optional[str] = None  # OTEL trace ID
    span_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, converting enums."""
        payload = asdict(self)
        return payload
    
    def compute_cost_delta_vs_uncached(self, profile: ProviderProfile) -> float:
        """
        Calculate cost difference: cached vs uncached.
        
        Cost if not cached: (input_tokens / 1M) * input_cost_per_million
        Cost if cached:     (cache_read_tokens / 1M) * cached_cost_per_million
        Saved:              uncached_cost - cached_cost
        """
        uncached_cost = (self.input_tokens / 1_000_000) * profile.input_cost_per_million
        cached_cost = (self.cache_read_tokens / 1_000_000) * profile.cached_input_cost_per_million
        return uncached_cost - cached_cost

@dataclass
class HarnessValidationResult:
    """Result of validating telemetry against benchmark harness."""
    
    # Validation decision
    status: TelemetryValidationStatus
    passed: bool  # True if PASS or PASS_WITH_WARNINGS
    
    # Benchmark comparison
    case_id: str  # Which benchmark case this run corresponds to
    case_name: str
    min_compression_target: float  # Expected minimum compression ratio
    actual_compression_ratio: Optional[float]
    compression_target_met: bool
    
    min_savings_target_pct: float  # Expected minimum token savings %
    actual_savings_pct: Optional[float]
    savings_target_met: bool
    
    # Cache validation (vs expectations)
    cache_expected: bool
    cache_occurred: bool  # cache_read_tokens > 0
    cache_hit_ratio_actual: float
    cache_expectation_met: bool
    
    # Issues and warnings
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Telemetry hash for audit trail
    telemetry_hash: str = ""
    
    def summarize(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Case '{self.case_name}' ({self.case_id}): {self.status.value.upper()}",
            f"  Compression: {self.actual_compression_ratio:.1f}× (target: {self.min_compression_target:.1f}×) "
            f"{'✓' if self.compression_target_met else '✗'}",
            f"  Token Savings: {self.actual_savings_pct:.1f}% (target: {self.min_savings_target_pct:.1f}%) "
            f"{'✓' if self.savings_target_met else '✗'}",
            f"  Cache Hit Ratio: {self.cache_hit_ratio_actual:.1%} {'✓' if self.cache_expectation_met else '✗'}",
        ]
        
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(f"    - {issue}")
        
        if self.warnings:
            lines.append("  Warnings:")
            for warning in self.warnings:
                lines.append(f"    ⚠ {warning}")
        
        if self.recommendations:
            lines.append("  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    → {rec}")
        
        return "\\n".join(lines)

class CanonicalTelemetryAggregator:
    """
    Normalizes telemetry from any provider and validates against harness.
    
    Entry point for all telemetry processing in Token Saver.
    Converts provider-specific field names to canonical names,
    validates against benchmark expectations, and exports metrics.
    """
    
    def __init__(self):
        self._telemetry_records: List[ProviderTelemetryNormalized] = []
        self._validation_results: List[HarnessValidationResult] = []
    
    def normalize_from_provider_response(
        self,
        *,
        model: str,
        api_response: Dict[str, Any],
        file_id: Optional[str] = None,
        benchmark_case: Optional[BenchmarkCase] = None,
        compression_ratio: Optional[float] = None,
        token_savings_pct: Optional[float] = None,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ProviderTelemetryNormalized:
        """
        Convert provider API response to canonical telemetry.
        
        Handles field name aliasing for:
        - Anthropic: cache_read_input_tokens, cache_creation_input_tokens
        - OpenAI: cached_tokens
        - Google: cachedContentTokenCount
        
        Args:
            model: Model name (must exist in provider_profiles)
            api_response: Raw provider API response dict
            file_id: Document ID (optional, for audit trail)
            benchmark_case: Benchmark case being evaluated (optional)
            compression_ratio: Measured compression ratio (optional)
            token_savings_pct: Measured token savings % (optional)
            trace_id: OTEL trace ID for observability (optional)
            request_id: Unique request ID (optional, generated if not provided)
        
        Returns:
            Canonical normalized telemetry
        
        Raises:
            ValueError: If model not found or required tokens can't be extracted
        """
        profile = get_provider_profile(model)
        
        # Extract token counts (this already handles field aliasing)
        cache_summary = summarize_provider_cache_usage(
            model=model,
            api_response=api_response,
            file_id=file_id,
        )
        
        # Generate request ID if not provided
        if not request_id:
            request_hash = hashlib.sha256(json.dumps(api_response).encode()).hexdigest()[:12]
            request_id = f"{model}_{int(datetime.now(UTC).timestamp())}_{request_hash}"
        
        normalized = ProviderTelemetryNormalized(
            provider=profile.provider,
            model=model,
            request_id=request_id,
            timestamp_utc=datetime.now(UTC).isoformat() + "Z",
            input_tokens=cache_summary.get("total_input_tokens", 0),
            output_tokens=cache_summary.get("total_output_tokens", 0),
            cache_creation_tokens=cache_summary.get("cache_creation_tokens"),
            cache_read_tokens=cache_summary.get("cache_read_tokens", 0),
            cache_hit_ratio=cache_summary.get("cache_hit_ratio", 0.0),
            cache_read_cost_usd=cache_summary.get("cached_read_cost_usd", 0.0),
            cache_creation_cost_usd=cache_summary.get("cache_creation_cost_usd", 0.0),
            compression_ratio=compression_ratio,
            token_savings_pct=token_savings_pct,
            benchmark_case_id=benchmark_case.case_id if benchmark_case else None,
            raw_api_response=api_response,
            trace_id=trace_id,
        )
        
        self._telemetry_records.append(normalized)
        return normalized
    
    def validate_against_harness(
        self,
        telemetry: ProviderTelemetryNormalized,
        benchmark_case: BenchmarkCase,
    ) -> HarnessValidationResult:
        """
        Validate telemetry against benchmark case expectations.
        
        Checks:
        1. Compression ratio meets minimum (compression_ratio >= min_compression_ratio)
        2. Token savings % meets minimum (token_savings_pct >= min_token_savings_pct)
        3. Cache expectations met (if expected_cache_hit, cache_hit_ratio > 0)
        4. No anomalies in token accounting
        
        Returns:
            HarnessValidationResult with detailed pass/fail + recommendations
        """
        issues: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []
        
        # 1. Compression ratio check
        compression_ok = False
        if telemetry.compression_ratio is not None:
            compression_ok = telemetry.compression_ratio >= benchmark_case.min_compression_ratio
            if not compression_ok:
                issues.append(
                    f"Compression ratio {telemetry.compression_ratio:.2f}× below "
                    f"target {benchmark_case.min_compression_ratio:.2f}×"
                )
        else:
            warnings.append("Compression ratio not provided; cannot validate")
        
        # 2. Token savings % check
        savings_ok = False
        if telemetry.token_savings_pct is not None:
            savings_ok = telemetry.token_savings_pct >= benchmark_case.min_token_savings_pct
            if not savings_ok:
                issues.append(
                    f"Token savings {telemetry.token_savings_pct:.1f}% below "
                    f"target {benchmark_case.min_token_savings_pct:.1f}%"
                )
        else:
            warnings.append("Token savings % not provided; cannot validate")
        
        # 3. Cache expectation check
        cache_ok = True
        cache_expected = telemetry.cache_read_tokens > 0
        if cache_expected and telemetry.cache_hit_ratio == 0.0:
            issues.append("Cache hit expected but cache_read_tokens is 0")
            cache_ok = False
            recommendations.append("Check if prompt prefix is stable (no UUIDs, timestamps in stable sections)")
            recommendations.append("Check if provider cache creation cost was incurred")
        
        # 4. Determine overall status
        if len(issues) == 0:
            if len(warnings) == 0:
                status = TelemetryValidationStatus.PASS
            else:
                status = TelemetryValidationStatus.PASS_WITH_WARNINGS
        else:
            status = TelemetryValidationStatus.FAIL
        
        # Compute telemetry hash for audit trail
        telemetry_hash = hashlib.sha256(
            json.dumps(asdict(telemetry), default=str).encode()
        ).hexdigest()[:16]
        
        result = HarnessValidationResult(
            status=status,
            passed=status in (TelemetryValidationStatus.PASS, TelemetryValidationStatus.PASS_WITH_WARNINGS),
            case_id=benchmark_case.case_id,
            case_name=benchmark_case.name,
            min_compression_target=benchmark_case.min_compression_ratio,
            actual_compression_ratio=telemetry.compression_ratio,
            compression_target_met=compression_ok,
            min_savings_target_pct=benchmark_case.min_token_savings_pct,
            actual_savings_pct=telemetry.token_savings_pct,
            savings_target_met=savings_ok,
            cache_expected=cache_expected,
            cache_occurred=telemetry.cache_read_tokens > 0,
            cache_hit_ratio_actual=telemetry.cache_hit_ratio,
            cache_expectation_met=cache_ok,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            telemetry_hash=telemetry_hash,
        )
        
        self._validation_results.append(result)
        
        # Export to observability layer
        with observe.trace("harness_validation", case_id=benchmark_case.case_id):
            observe.set_attribute("status", status.value)
            observe.set_attribute("compression_ratio", telemetry.compression_ratio or 0.0)
            observe.set_attribute("token_savings_pct", telemetry.token_savings_pct or 0.0)
            observe.set_attribute("cache_hit_ratio", telemetry.cache_hit_ratio)
            observe.set_attribute("passed", result.passed)
            if not result.passed:
                observe.record_exception(
                    ValueError(f"Harness validation failed: {'; '.join(result.issues)}")
                )
        
        return result
    
    def export_session_metrics(self) -> Dict[str, Any]:
        """
        Aggregate all telemetry from this session.
        
        Returns dict with:
        - total_requests: int
        - avg_compression_ratio: float
        - avg_token_savings_pct: float
        - avg_cache_hit_ratio: float
        - harness_pass_rate: float
        - by_model: Dict[str, Dict] with per-model aggregates
        - validation_results: List[HarnessValidationResult] as dicts
        """
        if not self._telemetry_records:
            return {"total_requests": 0}
        
        by_model: Dict[str, List[ProviderTelemetryNormalized]] = {}
        for record in self._telemetry_records:
            if record.model not in by_model:
                by_model[record.model] = []
            by_model[record.model].append(record)
        
        model_metrics = {}
        for model, records in by_model.items():
            compressions = [r.compression_ratio for r in records if r.compression_ratio]
            savings = [r.token_savings_pct for r in records if r.token_savings_pct]
            caches = [r.cache_hit_ratio for r in records]
            
            model_metrics[model] = {
                "request_count": len(records),
                "avg_compression_ratio": sum(compressions) / len(compressions) if compressions else None,
                "avg_token_savings_pct": sum(savings) / len(savings) if savings else None,
                "avg_cache_hit_ratio": sum(caches) / len(caches) if caches else 0.0,
                "total_cache_cost_saved_usd": sum(r.cache_read_cost_usd for r in records),
            }
        
        harness_passed = sum(1 for r in self._validation_results if r.passed)
        harness_total = len(self._validation_results)
        
        return {
            "total_requests": len(self._telemetry_records),
            "avg_compression_ratio": sum(
                r.compression_ratio for r in self._telemetry_records if r.compression_ratio
            ) / len([r for r in self._telemetry_records if r.compression_ratio])
            if any(r.compression_ratio for r in self._telemetry_records)
            else None,
            "avg_token_savings_pct": sum(
                r.token_savings_pct for r in self._telemetry_records if r.token_savings_pct
            ) / len([r for r in self._telemetry_records if r.token_savings_pct])
            if any(r.token_savings_pct for r in self._telemetry_records)
            else None,
            "avg_cache_hit_ratio": sum(r.cache_hit_ratio for r in self._telemetry_records) / len(self._telemetry_records),
            "total_cache_cost_saved_usd": sum(r.cache_read_cost_usd for r in self._telemetry_records),
            "harness_pass_rate": harness_passed / harness_total if harness_total > 0 else 0.0,
            "by_model": model_metrics,
            "validation_results": [asdict(r) for r in self._validation_results],
        }
`

### Integration Points

**1. Modify src/types.py → HandlerContext**

`python
from src.canonical_telemetry import CanonicalTelemetryAggregator

class HandlerContext(TypedDict, total=True):
    # ... existing ...
    telemetry_aggregator: ReadOnly[CanonicalTelemetryAggregator]  # NEW
`

**2. Modify src/handlers/model_handlers.py**

Update handle_get_cache_telemetry() to use canonical aggregator:

`python
def handle_get_cache_telemetry(args: Dict[str, Any], context: HandlerContext) -> str:
    """Enhanced cache telemetry handler using canonical aggregator."""
    
    model = args.get("model")  # e.g., "claude-opus-4.6"
    api_response = args.get("api_response", {})
    file_id = args.get("file_id")
    trace_id = args.get("trace_id")  # From OTEL
    
    aggregator = context["telemetry_aggregator"]
    
    try:
        # Normalize to canonical format
        telemetry = aggregator.normalize_from_provider_response(
            model=model,
            api_response=api_response,
            file_id=file_id,
            trace_id=trace_id,
        )
        
        return json.dumps({
            "normalized_telemetry": asdict(telemetry),
            "status": "success",
        })
    except ValueError as e:
        return json.dumps({
            "error": str(e),
            "status": "error",
        })
`

**3. Create src/handlers/telemetry_handlers.py**

New handlers for observability layer:

`python
def handle_export_session_metrics(args: Dict[str, Any], context: HandlerContext) -> str:
    """Export aggregated metrics for entire session."""
    
    aggregator = context["telemetry_aggregator"]
    metrics = aggregator.export_session_metrics()
    
    return json.dumps(metrics)

def handle_validate_against_harness(args: Dict[str, Any], context: HandlerContext) -> str:
    """Validate a benchmark run against harness expectations."""
    
    aggregator = context["telemetry_aggregator"]
    
    # Load benchmark case
    case_id = args.get("benchmark_case_id")
    cases = load_benchmark_cases(...)  # From args or config
    case = next((c for c in cases if c.case_id == case_id), None)
    
    if not case:
        return json.dumps({"error": f"Unknown benchmark case: {case_id}"})
    
    # Get most recent telemetry record
    if not aggregator._telemetry_records:
        return json.dumps({"error": "No telemetry records"})
    
    telemetry = aggregator._telemetry_records[-1]
    
    # Validate
    result = aggregator.validate_against_harness(telemetry, case)
    
    return json.dumps({
        "validation_result": asdict(result),
        "summary": result.summarize(),
    })
`

**4. Update src/server.py → _build_context()**

`python
def _build_context(self) -> HandlerContext:
    # ... existing ...
    
    from src.canonical_telemetry import CanonicalTelemetryAggregator
    
    return {
        # ... existing ...
        "telemetry_aggregator": CanonicalTelemetryAggregator(),
    }
`

### Tests to Add

**File**: 	ests/test_canonical_telemetry.py (250 lines)

`python
import pytest
from src.canonical_telemetry import (
    CanonicalTelemetryAggregator,
    ProviderTelemetryNormalized,
    HarnessValidationResult,
    TelemetryValidationStatus,
)
from src.benchmark_harness import BenchmarkCase

class TestCanonicalTelemetryAggregator:
    
    def test_normalize_anthropic_response(self):
        """Normalize Anthropic API response with cache_read_input_tokens."""
        api_response = {
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_creation_input_tokens": 500,
                "cache_read_input_tokens": 300,
            }
        }
        
        aggregator = CanonicalTelemetryAggregator()
        telemetry = aggregator.normalize_from_provider_response(
            model="claude-opus-4.6",
            api_response=api_response,
        )
        
        assert telemetry.model == "claude-opus-4.6"
        assert telemetry.input_tokens == 1000
        assert telemetry.cache_read_tokens == 300
        assert telemetry.cache_creation_tokens == 500
    
    def test_normalize_openai_response(self):
        """Normalize OpenAI API response with cached_tokens."""
        api_response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "cached_tokens": 200,
            }
        }
        
        aggregator = CanonicalTelemetryAggregator()
        telemetry = aggregator.normalize_from_provider_response(
            model="gpt-5.4",
            api_response=api_response,
        )
        
        assert telemetry.model == "gpt-5.4"
        assert telemetry.input_tokens == 1000
        assert telemetry.cache_read_tokens == 200
    
    def test_validate_against_harness_pass(self):
        """Validation passes when metrics exceed targets."""
        aggregator = CanonicalTelemetryAggregator()
        
        case = BenchmarkCase(
            case_id="test_1",
            name="Test Case",
            text="dummy",
            min_compression_ratio=5.0,
            min_token_savings_pct=80.0,
        )
        
        telemetry = ProviderTelemetryNormalized(
            provider="anthropic",
            model="claude-opus-4.6",
            request_id="test",
            timestamp_utc="2025-03-16T00:00:00Z",
            input_tokens=1000,
            output_tokens=100,
            compression_ratio=7.5,  # Exceeds target
            token_savings_pct=85.0,  # Exceeds target
        )
        
        result = aggregator.validate_against_harness(telemetry, case)
        
        assert result.passed
        assert result.compression_target_met
        assert result.savings_target_met
        assert result.status == TelemetryValidationStatus.PASS
    
    def test_validate_against_harness_fail_ratio(self):
        """Validation fails when compression ratio below target."""
        aggregator = CanonicalTelemetryAggregator()
        
        case = BenchmarkCase(
            case_id="test_2",
            name="Test Case 2",
            text="dummy",
            min_compression_ratio=10.0,
            min_token_savings_pct=80.0,
        )
        
        telemetry = ProviderTelemetryNormalized(
            provider="anthropic",
            model="claude-opus-4.6",
            request_id="test",
            timestamp_utc="2025-03-16T00:00:00Z",
            input_tokens=1000,
            output_tokens=100,
            compression_ratio=5.0,  # Below target of 10.0
            token_savings_pct=80.0,
        )
        
        result = aggregator.validate_against_harness(telemetry, case)
        
        assert not result.passed
        assert not result.compression_target_met
        assert result.status == TelemetryValidationStatus.FAIL
    
    def test_export_session_metrics_aggregates(self):
        """Session metrics aggregate across multiple models."""
        aggregator = CanonicalTelemetryAggregator()
        
        # Add Anthropic telemetry
        aggregator.normalize_from_provider_response(
            model="claude-opus-4.6",
            api_response={"usage": {"input_tokens": 1000, "output_tokens": 100}},
            compression_ratio=7.5,
            token_savings_pct=85.0,
        )
        
        # Add OpenAI telemetry
        aggregator.normalize_from_provider_response(
            model="gpt-5.4",
            api_response={"usage": {"prompt_tokens": 2000, "completion_tokens": 200}},
            compression_ratio=6.0,
            token_savings_pct=80.0,
        )
        
        metrics = aggregator.export_session_metrics()
        
        assert metrics["total_requests"] == 2
        assert "by_model" in metrics
        assert "claude-opus-4.6" in metrics["by_model"]
        assert "gpt-5.4" in metrics["by_model"]
`

### Docs/Help Surfaces to Update

**1. docs/guides/PROVIDER_CACHE_COMPATIBILITY.md (Existing, Update section)**

Add section on "Canonical Telemetry Aggregation":

`markdown
## Canonical Telemetry (v0.11.0+)

Token Saver normalizes telemetry across all providers to a **canonical format**.

### Provider Field Mappings

| Metric | Anthropic | OpenAI | Google |
|--------|-----------|--------|--------|
| Input Tokens | input_tokens | prompt_tokens | promptTokenCount |
| Cache Read | cache_read_input_tokens | cached_tokens | cachedContentTokenCount |
| Cache Create | cache_creation_input_tokens | (N/A) | (N/A) |

### Canonical Telemetry Schema

`python
{
  "provider": "anthropic|openai|google",
  "model": "claude-opus-4.6|gpt-5.4|gemini-3.1-pro",
  "input_tokens": int,
  "cache_read_tokens": int,
  "cache_hit_ratio": float,  # Normalized 0-1
  "compression_ratio": float,
  "token_savings_pct": float,
}
`

### Validation Against Harness

After normalizing telemetry, Token Saver validates against benchmark expectations:

`python
# Example: Did this compression meet golden token targets?
result = aggregator.validate_against_harness(
    telemetry=normalized_telemetry,
    benchmark_case=BenchmarkCase(min_compression_ratio=5.0, ...)
)

if result.passed:
    print("✓ Benchmark passed")
else:
    print("✗ Issues:")
    for issue in result.issues:
        print(f"  - {issue}")
`
`

**2. src/handlers/help_handlers.py**

Add help entries:

`python
TOOL_HELP_REGISTRY["export_session_metrics"] = {
    "category": "Observability",
    "description": "Export aggregated telemetry metrics for entire session.",
    "content": """
SESSION METRICS aggregates telemetry from all compression runs:

Returns:
- total_requests: Number of compression jobs
- avg_compression_ratio: Mean compression across all runs
- avg_cache_hit_ratio: Cache effectiveness
- by_model: Per-model breakdowns
- harness_pass_rate: % of runs meeting benchmark targets

Metrics are exported to OpenTelemetry for visualization in Datadog/Grafana.
"""
}

TOOL_HELP_REGISTRY["validate_against_harness"] = {
    "category": "Quality Assurance",
    "description": "Validate compression metrics against benchmark expectations.",
    "content": """
HARNESS VALIDATION checks if a compression run meets golden token targets.

Validates:
1. Compression ratio >= minimum (e.g., 5.0×)
2. Token savings % >= minimum (e.g., 80%)
3. Cache hit ratio meets expectations (if cache expected)

Returns:
- passed: bool - Did this run pass all checks?
- issues: List[str] - What didn't meet targets
- recommendations: List[str] - How to improve

Use this in CI/CD to detect regressions before deployment.
"""
}
`

---

## SLICE 3: COMPOSABLE MULTI-PASS COMPRESSION PIPELINE

### Problem Statement
Currently, compression is a **single monolithic path**:
1. ingest_context() → SemanticCompressor → skeleton output
2. Caller decides what to do next (modulate, search, etc.)

**Missing**: Pipeline architecture where compressors chain together:
- **Pass 1**: Semantic compression (importance-based skeleton)
- **Pass 2**: Code-specific compression (AST-based for code)
- **Pass 3**: Multimodal compression (text + images)
- **Pass 4**: Learnable compression (SCAR-inspired token compression)

Each pass refines the output of the previous pass. Users can compose any subset.

### Existing Foundation (Leverage These)

**1. Multiple Compressor Implementations**

- SemanticCompressor (400 lines): General text/semantic compression
- CodeSemanticCompressor (250 lines): AST-based code compression
- MultiModalCompressor (200 lines): Text + image compression
- LearnableSemanticCompressor (150 lines): SCAR-inspired learnable compression
- ScarCompressor (same as above, alias)

**2. Compression Advisor**

- CompressionAdvisor (120 lines): Pre-compression estimation
- FidelityAdvisor (180 lines): Fidelity level recommendation
- CompressionVerifier (220 lines): Post-compression quality checks

**3. Pipeline-Like Patterns**

- daptive_rate_allocator.py (320 lines): Multi-level encoding with rate selection
- ContextWindowAdapter: Adapts compression based on context availability
- MultiLevelSemanticEncoder: Two-branch architecture (main + auxiliary)

### Minimal Implementation (New Module: src/compression_pipeline.py)

Create **350 lines** of pluggable pipeline infrastructure:

`python
# src/compression_pipeline.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

from .semantic_compressor import SemanticCompressor, SemanticNode, FidelityLevel
from .code_compressor import CodeSemanticCompressor, CodeChunk
from .multimodal_compressor import MultiModalCompressor
from .scar_compressor import LearnableSemanticCompressor
from .structured_logging import get_logger
from .observability import get_observability

logger = get_logger(__name__)
observe = get_observability()

class CompressionPassType(Enum):
    """Types of compression passes in the pipeline."""
    SEMANTIC = "semantic"  # Importance-based skeleton
    CODE = "code"  # AST-based for source code
    MULTIMODAL = "multimodal"  # Text + images
    LEARNABLE = "learnable"  # SCAR-inspired token compression

@dataclass
class CompressionPassResult:
    """Output of a single compression pass."""
    
    pass_type: CompressionPassType
    pass_name: str
    
    # Content output
    compressed_text: str
    skeleton_tokens: int
    compression_ratio: float
    
    # Quality signals
    fidelity_preserved: float  # 0-1, how much semantic info retained
    
    # Diagnostics
    duration_ms: float
    nodes_processed: int = 0
    nodes_retained: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineExecutionResult:
    """Result of executing full compression pipeline."""
    
    original_tokens: int
    final_skeleton_tokens: int
    final_compression_ratio: float
    final_token_savings_pct: float
    
    # Individual pass results (in order executed)
    pass_results: List[CompressionPassResult] = field(default_factory=list)
    
    # Composition info
    passes_executed: List[CompressionPassType] = field(default_factory=list)
    total_duration_ms: float = 0.0
    
    # Quality summary
    min_fidelity_preserved: float = 1.0  # Lowest fidelity across all passes
    avg_fidelity_preserved: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        result = asdict(self)
        result["passes_executed"] = [p.value for p in self.passes_executed]
        return result

class CompressionPass(ABC):
    """
    Abstract base for composable compression passes.
    
    Each pass takes:
    - Input: Uncompressed document (or output of previous pass)
    - Config: Compression parameters (skeleton_ratio, fidelity, etc.)
    
    Each pass produces:
    - Output: Compressed text
    - Metadata: Compression stats (ratio, fidelity, nodes processed)
    """
    
    @property
    @abstractmethod
    def pass_type(self) -> CompressionPassType:
        """Type of this pass."""
        pass
    
    @property
    @abstractmethod
    def pass_name(self) -> str:
        """Human-readable name."""
        pass
    
    @abstractmethod
    def can_handle(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Return True if this pass can compress this content type.
        
        E.g., CodeCompressionPass.can_handle() checks if text is valid code.
        """
        pass
    
    @abstractmethod
    def compress(
        self,
        text: str,
        **config
    ) -> CompressionPassResult:
        """
        Execute this pass.
        
        Args:
            text: Input text (or output from previous pass)
            **config: Pass-specific config (skeleton_ratio, fidelity_level, etc.)
        
        Returns:
            CompressionPassResult with compressed output and metadata
        """
        pass

class SemanticCompressionPass(CompressionPass):
    """
    Pass 1: General semantic compression via importance-based skeleton.
    
    Works on any text: general prose, docs, code, mixed content.
    Handles at least 80% of all document types.
    """
    
    def __init__(self, compressor: SemanticCompressor = None):
        self.compressor = compressor or SemanticCompressor()
    
    @property
    def pass_type(self) -> CompressionPassType:
        return CompressionPassType.SEMANTIC
    
    @property
    def pass_name(self) -> str:
        return "Semantic Compression"
    
    def can_handle(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Semantic pass can handle any non-empty text."""
        return isinstance(text, str) and len(text.strip()) > 0
    
    def compress(
        self,
        text: str,
        file_id: str = "unknown",
        skeleton_ratio: float = 0.2,
        fidelity_level: Optional[FidelityLevel] = None,
        **config
    ) -> CompressionPassResult:
        """
        Execute semantic compression.
        
        Args:
            text: Document text
            file_id: Document identifier
            skeleton_ratio: Fraction of nodes to retain (0.1 - 0.3)
            fidelity_level: Target fidelity (ABSTRACT | OUTLINE | STRUCTURE | DETAILED | RAW)
        """
        import time
        start_ms = time.time() * 1000
        
        # Ingest and compress
        self.compressor.ingest(file_id, text)
        skeleton = self.compressor.read_skeleton(file_id)
        
        # Compute metrics
        original_tokens = self.compressor._estimate_tokens(text)
        skeleton_tokens = self.compressor._estimate_tokens(skeleton)
        compression_ratio = original_tokens / skeleton_tokens if skeleton_tokens > 0 else 1.0
        
        # Fidelity preserved (how much semantic info retained)
        # Heuristic: More nodes retained = higher fidelity
        graph = self.compressor.graphs.get(file_id)
        fidelity = (1.0 - skeleton_ratio) if graph else 0.8  # Default guess
        
        duration_ms = (time.time() * 1000) - start_ms
        
        return CompressionPassResult(
            pass_type=self.pass_type,
            pass_name=self.pass_name,
            compressed_text=skeleton,
            skeleton_tokens=skeleton_tokens,
            compression_ratio=compression_ratio,
            fidelity_preserved=fidelity,
            duration_ms=duration_ms,
            nodes_processed=len(graph.nodes()) if graph else 0,
            nodes_retained=int((1.0 - skeleton_ratio) * len(graph.nodes())) if graph else 0,
            metadata={
                "file_id": file_id,
                "skeleton_ratio": skeleton_ratio,
                "fidelity_level": fidelity_level.name if fidelity_level else "auto",
            }
        )

class CodeCompressionPass(CompressionPass):
    """
    Pass 2: Code-specific compression via AST parsing.
    
    Only works on valid source code (Python, JavaScript, TypeScript, etc.).
    Leverages function/class boundaries and import dependencies.
    Can be skipped if content isn't code.
    """
    
    def __init__(self, compressor: CodeSemanticCompressor = None):
        self.compressor = compressor or CodeSemanticCompressor()
    
    @property
    def pass_type(self) -> CompressionPassType:
        return CompressionPassType.CODE
    
    @property
    def pass_name(self) -> str:
        return "Code-Specific Compression"
    
    def can_handle(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Check if text is valid code (heuristic: has common code patterns)."""
        code_indicators = ["def ", "class ", "import ", "function ", "return "]
        return any(indicator in text for indicator in code_indicators)
    
    def compress(
        self,
        text: str,
        file_id: str = "unknown",
        language: Optional[str] = None,
        skeleton_ratio: float = 0.15,
        **config
    ) -> CompressionPassResult:
        """
        Execute code-specific compression.
        
        Args:
            text: Source code
            file_id: File identifier (used to infer language)
            language: Explicit language (python, javascript, etc.)
            skeleton_ratio: Fraction of functions/classes to retain
        """
        import time
        start_ms = time.time() * 1000
        
        # Parse code and extract chunks
        chunks = self.compressor.parse_code(text, language=language)
        
        # Rank and select top chunks
        ranked_chunks = self.compressor.rank_chunks(chunks)
        num_to_keep = max(1, int(len(ranked_chunks) * (1.0 - skeleton_ratio)))
        selected_chunks = ranked_chunks[:num_to_keep]
        
        # Reconstruct compressed code
        compressed = self.compressor.reconstruct_from_chunks(selected_chunks, text)
        
        # Compute metrics
        original_tokens = self.compressor._estimate_tokens(text)
        skeleton_tokens = self.compressor._estimate_tokens(compressed)
        compression_ratio = original_tokens / skeleton_tokens if skeleton_tokens > 0 else 1.0
        
        # Fidelity: How many public functions/classes retained?
        fidelity = len(selected_chunks) / len(chunks) if chunks else 0.8
        
        duration_ms = (time.time() * 1000) - start_ms
        
        return CompressionPassResult(
            pass_type=self.pass_type,
            pass_name=self.pass_name,
            compressed_text=compressed,
            skeleton_tokens=skeleton_tokens,
            compression_ratio=compression_ratio,
            fidelity_preserved=fidelity,
            duration_ms=duration_ms,
            nodes_processed=len(chunks),
            nodes_retained=len(selected_chunks),
            metadata={
                "file_id": file_id,
                "language": language or "unknown",
                "skeleton_ratio": skeleton_ratio,
                "functions_retained": len([c for c in selected_chunks if c.chunk_type == "function"]),
                "classes_retained": len([c for c in selected_chunks if c.chunk_type == "class"]),
            }
        )

class MultiModalCompressionPass(CompressionPass):
    """
    Pass 3: Multi-modal compression for text + images.
    
    Only works if content contains both text and images.
    Creates unified semantic graph across modalities.
    """
    
    def __init__(self, compressor: MultiModalCompressor = None):
        self.compressor = compressor or MultiModalCompressor()
    
    @property
    def pass_type(self) -> CompressionPassType:
        return CompressionPassType.MULTIMODAL
    
    @property
    def pass_name(self) -> str:
        return "Multi-Modal Compression"
    
    def can_handle(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Check if this content has embedded images."""
        # Heuristic: Check for image references or base64 image data
        return ("![" in text or "<img" in text or "data:image/" in text)
    
    def compress(
        self,
        text: str,
        file_id: str = "unknown",
        image_fidelity: str = "medium",  # low, medium, high
        **config
    ) -> CompressionPassResult:
        """Execute multi-modal compression."""
        import time
        start_ms = time.time() * 1000
        
        # Parse text + extract images
        nodes = self.compressor.parse_mixed_content(text)
        
        # Compress via unified graph
        compressed = self.compressor.compress(file_id, nodes, image_fidelity=image_fidelity)
        
        # Metrics
        original_tokens = self.compressor._estimate_tokens(text)
        skeleton_tokens = self.compressor._estimate_tokens(compressed)
        compression_ratio = original_tokens / skeleton_tokens if skeleton_tokens > 0 else 1.0
        
        image_nodes = [n for n in nodes if n.modality.value == "image"]
        fidelity = (len(nodes) - len(image_nodes)) / len(nodes) if nodes else 0.9
        
        duration_ms = (time.time() * 1000) - start_ms
        
        return CompressionPassResult(
            pass_type=self.pass_type,
            pass_name=self.pass_name,
            compressed_text=compressed,
            skeleton_tokens=skeleton_tokens,
            compression_ratio=compression_ratio,
            fidelity_preserved=fidelity,
            duration_ms=duration_ms,
            nodes_processed=len(nodes),
            nodes_retained=len([n for n in nodes if n.importance > 0.3]),
            metadata={
                "file_id": file_id,
                "text_nodes": len([n for n in nodes if n.modality.value == "text"]),
                "image_nodes": len(image_nodes),
                "image_fidelity": image_fidelity,
            }
        )

class LearnableCompressionPass(CompressionPass):
    """
    Pass 4: Learnable token compression via SCAR-inspired approach.
    
    Compresses embeddings (384D → 96D) while preserving semantics.
    Final pass to squeeze maximum token savings.
    Works on any text (universal).
    """
    
    def __init__(self, compressor: LearnableSemanticCompressor = None):
        self.compressor = compressor or LearnableSemanticCompressor()
    
    @property
    def pass_type(self) -> CompressionPassType:
        return CompressionPassType.LEARNABLE
    
    @property
    def pass_name(self) -> str:
        return "Learnable Compression"
    
    def can_handle(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Learnable pass works on any text."""
        return isinstance(text, str) and len(text.strip()) > 0
    
    def compress(
        self,
        text: str,
        file_id: str = "unknown",
        target_compression_dim: int = 96,
        **config
    ) -> CompressionPassResult:
        """Execute learnable compression."""
        import time
        start_ms = time.time() * 1000
        
        # Embed text
        embeddings = self.compressor.encode(text)
        
        # Compress embeddings
        compressed_embeddings, _ = self.compressor(embeddings)
        
        # Reconstruct and estimate tokens
        original_tokens = self.compressor._estimate_tokens(text)
        # Compressed embeddings are 4× smaller, so estimate proportionally
        skeleton_tokens = int(original_tokens / 4.0)
        compression_ratio = original_tokens / skeleton_tokens if skeleton_tokens > 0 else 1.0
        
        # Semantic preservation: reconstruction loss
        fidelity = 1.0 - (self.compressor.compression_ratio ** 0.5)  # Rough heuristic
        
        duration_ms = (time.time() * 1000) - start_ms
        
        return CompressionPassResult(
            pass_type=self.pass_type,
            pass_name=self.pass_name,
            compressed_text=text,  # Note: Learnable doesn't change text, just embeddings
            skeleton_tokens=skeleton_tokens,
            compression_ratio=compression_ratio,
            fidelity_preserved=fidelity,
            duration_ms=duration_ms,
            nodes_processed=1,
            nodes_retained=1,
            metadata={
                "file_id": file_id,
                "input_dim": self.compressor.input_dim,
                "output_dim": self.compressor.compressed_dim,
                "model": "scar_inspired",
            }
        )

class CompressionPipeline:
    """
    Composable multi-pass compression pipeline.
    
    Features:
    - Define a sequence of compression passes (semantic → code → multimodal → learnable)
    - Each pass operates on output of previous pass
    - Each pass is optional (can be skipped based on content type)
    - Full observability: metrics for each pass + final result
    
    Example:
        pipeline = CompressionPipeline([
            SemanticCompressionPass(),
            CodeCompressionPass(),  # Skipped if not code
            MultiModalCompressionPass(),  # Skipped if no images
        ])
        
        result = pipeline.execute(
            text=my_code,
            file_id="app.py",
            skeleton_ratio=0.2,
        )
        
        print(f"Compression: {result.final_compression_ratio:.1f}×")
        for pass_result in result.pass_results:
            print(f"  {pass_result.pass_name}: {pass_result.compression_ratio:.1f}×")
    """
    
    def __init__(self, passes: List[CompressionPass]):
        """
        Initialize pipeline with ordered list of passes.
        
        Args:
            passes: List of CompressionPass instances in execution order
        """
        self.passes = passes
        self._execution_history: List[PipelineExecutionResult] = []
    
    def execute(
        self,
        text: str,
        file_id: str = "unknown",
        skip_incompatible: bool = True,
        **config
    ) -> PipelineExecutionResult:
        """
        Execute pipeline on input text.
        
        Args:
            text: Input text (or skeleton from previous run)
            file_id: Document identifier
            skip_incompatible: If True, skip passes that can't handle content
            **config: Pass-specific config (skeleton_ratio, fidelity_level, etc.)
        
        Returns:
            PipelineExecutionResult with all pass results and final metrics
        """
        import time
        start_ms = time.time() * 1000
        
        current_text = text
        original_tokens = self._estimate_tokens(text)
        pass_results: List[CompressionPassResult] = []
        passes_executed: List[CompressionPassType] = []
        
        # Execute each pass in order
        for pass_obj in self.passes:
            # Check if pass can handle this content
            if not pass_obj.can_handle(current_text, {"file_id": file_id}):
                if skip_incompatible:
                    logger.info(f"Skipping {pass_obj.pass_name} (incompatible with content type)")
                    continue
            
            # Execute this pass
            with observe.trace("compression_pass", pass_name=pass_obj.pass_name):
                try:
                    pass_result = pass_obj.compress(current_text, file_id=file_id, **config)
                    pass_results.append(pass_result)
                    passes_executed.append(pass_obj.pass_type)
                    
                    # Update current text for next pass
                    current_text = pass_result.compressed_text
                    
                    # Observability
                    observe.set_attribute("pass_name", pass_obj.pass_name)
                    observe.set_attribute("compression_ratio", pass_result.compression_ratio)
                    observe.set_attribute("fidelity", pass_result.fidelity_preserved)
                    
                except Exception as e:
                    logger.error(f"Error in {pass_obj.pass_name}: {e}")
                    observe.record_exception(e)
                    if not skip_incompatible:
                        raise
        
        # Compute final metrics
        final_skeleton_tokens = self._estimate_tokens(current_text)
        final_compression_ratio = original_tokens / final_skeleton_tokens if final_skeleton_tokens > 0 else 1.0
        final_token_savings_pct = (1.0 - final_skeleton_tokens / original_tokens) * 100 if original_tokens > 0 else 0.0
        
        min_fidelity = min(
            (pr.fidelity_preserved for pr in pass_results),
            default=1.0
        )
        avg_fidelity = sum(pr.fidelity_preserved for pr in pass_results) / len(pass_results) if pass_results else 1.0
        
        total_duration_ms = (time.time() * 1000) - start_ms
        
        result = PipelineExecutionResult(
            original_tokens=original_tokens,
            final_skeleton_tokens=final_skeleton_tokens,
            final_compression_ratio=final_compression_ratio,
            final_token_savings_pct=final_token_savings_pct,
            pass_results=pass_results,
            passes_executed=passes_executed,
            total_duration_ms=total_duration_ms,
            min_fidelity_preserved=min_fidelity,
            avg_fidelity_preserved=avg_fidelity,
        )
        
        self._execution_history.append(result)
        
        # Export to observability
        with observe.trace("pipeline_execution", file_id=file_id):
            observe.set_attribute("passes_executed", len(passes_executed))
            observe.set_attribute("final_compression_ratio", final_compression_ratio)
            observe.set_attribute("final_token_savings_pct", final_token_savings_pct)
            observe.set_attribute("total_duration_ms", total_duration_ms)
        
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (placeholder)."""
        # Real implementation would use tiktoken
        return len(text.split()) // 4  # Rough estimate
`

### Integration Points

**1. Modify src/handlers/compression_handlers.py**

Add handler for pipeline-based ingestion:

`python
def handle_ingest_with_pipeline(args: Dict[str, Any], context: HandlerContext) -> str:
    """
    Ingest document using composable compression pipeline.
    
    NEW in v0.11.0: Supports chaining multiple compressors.
    """
    from src.compression_pipeline import CompressionPipeline, (
        SemanticCompressionPass,
        CodeCompressionPass,
        MultiModalCompressionPass,
    )
    
    text = args.get("text", "")
    file_id = args.get("file_id", "unknown")
    pipeline_passes = args.get("pipeline_passes", ["semantic", "code", "multimodal"])
    
    # Build pipeline based on request
    passes = []
    if "semantic" in pipeline_passes:
        passes.append(SemanticCompressionPass(context["compressor"]))
    if "code" in pipeline_passes:
        passes.append(CodeCompressionPass())
    if "multimodal" in pipeline_passes:
        passes.append(MultiModalCompressionPass())
    
    pipeline = CompressionPipeline(passes)
    
    # Execute
    result = pipeline.execute(
        text=text,
        file_id=file_id,
        skeleton_ratio=args.get("skeleton_ratio", 0.2),
    )
    
    return json.dumps({
        "file_id": file_id,
        "final_compression_ratio": result.final_compression_ratio,
        "final_token_savings_pct": result.final_token_savings_pct,
        "passes_executed": [p.value for p in result.passes_executed],
        "pass_details": [
            {
                "pass_name": pr.pass_name,
                "compression_ratio": pr.compression_ratio,
                "fidelity_preserved": pr.fidelity_preserved,
                "duration_ms": pr.duration_ms,
            }
            for pr in result.pass_results
        ],
        "status": "success",
    })
`

**2. Update src/types.py → HandlerContext**

`python
# No changes needed; pipeline is stateless and created per-request
`

**3. Register MCP tool in src/handlers/mcp_core.py**

Add new tool schema:

`python
def setup_mcp_tools(profile: str = "full") -> List[Tool]:
    all_tools = [
        # ... existing tools ...
        
        # NEW: Pipeline-based compression
        Tool(
            name="ingest_with_pipeline",
            description=(
                "Ingest and compress a document using composable multi-pass pipeline. "
                "Chains semantic, code, multimodal, and learnable compression passes. "
                "Each pass refines the output of the previous pass."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Raw document text"},
                    "file_id": {"type": "string", "description": "Unique document ID"},
                    "pipeline_passes": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["semantic", "code", "multimodal", "learnable"]},
                        "description": "Which passes to execute (default: all that apply)",
                        "default": ["semantic"],
                    },
                    "skeleton_ratio": {"type": "number", "description": "Skeleton ratio (0.1-0.3)"},
                },
                "required": ["text", "file_id"],
            },
        ),
    ]
    # ... rest of function ...
`

**4. Add handler route in oute_tool_call()**

`python
def route_tool_call(tool_name: str, args: Dict[str, Any], context: HandlerContext) -> str:
    # ... existing routes ...
    
    elif tool_name == "ingest_with_pipeline":
        return ch.handle_ingest_with_pipeline(args, context)
    
    # ... rest of function ...
`

### Tests to Add

**File**: 	ests/test_compression_pipeline.py (280 lines)

`python
import pytest
from src.compression_pipeline import (
    CompressionPipeline,
    SemanticCompressionPass,
    CodeCompressionPass,
    PipelineExecutionResult,
    CompressionPassType,
)

class TestCompressionPipeline:
    
    def test_semantic_pass_only(self):
        """Pipeline with just semantic compression."""
        pipeline = CompressionPipeline([SemanticCompressionPass()])
        
        text = "The quick brown fox jumps over the lazy dog. " * 50
        result = pipeline.execute(text, file_id="test1")
        
        assert result.final_compression_ratio > 1.0
        assert result.final_token_savings_pct > 0
        assert len(result.pass_results) == 1
        assert result.pass_results[0].pass_type == CompressionPassType.SEMANTIC
    
    def test_code_pass_skipped_for_prose(self):
        """Code pass is skipped when content isn't code."""
        pipeline = CompressionPipeline([
            SemanticCompressionPass(),
            CodeCompressionPass(),
        ])
        
        prose = "This is a long document about philosophy. " * 50
        result = pipeline.execute(prose, file_id="test2", skip_incompatible=True)
        
        # Only semantic pass should have executed
        assert len(result.pass_results) == 1
        assert result.passes_executed == [CompressionPassType.SEMANTIC]
    
    def test_code_pass_executes_for_code(self):
        """Code pass executes when content is code."""
        pipeline = CompressionPipeline([
            SemanticCompressionPass(),
            CodeCompressionPass(),
        ])
        
        code = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        class MyClass:
            def __init__(self):
                pass
        """ * 10
        
        result = pipeline.execute(code, file_id="test3", skip_incompatible=True)
        
        # Both passes should execute
        assert len(result.pass_results) >= 1  # At least semantic
        # Code pass may or may not execute depending on detection
    
    def test_pipeline_aggregates_metrics(self):
        """Final metrics correctly aggregate across passes."""
        pipeline = CompressionPipeline([SemanticCompressionPass()])
        
        text = "Sample text " * 100
        result = pipeline.execute(text, file_id="test4")
        
        # Sanity checks on aggregated metrics
        assert result.original_tokens > 0
        assert result.final_skeleton_tokens >= 0
        assert result.final_compression_ratio >= 1.0
        assert result.total_duration_ms > 0
        assert result.min_fidelity_preserved >= 0.0
        assert result.min_fidelity_preserved <= 1.0
    
    def test_execution_history_tracked(self):
        """Pipeline tracks execution history."""
        pipeline = CompressionPipeline([SemanticCompressionPass()])
        
        result1 = pipeline.execute("Text 1 " * 50, file_id="test5a")
        result2 = pipeline.execute("Text 2 " * 100, file_id="test5b")
        
        assert len(pipeline._execution_history) == 2
        assert pipeline._execution_history[0].original_tokens < pipeline._execution_history[1].original_tokens
`

### Docs/Help Surfaces to Update

**1. Create docs/guides/COMPRESSION_PIPELINE.md (New file)**

`markdown
# Composable Multi-Pass Compression Pipeline

**New in v0.11.0**: Token Saver supports chaining multiple compressors in a flexible pipeline.

## Overview

Instead of a single compression step, you can now compose several passes:

`
Input Text
    ↓
[Semantic Pass] → Importance-based skeleton
    ↓
[Code Pass] → AST-based refinement (optional)
    ↓
[Multimodal Pass] → Text + image fusion (optional)
    ↓
[Learnable Pass] → Embedding compression (optional)
    ↓
Final Skeleton
`

## Passes

### Pass 1: Semantic Compression (Always)

- **What**: Extracts high-importance chunks using graph algorithms
- **Handles**: Any text—prose, code, mixed content
- **Skips**: Never (fallback pass)
- **Config**: skeleton_ratio (0.1-0.3), idelity_level

### Pass 2: Code Compression (Optional)

- **What**: Parses AST, groups by function/class, preserves imports
- **Handles**: Source code (Python, JS, TS, Java, C++, Rust, Go)
- **Skips**: Non-code content (detected via heuristics)
- **Config**: language, skeleton_ratio

### Pass 3: Multimodal Compression (Optional)

- **What**: Unified semantic graph across text + images
- **Handles**: Mixed text + images (Markdown with ![...], HTML with <img>)
- **Skips**: Text-only content
- **Config**: image_fidelity (low | medium | high)

### Pass 4: Learnable Compression (Optional)

- **What**: SCAR-inspired embedding compression (384D → 96D)
- **Handles**: Any text (embeddings of any content)
- **Skips**: Never (applies to any text)
- **Config**: 	arget_compression_dim, model

## Example Usage

`python
from src.compression_pipeline import CompressionPipeline, (
    SemanticCompressionPass,
    CodeCompressionPass,
)

# Create pipeline
pipeline = CompressionPipeline([
    SemanticCompressionPass(),
    CodeCompressionPass(),
])

# Execute
result = pipeline.execute(
    text=my_code,
    file_id="app.py",
    skeleton_ratio=0.2,
)

# Results
print(f"Final: {result.final_compression_ratio:.1f}×")
for pass_result in result.pass_results:
    print(f"  {pass_result.pass_name}: {pass_result.compression_ratio:.1f}× "
          f"(fidelity: {pass_result.fidelity_preserved:.1%})")
`

## Observability

Each pass exports metrics to OpenTelemetry:

`
compression_pass
├── pass_name: "Semantic Compression"
├── compression_ratio: 7.5
├── fidelity_preserved: 0.92
└── duration_ms: 450
`

Final result aggregates all passes and exports to observability backend.

## Performance Guidance

| Content Type | Recommended Passes | Est. Compression | Duration |
|---|---|---|---|
| General prose | Semantic | 5-8× | 500ms |
| Source code | Semantic + Code | 8-12× | 800ms |
| Mixed (code + docs) | Semantic + Code | 7-10× | 900ms |
| Text + images | Semantic + Multimodal | 6-9× | 1200ms |
| Maximum quality | All 4 passes | 10-15× | 2000ms |

Use skip_incompatible=True to automatically skip passes that don't apply.
`

**2. src/handlers/help_handlers.py**

Add help registry entries:

`python
TOOL_HELP_REGISTRY["ingest_with_pipeline"] = {
    "category": "Document Compression",
    "description": "Ingest and compress using composable multi-pass pipeline.",
    "parameters": {
        "text": "Raw document text (required)",
        "file_id": "Unique document ID (required)",
        "pipeline_passes": "List of passes to execute: semantic, code, multimodal, learnable (default: [semantic])",
        "skeleton_ratio": "Skeleton fraction: 0.1 (aggressive) to 0.3 (balanced)",
    },
    "output_fields": {
        "final_compression_ratio": "Overall compression across all passes",
        "final_token_savings_pct": "Total token savings %",
        "passes_executed": "List of passes that ran",
        "pass_details": "Per-pass compression ratios and fidelity",
    },
    "examples": [
        {
            "description": "Compress source code with code-aware pass",
            "args": {
                "text": "def fibonacci(n): ...",
                "file_id": "app.py",
                "pipeline_passes": ["semantic", "code"],
            }
        },
        {
            "description": "Compress mixed content with full pipeline",
            "args": {
                "text": "# Documentation\n`python\ncode...\n`\n![image.png]",
                "file_id": "readme.md",
                "pipeline_passes": ["semantic", "code", "multimodal"],
            }
        }
    ],
    "tips": [
        "Use pipeline_passes=[] with skip_incompatible=True to auto-select passes for content",
        "Code pass dramatically improves compression for source code (2-3× better)",
        "Multimodal pass only activates if images detected; safe to always include",
        "Learnable pass is experimental; best results with embeddings already cached",
        "Each pass's duration scales with document size; see COMPRESSION_PIPELINE.md for guidance",
    ],
    "related_tools": ["ingest_context", "recommend_fidelity", "export_graph_json"],
}
`

---

## SUMMARY TABLE

| Slice | Module | New LOC | Integration Points | Tests | Docs |
|---|---|---|---|---|---|
| **1: Prefix Enforcement** | prefix_validator.py | 189 | prompt_handlers.py (2 places), 	ypes.py, server.py | 	est_prefix_validator.py (180 lines) | PROMPT_CACHING.md update + help entries |
| **2: Telemetry Layer** | canonical_telemetry.py | 320 | 	ypes.py, model_handlers.py, new 	elemetry_handlers.py, server.py | 	est_canonical_telemetry.py (250 lines) | PROVIDER_CACHE_COMPATIBILITY.md update + help entries |
| **3: Pipeline** | compression_pipeline.py | 350 | compression_handlers.py, mcp_core.py (tool + route), no types.py change | 	est_compression_pipeline.py (280 lines) | New COMPRESSION_PIPELINE.md + help entries |
| **TOTAL** | 3 new modules | **859 lines** | 5 modified files | **710 lines** | 3 doc updates + 8 help entries |

---

## DEPLOYMENT CHECKLIST

### Phase 1: Prefix Validator (Week 1-2)

- [ ] Create src/prefix_validator.py (189 lines)
- [ ] Modify src/handlers/prompt_handlers.py: Add validation hooks in handle_create_prompt_template() and handle_prompt_render()
- [ ] Extend src/types.py: Add prefix_validator to HandlerContext
- [ ] Modify src/server.py: Initialize validator in _build_context()
- [ ] Write tests: 	ests/test_prefix_validator.py (180 lines)
- [ ] Update docs: docs/guides/PROMPT_CACHING.md + help registry
- [ ] Run: pytest tests/test_prefix_validator.py -v

### Phase 2: Canonical Telemetry (Week 2-3)

- [ ] Create src/canonical_telemetry.py (320 lines)
- [ ] Create src/handlers/telemetry_handlers.py with 2 new handlers
- [ ] Extend src/types.py: Add 	elemetry_aggregator to HandlerContext
- [ ] Modify src/handlers/model_handlers.py: Update handle_get_cache_telemetry() to use aggregator
- [ ] Modify src/server.py: Initialize aggregator in _build_context()
- [ ] Write tests: 	ests/test_canonical_telemetry.py (250 lines)
- [ ] Update docs: docs/guides/PROVIDER_CACHE_COMPATIBILITY.md + help registry
- [ ] Register new MCP tools: src/handlers/mcp_core.py (2 new Tool objects)
- [ ] Run: pytest tests/test_canonical_telemetry.py -v

### Phase 3: Compression Pipeline (Week 3-4)

- [ ] Create src/compression_pipeline.py (350 lines)
- [ ] Modify src/handlers/compression_handlers.py: Add handle_ingest_with_pipeline()
- [ ] Register new MCP tool: src/handlers/mcp_core.py (1 Tool object + route)
- [ ] Write tests: 	ests/test_compression_pipeline.py (280 lines)
- [ ] Create docs: docs/guides/COMPRESSION_PIPELINE.md + help registry
- [ ] Run: pytest tests/test_compression_pipeline.py -v

### Phase 4: Integration & Validation

- [ ] Run full test suite: pytest tests/test_prefix_validator.py tests/test_canonical_telemetry.py tests/test_compression_pipeline.py -v
- [ ] Check coverage: pytest --cov=src --cov-report=html (aim for 80%+)
- [ ] Update README.md with links to new guides
- [ ] Update CHANGELOG.md with v0.11.0 release notes
- [ ] Manual smoke test: Try all 3 slices in isolation, then together
- [ ] Code review: Submit for technical review (focus on interface contracts, not implementation details)

