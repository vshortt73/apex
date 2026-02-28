"""Shared dataclasses and enums for APEX."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class Dimension(str, enum.Enum):
    FACTUAL_RECALL = "factual_recall"
    APPLICATION = "application"
    SALIENCE = "salience"


class ScoreMethod(str, enum.Enum):
    EXACT_MATCH = "exact_match"
    SEMANTIC = "semantic"
    PROGRAMMATIC = "programmatic"
    EVALUATOR = "evaluator"


class FillerType(str, enum.Enum):
    NEUTRAL = "neutral"
    EMOTIONAL = "emotional"


@dataclass(frozen=True)
class FillerPassage:
    filler_id: str
    content: str
    domain: str
    token_count_estimate: int
    flesch_kincaid_grade: float
    panas_positive: float
    panas_negative: float
    notes: str = ""


@dataclass(frozen=True)
class Probe:
    probe_id: str
    dimension: Dimension
    content: str
    content_type: str
    token_counts: dict[str, int]
    intrinsic_salience: dict[str, float]
    domain: str
    confounding_factors: str
    evaluation_query_id: str
    score_method: ScoreMethod
    version: str = "1.0"


@dataclass(frozen=True)
class TestQuery:
    query_id: str
    probe_id: str
    dimension: Dimension
    query: str
    expected_answer: str | None = None
    expected_answer_secondary: str | None = None
    rubric: str | None = None
    score_method: ScoreMethod = ScoreMethod.EXACT_MATCH


@dataclass
class AssembledPrompt:
    probe: Probe
    test_query: TestQuery
    full_text: str
    target_position_tokens: int
    target_position_percent: float
    actual_token_count: int
    context_length_target: int
    filler_ids_before: list[str] = field(default_factory=list)
    filler_ids_after: list[str] = field(default_factory=list)
    seed: int = 0


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    backend: str
    model_name: str
    architecture: str = "unknown"
    parameters: str = "unknown"
    quantization: str = "none"
    max_context_window: int = 4096
    tokenizer: str = "approximate"


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    content: str
    latency_ms: int
    finish_reason: str = "stop"
    refused: bool = False


@dataclass
class ProbeResult:
    model_id: str
    model_architecture: str
    model_parameters: str
    quantization: str
    max_context_window: int
    context_length: int
    context_fill_ratio: float
    target_position: int
    target_position_percent: float
    dimension: str
    content_type: str
    probe_id: str
    probe_content: str
    filler_type: str
    test_query_id: str
    temperature: float
    run_number: int
    total_runs: int
    score: float | None
    score_method: str
    raw_response: str
    raw_test_response: str
    evaluator_model_id: str | None = None
    evaluator_justification: str | None = None
    latency_ms: int = 0
    timestamp: str = ""
    library_version: str = "1.0"
    framework_version: str = "1.0.0"
    refused: bool = False
