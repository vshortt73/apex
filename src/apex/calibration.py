"""Calibration subsystem: frozen prompt generation, validation, and baseline establishment."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from apex.assembler import PromptAssembler
from apex.calibration_store import CalibrationStore
from apex.libraries import ProbeLibrary
from apex.models.base import ModelAdapter
from apex.scoring.base import ScoringDispatcher
from apex.tokenizers import TokenizerBackend, get_tokenizer
from apex.types import (
    CalibrationBaseline,
    CalibrationPrompt,
    ChatMessage,
    ChatResponse,
    FillerType,
)

logger = logging.getLogger(__name__)

CALIBRATION_SEED = 42
CALIBRATION_POSITIONS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]
CALIBRATION_CONTEXT_LENGTHS = [4096, 8192]

SYSTEM_MESSAGE = "You are a helpful assistant."


def content_hash(text: str) -> str:
    """SHA256 hex digest of the full text."""
    return hashlib.sha256(text.encode()).hexdigest()


def probe_hash(probe_content: str) -> str:
    """First 16 hex chars of SHA256 of probe content."""
    return hashlib.sha256(probe_content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stage 1: Generate frozen prompt matrix
# ---------------------------------------------------------------------------


class CalibrationGenerator:
    """Generates a matrix of frozen calibration prompts."""

    def __init__(
        self,
        library: ProbeLibrary,
        tokenizer_spec: str = "approximate",
        filler_type: FillerType = FillerType.NEUTRAL,
    ) -> None:
        self.library = library
        tokenizer = get_tokenizer(tokenizer_spec)
        fillers = library.get_fillers(filler_type)
        self._assembler = PromptAssembler(tokenizer, fillers)

    def generate(
        self,
        positions: list[float] | None = None,
        context_lengths: list[int] | None = None,
        seed: int = CALIBRATION_SEED,
    ) -> list[CalibrationPrompt]:
        """Generate frozen prompts for all probes x positions x context lengths."""
        positions = positions or CALIBRATION_POSITIONS
        context_lengths = context_lengths or CALIBRATION_CONTEXT_LENGTHS
        probes = self.library.get_probes()
        now = datetime.now(timezone.utc).isoformat()
        results: list[CalibrationPrompt] = []

        for p in probes:
            query = self.library.get_query_for_probe(p.probe_id)
            if query is None:
                logger.warning("No query for probe %s — skipping", p.probe_id)
                continue
            for ctx_len in context_lengths:
                for pos in positions:
                    assembled = self._assembler.assemble_fixed_filler(
                        probe=p,
                        test_query=query,
                        position_percent=pos,
                        context_length=ctx_len,
                        config_seed=seed,
                    )
                    prompt = CalibrationPrompt(
                        probe_id=p.probe_id,
                        dimension=p.dimension.value,
                        position_percent=pos,
                        context_length=ctx_len,
                        seed=assembled.seed,
                        full_text=assembled.full_text,
                        actual_token_count=assembled.actual_token_count,
                        target_position_tokens=assembled.target_position_tokens,
                        filler_ids_before=",".join(assembled.filler_ids_before),
                        filler_ids_after=",".join(assembled.filler_ids_after),
                        probe_hash=probe_hash(p.content),
                        content_hash=content_hash(assembled.full_text),
                        generated_at=now,
                    )
                    results.append(prompt)

        return results


# ---------------------------------------------------------------------------
# Stage 2: Validate frozen prompts
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    probe_id: str
    position_percent: float
    context_length: int
    passed: bool
    checks: dict[str, bool]
    messages: list[str]


class CalibrationValidator:
    """Validates frozen calibration prompts against the current library."""

    def __init__(
        self,
        library: ProbeLibrary,
        tokenizer_spec: str = "approximate",
        position_tolerance: float = 0.10,
        fill_tolerance: float = 0.15,
    ) -> None:
        self.library = library
        self._tokenizer = get_tokenizer(tokenizer_spec)
        self._position_tolerance = position_tolerance
        self._fill_tolerance = fill_tolerance

    def validate(self, prompts: list[dict]) -> list[ValidationResult]:
        """Validate a list of prompt dicts from the store."""
        results: list[ValidationResult] = []
        for p in prompts:
            checks: dict[str, bool] = {}
            messages: list[str] = []

            probe_id = p["probe_id"]
            probe = self.library.probes.get(probe_id)

            # 1. Probe content present in full_text
            if probe is None:
                checks["probe_present"] = False
                messages.append(f"Unknown probe_id: {probe_id}")
            else:
                present = probe.content in p["full_text"]
                checks["probe_present"] = present
                if not present:
                    messages.append("Probe content not found in full_text")

            # 2. Content hash integrity
            recomputed = content_hash(p["full_text"])
            hash_ok = recomputed == p["content_hash"]
            checks["content_hash"] = hash_ok
            if not hash_ok:
                messages.append(f"Content hash mismatch: stored={p['content_hash'][:16]}... recomputed={recomputed[:16]}...")

            # 3. Probe hash freshness
            if probe is not None:
                current_ph = probe_hash(probe.content)
                ph_ok = current_ph == p["probe_hash"]
                checks["probe_hash_fresh"] = ph_ok
                if not ph_ok:
                    messages.append("Probe content has changed since generation (probe_hash mismatch)")
            else:
                checks["probe_hash_fresh"] = False

            # 4. Token position within tolerance (as fraction of context_length)
            ctx_len = p["context_length"]
            target_pos_tokens = p["target_position_tokens"]
            expected_pos_tokens = int(ctx_len * p["position_percent"])
            if ctx_len > 0:
                deviation = abs(target_pos_tokens - expected_pos_tokens) / ctx_len
            else:
                deviation = 0.0
            pos_ok = deviation <= self._position_tolerance
            checks["position_tolerance"] = pos_ok
            if not pos_ok:
                messages.append(f"Token position deviation {deviation:.1%} of context exceeds {self._position_tolerance:.0%} tolerance")

            # 5. Context length fill within tolerance
            actual = p["actual_token_count"]
            if ctx_len > 0:
                fill_ratio = actual / ctx_len
                fill_ok = fill_ratio >= (1.0 - self._fill_tolerance)
            else:
                fill_ok = False
            checks["fill_tolerance"] = fill_ok
            if not fill_ok:
                messages.append(f"Fill ratio {fill_ratio:.1%} below {1.0 - self._fill_tolerance:.0%} threshold")

            passed = all(checks.values())
            results.append(ValidationResult(
                probe_id=probe_id,
                position_percent=p["position_percent"],
                context_length=p["context_length"],
                passed=passed,
                checks=checks,
                messages=messages,
            ))

        return results


# ---------------------------------------------------------------------------
# Stage 3: Establish two-tier baselines
# ---------------------------------------------------------------------------


class BaselineRunner:
    """Runs two-tier baselines: bare (probe only) or anchored (filler at pos 0.05)."""

    def __init__(
        self,
        library: ProbeLibrary,
        dispatcher: ScoringDispatcher,
        adapter: ModelAdapter,
        store: CalibrationStore | None = None,
    ) -> None:
        self.library = library
        self.dispatcher = dispatcher
        self.adapter = adapter
        self.store = store

    def run_baselines(
        self,
        baseline_type: str = "bare",
        probe_ids: list[str] | None = None,
    ) -> list[CalibrationBaseline]:
        """Run baselines for all or selected probes.

        Args:
            baseline_type: "bare" (no filler) or "anchored" (filler at pos 0.05)
            probe_ids: Optional list to filter probes
        """
        if probe_ids:
            probes = self.library.get_probes(probe_ids=probe_ids)
        else:
            probes = self.library.get_probes()

        model_info = self.adapter.get_model_info()
        results: list[CalibrationBaseline] = []

        for p in probes:
            query = self.library.get_query_for_probe(p.probe_id)
            if query is None:
                logger.warning("No query for probe %s — skipping baseline", p.probe_id)
                continue

            if baseline_type == "anchored":
                # Run anchored baseline for each context length in the frozen prompt matrix
                baselines = self._run_anchored_baselines(p, query, model_info.model_id)
                results.extend(baselines)
            else:
                baseline = self._run_single_baseline(
                    p, query, model_info.model_id, baseline_type="bare", turn1_input=p.content,
                )
                results.append(baseline)

        return results

    def _run_anchored_baselines(self, probe, query, model_id: str) -> list[CalibrationBaseline]:
        """Run anchored baselines using frozen prompts at endpoints (0.05 and 0.95).

        For each context length, runs both endpoint positions and keeps the
        higher score as the anchor — this represents the best-case-with-filler
        performance, isolating the filler effect from position effects.
        """
        if self.store is None:
            logger.error("CalibrationStore required for anchored baselines")
            return []

        frozen = self.store.get_prompts(probe_id=probe.probe_id)
        # Filter to endpoint positions (0.05 and 0.95)
        endpoint_prompts = [
            fp for fp in frozen
            if abs(fp["position_percent"] - 0.05) < 0.001
            or abs(fp["position_percent"] - 0.95) < 0.001
        ]

        if not endpoint_prompts:
            logger.warning("No frozen prompts at endpoints for %s — skipping anchored", probe.probe_id)
            return []

        # Group by context_length, run both endpoints, keep the higher score
        by_ctx: dict[int, list[dict]] = {}
        for fp in endpoint_prompts:
            by_ctx.setdefault(fp["context_length"], []).append(fp)

        results: list[CalibrationBaseline] = []
        for ctx_len, fps in sorted(by_ctx.items()):
            best_baseline: CalibrationBaseline | None = None
            for fp in fps:
                baseline = self._run_single_baseline(
                    probe, query, model_id,
                    baseline_type="anchored",
                    turn1_input=fp["full_text"],
                )
                if best_baseline is None:
                    best_baseline = baseline
                elif baseline.score is not None and (
                    best_baseline.score is None or baseline.score > best_baseline.score
                ):
                    best_baseline = baseline
            if best_baseline is not None:
                results.append(best_baseline)

        return results

    def _run_single_baseline(
        self, probe, query, model_id: str, baseline_type: str, turn1_input: str,
    ) -> CalibrationBaseline:
        """Execute the two-turn protocol for a single baseline."""
        now = datetime.now(timezone.utc).isoformat()
        error = None
        raw_response = ""
        raw_test_response = ""
        score_val = None
        justification = None

        # Turn 1
        try:
            turn1 = self.adapter.single_turn(SYSTEM_MESSAGE, turn1_input)
            raw_response = turn1.content
        except Exception as e:
            logger.error("Baseline turn 1 failed for %s: %s", probe.probe_id, e)
            error = str(e)
            baseline = CalibrationBaseline(
                probe_id=probe.probe_id,
                dimension=probe.dimension.value,
                model_id=model_id,
                baseline_type=baseline_type,
                score=None,
                score_method=probe.score_method.value,
                justification=None,
                raw_response="",
                raw_test_response="",
                error=error,
                timestamp=now,
            )
            if self.store:
                self.store.write_baseline(baseline)
            return baseline

        # Turn 2
        messages = [
            ChatMessage(role="system", content=SYSTEM_MESSAGE),
            ChatMessage(role="user", content=turn1_input),
            ChatMessage(role="assistant", content=turn1.content),
            ChatMessage(role="user", content=query.query),
        ]
        try:
            turn2 = self.adapter.chat(messages)
            raw_test_response = turn2.content
        except Exception as e:
            logger.error("Baseline turn 2 failed for %s: %s", probe.probe_id, e)
            error = str(e)
            baseline = CalibrationBaseline(
                probe_id=probe.probe_id,
                dimension=probe.dimension.value,
                model_id=model_id,
                baseline_type=baseline_type,
                score=None,
                score_method=probe.score_method.value,
                justification=None,
                raw_response=raw_response,
                raw_test_response="",
                error=error,
                timestamp=now,
            )
            if self.store:
                self.store.write_baseline(baseline)
            return baseline

        # Score
        try:
            score_val, _eval_model_id, justification = self.dispatcher.score(
                probe, query, turn2.content
            )
        except Exception as e:
            logger.error("Baseline scoring failed for %s: %s", probe.probe_id, e)
            error = str(e)

        baseline = CalibrationBaseline(
            probe_id=probe.probe_id,
            dimension=probe.dimension.value,
            model_id=model_id,
            baseline_type=baseline_type,
            score=score_val,
            score_method=probe.score_method.value,
            justification=justification,
            raw_response=raw_response,
            raw_test_response=raw_test_response,
            error=error,
            timestamp=now,
        )
        if self.store:
            self.store.write_baseline(baseline)
        return baseline
