"""Probe execution engine — orchestrates the two-turn protocol."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from tqdm import tqdm

from apex import __version__
from apex.assembler import PromptAssembler
from apex.config import ModelConfig, RunConfig
from apex.libraries import ProbeLibrary
from apex.models.base import ModelAdapter, get_adapter
from apex.scoring.base import ScoringDispatcher
from apex.storage import ResultStore
from apex.tokenizers import get_tokenizer
from apex.types import (
    ChatMessage,
    ChatResponse,
    Dimension,
    FillerType,
    ModelInfo,
    ProbeResult,
)

# Type alias for frozen prompt lookup
FrozenLookup = dict[tuple[str, float, int], dict]

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = "You are a helpful assistant."

_REFUSAL_PHRASES = [
    "i cannot",
    "i can't",
    "i'm unable to",
    "i am unable to",
    "as an ai",
    "i'm not able to",
    "i must decline",
    "i apologize, but i cannot",
    "content policy",
    "safety guidelines",
]


def _is_refusal(response: ChatResponse) -> bool:
    """Detect if a response is a refusal."""
    if not response.content or not response.content.strip():
        return True
    lower = response.content.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


class ProbeRunner:
    """Orchestrates probe execution across models, positions, and repetitions."""

    def __init__(self, config: RunConfig, library: ProbeLibrary, store: ResultStore) -> None:
        self.config = config
        self.library = library
        self.store = store

    def run(self) -> None:
        """Execute the full probe run."""
        self._run_uuid = str(uuid.uuid4())
        probes = self._select_probes()
        if not probes:
            logger.error("No probes selected — check config and library")
            return

        logger.info(
            "Starting run: %d probes × %d positions × %d context lengths × %d reps × %d models",
            len(probes),
            len(self.config.positions),
            len(self.config.context_lengths),
            self.config.repetitions,
            len(self.config.models),
        )

        for model_cfg in self.config.models:
            self._run_model(model_cfg, probes)

    def _select_probes(self):
        sel = self.config.probe_select
        if sel == "all":
            return self.library.get_probes()
        if isinstance(sel, list):
            return self.library.get_probes(probe_ids=sel)
        # Try as dimension name
        try:
            dim = Dimension(sel)
            return self.library.get_probes(dimension=dim)
        except ValueError:
            logger.error("Unknown probe selector: %s", sel)
            return []

    def _make_adapter(self, model_cfg: ModelConfig) -> ModelAdapter:
        """Create a ModelAdapter for the given config."""
        return get_adapter(
            backend=model_cfg.backend,
            model_name=model_cfg.model_name,
            base_url=model_cfg.base_url,
            api_key=model_cfg.api_key,
            temperature=self.config.temperature,
            model_info_overrides={
                "model_id": model_cfg.name,
                "architecture": model_cfg.architecture,
                "parameters": model_cfg.parameters,
                "quantization": model_cfg.quantization,
                "max_context_window": model_cfg.max_context_window,
                "tokenizer": model_cfg.tokenizer,
            },
        )

    def _make_evaluator_adapter(self) -> ModelAdapter | None:
        """Create an evaluator adapter if configured, else None."""
        if not self.config.evaluator_models:
            return None
        ev_cfg = self.config.evaluator_models[0]
        return get_adapter(
            backend=ev_cfg.backend,
            model_name=ev_cfg.model_name,
            base_url=ev_cfg.base_url,
            api_key=ev_cfg.api_key,
            temperature=0.0,
            model_info_overrides={"model_id": ev_cfg.name},
        )

    def _load_frozen_prompts(self) -> FrozenLookup:
        """Load frozen calibration prompts into a lookup dict."""
        from apex.calibration_store import CalibrationStore

        cal_store = CalibrationStore(self.config.database_dsn)
        frozen = cal_store.get_prompts()
        cal_store.close()

        lookup: FrozenLookup = {}
        for fp in frozen:
            key = (fp["probe_id"], fp["position_percent"], fp["context_length"])
            lookup[key] = fp

        logger.info("Loaded %d frozen calibration prompts", len(lookup))
        return lookup

    def _run_model(self, model_cfg: ModelConfig, probes) -> None:
        logger.info("Running model: %s (%s)", model_cfg.name, model_cfg.backend)

        adapter = self._make_adapter(model_cfg)
        model_info = adapter.get_model_info()
        tokenizer = get_tokenizer(model_cfg.tokenizer)

        evaluator_adapter = self._make_evaluator_adapter()
        dispatcher = ScoringDispatcher(evaluator_adapter)
        fillers = self.library.get_fillers(self.config.filler_type_enum)
        assembler = PromptAssembler(tokenizer, fillers)

        # Load frozen prompts if calibrated mode
        frozen_lookup: FrozenLookup = {}
        if self.config.use_calibration:
            frozen_lookup = self._load_frozen_prompts()
            if not frozen_lookup:
                logger.warning("Calibrated mode enabled but no frozen prompts found — falling back to assembler")

        # Build run queue
        completed = self.store.get_completed_runs(model_info.model_id)
        queue = []
        missing_frozen = 0
        for probe in probes:
            query = self.library.get_query_for_probe(probe.probe_id)
            if query is None:
                logger.warning("No query for probe %s — skipping", probe.probe_id)
                continue
            for ctx_len in self.config.context_lengths:
                for pos in self.config.positions:
                    for run_num in range(1, self.config.repetitions + 1):
                        key = (probe.probe_id, pos, ctx_len, run_num)
                        if key not in completed:
                            frozen_key = (probe.probe_id, pos, ctx_len)
                            frozen = frozen_lookup.get(frozen_key) if frozen_lookup else None
                            if frozen_lookup and frozen is None:
                                missing_frozen += 1
                                logger.debug(
                                    "No frozen prompt for %s pos=%.2f ctx=%d — will use assembler",
                                    probe.probe_id, pos, ctx_len,
                                )
                            queue.append((probe, query, pos, ctx_len, run_num, frozen))

        if missing_frozen > 0:
            logger.warning(
                "%d queue items have no frozen prompt — assembler fallback will be used",
                missing_frozen,
            )

        if not queue:
            logger.info("All probes already completed for %s", model_info.model_id)
            return

        logger.info(
            "%d probes to run for %s (%d already completed)",
            len(queue), model_info.model_id, len(completed),
        )

        if self.config.workers <= 1:
            # Sequential path — zero overhead, same behavior as before
            for probe, query, pos, ctx_len, run_num, frozen in tqdm(queue, desc=model_info.model_id):
                self._execute_probe(
                    adapter, model_info, assembler, dispatcher,
                    probe, query, pos, ctx_len, run_num,
                    frozen_prompt=frozen,
                )
        else:
            # Parallel path — per-thread resources via threading.local()
            self._run_model_parallel(
                model_cfg, model_info, assembler, queue,
            )

    def _run_model_parallel(
        self,
        model_cfg: ModelConfig,
        model_info: ModelInfo,
        assembler: PromptAssembler,
        queue: list,
    ) -> None:
        """Execute probes in parallel using ThreadPoolExecutor."""
        local = threading.local()
        thread_stores: list[ResultStore] = []
        stores_lock = threading.Lock()

        def _get_thread_resources():
            """Lazily initialize per-thread adapter, store, and dispatcher."""
            if not hasattr(local, "adapter"):
                local.adapter = self._make_adapter(model_cfg)
                local.store = ResultStore(self.config.database_dsn)
                local.dispatcher = ScoringDispatcher(self._make_evaluator_adapter())
                with stores_lock:
                    thread_stores.append(local.store)
            return local.adapter, local.store, local.dispatcher

        def _worker(item):
            probe, query, pos, ctx_len, run_num, frozen = item
            adapter, store, dispatcher = _get_thread_resources()
            self._execute_probe(
                adapter, model_info, assembler, dispatcher,
                probe, query, pos, ctx_len, run_num,
                store=store,
                frozen_prompt=frozen,
            )

        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            futures = [pool.submit(_worker, item) for item in queue]
            for future in tqdm(as_completed(futures), total=len(futures), desc=model_info.model_id):
                exc = future.exception()
                if exc is not None:
                    logger.error("Worker failed: %s", exc)

        # Cleanup per-thread stores
        for store in thread_stores:
            try:
                store.close()
            except Exception:
                pass

    def _execute_probe(
        self,
        adapter: ModelAdapter,
        model_info: ModelInfo,
        assembler: PromptAssembler,
        dispatcher: ScoringDispatcher,
        probe,
        query,
        position_percent: float,
        context_length: int,
        run_number: int,
        store: ResultStore | None = None,
        frozen_prompt: dict | None = None,
    ) -> None:
        """Execute a single probe: two-turn protocol, score, record.

        If frozen_prompt is provided (calibrated mode), uses the pre-assembled
        text from the calibration_prompts table instead of assembling fresh.
        """
        if frozen_prompt is not None:
            full_text = frozen_prompt["full_text"]
            target_position_tokens = frozen_prompt["target_position_tokens"]
            filler_type = "calibrated"
        else:
            assembled = assembler.assemble(
                probe=probe,
                test_query=query,
                position_percent=position_percent,
                context_length=context_length,
                config_seed=self.config.seed,
                run_number=run_number,
            )
            full_text = assembled.full_text
            target_position_tokens = assembled.target_position_tokens
            filler_type = self.config.filler_type

        # Turn 1: Send filler+probe
        refused = False
        try:
            turn1_response = self._call_with_retry(
                adapter.single_turn, SYSTEM_MESSAGE, full_text
            )
        except Exception as e:
            logger.error("Turn 1 failed for %s at %.0f%%: %s", probe.probe_id, position_percent * 100, e)
            turn1_response = ChatResponse(content="", latency_ms=0, refused=True)
            refused = True

        if _is_refusal(turn1_response):
            refused = True

        # Turn 2: Send test query with full history
        turn2_response = ChatResponse(content="", latency_ms=0)
        if not refused:
            messages = [
                ChatMessage(role="system", content=SYSTEM_MESSAGE),
                ChatMessage(role="user", content=full_text),
                ChatMessage(role="assistant", content=turn1_response.content),
                ChatMessage(role="user", content=query.query),
            ]
            try:
                turn2_response = self._call_with_retry(adapter.chat, messages)
            except Exception as e:
                logger.error("Turn 2 failed for %s: %s", probe.probe_id, e)
                turn2_response = ChatResponse(content="", latency_ms=0, refused=True)
                refused = True

            if _is_refusal(turn2_response):
                refused = True

        # Score
        score_val = None
        evaluator_model_id = None
        justification = None
        if not refused and turn2_response.content:
            try:
                score_val, evaluator_model_id, justification = dispatcher.score(
                    probe, query, turn2_response.content
                )
            except Exception as e:
                logger.error("Scoring failed for %s: %s", probe.probe_id, e)

        # Record
        result = ProbeResult(
            model_id=model_info.model_id,
            model_architecture=model_info.architecture,
            model_parameters=model_info.parameters,
            quantization=model_info.quantization,
            max_context_window=model_info.max_context_window,
            context_length=context_length,
            context_fill_ratio=context_length / model_info.max_context_window,
            target_position=target_position_tokens,
            target_position_percent=position_percent,
            dimension=probe.dimension.value,
            content_type=probe.content_type,
            probe_id=probe.probe_id,
            probe_content=probe.content,
            filler_type=filler_type,
            test_query_id=query.query_id,
            temperature=self.config.temperature,
            run_number=run_number,
            total_runs=self.config.repetitions,
            score=score_val,
            score_method=probe.score_method.value,
            raw_response=turn1_response.content,
            raw_test_response=turn2_response.content,
            evaluator_model_id=evaluator_model_id,
            evaluator_justification=justification,
            latency_ms=turn1_response.latency_ms + turn2_response.latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            library_version=self.library.library_version,
            framework_version=__version__,
            refused=refused,
            run_uuid=self._run_uuid,
        )
        target_store = store if store is not None else self.store
        target_store.write_result(result)

    def _call_with_retry(self, fn, *args, max_retries: int = 3, base_delay: float = 2.0):
        """Call fn with exponential backoff retry."""
        last_exc = None
        for attempt in range(max_retries):
            try:
                return fn(*args)
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Retry %d/%d after %.1fs: %s", attempt + 1, max_retries, delay, e
                    )
                    time.sleep(delay)
        raise last_exc
