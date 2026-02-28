"""Prompt assembly: filler + probe at exact token position."""

from __future__ import annotations

import hashlib
import random

from apex.tokenizers import TokenizerBackend
from apex.types import AssembledPrompt, FillerPassage, Probe, TestQuery


class PromptAssembler:
    """Assembles prompts by placing a probe at a target position within filler text."""

    def __init__(self, tokenizer: TokenizerBackend, fillers: list[FillerPassage]) -> None:
        self.tokenizer = tokenizer
        self.fillers = fillers

    def _make_seed(
        self,
        config_seed: int,
        probe_id: str,
        position_percent: float,
        context_length: int,
        run_number: int,
    ) -> int:
        raw = f"{config_seed}:{probe_id}:{position_percent}:{context_length}:{run_number}"
        return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)

    def assemble(
        self,
        probe: Probe,
        test_query: TestQuery,
        position_percent: float,
        context_length: int,
        config_seed: int = 42,
        run_number: int = 1,
    ) -> AssembledPrompt:
        """Build a prompt with probe at target position within filler.

        Algorithm:
        1. Compute deterministic seed from parameters
        2. Shuffle filler pool with seeded RNG
        3. Count probe tokens
        4. Compute target position in tokens
        5. Greedily pack whole filler passages before probe up to target position
        6. Insert probe between \\n\\n delimiters
        7. Greedily pack whole filler passages after probe up to context_length
        8. Record actual token count
        """
        seed = self._make_seed(config_seed, probe.probe_id, position_percent, context_length, run_number)
        rng = random.Random(seed)

        shuffled = list(self.fillers)
        rng.shuffle(shuffled)

        probe_text = probe.content
        probe_tokens = self.tokenizer.count_tokens(probe_text)

        target_pos_tokens = int(context_length * position_percent)

        # Pack filler before probe
        before_parts: list[str] = []
        before_ids: list[str] = []
        before_tokens = 0
        filler_idx = 0

        while filler_idx < len(shuffled):
            fp = shuffled[filler_idx]
            fp_tokens = self.tokenizer.count_tokens(fp.content)
            delimiter_tokens = self.tokenizer.count_tokens("\n\n")
            needed = fp_tokens + (delimiter_tokens if before_parts else 0)
            if before_tokens + needed + probe_tokens > target_pos_tokens + probe_tokens:
                # Would push probe past target position
                if before_tokens + needed > target_pos_tokens:
                    break
            if before_tokens + needed > target_pos_tokens:
                break
            if before_parts:
                before_parts.append("\n\n")
                before_tokens += delimiter_tokens
            before_parts.append(fp.content)
            before_ids.append(fp.filler_id)
            before_tokens += fp_tokens
            filler_idx += 1

        # Insert probe
        if before_parts:
            before_parts.append("\n\n")
            before_tokens += self.tokenizer.count_tokens("\n\n")

        # Pack filler after probe
        after_parts: list[str] = []
        after_ids: list[str] = []
        after_tokens = 0
        tokens_so_far = before_tokens + probe_tokens

        while filler_idx < len(shuffled):
            fp = shuffled[filler_idx]
            fp_tokens = self.tokenizer.count_tokens(fp.content)
            delimiter_tokens = self.tokenizer.count_tokens("\n\n")
            needed = delimiter_tokens + fp_tokens
            if tokens_so_far + after_tokens + needed > context_length:
                break
            after_parts.append("\n\n")
            after_parts.append(fp.content)
            after_ids.append(fp.filler_id)
            after_tokens += needed
            filler_idx += 1

        # Assemble final text
        full_text = "".join(before_parts) + probe_text + "".join(after_parts)
        actual_tokens = self.tokenizer.count_tokens(full_text)

        return AssembledPrompt(
            probe=probe,
            test_query=test_query,
            full_text=full_text,
            target_position_tokens=before_tokens,
            target_position_percent=position_percent,
            actual_token_count=actual_tokens,
            context_length_target=context_length,
            filler_ids_before=before_ids,
            filler_ids_after=after_ids,
            seed=seed,
        )
