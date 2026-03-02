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

    def assemble_fixed_filler(
        self,
        probe: Probe,
        test_query: TestQuery,
        position_percent: float,
        context_length: int,
        config_seed: int = 42,
    ) -> AssembledPrompt:
        """Build a prompt with fixed filler — only the probe insertion point moves.

        Unlike ``assemble()``, the filler shuffle seed is position-independent:
        ``config_seed:probe_id:context_length``.  A single filler body is packed
        to fill ``context_length`` tokens (minus the probe), then the probe is
        inserted at the target split point.  This guarantees identical filler
        text across all positions for a given ``(probe_id, context_length)``.
        """
        # Position-independent seed → same filler order for every position
        raw = f"{config_seed}:{probe.probe_id}:{context_length}"
        seed = int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        shuffled = list(self.fillers)
        rng.shuffle(shuffled)

        probe_text = probe.content
        probe_tokens = self.tokenizer.count_tokens(probe_text)
        delimiter_tokens = self.tokenizer.count_tokens("\n\n")

        # Budget for filler (total context minus probe and its two delimiters)
        filler_budget = context_length - probe_tokens - 2 * delimiter_tokens

        # Pack a single filler body up to the budget
        filler_parts: list[str] = []
        filler_ids: list[str] = []
        filler_tokens = 0
        for fp in shuffled:
            fp_tokens = self.tokenizer.count_tokens(fp.content)
            cost = fp_tokens + (delimiter_tokens if filler_parts else 0)
            if filler_tokens + cost > filler_budget:
                break
            if filler_parts:
                filler_parts.append("\n\n")
                filler_tokens += delimiter_tokens
            filler_parts.append(fp.content)
            filler_ids.append(fp.filler_id)
            filler_tokens += fp_tokens

        filler_body = "".join(filler_parts)

        # Find the split point closest to target position (in tokens)
        target_pos_tokens = int(context_length * position_percent)

        # Walk filler body token-by-token at passage boundaries to find
        # the best split point.
        segments: list[tuple[str, str]] = []  # (text, filler_id)
        idx = 0
        for filler_id in filler_ids:
            # Find this passage in the filler body starting from idx
            fp = next(f for f in self.fillers if f.filler_id == filler_id)
            start = filler_body.find(fp.content, idx)
            segments.append((fp.content, filler_id))
            idx = start + len(fp.content)

        # Build cumulative token counts at each passage boundary
        cumulative: list[int] = [0]
        for seg_text, _ in segments:
            prev = cumulative[-1]
            seg_tokens = self.tokenizer.count_tokens(seg_text)
            cost = seg_tokens + (delimiter_tokens if prev > 0 else 0)
            cumulative.append(prev + cost)

        # Pick the boundary closest to target_pos_tokens
        best_split = 0
        best_dist = abs(cumulative[0] - target_pos_tokens)
        for i, cum in enumerate(cumulative):
            dist = abs(cum - target_pos_tokens)
            if dist < best_dist:
                best_dist = dist
                best_split = i

        before_ids = [sid for _, sid in segments[:best_split]]
        after_ids = [sid for _, sid in segments[best_split:]]

        # Reconstruct text
        before_text = "\n\n".join(seg_text for seg_text, _ in segments[:best_split])
        after_text = "\n\n".join(seg_text for seg_text, _ in segments[best_split:])

        if before_text and after_text:
            full_text = before_text + "\n\n" + probe_text + "\n\n" + after_text
        elif before_text:
            full_text = before_text + "\n\n" + probe_text
        elif after_text:
            full_text = probe_text + "\n\n" + after_text
        else:
            full_text = probe_text

        actual_tokens = self.tokenizer.count_tokens(full_text)
        before_tokens = self.tokenizer.count_tokens(before_text) + (delimiter_tokens if before_text else 0) if before_text else 0

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
