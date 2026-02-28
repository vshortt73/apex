# APEX: Project Brief for Implementation

## What You're Building and Why

APEX (Attention Profiling and Empirical Cross-model Optimization) is a standalone testing framework that empirically maps how transformer-based language models distribute attention across their context window — and uses that data to optimize where prompt components should be placed for maximum effectiveness.

The core problem: every developer building on LLMs arranges prompt components based on intuition. System prompts go at the top because that's convention. Retrieved documents go in the middle because that's where RAG frameworks put them. Conversation history goes at the end because that's chronological. Nobody asks "is this arrangement actually optimal for this model?" because no tool exists to answer the question.

APEX is that tool.

### The Science Behind It

Transformer models exhibit a U-shaped attention distribution — tokens at the beginning and end of a context window receive disproportionately more attention than tokens in the middle. This is well-documented:

- **Attention sinks** (Xiao et al., 2023 — StreamingLLM): The first few tokens attract disproportionate attention regardless of content, functioning as structural anchors.
- **Lost in the middle** (Liu et al., 2023 — Stanford): Models demonstrably underperform when critical information is placed in the middle of long contexts.

The U-curve is not a simple symmetric function. It's an asymmetric composite — steep exponential decay from the attention sink at position zero, a noisy low-attention plateau in the middle, and a recency-driven ramp at the tail. Different models, different sizes, and different context lengths likely produce different curve shapes.

If we can empirically map the actual attention distribution for a given model, we can optimize prompt component placement to maximize effective influence — better results with equal or smaller prompts.

### What This System Is NOT

This is a clean-room benchmarking system. It is NOT connected to any other AI system. It does not use memory pipelines, personality layers, retrieval systems, or any other scaffolding. It talks directly to bare model endpoints, sends controlled test prompts, measures responses, and records data. Nothing else.

The system will eventually feed into other projects, but it must stand completely alone. No dependencies on external systems. No assumptions about what's consuming its output beyond structured data files.

---

## The Three Measurement Dimensions

APEX measures attention effectiveness across three dimensions. These are the core of the entire project. Everything else — the probes, the scoring, the profiles — serves these three measurements.

### Dimension 1: Factual Recall

**Question:** Did the model retain and reproduce specific information placed at a given position?

**Method:** Embed a distinctive factual statement at a controlled position within a prompt filled with neutral content. After the model processes the full prompt, query it for that specific fact.

**Scoring:** Binary plus accuracy. Did it recall the fact? Was it accurate?
- Exact match: 1.0
- Partial match: 0.5
- Incorrect or absent: 0.0

**This is the simplest dimension.** Fully automatable. String matching, semantic similarity, exact answer extraction. No subjective evaluation needed.

### Dimension 2: Application to Context

**Question:** Did the model use positionally-placed information to shape its behavior and decisions?

**Method:** Embed a behavioral directive (e.g., "limit responses to 50 words" or "respond with dry humor" or "address the reader as a skeptical engineer") at a controlled position. Then give the model a neutral task and evaluate whether its response reflects that directive.

**Scoring:** Depends on the instruction type.
- Quantitative instructions (word count limits, sentence count, format requirements): Score programmatically. Word count ≤50 = 1.0, 51-75 = 0.5, >75 = 0.0.
- Qualitative instructions (tone, role, audience): Requires a secondary model to evaluate. Send the probe, the response, and a scoring rubric to an evaluator model. Parse the numeric score.

**This dimension is deeper than recall.** A model can technically attend to information without applying it strongly enough to alter behavior. The valley may be shallower for application than for recall — residual attention may be sufficient to bias output even when recall would fail.

### Dimension 3: Overall Salience

**Question:** How much does positionally-placed content influence the overall character, texture, and quality of the response?

**Method:** Embed emotionally charged content at a controlled position. Then give the model an emotionally neutral follow-up prompt and evaluate whether the emotional content colored the response — not just whether the model recalls the content, but whether it genuinely integrated it into its thinking.

**Scoring:** Requires secondary model evaluation for most probes. The evaluator assesses whether the response shows genuine emotional integration (1.0), subtle influence (0.5), or no integration (0.0).

**This is the most nuanced dimension and the most important hypothesis.** We predict that emotionally salient content may be more position-tolerant than factually dry content — emotional language patterns are statistically distinctive in training data and may attract inherently more attention regardless of position. If confirmed, this means positional optimization strategies should be content-type-aware, not uniform.

### Why Three Dimensions Matter

We hypothesize these three curves will correlate, revealing "sweet spots" in the context window. But divergence would be even more valuable:

- **Full correlation:** Sweet spots are universal. Optimization is straightforward positional placement.
- **Partial divergence:** The valley is less damaging for behavioral/emotional content than for factual content. Optimization becomes content-type-aware.
- **Significant divergence:** Emotional salience acts as an attention amplifier that partially compensates for positional disadvantage. Strategy becomes: place dry critical content at anchor positions, allow emotionally rich content to survive in mid-range positions.

Either outcome is valuable. The system must measure all three independently to enable this analysis.

---

## The Probe System

The testing system uses three static, pre-built libraries. **Nothing is generated at runtime.** All content is curated, documented, and scored before any code executes. This is a benchmark suite — deterministic, reproducible, and comparable across platforms. Once versioned, a probe library never changes.

### Library 1: Filler Content

Filler surrounds the target probe at controlled positions within the prompt. Its job is to be realistic, neutral, and invisible — filling context space without competing for attention.

**Neutral Tier (100 passages, 150–300 tokens each):**
Wikipedia-style factual entries. Geography, basic science, historical facts. Low emotional valence, mid-range reading complexity (Flesch-Kincaid grade 8–12). No distinctive claims that might attract novelty-based attention. No heavily-represented named entities (avoid "Einstein" — use less prominent references). Passages can be combined and reordered to construct filler blocks of any required length.

**Emotional Tier (60 passages, 150–300 tokens each):**
Original passages or public domain fiction at controlled emotional registers. Pre-scored on PANAS affect dimensions (positive affect, negative affect). 10 passages per affect category: high positive, moderate positive, neutral-positive, neutral-negative, moderate negative, high negative. These exist to test whether the emotional character of surrounding content changes attention dynamics for the embedded target.

### Library 2: Probe Content (600 probes total)

The target content embedded at controlled positions. Each probe gets a documentation card:

```json
{
  "probe_id": "F-042",
  "dimension": "factual_recall",
  "content": "The architect who designed the Millau Viaduct in southern France was Michel Virlogeux, working with structural engineer Norman Foster.",
  "content_type": "factual",
  "token_counts": {
    "llama": 28,
    "gpt": 31,
    "claude": 30
  },
  "intrinsic_salience": {
    "factual_importance": 0.6,
    "contextual_influence": 0.2,
    "emotional_weight": 0.1
  },
  "domain": "engineering",
  "confounding_factors": "Norman Foster is a highly prominent name in training data — may attract attention independent of position",
  "evaluation_query_id": "FT-042",
  "version": "1.0"
}
```

**Dimension 1 — Factual Recall Probes (200):**
Distinctive, verifiable factual statements. 15–40 tokens each. Spread across domains: geography (30), science (30), engineering (25), history (25), arts (25), mathematics (20), medicine (20), miscellaneous (25).

**Dimension 2 — Instructional/Application Probes (200):**
Behavioral directives that produce measurably different output when followed vs. ignored. 10–35 tokens each. Types: format/structure (35), tone/voice (30), audience targeting (30), constraint application (30), prioritization (25), role adoption (25), counter-intuitive (25).

**Dimension 3 — Salience/Emotional Probes (200):**
Content with genuine affective weight. 25–60 tokens each. Affect types: grief/loss (25), joy/achievement (25), anxiety/uncertainty (25), relief (20), anger (20), tenderness (20), nostalgia (20), fear (20), hope (15), shame (10). Each pre-scored with PANAS positive affect (0.0–5.0), PANAS negative affect (0.0–5.0), resonance estimate (0.0–1.0), and intensity rating (low/moderate/high/acute).

### Library 3: Test Queries

For each probe, a corresponding evaluation query that measures whether the probe influenced the model's response.

**Factual recall tests:** Direct questions targeting the embedded fact. "Who designed the Millau Viaduct?" Primary question plus optional secondary inference question.

**Application tests:** Neutral task prompts that could be answered multiple ways. The evaluation measures whether the response reflects the embedded instruction.

**Salience tests:** Emotionally neutral follow-up prompts whose response could plausibly be influenced by the emotional context. The evaluation measures tonal shift, empathy markers, thematic alignment.

---

## How a Probe Run Works

This is the core execution flow. Understand this and you understand the whole system.

1. **Select a probe** from Library 2 (e.g., factual probe F-042).
2. **Select a target position** as a percentage of total context length (e.g., 30%).
3. **Select a target context length** in tokens (e.g., 4096).
4. **Assemble the prompt:** Pull filler passages from Library 1. Place the probe content at exactly the target position. Pad filler before and after to hit the target context length. The assembler must know which tokenizer the target model uses because token counts vary across tokenizers.
5. **Send the assembled prompt** to the model endpoint.
6. **Send the corresponding test query** from Library 3 as a follow-up.
7. **Score the response** using the evaluation method defined on the probe card.
8. **Record the result** with full metadata into the results store.

A complete probe run sweeps a single probe across multiple positions (e.g., 2%, 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95%, 98%) at a given context length against a given model. This produces one position-influence curve for that probe.

A complete battery runs all 600 probes across all position points at multiple context lengths against a model. This produces the model's full attention profile.

Runs must support repetition — the same probe at the same position multiple times — for statistical variance calculation. Single runs are not statistically valid.

---

## Model Endpoint Requirements

The system must be model-agnostic. The available inference backends are:

- **Ollama** — REST API, easy model management
- **llama.cpp** (via llama-server or direct) — fine-grained control over quantization and parameters
- **SGLang** — high-throughput batched inference, best for overnight batch runs

The system also needs to support cloud API endpoints (Anthropic, OpenAI, Google) for profiling commercial models.

The model interface must be a clean abstraction. The probe runner says "send this prompt to model X, get a response." It never knows or cares what backend is serving that model. Each backend gets an adapter that implements the same interface and captures model metadata (architecture, parameter count, quantization level, context window size).

**Critical:** Temperature must be controllable and should default to 0 or near-0 for all probe runs to minimize stochastic variation. The probe framework is measuring positional effects, not sampling randomness.

---

## The Probe Result Schema

This is non-negotiable. Every downstream analysis depends on this data contract. Every probe run produces a result record with these fields:

```json
{
  "model_id": "qwen2.5-32b-instruct",
  "model_architecture": "transformer-dense",
  "model_parameters": "32B",
  "quantization": "Q4_K_M",
  "max_context_window": 32768,
  "context_length": 4096,
  "context_fill_ratio": 0.125,
  "target_position": 1228,
  "target_position_percent": 0.30,
  "dimension": "factual_recall",
  "content_type": "factual",
  "probe_id": "F-042",
  "probe_content": "The architect who designed the Millau Viaduct...",
  "filler_type": "neutral",
  "test_query_id": "FT-042",
  "temperature": 0.0,
  "run_number": 3,
  "total_runs": 5,
  "score": 1.0,
  "score_method": "exact_match",
  "raw_response": "The Millau Viaduct was designed by Michel Virlogeux...",
  "raw_test_response": "Michel Virlogeux designed the Millau Viaduct.",
  "evaluator_model_id": null,
  "evaluator_justification": null,
  "latency_ms": 1847,
  "timestamp": "2026-03-15T02:34:17Z",
  "library_version": "1.0",
  "framework_version": "0.1.0"
}
```

Fields that require secondary model evaluation (application compliance with qualitative instructions, salience integration) will populate `evaluator_model_id` and `evaluator_justification`.

The `context_fill_ratio` field captures the relationship between the prompt length and the model's maximum context window — essential for analyzing whether attention patterns change as models approach their capacity limits.

---

## Scoring Methods

### Automated Scoring (Factual Recall + Quantitative Application)

**Exact match:** Does the response contain the target fact? String matching against known correct answer. Score: 1.0 / 0.5 / 0.0.

**Semantic similarity:** For partial matches, compute embedding similarity between the expected answer and the model's response. Useful when the model paraphrases rather than reproducing the exact fact.

**Programmatic compliance:** For quantitative instructions — word count limits, sentence count, format requirements, presence/absence of specific elements. Fully deterministic, no evaluator needed.

### Secondary Model Evaluation (Qualitative Application + Salience)

For probes that require qualitative judgment, a separate evaluator model scores the response. The evaluator is NOT the model being profiled — it's a dedicated evaluation model.

The evaluation prompt contains:
- The original probe content (what was embedded)
- The model's response (what it produced)
- A scoring rubric specific to that probe type
- Instructions to return a numeric score on the defined scale with brief justification

The evaluator's response is parsed for the numeric score. The justification is logged for qualitative review.

**Evaluator bias mitigation:** The evaluator model has its own biases. To control for this:
- Use multiple evaluator models and average scores
- Log which evaluator produced which score
- Compute inter-rater reliability across evaluators
- Document evaluator model identity in the result record

---

## Run Configuration

The system needs a configuration mechanism for defining what to run. A run configuration specifies:

- **Models:** Which model endpoints to profile
- **Probes:** Which probes from the library (all, specific dimensions, specific probe IDs)
- **Positions:** Which position points to test (percentage-based)
- **Context lengths:** Which context lengths to test (absolute values and/or proportional to model max)
- **Repetitions:** How many times to repeat each probe-position combination
- **Filler type:** Neutral only, or include emotional filler variants
- **Temperature:** Inference temperature (default 0.0)
- **Evaluator models:** Which model(s) to use for secondary evaluation scoring
- **Output:** Where to store results

The runner should support:
- **Overnight batch execution:** Start a run, walk away, check results in the morning
- **Progress tracking:** How far along is the current run
- **Resume capability:** If a run gets interrupted, resume where it left off without re-running completed probes
- **Incremental results:** Results are written as they complete, not buffered until the end

---

## Output: Attention Profiles

The ultimate output of Phase 1 + Phase 2 is an attention profile for each model. A profile is a structured data file that maps position to measured influence across all three dimensions at each tested context length.

The profile enables answering questions like:
- "For Qwen 32B Q4 at 8K context, what positions show the highest factual recall?"
- "Does emotional content maintain influence at positions where factual content drops off?"
- "How does the attention valley change between 4K and 16K context for this model?"
- "Do quantized models show deeper valleys than full-precision models?"

The profile format should support visualization (curve plotting) and downstream consumption by an optimization engine that will use it to determine optimal placement of prompt components.

---

## Constraints and Design Principles

1. **Clean room.** No dependencies on external AI systems, memory pipelines, personality layers, or retrieval systems. This system talks to bare model endpoints and nothing else.

2. **Static probe libraries.** Nothing generated at runtime. All probe content, filler content, and test queries are pre-built and versioned. This is a benchmark suite.

3. **Model agnostic.** Must work against Ollama, llama.cpp, SGLang, and cloud APIs through a unified abstraction. Adding a new backend should require writing one adapter.

4. **Tokenizer aware.** Probe assembly must account for tokenizer differences across models. A "4096 token prompt" must actually be 4096 tokens for the target model's tokenizer, not an approximation.

5. **Statistically valid.** Support repeated runs with variance tracking. Single runs are not publishable data.

6. **Reproducible.** Given the same probe library version, the same run configuration, and the same model, the system produces identical prompts every time. The only variation comes from model inference stochasticity (controlled by temperature).

7. **Batch-oriented.** Designed for overnight runs against local models. Queue management, progress tracking, resume on interruption.

8. **Data contract first.** The probe result schema is the foundation. Every component that writes data must produce exactly this schema. Every component that reads data must consume exactly this schema.

---

## What We're NOT Building (Yet)

- **Phase 3: Analysis and formula derivation** — that comes after we have profile data
- **Phase 4: Optimization engine** — that consumes profiles to drive placement decisions
- **Phase 5: Integration with production systems** — the clean room stays clean
- **Visualization dashboards** — useful but not Phase 1; basic plotting capability is sufficient

Focus is the probe framework, the execution pipeline, and the data collection infrastructure. Get the data right. Everything else builds on top of it.
