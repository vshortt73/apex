# APEX Setup and Running Instructions

## Prerequisites

- Python 3.11 or later
- A model inference backend (at least one of the following):
  - [Ollama](https://ollama.com) (local, easiest to start with)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp) server
  - [SGLang](https://github.com/sgl-project/sglang) server
  - OpenAI API key
  - Anthropic API key
  - Google AI API key

## Installation

```bash
cd /programs/apex

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install APEX and all dependencies
pip install -e .

# Optional: install dev tools (pytest, ruff)
pip install -e ".[dev]"
```

### Dependencies installed

| Package | Purpose |
|---|---|
| pyyaml | Configuration file parsing |
| httpx | HTTP client for local model backends |
| anthropic | Anthropic API adapter |
| openai | OpenAI API adapter |
| google-genai | Google Gemini API adapter |
| tiktoken | OpenAI-compatible tokenization |
| transformers | HuggingFace tokenizer support |
| sentence-transformers | Semantic similarity scoring |
| tqdm | Progress bars |

## Configuration

Copy the example config and edit it for your environment:

```bash
cp apex.yaml.example apex.yaml
```

### Config sections

**run** — Global run parameters:

```yaml
run:
  seed: 42              # RNG seed for deterministic filler ordering
  temperature: 0.0      # Model temperature (0.0 recommended for benchmarking)
  repetitions: 3        # Repeat each probe-position combo N times
  filler_type: neutral  # neutral | emotional
  use_calibration: false # true = use frozen calibration prompts instead of assembling
```

**data** — Library and output paths:

```yaml
data:
  directory: data       # Path to probe/filler/query JSON libraries
  output_db: results.db # SQLite output file
```

**probes** — Which probes to run:

```yaml
probes:
  select: all                    # Run all 60 seed probes
  # select: factual_recall       # Run only one dimension
  # select: [F-001, I-001, S-001] # Run specific probe IDs
```

**positions** — Where in the context window to place probes (0.0–1.0 exclusive):

```yaml
positions:
  - 0.02
  - 0.10
  - 0.30
  - 0.50
  - 0.70
  - 0.90
  - 0.98
```

**context_lengths** — Target prompt sizes in tokens:

```yaml
context_lengths:
  - 4096
  - 8192
```

**models** — Models to profile. Each entry requires:

```yaml
models:
  - name: qwen2.5-7b           # Unique identifier for this model
    backend: ollama             # ollama | llamacpp | sglang | openai | anthropic | google
    model_name: qwen2.5:7b     # Backend-specific model name
    tokenizer: approximate      # approximate | tiktoken:<model> | anthropic | <HF model path>
    max_context_window: 32768   # Model's maximum context window
    architecture: transformer-dense  # Optional metadata
    parameters: "7B"            # Optional metadata
    quantization: Q4_K_M        # Optional metadata
```

For cloud APIs, set environment variables for authentication:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

Or specify `api_key` directly in the model config (not recommended for shared configs).

**evaluator_models** — Models used for secondary evaluation scoring (application and salience probes that require qualitative judgment):

```yaml
evaluator_models:
  - name: eval-ollama
    backend: ollama
    model_name: qwen2.5:7b
    tokenizer: approximate
    max_context_window: 32768
```

### Backend-specific notes

| Backend | Default URL | Notes |
|---|---|---|
| ollama | `http://localhost:11434` | Set `base_url` if non-default |
| llamacpp | `http://localhost:8080` | OpenAI-compatible `/v1/chat/completions` |
| sglang | `http://localhost:30000` | OpenAI-compatible endpoint |
| openai | (API) | Uses `OPENAI_API_KEY` env var |
| anthropic | (API) | Uses `ANTHROPIC_API_KEY` env var |
| google | (API) | Uses `GOOGLE_API_KEY` env var |

## Seed Data

The framework ships with a seed probe library (60 probes total):

```
data/
├── filler/
│   └── filler_neutral_seed.json    # 20 neutral filler passages
├── probes/
│   ├── probes_factual_recall_seed.json  # 20 factual recall probes
│   ├── probes_application_seed.json     # 20 instructional probes
│   └── probes_salience_seed.json        # 20 emotional salience probes
└── queries/                              # (empty — queries are inline in probe files)
```

The library loader automatically extracts test queries from the inline `test_query` field in each probe entry. No separate query files are needed for the seed library.

## Running

### Validate configuration

Always validate before running. This checks config syntax, library loading, and estimates total work:

```bash
python -m apex validate apex.yaml
```

Example output:

```
Config: apex.yaml — OK
  Models: 1
    - qwen2.5-7b (ollama, qwen2.5:7b)
  Positions: [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
  Context lengths: [4096, 8192]
  Repetitions: 3
  Library version: 0.1.0-seed
  Fillers loaded: 20
  Probes loaded: 60
  Queries loaded: 60
  Estimated probe executions: 4680

Validation passed.
```

### Execute a probe run

```bash
python -m apex run apex.yaml
```

Add `-v` for verbose logging:

```bash
python -m apex -v run apex.yaml
```

Results are written to SQLite incrementally as each probe completes. The run can be safely interrupted and resumed — on restart, completed probes are skipped automatically.

### Minimal test run

For a quick smoke test, create a minimal config:

```yaml
run:
  seed: 42
  temperature: 0.0
  repetitions: 1
  filler_type: neutral

data:
  directory: data
  output_db: test_results.db

probes:
  select: [F-001, I-001, S-001]  # Just 3 probes

positions: [0.1, 0.5, 0.9]       # 3 positions

context_lengths: [2048]           # 1 context length

models:
  - name: qwen2.5-7b
    backend: ollama
    model_name: qwen2.5:7b
    tokenizer: approximate
    max_context_window: 32768

evaluator_models:
  - name: eval-qwen
    backend: ollama
    model_name: qwen2.5:7b
    tokenizer: approximate
    max_context_window: 32768
```

This runs 9 probe executions (3 probes x 3 positions x 1 context length x 1 rep) — enough to verify everything works end-to-end.

### Check run status

```bash
python -m apex status results.db
python -m apex status results.db --model qwen2.5-7b
```

### Export results to JSON

```bash
python -m apex export results.db -o results.json
python -m apex export results.db -o factual.json --dimension factual_recall
python -m apex export results.db -o qwen.json --model qwen2.5-7b
```

### Rescore existing results

Re-run scoring on stored results without re-running inference. This is useful when:
- Scoring logic has been updated (e.g., switching from 3-tier to continuous scoring)
- Evaluator-scored probes failed during a run (e.g., evaluator server was unreachable, producing NULL scores)

```bash
# Rescore all programmatic + exact_match results
python -m apex rescore results.db --data-dir data

# Rescore only NULL evaluator results with an evaluator model
python -m apex rescore postgresql://user@localhost:5432/apex \
  --evaluator-backend llamacpp \
  --evaluator-model Qwen_Qwen3-30B-A3B-Q4_K_M \
  --evaluator-url http://node2:8080 \
  --null-only

# Preview changes without writing
python -m apex rescore results.db --dry-run

# Filter by model and/or score method
python -m apex rescore results.db --model my-model --score-method evaluator --null-only
```

Without `--evaluator-backend` and `--evaluator-model`, only `programmatic` and `exact_match` results are rescored. Evaluator-method results require an evaluator adapter to be specified.

When rescoring evaluator results, the `evaluator_model_id` column is also updated to reflect which model actually performed the scoring.

The dashboard's Run Control tab also includes a "Rescore NULL Results" card that queries for NULL evaluator-scored results and launches a rescore subprocess with one click.

## Calibration

APEX probes lack baseline calibration by default. A raw score of 0.3 at some position could mean the probe is hard, filler is interfering, or position is bad — there's no way to tell. The calibration subsystem decomposes this into three separable factors using a two-tier baseline.

### Overview

| Tier | What it measures | Command |
|------|-----------------|---------|
| **Bare baseline** | Probe ceiling — score with no filler, no position effects | `--type bare` |
| **Anchored baseline** | Best-with-filler — filler present, optimal endpoint position | `--type anchored` |
| **Full run** | Combined position + filler + probe interaction | Normal `apex run` |

From these: `filler_factor = anchored / bare` (filler cost) and `position_factor = raw / anchored` (pure position effect).

### Step 1: Generate frozen prompt matrix

```bash
# Default: 60 probes × 19 positions × 2 context lengths = 2,280 prompts
python -m apex calibrate generate \
  --db results.db \
  --data-dir data

# Custom context lengths
python -m apex calibrate generate \
  --db results.db \
  --data-dir data \
  --context-lengths 4096 8192 16384

# PostgreSQL via env var
export APEX_DATABASE_URL="postgresql://user:pass@localhost:5432/apex"
python -m apex calibrate generate --data-dir data

# Regenerate from scratch
python -m apex calibrate generate --db results.db --data-dir data --force
```

Frozen prompts are assembled with `seed=42, run_number=1` and stored with SHA256 content hashes for integrity verification. Generation is idempotent — same inputs produce identical prompts, and upsert prevents duplicates.

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `results.db` | Database path or PostgreSQL DSN |
| `--data-dir` | `data` | Path to probe/filler/query libraries |
| `--tokenizer` | `approximate` | Tokenizer spec for token counting |
| `--seed` | `42` | RNG seed for filler ordering |
| `--context-lengths` | `4096 8192` | Target context lengths |
| `--force` | off | Delete existing prompts before generating |

### Step 2: Validate frozen prompts

```bash
python -m apex calibrate validate \
  --db results.db \
  --data-dir data \
  --verbose
```

Runs five checks per prompt:

1. Probe content is a substring of full_text
2. Content hash integrity (recomputed SHA256 matches stored)
3. Probe hash freshness (warns if probe text changed since generation)
4. Token position within 10% of context length
5. Context length fill ratio above 85%

Exits with code 1 on any failure. Run this after updating probe libraries to catch drift — if probes changed, regenerate with `--force`.

### Step 3: Run bare baselines (probe alone, no filler)

```bash
# Uses first model in config
python -m apex calibrate baseline configs/run.yaml \
  --db results.db \
  --type bare

# Specific model
python -m apex calibrate baseline configs/run.yaml \
  --db results.db \
  --type bare \
  --model qwen2.5-7b

# Specific probes only
python -m apex calibrate baseline configs/run.yaml \
  --db results.db \
  --type bare \
  --probe-ids F-001 F-002 F-003

# Overwrite existing baselines
python -m apex calibrate baseline configs/run.yaml \
  --db results.db \
  --type bare \
  --force
```

Each probe is run through the standard two-turn protocol with just the probe content (no filler). This establishes ceiling scores — what each probe gets with zero competition.

### Step 4: Run anchored baselines (filler present, optimal position)

```bash
python -m apex calibrate baseline configs/run.yaml \
  --db results.db \
  --type anchored
```

**Requires step 1 first** — frozen prompts must exist. For each probe and context length, runs both endpoint positions (0.05 and 0.95) from the frozen matrix and keeps the higher score. This represents best-case-with-filler, isolating the filler effect from position effects.

| Flag | Default | Description |
|------|---------|-------------|
| `config` | (required) | YAML config file (for model/evaluator setup) |
| `--db` | from config | Database path or PostgreSQL DSN |
| `--data-dir` | from config | Path to data directory |
| `--model` | first in config | Model name from config |
| `--type` | `bare` | `bare` (probe only) or `anchored` (filler at endpoints) |
| `--probe-ids` | all | Specific probe IDs to baseline |
| `--force` | off | Delete existing baselines before running |

### Step 5: Run a calibrated sweep

With frozen prompts and baselines in place, run the positional sweep using calibrated prompts:

```bash
# Via CLI flag
python -m apex run run.yaml --calibrated

# Or via config (set use_calibration: true in the run section)
python -m apex run run_calibrated.yaml

# Or from the dashboard: Run Control → check "Use calibrated prompts" → Launch
```

Results are stored with `filler_type = 'calibrated'` and are visible in the dashboard's Calibration tab alongside dynamic runs.

### Step 6: Use the decomposition

After running both baseline types and a calibrated run, the three-factor decomposition is available:

```python
from apex.calibration_store import CalibrationStore

store = CalibrationStore("results.db")

# Individual baselines
bare = store.get_baseline_for_probe("F-001", "qwen2.5-7b", "bare")        # e.g. 1.0
anchored = store.get_baseline_for_probe("F-001", "qwen2.5-7b", "anchored") # e.g. 0.85

# Filler factor (convenience method)
filler_factor = store.get_filler_factor("F-001", "qwen2.5-7b")  # 0.85 → filler costs 15%

# Position factor (compute from your run data)
position_factor = raw_score / anchored  # pure positional effect, filler removed
```

### Interpreting results

- **filler_factor close to 1.0** — filler is neutral, not competing with this probe
- **filler_factor significantly < 1.0** — filler itself degrades performance independent of position
- **filler_factor varying by dimension** — filler may not be truly neutral for all probe types (e.g., emotionally flat filler might suppress salience scores)
- **position_factor** — pure positional effect with filler influence factored out

### Dashboard: Calibration tab

The Calibration tab in the dashboard provides four visualizations:

- **Status panel** — Shows frozen prompt count, per-model baseline counts, and next-step guidance
- **Baseline bar chart** — Grouped bars showing bare, anchored, and filler_factor scores per probe, filterable by dimension
- **Normalized curves** — Position factor (score / anchored baseline) across the context window, with CI bands and y=1.0 reference line
- **Calibrated vs Dynamic** — Overlaid curves comparing calibrated runs (frozen prompts) against dynamic runs (assembled prompts) for a single dimension

The tab degrades gracefully: if no calibration data exists, it shows setup instructions. If baselines exist but no calibrated runs, it directs the user to Run Control.

### Recommended execution order

```bash
# 1. Generate frozen prompts
python -m apex calibrate generate --db results.db --data-dir data

# 2. Validate
python -m apex calibrate validate --db results.db --data-dir data

# 3. Run bare baselines
python -m apex calibrate baseline run.yaml --db results.db --type bare

# 4. Run anchored baselines
python -m apex calibrate baseline run.yaml --db results.db --type anchored

# 5. Run calibrated sweep (frozen prompts)
python -m apex run run.yaml --calibrated

# 6. Analyze — Calibration tab shows baseline decomposition and normalized curves
```

## Resume and Interruption

APEX is designed for long overnight runs. If a run is interrupted (Ctrl+C, crash, power loss):

1. Results already written to SQLite are preserved
2. Re-run the same command: `python -m apex run apex.yaml`
3. The runner queries completed `(model_id, probe_id, position, context_length, run_number)` tuples and skips them
4. Only remaining probes are executed

No special flags or recovery steps are needed.

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

All tests use mocked model calls and temporary databases — no live inference backend required.

## Tokenizer Selection

The `tokenizer` field in each model config controls how token counts are calculated during prompt assembly. Options:

| Value | When to use |
|---|---|
| `approximate` | Default fallback. Estimates ~4 chars/token. Good enough for initial testing. |
| `tiktoken:<model>` | OpenAI models. e.g., `tiktoken:gpt-4o`. Exact token counts. |
| `anthropic` | Anthropic models. Requires `ANTHROPIC_API_KEY`. |
| `<HF model path>` | HuggingFace models. e.g., `Qwen/Qwen2.5-7B-Instruct`. Downloads tokenizer on first use. |

For accurate benchmarking, use the model's actual tokenizer rather than `approximate`.

## Scoring Methods

Each probe declares how its responses should be scored:

| Method | Used by | How it works |
|---|---|---|
| `exact_match` | Factual recall probes | Checks if expected answer appears in response. 1.0/0.5/0.0. |
| `programmatic` | Application probes (quantitative) | Word count, sentence count, format checks, forbidden terms. |
| `evaluator` | Application probes (qualitative) + all salience probes | Sends response + rubric to evaluator model, parses JSON score. |
| `semantic` | Available for future probes | Cosine similarity via sentence-transformers. |

Evaluator-based scoring requires at least one model in `evaluator_models`. Without it, evaluator probes will record `score=null`. The dashboard's preflight check will warn when evaluator-method probes are selected but no evaluator model is configured. NULL scores can be filled in later using the `rescore` command.
