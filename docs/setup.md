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

Evaluator-based scoring requires at least one model in `evaluator_models`. Without it, evaluator probes will record `score=null`.
