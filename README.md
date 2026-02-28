# APEX

**Attention Profiling and Empirical Cross-model Optimization**

APEX is a benchmarking framework that empirically maps how large language models distribute attention across their context window. It measures whether information placed at specific positions in a prompt actually influences the model's behavior — and uses that data to identify optimal placement strategies.

## The Problem

Every developer building on LLMs arranges prompt components based on intuition. System prompts go at the top because that's convention. Retrieved documents go in the middle because that's where RAG frameworks put them. Nobody asks "is this arrangement actually optimal for this model?" because no tool exists to answer the question.

APEX is that tool.

## How It Works

### The Two-Turn Protocol

APEX uses a controlled two-turn conversation to measure positional influence:

1. **Turn 1**: A prompt is assembled with neutral filler text and a probe instruction embedded at a specific position (e.g., 30% through the context). The model processes this and responds. This response is stored but not scored.

2. **Turn 2**: A test query is sent as a follow-up, with the full Turn 1 conversation as history. The model's response to this query is scored — did the embedded instruction actually influence behavior?

By sweeping the same probe across many positions (2%, 10%, 30%, 50%, 70%, 90%, 98%), APEX produces a position-influence curve showing where in the context window the model pays attention.

### Three Measurement Dimensions

APEX measures attention across three independent dimensions:

**Factual Recall** — Did the model retain and reproduce specific information? Probes embed distinctive facts; test queries ask for those facts. Scored by exact/partial string matching.

**Application** — Did the model follow positionally-placed instructions? Probes embed behavioral directives (e.g., "limit responses to 50 words"); test queries give a neutral task. Scored programmatically (word counts, format checks) or by an evaluator model (tone, style).

**Salience** — Did emotionally charged content influence the response? Probes embed emotionally weighted passages; test queries are emotionally neutral. Scored by an evaluator model assessing emotional integration.

These three curves may correlate (universal sweet spots) or diverge (content-type-aware optimization needed). Both outcomes are valuable.

### Prompt Assembly

The assembler constructs deterministic prompts using seeded RNG:

- Filler passages are shuffled deterministically (same seed = same order)
- The probe is inserted at the exact target position in token space
- Filler is packed before and after to hit the target context length
- Token counts use the model's actual tokenizer (not approximations, when available)

## Architecture

```
src/apex/
├── cli.py              CLI entry point (run, status, export, validate, dashboard)
├── config.py           YAML config loading → RunConfig/ModelConfig dataclasses
├── types.py            All dataclasses and enums (Probe, TestQuery, ProbeResult, etc.)
├── libraries.py        Load JSON probe/filler/query libraries
├── tokenizers.py       TokenizerBackend ABC (HuggingFace, tiktoken, anthropic, approximate)
├── assembler.py        Deterministic prompt assembly with seeded RNG
├── runner.py           ProbeRunner — orchestrates the two-turn execution loop
├── storage.py          ResultStore — write results with upsert/resume support
├── db.py               Database backend abstraction (SQLite + PostgreSQL)
├── migrate.py          Schema migrations
│
├── models/             Model adapters (uniform interface)
│   ├── base.py         ModelAdapter ABC — single_turn(), chat()
│   ├── llamacpp.py     llama.cpp server (OpenAI-compatible)
│   ├── ollama.py       Ollama REST API
│   ├── sglang.py       SGLang server
│   ├── openai.py       OpenAI API
│   ├── anthropic.py    Anthropic API
│   └── google.py       Google Gemini API
│
├── scoring/            Score dispatch by probe's declared method
│   ├── exact_match.py  String matching (factual recall)
│   ├── programmatic.py Word count, sentence count, format checks (application)
│   ├── semantic.py     Cosine similarity via sentence-transformers
│   └── evaluator.py    Secondary model evaluation with rubric
│
└── dashboard/          Interactive Dash web application
    ├── app.py          Dash app builder, tab routing
    ├── config.py       Dashboard configuration (YAML persistence, auto-detection)
    ├── queries.py      Read-only database queries → pandas DataFrames
    ├── styles.py       Dark theme, Wong colorblind-safe palette
    ├── export.py       PNG/SVG/CSV figure export
    ├── services/
    │   ├── infra.py         GPU stats, llama-server lifecycle
    │   ├── model_catalog.py .gguf file scanner
    │   └── process_manager.py  APEX run subprocess management
    └── views/
        ├── run_monitor.py          Live progress, activity feed, error tracker
        ├── run_control.py          Config builder, preflight checks, launch
        ├── infrastructure.py       Server management, GPU monitoring
        ├── settings.py             Dashboard configuration UI
        ├── summary.py              Aggregate statistics per model
        ├── curve_explorer.py       Position vs score curves
        ├── dimension_comparison.py Cross-dimension correlation analysis
        ├── context_scaling.py      Context length sensitivity
        ├── cross_model.py          Multi-model comparison
        └── probe_detail.py         Single probe drill-down
```

## Quick Start

### Prerequisites

- Python 3.11+
- At least one inference backend: [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com), or a cloud API key
- PostgreSQL (recommended) or SQLite for result storage

### Installation

```bash
cd /programs/apex
python3 -m venv .venv
source .venv/bin/activate

# Core framework
pip install -e .

# Dashboard (interactive visualization)
pip install -e ".[dashboard]"

# Development tools
pip install -e ".[dev]"
```

### Minimal Test Run

Create a config file `test.yaml`:

```yaml
run:
  seed: 42
  temperature: 0.0
  repetitions: 1
  filler_type: neutral

data:
  directory: data

database:
  url: postgresql://user:pass@localhost:5432/apex

probes:
  select: [F-001, I-001, S-001]

positions: [0.1, 0.5, 0.9]

context_lengths: [2048]

models:
  - name: my-model
    backend: llamacpp
    model_name: my-model
    tokenizer: approximate
    max_context_window: 8192
```

```bash
# Validate first
python -m apex validate test.yaml

# Run (9 probe executions: 3 probes x 3 positions x 1 context length)
python -m apex run test.yaml

# Check status
python -m apex status test.yaml
```

### Dashboard

```bash
# Launch with PostgreSQL
./scripts/dashboard.sh

# Or with explicit database
python -m apex dashboard postgresql://user:pass@localhost:5432/apex

# With SQLite
python -m apex dashboard results.db
```

The dashboard opens at `http://localhost:8050` with 10 tabs:

| Tab | Purpose |
|-----|---------|
| Run Monitor | Live progress bars, activity feed, error tracker |
| Run Control | Build configs, preflight check, launch runs |
| Infrastructure | Start/stop llama-servers, GPU monitoring |
| Summary | Aggregate stats per model |
| Curve Explorer | Position vs score curves by dimension |
| Dimension Comparison | Cross-dimension correlation, sweet spots |
| Context Scaling | How scores change with context length |
| Cross-Model | Multi-model comparison on same axes |
| Probe Detail | Single probe drill-down with raw responses |
| Settings | Infrastructure paths, nodes, database, defaults |

## CLI Commands

```bash
python -m apex run <config.yaml>        # Execute a probe run
python -m apex validate <config.yaml>   # Validate config without running
python -m apex status <db>              # Show run progress and statistics
python -m apex export <db> -o out.json  # Export results to JSON
python -m apex dashboard [db]           # Launch interactive dashboard
```

## Configuration

See [docs/setup.md](docs/setup.md) for complete configuration reference including:
- Model backend setup (llama.cpp, Ollama, SGLang, OpenAI, Anthropic, Google)
- Probe selection (all, by dimension, specific IDs)
- Position and context length presets
- Evaluator model configuration
- Tokenizer selection

## Probe Libraries

APEX ships with a seed library of 60 probes across 3 dimensions:

```
data/
├── filler/
│   └── filler_neutral_seed.json       20 neutral filler passages
├── probes/
│   ├── probes_factual_recall_seed.json 20 factual recall probes
│   ├── probes_application_seed.json    20 instructional/application probes
│   └── probes_salience_seed.json       20 emotional salience probes
└── queries/                            Test queries (inline in probe files)
```

Probe libraries are static and versioned. Nothing is generated at runtime. Given the same library version, run config, and model, APEX produces identical prompts every time.

## Scoring Methods

| Method | Dimension | How It Works |
|--------|-----------|-------------|
| `exact_match` | Factual Recall | Checks if expected answer appears in response (1.0 / 0.5 / 0.0) |
| `programmatic` | Application | Word count, sentence count, format checks, forbidden terms |
| `evaluator` | Application + Salience | Sends response + rubric to evaluator model, parses scored JSON |
| `semantic` | (Available) | Cosine similarity via sentence-transformers |

## Database

APEX supports both SQLite and PostgreSQL:

- **SQLite**: Good for single-machine runs. WAL mode for concurrent read/write.
- **PostgreSQL**: Recommended for multi-node setups and the dashboard.

Results are written incrementally as each probe completes. Runs can be safely interrupted and resumed — the runner queries completed tuples and skips them automatically.

The unique constraint on `(model_id, probe_id, target_position_percent, context_length, run_number)` prevents duplicate results and enables resume.

## Resume and Interruption

APEX is designed for long overnight runs. If interrupted:

1. Results already written to the database are preserved
2. Re-run the same command — completed probes are skipped automatically
3. No special flags or recovery steps needed

## Tests

```bash
python -m pytest tests/ -v
```

85 tests covering assembler, config, libraries, tokenizers, storage, runner, scoring, dashboard queries, and app creation. All tests use mocked model calls and temporary databases.

## Design Principles

1. **Clean room** — No dependencies on external AI systems, memory pipelines, or retrieval systems
2. **Static probe libraries** — All content pre-built and versioned; nothing generated at runtime
3. **Model agnostic** — Unified adapter interface; adding a backend = one file
4. **Tokenizer aware** — Prompt assembly uses the model's actual tokenizer
5. **Reproducible** — Same config + same library = identical prompts (only model inference varies)
6. **Batch oriented** — Progress tracking, incremental writes, automatic resume
7. **Data contract first** — The ProbeResult schema is the foundation everything builds on

## Project

APEX v1.0.0 — ~8,000 lines of Python across 46 source files, 85 tests, 10-tab interactive dashboard.
