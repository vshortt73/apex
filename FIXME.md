# APEX — Known Issues

Bugs and UX issues found during field deployment (2026-03-03).
Items marked [PATCHED] have local workarounds applied on this system.

---

### 1. Dashboard database connection not documented
**Where:** `docs/setup.md`, `README.md`
**Problem:** Setup docs never explain how to configure the dashboard's PostgreSQL connection. Users hit an auth failure on first launch with no guidance.
**Fix:** Add a section covering the three config methods: `APEX_DATABASE_URL` env var, `configs/dashboard.yaml`, and CLI `--db` flag.

### 2. CLI uses relative path for dashboard config
**Where:** `src/apex/cli.py:103`
**Problem:** `DashboardConfig.load("configs/dashboard.yaml")` uses a relative path, so launching from any directory other than the project root silently falls back to hardcoded defaults (`apex:apex@localhost:5432/apex`). The `scripts/dashboard.sh` wrapper doesn't `cd` to the project root.
**Fix:** Either use an absolute path derived from the package location (like `app.py:14` already does), or add `cd "$PROJECT_ROOT"` to `dashboard.sh`.

### 3. Settings tab overwrites dashboard config on save
**Where:** `src/apex/dashboard/views/settings.py`
**Problem:** When the dashboard starts with defaults (because it couldn't find the config file per issue #2), the Settings "Save" button writes those defaults back to the config file, overwriting any correct credentials the user had set.
**Fix:** Relates to #2 — once the config is reliably loaded, this goes away. Additionally consider a "dirty" flag so Save only writes fields the user actually changed.

### 4. llama-server path input accepts directories — should want a file [PATCHED: no]
**Where:** `src/apex/dashboard/views/settings.py:273-285`
**Problem:** The "Test" button for llama-server binary checks `Path.is_file()`, but the input field's UX doesn't make it clear a full path to the executable is needed. The models field asks for a directory, making it inconsistent.
**Fix:** Either auto-append `llama-server` when a directory is given, or improve the placeholder/label to say "path to llama-server binary" and show a clearer error like "Path is a directory — provide the full path to the llama-server binary".

### 5. Multi-GPU systems get no GPU stats [PATCHED]
**Where:** `src/apex/dashboard/services/infra.py:get_gpu_stats()`
**Problem:** `nvidia-smi` returns one CSV line per GPU. The original code parsed the entire stdout as a single comma-split line, so multi-GPU output mangled the fields and `int()` conversion failed silently, returning `None`.
**Fix applied locally:** Changed return type from `GpuStats | None` to `list[GpuStats]`, parsing each line independently. Updated callers in `infrastructure.py` (renders per-GPU cards) and `run_monitor.py` (aggregates across GPUs in system bar). **This patch needs to be committed upstream.**
**Files changed:**
- `src/apex/dashboard/services/infra.py` — `get_gpu_stats()` returns `list[GpuStats]`
- `src/apex/dashboard/views/infrastructure.py` — `update_gpu_stats()` renders per-GPU cards
- `src/apex/dashboard/views/run_monitor.py` — `update_system_bar()` aggregates multi-GPU stats
