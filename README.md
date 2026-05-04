# MLLabAgent

MLLabAgent is a FastAPI + LangGraph assistant for analyzing MLflow experiment runs.

It can:
- show runs from MLflow
- show latest run
- show best run by metric
- compare two runs
- compare metrics and parameters
- summarize experiment differences using an LLM
- fallback to deterministic summary if the LLM fails

## Requirements

- Python 3.11 or 3.12
- uv

## Setup

```bash
uv sync
```

Optional: activate the created virtual environment manually.

```bash
source .venv/bin/activate
```

## Configure GEMINI_API_KEY

You can configure the key in one of the two ways below.

### Option 1: Temporary in current terminal session

```bash
export GEMINI_API_KEY="your_api_key_here"
```

This works only in the current shell session.

### Option 2: Project .env file (recommended)

Create a .env file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Then run uvicorn with env-file:

```bash
uv run python -m uvicorn --app-dir src ml_lab_agent.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

## Run Application

Recommended:

```bash
uv run python -m uvicorn --app-dir src ml_lab_agent.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

For local development with auto-reload:

```bash
uv run python -m uvicorn --app-dir src ml_lab_agent.main:app --host 127.0.0.1 --port 8000 --env-file .env --reload
```

## MLflow Demo Runs

Use two terminals.

1. Start MLflow tracking server on port 8080:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 8080
```

2. In another terminal, create demo runs in experiment `MLLabAgent Demo Runs`:

```bash
uv run python src/scripts/create_demo_runs.py --tracking-uri http://127.0.0.1:8080 --experiment-name "MLLabAgent Demo Runs"
```

3. Open UI:

```text
http://127.0.0.1:8080
```

## Supported Chat Requests

The agent accepts natural language requests about ML experiments. Examples:

- show run baseline_lr
- show latest run
- show best run by accuracy
- compare run baseline_lr and run tuned_cnn
- compare latest run and best run by f1_score
- summarize comparison of baseline_lr and tuned_cnn

## Architecture

High-level flow:

FastAPI -> LangGraph -> Services -> Repositories -> MLflow

- FastAPI exposes the `/chat` endpoint.
- LangGraph orchestrates the workflow and fallback paths.
- Services contain business logic such as run comparison, best-run selection, and identifier resolution.
- Repositories read experiment data from MLflow.
- LLM services parse natural language requests and generate experiment summaries.

## Environment Variables

Use `.env.example` as a template:

```bash
cp .env.example .env
```

Run tests (excluding integration tests):

```bash
uv run pytest -m "not integration"
```

## Stop Running App

- If running in foreground terminal: press Ctrl+C.
- If running in background or from another terminal:

```bash
pkill -f "uvicorn.*ml_lab_agent.main:app"
```

- If port is still occupied, find and kill by PID:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <PID>
```
