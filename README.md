# MLLabAgent

AI Agent for ML experiments analysis, run comparison, next steps recommendation, and report generation.

## Requirements

- Python 3.11 or 3.12
- Virtual environment (.venv)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
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
source .venv/bin/activate
python -m uvicorn --app-dir src ml_lab_agent.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

## Run Application

Use one of:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m ml_lab_agent.main
```

or

```bash
source .venv/bin/activate
python -m uvicorn --app-dir src ml_lab_agent.main:app --host 127.0.0.1 --port 8000
```

For local development with auto-reload:

```bash
source .venv/bin/activate
python -m uvicorn --app-dir src ml_lab_agent.main:app --host 127.0.0.1 --port 8000 --env-file .env --reload
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
