# DocMind AI Studio — Phase 1

Enterprise Multimodal Document Intelligence and Autonomous Research Agent.

## Phase 1 scope

This phase implements only:

- Project initialization
- Python configuration management
- Environment variable loading
- Structured logging
- Minimal FastAPI application
- `GET /health`
- Minimal Streamlit frontend
- Basic unit tests

Later RAG, PDF, retrieval, reranking, LangGraph, MCP, PostgreSQL,
evaluation, security, Docker, and other components are intentionally not
implemented in Phase 1.

## Requirements

- Python 3.11+
- VS Code
- PowerShell on Windows

## Setup in VS Code

Open the project folder in VS Code, then open the integrated terminal.

### 1. Create virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create `.env`

Copy `.env.example` to `.env`.

PowerShell:

```powershell
Copy-Item .env.example .env
```

## Run FastAPI

Terminal 1:

```powershell
uvicorn app.main:app --reload
```

Expected:

```text
Uvicorn running on http://127.0.0.1:8000
```

Open:

```text
http://127.0.0.1:8000/health
```

Expected JSON:

```json
{
  "status": "ok",
  "service": "docmind-ai"
}
```

FastAPI interactive documentation:

```text
http://127.0.0.1:8000/docs
```

## Run Streamlit

Keep FastAPI running and open a second VS Code terminal:

```powershell
streamlit run frontend/app.py
```

Streamlit normally opens:

```text
http://localhost:8501
```

## Run tests

```powershell
pytest -q
```

Expected result:

```text
3 passed
```

The exact test output may vary slightly by pytest version.

## Phase 1 architecture

```text
Browser
   |
   +----> Streamlit Frontend :8501
   |
   +----> FastAPI Backend :8000
                  |
                  +----> /health
```

No RAG or document processing is connected yet.

## Continue

After you have tested Phase 1, return to the chat and type:

`NEXT PHASE`
