# Enterprise Policy Assistant — RAG + LLM Evaluation Scaffold

A runnable portfolio project demonstrating policy retrieval, grounded responses with citations, JSON-schema validation, an evaluation harness, a FastAPI service, and a small MCP-style JSON-RPC endpoint.

## Important scope note

This is a **local demonstration scaffold**, not a production HR system and not evidence of a real 1,800-document deployment. It includes 8 synthetic sample policies and 10 evaluation cases so the pipeline can run immediately. The retriever is lexical BM25-like scoring rather than Azure AI Search; the answer path is deterministic demo mode rather than a live LLM call. The configuration endpoint shows whether Azure OpenAI / Anthropic credentials exist, but does not expose them. Replace the demo retriever and answering adapter before claiming provider integration or enterprise-scale metrics.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

## Endpoints

- `GET /health` — liveness check
- `GET /config` — reports whether provider credentials are configured
- `POST /ask` — retrieve relevant policy excerpts and return a typed answer with citations
- `POST /mcp` — minimal JSON-RPC `tools/list` and `tools/call` demonstration
- `GET /eval` — runs the bundled evaluation cases and reports schema pass and retrieval hit rates

Example request:

```json
{
  "question": "Who approves annual leave?",
  "top_k": 4
}
```

## Run tests

```bash
pytest -q
```

## Run with Docker

```bash
docker build -t enterprise-policy-assistant .
docker run --rm -p 8000:8000 enterprise-policy-assistant
```

## Architecture

1. `retrieval.py` tokenizes the question and scores local policy documents.
2. `answering.py` returns excerpts with citations and refuses to fabricate an answer when retrieval finds nothing.
3. `models.py` defines request/response contracts with Pydantic.
4. `evaluation.py` checks response structure with JSON Schema and measures expected-document retrieval.
5. `mcp_server.py` exposes a small tool discovery/call contract for experimentation.
6. `main.py` wires the API together.

## Next production steps

- Index approved source documents into Azure AI Search with access-control filters.
- Add an Azure OpenAI adapter with managed identity or Key Vault-backed secrets.
- Add prompt injection defenses, document-level authorization, audit logs, and PII handling.
- Use an official MCP SDK and transport if MCP protocol interoperability is required.
- Expand evaluation with groundedness, answer correctness, refusal quality, latency, and cost metrics.
