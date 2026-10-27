from pathlib import Path
from fastapi import FastAPI, HTTPException
from .models import AskRequest, Answer
from .retrieval import PolicyRetriever
from .answering import answer_question, provider_configuration
from .evaluation import run_evaluation
from .mcp_server import handle_rpc

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "policies.jsonl"
EVAL_PATH = ROOT / "eval" / "cases.json"
retriever = PolicyRetriever.from_jsonl(DATA_PATH)

app = FastAPI(
    title="Enterprise Policy Assistant",
    description="A grounded policy retrieval API with schema validation and a small evaluation harness.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, bool]:
    return provider_configuration()


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest) -> Answer:
    try:
        return answer_question(request, retriever)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Policy query failed.") from exc


@app.post("/mcp")
def mcp(payload: dict) -> dict:
    return handle_rpc(payload, retriever)


@app.get("/eval")
def evaluate() -> dict:
    return run_evaluation(EVAL_PATH, retriever)
