from fastapi.testclient import TestClient
from app.main import app
from app.mcp_server import handle_rpc
from app.retrieval import PolicyRetriever
from app.models import PolicyDocument

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_citations():
    response = client.post("/ask", json={"question": "Who approves annual leave?"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert "leave-001" in [c["document_id"] for c in body["citations"]]


def test_ask_rejects_too_short_question():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_eval_schema_and_retrieval_metrics():
    report = client.get("/eval").json()
    assert report["total"] == 10
    assert report["schema_pass_rate"] == 1.0
    assert 0 <= report["retrieval_hit_rate"] <= 1


def test_mcp_tools_list():
    retriever = PolicyRetriever([PolicyDocument(id="p1", title="Policy", text="Leave requires manager approval.")])
    response = handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, retriever)
    assert response["result"]["tools"][0]["name"] == "search_policy"


def test_mcp_tool_call():
    retriever = PolicyRetriever([PolicyDocument(id="p1", title="Leave", text="Annual leave requires manager approval.")])
    response = handle_rpc({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "search_policy", "arguments": {"question": "Who approves leave?"}}
    }, retriever)
    assert response["result"]["content"][0]["type"] == "text"


def test_mcp_unknown_method():
    retriever = PolicyRetriever([])
    response = handle_rpc({"jsonrpc": "2.0", "id": 3, "method": "bad/method"}, retriever)
    assert response["error"]["code"] == -32601
