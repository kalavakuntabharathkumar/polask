"""Minimal MCP-style JSON-RPC endpoint for tool integration.

This intentionally implements a small HTTP JSON-RPC surface instead of claiming full
MCP SDK compatibility. Use an official MCP SDK transport for production clients.
"""
from typing import Any
from .answering import answer_question
from .models import AskRequest
from .retrieval import PolicyRetriever


def handle_rpc(payload: dict[str, Any], retriever: PolicyRetriever) -> dict[str, Any]:
    method = payload.get("method")
    request_id = payload.get("id")
    params = payload.get("params") or {}

    if method == "tools/list":
        result = {
            "tools": [{
                "name": "search_policy",
                "description": "Search internal policy documents and return grounded excerpts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 3},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10}
                    },
                    "required": ["question"],
                    "additionalProperties": False
                }
            }]
        }
    elif method == "tools/call" and params.get("name") == "search_policy":
        arguments = params.get("arguments") or {}
        request = AskRequest(
            question=arguments.get("question", ""),
            top_k=arguments.get("top_k", 4)
        )
        result = {
            "content": [{
                "type": "text",
                "text": answer_question(request, retriever).model_dump_json()
            }]
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Unsupported method: {method}"}
        }

    return {"jsonrpc": "2.0", "id": request_id, "result": result}
