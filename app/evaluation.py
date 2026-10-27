from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator
from .models import AskRequest
from .answering import answer_question
from .retrieval import PolicyRetriever

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["answer", "citations", "confidence", "mode"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["document_id", "title", "excerpt"],
                "properties": {
                    "document_id": {"type": "string"},
                    "title": {"type": "string"},
                    "excerpt": {"type": "string"}
                },
                "additionalProperties": False
            }
        },
        "confidence": {"enum": ["low", "medium", "high"]},
        "mode": {"enum": ["demo", "azure_openai"]}
    },
    "additionalProperties": False
}


def run_evaluation(dataset_path: str | Path, retriever: PolicyRetriever) -> dict[str, Any]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    validator = Draft202012Validator(ANSWER_SCHEMA)
    results = []
    for case in dataset:
        response = answer_question(AskRequest(question=case["question"]), retriever).model_dump()
        schema_errors = list(validator.iter_errors(response))
        citation_ids = {citation["document_id"] for citation in response["citations"]}
        expected = set(case.get("expected_document_ids", []))
        results.append({
            "id": case["id"],
            "schema_valid": not schema_errors,
            "retrieval_hit": bool(expected.intersection(citation_ids)) if expected else None,
            "citation_count": len(citation_ids),
            "schema_errors": [error.message for error in schema_errors],
        })
    scored = [r for r in results if r["retrieval_hit"] is not None]
    return {
        "total": len(results),
        "schema_pass_rate": sum(r["schema_valid"] for r in results) / max(len(results), 1),
        "retrieval_hit_rate": sum(bool(r["retrieval_hit"]) for r in scored) / max(len(scored), 1),
        "cases": results,
    }
